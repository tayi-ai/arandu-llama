package schema

import (
	"context"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/arandu-io/hesape/process"
)

// PostgresSchemaState is the SchemaState that runs pg_dump, pg_restore and psql.
type PostgresSchemaState struct {
	*BaseSchemaState
}

// NewPostgresSchemaState builds a PostgresSchemaState for connection, using
// factory to run the pg_dump, pg_restore and psql commands.
func NewPostgresSchemaState(connection Connection, factory *process.Factory) *PostgresSchemaState {
	return &PostgresSchemaState{BaseSchemaState: NewBaseSchemaState(connection, factory)}
}

// Dump writes the connection's schema, plus the rows of the migration table,
// to path.
//
// Two pg_dump runs into one file: the schema, then the rows of the migration
// table appended to it. The second is what stops a restored database from
// replaying every migration it already has.
//
// --no-owner and --no-acl are the ones worth naming. Without them the dump
// carries the roles of the machine it came from, and loading it on another
// machine fails on a role that does not exist there -- which makes the squashed
// schema unusable exactly where it is most wanted, on a fresh checkout.
func (s *PostgresSchemaState) Dump(ctx context.Context, connection Connection, path string) error {
	schemaOnly := append(s.baseDumpCommand(), "--schema-only")
	if err := s.dumpInto(ctx, path, false, schemaOnly); err != nil {
		return err
	}

	hasTable, err := s.HasMigrationTable(ctx)
	if err != nil {
		return err
	}
	if !hasTable {
		return nil
	}

	dataOnly := append(s.baseDumpCommand(), "-t", s.GetMigrationTable(), "--data-only")
	return s.dumpInto(ctx, path, true, dataOnly)
}

// dumpInto runs one pg_dump and sends its standard output to path, truncating
// or appending depending on appending.
//
// The dump is written as it arrives rather than read back off the result: a
// schema is as big as the database it came from, and a result held as a string
// holds all of it at once. Quietly is what stops the process package from
// keeping a second copy for nobody.
func (s *PostgresSchemaState) dumpInto(ctx context.Context, path string, appending bool, args []string) error {
	flags := os.O_CREATE | os.O_WRONLY | os.O_TRUNC
	if appending {
		flags = os.O_CREATE | os.O_WRONLY | os.O_APPEND
	}

	file, err := os.OpenFile(path, flags, 0o600)
	if err != nil {
		return fmt.Errorf("schema: writing the Postgres dump to %s: %w", path, err)
	}
	defer file.Close()

	// The handler is called with one stream at a time and never from two
	// goroutines at once, so these two need no lock of their own; the run has
	// finished by the time they are read.
	var stderr strings.Builder
	var writeErr error
	streams := func(stream process.Stream, chunk string) {
		if stream == process.Err {
			stderr.WriteString(chunk)
			return
		}
		if _, err := io.WriteString(file, chunk); err != nil && writeErr == nil {
			writeErr = err
		}
	}

	result, err := s.MakeProcess(s.baseVariables(), args[0], args[1:]...).Quietly().Run(ctx, nil, streams)
	if err != nil {
		return fmt.Errorf("schema: pg_dump into %s: %w", path, err)
	}
	if writeErr != nil {
		return fmt.Errorf("schema: writing the Postgres dump to %s: %w", path, writeErr)
	}
	if result.Failed() {
		return fmt.Errorf("schema: pg_dump into %s: exit code %d: %s",
			path, result.ExitCode(), strings.TrimSpace(stderr.String()))
	}
	s.Output(stderr.String())
	return nil
}

// Load runs the schema file at path against the connection.
//
// The tool is chosen by the file's extension: a .sql file is plain text and
// goes through psql, anything else is pg_dump's own archive format and goes
// through pg_restore. Handing one to the other fails with a message about the
// file being corrupt, which is a misleading thing to read when the file is
// fine and only the reader is wrong.
func (s *PostgresSchemaState) Load(ctx context.Context, path string) error {
	var args []string

	if strings.HasSuffix(path, ".sql") {
		args = append([]string{"psql", "--file=" + path}, s.connectionFlags()...)
	} else {
		args = append([]string{"pg_restore", "--no-owner", "--no-acl", "--clean", "--if-exists"}, s.connectionFlags()...)
		args = append(args, path)
	}

	result, err := s.MakeProcess(s.baseVariables(), args[0], args[1:]...).Run(ctx, nil, nil)
	if err != nil {
		return fmt.Errorf("schema: loading %s with %s: %w", path, args[0], err)
	}
	if _, err := result.Throw(nil); err != nil {
		return fmt.Errorf("schema: loading %s with %s: %w", path, args[0], err)
	}
	return nil
}

// GetMigrationTable returns the migration table's name qualified with its
// schema.
//
// Postgres needs the qualification because pg_dump's -t matches against the
// search path, and a migrations table in a schema that is not first on that
// path would be dumped as nothing at all -- silently, since matching no table
// is not an error.
func (s *PostgresSchemaState) GetMigrationTable() string {
	schema, table, err := NewBuilder(s.GetConnection()).ParseSchemaAndTable(s.migrationTable, "public")
	if err != nil || schema == "" {
		return s.BaseSchemaState.GetMigrationTable()
	}
	return schema + "." + s.GetConnection().GetTablePrefix() + table
}

// baseDumpCommand builds the pg_dump invocation and flags shared by both dump
// passes.
func (s *PostgresSchemaState) baseDumpCommand() []string {
	return append([]string{"pg_dump", "--no-owner", "--no-acl"}, s.connectionFlags()...)
}

// connectionFlags is the host, port, user and database every one of these
// commands takes, passed as plain arguments because no shell parses them.
func (s *PostgresSchemaState) connectionFlags() []string {
	connection := s.GetConnection()
	return []string{
		"--host=" + s.configuredHost(),
		"--port=" + connection.GetConfig("port"),
		"--username=" + connection.GetConfig("username"),
		"--dbname=" + connection.GetConfig("database"),
	}
}

// baseVariables returns the one value that has to travel in the environment
// rather than on the command line: the password.
//
// PGPASSWORD is read by every libpq client. A password passed as an argument
// would be readable by every process on the machine through `ps`. Everything
// else is an ordinary argument, and connectionFlags says why that is safe.
func (s *PostgresSchemaState) baseVariables() map[string]string {
	return map[string]string{"PGPASSWORD": s.GetConnection().GetConfig("password")}
}
