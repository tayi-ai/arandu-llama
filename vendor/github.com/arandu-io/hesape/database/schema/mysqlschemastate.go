package schema

import (
	"bytes"
	"context"
	"fmt"
	"os"
	"regexp"
	"strings"

	"github.com/arandu-io/hesape/process"
)

// MySqlSchemaState is the SchemaState that runs mysqldump and mysql.
//
// It serves MariaDB as well: the one difference, omitting --set-gtid-purged, is
// a branch taken on Connection.IsMaria.
type MySqlSchemaState struct {
	*BaseSchemaState
}

// NewMySqlSchemaState builds a MySqlSchemaState for connection, using factory to
// run the mysqldump and mysql commands.
func NewMySqlSchemaState(connection Connection, factory *process.Factory) *MySqlSchemaState {
	return &MySqlSchemaState{BaseSchemaState: NewBaseSchemaState(connection, factory)}
}

// autoIncrementState matches the AUTO_INCREMENT=<n> clause mysqldump writes
// into a table's definition.
//
// mysqldump records where each table's counter had got to. Keeping it makes the
// dump differ on every run for reasons that are not schema changes, so the file
// churns in review and two developers who ran no migrations still get a diff.
var autoIncrementState = regexp.MustCompile(`(?i)\s+AUTO_INCREMENT=[0-9]+`)

// Dump writes the connection's schema, plus the rows of the migration table,
// to path.
//
// Three steps: dump the structure, strip the auto-increment counters, then
// append the rows of the migration table so a restored database does not
// replay the migrations it already has.
func (s *MySqlSchemaState) Dump(ctx context.Context, connection Connection, path string) error {
	args := append(s.baseDumpCommand(), "--routines", "--result-file="+path, "--no-data")
	if err := s.executeDumpProcess(ctx, args, nil); err != nil {
		return err
	}

	if err := s.removeAutoIncrementingState(path); err != nil {
		return err
	}

	hasTable, err := s.HasMigrationTable(ctx)
	if err != nil {
		return err
	}
	if !hasTable {
		return nil
	}
	return s.appendMigrationData(ctx, path)
}

// removeAutoIncrementingState strips the AUTO_INCREMENT=<n> clauses from the
// dump file at path.
func (s *MySqlSchemaState) removeAutoIncrementingState(path string) error {
	contents, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("schema: reading the MySQL dump at %s: %w", path, err)
	}
	if err := os.WriteFile(path, autoIncrementState.ReplaceAll(contents, nil), 0o600); err != nil {
		return fmt.Errorf("schema: rewriting the MySQL dump at %s: %w", path, err)
	}
	return nil
}

// appendMigrationData dumps the migration table's rows with mysqldump and
// appends them to the schema file at path.
func (s *MySqlSchemaState) appendMigrationData(ctx context.Context, path string) error {
	args := append(s.baseDumpCommand(),
		s.GetMigrationTable(),
		"--no-create-info", "--skip-extended-insert", "--skip-routines", "--compact", "--complete-insert")

	var stdout bytes.Buffer
	if err := s.executeDumpProcess(ctx, args, &stdout); err != nil {
		return err
	}

	file, err := os.OpenFile(path, os.O_APPEND|os.O_WRONLY, 0o600)
	if err != nil {
		return fmt.Errorf("schema: appending the migration rows to %s: %w", path, err)
	}
	defer file.Close()

	if _, err := file.Write(stdout.Bytes()); err != nil {
		return fmt.Errorf("schema: appending the migration rows to %s: %w", path, err)
	}
	return nil
}

// Load runs the schema file at path against the connection with the mysql
// client.
func (s *MySqlSchemaState) Load(ctx context.Context, path string) error {
	file, err := os.Open(path)
	if err != nil {
		return fmt.Errorf("schema: reading the MySQL schema file %s: %w", path, err)
	}
	defer file.Close()

	args := append([]string{"mysql"}, s.connectionFlags()...)
	args = append(args, "--database="+s.GetConnection().GetConfig("database"))

	result, err := s.MakeProcess(s.baseVariables(), args[0], args[1:]...).Input(file).Run(ctx, nil, nil)
	if err != nil {
		return fmt.Errorf("schema: loading %s with mysql: %w", path, err)
	}
	if _, err := result.Throw(nil); err != nil {
		return fmt.Errorf("schema: loading %s with mysql: %w", path, err)
	}
	return nil
}

// baseDumpCommand builds the mysqldump invocation and flags shared by Dump
// and appendMigrationData.
//
// --set-gtid-purged=OFF is added only for MySQL. MariaDB's mysqldump has no such
// option and refuses the whole command over it, which is why IsMaria is
// checked before adding it rather than letting executeDumpProcess retry.
func (s *MySqlSchemaState) baseDumpCommand() []string {
	args := append([]string{"mysqldump"}, s.connectionFlags()...)
	args = append(args,
		"--no-tablespaces", "--skip-add-locks", "--skip-comments",
		"--skip-set-charset", "--tz-utc", "--column-statistics=0")

	if !s.GetConnection().IsMaria() {
		args = append(args, "--set-gtid-purged=OFF")
	}
	return append(args, s.GetConnection().GetConfig("database"))
}

// connectionFlags builds the --user flag plus either --socket, or --host and
// --port, taken from the connection's configuration.
//
// A configured Unix socket wins over host and port, because the two cannot both
// be given: mysqldump takes whichever it sees and the other is silently ignored,
// so passing both makes the connection depend on argument order.
//
// connectionFlags carries no SSL flags: the Go driver takes TLS through a
// registered config name in the DSN instead of discrete options. An operator
// who needs a dump over TLS sets it in the environment the client already
// reads, which processEnvironment inherits.
func (s *MySqlSchemaState) connectionFlags() []string {
	connection := s.GetConnection()

	flags := []string{"--user=" + connection.GetConfig("username")}

	if socket := connection.GetConfig("unix_socket"); socket != "" {
		return append(flags, "--socket="+socket)
	}
	return append(flags,
		"--host="+s.configuredHost(),
		"--port="+connection.GetConfig("port"))
}

// baseVariables returns the one environment variable the dump and load
// commands need: the password, which must not appear on a command line.
//
// MYSQL_PWD is how the client takes a password without it being visible in
// `ps`.
func (s *MySqlSchemaState) baseVariables() map[string]string {
	return map[string]string{"MYSQL_PWD": s.GetConnection().GetConfig("password")}
}

// executeDumpProcess runs the dump, and drops an option the installed client
// does not know rather than failing over it.
//
// Both options it retries without are recent additions that older and
// alternative clients reject outright. --column-statistics=0 is not understood
// before mysqldump 8.0, and --set-gtid-purged is a MySQL option MariaDB does not
// have; in each case the client exits without writing anything, and the message
// names the option. This drops each of the two at most once and cannot loop.
func (s *MySqlSchemaState) executeDumpProcess(ctx context.Context, args []string, stdout *bytes.Buffer) error {
	err := s.runDump(ctx, args, stdout)
	if err == nil {
		return nil
	}

	for _, option := range []struct {
		flag    string
		mention []string
	}{
		{"--column-statistics=0", []string{"column-statistics", "column_statistics"}},
		{"--set-gtid-purged=OFF", []string{"set-gtid-purged"}},
	} {
		if !mentionsAny(err.Error(), option.mention) {
			continue
		}
		if stdout != nil {
			stdout.Reset()
		}
		args = withoutArg(args, option.flag)
		if err = s.runDump(ctx, args, stdout); err == nil {
			return nil
		}
	}
	return err
}

// runDump runs one mysqldump, sending its standard output to the buffer when
// one was given and letting --result-file place it otherwise.
//
// A failed dump is reported through Throw, whose error carries both of the
// program's streams: that is what executeDumpProcess reads to decide whether the
// client rejected an option it does not have.
func (s *MySqlSchemaState) runDump(ctx context.Context, args []string, stdout *bytes.Buffer) error {
	result, err := s.MakeProcess(s.baseVariables(), args[0], args[1:]...).Run(ctx, nil, nil)
	if err != nil {
		return fmt.Errorf("schema: mysqldump: %w", err)
	}
	if _, err := result.Throw(nil); err != nil {
		return fmt.Errorf("schema: mysqldump: %w", err)
	}

	if stdout != nil {
		stdout.WriteString(result.Output())
	}
	s.Output(result.ErrorOutput())
	return nil
}

// mentionsAny reports whether message contains any of needles.
func mentionsAny(message string, needles []string) bool {
	for _, needle := range needles {
		if strings.Contains(message, needle) {
			return true
		}
	}
	return false
}

// withoutArg removes flag from args by exact match on the argument list,
// which cannot accidentally match a substring of a password or a table name
// the way a search over the whole command line can.
func withoutArg(args []string, flag string) []string {
	out := make([]string, 0, len(args))
	for _, arg := range args {
		if arg == flag {
			continue
		}
		out = append(out, arg)
	}
	return out
}
