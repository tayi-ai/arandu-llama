package schema

import (
	"context"
	"strings"

	"github.com/arandu-io/hesape/process"
)

// SchemaState is the pair of operations behind `schema:dump` and the squashed
// migration:
// write the whole schema to a file, and load one back. Both run the engine's own
// tool -- mysqldump, pg_dump, sqlite3 -- because no driver can produce a dump
// another server will accept.
//
// Dump and Load are the interface; everything else is BaseSchemaState, which a
// driver's state embeds. The process factory is a field so a test can fake the
// tools instead of needing them installed.
type SchemaState interface {
	// Dump writes the connection's schema to path.
	Dump(ctx context.Context, connection Connection, path string) error

	// Load runs the schema file at path against the connection.
	Load(ctx context.Context, path string) error
}

// The three concrete states implement the contract, checked here rather than
// at the first call site, so a state missing Dump or Load fails to compile
// instead of only failing when something tries to use it.
var (
	_ SchemaState = (*MySqlSchemaState)(nil)
	_ SchemaState = (*PostgresSchemaState)(nil)
	_ SchemaState = (*SqliteSchemaState)(nil)
)

// BaseSchemaState is the half of a SchemaState no driver disagrees about: the
// connection, the migration table's name, the process factory and the output
// handler. A driver's schema state embeds it and adds Dump and Load.
type BaseSchemaState struct {
	connection     Connection
	migrationTable string
	factory        *process.Factory
	output         func(line string)
}

// NewBaseSchemaState builds a BaseSchemaState for connection, with the
// migration table named "migrations" until WithMigrationTable overrides it. A
// nil factory means a new one, and the output handler defaults to one that
// discards every line.
func NewBaseSchemaState(connection Connection, factory *process.Factory) *BaseSchemaState {
	if factory == nil {
		factory = process.NewFactory()
	}
	return &BaseSchemaState{
		connection:     connection,
		migrationTable: "migrations",
		factory:        factory,
		output:         func(string) {},
	}
}

// MakeProcess describes the program name and args, without starting it.
//
// It takes the program and its arguments apart rather than a shell string,
// because a shell string assembled from a database configuration is an
// injection with a familiar shape -- and the process package it goes through has
// no string form to hand a shell.
//
// The environment is what a client program has to be told without it appearing
// on a command line: PGPASSWORD and MYSQL_PWD are how pg_dump and mysqldump take
// a password without it being visible in `ps`. It is added to the environment
// this process already has rather than replacing it, because these programs also
// read PGHOST, PGSSLMODE, HOME and the rest, and a schema dump that ignored the
// operator's environment would fail in ways that have nothing to do with the
// schema.
//
// There is no deadline: a dump takes as long as the database is big, and a limit
// measured in seconds is a limit that fails on the databases worth dumping.
func (s *BaseSchemaState) MakeProcess(environment map[string]string, name string, args ...string) *process.PendingProcess {
	return s.factory.NewPendingProcess().
		Command(append([]string{name}, args...)...).
		Env(environment).
		Forever()
}

// HasMigrationTable reports whether the connection's migration table exists.
func (s *BaseSchemaState) HasMigrationTable(ctx context.Context) (bool, error) {
	return NewBuilder(s.connection).HasTable(ctx, s.migrationTable)
}

// GetMigrationTable returns the migration table's name, with the connection's
// table prefix applied.
func (s *BaseSchemaState) GetMigrationTable() string {
	return s.connection.GetTablePrefix() + s.migrationTable
}

// WithMigrationTable sets the migration table's name and returns s.
func (s *BaseSchemaState) WithMigrationTable(table string) *BaseSchemaState {
	s.migrationTable = table
	return s
}

// HandleOutputUsing installs output as the handler for each line the
// underlying command writes; a nil output installs one that discards
// everything.
func (s *BaseSchemaState) HandleOutputUsing(output func(line string)) *BaseSchemaState {
	if output == nil {
		output = func(string) {}
	}
	s.output = output
	return s
}

// Output hands one line to the installed output handler. The handler is an
// unexported field holding a function, reachable only from this package, so
// the call needs a name outside it.
func (s *BaseSchemaState) Output(line string) { s.output(line) }

// GetConnection returns the connection the state works against.
func (s *BaseSchemaState) GetConnection() Connection { return s.connection }

// configuredHost returns the connection's configured host, taking the first
// entry when it is a comma-separated list.
//
// A read/write cluster is configured with a list of hosts, and a dump is one
// connection to one server, so this takes the first. GetConfig returns a
// string here rather than a list, so the split is done on the way past.
func (s *BaseSchemaState) configuredHost() string {
	host := s.connection.GetConfig("host")
	if first, _, found := strings.Cut(host, ","); found {
		return strings.TrimSpace(first)
	}
	return host
}
