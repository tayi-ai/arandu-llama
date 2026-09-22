package database

import (
	"context"

	"github.com/arandu-io/hesape/database/migrations"
	"github.com/arandu-io/hesape/database/schema"
)

// ForMigrations adapts a Connection to migrations.Connection.
//
// The migrations package declares its own narrow connection interface, because
// it cannot import this one -- this one resolves migrations. The two
// signatures differ in exactly two places, and both differences are the reason
// this adapter is four methods rather than a type assertion:
//
//   - Select there takes no read-replica flag, and always reads through the
//     write pool. A migration that read from a replica would be reading a
//     schema the replica has not been told about yet.
//   - Pretend there returns the statements as strings, because the migrator
//     prints them and has no use for the bindings.
//
// The result also satisfies migrations.TransactionalConnection and
// migrations.PretendingConnection, so `aru migrate` gets its transaction
// wrapping and its --pretend from the same value.
func ForMigrations(connection *Connection) migrations.Connection {
	return migrationConnection{connection}
}

// migrationConnection is the adapter type ForMigrations constructs.
type migrationConnection struct{ *Connection }

// Schema satisfies migrations.Connection.Schema: the Blueprint, over the same
// connection that runs Statement.
//
// The same connection matters more than it reads. A builder over a second
// handle would sit outside the transaction the Migrator opened for this
// migration, so a DDL failure would roll back half of it -- and it would sit
// outside --pretend, so a dry run would create tables.
func (m migrationConnection) Schema() *schema.Builder {
	return schema.NewBuilder(ForSchema(m.Connection))
}

// Statement satisfies migrations.Connection.Statement.
//
// The migrations component writes its placeholders as "?", like everything
// else in this framework, and PostgreSQL wants "$1". The translation used to
// happen here, because this was the first side of the pair that knew which
// engine it was talking to. It now happens on the connection, for every
// statement rather than only the migrator's -- so this method is the call it
// forwards and nothing else.
func (m migrationConnection) Statement(ctx context.Context, query string, bindings []any) (bool, error) {
	return m.Connection.Statement(ctx, query, bindings)
}

// Select satisfies migrations.Connection.Select: it runs the query on the
// write pool, because a migration that read from a replica would be reading a
// schema the replica has not been told about yet.
func (m migrationConnection) Select(ctx context.Context, query string, bindings []any) ([]map[string]any, error) {
	return m.Connection.Select(ctx, query, bindings, false)
}

// SupportsSchemaTransactions satisfies
// migrations.TransactionalConnection.SupportsSchemaTransactions: it reports
// whether this driver can roll a schema change back.
//
// PostgreSQL and SQLite roll DDL back; MySQL commits it the moment it runs, so
// a failed migration there leaves the half it finished behind whatever anybody
// wants. Saying so is what keeps the Migrator from opening a transaction that
// would tell the operator a lie.
func (m migrationConnection) SupportsSchemaTransactions() bool {
	switch m.GetDriverName() {
	case string(DialectPostgres), string(DialectSQLite):
		return true
	default:
		return false
	}
}

// Transaction satisfies migrations.TransactionalConnection.Transaction: it
// runs callback inside a transaction, with a single attempt and no retry.
func (m migrationConnection) Transaction(_ context.Context, callback func() error) error {
	return m.Connection.Transaction(callback, 1)
}

// Pretend satisfies migrations.PretendingConnection.Pretend: it runs callback
// without executing its statements and returns them as strings.
func (m migrationConnection) Pretend(ctx context.Context, callback func() error) ([]string, error) {
	entries, err := m.Connection.Pretend(ctx, func(*Connection) error { return callback() })
	if err != nil {
		return nil, err
	}

	out := make([]string, 0, len(entries))
	for _, entry := range entries {
		out = append(out, entry.Query)
	}
	return out, nil
}

// MigrationResolver adapts a ConnectionResolverInterface to
// migrations.Resolver, which is the same adaptation one level up.
type MigrationResolver struct {
	// Resolver is the resolver being adapted: a DatabaseManager, or a
	// ConnectionResolver built by hand.
	Resolver ConnectionResolverInterface
}

// Connection satisfies migrations.Resolver.Connection: it resolves the named
// connection and adapts it with ForMigrations, or fails if it is not a
// *Connection.
func (r MigrationResolver) Connection(name string) (migrations.Connection, error) {
	connection, err := r.Resolver.Connection(name)
	if err != nil {
		return nil, err
	}

	concrete, ok := connection.(*Connection)
	if !ok {
		return nil, errNotAMigratableConnection
	}
	return ForMigrations(concrete), nil
}

// GetDefaultConnection satisfies migrations.Resolver.GetDefaultConnection,
// forwarding to the wrapped resolver.
func (r MigrationResolver) GetDefaultConnection() string { return r.Resolver.GetDefaultConnection() }

// SetDefaultConnection satisfies migrations.Resolver.SetDefaultConnection,
// forwarding to the wrapped resolver.
func (r MigrationResolver) SetDefaultConnection(name string) { r.Resolver.SetDefaultConnection(name) }

// errNotAMigratableConnection is what MigrationResolver.Connection returns
// when the resolved connection is not a *Connection. It is possible only in a
// test that substituted the interface, and saying which one is not there
// beats a nil dereference.
var errNotAMigratableConnection = errNotMigratable{}

type errNotMigratable struct{}

func (errNotMigratable) Error() string {
	return "database: this connection is not a *database.Connection, so migrations cannot run on it"
}

// The adapters satisfy the interfaces they exist for, checked here rather than
// discovered at the call site.
var (
	_ migrations.Connection              = migrationConnection{}
	_ migrations.TransactionalConnection = migrationConnection{}
	_ migrations.PretendingConnection    = migrationConnection{}
	_ migrations.Resolver                = MigrationResolver{}
)
