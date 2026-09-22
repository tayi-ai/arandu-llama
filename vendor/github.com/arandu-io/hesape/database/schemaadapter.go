package database

import (
	"context"
	"strings"
	"sync"

	"github.com/arandu-io/hesape/database/schema"
	"github.com/arandu-io/hesape/database/schema/grammars"
)

// ForSchema adapts a Connection to schema.Connection.
//
// It is the seam between the connection this package owns and the schema
// package, and it exists rather than the concrete Connection growing the
// seventeen methods of that interface. Two reasons, and the second is the one
// that decides:
//
// The signatures do not line up. schema.Connection wants Select(ctx, query) and
// this connection answers Select(ctx, query, bindings, useReadPDO); it wants
// GetServerVersion() string and this one answers (string, error) and takes a
// context; it wants GetConfig(option) string where this one answers any. Every
// one of those is a wrapper, and a wrapper on the concrete type would be a
// second spelling of a method that already exists.
//
// And an adapter keeps the change local. Making Connection implement
// schema.Connection would put six processors, a grammar choice and a version
// lookup on the type every repository, every migration and the Collector reach
// through -- which is the type least able to absorb an unrelated change.
//
// # No Grant
//
// Nothing here takes one. DDL names a table, not a row: there is no tenant to
// scope by, no subject to attribute to, and no request it came from. The path to
// application rows is the one that needs a Grant, and it still has one on every
// method.
func ForSchema(connection *Connection) schema.Connection {
	adapter := &schemaConnection{connection: connection}
	// The grammar reads the connection back -- for the version, for the driver,
	// for the foreign key state -- so it is built over the adapter and held.
	adapter.grammar = grammarFor(connection.GetDriverName(), adapter)
	return adapter
}

// grammarFor picks the schema grammar by driver name.
//
// An unknown driver gets the SQLite grammar rather than nil, because a nil
// grammar fails inside the compiler with a message about a nil pointer, and
// this way it fails where the statement is refused with the name of the driver
// in it.
func grammarFor(driver string, connection schema.Connection) schema.Grammar {
	switch strings.ToLower(driver) {
	case "pgsql", "postgres", "postgresql":
		return grammars.NewPostgresGrammar(connection)
	case "mysql", "mariadb":
		return grammars.NewMySQLGrammar(connection)
	default:
		return grammars.NewSQLiteGrammar(connection)
	}
}

// schemaConnection is the adapter ForSchema constructs.
type schemaConnection struct {
	connection *Connection
	grammar    schema.Grammar

	// prepare guards the one-time reads that need a context.
	//
	// schema.Connection answers GetServerVersion and
	// ForeignKeyConstraintsEnabled without one, and both are questions for the
	// server. They are read here, once, from the context of the operation that
	// asked -- rather than from a context.Background() invented to paper over
	// the mismatch, which is what would hide a cancelled migration hanging on a
	// version query.
	prepare       sync.Once
	serverVersion string
	foreignKeys   bool
}

// Prepare reads what the interface answers without a context, using one.
//
// It is idempotent and it is called by the adapter itself before anything that
// needs the answers. A caller may call it earlier to pay the cost where it can
// see it.
func (c *schemaConnection) Prepare(ctx context.Context) {
	c.prepare.Do(func() {
		if version, err := c.connection.GetServerVersion(ctx); err == nil {
			c.serverVersion = version
		}
		c.foreignKeys = c.readForeignKeys(ctx)
	})
}

// readForeignKeys asks the engine whether it is enforcing foreign keys.
//
// Only SQLite has an answer that changes: the other two enforce them always,
// and a query asking would be a round trip for a constant.
func (c *schemaConnection) readForeignKeys(ctx context.Context) bool {
	if !strings.EqualFold(c.connection.GetDriverName(), "sqlite") {
		return true
	}
	value, err := c.connection.Scalar(ctx, "pragma foreign_keys", nil, false)
	if err != nil {
		// A pragma that cannot be read is not a reason to refuse the migration.
		// The grammar uses this to put the setting back the way it found it, and
		// on is the setting a database is in unless somebody changed it.
		return true
	}
	switch v := value.(type) {
	case bool:
		return v
	case int64:
		return v != 0
	case string:
		return v == "1" || strings.EqualFold(v, "on") || strings.EqualFold(v, "true")
	}
	return true
}

func (c *schemaConnection) GetSchemaGrammar() schema.Grammar { return c.grammar }

// GetConfig narrows the connection's any to the string the schema reads.
func (c *schemaConnection) GetConfig(option string) string {
	value, _ := c.connection.GetConfig(option).(string)
	return value
}

func (c *schemaConnection) GetTablePrefix() string { return c.connection.GetTablePrefix() }
func (c *schemaConnection) GetDriverName() string  { return c.connection.GetDriverName() }

func (c *schemaConnection) GetServerVersion() string {
	c.Prepare(context.Background())
	return c.serverVersion
}

// IsMaria reports whether this is MariaDB rather than MySQL.
//
// The driver name answers it when the project registered one; the version
// string answers it when the project registered "mysql" for both, which is what
// most drivers do.
func (c *schemaConnection) IsMaria() bool {
	if strings.EqualFold(c.connection.GetDriverName(), "mariadb") {
		return true
	}
	return strings.Contains(strings.ToLower(c.GetServerVersion()), "mariadb")
}

func (c *schemaConnection) ForeignKeyConstraintsEnabled() bool {
	c.Prepare(context.Background())
	return c.foreignKeys
}

// Statement runs DDL. It carries no bindings, because DDL carries none.
func (c *schemaConnection) Statement(ctx context.Context, statement string) error {
	c.Prepare(ctx)
	_, err := c.connection.Statement(ctx, statement, nil)
	return err
}

// Select reads metadata, always from the write connection: a table created a
// moment ago may not have reached a replica yet.
func (c *schemaConnection) Select(ctx context.Context, statement string) ([]schema.Record, error) {
	c.Prepare(ctx)
	rows, err := c.connection.Select(ctx, statement, nil, false)
	if err != nil {
		return nil, err
	}
	return asSchemaRecords(rows), nil
}

// Scalar reads one value, from the write connection for the same reason.
func (c *schemaConnection) Scalar(ctx context.Context, statement string) (any, error) {
	c.Prepare(ctx)
	return c.connection.Scalar(ctx, statement, nil, false)
}

// The adapter satisfies the interface it exists for, checked here rather than
// at the call site.
var _ schema.Connection = (*schemaConnection)(nil)
