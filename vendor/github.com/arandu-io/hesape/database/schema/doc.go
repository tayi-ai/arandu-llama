// Package schema declares tables and indexes: the Blueprint a migration writes
// against, the Builder that executes it, and the dump-and-load schema state.
//
// # Where the SQL comes from
//
// A Blueprint is a list of columns and commands. It builds no SQL of its own:
// ToSQL walks the commands and hands each one to the Grammar, which is where the
// three dialects differ. The dispatch is an explicit switch in compileCommand,
// so a grammar that does not support a command returns an error rather than
// being silently absent. A fluent command a driver does not declare is skipped
// by GetFluentCommands, which only adds the commands the grammar asked for.
//
// There is one Builder rather than one per driver, because what differs between
// drivers is the SQL and the SQL is the Grammar's. The exception is
// RefreshDatabaseFile, which touches a file rather than the database and so
// checks the driver itself.
//
// # No authorization credential on this path
//
// Nothing in this component asks for one, and that is a decision rather than an
// omission. Blueprint and Grammar build strings and decide nothing; Builder
// executes, and no method on it asks for one either.
//
// DDL has nothing to offer such a credential. A create table names a table, not
// rows, so there is no tenant column to filter on, and the statement is run as a
// pipeline step with no request behind it, so there is no subject to attribute
// it to. The only credential this component could hold is one its caller
// invented, and a parameter that looks like enforcement while enforcing nothing
// is worse than no parameter, because a reader stops looking.
//
// The path to application rows is a different one and is unaffected: it is still
// closed on every method. What a migration writes here looks like this:
//
//	if err := builder.Create(ctx, "users", func(t *schema.Blueprint) {
//	    t.ID("")
//	    t.String("email", 255).Unique()
//	    t.Timestamps()
//	}); err != nil { ... }
//
// # Naming
//
// An initialism is upper case: ID, UUID, ULID, JSON, JSONB, SRID, IPAddress.
// Optional arguments of one type are variadic -- String(column, length...),
// Index(columns, name...) -- and where they are not, the parameter is required
// and the empty string selects the default: ULID("") names its column 'ulid'
// and SoftDeletes("") names 'deleted_at'. A value that is both readable and
// settable keeps the setter as the plain verb and prefixes the reader with Get:
// GetTable, GetEngine, GetCharset, GetCollation, GetTemporary, GetAfter,
// GetColumns, GetCommands, GetState, and on ColumnDefinition GetName, GetType,
// GetNullable, GetDefault, GetComment.
//
// MariaDB and MySQL share one schema state: mysqldump is the same program, and
// the one difference -- omitting --set-gtid-purged -- is a branch
// MySqlSchemaState takes on Connection.IsMaria.
package schema
