package schema

import "fmt"

// Grammar is what a Blueprint needs of the thing that spells its DDL.
//
// It is an interface, declared in the package that consumes it, because the
// concrete grammars live in schema/grammars and naming them here would close an
// import cycle: grammars imports schema for *Blueprint, so schema cannot import
// grammars for the type. It is the same arrangement query makes for its own
// grammar.
//
// Every command compiler has one signature, ([]string, error). A nil slice means
// the command produced no statement, which is how SQLite says "handled on table
// creation". A grammar that does not support a command returns an error rather
// than being absent, because a missing method that silently skips the command is
// how a migration ends up half applied with nothing said.
type Grammar interface {
	// CompileCreateDatabase compiles the statement that creates a database
	// named name.
	CompileCreateDatabase(name string) (string, error)

	// CompileDropDatabaseIfExists compiles the statement that drops the
	// database named name, if it exists.
	CompileDropDatabaseIfExists(name string) (string, error)

	// CompileSchemas compiles the query that lists the connection's
	// schemas.
	CompileSchemas() (string, error)

	// CompileTableExists compiles a query that reports whether table
	// exists, scoped to schema. An empty result means the driver has no
	// cheap existence query, and Builder falls back to listing the tables.
	CompileTableExists(schema, table string) string

	// CompileTables compiles the query that lists tables in the given
	// schemas. The optional withSize opts into reporting each table's
	// size: on SQLite, sizes cost a join against dbstat, so they are
	// computed only when asked for. The other drivers report the size
	// either way and ignore this argument.
	CompileTables(schemas []string, withSize ...bool) (string, error)

	// CompileViews compiles the query that lists views in the given
	// schemas.
	CompileViews(schemas []string) (string, error)

	// CompileTypes compiles the query that lists user-defined types in the
	// given schemas.
	CompileTypes(schemas []string) (string, error)

	// CompileColumns compiles the query that lists table's columns, scoped
	// to schema.
	CompileColumns(schema, table string) (string, error)

	// CompileIndexes compiles the query that lists table's indexes, scoped
	// to schema.
	CompileIndexes(schema, table string) (string, error)

	// CompileForeignKeys compiles the query that lists table's foreign
	// keys, scoped to schema.
	CompileForeignKeys(schema, table string) (string, error)

	// CompileCreate compiles the statements that create the blueprint's
	// table.
	CompileCreate(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileAdd compiles the statements that add the blueprint's new
	// columns to an existing table.
	CompileAdd(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileChange compiles the statements that alter the blueprint's
	// changed columns.
	CompileChange(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileAlter compiles the table rebuild the SQLite grammar needs
	// when the other drivers can alter the table directly. It is on the
	// interface rather than restricted to the SQLite implementation
	// because the blueprint dispatches by command name and has no way to
	// ask a Go interface whether a method exists.
	CompileAlter(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileRenameColumn compiles the statements that rename one of the
	// blueprint's columns.
	CompileRenameColumn(blueprint *Blueprint, command *Command) ([]string, error)

	// CompilePrimary compiles the statements that add a primary key.
	CompilePrimary(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileUnique compiles the statements that add a unique index.
	CompileUnique(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileIndex compiles the statements that add an index.
	CompileIndex(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileFullText compiles the statements that add a full-text index.
	CompileFullText(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileSpatialIndex compiles the statements that add a spatial
	// index.
	CompileSpatialIndex(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileVectorIndex compiles the statements that add a vector index.
	CompileVectorIndex(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileForeign compiles the statements that add a foreign key
	// constraint.
	CompileForeign(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileDrop compiles the statements that drop the table.
	CompileDrop(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileDropIfExists compiles the statements that drop the table if
	// it exists.
	CompileDropIfExists(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileDropColumn compiles the statements that drop the blueprint's
	// columns.
	CompileDropColumn(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileDropPrimary compiles the statements that drop the primary
	// key.
	CompileDropPrimary(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileDropUnique compiles the statements that drop a unique index.
	CompileDropUnique(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileDropIndex compiles the statements that drop an index.
	CompileDropIndex(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileDropFullText compiles the statements that drop a full-text
	// index.
	CompileDropFullText(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileDropSpatialIndex compiles the statements that drop a spatial
	// index.
	CompileDropSpatialIndex(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileDropForeign compiles the statements that drop a foreign key
	// constraint.
	CompileDropForeign(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileRename compiles the statements that rename the table.
	CompileRename(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileRenameIndex compiles the statements that rename an index.
	CompileRenameIndex(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileTableComment compiles the statements that set the table's
	// comment.
	CompileTableComment(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileComment compiles the fluent command that puts a column
	// comment in a statement of its own.
	CompileComment(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileAutoIncrementStartingValues compiles the statements that set
	// a column's auto-increment starting value.
	CompileAutoIncrementStartingValues(blueprint *Blueprint, command *Command) ([]string, error)

	// CompileDropAllTables compiles the statement that drops every table
	// named in tables.
	CompileDropAllTables(tables []string) (string, error)

	// CompileDropAllViews compiles the statement that drops every view
	// named in views.
	CompileDropAllViews(views []string) (string, error)

	// CompileDropAllTypes compiles the statement that drops every type
	// named in types.
	CompileDropAllTypes(types []string) (string, error)

	// CompileEnableForeignKeyConstraints compiles the statement that turns
	// foreign key enforcement on.
	CompileEnableForeignKeyConstraints() (string, error)

	// CompileDisableForeignKeyConstraints compiles the statement that
	// turns foreign key enforcement off.
	CompileDisableForeignKeyConstraints() (string, error)

	// GetFluentCommands returns the names of the commands a driver runs
	// outside the create or alter statement, one per column.
	GetFluentCommands() []string

	// GetAlterCommands returns the names of the commands that force a
	// table rebuild. Every other driver returns none.
	GetAlterCommands() []string

	// SupportsSchemaTransactions reports whether the driver can run schema
	// changes inside a transaction.
	SupportsSchemaTransactions() bool

	// Wrap quotes value as an identifier, using this driver's quoting
	// rules.
	Wrap(value any) string

	// WrapTable quotes table as an identifier. The optional prefix
	// replaces the connection's table prefix, which is how the SQLite
	// rebuild names its temporary table.
	WrapTable(table any, prefix ...string) string

	// MaxIdentifierLength returns the longest identifier this driver
	// accepts, in bytes. Zero means the driver imposes no limit worth
	// enforcing.
	//
	// The limit belongs to the driver and not to the Blueprint, because the
	// drivers disagree on both the number and on what happens past it: one
	// truncates the name and carries on, another refuses the statement. A
	// conventional name is shortened to fit before it ever reaches the
	// server, so neither behaviour is ever reached.
	MaxIdentifierLength() int
}

// compileCommand hands one command to the grammar.
//
// The dispatch is written out as a switch on the command name, since Go has
// no way to call a method by that name at runtime. An unknown command
// returns an UnknownCommandError rather than being skipped in silence.
func compileCommand(g Grammar, blueprint *Blueprint, command *Command) ([]string, error) {
	switch command.Name {
	case "create":
		return g.CompileCreate(blueprint, command)
	case "add":
		return g.CompileAdd(blueprint, command)
	case "change":
		return g.CompileChange(blueprint, command)
	case "alter":
		return g.CompileAlter(blueprint, command)
	case "renameColumn":
		return g.CompileRenameColumn(blueprint, command)
	case "primary":
		return g.CompilePrimary(blueprint, command)
	case "unique":
		return g.CompileUnique(blueprint, command)
	case "index":
		return g.CompileIndex(blueprint, command)
	case "fulltext":
		return g.CompileFullText(blueprint, command)
	case "spatialIndex":
		return g.CompileSpatialIndex(blueprint, command)
	case "vectorIndex":
		return g.CompileVectorIndex(blueprint, command)
	case "foreign":
		return g.CompileForeign(blueprint, command)
	case "drop":
		return g.CompileDrop(blueprint, command)
	case "dropIfExists":
		return g.CompileDropIfExists(blueprint, command)
	case "dropColumn":
		return g.CompileDropColumn(blueprint, command)
	case "dropPrimary":
		return g.CompileDropPrimary(blueprint, command)
	case "dropUnique":
		return g.CompileDropUnique(blueprint, command)
	case "dropIndex":
		return g.CompileDropIndex(blueprint, command)
	case "dropFullText":
		return g.CompileDropFullText(blueprint, command)
	case "dropSpatialIndex":
		return g.CompileDropSpatialIndex(blueprint, command)
	case "dropForeign":
		return g.CompileDropForeign(blueprint, command)
	case "rename":
		return g.CompileRename(blueprint, command)
	case "renameIndex":
		return g.CompileRenameIndex(blueprint, command)
	case "tableComment":
		return g.CompileTableComment(blueprint, command)
	case "comment":
		return g.CompileComment(blueprint, command)
	case "autoIncrementStartingValues":
		return g.CompileAutoIncrementStartingValues(blueprint, command)
	default:
		return nil, &UnknownCommandError{Name: command.Name}
	}
}

// UnknownCommandError is returned when a blueprint carries a command no grammar
// knows how to compile.
type UnknownCommandError struct{ Name string }

func (e *UnknownCommandError) Error() string {
	return "schema: no grammar compiles the blueprint command " + e.Name
}

// IdentifierTooLongError is returned when a name the caller wrote out by hand
// is longer than the driver accepts.
//
// A conventional name is shortened to fit and never reaches this, so the error
// only ever reports a name the caller chose. It is an error rather than a
// silent shortening because the caller who names an index means to use that
// name again -- to drop it, to rename it, to read it in a report -- and a name
// quietly changed underneath is worse than a migration that refuses to run.
type IdentifierTooLongError struct {
	// Name is the identifier as the caller wrote it.
	Name string

	// Limit is the longest identifier the driver accepts, in bytes.
	Limit int
}

func (e *IdentifierTooLongError) Error() string {
	return fmt.Sprintf(
		"schema: identifier %q is %d bytes, over this driver's limit of %d; name it something shorter",
		e.Name, len(e.Name), e.Limit,
	)
}
