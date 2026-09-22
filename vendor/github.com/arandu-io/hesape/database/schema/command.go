package schema

// Command is one command of a Blueprint.
//
// Each Name reads a different subset of the fields; the union of them is spelled
// out here so the grammar can read them, exactly as query.Where does for the
// query builder's clauses.
//
// IndexDefinition and ForeignKeyDefinition are not separate values but typed
// handles over this struct, so whatever the caller chains onto them is what the
// grammar compiles.
type Command struct {
	// Name is the command: create, add, change, dropColumn, renameIndex,
	// tableComment. The grammar is selected by it.
	Name string

	// ShouldBeSkipped marks a command the grammar has already folded into
	// another statement. The MySQL grammar sets it on the primary key command
	// it folded into the create table statement.
	ShouldBeSkipped bool

	// Column is the column an add, change, comment or
	// autoIncrementStartingValues command carries.
	Column *ColumnDefinition

	// Columns is the column list of an index, foreign key or dropColumn
	// command. It is []any rather than []string because a raw index passes an
	// Expression, as rawIndex does.
	Columns []any

	// From and To carry rename, renameColumn and renameIndex.
	From string
	To   string

	// Index is the index name an index or drop index command works on.
	Index string

	// Algorithm is the index algorithm: using btree, using hash, using gist.
	Algorithm string

	// OperatorClass is the Postgres operator class of a spatial or vector index.
	OperatorClass string

	// Language is the Postgres full text search configuration.
	Language string

	// Comment is the table comment of a tableComment command.
	Comment string

	// References, On, OnDelete and OnUpdate are the foreign key.
	References []any
	On         any
	OnDelete   string
	OnUpdate   string

	// Deferrable, InitiallyImmediate, NotValid and NullsNotDistinct are the
	// Postgres constraint modifiers. They are pointers because "not set" must
	// be distinguishable from false: an unset deferrable emits nothing, and a
	// false one emits "not deferrable".
	Deferrable         *bool
	InitiallyImmediate *bool
	NotValid           *bool
	NullsNotDistinct   *bool

	// Online is Postgres' concurrent index build.
	Online bool

	// Lock and Instant are the MySQL DDL lock mode and the instant algorithm.
	Lock    string
	Instant bool

	// isColumnDefinition marks the placeholder a bare ColumnDefinition occupies
	// in the command list while the table is being altered. addImpliedCommands
	// rewrites it into an add or change command, and the marker is how that
	// rewrite finds it here.
	isColumnDefinition bool
}

// NewCommand returns a new Command named name.
func NewCommand(name string) *Command {
	return &Command{Name: name}
}
