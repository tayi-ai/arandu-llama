package schema

import "context"

// Record is one row as the connection hands it back: column name to value.
// It is the same alias query uses, for the same reason.
type Record = map[string]any

// Connection is what this component needs of the thing that runs its
// statements.
//
// It is an interface declared in the package that consumes it, because the
// concrete connection lives in the parent package and a Go import the other way
// would close a cycle. The method set is only what the schema component
// actually calls -- nothing here is a general purpose database handle.
//
// The Process* methods are on the connection rather than on a Processor value
// because the post processor is per driver, and this package must not know which
// driver it is talking to.
type Connection interface {
	// GetSchemaGrammar returns the grammar this connection compiles schema
	// statements with.
	GetSchemaGrammar() Grammar

	// GetConfig returns the connection's value for option, or the empty
	// string if it was never set -- true for every option this component
	// reads (charset, collation, engine, prefix_indexes, use_native_json,
	// use_native_jsonb).
	GetConfig(option string) string

	// GetTablePrefix returns the prefix this connection adds to every table
	// name.
	GetTablePrefix() string

	// GetDriverName reports the driver name: mysql, mariadb, pgsql or
	// sqlite. Blueprint switches on it to select driver-specific behavior.
	GetDriverName() string

	// GetServerVersion returns the database server's version string.
	GetServerVersion() string

	// IsMaria reports whether the connection is talking to MariaDB rather
	// than MySQL. It is false on every connection that is not MySQL, as the
	// MySQL grammar is the only caller.
	IsMaria() bool

	// ForeignKeyConstraintsEnabled reports whether the connection is currently
	// enforcing foreign keys.
	//
	// The SQLite grammar reads it with "pragma foreign_keys" in the middle
	// of compiling a table rebuild, which has to switch them off and then
	// put them back exactly as it found them -- switching them on
	// unconditionally would start enforcing constraints a caller had
	// deliberately suspended. The read is on the connection here so that no
	// pragma of one driver's is written into a package that must not know
	// which driver it is talking to.
	ForeignKeyConstraintsEnabled() bool

	// Statement runs a statement that returns no rows.
	Statement(ctx context.Context, query string) error

	// Select runs a query and returns the rows. Schema reads always go to
	// the write connection, because a table created a moment ago may not
	// have reached a replica yet.
	Select(ctx context.Context, query string) ([]Record, error)

	// Scalar runs a query and returns the first column of the first row.
	Scalar(ctx context.Context, query string) (any, error)

	// ProcessTables turns raw query rows into a slice of TableInfo.
	ProcessTables(rows []Record) []TableInfo

	// ProcessViews turns raw query rows into a slice of ViewInfo.
	ProcessViews(rows []Record) []ViewInfo

	// ProcessColumns turns raw query rows into a slice of ColumnInfo.
	ProcessColumns(rows []Record) []ColumnInfo

	// ProcessIndexes turns raw query rows into a slice of IndexInfo.
	ProcessIndexes(rows []Record) []IndexInfo

	// ProcessForeignKeys turns raw query rows into a slice of ForeignKeyInfo.
	ProcessForeignKeys(rows []Record) []ForeignKeyInfo

	// ProcessSchemas turns raw query rows into a slice of SchemaInfo.
	ProcessSchemas(rows []Record) []SchemaInfo
}

// SchemaInfo is one entry of the slice Builder.GetSchemas returns.
type SchemaInfo struct {
	Name    string
	Path    string
	Default bool
}

// TableInfo is one entry of the slice Builder.GetTables returns.
type TableInfo struct {
	Name                string
	Schema              string
	SchemaQualifiedName string
	Size                int64
	Comment             string
	Collation           string
	Engine              string
}

// ViewInfo is one entry of the slice Builder.GetViews returns.
type ViewInfo struct {
	Name                string
	Schema              string
	SchemaQualifiedName string
	Definition          string
}

// ColumnInfo is one entry of the slice Builder.GetColumns returns.
//
// TypeName is the bare type, 'varchar', and Type is the full definition,
// 'varchar(255)'. GetColumnType returns one or the other, and which one is
// the whole point of its fullDefinition argument.
type ColumnInfo struct {
	Name          string
	Type          string
	TypeName      string
	Nullable      bool
	Default       any
	AutoIncrement bool
	Comment       string
	Collation     string
	Generation    *Generation
}

// Generation is the generation key of a ColumnInfo: how a computed column is
// computed, or nil when the column is not computed.
type Generation struct {
	Type       string
	Expression string
}

// IndexInfo is one entry of the slice Builder.GetIndexes returns.
type IndexInfo struct {
	Name    string
	Columns []string
	Type    string
	Unique  bool
	Primary bool
}

// ForeignKeyInfo is one entry of the slice Builder.GetForeignKeys returns.
type ForeignKeyInfo struct {
	Name           string
	Columns        []string
	ForeignSchema  string
	ForeignTable   string
	ForeignColumns []string
	OnUpdate       string
	OnDelete       string
}

// Model is what ForeignIDFor needs of a model.
//
// The caller passes the model itself, and the interface names only the four
// methods Blueprint calls on it.
type Model interface {
	// GetTable returns the model's table name.
	GetTable() string

	// GetForeignKey returns the column name used when referencing this
	// model with a foreign key.
	GetForeignKey() string

	// GetKeyName returns the model's primary key column name.
	GetKeyName() string

	// GetKeyType returns the model's primary key type: "int" or "string".
	GetKeyType() string
}

// ULIDModel is how a model says it keys on ULIDs rather than UUIDs.
//
// A model that does not implement this interface is treated as a UUID
// model.
type ULIDModel interface {
	// UsesULIDs reports whether the model's key is a ULID.
	UsesULIDs() bool
}
