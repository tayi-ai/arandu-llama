package grammars

import (
	"fmt"
	"strings"

	"github.com/arandu-io/hesape/database/query"
	"github.com/arandu-io/hesape/database/schema"
)

// SQLiteGrammar turns a Blueprint into SQLite DDL.
//
// SQLite cannot alter most of a table, so several commands compile to nothing
// here and are carried out by the table rebuild CompileAlter emits instead.
type SQLiteGrammar struct{ *BaseGrammar }

// NewSQLiteGrammar constructs a SQLiteGrammar bound to connection, with
// the column modifiers and serial types SQLite needs.
func NewSQLiteGrammar(connection schema.Connection) *SQLiteGrammar {
	g := &SQLiteGrammar{&BaseGrammar{
		conn:      connection,
		modifiers: []string{"Increment", "Nullable", "Default", "Collate", "VirtualAs", "StoredAs"},
		serials:   []string{"bigInteger", "integer", "mediumInteger", "smallInteger", "tinyInteger"},
	}}
	g.self = g
	return g
}

// GetAlterCommands lists the commands SQLite cannot express as an alter
// statement -- change, primary, dropPrimary, foreign, and dropForeign --
// and which therefore force the table to be rebuilt. dropColumn joins the
// list too on a server older than 3.35, before SQLite could drop a column
// directly.
func (g *SQLiteGrammar) GetAlterCommands() []string {
	commands := []string{"change", "primary", "dropPrimary", "foreign", "dropForeign"}
	if !atLeast(g.conn.GetServerVersion(), "3.35") {
		commands = append(commands, "dropColumn")
	}
	return commands
}

// CompileSQLCreateStatement builds the query that reads the CREATE
// statement SQLite stored in sqlite_master for name, matching typ's first
// value as the object kind or "table" when typ is omitted.
func (g *SQLiteGrammar) CompileSQLCreateStatement(schemaName, name string, typ ...string) string {
	kind := "table"
	if len(typ) > 0 && typ[0] != "" {
		kind = typ[0]
	}
	return fmt.Sprintf(`select "sql" from %s.sqlite_master where type = %s and name = %s`,
		g.wrapValue(defaultSchema(schemaName)), g.QuoteString(kind), g.QuoteString(name))
}

// CompileTableExists builds the query that reports whether table exists
// in schemaName's sqlite_master.
func (g *SQLiteGrammar) CompileTableExists(schemaName, table string) string {
	return fmt.Sprintf(`select exists (select 1 from %s.sqlite_master where name = %s and type = 'table') as "exists"`,
		g.wrapValue(defaultSchema(schemaName)), g.QuoteString(table))
}

// CompileTables builds the query that lists tables across schemas. Asking
// for the size joins against dbstat, a virtual table not every build of
// SQLite carries -- CompileDbstatExists is how the caller finds out before
// asking.
func (g *SQLiteGrammar) CompileTables(schemas []string, withSize ...bool) (string, error) {
	where := ""
	switch {
	case len(schemas) > 1:
		where = " tl.schema in (" + g.QuoteString(schemas) + ") and"
	case len(schemas) == 1:
		where = " tl.schema = " + g.QuoteString(schemas[0]) + " and"
	}

	size := ""
	if len(withSize) > 0 && withSize[0] {
		size = ", (select sum(s.pgsize) " +
			"from (select tl.name as name union select il.name as name from pragma_index_list(tl.name, tl.schema) as il) as es " +
			"join dbstat(tl.schema) as s on s.name = es.name) as size"
	}

	return "select tl.name as name, tl.schema as schema" + size +
		" from pragma_table_list as tl where" + where +
		` tl.type in ('table', 'virtual') and tl.name not like 'sqlite\_%' escape '\' ` +
		"order by tl.schema, tl.name", nil
}

// CompileDbstatExists builds the query that reports whether this build of
// SQLite has the dbstat virtual table that table sizes come from.
func (g *SQLiteGrammar) CompileDbstatExists() string {
	return "select exists (select 1 from pragma_compile_options where compile_options = 'ENABLE_DBSTAT_VTAB') as enabled"
}

// CompileLegacyTables builds the query that lists tables for a server old
// enough to lack pragma_table_list.
func (g *SQLiteGrammar) CompileLegacyTables(schemaName string, withSize ...bool) string {
	name := defaultSchema(schemaName)
	if len(withSize) > 0 && withSize[0] {
		return fmt.Sprintf(
			"select m.tbl_name as name, %s as schema, sum(s.pgsize) as size from %s.sqlite_master as m "+
				"join dbstat(%s) as s on s.name = m.name "+
				`where m.type in ('table', 'index') and m.tbl_name not like 'sqlite\_%%' escape '\' `+
				"group by m.tbl_name order by m.tbl_name",
			g.QuoteString(name), g.wrapValue(name), g.QuoteString(name))
	}
	return fmt.Sprintf(
		"select name, %s as schema from %s.sqlite_master "+
			`where type = 'table' and name not like 'sqlite\_%%' escape '\' order by name`,
		g.QuoteString(name), g.wrapValue(name))
}

// CompileRebuild builds the VACUUM statement that rebuilds the named
// schema, defaulting to "main".
func (g *SQLiteGrammar) CompileRebuild(schemaName ...string) string {
	return "vacuum " + g.wrapValue(defaultSchema(firstOr(schemaName, "main")))
}

// CompileViews builds the query that lists views across schemas, from
// pragma_table_list.
func (g *SQLiteGrammar) CompileViews(schemas []string) (string, error) {
	where := ""
	switch {
	case len(schemas) > 1:
		where = "schema in (" + g.QuoteString(schemas) + ") and "
	case len(schemas) == 1:
		where = "schema = " + g.QuoteString(schemas[0]) + " and "
	}
	return "select name, schema, sql as definition from pragma_table_list where " + where +
		"type = 'view' order by schema, name", nil
}

// CompileColumns builds the query that lists table's columns, in column
// order, from pragma_table_xinfo.
func (g *SQLiteGrammar) CompileColumns(schemaName, table string) (string, error) {
	return fmt.Sprintf(
		`select name, type, not "notnull" as "nullable", dflt_value as "default", pk as "primary", hidden as "extra" `+
			"from pragma_table_xinfo(%s, %s) order by cid asc",
		g.QuoteString(table), g.QuoteString(defaultSchema(schemaName))), nil
}

// CompileIndexes builds the query that lists table's indexes, unioning
// the implicit primary-key index from pragma_table_xinfo with the named
// indexes from pragma_index_list.
func (g *SQLiteGrammar) CompileIndexes(schemaName, table string) (string, error) {
	quotedTable := g.QuoteString(table)
	quotedSchema := g.QuoteString(defaultSchema(schemaName))
	return fmt.Sprintf(
		`select 'primary' as name, group_concat(col) as columns, 1 as "unique", 1 as "primary" `+
			"from (select name as col from pragma_table_xinfo(%s, %s) where pk > 0 order by pk, cid) group by name "+
			`union select name, group_concat(col) as columns, "unique", origin = 'pk' as "primary" `+
			"from (select il.*, ii.name as col from pragma_index_list(%s, %s) il, pragma_index_info(il.name, %s) ii order by il.seq, ii.seqno) "+
			`group by name, "unique", "primary"`,
		quotedTable, quotedSchema, quotedTable, quotedSchema, quotedSchema), nil
}

// CompileForeignKeys builds the query that lists table's foreign keys
// from pragma_foreign_key_list, grouping each key's columns together.
func (g *SQLiteGrammar) CompileForeignKeys(schemaName, table string) (string, error) {
	quotedSchema := g.QuoteString(defaultSchema(schemaName))
	return fmt.Sprintf(
		`select group_concat("from") as columns, %s as foreign_schema, "table" as foreign_table, `+
			`group_concat("to") as foreign_columns, on_update, on_delete `+
			"from (select * from pragma_foreign_key_list(%s, %s) order by id desc, seq) "+
			`group by id, "table", on_update, on_delete`,
		quotedSchema, g.QuoteString(table), quotedSchema), nil
}

// CompileCreate builds the CREATE TABLE statement for blueprint. SQLite
// takes its foreign keys and its primary key inside the create statement,
// because it cannot add either afterwards.
func (g *SQLiteGrammar) CompileCreate(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	columns, err := g.getColumns(blueprint)
	if err != nil {
		return nil, err
	}
	create := "create"
	if blueprint.GetTemporary() {
		create = "create temporary"
	}
	return one(fmt.Sprintf("%s table %s (%s%s%s)",
		create,
		g.WrapTable(blueprint),
		strings.Join(columns, ", "),
		g.addForeignKeys(g.getCommandsByName(blueprint, "foreign")),
		g.addPrimaryKeys(g.getCommandByName(blueprint, "primary")))), nil
}

// addForeignKeys renders the ", foreign key(...) references ..." clause
// for each of foreignKeys, for inclusion in a create statement.
func (g *SQLiteGrammar) addForeignKeys(foreignKeys []*schema.Command) string {
	sql := ""
	for _, foreign := range foreignKeys {
		sql += g.getForeignKey(foreign)
	}
	return sql
}

// getForeignKey renders foreign's own foreign-key clause, including its
// ON DELETE and ON UPDATE rules when set.
func (g *SQLiteGrammar) getForeignKey(foreign *schema.Command) string {
	sql := fmt.Sprintf(", foreign key(%s) references %s(%s)",
		g.Columnize(foreign.Columns),
		g.WrapTable(foreign.On),
		g.Columnize(foreign.References))

	if foreign.OnDelete != "" {
		sql += " on delete " + foreign.OnDelete
	}
	if foreign.OnUpdate != "" {
		sql += " on update " + foreign.OnUpdate
	}
	return sql
}

// addPrimaryKeys renders the ", primary key (...)" clause for primary, or
// an empty string when there is no primary key to add.
func (g *SQLiteGrammar) addPrimaryKeys(primary *schema.Command) string {
	if primary == nil {
		return ""
	}
	return ", primary key (" + g.Columnize(primary.Columns) + ")"
}

// CompileAdd builds the ALTER TABLE ADD COLUMN statement for
// command.Column.
func (g *SQLiteGrammar) CompileAdd(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	definition, err := g.getColumn(blueprint, command.Column)
	if err != nil {
		return nil, err
	}
	return one("alter table " + g.WrapTable(blueprint) + " add column " + definition), nil
}

// CompilePrimary emits nothing: SQLite takes its primary key inside the
// create statement or the table rebuild, never as a separate alter.
func (g *SQLiteGrammar) CompilePrimary(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return nil, nil
}

// CompileChange emits nothing: a changed column is handled by the table
// rebuild in CompileAlter, since SQLite cannot alter a column in place.
func (g *SQLiteGrammar) CompileChange(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return nil, nil
}

// CompileForeign emits nothing: SQLite takes its foreign keys inside the
// create statement or the table rebuild, never as a separate alter.
func (g *SQLiteGrammar) CompileForeign(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return nil, nil
}

// CompileUnique builds the CREATE UNIQUE INDEX statement for command's
// columns.
func (g *SQLiteGrammar) CompileUnique(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	schemaName, table, err := schema.ParseSchemaAndTable(blueprint.GetTable())
	if err != nil {
		return nil, err
	}
	return one(fmt.Sprintf("create unique index %s%s on %s (%s)",
		g.schemaPrefix(schemaName), g.Wrap(command.Index), g.WrapTable(table), g.Columnize(command.Columns))), nil
}

// CompileIndex builds the CREATE INDEX statement for command's columns.
func (g *SQLiteGrammar) CompileIndex(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	schemaName, table, err := schema.ParseSchemaAndTable(blueprint.GetTable())
	if err != nil {
		return nil, err
	}
	return one(fmt.Sprintf("create index %s%s on %s (%s)",
		g.schemaPrefix(schemaName), g.Wrap(command.Index), g.WrapTable(table), g.Columnize(command.Columns))), nil
}

// CompileDrop builds the DROP TABLE statement for blueprint.
func (g *SQLiteGrammar) CompileDrop(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return one("drop table " + g.WrapTable(blueprint)), nil
}

// CompileDropIfExists builds the DROP TABLE IF EXISTS statement for
// blueprint.
func (g *SQLiteGrammar) CompileDropIfExists(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return one("drop table if exists " + g.WrapTable(blueprint)), nil
}

// CompileDropAllTables builds the statements that clear every table, index
// and trigger out of SQLite's catalogue.
//
// SQLite clears its catalogue rather than naming tables individually, so the
// list is not read: what it names is what "all tables" means here anyway, and
// a shorter list would not make this drop less.
//
// This used to be documented as taking schema names where the other grammars
// take table names -- a divergence from its own interface, written down and
// never checked against the caller. Builder.DropAllTables passes the qualified
// table names, so the schema came out as "main.arandu_migrations" and SQLite
// answered that no such schema existed. It is "main" now, which is the only
// schema a connection has without an ATTACH.
//
// The pragma is what makes the delete legal: sqlite_master is read-only
// otherwise, and the statement is refused with "table sqlite_master may not be
// modified". It is turned back off in the same batch so the connection does not
// carry a writable catalogue into whatever runs next.
//
// The vacuum is not tidiness either. Deleting the rows empties the catalogue
// table, and the connection goes on answering pragma_table_list from the schema
// it already has in memory -- so the tables are gone from disk and still listed,
// which is the shape of a wipe that reports success and drops nothing. The
// vacuum rebuilds the file and reloads the schema. It cannot run inside a
// transaction, which is why this is a wipe a caller runs before migrating
// rather than a step inside one.
func (g *SQLiteGrammar) CompileDropAllTables([]string) (string, error) {
	return "pragma writable_schema = 1; " +
		"delete from \"main\".sqlite_master where type in ('table', 'index', 'trigger'); " +
		"pragma writable_schema = 0; " +
		"vacuum;", nil
}

// CompileDropAllViews builds the statements that clear every view out of
// SQLite's catalogue, for the reasons CompileDropAllTables gives.
func (g *SQLiteGrammar) CompileDropAllViews([]string) (string, error) {
	return "pragma writable_schema = 1; " +
		"delete from \"main\".sqlite_master where type in ('view'); " +
		"pragma writable_schema = 0; " +
		"vacuum;", nil
}

// CompileDropColumn builds the ALTER TABLE DROP COLUMN statements for
// command's columns. Before SQLite 3.35 there is no "drop column", and the
// rebuild does it instead.
func (g *SQLiteGrammar) CompileDropColumn(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	if !atLeast(g.conn.GetServerVersion(), "3.35") {
		return nil, nil
	}
	table := g.WrapTable(blueprint)
	var statements []string
	for _, column := range g.PrefixArray("drop column", g.WrapArray(command.Columns)) {
		statements = append(statements, "alter table "+table+" "+column)
	}
	return statements, nil
}

// CompileDropPrimary emits nothing: dropping a primary key is handled by
// the table rebuild in CompileAlter, since SQLite cannot drop one in place.
func (g *SQLiteGrammar) CompileDropPrimary(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return nil, nil
}

// CompileDropUnique builds the DROP INDEX statement for a unique key,
// which on SQLite is the same as dropping any other index.
func (g *SQLiteGrammar) CompileDropUnique(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return g.CompileDropIndex(blueprint, command)
}

// CompileDropIndex builds the DROP INDEX statement for command.Index.
func (g *SQLiteGrammar) CompileDropIndex(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	schemaName, _, err := schema.ParseSchemaAndTable(blueprint.GetTable())
	if err != nil {
		return nil, err
	}
	return one("drop index " + g.schemaPrefix(schemaName) + g.Wrap(command.Index)), nil
}

// CompileDropForeign refuses to drop a foreign key by name, because
// SQLite's foreign keys have no names to drop by; the caller must name the
// columns instead. Dropping by columns is handled by the table rebuild in
// CompileAlter.
func (g *SQLiteGrammar) CompileDropForeign(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	if len(command.Columns) == 0 {
		return nil, unsupported("dropping foreign keys by name")
	}
	return nil, nil
}

// CompileRename builds the ALTER TABLE RENAME TO statement from
// blueprint to command.To.
func (g *SQLiteGrammar) CompileRename(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return one("alter table " + g.WrapTable(blueprint) + " rename to " + g.WrapTable(command.To)), nil
}

// CompileRenameIndex refuses to rename an index.
//
// SQLite has no rename for an index: doing it means reading the index back
// off the server, dropping it, and creating it again under the new name.
// That read happens in the middle of compiling, which this grammar does
// without a connection, so the operation is refused with the reason instead
// of emitting a statement SQLite would reject.
func (g *SQLiteGrammar) CompileRenameIndex(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return nil, fmt.Errorf("%w: renaming an index on SQLite means dropping and recreating it, which needs the index read back from the server; drop and create it in the migration instead", schema.ErrUnsupported)
}

// CompileEnableForeignKeyConstraints builds the pragma statement that turns
// foreign key enforcement on.
func (g *SQLiteGrammar) CompileEnableForeignKeyConstraints() (string, error) {
	return g.Pragma("foreign_keys", "1"), nil
}

// CompileDisableForeignKeyConstraints builds the pragma statement that
// turns foreign key enforcement off.
func (g *SQLiteGrammar) CompileDisableForeignKeyConstraints() (string, error) {
	return g.Pragma("foreign_keys", "0"), nil
}

// Pragma renders "pragma key" to read its value, or "pragma key = value" to
// set it when value is not empty.
func (g *SQLiteGrammar) Pragma(key, value string) string {
	if value == "" {
		return "pragma " + key
	}
	return "pragma " + key + " = " + value
}

// CompileAlter builds the table rebuild: the sequence of statements SQLite
// needs for anything an alter statement cannot express.
//
// SQLite can add a column and little else, so anything further is done by
// building the table the blueprint describes under a temporary name, copying
// the rows across, dropping the original, and renaming. The shape being
// built comes from the BlueprintState, which read the table before the
// first command was compiled and has been kept current since.
func (g *SQLiteGrammar) CompileAlter(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	state := blueprint.GetState()
	if state == nil {
		return nil, nil
	}

	var columnNames []string
	autoIncrementColumn := ""
	var columns []string

	for _, column := range state.GetColumns() {
		if column.GetAutoIncrement() {
			autoIncrementColumn = column.GetName()
		}
		if !isGenerated(column) {
			columnNames = append(columnNames, g.Wrap(column))
		}

		definition := column.GetFullTypeDefinition()
		if definition == "" {
			typ, err := g.typeOf(column)
			if err != nil {
				return nil, err
			}
			definition = typ
		}
		modified, err := g.addModifiers(g.Wrap(column)+" "+definition, blueprint, column)
		if err != nil {
			return nil, err
		}
		columns = append(columns, modified)
	}

	var indexes []string
	for _, index := range state.GetIndexes() {
		var statements []string
		var err error
		switch index.GetCommand().Name {
		case "unique":
			statements, err = g.CompileUnique(blueprint, index.GetCommand())
		default:
			statements, err = g.CompileIndex(blueprint, index.GetCommand())
		}
		if err != nil {
			return nil, err
		}
		indexes = append(indexes, statements...)
	}

	_, tableName, err := schema.ParseSchemaAndTable(blueprint.GetTable())
	if err != nil {
		return nil, err
	}

	tempTable := g.WrapTable(blueprint, "__temp__"+g.conn.GetTablePrefix())
	table := g.WrapTable(blueprint)
	names := strings.Join(columnNames, ", ")

	foreignKeys := make([]*schema.Command, 0, len(state.GetForeignKeys()))
	for _, foreign := range state.GetForeignKeys() {
		foreignKeys = append(foreignKeys, foreign.GetCommand())
	}

	primaryKey := ""
	if autoIncrementColumn == "" && state.GetPrimaryKey() != nil {
		primaryKey = g.addPrimaryKeys(state.GetPrimaryKey().GetCommand())
	}

	enabled := g.conn.ForeignKeyConstraintsEnabled()

	var statements []string
	if enabled {
		disable, err := g.CompileDisableForeignKeyConstraints()
		if err != nil {
			return nil, err
		}
		statements = append(statements, disable)
	}

	statements = append(statements,
		fmt.Sprintf("create table %s (%s%s%s)", tempTable, strings.Join(columns, ", "), g.addForeignKeys(foreignKeys), primaryKey),
		fmt.Sprintf("insert into %s (%s) select %s from %s", tempTable, names, names, table),
		"drop table "+table,
		fmt.Sprintf("alter table %s rename to %s", tempTable, g.WrapTable(tableName)),
	)
	statements = append(statements, indexes...)

	if enabled {
		enable, err := g.CompileEnableForeignKeyConstraints()
		if err != nil {
			return nil, err
		}
		statements = append(statements, enable)
	}

	return statements, nil
}

// schemaPrefix renders the "schema." an index statement carries when the table
// was named with one.
func (g *SQLiteGrammar) schemaPrefix(schemaName string) string {
	if schemaName == "" {
		return ""
	}
	return g.wrapValue(schemaName) + "."
}

// typeOf spells a column type. SQLite has five storage classes, so most column
// types collapse onto varchar, integer or text.
func (g *SQLiteGrammar) typeOf(column *schema.ColumnDefinition) (string, error) {
	switch column.GetType() {
	case "char", "string", "uuid", "ipAddress", "macAddress":
		return "varchar", nil
	case "tinyText", "text", "mediumText", "longText":
		return "text", nil
	case "integer", "bigInteger", "mediumInteger", "tinyInteger", "smallInteger":
		return "integer", nil
	case "float":
		return "float", nil
	case "double":
		return "double", nil
	case "decimal":
		return "numeric", nil
	case "boolean":
		return "tinyint(1)", nil
	case "enum":
		return fmt.Sprintf(`varchar check ("%s" in (%s))`,
			column.GetName(), g.QuoteString(column.GetAllowed())), nil
	case "json":
		if g.conn.GetConfig("use_native_json") != "" {
			return "json", nil
		}
		return "text", nil
	case "jsonb":
		if g.conn.GetConfig("use_native_jsonb") != "" {
			return "jsonb", nil
		}
		return "text", nil
	case "date":
		if column.GetUseCurrent() {
			column.Default(query.Raw("CURRENT_DATE"))
		}
		return "date", nil
	case "dateTime", "dateTimeTz", "timestamp", "timestampTz":
		if column.GetUseCurrent() {
			column.Default(query.Raw("CURRENT_TIMESTAMP"))
		}
		return "datetime", nil
	case "time", "timeTz":
		return "time", nil
	case "year":
		if column.GetUseCurrent() {
			column.Default(query.Raw("(CAST(strftime('%Y', 'now') AS INTEGER))"))
		}
		return "integer", nil
	case "binary":
		return "blob", nil
	case "geometry", "geography":
		return "geometry", nil
	case "computed":
		return "", fmt.Errorf("%w: SQLite requires a type, see the VirtualAs and StoredAs modifiers", schema.ErrUnsupported)
	case "raw":
		return typeRaw(column), nil
	default:
		return "", fmt.Errorf("%w: the column type %q", schema.ErrUnsupported, column.GetType())
	}
}

// modify renders the SQL fragment for one column modifier, dispatching on
// modifier's name across SQLite's column-level clauses.
func (g *SQLiteGrammar) modify(modifier string, blueprint *schema.Blueprint, column *schema.ColumnDefinition) ([]string, error) {
	switch modifier {
	case "Increment":
		if g.isSerial(column) && column.GetAutoIncrement() {
			return one(" primary key autoincrement"), nil
		}
	case "Nullable":
		if !isGenerated(column) {
			if isNullable(column) {
				return nil, nil
			}
			return one(" not null"), nil
		}
		if n := column.GetNullable(); n != nil && !*n {
			return one(" not null"), nil
		}
	case "Default":
		if column.GetDefault() != nil && column.GetVirtualAs() == nil &&
			column.GetVirtualAsJSON() == "" && column.GetStoredAs() == nil {
			return one(" default " + g.GetDefaultValue(column.GetDefault())), nil
		}
	case "Collate":
		if column.GetCollation() != "" {
			return one(" collate '" + column.GetCollation() + "'"), nil
		}
	case "VirtualAs":
		if json := column.GetVirtualAsJSON(); json != "" {
			return one(" as (" + g.wrapJSONSelector(json) + ")"), nil
		}
		if column.GetVirtualAs() != nil {
			return one(" as (" + stringOf(column.GetVirtualAs()) + ")"), nil
		}
	case "StoredAs":
		if json := column.GetStoredAsJSON(); json != "" {
			return one(" as (" + g.wrapJSONSelector(json) + ") stored"), nil
		}
		if column.GetStoredAs() != nil {
			return one(" as (" + stringOf(column.GetStoredAs()) + ") stored"), nil
		}
	}
	return nil, nil
}

// wrapJSONSelector renders value's field->path JSON accessor as
// json_extract(...).
func (g *SQLiteGrammar) wrapJSONSelector(value string) string {
	field, path := wrapJSONFieldAndPath(value, g.wrapValue)
	return "json_extract(" + field + path + ")"
}

// defaultSchema returns schemaName, or "main" when schemaName is empty.
func defaultSchema(schemaName string) string {
	if schemaName == "" {
		return "main"
	}
	return schemaName
}

func firstOr(values []string, fallback string) string {
	if len(values) > 0 && values[0] != "" {
		return values[0]
	}
	return fallback
}
