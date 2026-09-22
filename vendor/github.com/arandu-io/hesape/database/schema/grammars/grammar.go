package grammars

import (
	"fmt"
	"strings"

	"github.com/arandu-io/hesape/database/query"
	"github.com/arandu-io/hesape/database/schema"
)

// dialect is what BaseGrammar needs back from the driver grammar embedding it.
//
// Go embedding does not dispatch back to the outer type, so each driver
// grammar hands itself to the base at construction and the base calls back
// through this interface. It is unexported because it is the mechanism, not
// the contract -- schema.Grammar is the contract.
type dialect interface {
	// wrapValue quotes one identifier segment in the driver's own style.
	wrapValue(value string) string

	// typeOf dispatches to the driver's type method for the column's type,
	// returning the SQL type or an error when the type has no mapping.
	typeOf(column *schema.ColumnDefinition) (string, error)

	// modify applies one named column modifier for the driver. It returns a
	// list because Postgres' generatedAs modifier can produce more than one
	// fragment when a column is being changed; every other modifier returns
	// none or one. The error reports a generated column that cannot be
	// modified in place.
	modify(modifier string, blueprint *schema.Blueprint, column *schema.ColumnDefinition) ([]string, error)
}

// BaseGrammar is everything a schema grammar can spell without knowing its
// dialect.
//
// A driver grammar embeds it and overrides what its dialect spells differently.
// Everything the base cannot spell on its own returns an error, so a driver that
// does not override it reports the same refusal.
type BaseGrammar struct {
	conn schema.Connection
	self dialect

	transactions   bool
	modifiers      []string
	serials        []string
	fluentCommands []string
}

// GetConnection returns the connection the grammar was built with.
func (g *BaseGrammar) GetConnection() schema.Connection { return g.conn }

// GetFluentCommands returns the commands a driver runs outside the create
// or alter statement, one per column.
func (g *BaseGrammar) GetFluentCommands() []string { return g.fluentCommands }

// GetAlterCommands returns the command names whose presence forces a table
// rebuild. The base implementation returns nil; only the SQLite grammar
// names any.
func (g *BaseGrammar) GetAlterCommands() []string { return nil }

// SupportsSchemaTransactions reports whether this dialect can run DDL
// statements inside a transaction.
func (g *BaseGrammar) SupportsSchemaTransactions() bool { return g.transactions }

// MaxIdentifierLength returns the longest identifier this driver accepts, in
// bytes.
//
// The base returns zero, which means no limit worth enforcing: SQLite is the
// driver that answers this way, since it caps neither an index name nor a
// column name at any length a migration would reach. Postgres and MySQL
// override it with their own.
func (g *BaseGrammar) MaxIdentifierLength() int { return 0 }

// Wrap quotes value as a SQL identifier. A column definition is unwrapped to
// its name before quoting, and an expression passes through untouched.
func (g *BaseGrammar) Wrap(value any) string {
	if column, ok := value.(*schema.ColumnDefinition); ok {
		value = column.GetName()
	}
	if query.IsExpression(value) {
		return stringOf(value)
	}
	name := stringOf(value)
	if i := aliasIndex(name); i >= 0 {
		return g.Wrap(strings.TrimSpace(name[:i])) + " as " +
			g.self.wrapValue(strings.TrimSpace(name[i+4:]))
	}
	return g.wrapSegments(name)
}

// WrapTable quotes table as a SQL table identifier. A blueprint is unwrapped
// to its table name before quoting.
//
// The optional prefix replaces the connection's table prefix, which is how the
// SQLite rebuild names the temporary table it copies rows into. Note that only
// the last dotted segment takes the prefix: schema.table prefixes the table,
// not the schema.
func (g *BaseGrammar) WrapTable(table any, prefix ...string) string {
	if blueprint, ok := table.(*schema.Blueprint); ok {
		table = blueprint.GetTable()
	}
	if query.IsExpression(table) {
		return stringOf(table)
	}

	tablePrefix := g.conn.GetTablePrefix()
	if len(prefix) > 0 {
		tablePrefix = prefix[0]
	}

	name := stringOf(table)
	if i := aliasIndex(name); i >= 0 {
		return g.WrapTable(strings.TrimSpace(name[:i]), tablePrefix) + " as " +
			g.self.wrapValue(tablePrefix+strings.TrimSpace(name[i+4:]))
	}
	if i := strings.LastIndex(name, "."); i >= 0 {
		return g.wrapSegments(name[:i] + "." + tablePrefix + name[i+1:])
	}
	return g.self.wrapValue(tablePrefix + name)
}

// WrapArray quotes every value in values as a SQL identifier, preserving
// order.
func (g *BaseGrammar) WrapArray(values []any) []string {
	out := make([]string, len(values))
	for i, value := range values {
		out[i] = g.Wrap(value)
	}
	return out
}

// Columnize wraps each column as a SQL identifier and joins them with ", ".
func (g *BaseGrammar) Columnize(columns []any) string {
	return strings.Join(g.WrapArray(columns), ", ")
}

// QuoteString quotes value as one or more SQL string literals. A []string or
// []any is quoted element by element and joined with ", "; any other value
// is quoted as a single literal.
func (g *BaseGrammar) QuoteString(value any) string {
	switch v := value.(type) {
	case []string:
		out := make([]string, len(v))
		for i, item := range v {
			out[i] = quote(item)
		}
		return strings.Join(out, ", ")
	case []any:
		out := make([]string, len(v))
		for i, item := range v {
			out[i] = quote(stringOf(item))
		}
		return strings.Join(out, ", ")
	default:
		return quote(stringOf(value))
	}
}

// PrefixArray prepends prefix and a space to every value, returning the
// results in the same order.
func (g *BaseGrammar) PrefixArray(prefix string, values []string) []string {
	out := make([]string, len(values))
	for i, value := range values {
		out[i] = prefix + " " + value
	}
	return out
}

// GetDefaultValue formats value so it can stand in a "default" clause: an
// expression passes through untouched, a bool becomes the string '1' or
// '0', and anything else is quoted as a string literal.
func (g *BaseGrammar) GetDefaultValue(value any) string {
	if query.IsExpression(value) {
		return stringOf(value)
	}
	if b, ok := value.(bool); ok {
		if b {
			return "'1'"
		}
		return "'0'"
	}
	return quote(stringOf(value))
}

// getColumns compiles the blueprint's added columns into their column
// definition SQL, in order, stopping at the first column that fails to
// compile.
func (g *BaseGrammar) getColumns(blueprint *schema.Blueprint) ([]string, error) {
	added := blueprint.GetAddedColumns()
	columns := make([]string, 0, len(added))
	for _, column := range added {
		sql, err := g.getColumn(blueprint, column)
		if err != nil {
			return nil, err
		}
		columns = append(columns, sql)
	}
	return columns, nil
}

// getColumn compiles one column to its definition SQL: the quoted column
// name, the dialect's SQL type, and every modifier the grammar declares.
func (g *BaseGrammar) getColumn(blueprint *schema.Blueprint, column *schema.ColumnDefinition) (string, error) {
	typ, err := g.self.typeOf(column)
	if err != nil {
		return "", err
	}
	return g.addModifiers(g.Wrap(column)+" "+typ, blueprint, column)
}

// addModifiers appends every modifier fragment the driver produces for
// column, in the grammar's declared modifier order, to sql.
func (g *BaseGrammar) addModifiers(sql string, blueprint *schema.Blueprint, column *schema.ColumnDefinition) (string, error) {
	for _, modifier := range g.modifiers {
		fragments, err := g.self.modify(modifier, blueprint, column)
		if err != nil {
			return "", err
		}
		for _, fragment := range fragments {
			sql += fragment
		}
	}
	return sql, nil
}

// getCommandByName returns the first command in blueprint named name, or
// nil if there is none.
func (g *BaseGrammar) getCommandByName(blueprint *schema.Blueprint, name string) *schema.Command {
	for _, command := range blueprint.GetCommands() {
		if command.Name == name {
			return command
		}
	}
	return nil
}

// getCommandsByName returns every command in blueprint named name, in the
// order they were added.
func (g *BaseGrammar) getCommandsByName(blueprint *schema.Blueprint, name string) []*schema.Command {
	var commands []*schema.Command
	for _, command := range blueprint.GetCommands() {
		if command.Name == name {
			commands = append(commands, command)
		}
	}
	return commands
}

// hasCommand reports whether blueprint has a command named name.
func (g *BaseGrammar) hasCommand(blueprint *schema.Blueprint, name string) bool {
	return g.getCommandByName(blueprint, name) != nil
}

// isSerial reports whether column's type is one the driver can make
// auto-incrementing, per the grammar's declared serials list.
func (g *BaseGrammar) isSerial(column *schema.ColumnDefinition) bool {
	for _, serial := range g.serials {
		if serial == column.GetType() {
			return true
		}
	}
	return false
}

// CompileCreateDatabase builds the SQL to create a database named name.
func (g *BaseGrammar) CompileCreateDatabase(name string) (string, error) {
	return "create database " + g.self.wrapValue(name), nil
}

// CompileDropDatabaseIfExists builds the SQL to drop a database named name
// if it exists.
func (g *BaseGrammar) CompileDropDatabaseIfExists(name string) (string, error) {
	return "drop database if exists " + g.self.wrapValue(name), nil
}

// CompileSchemas builds the query that lists the server's schemas. The base
// grammar has no catalogue to query, so it refuses.
func (g *BaseGrammar) CompileSchemas() (string, error) {
	return "", unsupported("retrieving schemas")
}

// CompileTableExists builds the query that reports whether table exists in
// schemaName. An empty result means the driver has no cheap existence
// check, and Builder falls back to listing the tables instead.
func (g *BaseGrammar) CompileTableExists(schemaName, table string) string { return "" }

// CompileTables builds the query that lists the tables in schemas. The base
// grammar has no catalogue to query, so it refuses.
func (g *BaseGrammar) CompileTables(schemas []string, withSize ...bool) (string, error) {
	return "", unsupported("retrieving tables")
}

// CompileViews builds the query that lists the views in schemas. The base
// grammar has no catalogue to query, so it refuses.
func (g *BaseGrammar) CompileViews(schemas []string) (string, error) {
	return "", unsupported("retrieving views")
}

// CompileTypes builds the query that lists the user-defined types in
// schemas. The base grammar has no catalogue to query, so it refuses.
func (g *BaseGrammar) CompileTypes(schemas []string) (string, error) {
	return "", unsupported("retrieving user-defined types")
}

// CompileColumns builds the query that describes the columns of table in
// schemaName. The base grammar has no catalogue to query, so it refuses.
func (g *BaseGrammar) CompileColumns(schemaName, table string) (string, error) {
	return "", unsupported("retrieving columns")
}

// CompileIndexes builds the query that describes the indexes of table in
// schemaName. The base grammar has no catalogue to query, so it refuses.
func (g *BaseGrammar) CompileIndexes(schemaName, table string) (string, error) {
	return "", unsupported("retrieving indexes")
}

// CompileForeignKeys builds the query that describes the foreign keys of
// table in schemaName. The base grammar has no catalogue to query, so it
// refuses.
func (g *BaseGrammar) CompileForeignKeys(schemaName, table string) (string, error) {
	return "", unsupported("retrieving foreign keys")
}

// CompileRenameColumn builds the SQL that renames command.From to
// command.To on blueprint's table.
func (g *BaseGrammar) CompileRenameColumn(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return one(fmt.Sprintf("alter table %s rename column %s to %s",
		g.WrapTable(blueprint), g.Wrap(command.From), g.Wrap(command.To))), nil
}

// CompileChange builds the SQL that alters an existing column's type or
// modifiers. The base grammar refuses; every driver that supports the
// operation overrides it.
func (g *BaseGrammar) CompileChange(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return nil, unsupported("modifying columns")
}

// CompileAlter builds the statements a table rebuild needs. The base
// implementation compiles nothing: no driver but SQLite emits an alter
// command, so there is nothing to refuse either.
func (g *BaseGrammar) CompileAlter(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return nil, nil
}

// CompileFullText builds the SQL that creates a full-text index. The base
// grammar refuses; a driver that supports full-text indexing overrides it.
func (g *BaseGrammar) CompileFullText(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return nil, unsupported("fulltext index creation")
}

// CompileDropFullText builds the SQL that drops a full-text index. The base
// grammar refuses; a driver that supports full-text indexing overrides it.
func (g *BaseGrammar) CompileDropFullText(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return nil, unsupported("fulltext index removal")
}

// CompileVectorIndex builds the SQL that creates a vector similarity index.
// The base grammar refuses; a driver that supports vector indexing
// overrides it.
func (g *BaseGrammar) CompileVectorIndex(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return nil, unsupported("vector indexes")
}

// CompileForeign builds the SQL that adds a foreign key constraint named
// command.Index, referencing command.References on command.On, with
// on-delete and on-update clauses when the command sets them.
func (g *BaseGrammar) CompileForeign(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	sql := fmt.Sprintf("alter table %s add constraint %s ",
		g.WrapTable(blueprint), g.Wrap(command.Index))

	sql += fmt.Sprintf("foreign key (%s) references %s (%s)",
		g.Columnize(command.Columns),
		g.WrapTable(command.On),
		g.Columnize(command.References))

	if command.OnDelete != "" {
		sql += " on delete " + command.OnDelete
	}
	if command.OnUpdate != "" {
		sql += " on update " + command.OnUpdate
	}

	return one(sql), nil
}

// CompileDropForeign builds the SQL that drops a foreign key constraint
// named command.Index. The base grammar refuses because the drop syntax is
// dialect-specific; every driver overrides it.
func (g *BaseGrammar) CompileDropForeign(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return nil, unsupported("dropping foreign keys")
}

// CompileComment builds the SQL for a standalone column comment command.
// Only Postgres declares one; the base compiles nothing.
func (g *BaseGrammar) CompileComment(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return nil, nil
}

// CompileAutoIncrementStartingValues builds the SQL that sets an
// auto-increment column's starting value. Only MySQL and Postgres declare
// it as a fluent command, so the base grammar compiles nothing.
func (g *BaseGrammar) CompileAutoIncrementStartingValues(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return nil, nil
}

// CompileDropAllTables builds the SQL that drops every table in tables in
// one statement. The base grammar refuses; every driver overrides it.
func (g *BaseGrammar) CompileDropAllTables(tables []string) (string, error) {
	return "", unsupported("dropping all tables")
}

// CompileDropAllViews builds the SQL that drops every view in views in one
// statement. The base grammar refuses; every driver overrides it.
func (g *BaseGrammar) CompileDropAllViews(views []string) (string, error) {
	return "", unsupported("dropping all views")
}

// CompileDropAllTypes builds the SQL that drops every type in types in one
// statement. The base grammar refuses; only a driver with user-defined
// types overrides it.
func (g *BaseGrammar) CompileDropAllTypes(types []string) (string, error) {
	return "", unsupported("dropping all types")
}

// CompileSpatialIndex builds the SQL that creates a spatial index. The base
// grammar refuses; a driver that supports spatial indexing overrides it.
func (g *BaseGrammar) CompileSpatialIndex(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return nil, unsupported("spatial indexes")
}

// CompileDropSpatialIndex builds the SQL that drops a spatial index. The
// base grammar refuses; a driver that supports spatial indexing overrides
// it.
func (g *BaseGrammar) CompileDropSpatialIndex(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return nil, unsupported("spatial indexes")
}

// CompileTableComment builds the SQL that sets a table's comment. The base
// grammar refuses; a driver that supports table comments overrides it.
func (g *BaseGrammar) CompileTableComment(blueprint *schema.Blueprint, command *schema.Command) ([]string, error) {
	return nil, unsupported("table comments")
}

// wrapValue quotes value with the SQL standard double quote, or returns it
// unchanged if it is the wildcard "*". A quote inside the identifier is
// doubled rather than stripped, because stripping it would silently rename
// the column.
func (g *BaseGrammar) wrapValue(value string) string {
	if value == "*" {
		return value
	}
	return `"` + strings.ReplaceAll(value, `"`, `""`) + `"`
}

func (g *BaseGrammar) wrapSegments(name string) string {
	segments := strings.Split(name, ".")
	out := make([]string, len(segments))
	for i, segment := range segments {
		out[i] = g.self.wrapValue(segment)
	}
	return strings.Join(out, ".")
}

// typeRaw returns column's raw SQL type definition unchanged.
func typeRaw(column *schema.ColumnDefinition) string { return column.GetDefinition() }

// one wraps a single statement in the list every compiler returns.
func one(sql string) []string { return []string{sql} }

// unsupported builds the error a driver returns for a DDL operation it does
// not implement, wrapping schema.ErrUnsupported with what.
func unsupported(what string) error {
	return fmt.Errorf("%w: %s", schema.ErrUnsupported, what)
}

func quote(value string) string {
	return "'" + strings.ReplaceAll(value, "'", "''") + "'"
}

// addSlashes escapes a value for a MySQL column comment: backslash, single
// quote, double quote and NUL are each escaped, in place of doubling the
// quote as other identifiers do. The backslash is escaped first, so that an
// escape it adds is not escaped again.
func addSlashes(value string) string {
	value = strings.ReplaceAll(value, `\`, `\\`)
	value = strings.ReplaceAll(value, "'", `\'`)
	value = strings.ReplaceAll(value, `"`, `\"`)
	return strings.ReplaceAll(value, "\x00", `\0`)
}

// stringOf renders the string, expression or number a grammar is handed.
func stringOf(value any) string {
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	case query.Expression:
		return v.String()
	case *query.Expression:
		return v.String()
	case fmt.Stringer:
		return v.String()
	default:
		return fmt.Sprint(value)
	}
}

func aliasIndex(name string) int {
	return strings.Index(strings.ToLower(name), " as ")
}

// atLeast reports whether the connection's server version is at or past want,
// comparing the dotted, numeric segments of each from left to right.
func atLeast(version, want string) bool {
	return compareVersions(version, want) >= 0
}

func compareVersions(a, b string) int {
	as, bs := strings.Split(numericPrefix(a), "."), strings.Split(numericPrefix(b), ".")
	for i := 0; i < len(as) || i < len(bs); i++ {
		var x, y int
		if i < len(as) {
			fmt.Sscanf(as[i], "%d", &x)
		}
		if i < len(bs) {
			fmt.Sscanf(bs[i], "%d", &y)
		}
		if x != y {
			if x < y {
				return -1
			}
			return 1
		}
	}
	return 0
}

// numericPrefix keeps the leading digits and dots of a version string, so that
// "10.5.2-MariaDB" compares as 10.5.2.
func numericPrefix(version string) string {
	for i := 0; i < len(version); i++ {
		if (version[i] < '0' || version[i] > '9') && version[i] != '.' {
			return version[:i]
		}
	}
	return version
}

// intOr dereferences value, or returns fallback if value is nil.
func intOr(value *int, fallback int) int {
	if value == nil {
		return fallback
	}
	return *value
}

// isNullable reports column's nullable attribute, treating an unset
// attribute as false.
func isNullable(column *schema.ColumnDefinition) bool {
	nullable := column.GetNullable()
	return nullable != nil && *nullable
}

// isGenerated reports whether the column is computed, in which case the
// nullable modifier is written differently or not at all.
func isGenerated(column *schema.ColumnDefinition) bool {
	return column.GetVirtualAs() != nil || column.GetVirtualAsJSON() != "" ||
		column.GetStoredAs() != nil || column.GetStoredAsJSON() != ""
}

// startingValue returns column's starting value: GetStartingValue if set,
// otherwise GetFrom, otherwise zero.
func startingValue(column *schema.ColumnDefinition) int {
	if value := column.GetStartingValue(); value != nil {
		return *value
	}
	if value := column.GetFrom(); value != nil {
		return *value
	}
	return 0
}

// wrapJSONFieldAndPath splits column on the first "->" into a field and a
// JSON path. It wraps the field with wrap and renders the path, if any, as
// a ", " followed by the quoted JSON path expression.
func wrapJSONFieldAndPath(column string, wrap func(string) string) (field, path string) {
	parts := strings.SplitN(column, "->", 2)
	field = wrap(parts[0])
	if len(parts) > 1 {
		path = ", " + wrapJSONPath(parts[1])
	}
	return field, path
}

// wrapJSONPath renders value as a quoted JSON path expression: an escaped
// or literal single quote becomes a doubled quote, and each "->"-separated
// segment is wrapped by wrapJSONPathSegment and joined with ".". The result
// is quoted as '$path' when path starts with an array index, or '$.path'
// otherwise.
func wrapJSONPath(value string) string {
	value = strings.ReplaceAll(value, `\'`, "''")
	value = strings.ReplaceAll(value, "'", "''")

	segments := strings.Split(value, "->")
	for i, segment := range segments {
		segments[i] = wrapJSONPathSegment(segment)
	}
	jsonPath := strings.Join(segments, ".")

	if strings.HasPrefix(jsonPath, "[") {
		return "'$" + jsonPath + "'"
	}
	return "'$." + jsonPath + "'"
}

// wrapJSONPathSegment double-quotes a JSON path segment's key, leaving a
// trailing array-index bracket like "[0]" outside the quotes.
func wrapJSONPathSegment(segment string) string {
	if i := strings.Index(segment, "["); i >= 0 && strings.HasSuffix(segment, "]") {
		key, brackets := segment[:i], segment[i:]
		if key != "" {
			return `"` + key + `"` + brackets
		}
		return brackets
	}
	return `"` + segment + `"`
}
