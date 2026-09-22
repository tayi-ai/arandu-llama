package grammars

import (
	"strings"

	"github.com/arandu-io/hesape/database/query"
)

// SQLiteGrammar is the grammar for SQLite.
//
// What it changes about standard SQL: there is no row lock at all, so the lock
// clause compiles to nothing; a union subquery has to be wrapped in a select
// because SQLite will not parenthesise one; the date parts come out of
// strftime; an upsert is ON CONFLICT DO UPDATE; the JSON operators are the
// json_* functions; and an update or delete narrowed by a limit goes through
// rowid.
//
// The right join is not on that list: SQLite has had one since 3.39, and the
// standard compilation is correct for it.
type SQLiteGrammar struct {
	*Grammar
}

var _ query.Grammar = (*SQLiteGrammar)(nil)

// NewSQLiteGrammar creates a SQLiteGrammar.
func NewSQLiteGrammar() *SQLiteGrammar {
	g := &SQLiteGrammar{Grammar: NewGrammar()}
	g.Grammar.self = g
	return g
}

// sqliteOperators lists the extra comparison operators SQLite supports
// beyond the standard set.
var sqliteOperators = []string{
	"=", "<", ">", "<=", ">=", "<>", "!=",
	"like", "not like", "ilike",
	"&", "|", "<<", ">>",
}

// GetOperators returns the standard operators plus SQLite's extras.
func (g *SQLiteGrammar) GetOperators() []string {
	return append(g.Grammar.GetOperators(), sqliteOperators...)
}

// CompileLock builds the SQL locking clause. SQLite locks the file, not
// the row, so there is nothing to say.
func (g *SQLiteGrammar) CompileLock(q *query.Builder, value any) string { return "" }

// WrapUnion wraps a union member in a select. SQLite does not accept a
// parenthesised select as a union operand, so it gets a select around it
// instead.
func (g *SQLiteGrammar) WrapUnion(sql string) string {
	return "select * from (" + sql + ")"
}

// WhereBasic compiles a simple where clause. The null safe equality is
// spelled IS.
func (g *SQLiteGrammar) WhereBasic(q *query.Builder, where query.Where) string {
	if where.Operator == "<=>" {
		d := g.self
		return d.Wrap(where.Column) + " IS " + d.Parameter(where.Value)
	}
	return g.Grammar.WhereBasic(q, where)
}

// WhereLike compiles a like where clause.
//
// SQLite's like is case insensitive and has no case sensitive form, so the
// case sensitive question is resolved with glob -- which is case sensitive
// and uses different wildcards. PrepareWhereLikeBinding translates the
// pattern.
func (g *SQLiteGrammar) WhereLike(q *query.Builder, where query.Where) string {
	if !where.CaseSensitive {
		return g.Grammar.WhereLike(q, where)
	}

	where.Operator = "glob"
	if where.Not {
		where.Operator = "not glob"
	}

	return g.self.WhereBasic(q, where)
}

// PrepareWhereLikeBinding rewrites a like pattern as a glob pattern, since
// the two spell their wildcards the other way round.
func (g *SQLiteGrammar) PrepareWhereLikeBinding(value string, caseSensitive bool) string {
	if !caseSensitive {
		return value
	}
	return strings.NewReplacer("*", "[*]", "?", "[?]", "%", "*", "_", "?").Replace(value)
}

// WhereDate compiles a where clause that compares a column's date part.
func (g *SQLiteGrammar) WhereDate(q *query.Builder, where query.Where) string {
	return g.self.DateBasedWhere("%Y-%m-%d", q, where)
}

// WhereDay compiles a where clause that compares a column's day part.
func (g *SQLiteGrammar) WhereDay(q *query.Builder, where query.Where) string {
	return g.self.DateBasedWhere("%d", q, where)
}

// WhereMonth compiles a where clause that compares a column's month part.
func (g *SQLiteGrammar) WhereMonth(q *query.Builder, where query.Where) string {
	return g.self.DateBasedWhere("%m", q, where)
}

// WhereYear compiles a where clause that compares a column's year part.
func (g *SQLiteGrammar) WhereYear(q *query.Builder, where query.Where) string {
	return g.self.DateBasedWhere("%Y", q, where)
}

// WhereTime compiles a where clause that compares a column's time part.
func (g *SQLiteGrammar) WhereTime(q *query.Builder, where query.Where) string {
	return g.self.DateBasedWhere("%H:%M:%S", q, where)
}

// DateBasedWhere compiles a date-part where clause. SQLite has no date
// type, so both sides are compared as text.
func (g *SQLiteGrammar) DateBasedWhere(typ string, q *query.Builder, where query.Where) string {
	d := g.self
	return "strftime('" + typ + "', " + d.Wrap(where.Column) + ") " + where.Operator +
		" cast(" + d.Parameter(where.Value) + " as text)"
}

// CompileIndexHint builds the index hint component. SQLite has one form
// of hint and it is a demand, so anything but a forced index compiles to
// nothing.
//
// An index name that is not a bare identifier is dropped rather than
// interpolated; see MySQLGrammar.CompileIndexHint for why that is a string
// and not an error.
func (g *SQLiteGrammar) CompileIndexHint(q *query.Builder, indexHint *query.IndexHint) string {
	if indexHint.Type != "force" {
		return ""
	}
	if !indexName.MatchString(indexHint.Index) {
		return ""
	}
	return "indexed by " + indexHint.Index
}

// CompileJSONLength builds the SQL fragment comparing a JSON column's
// length.
func (g *SQLiteGrammar) CompileJSONLength(column any, operator, value string) (string, error) {
	field, path := g.wrapJSONFieldAndPath(column)
	return "json_array_length(" + field + path + ") " + operator + " " + value, nil
}

// CompileJSONContains builds the SQL fragment testing whether a JSON
// column contains a value. SQLite has no containment operator, so the
// array is walked.
func (g *SQLiteGrammar) CompileJSONContains(column any, value string) (string, error) {
	field, path := g.wrapJSONFieldAndPath(column)
	return "exists (select 1 from json_each(" + field + path + ") where " +
		g.self.Wrap("json_each.value") + " is " + value + ")", nil
}

// PrepareBindingForJSONContains prepares a binding for a JSON contains
// comparison: the value is compared against one element, so it stays a
// scalar rather than becoming JSON text.
func (g *SQLiteGrammar) PrepareBindingForJSONContains(binding any) (any, error) {
	return binding, nil
}

// CompileJSONContainsKey builds the SQL fragment testing whether a JSON
// column contains a key.
func (g *SQLiteGrammar) CompileJSONContainsKey(column any) (string, error) {
	field, path := g.wrapJSONFieldAndPath(column)
	return "json_type(" + field + path + ") is not null", nil
}

// CompileGroupLimit builds a select with a group limit's window function,
// on servers that support one.
//
// Window functions arrived in SQLite 3.25.0. Before that the group limit is
// dropped and an ordinary select is compiled instead -- every row of every
// group, and the caller cuts them itself. A connection that cannot report a
// version is taken to be current.
func (g *SQLiteGrammar) CompileGroupLimit(q *query.Builder) string {
	if g.Grammar.compilationError(q) != nil {
		return ""
	}
	if version, ok := serverVersion(q.GetConnection()); ok && versionLess(version, "3.25.0") {
		return g.compileSelectWithoutGroupLimit(q)
	}
	return g.Grammar.CompileGroupLimit(q)
}

// CompileUpdate builds the SQL for an update statement, routing through a
// rowid subquery when the update has joins or a limit.
func (g *SQLiteGrammar) CompileUpdate(q *query.Builder, values map[string]any) string {
	if g.Grammar.compilationError(q) != nil {
		return ""
	}
	if len(q.Joins) > 0 || q.GetLimit() != nil {
		return g.compileUpdateWithJoinsOrLimit(q, values)
	}
	return g.Grammar.CompileUpdate(q, values)
}

// CompileInsertOrIgnore builds an insert statement that ignores
// conflicting rows.
func (g *SQLiteGrammar) CompileInsertOrIgnore(q *query.Builder, values []map[string]any) string {
	return strings.Replace(g.self.CompileInsert(q, values), "insert", "insert or ignore", 1)
}

// CompileInsertOrIgnoreReturning builds an insert statement that ignores
// conflicting rows and returns the given columns.
func (g *SQLiteGrammar) CompileInsertOrIgnoreReturning(q *query.Builder, values []map[string]any, uniqueBy, returning []string) (string, error) {
	if err := g.Grammar.compilationError(q); err != nil {
		return "", err
	}
	d := g.self
	return d.CompileInsert(q, values) +
		" on conflict (" + d.Columnize(toAny(uniqueBy)) + ") do nothing" +
		" returning " + d.Columnize(toAny(returning)), nil
}

// CompileInsertOrIgnoreUsing builds an insert-from-select statement that
// ignores conflicting rows.
func (g *SQLiteGrammar) CompileInsertOrIgnoreUsing(q *query.Builder, columns []any, sql string) (string, error) {
	if err := g.Grammar.compilationError(q); err != nil {
		return "", err
	}
	return strings.Replace(g.self.CompileInsertUsing(q, columns, sql), "insert", "insert or ignore", 1), nil
}

// CompileUpdateColumns builds the SQL set list for an update statement.
//
// SQLite cannot set one path of a document at a time, so every path written
// into the same column is collected and applied as one patch.
func (g *SQLiteGrammar) CompileUpdateColumns(q *query.Builder, values map[string]any) string {
	d := g.self
	groups := g.groupJSONColumnsForUpdate(values)

	merged := make(map[string]any, len(values))
	for key, value := range values {
		if isJSONSelector(key) {
			continue
		}
		merged[key] = value
	}
	for key, value := range groups {
		merged[key] = value
	}

	parts := make([]string, 0, len(merged))
	for _, key := range sortedKeys(merged) {
		column := lastSegment(key, ".")

		if _, patched := groups[key]; patched {
			parts = append(parts, d.Wrap(column)+" = "+g.compileJSONPatch(column, merged[key]))
			continue
		}

		parts = append(parts, d.Wrap(column)+" = "+d.Parameter(merged[key]))
	}

	return strings.Join(parts, ", ")
}

// CompileUpsert builds an insert statement that updates the given columns
// on a conflicting row.
func (g *SQLiteGrammar) CompileUpsert(q *query.Builder, values []map[string]any, uniqueBy []string, update []string) string {
	if g.Grammar.compilationError(q) != nil {
		return ""
	}
	d := g.self

	sql := d.CompileInsert(q, values) + " on conflict (" + d.Columnize(toAny(uniqueBy)) + ") do update set "

	columns := make([]string, 0, len(update))
	for _, column := range update {
		columns = append(columns, d.Wrap(column)+" = "+d.WrapValue("excluded")+"."+d.Wrap(column))
	}

	return sql + strings.Join(columns, ", ")
}

// groupJSONColumnsForUpdate rebuilds every JSON path in the update as the
// nested document it describes, keyed by its column.
func (g *SQLiteGrammar) groupJSONColumnsForUpdate(values map[string]any) map[string]any {
	groups := make(map[string]any)

	for _, key := range sortedKeys(values) {
		if !isJSONSelector(key) {
			continue
		}
		path := strings.ReplaceAll(after(key, "."), "->", ".")
		setPath(groups, strings.Split(path, "."), values[key])
	}

	return groups
}

// compileJSONPatch builds the json_patch expression that merges a value
// into an existing column.
func (g *SQLiteGrammar) compileJSONPatch(column string, value any) string {
	d := g.self
	return "json_patch(ifnull(" + d.Wrap(column) + ", json('{}')), json(" + d.Parameter(value) + "))"
}

// compileUpdateWithJoinsOrLimit builds an update routed through a rowid
// subquery, for an update that has joins or a limit.
//
// SQLite has no UPDATE FROM and no limit on an update, so the rows are
// named by rowid and chosen by a select that can have both. The select is
// compiled from a clone, which reaches the same rows without adding a
// column to the caller's builder.
func (g *SQLiteGrammar) compileUpdateWithJoinsOrLimit(q *query.Builder, values map[string]any) string {
	d := g.self
	table := d.WrapTable(q.GetFrom())
	columns := d.CompileUpdateColumns(q, values)

	alias := lastAliasSegment(text(q.GetFrom()))
	selectSQL := d.CompileSelect(q.Clone().Select(alias + ".rowid"))

	return "update " + table + " set " + columns + " where " + d.Wrap("rowid") + " in (" + selectSQL + ")"
}

// PrepareBindingsForUpdate orders the bindings for an update statement.
// The JSON paths are bound as the one patch document they were compiled
// into.
func (g *SQLiteGrammar) PrepareBindingsForUpdate(bindings map[string][]any, values map[string]any) []any {
	groups := g.groupJSONColumnsForUpdate(values)

	merged := make(map[string]any, len(values))
	for key, value := range values {
		if isJSONSelector(key) {
			continue
		}
		merged[key] = value
	}
	for key, value := range groups {
		merged[key] = value
	}

	out := make([]any, 0, len(merged)+8)
	for _, key := range sortedKeys(merged) {
		value := merged[key]
		if isStructured(value) {
			if encoded, err := encodeJSON(value); err == nil {
				value = encoded
			}
		}
		out = append(out, value)
	}

	for _, key := range bindingOrder {
		if key == "select" {
			continue
		}
		out = append(out, bindings[key]...)
	}

	return out
}

// CompileDelete builds the SQL for a delete statement, routing through a
// rowid subquery when the delete has joins or a limit.
func (g *SQLiteGrammar) CompileDelete(q *query.Builder) string {
	if g.Grammar.compilationError(q) != nil {
		return ""
	}
	if len(q.Joins) > 0 || q.GetLimit() != nil {
		return g.compileDeleteWithJoinsOrLimit(q)
	}
	return g.Grammar.CompileDelete(q)
}

// compileDeleteWithJoinsOrLimit builds a delete routed through a rowid
// subquery, for a delete that has joins or a limit.
func (g *SQLiteGrammar) compileDeleteWithJoinsOrLimit(q *query.Builder) string {
	d := g.self
	table := d.WrapTable(q.GetFrom())

	alias := lastAliasSegment(text(q.GetFrom()))
	selectSQL := d.CompileSelect(q.Clone().Select(alias + ".rowid"))

	return "delete from " + table + " where " + d.Wrap("rowid") + " in (" + selectSQL + ")"
}

// CompileTruncate builds the SQL that empties a table.
//
// SQLite has no truncate: the rows are deleted and the auto increment
// counter is reset by hand. Both statements are returned, and the order
// they run in does not matter -- one clears the table, the other clears the
// counter, and neither depends on the other.
//
// The schema is split off the table name here rather than asked of the
// schema builder, because a query grammar reaching into a schema builder to
// read a string is a dependency for nothing.
func (g *SQLiteGrammar) CompileTruncate(q *query.Builder) map[string][]any {
	if g.Grammar.compilationError(q) != nil {
		return nil
	}
	d := g.self
	from := text(q.GetFrom())

	schema, table := "", from
	if i := strings.Index(from, "."); i >= 0 {
		schema, table = from[:i], from[i+1:]
	}
	if schema != "" {
		schema = d.WrapValue(schema) + "."
	}

	return map[string][]any{
		"delete from " + schema + "sqlite_sequence where name = ?": {d.GetTablePrefix() + table},
		"delete from " + d.WrapTable(q.GetFrom()):                  {},
	}
}

// WrapJSONSelector quotes a JSON path selector.
func (g *SQLiteGrammar) WrapJSONSelector(value string) string {
	field, path := g.wrapJSONFieldAndPath(value)
	return "json_extract(" + field + path + ")"
}

// after returns what follows the first occurrence of separator, or the
// whole string when there is none.
func after(value, separator string) string {
	if i := strings.Index(value, separator); i >= 0 {
		return value[i+len(separator):]
	}
	return value
}

// setPath walks the nested maps into existence along a dotted key path and
// writes the value at the end of it.
func setPath(target map[string]any, path []string, value any) {
	for i, key := range path {
		if i == len(path)-1 {
			target[key] = value
			return
		}

		next, ok := target[key].(map[string]any)
		if !ok {
			next = make(map[string]any)
			target[key] = next
		}
		target = next
	}
}
