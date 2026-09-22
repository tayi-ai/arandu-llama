package grammars

import (
	"regexp"
	"strconv"
	"strings"

	"github.com/arandu-io/hesape/database/query"
)

// MySQLGrammar is the grammar for MySQL.
//
// What it changes about standard SQL, and nothing else: identifiers are quoted
// with a backtick, a random order is RAND(), a shared lock is "lock in share
// mode", an upsert is ON DUPLICATE KEY UPDATE, the JSON operators are the
// json_* functions, and update and delete accept an order and a limit.
type MySQLGrammar struct {
	*Grammar
}

var _ query.Grammar = (*MySQLGrammar)(nil)

// NewMySQLGrammar creates a MySQLGrammar.
func NewMySQLGrammar() *MySQLGrammar {
	g := &MySQLGrammar{Grammar: NewGrammar()}
	g.Grammar.self = g
	return g
}

// GetOperators returns the standard operators plus the ones MySQL adds.
//
// query.BaseGrammar already provides the shared list, so the driver's own
// operators are appended to it.
func (g *MySQLGrammar) GetOperators() []string {
	return append(g.Grammar.GetOperators(), "sounds like")
}

// CompileSelect builds the SQL for a select statement, adding the
// execution time hint when the query asked for a timeout.
func (g *MySQLGrammar) CompileSelect(q *query.Builder) string {
	if g.Grammar.compilationError(q) != nil {
		return ""
	}
	sql := g.Grammar.CompileSelect(q)

	timeout := q.GetTimeout()
	if timeout == nil {
		return sql
	}

	milliseconds := *timeout * 1000

	if len(sql) >= 6 && strings.EqualFold(sql[:6], "select") {
		return "select /*+ MAX_EXECUTION_TIME(" + strconv.Itoa(milliseconds) + ") */" + sql[6:]
	}

	return sql
}

// WhereLike compiles a like where clause. MySQL spells a case sensitive
// like "like binary".
func (g *MySQLGrammar) WhereLike(q *query.Builder, where query.Where) string {
	operator := ""
	if where.Not {
		operator = "not "
	}
	if where.CaseSensitive {
		operator += "like binary"
	} else {
		operator += "like"
	}

	where.Operator = operator

	return g.self.WhereBasic(q, where)
}

// WhereNull compiles an "is null" where clause.
//
// A JSON path that is absent and a JSON path holding the literal null are
// two different things to MySQL and one thing to everybody asking the
// question, so both are tested.
func (g *MySQLGrammar) WhereNull(q *query.Builder, where query.Where) string {
	column := text(g.self.GetValue(where.Column))

	if isJSONSelector(column) {
		field, path := g.wrapJSONFieldAndPath(column)
		return "(json_extract(" + field + path + ") is null OR json_type(json_extract(" + field + path + ")) = 'NULL')"
	}

	return g.Grammar.WhereNull(q, where)
}

// WhereNotNull compiles an "is not null" where clause, testing both the
// absent and the literal-null case for a JSON path the same way WhereNull
// does.
func (g *MySQLGrammar) WhereNotNull(q *query.Builder, where query.Where) string {
	column := text(g.self.GetValue(where.Column))

	if isJSONSelector(column) {
		field, path := g.wrapJSONFieldAndPath(column)
		return "(json_extract(" + field + path + ") is not null AND json_type(json_extract(" + field + path + ")) != 'NULL')"
	}

	return g.Grammar.WhereNotNull(q, where)
}

// WhereFullText compiles a full-text search where clause.
func (g *MySQLGrammar) WhereFullText(q *query.Builder, where query.Where) (string, error) {
	d := g.self
	columns := d.Columnize(where.Columns)
	value := d.Parameter(where.Value)

	mode := " in natural language mode"
	if text(where.Options["mode"]) == "boolean" {
		mode = " in boolean mode"
	}

	expanded := ""
	if truthy(where.Options["expanded"]) && text(where.Options["mode"]) != "boolean" {
		expanded = " with query expansion"
	}

	return "match (" + columns + ") against (" + value + mode + expanded + ")", nil
}

// indexName matches a bare identifier: letters, digits, underscore and
// dollar sign.
var indexName = regexp.MustCompile(`^[a-zA-Z0-9_$]+$`)

// CompileIndexHint builds the index hint component.
//
// The index name is written into the statement rather than bound, because
// no engine binds an identifier, so a name that is not a bare identifier is
// dropped rather than returned as an error: the hint compiles inside
// CompileSelect, which returns a string, and dropping a hint changes the
// plan and never the rows -- while interpolating it would be an injection.
func (g *MySQLGrammar) CompileIndexHint(q *query.Builder, indexHint *query.IndexHint) string {
	index := indexHint.Index

	for _, name := range strings.Split(index, ",") {
		if !indexName.MatchString(strings.TrimSpace(name)) {
			return ""
		}
	}

	switch indexHint.Type {
	case "hint":
		return "use index (" + index + ")"
	case "force":
		return "force index (" + index + ")"
	default:
		return "ignore index (" + index + ")"
	}
}

// legacyGroupLimiter is the one question MySQLGrammar asks itself that
// MariaDBGrammar resolves differently.
type legacyGroupLimiter interface {
	UseLegacyGroupLimit(q *query.Builder) bool
}

// CompileGroupLimit builds a select with a group limit's window function,
// or the legacy session-variable form on a server old enough to need it.
func (g *MySQLGrammar) CompileGroupLimit(q *query.Builder) string {
	if g.Grammar.compilationError(q) != nil {
		return ""
	}
	if limiter, ok := g.self.(legacyGroupLimiter); ok && limiter.UseLegacyGroupLimit(q) {
		return g.compileLegacyGroupLimit(q)
	}
	return g.Grammar.CompileGroupLimit(q)
}

// UseLegacyGroupLimit reports whether the session-variable form of a group
// limit is needed: window functions arrived in MySQL 8.0.11.
//
// MariaDB has a grammar of its own, which always returns false here, so
// nothing here has to ask whether the connection is really MariaDB. A
// connection that cannot report a version is taken to be current.
func (g *MySQLGrammar) UseLegacyGroupLimit(q *query.Builder) bool {
	version, ok := serverVersion(q.GetConnection())
	if !ok {
		return false
	}
	return versionLess(version, "8.0.11")
}

// compileLegacyGroupLimit builds the row numbering done with session
// variables, for a server without window functions.
func (g *MySQLGrammar) compileLegacyGroupLimit(q *query.Builder) string {
	d := g.self
	groupLimit := q.GetGroupLimit()
	limit := groupLimit.Value

	offset := q.GetOffset()
	if offset != nil {
		limit += *offset
		q = q.CloneWithout("offset")
	}

	column := d.Wrap(lastSegment(groupLimit.Column, "."))

	partition := ", @laravel_row := if(@laravel_group = " + column + ", @laravel_row + 1, 1) as `laravel_row`"
	partition += ", @laravel_group := " + column

	q = q.Clone()
	q.Orders = append([]query.Order{{Column: groupLimit.Column, Direction: "asc"}}, q.Orders...)

	sql := concatenate(g.compileComponents(q))

	from := "(select @laravel_row := 0, @laravel_group := 0) as `laravel_vars`, (" + sql + ") as `laravel_table`"

	sql = "select `laravel_table`.*" + partition + " from " + from + " having `laravel_row` <= " + strconv.Itoa(limit)

	if offset != nil {
		sql += " and `laravel_row` > " + strconv.Itoa(*offset)
	}

	return sql + " order by `laravel_row`"
}

// CompileInsert builds the SQL for an insert statement. MySQL writes a row
// of defaults as an empty column list rather than with the "default
// values" keyword.
func (g *MySQLGrammar) CompileInsert(q *query.Builder, values []map[string]any) string {
	if len(values) == 0 {
		values = []map[string]any{{}}
	}
	return g.Grammar.CompileInsert(q, values)
}

// CompileInsertOrIgnore builds an insert statement that ignores
// conflicting rows.
func (g *MySQLGrammar) CompileInsertOrIgnore(q *query.Builder, values []map[string]any) string {
	return strings.Replace(g.self.CompileInsert(q, values), "insert", "insert ignore", 1)
}

// CompileInsertOrIgnoreUsing builds an insert-from-select statement that
// ignores conflicting rows.
func (g *MySQLGrammar) CompileInsertOrIgnoreUsing(q *query.Builder, columns []any, sql string) (string, error) {
	if err := g.Grammar.compilationError(q); err != nil {
		return "", err
	}
	return strings.Replace(g.self.CompileInsertUsing(q, columns, sql), "insert", "insert ignore", 1), nil
}

// CompileJSONContains builds the SQL fragment testing whether a JSON
// column contains a value.
func (g *MySQLGrammar) CompileJSONContains(column any, value string) (string, error) {
	field, path := g.wrapJSONFieldAndPath(column)
	return "json_contains(" + field + ", " + value + path + ")", nil
}

// CompileJSONOverlaps builds the SQL fragment testing whether a JSON
// column overlaps a value.
func (g *MySQLGrammar) CompileJSONOverlaps(column any, value string) (string, error) {
	field, path := g.wrapJSONFieldAndPath(column)
	return "json_overlaps(" + field + ", " + value + path + ")", nil
}

// CompileJSONContainsKey builds the SQL fragment testing whether a JSON
// column contains a key.
func (g *MySQLGrammar) CompileJSONContainsKey(column any) (string, error) {
	field, path := g.wrapJSONFieldAndPath(column)
	return "ifnull(json_contains_path(" + field + ", 'one'" + path + "), 0)", nil
}

// CompileJSONLength builds the SQL fragment comparing a JSON column's
// length.
func (g *MySQLGrammar) CompileJSONLength(column any, operator, value string) (string, error) {
	field, path := g.wrapJSONFieldAndPath(column)
	return "json_length(" + field + path + ") " + operator + " " + value, nil
}

// CompileJSONValueCast casts a value expression to JSON.
func (g *MySQLGrammar) CompileJSONValueCast(value string) string {
	return "cast(" + value + " as json)"
}

// CompileRandom builds a random ordering, seeded when the seed parses as a
// number.
//
// The seed is written into the statement, so a seed that is not a number is
// dropped and the unseeded form is emitted instead: a random order that is
// not reproducible is a smaller thing than interpolating a caller's string
// into SQL.
func (g *MySQLGrammar) CompileRandom(seed string) string {
	if seed == "" {
		return "RAND()"
	}
	number, err := strconv.Atoi(seed)
	if err != nil {
		return "RAND()"
	}
	return "RAND(" + strconv.Itoa(number) + ")"
}

// CompileLock builds the SQL locking clause.
func (g *MySQLGrammar) CompileLock(q *query.Builder, value any) string {
	if lock, ok := value.(string); ok {
		return lock
	}
	if lock, ok := value.(bool); ok && lock {
		return "for update"
	}
	return "lock in share mode"
}

// CompileUpdateColumns builds the SQL set list for an update statement. A
// JSON path is updated in place with json_set rather than by replacing the
// document.
func (g *MySQLGrammar) CompileUpdateColumns(q *query.Builder, values map[string]any) string {
	d := g.self
	parts := make([]string, 0, len(values))

	for _, column := range sortedKeys(values) {
		if isJSONSelector(column) {
			parts = append(parts, g.compileJSONUpdateColumn(column, values[column]))
			continue
		}
		parts = append(parts, d.Wrap(column)+" = "+d.Parameter(values[column]))
	}

	return strings.Join(parts, ", ")
}

// compileJSONUpdateColumn builds the json_set expression that updates one
// JSON path in place.
func (g *MySQLGrammar) compileJSONUpdateColumn(key string, value any) string {
	var parameter string

	switch {
	case isBool(value):
		parameter = "false"
		if value == true {
			parameter = "true"
		}
	case isStructured(value):
		parameter = "cast(? as json)"
	default:
		parameter = g.self.Parameter(value)
	}

	field, path := g.wrapJSONFieldAndPath(key)

	return field + " = json_set(" + field + path + ", " + parameter + ")"
}

// CompileUpsert builds an insert statement that updates the given columns
// on a conflicting row.
//
// # One shape, not two
//
// query.Grammar declares the update list as a list of column names, so each
// column keeps the value the insert would have given it -- which is the
// form every caller of upsert uses, rather than also accepting a map of
// column to a different value.
func (g *MySQLGrammar) CompileUpsert(q *query.Builder, values []map[string]any, uniqueBy []string, update []string) string {
	if g.Grammar.compilationError(q) != nil {
		return ""
	}
	d := g.self
	useUpsertAlias := configBool(q.GetConnection(), "use_upsert_alias")

	sql := d.CompileInsert(q, values)

	if useUpsertAlias {
		sql += " as laravel_upsert_alias"
	}

	sql += " on duplicate key update "

	columns := make([]string, 0, len(update))
	for _, column := range update {
		if useUpsertAlias {
			columns = append(columns, d.Wrap(column)+" = "+d.Wrap("laravel_upsert_alias")+"."+d.Wrap(column))
			continue
		}
		columns = append(columns, d.Wrap(column)+" = values("+d.Wrap(column)+")")
	}

	return sql + strings.Join(columns, ", ")
}

// CompileJoinLateral builds the SQL for a lateral join.
func (g *MySQLGrammar) CompileJoinLateral(join *query.JoinClause, expression string) (string, error) {
	return strings.TrimSpace(join.Type + " join lateral " + expression + " on true"), nil
}

// SupportsStraightJoins reports true: MySQL supports a straight_join.
func (g *MySQLGrammar) SupportsStraightJoins() (bool, error) { return true, nil }

// CompileUpdateWithoutJoins builds the SQL for an update with no joins.
// MySQL takes an order and a limit on an update.
func (g *MySQLGrammar) CompileUpdateWithoutJoins(q *query.Builder, table, columns, where string) string {
	if g.Grammar.compilationError(q) != nil {
		return ""
	}
	sql := g.Grammar.CompileUpdateWithoutJoins(q, table, columns, where)
	return sql + g.orderAndLimit(q)
}

// CompileDeleteWithoutJoins builds the SQL for a delete with no joins,
// adding an order and a limit when the query has them.
func (g *MySQLGrammar) CompileDeleteWithoutJoins(q *query.Builder, table, where string) string {
	if g.Grammar.compilationError(q) != nil {
		return ""
	}
	sql := g.Grammar.CompileDeleteWithoutJoins(q, table, where)
	return sql + g.orderAndLimit(q)
}

// CompileDeleteWithJoins builds the SQL for a delete that joins other
// tables.
//
// Standard MySQL rejects an order or a limit on a joined delete and some
// compatible servers accept them, so they are emitted when they were asked
// for and the server is left to say what it supports.
func (g *MySQLGrammar) CompileDeleteWithJoins(q *query.Builder, table, where string) string {
	if g.Grammar.compilationError(q) != nil {
		return ""
	}
	sql := g.Grammar.CompileDeleteWithJoins(q, table, where)
	return sql + g.orderAndLimit(q)
}

func (g *MySQLGrammar) orderAndLimit(q *query.Builder) string {
	d := g.self
	sql := ""
	if len(q.Orders) > 0 {
		sql += " " + d.CompileOrders(q, q.Orders)
	}
	if limit := q.GetLimit(); limit != nil {
		sql += " " + d.CompileLimit(q, *limit)
	}
	return sql
}

// PrepareBindingsForUpdate orders the bindings for an update statement.
//
// A boolean written into a JSON path is compiled into the statement rather
// than bound, so its binding is dropped here; an array or an object becomes
// its JSON text.
func (g *MySQLGrammar) PrepareBindingsForUpdate(bindings map[string][]any, values map[string]any) []any {
	prepared := make(map[string]any, len(values))

	for column, value := range values {
		if isJSONSelector(column) && isBool(value) {
			continue
		}
		if isStructured(value) {
			encoded, err := encodeJSON(value)
			if err == nil {
				value = encoded
			}
		}
		prepared[column] = value
	}

	return g.Grammar.PrepareBindingsForUpdate(bindings, prepared)
}

// CompileThreadCount builds the SQL that counts open connections.
func (g *MySQLGrammar) CompileThreadCount() string {
	return "select variable_value as `Value` from performance_schema.session_status where variable_name = 'threads_connected'"
}

// WrapValue quotes one identifier segment. MySQL quotes an identifier with
// a backtick, and doubles a backtick inside one.
func (g *MySQLGrammar) WrapValue(value string) string {
	if value == "*" {
		return value
	}
	return "`" + strings.ReplaceAll(value, "`", "``") + "`"
}

// WrapJSONSelector quotes a JSON path selector, unquoting the extracted
// value.
func (g *MySQLGrammar) WrapJSONSelector(value string) string {
	field, path := g.wrapJSONFieldAndPath(value)
	return "json_unquote(json_extract(" + field + path + "))"
}

// WrapJSONBooleanSelector quotes a JSON path selector for a boolean
// comparison. The value is compared unquoted, so the extract is not
// unwrapped.
func (g *MySQLGrammar) WrapJSONBooleanSelector(value string) string {
	field, path := g.wrapJSONFieldAndPath(value)
	return "json_extract(" + field + path + ")"
}
