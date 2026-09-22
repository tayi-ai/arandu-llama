package grammars

import (
	"regexp"
	"slices"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"

	"github.com/arandu-io/hesape/database/query"
)

// PostgresGrammar is the grammar for Postgres.
//
// What it changes about standard SQL: a like is compared against the column
// cast to text, a bitwise comparison is cast to bool, a date comparison is an
// extract, distinct takes columns, an upsert is ON CONFLICT DO UPDATE, an
// insert can return the identifier it assigned, an update or delete narrowed by
// a limit goes through ctid, and the JSON operators are -> and ->> rather than
// functions.
//
// Placeholders stay "?" here, as they do in every grammar. See Grammar.Parameter
// for where the $1, $2 numbering happens and why it happens there.
type PostgresGrammar struct {
	*Grammar
}

var _ query.Grammar = (*PostgresGrammar)(nil)

// NewPostgresGrammar returns a new PostgresGrammar, its embedded Grammar's
// self reference pointed at it so dialect overrides take effect.
func NewPostgresGrammar() *PostgresGrammar {
	g := &PostgresGrammar{Grammar: NewGrammar()}
	g.Grammar.self = g
	return g
}

// postgresOperators lists the operators unique to Postgres, appended to the
// base grammar's set.
var postgresOperators = []string{
	"=", "<", ">", "<=", ">=", "<>", "!=",
	"like", "not like", "between", "ilike", "not ilike",
	"~", "&", "|", "#", "<<", ">>", "<<=", ">>=",
	"&&", "@>", "<@", "?", "?|", "?&", "||", "-", "@?", "@@", "#-",
	"is distinct from", "is not distinct from",
}

// postgresBitwiseOperators lists the operators Postgres treats as bitwise.
var postgresBitwiseOperators = []string{"~", "&", "|", "#", "<<", ">>", "<<=", ">>="}

// The two pieces of package level state below are process wide, and a grammar
// is read from every request, so both are guarded: a test that flips one while
// another goroutine compiles a statement would otherwise be a data race rather
// than a surprise.
var (
	customOperatorsMu sync.RWMutex
	customOperators   []string
	customOperator    = regexp.MustCompile("^[+\\-*/<>=~!@#%^&|?]+$")

	cascadeTruncate = newAtomicTrue()
)

func newAtomicTrue() *atomic.Bool {
	value := &atomic.Bool{}
	value.Store(true)
	return value
}

// CustomOperators registers operators an extension adds, which Builder has to
// accept before it will compile them.
func CustomOperators(operators []string) {
	customOperatorsMu.Lock()
	defer customOperatorsMu.Unlock()

	for _, operator := range operators {
		if customOperator.MatchString(operator) &&
			!strings.Contains(operator, "--") &&
			!strings.Contains(operator, "/*") &&
			!strings.Contains(operator, "*/") {
			customOperators = append(customOperators, operator)
		}
	}
}

// CascadeOnTruncate sets whether truncating a table also truncates the tables
// whose foreign keys point at it.
func CascadeOnTruncate(value bool) { cascadeTruncate.Store(value) }

// CascadeOnTrucate is a misspelling kept for compatibility.
//
// Deprecated: use CascadeOnTruncate.
func CascadeOnTrucate(value bool) { CascadeOnTruncate(value) }

// GetOperators returns the base operators, Postgres's own, and any registered
// by CustomOperators, sorted and de-duplicated.
func (g *PostgresGrammar) GetOperators() []string {
	customOperatorsMu.RLock()
	defer customOperatorsMu.RUnlock()

	out := append(g.Grammar.GetOperators(), postgresOperators...)
	out = append(out, customOperators...)

	slices.Sort(out)
	return slices.Compact(out)
}

// GetBitwiseOperators returns the base bitwise operators plus Postgres's own.
func (g *PostgresGrammar) GetBitwiseOperators() []string {
	return append(g.Grammar.GetBitwiseOperators(), postgresBitwiseOperators...)
}

// WhereBasic compiles a basic where clause, casting the column to text first
// when the operator is a like.
//
// A like against a non-text column is an error in Postgres and a silent cast
// everywhere else, so the column is cast rather than the comparison refused.
func (g *PostgresGrammar) WhereBasic(q *query.Builder, where query.Where) string {
	d := g.self
	if strings.Contains(strings.ToLower(where.Operator), "like") {
		return d.Wrap(where.Column) + "::text " + where.Operator + " " + d.Parameter(where.Value)
	}
	return g.Grammar.WhereBasic(q, where)
}

// WhereBitwise compiles a bitwise where clause, casting the numeric result to
// bool: the operator itself returns a number, and a where wants a boolean.
func (g *PostgresGrammar) WhereBitwise(q *query.Builder, where query.Where) string {
	d := g.self
	value := d.Parameter(where.Value)
	operator := strings.ReplaceAll(where.Operator, "?", "??")

	return "(" + d.Wrap(where.Column) + " " + operator + " " + value + ")::bool"
}

// WhereLike compiles a like comparison. Postgres is the one engine whose like
// is case sensitive, so the case insensitive form is ilike.
func (g *PostgresGrammar) WhereLike(q *query.Builder, where query.Where) string {
	operator := ""
	if where.Not {
		operator = "not "
	}
	if where.CaseSensitive {
		operator += "like"
	} else {
		operator += "ilike"
	}

	where.Operator = operator

	return g.self.WhereBasic(q, where)
}

// WhereDate compiles a date comparison, casting the column to date.
func (g *PostgresGrammar) WhereDate(q *query.Builder, where query.Where) string {
	d := g.self
	column := d.Wrap(where.Column)
	if isJSONSelector(where.Column) {
		column = "(" + column + ")"
	}
	return column + "::date " + where.Operator + " " + d.Parameter(where.Value)
}

// WhereTime compiles a time comparison, casting the column to time.
func (g *PostgresGrammar) WhereTime(q *query.Builder, where query.Where) string {
	d := g.self
	column := d.Wrap(where.Column)
	if isJSONSelector(where.Column) {
		column = "(" + column + ")"
	}
	return column + "::time " + where.Operator + " " + d.Parameter(where.Value)
}

// DateBasedWhere compiles a day/month/year comparison as an extract over the
// column.
func (g *PostgresGrammar) DateBasedWhere(typ string, q *query.Builder, where query.Where) string {
	d := g.self
	return "extract(" + typ + " from " + d.Wrap(where.Column) + ") " + where.Operator + " " + d.Parameter(where.Value)
}

// fullTextLanguages lists the languages Postgres ships a text search
// configuration for.
var fullTextLanguages = []string{
	"simple", "arabic", "danish", "dutch", "english", "finnish", "french",
	"german", "hungarian", "indonesian", "irish", "italian", "lithuanian",
	"nepali", "norwegian", "portuguese", "romanian", "russian", "spanish",
	"swedish", "tamil", "turkish",
}

// WhereFullText compiles a full text search clause.
//
// The language is written into the statement, so it is checked against the
// list of the ones Postgres ships, and anything else falls back to english --
// which is why the value is not simply quoted.
func (g *PostgresGrammar) WhereFullText(q *query.Builder, where query.Where) (string, error) {
	d := g.self

	language := text(where.Options["language"])
	if !slices.Contains(fullTextLanguages, language) {
		language = "english"
	}

	isVector := truthy(where.Options["vector"])

	parts := make([]string, 0, len(where.Columns))
	for _, column := range where.Columns {
		if isVector {
			parts = append(parts, d.Wrap(column))
			continue
		}
		parts = append(parts, "to_tsvector('"+language+"', "+d.Wrap(column)+")")
	}
	columns := strings.Join(parts, " || ")

	mode := "plainto_tsquery"
	switch text(where.Options["mode"]) {
	case "phrase":
		mode = "phraseto_tsquery"
	case "websearch":
		mode = "websearch_to_tsquery"
	case "raw":
		mode = "to_tsquery"
	}

	return "(" + columns + ") @@ " + mode + "('" + language + "', " + d.Parameter(where.Value) + ")", nil
}

// CompileColumns compiles the select list. Only Postgres takes columns for its
// distinct.
func (g *PostgresGrammar) CompileColumns(q *query.Builder, columns []any) string {
	d := g.self

	if q.GetAggregate() != nil {
		return ""
	}

	switch distinct := q.GetDistinct().(type) {
	case []any:
		return "select distinct on (" + d.Columnize(distinct) + ") " + d.Columnize(columns)
	case bool:
		if distinct {
			return "select distinct " + d.Columnize(columns)
		}
	}

	return "select " + d.Columnize(columns)
}

// CompileJSONContains compiles a JSON containment test using the @> operator.
func (g *PostgresGrammar) CompileJSONContains(column any, value string) (string, error) {
	wrapped := strings.ReplaceAll(g.self.Wrap(column), "->>", "->")
	return "(" + wrapped + ")::jsonb @> " + value, nil
}

// jsonArrayIndex matches a trailing array index, such as "[3]" or "[-1]", at
// the end of a JSON path segment.
var jsonArrayIndex = regexp.MustCompile(`\[(-?[0-9]+)\]$`)

// CompileJSONContainsKey compiles a test for whether a JSON path exists.
//
// An index into an array is a different question from a key in an object, so a
// path ending in one compiles to a length test instead. A negative index
// counts from the end, which is why it is compared by absolute value.
func (g *PostgresGrammar) CompileJSONContainsKey(column any) (string, error) {
	segments := strings.Split(text(column), "->")
	lastSegment := segments[len(segments)-1]
	segments = segments[:len(segments)-1]

	index, indexed := 0, false

	if number, err := strconv.Atoi(lastSegment); err == nil {
		index, indexed = number, true
	} else if matches := jsonArrayIndex.FindStringSubmatch(lastSegment); matches != nil {
		segments = append(segments, strings.TrimSuffix(lastSegment, matches[0]))
		index, _ = strconv.Atoi(matches[1])
		indexed = true
	}

	wrapped := strings.ReplaceAll(g.self.Wrap(strings.Join(segments, "->")), "->>", "->")

	if indexed {
		length := index + 1
		if index < 0 {
			length = -index
		}
		return "case when jsonb_typeof((" + wrapped + ")::jsonb) = 'array' then " +
			"jsonb_array_length((" + wrapped + ")::jsonb) >= " + strconv.Itoa(length) +
			" else false end", nil
	}

	key := "'" + strings.ReplaceAll(lastSegment, "'", "''") + "'"

	return "coalesce((" + wrapped + ")::jsonb ?? " + key + ", false)", nil
}

// CompileJSONLength compiles a comparison against the length of a JSON array.
func (g *PostgresGrammar) CompileJSONLength(column any, operator, value string) (string, error) {
	wrapped := strings.ReplaceAll(g.self.Wrap(column), "->>", "->")
	return "jsonb_array_length((" + wrapped + ")::jsonb) " + operator + " " + value, nil
}

// CompileHaving compiles a having clause, delegating to CompileHavingBitwise
// when the having is a bitwise comparison.
func (g *PostgresGrammar) CompileHaving(having query.Having) string {
	if strings.EqualFold(having.Type, "Bitwise") {
		return g.CompileHavingBitwise(having)
	}
	return g.Grammar.CompileHaving(having)
}

// CompileHavingBitwise compiles a bitwise having clause, casting the numeric
// result to bool.
func (g *PostgresGrammar) CompileHavingBitwise(having query.Having) string {
	d := g.self
	return "(" + d.Wrap(having.Column) + " " + having.Operator + " " + d.Parameter(having.Value) + ")::bool"
}

// CompileLock compiles a row lock: a string value is used as written, true
// compiles to "for update", and anything else to "for share".
func (g *PostgresGrammar) CompileLock(q *query.Builder, value any) string {
	if lock, ok := value.(string); ok {
		return lock
	}
	if lock, ok := value.(bool); ok && lock {
		return "for update"
	}
	return "for share"
}

// CompileInsertOrIgnore compiles an insert that silently skips a row already
// present, via "on conflict do nothing".
func (g *PostgresGrammar) CompileInsertOrIgnore(q *query.Builder, values []map[string]any) string {
	if g.Grammar.compilationError(q) != nil {
		return ""
	}
	return g.self.CompileInsert(q, values) + " on conflict do nothing"
}

// CompileInsertOrIgnoreReturning compiles an insert that skips a conflicting
// row and returns the given columns for the rows it did insert.
func (g *PostgresGrammar) CompileInsertOrIgnoreReturning(q *query.Builder, values []map[string]any, uniqueBy, returning []string) (string, error) {
	if err := g.Grammar.compilationError(q); err != nil {
		return "", err
	}
	d := g.self
	return d.CompileInsert(q, values) +
		" on conflict (" + d.Columnize(toAny(uniqueBy)) + ") do nothing" +
		" returning " + d.Columnize(toAny(returning)), nil
}

// CompileInsertOrIgnoreUsing compiles an insert-from-select that silently
// skips a row already present.
func (g *PostgresGrammar) CompileInsertOrIgnoreUsing(q *query.Builder, columns []any, sql string) (string, error) {
	if err := g.Grammar.compilationError(q); err != nil {
		return "", err
	}
	return g.self.CompileInsertUsing(q, columns, sql) + " on conflict do nothing", nil
}

// CompileInsertGetID compiles an insert with a returning clause naming the
// sequence column.
//
// Postgres hands the identifier back as a row, which is why PostgresProcessor
// reads it from the result set instead of asking the connection for the last
// one it assigned.
func (g *PostgresGrammar) CompileInsertGetID(q *query.Builder, values map[string]any, sequence string) string {
	if g.Grammar.compilationError(q) != nil {
		return ""
	}
	if sequence == "" {
		sequence = "id"
	}
	return g.self.CompileInsert(q, []map[string]any{values}) + " returning " + g.self.Wrap(sequence)
}

// CompileUpdate compiles an update statement, routing through
// compileUpdateWithJoinsOrLimit when the query has joins or a limit.
func (g *PostgresGrammar) CompileUpdate(q *query.Builder, values map[string]any) string {
	if g.Grammar.compilationError(q) != nil {
		return ""
	}
	if len(q.Joins) > 0 || q.GetLimit() != nil {
		return g.compileUpdateWithJoinsOrLimit(q, values)
	}
	return g.Grammar.CompileUpdate(q, values)
}

// CompileUpdateColumns compiles the set list of an update. The list names
// columns of the table being updated, so a qualified name loses its table.
func (g *PostgresGrammar) CompileUpdateColumns(q *query.Builder, values map[string]any) string {
	d := g.self
	parts := make([]string, 0, len(values))

	for _, key := range sortedKeys(values) {
		column := lastSegment(key, ".")

		if isJSONSelector(key) {
			parts = append(parts, g.compileJSONUpdateColumn(column, values[key]))
			continue
		}

		parts = append(parts, d.Wrap(column)+" = "+d.Parameter(values[key]))
	}

	return strings.Join(parts, ", ")
}

// CompileUpsert compiles an insert with an "on conflict do update" clause.
//
// The update list is a list of column names, as query.Grammar declares it, so
// each one takes the value the conflicting insert carried -- that is what
// "excluded" names.
func (g *PostgresGrammar) CompileUpsert(q *query.Builder, values []map[string]any, uniqueBy []string, update []string) string {
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

// CompileJoinLateral compiles a lateral join clause, appending "on true" since
// the join condition is expressed inside the lateral subquery itself.
func (g *PostgresGrammar) CompileJoinLateral(join *query.JoinClause, expression string) (string, error) {
	return strings.TrimSpace(join.Type + " join lateral " + expression + " on true"), nil
}

// compileJSONUpdateColumn compiles a set clause for one JSON path, using
// jsonb_set to write the value at that path without touching the rest of the
// document.
func (g *PostgresGrammar) compileJSONUpdateColumn(key string, value any) string {
	d := g.self

	segments := strings.Split(key, "->")
	field := d.Wrap(segments[0])
	path := "'{" + strings.Join(g.wrapJSONPathAttributes(segments[1:], `"`), ",") + "}'"

	return field + " = jsonb_set(" + field + "::jsonb, " + path + ", " + d.Parameter(value) + ")"
}

// CompileUpdateFrom compiles an update whose joins compile as a from clause,
// since Postgres lists the joined tables there rather than beside the table
// being updated.
func (g *PostgresGrammar) CompileUpdateFrom(q *query.Builder, values map[string]any) string {
	if g.Grammar.compilationError(q) != nil {
		return ""
	}
	d := g.self
	table := d.WrapTable(q.GetFrom())
	columns := d.CompileUpdateColumns(q, values)

	from := ""
	if len(q.Joins) > 0 {
		tables := make([]string, 0, len(q.Joins))
		for _, join := range q.Joins {
			tables = append(tables, d.WrapTable(join.Table))
		}
		from = " from " + strings.Join(tables, ", ")
	}

	return strings.TrimSpace("update " + table + " set " + columns + from + " " + g.compileUpdateWheres(q))
}

// compileUpdateWheres compiles the where clause of an update-from, folding in
// the join conditions compiled by compileUpdateJoinWheres.
func (g *PostgresGrammar) compileUpdateWheres(q *query.Builder) string {
	baseWheres := g.self.CompileWheres(q)

	if len(q.Joins) == 0 {
		return baseWheres
	}

	joinWheres := g.compileUpdateJoinWheres(q)

	if strings.TrimSpace(baseWheres) == "" {
		return "where " + removeLeadingBoolean(joinWheres)
	}

	return baseWheres + " " + joinWheres
}

// compileUpdateJoinWheres compiles the join conditions of an update as where
// clauses, because there is no join clause left to hang them on.
func (g *PostgresGrammar) compileUpdateJoinWheres(q *query.Builder) string {
	parts := make([]string, 0)

	for _, join := range q.Joins {
		for _, where := range join.Wheres {
			boolean := where.Boolean
			if boolean == "" {
				boolean = "and"
			}
			parts = append(parts, boolean+" "+g.compileWhere(q, where))
		}
	}

	return strings.Join(parts, " ")
}

// PrepareBindingsForUpdateFrom orders the bindings of an update-from to match
// the set list, then the where clause, then every other clause in order.
func (g *PostgresGrammar) PrepareBindingsForUpdateFrom(bindings map[string][]any, values map[string]any) []any {
	out := make([]any, 0, len(values)+8)

	for _, column := range sortedKeys(values) {
		out = append(out, g.jsonBinding(column, values[column]))
	}

	out = append(out, bindings["where"]...)

	for _, key := range bindingOrder {
		if key == "select" || key == "where" {
			continue
		}
		out = append(out, bindings[key]...)
	}

	return out
}

// compileUpdateWithJoinsOrLimit compiles an update with joins or a limit.
//
// Postgres has no limit on an update, so the rows are named by their ctid --
// their physical address -- and chosen by a select that does have one. The
// select is compiled from a clone, so the ctid column added for it never
// leaks onto the caller's own builder.
func (g *PostgresGrammar) compileUpdateWithJoinsOrLimit(q *query.Builder, values map[string]any) string {
	d := g.self
	table := d.WrapTable(q.GetFrom())
	columns := d.CompileUpdateColumns(q, values)

	alias := lastAliasSegment(text(q.GetFrom()))
	selectSQL := d.CompileSelect(q.Clone().Select(alias + ".ctid"))

	return "update " + table + " set " + columns + " where " + d.Wrap("ctid") + " in (" + selectSQL + ")"
}

// PrepareBindingsForUpdate orders the bindings of an update to match the set
// list followed by every other clause.
//
// The join bindings are not moved to the front the way the base grammar moves
// them, because a Postgres update compiles its joins after the set list. An
// Expression stays in the list as itself; see Grammar.PrepareBindingsForUpdate.
func (g *PostgresGrammar) PrepareBindingsForUpdate(bindings map[string][]any, values map[string]any) []any {
	out := make([]any, 0, len(values)+8)

	for _, column := range sortedKeys(values) {
		out = append(out, g.jsonBinding(column, values[column]))
	}

	for _, key := range bindingOrder {
		if key == "select" {
			continue
		}
		out = append(out, bindings[key]...)
	}

	return out
}

// jsonBinding is the value transformation both of the Postgres binding
// preparers do: an array or an object, and anything bound to a JSON path,
// travels as JSON text.
func (g *PostgresGrammar) jsonBinding(column string, value any) any {
	if isStructured(value) || (isJSONSelector(column) && !query.IsExpression(value)) {
		if encoded, err := encodeJSON(value); err == nil {
			return encoded
		}
	}
	return value
}

// CompileDelete compiles a delete statement, routing through
// compileDeleteWithJoinsOrLimit when the query has joins or a limit.
func (g *PostgresGrammar) CompileDelete(q *query.Builder) string {
	if g.Grammar.compilationError(q) != nil {
		return ""
	}
	if len(q.Joins) > 0 || q.GetLimit() != nil {
		return g.compileDeleteWithJoinsOrLimit(q)
	}
	return g.Grammar.CompileDelete(q)
}

// compileDeleteWithJoinsOrLimit compiles a delete with joins or a limit,
// naming the rows by ctid and choosing them with a select.
func (g *PostgresGrammar) compileDeleteWithJoinsOrLimit(q *query.Builder) string {
	d := g.self
	table := d.WrapTable(q.GetFrom())

	alias := lastAliasSegment(text(q.GetFrom()))
	selectSQL := d.CompileSelect(q.Clone().Select(alias + ".ctid"))

	return "delete from " + table + " where " + d.Wrap("ctid") + " in (" + selectSQL + ")"
}

// CompileTruncate compiles a truncate statement, appending cascade when
// CascadeOnTruncate is enabled.
func (g *PostgresGrammar) CompileTruncate(q *query.Builder) map[string][]any {
	if g.Grammar.compilationError(q) != nil {
		return nil
	}
	sql := "truncate " + g.self.WrapTable(q.GetFrom()) + " restart identity"
	if cascadeTruncate.Load() {
		sql += " cascade"
	}
	return map[string][]any{sql: {}}
}

// CompileThreadCount compiles a query counting the server's active
// connections.
func (g *PostgresGrammar) CompileThreadCount() string {
	return `select count(*) as "Value" from pg_stat_activity`
}

// WrapJSONSelector wraps a JSON path selector. The last step of the path is
// ->> so the value comes back as text, and the ones before it are -> so they
// stay JSON.
func (g *PostgresGrammar) WrapJSONSelector(value string) string {
	path := strings.Split(value, "->")

	field := g.wrapSegments(strings.Split(path[0], "."))

	wrappedPath := g.wrapJSONPathAttributes(path[1:], "'")
	if len(wrappedPath) == 0 {
		return field
	}

	attribute := wrappedPath[len(wrappedPath)-1]
	wrappedPath = wrappedPath[:len(wrappedPath)-1]

	if len(wrappedPath) > 0 {
		return field + "->" + strings.Join(wrappedPath, "->") + "->>" + attribute
	}

	return field + "->>" + attribute
}

// WrapJSONBooleanSelector wraps a JSON path selector for a boolean comparison,
// keeping the result as jsonb rather than unwrapping it to text.
func (g *PostgresGrammar) WrapJSONBooleanSelector(value string) string {
	return "(" + strings.ReplaceAll(g.self.WrapJSONSelector(value), "->>", "->") + ")::jsonb"
}

// WrapJSONBooleanValue wraps a literal boolean value as jsonb, to compare
// against a boolean JSON selector.
func (g *PostgresGrammar) WrapJSONBooleanValue(value string) string {
	return "'" + value + "'::jsonb"
}

// wrapJSONPathAttributes wraps each attribute of a JSON path, quoting object
// keys and leaving array indexes bare.
//
// The quote is a parameter because the same path is spelled with single quotes
// inside an operator chain and with double quotes inside the array literal
// jsonb_set takes.
func (g *PostgresGrammar) wrapJSONPathAttributes(path []string, quote string) []string {
	out := make([]string, 0, len(path))

	for _, attribute := range path {
		for _, key := range parseJSONPathArrayKeys(attribute) {
			if _, err := strconv.Atoi(key); err == nil {
				out = append(out, key)
				continue
			}
			out = append(out, quote+key+quote)
		}
	}

	return out
}

// jsonPathKeys matches the contents of each bracketed segment of a JSON path,
// such as the 0 and 1 in "tags[0][1]".
var jsonPathKeys = regexp.MustCompile(`\[([^\]]+)\]`)

// parseJSONPathArrayKeys splits one path segment into its attribute and any
// trailing array indexes: "tags[0][1]" is one attribute and two indexes.
func parseJSONPathArrayKeys(attribute string) []string {
	parts := jsonPathArrayKeys.FindString(attribute)
	if parts == "" {
		return []string{attribute}
	}

	out := make([]string, 0, 2)
	if key := strings.TrimSuffix(attribute, parts); key != "" {
		out = append(out, key)
	}

	for _, match := range jsonPathKeys.FindAllStringSubmatch(parts, -1) {
		if match[1] != "" {
			out = append(out, match[1])
		}
	}

	return out
}

// SubstituteBindingsIntoRawSQL substitutes bindings into raw SQL for display.
// The doubled question marks the operators carry are written back as single
// ones, since the result is for reading rather than for running.
func (g *PostgresGrammar) SubstituteBindingsIntoRawSQL(sql string, bindings []any) (string, error) {
	out, err := g.Grammar.SubstituteBindingsIntoRawSQL(sql, bindings)
	if err != nil {
		return "", err
	}

	for _, operator := range postgresOperators {
		if !strings.Contains(operator, "?") {
			continue
		}
		out = strings.ReplaceAll(out, strings.ReplaceAll(operator, "?", "??"), operator)
	}

	return out, nil
}

// lastAliasSegment splits a from clause on " as " and returns the last
// segment: the name a query refers to its own table by.
func lastAliasSegment(from string) string {
	segments := aliasPattern.Split(from, -1)
	return segments[len(segments)-1]
}
