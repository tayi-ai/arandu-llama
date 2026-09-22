package grammars

import (
	"errors"
	"fmt"
	"regexp"
	"slices"
	"strconv"
	"strings"

	"github.com/arandu-io/hesape/database/query"
)

// dialect is the set of methods the compile pipeline calls on itself.
//
// Go resolves a call through an embedded type at compile time, against the
// type that declared it. Without indirection, a driver that overrides
// compileLock would never be seen by compileSelect: the base version would
// always run, every driver difference would silently disappear -- backticks
// would never appear, "for update" would never appear, and nothing would
// fail.
//
// The self reference restores the dynamic dispatch that indirection removes:
// Grammar holds the outermost grammar, and every internal call goes through
// it. It is unexported because the four grammars that satisfy it live in
// this package, and nothing outside implements a dialect without also being
// one of them.
type dialect interface {
	// Identifier and value wrapping.
	Wrap(value any) string
	WrapTable(table any) string
	WrapValue(value string) string
	WrapArray(values []any) []string
	WrapJSONSelector(value string) string
	WrapJSONBooleanSelector(value string) string
	WrapJSONBooleanValue(value string) string
	WrapUnion(sql string) string
	Columnize(columns []any) string
	Parameter(value any) string
	Parameterize(values []any) string
	QuoteString(value any) string
	Escape(value any, binary bool) (string, error)
	GetValue(expression any) any
	GetTablePrefix() string

	// The select pipeline.
	CompileSelect(q *query.Builder) string
	CompileAggregate(q *query.Builder, aggregate *query.Aggregate) string
	CompileColumns(q *query.Builder, columns []any) string
	CompileFrom(q *query.Builder, table any) string
	CompileIndexHint(q *query.Builder, indexHint *query.IndexHint) string
	CompileJoins(q *query.Builder, joins []*query.JoinClause) string
	CompileWheres(q *query.Builder) string
	CompileGroups(q *query.Builder, groups []any) string
	CompileHavings(q *query.Builder) string
	CompileHaving(having query.Having) string
	CompileOrders(q *query.Builder, orders []query.Order) string
	CompileLimit(q *query.Builder, limit int) string
	CompileOffset(q *query.Builder, offset int) string
	CompileLock(q *query.Builder, value any) string
	CompileGroupLimit(q *query.Builder) string
	CompileRandom(seed string) string
	SupportsStraightJoins() (bool, error)
	CompileJoinLateral(join *query.JoinClause, expression string) (string, error)

	// The where compilers. Each one is an override point on the interface,
	// whether or not a driver in this package actually redefines it, because
	// the dispatch in compileWhere has to reach whichever version the active
	// driver provides.
	WhereRaw(q *query.Builder, where query.Where) string
	WhereBasic(q *query.Builder, where query.Where) string
	WhereBitwise(q *query.Builder, where query.Where) string
	WhereLike(q *query.Builder, where query.Where) string
	WhereIn(q *query.Builder, where query.Where) string
	WhereNotIn(q *query.Builder, where query.Where) string
	WhereInRaw(q *query.Builder, where query.Where) string
	WhereNotInRaw(q *query.Builder, where query.Where) string
	WhereNull(q *query.Builder, where query.Where) string
	WhereNotNull(q *query.Builder, where query.Where) string
	WhereBetween(q *query.Builder, where query.Where) string
	WhereBetweenColumns(q *query.Builder, where query.Where) string
	WhereValueBetween(q *query.Builder, where query.Where) string
	WhereDate(q *query.Builder, where query.Where) string
	WhereTime(q *query.Builder, where query.Where) string
	WhereDay(q *query.Builder, where query.Where) string
	WhereMonth(q *query.Builder, where query.Where) string
	WhereYear(q *query.Builder, where query.Where) string
	DateBasedWhere(typ string, q *query.Builder, where query.Where) string
	WhereColumn(q *query.Builder, where query.Where) string
	WhereNested(q *query.Builder, where query.Where) string
	WhereSub(q *query.Builder, where query.Where) string
	WhereExists(q *query.Builder, where query.Where) string
	WhereNotExists(q *query.Builder, where query.Where) string
	WhereRowValues(q *query.Builder, where query.Where) string
	WhereJSONBoolean(q *query.Builder, where query.Where) string
	WhereJSONContains(q *query.Builder, where query.Where) string
	WhereJSONOverlaps(q *query.Builder, where query.Where) string
	WhereJSONContainsKey(q *query.Builder, where query.Where) string
	WhereJSONLength(q *query.Builder, where query.Where) string
	WhereFullText(q *query.Builder, where query.Where) (string, error)
	WhereExpression(q *query.Builder, where query.Where) string

	// The JSON operators, which is where the three dialects agree least.
	CompileJSONContains(column any, value string) (string, error)
	CompileJSONOverlaps(column any, value string) (string, error)
	CompileJSONContainsKey(column any) (string, error)
	CompileJSONLength(column any, operator, value string) (string, error)
	CompileJSONValueCast(value string) string
	PrepareBindingForJSONContains(binding any) (any, error)

	// The write pipeline.
	CompileInsert(q *query.Builder, values []map[string]any) string
	CompileInsertUsing(q *query.Builder, columns []any, sql string) string
	CompileInsertGetID(q *query.Builder, values map[string]any, sequence string) string
	CompileUpdate(q *query.Builder, values map[string]any) string
	CompileUpdateColumns(q *query.Builder, values map[string]any) string
	CompileUpdateWithoutJoins(q *query.Builder, table, columns, where string) string
	CompileUpdateWithJoins(q *query.Builder, table, columns, where string) string
	CompileDelete(q *query.Builder) string
	CompileDeleteWithoutJoins(q *query.Builder, table, where string) string
	CompileDeleteWithJoins(q *query.Builder, table, where string) string
	CompileTruncate(q *query.Builder) map[string][]any
	PrepareBindingsForUpdate(bindings map[string][]any, values map[string]any) []any
	SubstituteBindingsIntoRawSQL(sql string, bindings []any) (string, error)
}

// Grammar is the standard SQL a driver grammar starts from.
//
// It embeds query.BaseGrammar, which already carries what no dialect disagrees
// about -- the operator lists, the table prefix, the savepoint statements, the
// date format, literal quoting and escaping. What it adds is the compile
// pipeline.
//
// # It is incomplete, and the compiler knows it
//
// CompileInsertOrIgnore and CompileUpsert are absent: every engine spells them
// differently and none can use the standard form. So *Grammar does not satisfy
// query.Grammar and cannot be handed to a builder at all -- the gap is found
// when the package compiles rather than when the statement runs.
type Grammar struct {
	query.BaseGrammar

	self dialect
}

// NewGrammar creates a Grammar with its self reference wired to itself, the
// value the compile pipeline dispatches through.
//
// A driver grammar embeds the result and then points self at itself instead,
// which is what lets the driver's own overrides take part in the dispatch.
func NewGrammar() *Grammar {
	g := &Grammar{}
	g.self = g
	return g
}

// compilationError runs the builder's final barrier under the policy of the
// dialect about to spell the SQL. Compile methods are public, so they cannot
// assume the caller came through Builder.ToSQL or an execution method, and a
// builder assembled for one dialect can be handed straight to another's
// compiler. They deliberately do not run callbacks: the builder owns that
// lifecycle, including callbacks queued for a later compilation.
func (g *Grammar) compilationError(q *query.Builder) error {
	if q == nil {
		return errors.New("query/grammars: cannot compile a nil query")
	}
	return q.ValidateForCompilationWith(g.compilingDialect())
}

// joinsError runs the same barrier over join clauses handed in as a
// parameter, under the dialect about to spell them.
func (g *Grammar) joinsError(joins []*query.JoinClause) error {
	dialect := g.compilingDialect()
	for _, join := range joins {
		if join == nil {
			continue
		}
		if err := join.ValidateForCompilationWith(dialect); err != nil {
			return err
		}
	}
	return nil
}

// compilingDialect returns the outermost grammar of the compile pipeline as
// an operator policy, or nil when it is not one: Grammar alone is incomplete
// on purpose and no builder can have been given it.
func (g *Grammar) compilingDialect() query.Grammar {
	if dialect, ok := g.self.(query.Grammar); ok {
		return dialect
	}
	return nil
}

// selectComponents lists the clauses of a select, in the order they are
// concatenated.
var selectComponents = []string{
	"aggregate", "columns", "from", "indexHint", "joins",
	"wheres", "groups", "havings", "orders", "limit", "offset", "lock",
}

// bindingOrder is the binding key order query.Builder uses internally.
//
// The list also lives in query, unexported, and PrepareBindingsForUpdate has
// to walk it to flatten the bindings map in the same order: a binding read
// out of order lands in the wrong placeholder, which is a wrong answer
// rather than an error.
var bindingOrder = []string{"select", "from", "join", "where", "groupBy", "having", "order", "union", "unionOrder"}

// component is one entry of compileComponents' return.
//
// A Go map has no defined order, and the order here is the statement, so the
// pieces are carried in a slice and looked up by name where compileGroupLimit
// needs them.
type component struct {
	name string
	sql  string
}

// CompileSelect builds the SQL for a select statement, or delegates to the
// group-limit or union-aggregate path when the query needs one.
func (g *Grammar) CompileSelect(q *query.Builder) string {
	if g.compilationError(q) != nil {
		return ""
	}
	if (len(q.Unions) > 0 || len(q.Havings) > 0) && q.GetAggregate() != nil {
		return g.compileUnionAggregate(q)
	}

	// A group limit needs a different shape entirely, which is what makes an
	// eager load with a limit per parent possible.
	if q.GetGroupLimit() != nil {
		if q.Columns == nil {
			q.Columns = []any{"*"}
		}
		return g.self.CompileGroupLimit(q)
	}

	return g.compileSelectWithoutGroupLimit(q)
}

// compileSelectWithoutGroupLimit is compileSelect's ordinary path: every
// select that does not need a group limit's window function.
//
// The group limit cannot be cleared from outside query -- Builder keeps it
// unexported and CloneWithout does not name it -- so this path is named
// directly instead of reached by mutating a cleared clone, which is the same
// statement without a round trip through a mutation.
func (g *Grammar) compileSelectWithoutGroupLimit(q *query.Builder) string {
	d := g.self

	original := q.Columns
	if q.Columns == nil {
		q.Columns = []any{"*"}
	}

	sql := strings.TrimSpace(concatenate(g.compileComponents(q)))

	if len(q.Unions) > 0 {
		sql = d.WrapUnion(sql) + " " + g.compileUnions(q)
	}

	q.Columns = original

	return sql
}

// compileComponents compiles each present component of the select and
// returns them in order.
//
// A component is included when its field is non-nil, even if it is an empty
// slice: the empty cases compile to the empty string and concatenate drops
// them, so testing for nil rather than length reaches the same statement.
func (g *Grammar) compileComponents(q *query.Builder) []component {
	d := g.self
	out := make([]component, 0, len(selectComponents))

	if aggregate := q.GetAggregate(); aggregate != nil {
		out = append(out, component{"aggregate", d.CompileAggregate(q, aggregate)})
	}
	if q.Columns != nil {
		out = append(out, component{"columns", d.CompileColumns(q, q.Columns)})
	}
	if from := q.GetFrom(); from != nil {
		out = append(out, component{"from", d.CompileFrom(q, from)})
	}
	if indexHint := q.GetIndexHint(); indexHint != nil {
		out = append(out, component{"indexHint", d.CompileIndexHint(q, indexHint)})
	}
	if len(q.Joins) > 0 {
		out = append(out, component{"joins", d.CompileJoins(q, q.Joins)})
	}
	out = append(out, component{"wheres", d.CompileWheres(q)})
	if len(q.Groups) > 0 {
		out = append(out, component{"groups", d.CompileGroups(q, q.Groups)})
	}
	if len(q.Havings) > 0 {
		out = append(out, component{"havings", d.CompileHavings(q)})
	}
	if len(q.Orders) > 0 {
		out = append(out, component{"orders", d.CompileOrders(q, q.Orders)})
	}
	if limit := q.GetLimit(); limit != nil {
		out = append(out, component{"limit", d.CompileLimit(q, *limit)})
	}
	if offset := q.GetOffset(); offset != nil {
		out = append(out, component{"offset", d.CompileOffset(q, *offset)})
	}
	if lock := q.GetLock(); lock != nil {
		out = append(out, component{"lock", d.CompileLock(q, lock)})
	}

	return out
}

// CompileAggregate builds the SQL for an aggregate function over the given
// columns.
func (g *Grammar) CompileAggregate(q *query.Builder, aggregate *query.Aggregate) string {
	d := g.self
	column := d.Columnize(aggregate.Columns)

	// A distinct aggregate counts distinct values, not distinct rows, so the
	// keyword goes inside the function call.
	switch distinct := q.GetDistinct().(type) {
	case []any:
		column = "distinct " + d.Columnize(distinct)
	case bool:
		if distinct && column != "*" {
			column = "distinct " + column
		}
	}

	return "select " + aggregate.Function + "(" + column + ") as aggregate"
}

// CompileColumns builds the select list, honouring the query's distinct
// setting.
func (g *Grammar) CompileColumns(q *query.Builder, columns []any) string {
	if q.GetAggregate() != nil {
		return ""
	}
	if distinct, ok := q.GetDistinct().(bool); ok && distinct {
		return "select distinct " + g.self.Columnize(columns)
	}
	if _, ok := q.GetDistinct().([]any); ok {
		return "select distinct " + g.self.Columnize(columns)
	}
	return "select " + g.self.Columnize(columns)
}

// CompileFrom builds the SQL from clause for the given table.
func (g *Grammar) CompileFrom(q *query.Builder, table any) string {
	return "from " + g.self.WrapTable(table)
}

// CompileIndexHint builds the index hint component, returning the empty
// string here because the base grammar targets no particular engine's hint
// syntax. That matches what an engine without index hints does with the
// request anyway -- a hint changes the plan, never the rows.
func (g *Grammar) CompileIndexHint(q *query.Builder, indexHint *query.IndexHint) string {
	return ""
}

// CompileJoins builds the SQL for the given join clauses.
//
// The joins are a parameter rather than being read off the query, so they are
// screened alongside it: a guard that clears the builder and then spells a
// slice nobody looked at is protection in name only. Each refusal is recorded
// on the join clause that carries it, and the whole call returns the empty
// string.
func (g *Grammar) CompileJoins(q *query.Builder, joins []*query.JoinClause) string {
	if g.compilationError(q) != nil || g.joinsError(joins) != nil {
		return ""
	}
	d := g.self
	parts := make([]string, 0, len(joins))

	for _, join := range joins {
		table := d.WrapTable(join.Table)

		tableAndNestedJoins := table
		if len(join.Joins) > 0 {
			tableAndNestedJoins = "(" + table + " " + d.CompileJoins(q, join.Joins) + ")"
		}

		// A lateral join has a shape of its own -- it takes no on clause, and
		// the engines that have one spell it differently -- so the whole of it
		// comes from the dialect. query.JoinClause carries a Lateral flag to
		// mark it.
		if join.Lateral {
			sql, err := d.CompileJoinLateral(join, tableAndNestedJoins)
			if err != nil {
				parts = append(parts, unsupportedJoin(err))
				continue
			}
			parts = append(parts, sql)
			continue
		}

		// A straight join is MySQL's; every other engine's SupportsStraightJoins
		// reports it unsupported, so the type is compiled straight through and
		// left for the engine itself to reject.
		joinWord := " join"
		if join.Type == "straight_join" {
			if supported, err := d.SupportsStraightJoins(); err == nil && supported {
				joinWord = ""
			}
		}

		parts = append(parts, strings.TrimSpace(
			join.Type+joinWord+" "+tableAndNestedJoins+" "+g.compileJoinConstraints(join),
		))
	}

	return strings.Join(parts, " ")
}

// CompileJoinLateral builds the SQL for a lateral join. The base
// implementation returns an error: only a driver that supports lateral joins
// overrides it.
//
// It takes a *query.JoinClause because query has no separate lateral join
// type; the Lateral flag on the ordinary clause marks it instead.
func (g *Grammar) CompileJoinLateral(join *query.JoinClause, expression string) (string, error) {
	return "", fmt.Errorf("query/grammars: this database engine does not support lateral joins")
}

// SupportsStraightJoins reports whether the dialect supports MySQL's
// straight_join. The base implementation returns an error: only MySQL
// overrides it.
func (g *Grammar) SupportsStraightJoins() (bool, error) {
	return false, fmt.Errorf("query/grammars: this database engine does not support straight joins")
}

// CompileWheres builds the SQL where clause for the query, or the empty
// string if it has no where clauses.
func (g *Grammar) CompileWheres(q *query.Builder) string {
	if g.compilationError(q) != nil {
		return ""
	}
	clauses := g.whereClauses(q)
	if clauses == "" {
		return ""
	}
	return "where " + clauses
}

// compileJoinConstraints is compileWheres for a join clause, where the
// conjunction is "on" rather than "where".
//
// The grammar is handed the JoinClause itself, rather than its embedded
// *Builder, because the embedded value has forgotten what wraps it.
func (g *Grammar) compileJoinConstraints(join *query.JoinClause) string {
	clauses := g.whereClauses(join.Builder)
	if clauses == "" {
		return ""
	}
	return "on " + clauses
}

// whereClauses compiles a query's where clauses into a single string, joined
// by their booleans, with the leading conjunction removed.
//
// Keeping the conjunction out until the caller adds it is what lets
// whereNested and compileJoinConstraints reuse this, without depending on
// the conjunction's length the way trimming a fixed prefix would.
func (g *Grammar) whereClauses(q *query.Builder) string {
	if len(q.Wheres) == 0 {
		return ""
	}

	parts := make([]string, 0, len(q.Wheres))
	for _, where := range q.Wheres {
		boolean := where.Boolean
		if boolean == "" {
			boolean = "and"
		}
		parts = append(parts, boolean+" "+g.compileWhere(q, where))
	}

	return removeLeadingBoolean(strings.Join(parts, " "))
}

// compileWhere dispatches a where clause to the method that compiles its
// type.
//
// The builder spells negation with a Not flag rather than a distinct type
// name -- "In" plus Not, rather than "NotIn" -- so whereType folds both
// spellings into the same case before this switch runs.
func (g *Grammar) compileWhere(q *query.Builder, where query.Where) string {
	d := g.self
	typ, not := whereType(where)
	where.Not = not

	switch typ {
	case "raw":
		return d.WhereRaw(q, where)
	case "basic":
		return d.WhereBasic(q, where)
	case "bitwise":
		return d.WhereBitwise(q, where)
	case "like":
		return d.WhereLike(q, where)
	case "in":
		if not {
			return d.WhereNotIn(q, where)
		}
		return d.WhereIn(q, where)
	case "inraw":
		if not {
			return d.WhereNotInRaw(q, where)
		}
		return d.WhereInRaw(q, where)
	case "null":
		if not {
			return d.WhereNotNull(q, where)
		}
		return d.WhereNull(q, where)
	case "between":
		return d.WhereBetween(q, where)
	case "betweencolumns":
		return d.WhereBetweenColumns(q, where)
	case "valuebetween":
		return d.WhereValueBetween(q, where)
	case "date":
		return d.WhereDate(q, where)
	case "time":
		return d.WhereTime(q, where)
	case "day":
		return d.WhereDay(q, where)
	case "month":
		return d.WhereMonth(q, where)
	case "year":
		return d.WhereYear(q, where)
	case "column":
		return d.WhereColumn(q, where)
	case "nested":
		return d.WhereNested(q, where)
	case "sub":
		return d.WhereSub(q, where)
	case "exists":
		if not {
			return d.WhereNotExists(q, where)
		}
		return d.WhereExists(q, where)
	case "rowvalues":
		return d.WhereRowValues(q, where)
	case "jsonboolean":
		return d.WhereJSONBoolean(q, where)
	case "jsoncontains":
		return d.WhereJSONContains(q, where)
	case "jsonoverlaps":
		return d.WhereJSONOverlaps(q, where)
	case "jsoncontainskey":
		return d.WhereJSONContainsKey(q, where)
	case "jsonlength":
		return d.WhereJSONLength(q, where)
	case "fulltext":
		sql, err := d.WhereFullText(q, where)
		if err != nil {
			return unsupportedClause(err)
		}
		return sql
	case "expression":
		return d.WhereExpression(q, where)
	default:
		// A clause the grammar cannot spell must not be dropped: a filter that
		// quietly disappears is a tenant reading another tenant's rows, so the
		// query is made false instead and carries the reason.
		return unsupportedClause(fmt.Errorf("query/grammars: no compiler for where clause type %q", where.Type))
	}
}

// whereType normalises a clause type to the switch's spelling and folds
// "Not..." type names into the Not flag.
func whereType(where query.Where) (typ string, not bool) {
	typ = strings.ToLower(where.Type)
	not = where.Not

	switch typ {
	case "notnull":
		return "null", true
	case "notin":
		return "in", true
	case "notinraw":
		return "inraw", true
	case "notexists":
		return "exists", true
	case "notbetween":
		return "between", true
	}

	return typ, not
}

// WhereRaw compiles a raw where clause, writing its SQL expression as-is.
func (g *Grammar) WhereRaw(q *query.Builder, where query.Where) string {
	return text(g.self.GetValue(where.SQL))
}

// WhereBasic compiles a simple "column operator value" where clause.
//
// The operator has its question marks doubled: Postgres spells several of
// its operators with one -- ?, ?| and ?& all test for a JSON key -- and a
// bare one would be read as a placeholder. PostgresGrammar's
// SubstituteBindingsIntoRawSQL undoes the doubling, which is the other half
// of the same convention.
func (g *Grammar) WhereBasic(q *query.Builder, where query.Where) string {
	d := g.self
	value := d.Parameter(where.Value)
	operator := strings.ReplaceAll(where.Operator, "?", "??")

	return d.Wrap(where.Column) + " " + operator + " " + value
}

// WhereBitwise compiles a bitwise where clause the same way as a basic one.
func (g *Grammar) WhereBitwise(q *query.Builder, where query.Where) string {
	return g.self.WhereBasic(q, where)
}

// WhereLike compiles a like where clause, refusing as a false clause when
// case sensitivity was asked for and the engine cannot provide it: answering
// with the case insensitive form instead would return rows nobody filtered
// for.
func (g *Grammar) WhereLike(q *query.Builder, where query.Where) string {
	if where.CaseSensitive {
		return unsupportedClause(fmt.Errorf("query/grammars: this database engine does not support case sensitive like operations"))
	}

	where.Operator = "like"
	if where.Not {
		where.Operator = "not like"
	}

	return g.self.WhereBasic(q, where)
}

// WhereIn compiles an "in" where clause.
//
// An empty list compiles to "0 = 1" rather than being dropped: a filter over
// an empty set matches nothing, and dropping it would match everything.
func (g *Grammar) WhereIn(q *query.Builder, where query.Where) string {
	if len(where.Values) == 0 {
		return "0 = 1"
	}
	d := g.self
	return d.Wrap(where.Column) + " in (" + d.Parameterize(where.Values) + ")"
}

// WhereNotIn compiles a "not in" where clause.
func (g *Grammar) WhereNotIn(q *query.Builder, where query.Where) string {
	if len(where.Values) == 0 {
		return "1 = 1"
	}
	d := g.self
	return d.Wrap(where.Column) + " not in (" + d.Parameterize(where.Values) + ")"
}

// WhereInRaw compiles an "in" where clause with its values written
// directly into the statement rather than bound, which is only safe because
// whereIntegerInRaw is the one caller and it has already cast every value to
// an integer.
func (g *Grammar) WhereInRaw(q *query.Builder, where query.Where) string {
	if len(where.Values) == 0 {
		return "0 = 1"
	}
	return g.self.Wrap(where.Column) + " in (" + joinValues(where.Values) + ")"
}

// WhereNotInRaw compiles a "not in" where clause the same way as
// WhereInRaw: values written directly into the statement.
func (g *Grammar) WhereNotInRaw(q *query.Builder, where query.Where) string {
	if len(where.Values) == 0 {
		return "1 = 1"
	}
	return g.self.Wrap(where.Column) + " not in (" + joinValues(where.Values) + ")"
}

// WhereNull compiles an "is null" where clause.
func (g *Grammar) WhereNull(q *query.Builder, where query.Where) string {
	return g.self.Wrap(where.Column) + " is null"
}

// WhereNotNull compiles an "is not null" where clause.
func (g *Grammar) WhereNotNull(q *query.Builder, where query.Where) string {
	return g.self.Wrap(where.Column) + " is not null"
}

// WhereBetween compiles a "between" where clause.
func (g *Grammar) WhereBetween(q *query.Builder, where query.Where) string {
	d := g.self
	between := "between"
	if where.Not {
		between = "not between"
	}
	first, last := ends(where.Values)
	return d.Wrap(where.Column) + " " + between + " " + d.Parameter(first) + " and " + d.Parameter(last)
}

// WhereBetweenColumns compiles a "between" where clause whose bounds are
// columns rather than bound values.
func (g *Grammar) WhereBetweenColumns(q *query.Builder, where query.Where) string {
	d := g.self
	between := "between"
	if where.Not {
		between = "not between"
	}
	first, last := ends(where.Values)
	return d.Wrap(where.Column) + " " + between + " " + d.Wrap(first) + " and " + d.Wrap(last)
}

// WhereValueBetween compiles a "between" where clause with the comparison
// the other way round: the value is the literal being tested, and the two
// columns are the bounds.
func (g *Grammar) WhereValueBetween(q *query.Builder, where query.Where) string {
	d := g.self
	between := "between"
	if where.Not {
		between = "not between"
	}
	first, last := ends(where.Columns)
	return d.Parameter(where.Value) + " " + between + " " + d.Wrap(first) + " and " + d.Wrap(last)
}

// WhereDate compiles a where clause that compares a column's date part.
func (g *Grammar) WhereDate(q *query.Builder, where query.Where) string {
	return g.self.DateBasedWhere("date", q, where)
}

// WhereTime compiles a where clause that compares a column's time part.
func (g *Grammar) WhereTime(q *query.Builder, where query.Where) string {
	return g.self.DateBasedWhere("time", q, where)
}

// WhereDay compiles a where clause that compares a column's day part.
func (g *Grammar) WhereDay(q *query.Builder, where query.Where) string {
	return g.self.DateBasedWhere("day", q, where)
}

// WhereMonth compiles a where clause that compares a column's month part.
func (g *Grammar) WhereMonth(q *query.Builder, where query.Where) string {
	return g.self.DateBasedWhere("month", q, where)
}

// WhereYear compiles a where clause that compares a column's year part.
func (g *Grammar) WhereYear(q *query.Builder, where query.Where) string {
	return g.self.DateBasedWhere("year", q, where)
}

// DateBasedWhere compiles the shared form the date, time, day, month and
// year where clauses all use.
func (g *Grammar) DateBasedWhere(typ string, q *query.Builder, where query.Where) string {
	d := g.self
	return typ + "(" + d.Wrap(where.Column) + ") " + where.Operator + " " + d.Parameter(where.Value)
}

// WhereColumn compiles a where clause comparing two columns: both sides are
// names, so neither becomes a binding.
func (g *Grammar) WhereColumn(q *query.Builder, where query.Where) string {
	d := g.self
	return d.Wrap(where.First) + " " + where.Operator + " " + d.Wrap(where.Second)
}

// WhereNested compiles a parenthesised group of where clauses.
func (g *Grammar) WhereNested(q *query.Builder, where query.Where) string {
	if where.Query == nil {
		return ""
	}
	return "(" + g.whereClauses(where.Query) + ")"
}

// WhereSub compiles a where clause comparing a column against a subquery.
//
// A clause of this type with no subquery is a builder bug. It compiles
// false here instead of panicking: a filter that cannot be built is still a
// filter, and a panic inside a grammar takes down the request that was about
// to be told no.
func (g *Grammar) WhereSub(q *query.Builder, where query.Where) string {
	if where.Query == nil {
		return unsupportedClause(errMissingSubquery)
	}
	d := g.self
	return d.Wrap(where.Column) + " " + where.Operator + " (" + d.CompileSelect(where.Query) + ")"
}

// WhereExists compiles an "exists" where clause over a subquery.
func (g *Grammar) WhereExists(q *query.Builder, where query.Where) string {
	if where.Query == nil {
		return unsupportedClause(errMissingSubquery)
	}
	return "exists (" + g.self.CompileSelect(where.Query) + ")"
}

// WhereNotExists compiles a "not exists" where clause over a subquery.
func (g *Grammar) WhereNotExists(q *query.Builder, where query.Where) string {
	if where.Query == nil {
		return unsupportedClause(errMissingSubquery)
	}
	return "not exists (" + g.self.CompileSelect(where.Query) + ")"
}

// errMissingSubquery is the clause that names a subquery and does not carry one.
var errMissingSubquery = errors.New("query/grammars: the where clause has no subquery to compile")

// WhereRowValues compiles a row-value comparison: a tuple of columns
// against a tuple of values.
func (g *Grammar) WhereRowValues(q *query.Builder, where query.Where) string {
	d := g.self
	return "(" + d.Columnize(where.Columns) + ") " + where.Operator + " (" + d.Parameterize(where.Values) + ")"
}

// WhereJSONBoolean compiles a where clause comparing a JSON boolean value.
func (g *Grammar) WhereJSONBoolean(q *query.Builder, where query.Where) string {
	d := g.self
	column := d.WrapJSONBooleanSelector(text(where.Column))
	value := d.WrapJSONBooleanValue(d.Parameter(where.Value))
	return column + " " + where.Operator + " " + value
}

// WhereJSONContains compiles a where clause testing whether a JSON column
// contains a value.
func (g *Grammar) WhereJSONContains(q *query.Builder, where query.Where) string {
	d := g.self
	sql, err := d.CompileJSONContains(where.Column, d.Parameter(where.Value))
	if err != nil {
		return unsupportedClause(err)
	}
	if where.Not {
		return "not " + sql
	}
	return sql
}

// WhereJSONOverlaps compiles a where clause testing whether a JSON column
// overlaps a value.
func (g *Grammar) WhereJSONOverlaps(q *query.Builder, where query.Where) string {
	d := g.self
	sql, err := d.CompileJSONOverlaps(where.Column, d.Parameter(where.Value))
	if err != nil {
		return unsupportedClause(err)
	}
	if where.Not {
		return "not " + sql
	}
	return sql
}

// WhereJSONContainsKey compiles a where clause testing whether a JSON
// column contains a key.
func (g *Grammar) WhereJSONContainsKey(q *query.Builder, where query.Where) string {
	sql, err := g.self.CompileJSONContainsKey(where.Column)
	if err != nil {
		return unsupportedClause(err)
	}
	if where.Not {
		return "not " + sql
	}
	return sql
}

// WhereJSONLength compiles a where clause comparing a JSON column's
// length.
func (g *Grammar) WhereJSONLength(q *query.Builder, where query.Where) string {
	d := g.self
	sql, err := d.CompileJSONLength(where.Column, where.Operator, d.Parameter(where.Value))
	if err != nil {
		return unsupportedClause(err)
	}
	return sql
}

// WhereFullText compiles a full-text search where clause. The base
// implementation returns an error: only a driver that supports full-text
// search overrides it.
func (g *Grammar) WhereFullText(q *query.Builder, where query.Where) (string, error) {
	return "", fmt.Errorf("query/grammars: this database engine does not support fulltext search operations")
}

// WhereExpression compiles a where clause that is itself a raw expression.
func (g *Grammar) WhereExpression(q *query.Builder, where query.Where) string {
	return text(g.self.GetValue(where.Column))
}

// CompileJSONContains builds the SQL fragment testing whether a JSON column
// contains a value. Go initialisms are upper case throughout, hence JSON
// rather than Json. The base implementation returns an error: only a driver
// that supports it overrides it.
func (g *Grammar) CompileJSONContains(column any, value string) (string, error) {
	return "", fmt.Errorf("query/grammars: this database engine does not support JSON contains operations")
}

// CompileJSONOverlaps builds the SQL fragment testing whether a JSON column
// overlaps a value. The base implementation returns an error: only a driver
// that supports it overrides it.
func (g *Grammar) CompileJSONOverlaps(column any, value string) (string, error) {
	return "", fmt.Errorf("query/grammars: this database engine does not support JSON overlaps operations")
}

// CompileJSONContainsKey builds the SQL fragment testing whether a JSON
// column contains a key. The base implementation returns an error: only a
// driver that supports it overrides it.
func (g *Grammar) CompileJSONContainsKey(column any) (string, error) {
	return "", fmt.Errorf("query/grammars: this database engine does not support JSON contains key operations")
}

// CompileJSONLength builds the SQL fragment comparing a JSON column's
// length. The base implementation returns an error: only a driver that
// supports it overrides it.
func (g *Grammar) CompileJSONLength(column any, operator, value string) (string, error) {
	return "", fmt.Errorf("query/grammars: this database engine does not support JSON length operations")
}

// CompileJSONValueCast wraps a value expression for a JSON comparison. The
// base implementation returns the value unchanged.
func (g *Grammar) CompileJSONValueCast(value string) string { return value }

// PrepareBindingForJSONContains encodes a binding for a JSON contains
// comparison.
func (g *Grammar) PrepareBindingForJSONContains(binding any) (any, error) {
	return encodeJSON(binding)
}

// CompileGroups builds the SQL group by clause.
func (g *Grammar) CompileGroups(q *query.Builder, groups []any) string {
	return "group by " + g.self.Columnize(groups)
}

// CompileHavings builds the SQL having clause, or the empty string if the
// query has no having clauses.
func (g *Grammar) CompileHavings(q *query.Builder) string {
	if g.compilationError(q) != nil {
		return ""
	}
	clauses := g.havingClauses(q)
	if clauses == "" {
		return ""
	}
	return "having " + clauses
}

// havingClauses is CompileHavings without the leading keyword, so that a
// nested having can reuse it.
func (g *Grammar) havingClauses(q *query.Builder) string {
	if q == nil || len(q.Havings) == 0 {
		return ""
	}

	parts := make([]string, 0, len(q.Havings))
	for _, having := range q.Havings {
		boolean := having.Boolean
		if boolean == "" {
			boolean = "and"
		}
		parts = append(parts, boolean+" "+g.self.CompileHaving(having))
	}

	return removeLeadingBoolean(strings.Join(parts, " "))
}

// CompileHaving dispatches a having clause to the method that compiles its
// type.
func (g *Grammar) CompileHaving(having query.Having) string {
	switch strings.ToLower(having.Type) {
	case "raw":
		return having.SQL
	case "between":
		return g.CompileHavingBetween(having)
	case "null":
		return g.CompileHavingNull(having)
	case "notnull":
		return g.CompileHavingNotNull(having)
	case "bit":
		return g.CompileHavingBit(having)
	case "expression":
		return g.CompileHavingExpression(having)
	case "nested":
		return g.CompileNestedHavings(having)
	default:
		return g.CompileBasicHaving(having)
	}
}

// CompileBasicHaving compiles a simple "column operator value" having
// clause.
func (g *Grammar) CompileBasicHaving(having query.Having) string {
	d := g.self
	return d.Wrap(having.Column) + " " + having.Operator + " " + d.Parameter(having.Value)
}

// CompileHavingBetween compiles a "between" having clause.
func (g *Grammar) CompileHavingBetween(having query.Having) string {
	d := g.self
	between := "between"
	if having.Not {
		between = "not between"
	}
	first, last := ends(having.Values)
	return d.Wrap(having.Column) + " " + between + " " + d.Parameter(first) + " and " + d.Parameter(last)
}

// CompileHavingNull compiles an "is null" having clause.
func (g *Grammar) CompileHavingNull(having query.Having) string {
	return g.self.Wrap(having.Column) + " is null"
}

// CompileHavingNotNull compiles an "is not null" having clause.
func (g *Grammar) CompileHavingNotNull(having query.Having) string {
	return g.self.Wrap(having.Column) + " is not null"
}

// CompileHavingBit compiles a having clause that tests a bitwise
// comparison against zero.
func (g *Grammar) CompileHavingBit(having query.Having) string {
	d := g.self
	return "(" + d.Wrap(having.Column) + " " + having.Operator + " " + d.Parameter(having.Value) + ") != 0"
}

// CompileHavingExpression compiles a having clause that is itself a raw
// expression.
func (g *Grammar) CompileHavingExpression(having query.Having) string {
	return text(g.self.GetValue(having.Column))
}

// CompileNestedHavings compiles a parenthesised group of having clauses.
func (g *Grammar) CompileNestedHavings(having query.Having) string {
	return "(" + g.havingClauses(having.Query) + ")"
}

// CompileOrders builds the SQL order by clause, or the empty string if the
// query has no orders.
func (g *Grammar) CompileOrders(q *query.Builder, orders []query.Order) string {
	if len(orders) == 0 {
		return ""
	}
	return "order by " + strings.Join(g.compileOrdersToArray(q, orders), ", ")
}

// compileOrdersToArray compiles each order into its own SQL fragment.
func (g *Grammar) compileOrdersToArray(q *query.Builder, orders []query.Order) []string {
	d := g.self
	out := make([]string, 0, len(orders))
	for _, order := range orders {
		if order.SQL != nil {
			out = append(out, text(d.GetValue(order.SQL)))
			continue
		}
		out = append(out, d.Wrap(order.Column)+" "+order.Direction)
	}
	return out
}

// CompileLimit builds the SQL limit clause.
func (g *Grammar) CompileLimit(q *query.Builder, limit int) string {
	return "limit " + strconv.Itoa(limit)
}

// CompileOffset builds the SQL offset clause.
func (g *Grammar) CompileOffset(q *query.Builder, offset int) string {
	return "offset " + strconv.Itoa(offset)
}

// CompileGroupLimit builds a select using a window function that numbers the
// rows of each group, so the outer query can cut each group at the same
// depth.
//
// The binding move has to happen on the query the caller holds, because the
// caller reads GetBindings after the SQL is compiled. The offset is dropped
// on a clone instead of on the caller's query, which reaches the same
// statement without leaving the builder changed behind the caller's back.
func (g *Grammar) CompileGroupLimit(q *query.Builder) string {
	if g.compilationError(q) != nil {
		return ""
	}
	d := g.self

	selectBindings := make([]any, 0, len(q.Bindings["select"])+len(q.Bindings["order"]))
	selectBindings = append(selectBindings, q.Bindings["select"]...)
	selectBindings = append(selectBindings, q.Bindings["order"]...)
	q.SetBindings(selectBindings, "select")
	q.SetBindings(nil, "order")

	groupLimit := q.GetGroupLimit()
	limit := groupLimit.Value

	offset := q.GetOffset()
	if offset != nil {
		limit += *offset
		q = q.CloneWithout("offset")
	}

	components := g.compileComponents(q)
	orders := take(&components, "orders")
	setComponent(components, "columns", get(components, "columns")+g.compileRowNumber(groupLimit.Column, orders))

	table := d.Wrap("laravel_table")
	row := d.Wrap("laravel_row")

	sql := "select * from (" + concatenate(components) + ") as " + table +
		" where " + row + " <= " + strconv.Itoa(limit)

	if offset != nil {
		sql += " and " + row + " > " + strconv.Itoa(*offset)
	}

	return sql + " order by " + row
}

// compileRowNumber writes the row-number window a group limit is built on.
//
// The aliases never leave the compiled statement: Get strips them out of every
// row.
func (g *Grammar) compileRowNumber(partition string, orders string) string {
	over := strings.TrimSpace("partition by " + g.self.Wrap(partition) + " " + orders)
	return ", row_number() over (" + over + ") as " + g.self.Wrap("laravel_row")
}

// compileUnions builds the SQL for the query's union clauses, including
// their orders, limit and offset.
func (g *Grammar) compileUnions(q *query.Builder) string {
	d := g.self
	var sql strings.Builder

	for _, union := range q.Unions {
		sql.WriteString(g.compileUnion(union))
	}
	if len(q.UnionOrders) > 0 {
		sql.WriteString(" " + d.CompileOrders(q, q.UnionOrders))
	}
	if q.UnionLimit != nil {
		sql.WriteString(" " + d.CompileLimit(q, *q.UnionLimit))
	}
	if q.UnionOffset != nil {
		sql.WriteString(" " + d.CompileOffset(q, *q.UnionOffset))
	}

	return strings.TrimLeft(sql.String(), " ")
}

// compileUnion builds the SQL for a single union clause.
func (g *Grammar) compileUnion(union query.Union) string {
	conjunction := " union "
	if union.All {
		conjunction = " union all "
	}
	return conjunction + g.self.WrapUnion(g.self.CompileSelect(union.Query))
}

// WrapUnion parenthesises a compiled union member's SQL.
func (g *Grammar) WrapUnion(sql string) string { return "(" + sql + ")" }

// compileUnionAggregate builds the SQL for an aggregate computed over a
// union of selects.
//
// The inner select runs against a clone with its aggregate cleared, so it
// does not compile the aggregate a second time -- reaching the same
// statement without mutating the caller's builder.
func (g *Grammar) compileUnionAggregate(q *query.Builder) string {
	d := g.self
	sql := d.CompileAggregate(q, q.GetAggregate())
	return sql + " from (" + d.CompileSelect(q.CloneWithout("aggregate")) + ") as " + d.WrapTable("temp_table")
}

// CompileExists builds a select that reports whether the query would
// return any rows.
func (g *Grammar) CompileExists(q *query.Builder) string {
	if g.compilationError(q) != nil {
		return ""
	}
	return "select exists(" + g.self.CompileSelect(q) + ") as " + g.self.Wrap("exists")
}

// CompileInsert builds the SQL for an insert statement.
//
// # Column order
//
// The column list comes from the first record's keys. A Go map has no
// defined order, so the columns are sorted by name: the statement has to be
// the same on every run, and the bindings the builder flattens have to line
// up with it. Every method here that walks a values map sorts it the same
// way.
func (g *Grammar) CompileInsert(q *query.Builder, values []map[string]any) string {
	if g.compilationError(q) != nil {
		return ""
	}
	d := g.self
	table := d.WrapTable(q.GetFrom())

	if len(values) == 0 {
		return "insert into " + table + " default values"
	}

	columns := sortedKeys(values[0])

	parameters := make([]string, 0, len(values))
	for _, record := range values {
		ordered := make([]any, 0, len(columns))
		for _, column := range columns {
			ordered = append(ordered, record[column])
		}
		parameters = append(parameters, "("+d.Parameterize(ordered)+")")
	}

	return "insert into " + table + " (" + d.Columnize(toAny(columns)) + ") values " + strings.Join(parameters, ", ")
}

// CompileInsertGetID builds the SQL for an insert whose generated id the
// caller wants back.
func (g *Grammar) CompileInsertGetID(q *query.Builder, values map[string]any, sequence string) string {
	if g.compilationError(q) != nil {
		return ""
	}
	return g.self.CompileInsert(q, []map[string]any{values})
}

// CompileInsertUsing builds the SQL for an insert whose values come from a
// select statement rather than literal rows.
func (g *Grammar) CompileInsertUsing(q *query.Builder, columns []any, sql string) string {
	if g.compilationError(q) != nil {
		return ""
	}
	d := g.self
	table := d.WrapTable(q.GetFrom())

	if len(columns) == 0 || (len(columns) == 1 && text(columns[0]) == "*") {
		return "insert into " + table + " " + sql
	}

	return "insert into " + table + " (" + d.Columnize(columns) + ") " + sql
}

// CompileInsertOrIgnoreReturning builds the SQL for an insert that ignores
// conflicting rows and returns the given columns for the rows it did
// insert. The base implementation returns an error: only a driver that
// supports it overrides it.
func (g *Grammar) CompileInsertOrIgnoreReturning(q *query.Builder, values []map[string]any, uniqueBy, returning []string) (string, error) {
	if err := g.compilationError(q); err != nil {
		return "", err
	}
	return "", fmt.Errorf("query/grammars: this database engine does not support insert or ignore with returning")
}

// CompileInsertOrIgnoreUsing builds the SQL for an insert from a select
// statement that ignores conflicting rows. The base implementation returns
// an error: only a driver that supports it overrides it.
func (g *Grammar) CompileInsertOrIgnoreUsing(q *query.Builder, columns []any, sql string) (string, error) {
	if err := g.compilationError(q); err != nil {
		return "", err
	}
	return "", fmt.Errorf("query/grammars: this database engine does not support inserting while ignoring errors")
}

// CompileUpdate builds the SQL for an update statement.
func (g *Grammar) CompileUpdate(q *query.Builder, values map[string]any) string {
	if g.compilationError(q) != nil {
		return ""
	}
	d := g.self
	table := d.WrapTable(q.GetFrom())
	columns := d.CompileUpdateColumns(q, values)
	where := d.CompileWheres(q)

	if len(q.Joins) > 0 {
		return strings.TrimSpace(d.CompileUpdateWithJoins(q, table, columns, where))
	}
	return strings.TrimSpace(d.CompileUpdateWithoutJoins(q, table, columns, where))
}

// CompileUpdateColumns builds the SQL set list for an update statement.
func (g *Grammar) CompileUpdateColumns(q *query.Builder, values map[string]any) string {
	d := g.self
	parts := make([]string, 0, len(values))
	for _, column := range sortedKeys(values) {
		parts = append(parts, d.Wrap(column)+" = "+d.Parameter(values[column]))
	}
	return strings.Join(parts, ", ")
}

// CompileUpdateWithoutJoins builds the SQL for an update with no joins.
func (g *Grammar) CompileUpdateWithoutJoins(q *query.Builder, table, columns, where string) string {
	if g.compilationError(q) != nil {
		return ""
	}
	return "update " + table + " set " + columns + " " + where
}

// CompileUpdateWithJoins builds the SQL for an update that joins other
// tables.
func (g *Grammar) CompileUpdateWithJoins(q *query.Builder, table, columns, where string) string {
	if g.compilationError(q) != nil {
		return ""
	}
	joins := g.self.CompileJoins(q, q.Joins)
	return "update " + table + " " + joins + " set " + columns + " " + where
}

// PrepareBindingsForUpdate orders the bindings for an update statement to
// match the SQL CompileUpdate produced.
//
// The join bindings come first because the joins are compiled before the set
// list, then the values, then everything else in bindingOrder. Select
// bindings are dropped: an update has no select clause to bind them to.
//
// An Expression stays in the list as itself. CompileUpdateColumns wrote it
// into the statement rather than leaving a placeholder for it, so it must
// not be bound -- and query.Builder's CleanBindings is what drops it later,
// by recognising the type. Unwrapping it here would hide it from that check
// and shift every binding after it by one.
func (g *Grammar) PrepareBindingsForUpdate(bindings map[string][]any, values map[string]any) []any {
	out := make([]any, 0, len(values)+8)
	out = append(out, bindings["join"]...)

	for _, column := range sortedKeys(values) {
		out = append(out, values[column])
	}

	for _, key := range bindingOrder {
		if key == "select" || key == "join" {
			continue
		}
		out = append(out, bindings[key]...)
	}

	return out
}

// CompileDelete builds the SQL for a delete statement.
func (g *Grammar) CompileDelete(q *query.Builder) string {
	if g.compilationError(q) != nil {
		return ""
	}
	d := g.self
	table := d.WrapTable(q.GetFrom())
	where := d.CompileWheres(q)

	if len(q.Joins) > 0 {
		return strings.TrimSpace(d.CompileDeleteWithJoins(q, table, where))
	}
	return strings.TrimSpace(d.CompileDeleteWithoutJoins(q, table, where))
}

// CompileDeleteWithoutJoins builds the SQL for a delete with no joins.
func (g *Grammar) CompileDeleteWithoutJoins(q *query.Builder, table, where string) string {
	if g.compilationError(q) != nil {
		return ""
	}
	return "delete from " + table + " " + where
}

// CompileDeleteWithJoins builds the SQL for a delete that joins other
// tables.
func (g *Grammar) CompileDeleteWithJoins(q *query.Builder, table, where string) string {
	if g.compilationError(q) != nil {
		return ""
	}
	alias := lastSegment(table, " as ")
	joins := g.self.CompileJoins(q, q.Joins)
	return "delete " + alias + " from " + table + " " + joins + " " + where
}

// PrepareBindingsForDelete orders the bindings for a delete statement,
// dropping the select bindings a delete has no clause to bind.
func (g *Grammar) PrepareBindingsForDelete(bindings map[string][]any) []any {
	out := make([]any, 0, 8)
	for _, key := range bindingOrder {
		if key == "select" {
			continue
		}
		out = append(out, bindings[key]...)
	}
	return out
}

// CompileTruncate builds the SQL that empties a table.
func (g *Grammar) CompileTruncate(q *query.Builder) map[string][]any {
	if g.compilationError(q) != nil {
		return nil
	}
	return map[string][]any{"truncate table " + g.self.WrapTable(q.GetFrom()): {}}
}

// CompileLock builds the SQL locking clause, if the given value is a
// string.
func (g *Grammar) CompileLock(q *query.Builder, value any) string {
	if lock, ok := value.(string); ok {
		return lock
	}
	return ""
}

// CompileThreadCount builds the SQL that asks the engine how many threads
// or connections are active. The empty string means the engine cannot be
// asked.
func (g *Grammar) CompileThreadCount() string { return "" }

// SubstituteBindingsIntoRawSQL writes a statement's bindings directly into
// its placeholders, for display.
//
// It is for showing a statement, never for running one: the result is a
// string with the values written into it, which is exactly what a
// placeholder exists to avoid. It returns an error because escaping needs a
// connection, and a grammar without one refuses rather than emitting
// something that looks escaped and is not.
func (g *Grammar) SubstituteBindingsIntoRawSQL(sql string, bindings []any) (string, error) {
	d := g.self

	escaped := make([]string, 0, len(bindings))
	for _, binding := range bindings {
		_, binary := binding.([]byte)
		value, err := d.Escape(binding, binary)
		if err != nil {
			return "", err
		}
		escaped = append(escaped, value)
	}

	var out strings.Builder
	isStringLiteral := false

	for i := 0; i < len(sql); i++ {
		char := sql[i]
		next := byte(0)
		if i+1 < len(sql) {
			next = sql[i+1]
		}

		pair := string([]byte{char, next})
		switch {
		case next != 0 && (pair == `\'` || pair == "''" || pair == "??"):
			// An escaped quote, or a Postgres operator whose question mark was
			// doubled so it would not read as a placeholder.
			out.WriteString(pair)
			i++
		case char == '\'':
			out.WriteByte(char)
			isStringLiteral = !isStringLiteral
		case char == '?' && !isStringLiteral:
			if len(escaped) > 0 {
				out.WriteString(escaped[0])
				escaped = escaped[1:]
			} else {
				out.WriteByte('?')
			}
		default:
			out.WriteByte(char)
		}
	}

	return out.String(), nil
}

// Wrap quotes an identifier, leaving an Expression alone.
//
// query.BaseGrammar has a Wrap of its own, and it cannot be the one that runs:
// it quotes through an unexported wrapValue that Go binds at compile time, so
// MySQL's backtick could never reach it from this package. The wrapping family
// is therefore declared here over the exported WrapValue hook, and BaseGrammar
// stays embedded for everything that is not identifier quoting.
func (g *Grammar) Wrap(value any) string {
	d := g.self

	if query.IsExpression(value) {
		return text(d.GetValue(value))
	}

	name := text(value)

	if segments := aliasSplit(name); len(segments) > 1 {
		return d.Wrap(segments[0]) + " as " + d.WrapValue(segments[1])
	}

	if isJSONSelector(name) {
		return d.WrapJSONSelector(name)
	}

	return g.wrapSegments(strings.Split(name, "."))
}

// WrapTable quotes a table name, applying the table prefix.
//
// The table prefix is applied here and nowhere else, which is why a bare
// table name written into a where clause is a bug that only appears on a
// prefixed connection.
func (g *Grammar) WrapTable(table any) string {
	return g.wrapTableWithPrefix(table, g.self.GetTablePrefix())
}

// wrapTableWithPrefix is wrapTable's second parameter, which Go cannot spell as
// an optional argument without changing the signature query.Grammar declares.
func (g *Grammar) wrapTableWithPrefix(table any, prefix string) string {
	d := g.self

	if query.IsExpression(table) {
		return text(d.GetValue(table))
	}

	name := text(table)

	if segments := aliasSplit(name); len(segments) > 1 {
		return g.wrapTableWithPrefix(segments[0], prefix) + " as " + d.WrapValue(prefix+segments[1])
	}

	// A schema qualified name is prefixed on the table, not on the schema.
	if i := strings.LastIndex(name, "."); i >= 0 {
		name = name[:i+1] + prefix + name[i+1:]
		segments := strings.Split(name, ".")
		wrapped := make([]string, 0, len(segments))
		for _, segment := range segments {
			wrapped = append(wrapped, d.WrapValue(segment))
		}
		return strings.Join(wrapped, ".")
	}

	return d.WrapValue(prefix + name)
}

// wrapSegments quotes each dot-separated segment of a name. The first of
// several segments is a table, so it takes the prefix.
func (g *Grammar) wrapSegments(segments []string) string {
	d := g.self
	out := make([]string, 0, len(segments))
	for i, segment := range segments {
		if i == 0 && len(segments) > 1 {
			out = append(out, d.WrapTable(segment))
			continue
		}
		out = append(out, d.WrapValue(segment))
	}
	return strings.Join(out, ".")
}

// WrapValue quotes one identifier segment in standard double quotes. MySQL
// overrides it with a backtick.
//
// A quote inside the identifier is doubled rather than dropped, because
// dropping it would silently rename the column.
func (g *Grammar) WrapValue(value string) string {
	if value == "*" {
		return value
	}
	return `"` + strings.ReplaceAll(value, `"`, `""`) + `"`
}

// WrapArray quotes every value in the given slice.
func (g *Grammar) WrapArray(values []any) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		out = append(out, g.self.Wrap(value))
	}
	return out
}

// Columnize quotes and joins a list of columns with commas.
func (g *Grammar) Columnize(columns []any) string {
	return strings.Join(g.self.WrapArray(columns), ", ")
}

// Parameterize builds a comma-separated list of placeholders, one per
// value.
func (g *Grammar) Parameterize(values []any) string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		out = append(out, g.self.Parameter(value))
	}
	return strings.Join(out, ", ")
}

// Parameter returns a placeholder for the given value, or its literal SQL
// if it is an Expression.
//
// # Why every dialect writes "?", Postgres included
//
// Postgres numbers its placeholders, and this grammar still emits "?" for it.
// The numbering happens once, at the connection, in database.Dialect.Rebind,
// which walks the finished statement and rewrites each "?" to $1, $2 and so on
// while leaving the ones inside string literals and comments alone. Numbering
// here instead would mean the grammar had to know a fragment's position in a
// statement it has not finished building yet -- a subquery compiled on its own
// would start over at $1 -- and it would give the project two placeholder
// conventions where one will do.
func (g *Grammar) Parameter(value any) string {
	if query.IsExpression(value) {
		return text(g.self.GetValue(value))
	}
	return "?"
}

// GetValue unwraps an expression to the value it carries; anything else is
// returned unchanged.
func (g *Grammar) GetValue(expression any) any {
	switch e := expression.(type) {
	case query.Expression:
		return g.GetValue(e.Value())
	case *query.Expression:
		if e == nil {
			return nil
		}
		return g.GetValue(e.Value())
	}
	return expression
}

// IsExpression reports whether value is a query.Expression.
func (g *Grammar) IsExpression(value any) bool { return query.IsExpression(value) }

// SetTablePrefix sets the table prefix and returns the outermost grammar.
//
// query.BaseGrammar's own version returns nil, because it cannot name the
// grammar that embedded it; this returns g.self instead, the outermost
// grammar wired in by NewGrammar.
func (g *Grammar) SetTablePrefix(prefix string) query.Grammar {
	g.BaseGrammar.SetTablePrefix(prefix)
	if grammar, ok := g.self.(query.Grammar); ok {
		return grammar
	}
	return nil
}

// concatenate joins the pieces of a statement with spaces, dropping the
// empty ones.
func concatenate(components []component) string {
	parts := make([]string, 0, len(components))
	for _, c := range components {
		if c.sql != "" {
			parts = append(parts, c.sql)
		}
	}
	return strings.Join(parts, " ")
}

func get(components []component, name string) string {
	for _, c := range components {
		if c.name == name {
			return c.sql
		}
	}
	return ""
}

func setComponent(components []component, name, sql string) {
	for i := range components {
		if components[i].name == name {
			components[i].sql = sql
			return
		}
	}
}

// take removes a component and returns its SQL, which is what `unset` does to
// the orders before a group limit folds them into the window function.
func take(components *[]component, name string) string {
	for i, c := range *components {
		if c.name == name {
			*components = append((*components)[:i], (*components)[i+1:]...)
			return c.sql
		}
	}
	return ""
}

// leadingBoolean matches the "and " or "or " the builder writes in front of
// every clause for the compilers' convenience.
var leadingBoolean = regexp.MustCompile(`(?i)^(and |or )`)

// removeLeadingBoolean strips the leading "and " or "or " a compiled clause
// list starts with.
//
// The pattern is anchored to the start of the string: every clause list
// starts with the boolean, and anchoring it rules out eating a conjunction
// out of the middle of a raw clause.
func removeLeadingBoolean(value string) string {
	return leadingBoolean.ReplaceAllString(value, "")
}

// aliasPattern matches the " as " separator between a column or table and
// its alias, case-insensitively.
var aliasPattern = regexp.MustCompile(`(?i)\s+as\s+`)

// aliasSplit splits "column as alias" the way preg_split does.
func aliasSplit(value string) []string {
	return aliasPattern.Split(value, 2)
}

// lastSegment returns the part of value after the last occurrence of
// separator, or value unchanged if separator does not appear.
func lastSegment(value, separator string) string {
	if i := strings.LastIndex(strings.ToLower(value), separator); i >= 0 {
		return value[i+len(separator):]
	}
	return value
}

// unsupportedClause turns a where-compiler's refusal into a false clause.
//
// CompileSelect returns a string, because that is what query.Grammar
// declares and what a builder can use, so the refusal cannot travel as an
// error from here. It travels as a clause that is false instead: the query
// returns nothing rather than returning rows nobody filtered for, and the
// reason rides along in a comment so the log says what happened. Every entry
// point that can return an error does.
func unsupportedClause(err error) string {
	return "1 = 0 /* " + strings.ReplaceAll(err.Error(), "*/", "* /") + " */"
}

// unsupportedJoin is a join the grammar cannot spell.
//
// A where clause that cannot be compiled becomes a false one, because a filter
// nobody can spell must not widen the result. A join cannot take that route:
// dropping it changes which rows come back in a direction that depends on the
// join type, and an inner join dropped from a query is rows that were never
// meant to be there. So the fragment is deliberately not SQL, the engine
// refuses the statement, and the reason travels in the message.
func unsupportedJoin(err error) string {
	return "/* " + strings.ReplaceAll(err.Error(), "*/", "* /") + " */ unsupported join"
}

// sortedKeys is the column order of a values map. See CompileInsert.
func sortedKeys[T any](values map[string]T) []string {
	keys := make([]string, 0, len(values))
	for key := range values {
		keys = append(keys, key)
	}
	slices.Sort(keys)
	return keys
}

func toAny(values []string) []any {
	out := make([]any, 0, len(values))
	for _, value := range values {
		out = append(out, value)
	}
	return out
}

// ends returns the first and last values of a between clause's bounds.
func ends(values []any) (first, last any) {
	if len(values) == 0 {
		return nil, nil
	}
	return values[0], values[len(values)-1]
}

// joinValues writes values into the statement, for the raw "in" clauses whose
// caller has already made every value an integer.
func joinValues(values []any) string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		out = append(out, text(value))
	}
	return strings.Join(out, ", ")
}
