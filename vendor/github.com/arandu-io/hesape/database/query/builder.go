package query

import (
	"context"
	"errors"
	"strings"
)

// Builder is the fluent query builder.
//
// It builds SQL and does not decide who may run it. Authorization lives one
// layer up, in the repository that holds a security.Grant and filters by the
// tenant that Grant carries -- on reads exactly as on writes. A Builder reached
// without going through that layer is SQL nobody authorized, which is why
// nothing outside a repository should be constructing one.
//
// # Which fields are exported, and which are read through a getter
//
// The grammar reads the builder's state directly, so the state is exported.
// Nine of them cannot be: from, limit, offset, distinct, lock, aggregate,
// timeout, groupLimit and indexHint are all a field AND a fluent method, and Go
// holds both in one namespace. Those are unexported and reachable through
// GetFrom, GetLimit, GetOffset, GetColumns, GetDistinct, GetLock, GetAggregate,
// GetTimeout, GetGroupLimit and GetIndexHint.
type Builder struct {
	// Connection is the thing the query runs its statements against.
	connection Connection

	// Grammar is what compiles the query to SQL for a specific engine.
	Grammar Grammar

	// Processor is the hook that adjusts results as they come back from the
	// engine.
	Processor Processor

	// Bindings holds the query's parameter values, keyed by the seven
	// segments of a statement they belong to. The order of the keys is the
	// order they are concatenated in, which is why GetBindings walks
	// bindingOrder rather than ranging the map.
	Bindings map[string][]any

	// Columns is the select list. Nil means "not selected yet", which is
	// what lets Get default to * while an explicit Select of nothing selects
	// nothing.
	Columns []any

	// Wheres is the list of where clauses.
	Wheres []Where

	// Joins is the list of join clauses.
	Joins []*JoinClause

	// Groups is the GROUP BY column list.
	Groups []any

	// Havings is the list of having clauses.
	Havings []Having

	// Orders is the list of order-by clauses.
	Orders []Order

	// Unions is the list of queries unioned onto this one.
	Unions []Union

	// UnionOrders is the order-by list that applies to the union as a whole.
	UnionOrders []Order

	// UnionLimit is the limit that applies to the union as a whole.
	UnionLimit *int

	// UnionOffset is the offset that applies to the union as a whole.
	UnionOffset *int

	// BeforeQueryCallbacks run immediately before the query is compiled.
	BeforeQueryCallbacks []func(*Builder)

	// AfterQueryCallbacks each run over the result of Get, Pluck or Cursor
	// before it is handed back.
	AfterQueryCallbacks []func([]Record) []Record

	// subqueries records the queries compiled into a from, a select or a join,
	// so that scopeSubqueryClauses can put the tenant on each of them when the
	// statement runs. See pendingSub.
	subqueries []pendingSub

	from        any
	distinct    any
	indexHint   *IndexHint
	limit       *int
	groupLimit  *GroupLimit
	offset      *int
	lock        any
	aggregate   *Aggregate
	timeout     *int
	useWritePDO bool
	err         error
}

// Connection is what a query builder asks of the thing that runs its
// statements.
//
// It is declared here rather than imported from the database package because
// the interface belongs with its consumer in Go, and because database imports
// this package: naming it there would close the cycle.
// Every method takes a context, and it is the request's -- not one the
// connection was built with. A statement that cannot be cancelled outlives the
// request that asked for it, and a deadline that stops at the handler is a
// deadline the database never hears about.
type Connection interface {
	// Select runs a select and returns its rows.
	Select(ctx context.Context, query string, bindings []any, useReadPDO bool) ([]Record, error)

	// Insert runs an insert.
	Insert(ctx context.Context, query string, bindings []any) (bool, error)

	// Update runs an update and returns the number of rows affected.
	Update(ctx context.Context, query string, bindings []any) (int64, error)

	// Delete runs a delete and returns the number of rows affected.
	Delete(ctx context.Context, query string, bindings []any) (int64, error)

	// Statement runs a statement that returns neither rows nor a count.
	Statement(ctx context.Context, query string, bindings []any) (bool, error)
}

// Processor is the hook that lets a driver adjust results on the way out.
type Processor interface {
	// ProcessSelect adjusts the rows a select returned before they reach the
	// caller.
	ProcessSelect(query *Builder, results []Record) []Record

	// ProcessInsertGetID runs an insert and reports the ID of the inserted
	// row, read back from the named sequence.
	ProcessInsertGetID(ctx context.Context, query *Builder, sql string, values []any, sequence string) (int64, error)
}

// bindingOrder is the key order of Bindings. The list is the order the
// fragments appear in the compiled statement, and GetBindings depends on it:
// a binding read out of order is a value landing in the wrong placeholder,
// which is a wrong answer rather than an error.
var bindingOrder = []string{"select", "from", "join", "where", "groupBy", "having", "order", "union", "unionOrder"}

// NewBuilder creates a Builder for the given connection, grammar and
// processor, with an empty binding slice for each of the seven segments.
func NewBuilder(connection Connection, grammar Grammar, processor Processor) *Builder {
	b := &Builder{
		connection: connection,
		Grammar:    grammar,
		Processor:  processor,
		Bindings:   make(map[string][]any, len(bindingOrder)),
	}
	for _, key := range bindingOrder {
		b.Bindings[key] = nil
	}
	return b
}

// Select sets the columns to select, replacing any previous selection.
// AddSelect is how a column is appended instead.
func (b *Builder) Select(columns ...any) *Builder {
	b.Columns = nil
	b.Bindings["select"] = nil
	b.forgetSubqueriesOfKind("select")
	if len(columns) == 0 {
		columns = []any{"*"}
	}
	b.Columns = append(b.Columns, columns...)
	return b
}

// SelectRaw adds a raw expression to the select list, with its own bindings.
func (b *Builder) SelectRaw(expression string, bindings ...any) *Builder {
	b.AddSelect(Raw(expression))
	if len(bindings) > 0 {
		b.AddBinding(bindings, "select")
	}
	return b
}

// AddSelect appends columns to the select list.
func (b *Builder) AddSelect(columns ...any) *Builder {
	b.Columns = append(b.Columns, columns...)
	return b
}

// Distinct marks the query as distinct.
//
// With no argument it is a plain DISTINCT; with columns it is the engine's
// DISTINCT ON, which only Postgres compiles.
func (b *Builder) Distinct(columns ...any) *Builder {
	if len(columns) > 0 {
		b.distinct = columns
	} else {
		b.distinct = true
	}
	return b
}

// From sets the table the query reads from. The table may be a string or an
// Expression.
func (b *Builder) From(table any, as ...string) *Builder {
	b.forgetSubqueriesOfKind("from")
	if len(as) > 0 && as[0] != "" {
		b.from = stringify(table) + " as " + as[0]
		return b
	}
	b.from = table
	return b
}

// FromRaw sets the table to a raw expression, with its own bindings.
func (b *Builder) FromRaw(expression string, bindings ...any) *Builder {
	b.forgetSubqueriesOfKind("from")
	b.from = Raw(expression)
	if len(bindings) > 0 {
		b.AddBinding(bindings, "from")
	}
	return b
}

// Where adds a basic where clause, in either its two or three argument form.
//
// Called with two arguments the operator is "=": where('id', 1) and
// where('id', '=', 1) are the same clause. Called with a func it opens a
// nested group instead.
func (b *Builder) Where(column any, args ...any) *Builder {
	return b.addWhere("and", column, args...)
}

// OrWhere adds an "or" basic where clause.
func (b *Builder) OrWhere(column any, args ...any) *Builder {
	return b.addWhere("or", column, args...)
}

func (b *Builder) addWhere(boolean string, column any, args ...any) *Builder {
	if nested, ok := column.(func(*Builder)); ok {
		return b.WhereNested(nested, boolean)
	}

	operator, value := prepareValueAndOperator(args...)
	operator = b.acceptOperator(operator)
	if b.Err() != nil {
		return b
	}
	b.Wheres = append(b.Wheres, Where{
		Type:     "Basic",
		Column:   column,
		Operator: operator,
		Value:    value,
		Boolean:  boolean,
	})
	if !IsExpression(value) {
		b.AddBinding([]any{value}, "where")
	}
	return b
}

// prepareValueAndOperator resolves the operator and value from a variadic
// argument list: zero arguments is "=" against nil, one argument is a value
// compared with "=", and two or more are an operator and a value.
//
// Unlike the exported PrepareValueAndOperator, this does not refuse an
// operator with no value: three arguments including a nil value pass through
// as given, which compiles to a comparison against NULL and is visible in the
// SQL rather than hidden in an error.
func prepareValueAndOperator(args ...any) (operator string, value any) {
	switch len(args) {
	case 0:
		return "=", nil
	case 1:
		return "=", args[0]
	default:
		return stringify(args[0]), args[1]
	}
}

// WhereNested opens a nested group of clauses, built by the callback, and
// adds it under boolean.
func (b *Builder) WhereNested(callback func(*Builder), boolean string) *Builder {
	nested := b.ForNestedWhere()
	callback(nested)
	return b.AddNestedWhereQuery(nested, boolean)
}

// ForNestedWhere returns a new query sharing this one's table, for building
// a nested group of clauses.
func (b *Builder) ForNestedWhere() *Builder {
	nested := b.NewQuery()
	nested.from = b.from
	return nested
}

// AddNestedWhereQuery adds query as a parenthesised group of clauses under
// boolean.
//
// A group with no clauses is dropped rather than compiled, because "()" is a
// syntax error on every engine and an empty callback is an ordinary outcome of
// a conditional filter.
func (b *Builder) AddNestedWhereQuery(query *Builder, boolean string) *Builder {
	if query != nil && query.Err() != nil {
		b.setError(query.Err())
	}
	if query == nil {
		return b
	}
	if len(query.Wheres) == 0 {
		return b
	}
	if boolean == "" {
		boolean = "and"
	}
	b.Wheres = append(b.Wheres, Where{Type: "Nested", Query: query, Boolean: boolean})
	b.AddBinding(query.Bindings["where"], "where")
	return b
}

// WhereColumn adds a clause comparing two columns.
func (b *Builder) WhereColumn(first any, args ...any) *Builder {
	operator, second := prepareValueAndOperator(args...)
	operator = b.acceptOperator(operator)
	if b.Err() != nil {
		return b
	}
	b.Wheres = append(b.Wheres, Where{
		Type:     "Column",
		First:    first,
		Operator: operator,
		Second:   second,
		Boolean:  "and",
	})
	return b
}

// WhereRaw adds a raw SQL where clause, with its own bindings.
//
// The bindings are recorded on the clause as well as appended to the segment,
// because the tenant filter needs to rebuild the flat list -- scoping a
// subquery changes how many bindings it contributes, and the list can only be
// put back in order if every clause knows which of them are its own.
func (b *Builder) WhereRaw(sql string, bindings ...any) *Builder {
	b.Wheres = append(b.Wheres, Where{Type: "Raw", SQL: sql, Boolean: "and", Values: bindings})
	if len(bindings) > 0 {
		b.AddBinding(bindings, "where")
	}
	return b
}

// OrWhereRaw adds an "or" raw SQL where clause.
func (b *Builder) OrWhereRaw(sql string, bindings ...any) *Builder {
	b.Wheres = append(b.Wheres, Where{Type: "Raw", SQL: sql, Boolean: "or", Values: bindings})
	if len(bindings) > 0 {
		b.AddBinding(bindings, "where")
	}
	return b
}

// WhereBindings rebuilds the where segment from the clauses, in the order the
// compiled statement consumes them.
//
// It exists so that a clause carrying a subquery can be replaced -- which is
// what putting the tenant on a subquery does -- without the values afterwards
// sliding onto the wrong placeholders. Every clause type that contributes a
// binding is listed here, and a type that starts contributing one has to be
// added: a clause this does not know about loses its values silently, which is
// a wrong answer rather than an error.
func (b *Builder) WhereBindings() []any {
	out := make([]any, 0, len(b.Bindings["where"]))
	for _, where := range b.Wheres {
		switch where.Type {
		case "Basic":
			// A Basic clause carries a subquery only when WhereSubCount built
			// it, and then the subquery is on the left of the comparison: its
			// bindings come first because that is the order the clause
			// compiles in. Reading them off the clause is what lets this list be
			// rebuilt after scopeSubqueries has put the tenant on the subquery.
			if where.Query != nil {
				out = append(out, where.Query.GetBindings()...)
			}
			if !IsExpression(where.Value) {
				out = append(out, where.Value)
			}
		case "In", "Between":
			// An In carries either a list of values or a subquery, never both.
			// The subquery form is the one that loses its bindings silently if
			// this branch only ever looks at Values.
			if where.Query != nil {
				out = append(out, where.Query.GetBindings()...)
				continue
			}
			for _, value := range where.Values {
				if !IsExpression(value) {
					out = append(out, value)
				}
			}
		case "Raw":
			out = append(out, where.Values...)
		case "Like", "valueBetween", "Fulltext",
			"Date", "Time", "Day", "Month", "Year",
			"JsonContains", "JsonOverlaps", "JsonLength":
			// One value each, bound unless it compiled into the statement. The
			// clause carries the value that was bound and not the one that was
			// written, which is what makes this rebuildable -- see WhereLike.
			if !IsExpression(where.Value) {
				out = append(out, where.Value)
			}
		case "RowValues":
			for _, value := range where.Values {
				if !IsExpression(value) {
					out = append(out, value)
				}
			}
		case "Nested":
			if where.Query != nil {
				out = append(out, where.Query.WhereBindings()...)
			}
		case "Exists":
			if where.Query != nil {
				out = append(out, where.Query.GetBindings()...)
			}
		}
	}
	return out
}

// WhereIn adds a where-in clause.
//
// An empty list is kept rather than dropped: it compiles to "0 = 1", so a
// filter over an empty set returns nothing. Dropping it would return
// everything, which is the difference between an empty page and a data leak.
func (b *Builder) WhereIn(column any, values []any) *Builder {
	return b.addWhereIn(column, values, "and", false)
}

// OrWhereIn adds an "or" where-in clause.
func (b *Builder) OrWhereIn(column any, values []any) *Builder {
	return b.addWhereIn(column, values, "or", false)
}

// WhereNotIn adds a where-not-in clause.
func (b *Builder) WhereNotIn(column any, values []any) *Builder {
	return b.addWhereIn(column, values, "and", true)
}

// OrWhereNotIn adds an "or" where-not-in clause.
func (b *Builder) OrWhereNotIn(column any, values []any) *Builder {
	return b.addWhereIn(column, values, "or", true)
}

func (b *Builder) addWhereIn(column any, values []any, boolean string, not bool) *Builder {
	b.Wheres = append(b.Wheres, Where{
		Type: "In", Column: column, Values: values, Boolean: boolean, Not: not,
	})
	for _, value := range values {
		if !IsExpression(value) {
			b.AddBinding([]any{value}, "where")
		}
	}
	return b
}

// WhereNull adds a clause requiring the named columns to be NULL.
func (b *Builder) WhereNull(columns ...any) *Builder {
	return b.addWhereNull(columns, "and", false)
}

// OrWhereNull adds an "or" is-NULL clause.
func (b *Builder) OrWhereNull(columns ...any) *Builder {
	return b.addWhereNull(columns, "or", false)
}

// WhereNotNull adds a clause requiring the named columns not to be NULL.
func (b *Builder) WhereNotNull(columns ...any) *Builder {
	return b.addWhereNull(columns, "and", true)
}

// OrWhereNotNull adds an "or" is-not-NULL clause.
func (b *Builder) OrWhereNotNull(columns ...any) *Builder {
	return b.addWhereNull(columns, "or", true)
}

func (b *Builder) addWhereNull(columns []any, boolean string, not bool) *Builder {
	for _, column := range columns {
		b.Wheres = append(b.Wheres, Where{Type: "Null", Column: column, Boolean: boolean, Not: not})
	}
	return b
}

// WhereBetween adds a clause requiring column to fall between from and to.
func (b *Builder) WhereBetween(column any, from, to any) *Builder {
	return b.addWhereBetween(column, from, to, "and", false)
}

// OrWhereBetween adds an "or" between clause.
func (b *Builder) OrWhereBetween(column any, from, to any) *Builder {
	return b.addWhereBetween(column, from, to, "or", false)
}

// WhereNotBetween adds a not-between clause.
func (b *Builder) WhereNotBetween(column any, from, to any) *Builder {
	return b.addWhereBetween(column, from, to, "and", true)
}

// OrWhereNotBetween adds an "or" not-between clause.
func (b *Builder) OrWhereNotBetween(column any, from, to any) *Builder {
	return b.addWhereBetween(column, from, to, "or", true)
}

func (b *Builder) addWhereBetween(column, from, to any, boolean string, not bool) *Builder {
	b.Wheres = append(b.Wheres, Where{
		Type: "Between", Column: column, Values: []any{from, to}, Boolean: boolean, Not: not,
	})
	for _, value := range []any{from, to} {
		if !IsExpression(value) {
			b.AddBinding([]any{value}, "where")
		}
	}
	return b
}

// WhereExists adds an exists clause for the subquery the callback builds.
func (b *Builder) WhereExists(callback func(*Builder), boolean string, not bool) *Builder {
	query := b.NewQuery()
	callback(query)
	return b.AddWhereExistsQuery(query, boolean, not)
}

// AddWhereExistsQuery adds query as an exists (or not-exists) clause under
// boolean.
func (b *Builder) AddWhereExistsQuery(query *Builder, boolean string, not bool) *Builder {
	if boolean == "" {
		boolean = "and"
	}
	b.Wheres = append(b.Wheres, Where{Type: "Exists", Query: query, Boolean: boolean, Not: not})
	b.AddBinding(query.GetBindings(), "where")
	return b
}

// Join adds an inner join.
func (b *Builder) Join(table any, first any, args ...any) *Builder {
	return b.addJoin("inner", table, first, args...)
}

// LeftJoin adds a left join.
func (b *Builder) LeftJoin(table any, first any, args ...any) *Builder {
	return b.addJoin("left", table, first, args...)
}

// RightJoin adds a right join.
func (b *Builder) RightJoin(table any, first any, args ...any) *Builder {
	return b.addJoin("right", table, first, args...)
}

// CrossJoin adds a cross join. With no condition it is a bare cross join;
// with one it compiles as an inner join.
func (b *Builder) CrossJoin(table any, first ...any) *Builder {
	if len(first) == 0 {
		b.Joins = append(b.Joins, NewJoinClause(b, "cross", table))
		return b
	}
	return b.addJoin("cross", table, first[0], first[1:]...)
}

func (b *Builder) addJoin(typ string, table any, first any, args ...any) *Builder {
	return b.addJoinClause(typ, false, table, first, args...)
}

// GroupBy adds columns to the GROUP BY list.
func (b *Builder) GroupBy(groups ...any) *Builder {
	b.Groups = append(b.Groups, groups...)
	return b
}

// GroupByRaw adds a raw expression to the GROUP BY list, with its own
// bindings.
func (b *Builder) GroupByRaw(sql string, bindings ...any) *Builder {
	b.Groups = append(b.Groups, Raw(sql))
	if len(bindings) > 0 {
		b.AddBinding(bindings, "groupBy")
	}
	return b
}

// Having adds a having clause. The body is in havings.go, shared with
// OrHaving: the two differ by their conjunction and by nothing else.
func (b *Builder) Having(column any, args ...any) *Builder {
	return b.addHaving("and", column, args...)
}

// HavingRaw adds a raw having clause, with its own bindings.
func (b *Builder) HavingRaw(sql string, bindings ...any) *Builder {
	b.Havings = append(b.Havings, Having{Type: "Raw", SQL: sql, Boolean: "and"})
	if len(bindings) > 0 {
		b.AddBinding(bindings, "having")
	}
	return b
}

// OrderBy adds an order-by clause. A direction other than asc or desc is a
// caller writing SQL into a direction, so anything else reads as asc.
func (b *Builder) OrderBy(column any, direction ...string) *Builder {
	dir := "asc"
	if len(direction) > 0 && lower(direction[0]) == "desc" {
		dir = "desc"
	}
	order := Order{Column: column, Direction: dir}
	if len(b.Unions) > 0 {
		b.UnionOrders = append(b.UnionOrders, order)
	} else {
		b.Orders = append(b.Orders, order)
	}
	return b
}

// OrderByDesc adds a descending order-by clause.
func (b *Builder) OrderByDesc(column any) *Builder { return b.OrderBy(column, "desc") }

// Latest adds a descending order-by clause. The column defaults to
// created_at.
func (b *Builder) Latest(column ...any) *Builder {
	return b.OrderBy(defaultTimestampColumn(column), "desc")
}

// Oldest adds an ascending order-by clause. The column defaults to
// created_at.
func (b *Builder) Oldest(column ...any) *Builder {
	return b.OrderBy(defaultTimestampColumn(column), "asc")
}

func defaultTimestampColumn(column []any) any {
	if len(column) > 0 && column[0] != nil {
		return column[0]
	}
	return "created_at"
}

// OrderByRaw adds a raw order-by expression, with its own bindings.
func (b *Builder) OrderByRaw(sql string, bindings ...any) *Builder {
	order := Order{SQL: Raw(sql)}
	if len(b.Unions) > 0 {
		b.UnionOrders = append(b.UnionOrders, order)
		b.AddBinding(bindings, "unionOrder")
	} else {
		b.Orders = append(b.Orders, order)
		b.AddBinding(bindings, "order")
	}
	return b
}

// InRandomOrder orders the results randomly, optionally seeded.
func (b *Builder) InRandomOrder(seed ...string) *Builder {
	s := ""
	if len(seed) > 0 {
		s = seed[0]
	}
	return b.OrderByRaw(b.Grammar.CompileRandom(s))
}

// Reorder drops every order-by clause, and adds one back when given a
// column. Pass nil to drop the ordering and add nothing.
func (b *Builder) Reorder(column any, direction ...string) *Builder {
	b.Orders = nil
	b.UnionOrders = nil
	b.Bindings["order"] = nil
	b.Bindings["unionOrder"] = nil
	if column == nil {
		return b
	}
	return b.OrderBy(column, direction...)
}

// ReorderDesc drops every order-by clause and adds one back in descending
// order.
func (b *Builder) ReorderDesc(column any) *Builder { return b.Reorder(column, "desc") }

// Limit sets the row limit. A negative limit is dropped rather than applied.
func (b *Builder) Limit(value int) *Builder {
	if value < 0 {
		return b
	}
	if len(b.Unions) > 0 {
		b.UnionLimit = intPtr(value)
	} else {
		b.limit = intPtr(value)
	}
	return b
}

// Take is an alias of Limit.
func (b *Builder) Take(value int) *Builder { return b.Limit(value) }

// Offset sets the row offset. A negative offset reads as zero.
func (b *Builder) Offset(value int) *Builder {
	if value < 0 {
		value = 0
	}
	if len(b.Unions) > 0 {
		b.UnionOffset = intPtr(value)
	} else {
		b.offset = intPtr(value)
	}
	return b
}

// Skip is an alias of Offset.
func (b *Builder) Skip(value int) *Builder { return b.Offset(value) }

// ForPage sets the limit and offset for the given page of results.
//
// It is offset pagination, and it is the wrong tool past the first few pages:
// the engine counts and discards every row it skips, and a row inserted between
// two requests shifts the boundary so that a row is never shown. CursorPaginate
// in the pagination package names the boundary by value instead.
func (b *Builder) ForPage(page, perPage int) *Builder {
	if page < 1 {
		page = 1
	}
	return b.Offset((page - 1) * perPage).Limit(perPage)
}

// GroupLimit limits the number of rows per group of column, for a "top N per
// group" query.
func (b *Builder) GroupLimit(value int, column string) *Builder {
	if value >= 0 {
		b.groupLimit = &GroupLimit{Value: value, Column: column}
	}
	return b
}

// Union appends query as a union, or a union all when all is true.
func (b *Builder) Union(query *Builder, all ...bool) *Builder {
	isAll := len(all) > 0 && all[0]
	b.Unions = append(b.Unions, Union{Query: query, All: isAll})
	b.AddBinding(query.GetBindings(), "union")
	return b
}

// UnionAll appends query as a union all.
func (b *Builder) UnionAll(query *Builder) *Builder { return b.Union(query, true) }

// Lock sets the row-locking clause.
func (b *Builder) Lock(value any) *Builder {
	b.lock = value
	return b
}

// LockForUpdate locks the selected rows for update.
func (b *Builder) LockForUpdate() *Builder { return b.Lock(true) }

// SharedLock locks the selected rows with a shared lock.
func (b *Builder) SharedLock() *Builder { return b.Lock(false) }

// Timeout sets the statement timeout, in seconds.
func (b *Builder) Timeout(seconds int) *Builder {
	b.timeout = intPtr(seconds)
	return b
}

// UseIndex hints the engine to use the named index.
func (b *Builder) UseIndex(index string) *Builder {
	b.indexHint = NewIndexHint("hint", index)
	return b
}

// ForceIndex hints the engine to force the named index.
func (b *Builder) ForceIndex(index string) *Builder {
	b.indexHint = NewIndexHint("force", index)
	return b
}

// IgnoreIndex hints the engine to ignore the named index.
func (b *Builder) IgnoreIndex(index string) *Builder {
	b.indexHint = NewIndexHint("ignore", index)
	return b
}

// BeforeQuery registers a callback to run immediately before the query is
// compiled.
func (b *Builder) BeforeQuery(callback func(*Builder)) *Builder {
	b.BeforeQueryCallbacks = append(b.BeforeQueryCallbacks, callback)
	return b
}

// ApplyBeforeQueryCallbacks drains registered callbacks across the complete
// query graph, then validates every non-raw operator and order direction.
// Callbacks registered by callbacks are part of the same preparation and
// cannot run midway through SQL compilation. A callback cycle fails closed
// after a bounded number of rounds.
func (b *Builder) ApplyBeforeQueryCallbacks() {
	const maxRounds = 64
	for range maxRounds {
		b.applyBeforeQueryCallbacks(make(map[*Builder]bool))
		if !b.hasBeforeQueryCallbacks(make(map[*Builder]bool)) {
			b.ValidateForCompilation()
			return
		}
	}
	b.setError(errors.New("query: before-query callbacks did not settle"))
}

// ValidateForCompilation applies the final barrier to the complete query
// graph without executing callbacks: every non-raw operator and every order
// direction has to be one the grammar declares. Grammar compilers use it to
// reject direct public-field mutations while leaving callback lifecycle
// ownership to Builder execution methods.
func (b *Builder) ValidateForCompilation() error {
	if b == nil {
		return errors.New("query: cannot validate a nil builder")
	}
	return b.validateQueryGraph(make(map[*Builder]bool), b.Grammar, nil)
}

// ValidateForCompilationWith applies the same barrier under the operator
// policy of the grammar that is about to spell the SQL, which is not always
// the one the builder was given: a builder assembled for one dialect can be
// handed straight to another dialect's compiler, and the engine that receives
// the statement is the one that decides how each token reads.
//
// The grammar the builder carries stays in the conversation, but only for word
// operators, which is how a grammar extending a dialect keeps the operators it
// adds. A nil compiler leaves the operators every dialect shares.
func (b *Builder) ValidateForCompilationWith(compiler Grammar) error {
	if b == nil {
		return errors.New("query: cannot validate a nil builder")
	}
	return b.validateQueryGraph(make(map[*Builder]bool), compiler, b.Grammar)
}

// ToSQL runs the before-query callbacks and compiles the query to a select
// statement.
func (b *Builder) ToSQL() string {
	b.ApplyBeforeQueryCallbacks()
	if b.Err() != nil {
		return ""
	}
	if b.Grammar == nil {
		b.setError(errors.New("query: the builder has no grammar to compile with"))
		return ""
	}
	return b.Grammar.CompileSelect(b)
}

// Err returns the first final-validation error recorded by the builder or one
// of its child queries.
func (b *Builder) Err() error { return b.err }

func (b *Builder) setError(err error) {
	if err != nil && b.err == nil {
		b.err = err
	}
}

func (b *Builder) acceptOperator(operator string) string {
	canonical, err := normalizeOperator(b.Grammar, nil, operator)
	if err != nil {
		b.setError(err)
		return operator
	}
	return canonical
}

func (b *Builder) applyBeforeQueryCallbacks(visited map[*Builder]bool) {
	if b == nil {
		return
	}
	if visited[b] {
		return
	}
	visited[b] = true

	callbacks := b.BeforeQueryCallbacks
	b.BeforeQueryCallbacks = nil
	for _, callback := range callbacks {
		callback(b)
	}
	for i := range b.Wheres {
		b.Wheres[i].Query.applyBeforeQueryCallbacks(visited)
	}
	for i := range b.Havings {
		b.Havings[i].Query.applyBeforeQueryCallbacks(visited)
	}
	for _, join := range b.Joins {
		if join != nil {
			join.Builder.applyBeforeQueryCallbacks(visited)
		}
	}
	for _, union := range b.Unions {
		union.Query.applyBeforeQueryCallbacks(visited)
	}
	for _, sub := range b.subqueries {
		sub.query.applyBeforeQueryCallbacks(visited)
	}
}

func (b *Builder) hasBeforeQueryCallbacks(visited map[*Builder]bool) bool {
	if b == nil || visited[b] {
		return false
	}
	visited[b] = true
	if len(b.BeforeQueryCallbacks) > 0 {
		return true
	}
	for i := range b.Wheres {
		if b.Wheres[i].Query.hasBeforeQueryCallbacks(visited) {
			return true
		}
	}
	for i := range b.Havings {
		if b.Havings[i].Query.hasBeforeQueryCallbacks(visited) {
			return true
		}
	}
	for _, join := range b.Joins {
		if join != nil && join.Builder.hasBeforeQueryCallbacks(visited) {
			return true
		}
	}
	for _, union := range b.Unions {
		if union.Query.hasBeforeQueryCallbacks(visited) {
			return true
		}
	}
	for _, sub := range b.subqueries {
		if sub.query.hasBeforeQueryCallbacks(visited) {
			return true
		}
	}
	return false
}

// validateQueryGraph walks the whole query graph under one pair of policies.
//
// Both are fixed for the walk: one statement is spelled by one dialect,
// however many grammars the builders inside it were assembled with. declared
// is the grammar of the builder that was handed to the compiler, and is nil
// when the builder is being compiled by its own.
func (b *Builder) validateQueryGraph(visited map[*Builder]bool, compiler, declared Grammar) error {
	if b == nil {
		return nil
	}
	if visited[b] {
		return b.err
	}
	visited[b] = true

	if b.err != nil {
		return b.err
	}

	for i := range b.Wheres {
		where := &b.Wheres[i]
		if clauseOperatorNeedsValidation(where.Type, where.Operator) {
			canonical, err := normalizeOperator(compiler, declared, where.Operator)
			if err != nil {
				b.setError(err)
				return b.err
			}
			where.Operator = canonical
		}
		if err := b.validateChild(where.Query, visited, compiler, declared); err != nil {
			return err
		}
	}
	for i := range b.Havings {
		having := &b.Havings[i]
		if clauseOperatorNeedsValidation(having.Type, having.Operator) {
			canonical, err := normalizeOperator(compiler, declared, having.Operator)
			if err != nil {
				b.setError(err)
				return b.err
			}
			having.Operator = canonical
		}
		if err := b.validateChild(having.Query, visited, compiler, declared); err != nil {
			return err
		}
	}
	for i := range b.Orders {
		if err := normalizeOrderDirection(&b.Orders[i]); err != nil {
			b.setError(err)
			return b.err
		}
	}
	for i := range b.UnionOrders {
		if err := normalizeOrderDirection(&b.UnionOrders[i]); err != nil {
			b.setError(err)
			return b.err
		}
	}
	for _, join := range b.Joins {
		if join != nil {
			if err := b.validateChild(join.Builder, visited, compiler, declared); err != nil {
				return err
			}
		}
	}
	for _, union := range b.Unions {
		if err := b.validateChild(union.Query, visited, compiler, declared); err != nil {
			return err
		}
	}
	for _, sub := range b.subqueries {
		if err := b.validateChild(sub.query, visited, compiler, declared); err != nil {
			return err
		}
	}
	return b.err
}

func clauseOperatorNeedsValidation(typ, operator string) bool {
	if typ == "Raw" || typ == "Expression" {
		return false
	}
	if operator != "" {
		return true
	}
	switch typ {
	case "Basic", "Bitwise", "bit", "Column", "Date", "Time", "Day", "Month", "Year", "RowValues", "JsonLength", "Sub":
		return true
	default:
		return false
	}
}

func (b *Builder) validateChild(child *Builder, visited map[*Builder]bool, compiler, declared Grammar) error {
	if child == nil {
		return nil
	}
	if err := child.validateQueryGraph(visited, compiler, declared); err != nil {
		b.setError(err)
		return b.err
	}
	return nil
}

// NewQuery returns a new Builder sharing this one's connection, grammar and
// processor.
func (b *Builder) NewQuery() *Builder {
	return NewBuilder(b.connection, b.Grammar, b.Processor)
}

// AddBinding appends value to the named binding segment.
//
// An unknown binding type is dropped rather than appended to a key the
// grammar never reads. The seven names are in bindingOrder and are not open
// to extension.
func (b *Builder) AddBinding(value []any, typ string) *Builder {
	if typ == "" {
		typ = "where"
	}
	if _, ok := b.Bindings[typ]; !ok {
		return b
	}
	b.Bindings[typ] = append(b.Bindings[typ], value...)
	return b
}

// GetBindings returns every binding, flattened in the order the compiled
// statement consumes them.
func (b *Builder) GetBindings() []any {
	out := make([]any, 0)
	for _, key := range bindingOrder {
		out = append(out, b.Bindings[key]...)
	}
	return out
}

// GetRawBindings returns the bindings keyed by segment, unflattened.
func (b *Builder) GetRawBindings() map[string][]any { return b.Bindings }

// SetBindings replaces the bindings of the named segment.
func (b *Builder) SetBindings(bindings []any, typ string) *Builder {
	if typ == "" {
		typ = "where"
	}
	if _, ok := b.Bindings[typ]; !ok {
		return b
	}
	b.Bindings[typ] = bindings
	b.forgetSubqueriesOfSegment(typ)
	return b
}

// MergeBindings appends query's bindings onto this builder's, segment by
// segment.
func (b *Builder) MergeBindings(query *Builder) *Builder {
	for _, key := range bindingOrder {
		b.Bindings[key] = append(b.Bindings[key], query.Bindings[key]...)
	}
	return b
}

// UseWritePDO sends the read to the write connection, for the case a
// replica has not caught up yet: a row inserted and read back in the same
// request is not there on a replica that is two hundred milliseconds behind.
//
// The state behind it is unexported and read through UsingWritePDO, for the
// same reason From, Limit and Offset are: it is both a field and a fluent
// method, and Go holds both in one namespace.
func (b *Builder) UseWritePDO() *Builder {
	b.useWritePDO = true
	return b
}

// UsingWritePDO reports whether the read goes to the write connection.
// Mechanical, for the reason UseWritePDO gives.
func (b *Builder) UsingWritePDO() bool { return b.useWritePDO }

// AfterQuery registers a callback to run over the result after the query
// executes.
func (b *Builder) AfterQuery(callback func([]Record) []Record) *Builder {
	b.AfterQueryCallbacks = append(b.AfterQueryCallbacks, callback)
	return b
}

// ApplyAfterQueryCallbacks runs every registered AfterQuery callback over
// result. Each callback receives what the one before it returned, so they
// compose.
func (b *Builder) ApplyAfterQueryCallbacks(result []Record) []Record {
	for _, callback := range b.AfterQueryCallbacks {
		result = callback(result)
	}
	return result
}

// GetColumns returns the select list, with any Expression rendered to the
// SQL it carries rather than returned as the object.
//
// It returns an empty slice rather than nil when nothing was selected, so a
// caller ranging the result never has to special-case nil.
func (b *Builder) GetColumns() []any {
	out := make([]any, 0, len(b.Columns))
	for _, column := range b.Columns {
		if IsExpression(column) {
			out = append(out, stringify(column))
			continue
		}
		out = append(out, column)
	}
	return out
}

// GetLimit returns the row limit, if one was set. Mechanical, for the same
// reason as GetFrom.
func (b *Builder) GetLimit() *int { return b.limit }

// GetOffset returns the row offset, if one was set. Mechanical, for the same
// reason as GetFrom.
func (b *Builder) GetOffset() *int { return b.offset }

// GetConnection returns the connection the query runs against.
func (b *Builder) GetConnection() Connection { return b.connection }

// GetGrammar returns the grammar the query compiles through.
func (b *Builder) GetGrammar() Grammar { return b.Grammar }

// GetProcessor returns the processor that adjusts the query's results.
func (b *Builder) GetProcessor() Processor { return b.Processor }

// GetFrom returns the table the query reads from.
//
// The fluent From method claimed the name "from" in Go's single namespace, so
// the underlying state is unexported and reachable only through this getter.
func (b *Builder) GetFrom() any { return b.from }

// GetDistinct returns the DISTINCT state: false, true, or the columns of a
// DISTINCT ON. Mechanical, for the same reason as GetFrom.
func (b *Builder) GetDistinct() any { return b.distinct }

// GetLock returns the lock state. Mechanical, for the same reason as GetFrom.
func (b *Builder) GetLock() any { return b.lock }

// GetAggregate returns the aggregate being compiled, if any. Mechanical, for
// the same reason as GetFrom.
func (b *Builder) GetAggregate() *Aggregate { return b.aggregate }

// SetAggregate sets the aggregate function and columns being compiled.
//
// It also drops the ordering when there is no GROUP BY, which is easy to
// miss and is the half that matters: ORDER BY on an aggregate with no
// grouping is a sort of one row, and MySQL in strict mode refuses it outright
// when the ordering column is not in the select. The order bindings go with
// it, or the placeholders and the values stop lining up.
func (b *Builder) SetAggregate(function string, columns []any) *Builder {
	b.aggregate = &Aggregate{Function: function, Columns: columns}
	if len(b.Groups) == 0 {
		b.Orders = nil
		b.Bindings["order"] = nil
	}
	return b
}

// GetTimeout returns the statement timeout in seconds, if one was set.
// Mechanical, for the same reason as GetFrom.
func (b *Builder) GetTimeout() *int { return b.timeout }

// GetGroupLimit returns the per-group limit, if one was set. Mechanical, for
// the same reason as GetFrom.
func (b *Builder) GetGroupLimit() *GroupLimit { return b.groupLimit }

// GetIndexHint returns the index hint, if one was set. Mechanical, for the same
// reason as GetFrom.
func (b *Builder) GetIndexHint() *IndexHint { return b.indexHint }

// Clone returns a copy of the builder.
//
// The slices and the binding map are copied, so a clone that gains a where
// does not grow one on the query it was cloned from. That sharing is the bug
// behind every "the count query has the pagination limit on it" report.
func (b *Builder) Clone() *Builder {
	out := *b
	out.Columns = append([]any(nil), b.Columns...)
	out.Wheres = append([]Where(nil), b.Wheres...)
	out.Joins = append([]*JoinClause(nil), b.Joins...)
	out.Groups = append([]any(nil), b.Groups...)
	out.Havings = append([]Having(nil), b.Havings...)
	out.Orders = append([]Order(nil), b.Orders...)
	out.Unions = append([]Union(nil), b.Unions...)
	out.UnionOrders = append([]Order(nil), b.UnionOrders...)
	out.subqueries = append([]pendingSub(nil), b.subqueries...)
	out.Bindings = make(map[string][]any, len(b.Bindings))
	for key, values := range b.Bindings {
		out.Bindings[key] = append([]any(nil), values...)
	}
	return &out
}

// CloneWithout returns a clone with the named properties reset to their
// zero value.
func (b *Builder) CloneWithout(properties ...string) *Builder {
	out := b.Clone()
	for _, property := range properties {
		switch strings.ToLower(property) {
		case "columns":
			out.Columns = nil
			out.forgetSubqueriesOfKind("select")
		case "wheres":
			out.Wheres = nil
		case "joins":
			out.Joins = nil
			out.forgetSubqueriesOfKind("join")
		case "groups":
			out.Groups = nil
		case "havings":
			out.Havings = nil
		case "orders":
			out.Orders = nil
		case "unions":
			out.Unions = nil
		case "unionorders":
			out.UnionOrders = nil
		case "unionlimit":
			out.UnionLimit = nil
		case "unionoffset":
			out.UnionOffset = nil
		case "limit":
			out.limit = nil
		case "offset":
			out.offset = nil
		case "aggregate":
			out.aggregate = nil
		}
	}
	return out
}

// CloneWithoutBindings returns a clone with the named binding segments
// cleared.
func (b *Builder) CloneWithoutBindings(except ...string) *Builder {
	out := b.Clone()
	for _, typ := range except {
		if _, ok := out.Bindings[typ]; ok {
			out.Bindings[typ] = nil
			out.forgetSubqueriesOfSegment(typ)
		}
	}
	return out
}
