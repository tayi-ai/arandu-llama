package model

import (
	"context"
	"fmt"
	"reflect"
	"slices"
	"strings"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database/model/relations"
	"github.com/arandu-io/hesape/database/query"
)

// Builder is the query builder that hands back models instead of rows.
//
// Everything that runs takes an auth.Grant and filters by auth.Tenant(g) --
// reads exactly like writes. Everything that only builds does not: a
// Where or an OrderBy is a fragment of SQL, and a fragment authorizes nothing.
type Builder[T any] struct {
	query *query.Builder
	model *Model[T]

	eagerLoad     map[string]func(*query.Builder)
	scopes        map[string]Scope[T]
	removedScopes []string

	onDelete            func(context.Context, *Builder[T], auth.Grant) (int64, error)
	afterQueryCallbacks []func(Collection[T]) Collection[T]
	onCloneCallbacks    []func(*Builder[T])
	pendingAttributes   map[string]any

	// prepared says that the scopes and the tenant filter are already on this
	// builder's query.
	//
	// It exists because prepare is called by every method that runs, and those
	// methods call each other: Get prepares and then asks GetModels, which
	// prepares too. Without the flag the tenant filter and every global scope
	// landed in the where clause twice -- the same rows, the same bindings, and a
	// query nobody could read.
	prepared bool

	// err is what a builder method could not report.
	//
	// A method that returned an error could not be chained, and a chain that
	// has to be broken every second call is a chain nobody writes -- so the
	// error is held and returned by the first method that runs. Nothing is
	// ever executed with an error waiting.
	err error
}

// fail records an error for the first method that runs to report. See
// Builder.err.
func (b *Builder[T]) fail(err error) *Builder[T] {
	if b.err == nil {
		b.err = err
	}
	return b
}

// NewBuilder creates a Builder for q. The model is attached afterward,
// through SetModel.
func NewBuilder[T any](q *query.Builder) *Builder[T] {
	return &Builder[T]{query: q, eagerLoad: map[string]func(*query.Builder){}}
}

// DB is a connection that also carries the grammar its statements compile
// through and the processor their results are read back through.
//
// query.Connection is the five verbs and nothing else, on purpose: it is the
// contract a driver implements. The two below are not a driver's job, they are
// what a connection already knows about itself -- which grammar quotes its
// identifiers, which processor reads an inserted key back -- and asking for them
// by assertion is how Go spells a capability a narrow contract does not carry.
// Transactor is the same shape for the same reason.
type DB interface {
	query.Connection

	// GetQueryGrammar returns the grammar statements are compiled through.
	GetQueryGrammar() query.Grammar

	// GetPostProcessor returns the processor results are read back through.
	GetPostProcessor() query.Processor
}

// Query returns a query over T, on db, with T's global scopes registered.
//
//	user, err := model.Query[User](db).Where("email", "=", email).First(ctx, g)
//
// What it does that NewModel(table, connection, grammar, processor).NewQuery()
// does not is refuse to be told what is already known. The table is the name of
// T, pluralised and snake cased, which is what GetTable falls back to; the
// grammar and the processor are the connection's own. NewModel cannot work any
// of the three out, because all three arrive as arguments -- and an argument
// that repeats what a value already carries is an argument that can disagree
// with it.
//
// NewModel stays for the model that is not the default: another table, another
// key, a table with no tenant column, soft deletes. Configure that model once,
// keep it, and ask it for its query.
//
// It does not take a Grant, and that is the decision rather than an omission.
// Every terminal takes one already, because authorization belongs to the
// statement that runs and not to the sentence that builds it: a Grant held on
// the builder would be a second place a tenant could come from, and the two
// could differ. A builder authorizes nothing, which is why it can be built,
// stored and passed around without one.
func Query[T any](db DB) *Builder[T] {
	return NewModel[T]("", db, db.GetQueryGrammar(), db.GetPostProcessor()).NewQuery()
}

// NewTypedBuilder returns the Builder used for m's queries.
//
// A model that wants a wider query writes a type that embeds Builder[T] and
// overrides this method. It does not set the model, because NewModelQuery
// does that afterward.
func (m *Model[T]) NewTypedBuilder(q *query.Builder) *Builder[T] {
	return NewBuilder[T](q)
}

// NewBaseQueryBuilder returns a plain query.Builder scoped to m's table, with
// no model-level behavior attached.
func (m *Model[T]) NewBaseQueryBuilder() *query.Builder {
	q := query.NewBuilder(m.connection, m.Grammar, m.Processor)
	q.From(m.GetTable())
	return q
}

// NewModelQuery returns a builder with no global scopes and no eager loads.
func (m *Model[T]) NewModelQuery() *Builder[T] {
	return m.NewTypedBuilder(m.NewBaseQueryBuilder()).SetModel(m)
}

// NewQuery returns the model's query with its global scopes on.
func (m *Model[T]) NewQuery() *Builder[T] {
	return m.RegisterGlobalScopes(m.NewQueryWithoutScopes())
}

// Query returns the model's query with its global scopes on -- the same as
// NewQuery. Go has no static form of a generic method, so there is only one
// way to ask a model for its query.
func (m *Model[T]) Query() *Builder[T] { return m.NewQuery() }

// NewQueryWithoutScopes returns a builder for m with no global scopes
// applied.
func (m *Model[T]) NewQueryWithoutScopes() *Builder[T] { return m.NewModelQuery() }

// NewQueryWithoutRelationships returns the model's query with its global
// scopes on.
func (m *Model[T]) NewQueryWithoutRelationships() *Builder[T] {
	return m.RegisterGlobalScopes(m.NewModelQuery())
}

// NewQueryWithoutScope returns the model's query with its global scopes on,
// except the one named by identifier.
func (m *Model[T]) NewQueryWithoutScope(identifier string) *Builder[T] {
	return m.NewQuery().WithoutGlobalScope(identifier)
}

// On returns the model's query, run against another connection.
//
// The connection comes with the name: there is no resolver to look one up in.
func (m *Model[T]) On(name string, connection query.Connection) *Builder[T] {
	instance, err := m.NewInstance(nil, false)
	if err != nil {
		// NewInstance can only fail on a value that does not fit a field, and
		// nothing is being filled here.
		instance = m
	}
	return instance.SetConnection(name, connection).NewQuery()
}

// RegisterGlobalScopes copies the model's global scopes onto b, and returns
// b.
//
// It also registers the SoftDeletingScope when the model soft deletes. A
// model has no construction-time hook to do this once up front, so the
// registration happens here, every time the scopes are collected.
func (m *Model[T]) RegisterGlobalScopes(b *Builder[T]) *Builder[T] {
	if m.SoftDeletes && !m.HasGlobalScope(SoftDeletingScopeName) {
		m.AddGlobalScope(SoftDeletingScopeName, &SoftDeletingScope[T]{})
	}
	for identifier, scope := range m.globalScopes {
		b.WithGlobalScope(identifier, scope)
	}
	return b
}

// All returns every row of the model's table, subject to its global scopes.
func (m *Model[T]) All(ctx context.Context, g auth.Grant, columns ...any) (Collection[T], error) {
	return m.NewQuery().Get(ctx, g, columns...)
}

// SetModel attaches model to b, and points the query at its table.
func (b *Builder[T]) SetModel(model *Model[T]) *Builder[T] {
	b.model = model
	b.query.From(model.GetTable())
	return b
}

// GetModel returns the model this builder queries.
func (b *Builder[T]) GetModel() *Model[T] { return b.model }

// GetQuery returns the underlying query.Builder.
func (b *Builder[T]) GetQuery() *query.Builder { return b.query }

// SetQuery replaces the underlying query.Builder.
func (b *Builder[T]) SetQuery(q *query.Builder) *Builder[T] {
	b.query = q
	return b
}

// ToBase returns the underlying query.Builder with the scopes applied.
//
// It takes the Grant because applying the scopes is also where the tenant
// filter goes on, and a base builder handed out without it is a query
// somebody will run.
func (b *Builder[T]) ToBase(ctx context.Context, g auth.Grant) (*query.Builder, error) {
	prepared, err := b.prepare(g)
	if err != nil {
		return nil, err
	}
	return prepared.query, nil
}

// Qualify returns column qualified with the model's table.
func (b *Builder[T]) Qualify(column string) string { return b.model.QualifyColumn(column) }

// NewModelInstance returns a new, unsaved model of the builder's type, with
// attributes merged over any pending attributes from WithAttributes.
func (b *Builder[T]) NewModelInstance(attributes map[string]any) (*Model[T], error) {
	merged := copyMap(b.pendingAttributes)
	for key, value := range attributes {
		merged[key] = value
	}
	return b.model.NewInstance(merged, false)
}

// WithAttributes records values that filter the query and then fill whatever
// the query creates.
//
// asConditions defaults to true; passing false keeps the values for
// NewModelInstance without adding the where clauses -- which is what a
// relation does with a foreign key it already constrained another way.
//
// There is no separate single-column form: a map with one entry is that
// call.
func (b *Builder[T]) WithAttributes(attributes map[string]any, asConditions ...bool) *Builder[T] {
	if optionalBool(asConditions) {
		for _, column := range sortedKeys(attributes) {
			b.Where(b.Qualify(column), "=", attributes[column])
		}
	}
	if b.pendingAttributes == nil {
		b.pendingAttributes = map[string]any{}
	}
	for column, value := range attributes {
		b.pendingAttributes[column] = value
	}
	return b
}

// WithSavepointIfNeeded runs scope inside a savepoint when a transaction is
// already open, and plainly when none is.
//
// query.Connection does not declare a transaction level -- see Transactor
// for why this component does not widen it -- so the capability is asked for
// by a type assertion, and a connection that does not implement it runs the
// callback as if no transaction were open, the same as a level of zero.
func (b *Builder[T]) WithSavepointIfNeeded(scope func() error) error {
	nested, ok := b.query.GetConnection().(Savepointer)
	if !ok || nested.TransactionLevel() <= 0 {
		return scope()
	}
	return nested.Transaction(scope)
}

// clone returns a copy of b with its own query, scopes and callback slices,
// so that mutating the copy never touches b.
func (b *Builder[T]) clone() *Builder[T] {
	out := &Builder[T]{
		query:               b.query.Clone(),
		model:               b.model,
		eagerLoad:           map[string]func(*query.Builder){},
		scopes:              cloneScopes(b.scopes),
		removedScopes:       slices.Clone(b.removedScopes),
		onDelete:            b.onDelete,
		afterQueryCallbacks: slices.Clone(b.afterQueryCallbacks),
		pendingAttributes:   copyMap(b.pendingAttributes),
		prepared:            b.prepared,
		err:                 b.err,
	}
	for name, constraints := range b.eagerLoad {
		out.eagerLoad[name] = constraints
	}
	out.onCloneCallbacks = slices.Clone(b.onCloneCallbacks)
	for _, callback := range out.onCloneCallbacks {
		callback(out)
	}
	return out
}

// Clone returns a copy of b, safe to mutate independently.
func (b *Builder[T]) Clone() *Builder[T] { return b.clone() }

// prepare is where a query becomes runnable: the global scopes go on, and then
// the tenant filter.
//
// The tenant is read off the Grant and never from anywhere else. A Grant with no
// tenant -- the zero Grant, which is the only one constructible outside the auth
// package -- is refused here, before any SQL exists.
//
// The wheres already on the query are wrapped in one group before the tenant
// filter is appended, and that is not tidiness. `where a or b` with `and
// tenant = ?` appended reads as `a or (b and tenant = ?)`, so every row matching
// a comes back whoever it belongs to. Grouping first is what makes the filter
// mean what it says.
func (b *Builder[T]) prepare(g auth.Grant) (*Builder[T], error) {
	if b.err != nil {
		return nil, b.err
	}

	tenant := auth.Tenant(g)
	if tenant == "" || !auth.ValidTenant(tenant) {
		return nil, ErrNoTenant
	}
	if b.prepared {
		return b, nil
	}

	prepared := b.ApplyScopes()
	prepared.prepared = true

	if err := prepared.scopeToTenant(g, tenant); err != nil {
		return nil, err
	}
	return prepared, nil
}

// scopeToTenant is the only place a tenant lands on an model statement: the
// model's own filter, and then the nested half of the query builder's scoped.
//
// It is called on a builder nobody else holds -- prepare on the clone
// ApplyScopes made, ForceDelete on its own -- because the filter belongs to the
// statement and not to the query the caller kept: running the same builder twice
// under two Grants must not leave the first tenant's filter on the second
// tenant's statement.
//
// The second half is not a second filter, it is the rest of the same one.
// query.Builder.ScopeNested puts the tenant on the far side of a union, on the
// subquery of a `where exists`, a `where in` and a count comparison, and on the
// subqueries compiled into a from, a select or a join. Nothing here reached it
// until now: GetModels ran b.query.ToSQL() straight at the connection, so the
// filter written above was the whole of the tenant scoping, and it only ever
// named the outer table. That is what leaked --
// Users.WithCount("posts").Get(auth.SystemGrant("user.list", "acme")) emitted
// `select "users".*, (select count(*) from "posts" where "users"."id" =
// "posts"."user_id") as "posts_count" from "users" where "users"."tenant_id" =
// ?`, and every tenant's posts were counted into every row.
//
// The context is Background because this package carries none, for the reason
// its doc comment gives: the connection contract takes no context, so a
// signature that accepted one could not pass it on. All ScopeNested does with it
// is refuse to build a statement whose context is already cancelled, and there
// is nothing here for that to cancel.
func (b *Builder[T]) scopeToTenant(g auth.Grant, tenant string) error {
	if column := b.model.TenantColumn; column != "" {
		isolateWheres(b.query)
		b.query.Where(b.model.QualifyColumn(column), "=", tenant)
	}
	return b.query.ScopeNested(context.Background(), g)
}

// isolateWheres wraps every where already on the query in a single group, so
// that a filter added after them cannot be swallowed by an or. See prepare.
func isolateWheres(q *query.Builder) {
	if len(q.Wheres) < 2 {
		return
	}
	hasOr := false
	for _, where := range q.Wheres {
		if strings.Contains(where.Boolean, "or") {
			hasOr = true
			break
		}
	}
	if !hasOr {
		return
	}

	group := q.ForNestedWhere()
	group.Wheres = q.Wheres
	boolean := strings.ReplaceAll(q.Wheres[0].Boolean, " not", "")
	if boolean == "" {
		boolean = "and"
	}
	q.Wheres = []query.Where{{Type: "Nested", Query: group, Boolean: boolean}}
}

// WithGlobalScope registers scope under identifier, and extends b
// immediately if scope implements ScopeExtender.
func (b *Builder[T]) WithGlobalScope(identifier string, scope Scope[T]) *Builder[T] {
	if b.scopes == nil {
		b.scopes = map[string]Scope[T]{}
	}
	b.scopes[identifier] = scope
	if extender, ok := scope.(ScopeExtender[T]); ok {
		extender.Extend(b)
	}
	return b
}

// WithoutGlobalScope removes the scope registered under identifier, and
// records it as removed.
func (b *Builder[T]) WithoutGlobalScope(identifier string) *Builder[T] {
	delete(b.scopes, identifier)
	b.removedScopes = append(b.removedScopes, identifier)
	return b
}

// WithoutGlobalScopes removes the named scopes. With no argument it removes
// them all.
func (b *Builder[T]) WithoutGlobalScopes(identifiers ...string) *Builder[T] {
	if len(identifiers) == 0 {
		for identifier := range b.scopes {
			identifiers = append(identifiers, identifier)
		}
	}
	for _, identifier := range identifiers {
		b.WithoutGlobalScope(identifier)
	}
	return b
}

// RemovedScopes returns the identifiers of the scopes removed from this
// builder.
func (b *Builder[T]) RemovedScopes() []string { return slices.Clone(b.removedScopes) }

// ApplyScopes returns a copy of the builder with every registered scope
// applied.
//
// The wheres a scope adds are wrapped in a group when either side carries an
// or: without that, a scope's filter joins an or chain and stops filtering.
func (b *Builder[T]) ApplyScopes() *Builder[T] {
	if len(b.scopes) == 0 {
		return b.clone()
	}
	out := b.clone()
	for _, identifier := range sortedScopeNames(b.scopes) {
		scope := b.scopes[identifier]
		before := len(out.query.Wheres)
		scope.Apply(out, out.model)
		groupNewWheres(out.query, before)
	}
	return out
}

// sortedScopeNames keeps the order scopes are applied in stable. A Go map
// has no order, and two scopes that disagree about the order they run in are
// a bug that only shows up sometimes.
func sortedScopeNames[T any](scopes map[string]Scope[T]) []string {
	out := make([]string, 0, len(scopes))
	for identifier := range scopes {
		out = append(out, identifier)
	}
	slices.Sort(out)
	return out
}

// groupNewWheres wraps the wheres added since originalCount in their own
// group when needed, so a scope's filter cannot be absorbed by a
// surrounding or.
func groupNewWheres(q *query.Builder, originalCount int) {
	if len(q.Wheres) == originalCount {
		return
	}
	all := q.Wheres
	q.Wheres = nil
	groupWhereSliceForScope(q, all[:originalCount])
	groupWhereSliceForScope(q, all[originalCount:])
}

// groupWhereSliceForScope appends slice to q's wheres, wrapped in one nested
// group when any entry in it uses "or".
func groupWhereSliceForScope(q *query.Builder, slice []query.Where) {
	if len(slice) == 0 {
		return
	}
	hasOr := false
	for _, where := range slice {
		if strings.Contains(where.Boolean, "or") {
			hasOr = true
			break
		}
	}
	if !hasOr {
		q.Wheres = append(q.Wheres, slice...)
		return
	}
	group := q.ForNestedWhere()
	group.Wheres = slice
	q.Wheres = append(q.Wheres, query.Where{
		Type:    "Nested",
		Query:   group,
		Boolean: strings.ReplaceAll(slice[0].Boolean, " not", ""),
	})
}

// Where adds a where clause. Passing a func(*Builder[T]) instead of a column
// name adds a group built by calling that function with a fresh builder.
func (b *Builder[T]) Where(column any, args ...any) *Builder[T] {
	if nested, ok := column.(func(*Builder[T])); ok {
		return b.whereNested(nested, "and")
	}
	b.query.Where(column, args...)
	return b
}

// OrWhere adds an or-where clause. Passing a func(*Builder[T]) instead of a
// column name adds a group built by calling that function with a fresh
// builder.
func (b *Builder[T]) OrWhere(column any, args ...any) *Builder[T] {
	if nested, ok := column.(func(*Builder[T])); ok {
		return b.whereNested(nested, "or")
	}
	b.query.OrWhere(column, args...)
	return b
}

// whereNested runs callback against a fresh builder for the same model, and
// adds what it builds as one group joined with boolean.
func (b *Builder[T]) whereNested(callback func(*Builder[T]), boolean string) *Builder[T] {
	nested := b.model.NewQueryWithoutRelationships()
	callback(nested)
	if nested.err != nil {
		// The nested builder is thrown away once its wheres are merged, so an
		// error left on it would never reach anybody.
		b.fail(nested.err)
	}
	for name, constraints := range nested.eagerLoad {
		b.eagerLoad[name] = constraints
	}
	b.WithoutGlobalScopes(nested.removedScopes...)
	b.query.AddNestedWhereQuery(nested.query, boolean)
	return b
}

// WhereNot adds a where clause wrapped in a negated group: NOT (column
// args...).
func (b *Builder[T]) WhereNot(column any, args ...any) *Builder[T] {
	before := len(b.query.Wheres)
	b.Where(func(nested *Builder[T]) {
		nested.Where(column, args...)
	})
	return b.negateLastWhere(before)
}

// negateLastWhere flips the boolean of the group at index before to "...
// not", negating it.
//
// It negates nothing when the group turned out empty -- an empty nested
// where is dropped rather than compiled, and negating whatever came before
// it would change a clause the caller did not write.
func (b *Builder[T]) negateLastWhere(before int) *Builder[T] {
	if len(b.query.Wheres) == before {
		return b
	}
	last := &b.query.Wheres[len(b.query.Wheres)-1]
	if !strings.Contains(last.Boolean, "not") {
		last.Boolean += " not"
	}
	return b
}

// OrWhereNot adds an or-joined, negated group: OR NOT (column args...).
func (b *Builder[T]) OrWhereNot(column any, args ...any) *Builder[T] {
	before := len(b.query.Wheres)
	b.OrWhere(func(nested *Builder[T]) {
		nested.Where(column, args...)
	})
	return b.negateLastWhere(before)
}

// WhereKey filters by the model's primary key. A slice of ids adds a WHERE
// IN instead of an equality.
func (b *Builder[T]) WhereKey(id any) *Builder[T] {
	if ids, ok := id.([]any); ok {
		b.query.WhereIn(b.model.GetQualifiedKeyName(), ids)
		return b
	}
	return b.Where(b.model.GetQualifiedKeyName(), "=", id)
}

// WhereKeyNot excludes the model's primary key. A slice of ids adds a WHERE
// NOT IN instead of an inequality.
func (b *Builder[T]) WhereKeyNot(id any) *Builder[T] {
	if ids, ok := id.([]any); ok {
		b.query.WhereNotIn(b.model.GetQualifiedKeyName(), ids)
		return b
	}
	return b.Where(b.model.GetQualifiedKeyName(), "!=", id)
}

// Latest orders the query by column, or the model's created-at column when
// none is given, newest first.
func (b *Builder[T]) Latest(column ...string) *Builder[T] {
	b.query.Latest(timestampColumn(b.model.GetCreatedAtColumn(), column))
	return b
}

// Oldest orders the query by column, or the model's created-at column when
// none is given, oldest first.
func (b *Builder[T]) Oldest(column ...string) *Builder[T] {
	b.query.Oldest(timestampColumn(b.model.GetCreatedAtColumn(), column))
	return b
}

func timestampColumn(fallback string, column []string) any {
	if len(column) > 0 && column[0] != "" {
		return column[0]
	}
	if fallback == "" {
		return "created_at"
	}
	return fallback
}

// Hydrate turns rows into models: rows in, models out.
func (b *Builder[T]) Hydrate(items []query.Record) (Collection[T], error) {
	found, err := b.hydrate(items)
	if err != nil {
		return nil, err
	}
	return found.entities(), nil
}

// hydrate is Hydrate with the models still in hand.
//
// Every read in this package goes through it, and the ones that keep working on
// the result -- an eager load, a Fresh, the key a chunk resumes from -- stay on
// this side rather than on the Collection. See models.
func (b *Builder[T]) hydrate(items []query.Record) (models[T], error) {
	out := make(models[T], 0, len(items))
	for _, item := range items {
		model, err := b.model.NewFromBuilder(item)
		if err != nil {
			return nil, err
		}
		out = append(out, model)
	}
	return out, nil
}

// FromQuery returns models from SQL somebody wrote by hand.
//
// It takes the Grant like every other read. The SQL is the caller's, so the
// tenant cannot be added to it -- which is exactly why the Grant is still
// required: a query nobody authorized does not run, and the where clause that
// scopes it is the caller's to write.
func (b *Builder[T]) FromQuery(ctx context.Context, g auth.Grant, sql string, bindings []any) (Collection[T], error) {
	// The other method that skips prepare, and so the other one that has to read
	// the held error itself. See ForceDelete.
	if b.err != nil {
		return nil, b.err
	}
	if tenant := auth.Tenant(g); tenant == "" || !auth.ValidTenant(tenant) {
		return nil, ErrNoTenant
	}
	rows, err := b.model.connection.Select(ctx, sql, bindings, true)
	if err != nil {
		return nil, fmt.Errorf("model: selecting from %s: %w", b.model.GetTable(), err)
	}
	return b.Hydrate(rows)
}

// Get runs the query and returns the matching models, with their eager
// loads applied.
func (b *Builder[T]) Get(ctx context.Context, g auth.Grant, columns ...any) (Collection[T], error) {
	found, err := b.get(ctx, g, columns...)
	if err != nil {
		return nil, err
	}
	return b.results(found), nil
}

// get is Get with the models still in hand, and without the after-query
// callbacks.
//
// The callbacks are the caller's, and they are applied where the result is
// handed to one: results is the other half, and every read in this package that
// keeps working on the rows calls both.
func (b *Builder[T]) get(ctx context.Context, g auth.Grant, columns ...any) (models[T], error) {
	prepared, err := b.prepare(g)
	if err != nil {
		return nil, err
	}
	found, err := prepared.getModels(ctx, g, columns...)
	if err != nil {
		return nil, err
	}
	if len(found) > 0 {
		if err := prepared.eagerLoadRelations(ctx, g, found); err != nil {
			return nil, err
		}
	}
	return found, nil
}

// results is what get's models become for a caller: the after-query callbacks,
// over the rows.
func (b *Builder[T]) results(found models[T]) Collection[T] {
	return b.ApplyAfterQueryCallbacks(found.entities())
}

// result is results for a terminal that matched at most one row.
//
// A First is a Get with a limit of one, so its row is a one-row result and the
// callbacks see it as one. Nothing matched stays nothing matched: a callback
// that replaced an empty result would be answering a question nobody asked.
func (b *Builder[T]) result(model *Model[T]) *T {
	if model == nil {
		return nil
	}
	if len(b.afterQueryCallbacks) == 0 {
		return model.Entity
	}
	return b.results(models[T]{model}).First()
}

// GetModels returns the rows, hydrated, with nothing eager loaded.
//
// It is already prepared when Get calls it; called on its own it prepares
// itself, so there is no way to reach the rows without the Grant.
func (b *Builder[T]) GetModels(ctx context.Context, g auth.Grant, columns ...any) (Collection[T], error) {
	found, err := b.getModels(ctx, g, columns...)
	if err != nil {
		return nil, err
	}
	return found.entities(), nil
}

// getModels is GetModels with the models still in hand. See hydrate.
func (b *Builder[T]) getModels(ctx context.Context, g auth.Grant, columns ...any) (models[T], error) {
	prepared, err := b.prepare(g)
	if err != nil {
		return nil, err
	}
	if len(columns) > 0 && prepared.query.Columns == nil {
		prepared.query.Select(columns...)
	}
	rows, err := prepared.runSelect(ctx)
	if err != nil {
		return nil, err
	}
	return prepared.hydrate(rows)
}

// AfterQuery registers a callback run on the result of Get, allowed to
// replace it.
//
// It runs wherever a Collection is handed to a caller: Get, Chunk and the walks
// built on it, and the terminals that are a Get with a limit -- First, Sole,
// Find and their neighbours, whose row the callback sees as a result of one.
//
// It does not run on the reads this package makes for itself, and the reason is
// the signature. A callback takes rows and answers rows, and a row is not a way
// back to the model behind it, so a read whose next step is on the model cannot
// pass through one: the row Refresh reads back into a model it already holds,
// the rows Destroy loads in order to delete them, the aggregate LoadCount fills
// in, everything the relation tree reads through its own seam. Nor does it run
// on a paginator, whose page is not a Collection but items of whatever type the
// page was asked for.
func (b *Builder[T]) AfterQuery(callback func(Collection[T]) Collection[T]) *Builder[T] {
	b.afterQueryCallbacks = append(b.afterQueryCallbacks, callback)
	return b
}

// ApplyAfterQueryCallbacks runs the registered AfterQuery callbacks over
// result in order, threading each callback's replacement into the next.
func (b *Builder[T]) ApplyAfterQueryCallbacks(result Collection[T]) Collection[T] {
	for _, callback := range b.afterQueryCallbacks {
		if next := callback(result); next != nil {
			result = next
		}
	}
	return result
}

// First returns the first row matching the query, or (nil, nil) when there
// is none: no row is not a failure, and FirstOrFail is the spelling for when
// it is.
func (b *Builder[T]) First(ctx context.Context, g auth.Grant, columns ...any) (*T, error) {
	model, err := b.first(ctx, g, columns...)
	return b.result(model), err
}

// first is First with the model still in hand, and without the after-query
// callbacks. See get.
func (b *Builder[T]) first(ctx context.Context, g auth.Grant, columns ...any) (*Model[T], error) {
	found, err := b.Limit(1).get(ctx, g, columns...)
	if err != nil {
		return nil, err
	}
	if len(found) == 0 {
		return nil, nil
	}
	return found[0], nil
}

// FirstOrFail returns the first row matching the query, or an error when
// there is none.
func (b *Builder[T]) FirstOrFail(ctx context.Context, g auth.Grant, columns ...any) (*T, error) {
	model, err := b.firstOrFail(ctx, g, columns...)
	if err != nil {
		return nil, err
	}
	return b.result(model), nil
}

// firstOrFail is firstOrFail with the model still in hand, and without the
// after-query callbacks. See get.
func (b *Builder[T]) firstOrFail(ctx context.Context, g auth.Grant, columns ...any) (*Model[T], error) {
	model, err := b.first(ctx, g, columns...)
	if err != nil {
		return nil, err
	}
	if model == nil {
		return nil, modelNotFound(b.model.GetTable())
	}
	return model, nil
}

// FirstOr returns the first row matching the query, or what callback makes
// when there is none.
func (b *Builder[T]) FirstOr(ctx context.Context, g auth.Grant, callback func() (*T, error), columns ...any) (*T, error) {
	model, err := b.first(ctx, g, columns...)
	if err != nil {
		return nil, err
	}
	if model != nil {
		return b.result(model), nil
	}
	return callback()
}

// FirstWhere adds a where clause and returns the first matching row.
func (b *Builder[T]) FirstWhere(ctx context.Context, g auth.Grant, column any, args ...any) (*T, error) {
	return b.Where(column, args...).First(ctx, g)
}

// Sole returns the row matching the query, and fails unless it is the only
// one.
func (b *Builder[T]) Sole(ctx context.Context, g auth.Grant, columns ...any) (*T, error) {
	model, err := b.sole(ctx, g, columns...)
	if err != nil {
		return nil, err
	}
	return b.result(model), nil
}

// sole is Sole with the model still in hand, and without the after-query
// callbacks. See get.
func (b *Builder[T]) sole(ctx context.Context, g auth.Grant, columns ...any) (*Model[T], error) {
	found, err := b.Limit(2).get(ctx, g, columns...)
	if err != nil {
		return nil, err
	}
	switch len(found) {
	case 0:
		return nil, modelNotFound(b.model.GetTable())
	case 1:
		return found[0], nil
	default:
		return nil, fmt.Errorf("%w: %s", ErrMultipleRecordsFound, b.model.GetTable())
	}
}

// Find returns the row with the given primary key, or the rows for a slice
// of keys.
func (b *Builder[T]) Find(ctx context.Context, g auth.Grant, id any, columns ...any) (*T, error) {
	model, err := b.find(ctx, g, id, columns...)
	if err != nil {
		return nil, err
	}
	return b.result(model), nil
}

// find is Find with the model still in hand, and without the after-query
// callbacks. See get.
func (b *Builder[T]) find(ctx context.Context, g auth.Grant, id any, columns ...any) (*Model[T], error) {
	if ids, ok := id.([]any); ok {
		found, err := b.findMany(ctx, g, ids, columns...)
		if err != nil || len(found) == 0 {
			return nil, err
		}
		return found[0], nil
	}
	return b.WhereKey(id).first(ctx, g, columns...)
}

// FindMany returns the rows matching any of ids.
func (b *Builder[T]) FindMany(ctx context.Context, g auth.Grant, ids []any, columns ...any) (Collection[T], error) {
	found, err := b.findMany(ctx, g, ids, columns...)
	if err != nil {
		return nil, err
	}
	return b.results(found), nil
}

// findMany is FindMany with the models still in hand, and without the
// after-query callbacks. See get.
func (b *Builder[T]) findMany(ctx context.Context, g auth.Grant, ids []any, columns ...any) (models[T], error) {
	if len(ids) == 0 {
		return models[T]{}, nil
	}
	return b.WhereKey(ids).get(ctx, g, columns...)
}

// FindOrFail returns the row with the given primary key, or an error when
// there is none.
//
// Given a list it also fails when one id is missing: asking for three rows
// and getting two is not a shorter result, it is a wrong one.
func (b *Builder[T]) FindOrFail(ctx context.Context, g auth.Grant, id any, columns ...any) (*T, error) {
	model, err := b.findOrFail(ctx, g, id, columns...)
	if err != nil {
		return nil, err
	}
	return b.result(model), nil
}

// findOrFail is FindOrFail with the model still in hand, and without the
// after-query callbacks. See get.
func (b *Builder[T]) findOrFail(ctx context.Context, g auth.Grant, id any, columns ...any) (*Model[T], error) {
	if ids, ok := id.([]any); ok {
		found, err := b.findMany(ctx, g, ids, columns...)
		if err != nil {
			return nil, err
		}
		if len(found) != len(uniqueValues(ids)) {
			return nil, modelNotFound(b.model.GetTable(), ids...)
		}
		return found[0], nil
	}
	model, err := b.find(ctx, g, id, columns...)
	if err != nil {
		return nil, err
	}
	if model == nil {
		return nil, modelNotFound(b.model.GetTable(), id)
	}
	return model, nil
}

// FindOrNew returns the row with the given primary key, or a new unsaved
// model when there is none.
func (b *Builder[T]) FindOrNew(ctx context.Context, g auth.Grant, id any, columns ...any) (*T, error) {
	model, err := b.find(ctx, g, id, columns...)
	if err != nil {
		return nil, err
	}
	if model != nil {
		return b.result(model), nil
	}
	instance, err := b.NewModelInstance(nil)
	if err != nil {
		return nil, err
	}
	return instance.Entity, nil
}

// FirstOrNew returns the first row matching attributes, or a new unsaved
// model built from attributes and values when there is none.
func (b *Builder[T]) FirstOrNew(ctx context.Context, g auth.Grant, attributes, values map[string]any) (*T, error) {
	model, err := b.clone().whereAll(attributes).first(ctx, g)
	if err != nil {
		return nil, err
	}
	if model != nil {
		return b.result(model), nil
	}
	instance, err := b.NewModelInstance(mergeMaps(attributes, values))
	if err != nil {
		return nil, err
	}
	return instance.Entity, nil
}

// FirstOrCreate returns the first row matching attributes, or creates and
// returns one from attributes and values when there is none.
func (b *Builder[T]) FirstOrCreate(ctx context.Context, g auth.Grant, attributes, values map[string]any) (*T, error) {
	model, err := b.firstOrCreate(ctx, g, attributes, values)
	if err != nil {
		return nil, err
	}
	return b.result(model), nil
}

// firstOrCreate is FirstOrCreate with the model still in hand, and without the
// after-query callbacks. See get.
func (b *Builder[T]) firstOrCreate(ctx context.Context, g auth.Grant, attributes, values map[string]any) (*Model[T], error) {
	model, err := b.clone().whereAll(attributes).first(ctx, g)
	if err != nil {
		return nil, err
	}
	if model != nil {
		return model, nil
	}
	return b.createOrFirst(ctx, g, attributes, values)
}

// CreateOrFirst inserts a row from attributes and values, and if a unique
// index says somebody got there first, reads theirs instead.
//
// Nothing here classifies a driver error yet, so any insert failure sends it
// looking for the row, and the insert error is returned when there is none
// -- which keeps the race safe and never swallows a real failure.
func (b *Builder[T]) CreateOrFirst(ctx context.Context, g auth.Grant, attributes, values map[string]any) (*T, error) {
	model, err := b.createOrFirst(ctx, g, attributes, values)
	if err != nil {
		return nil, err
	}
	return b.result(model), nil
}

// createOrFirst is CreateOrFirst with the model still in hand, and without the
// after-query callbacks. See get.
func (b *Builder[T]) createOrFirst(ctx context.Context, g auth.Grant, attributes, values map[string]any) (*Model[T], error) {
	model, err := b.create(ctx, g, mergeMaps(attributes, values))
	if err == nil {
		return model, nil
	}
	existing, findErr := b.clone().whereAll(attributes).first(ctx, g)
	if findErr != nil || existing == nil {
		return nil, err
	}
	return existing, nil
}

// UpdateOrCreate finds or creates a row matching attributes, then fills it
// with values and saves it.
func (b *Builder[T]) UpdateOrCreate(ctx context.Context, g auth.Grant, attributes, values map[string]any) (*T, error) {
	model, err := b.firstOrCreate(ctx, g, attributes, values)
	if err != nil {
		return nil, err
	}
	if model.WasRecentlyCreated {
		return b.result(model), nil
	}
	if err := model.Fill(values); err != nil {
		return nil, err
	}
	if _, err := model.Save(ctx, g); err != nil {
		return nil, err
	}
	return b.result(model), nil
}

// whereAll adds one equality where clause per entry in attributes.
func (b *Builder[T]) whereAll(attributes map[string]any) *Builder[T] {
	for _, column := range sortedKeys(attributes) {
		b.Where(b.model.QualifyColumn(column), "=", attributes[column])
	}
	return b
}

// Value returns one column of the first row matching the query.
func (b *Builder[T]) Value(ctx context.Context, g auth.Grant, column string) (any, error) {
	model, err := b.first(ctx, g, column)
	if err != nil || model == nil {
		return nil, err
	}
	return model.GetAttribute(afterLastDot(column)), nil
}

// ValueOrFail returns one column of the first row matching the query, or an
// error when there is none.
func (b *Builder[T]) ValueOrFail(ctx context.Context, g auth.Grant, column string) (any, error) {
	model, err := b.firstOrFail(ctx, g, column)
	if err != nil {
		return nil, err
	}
	return model.GetAttribute(afterLastDot(column)), nil
}

// SoleValue returns one column of the row matching the query, and fails
// unless it is the only one.
func (b *Builder[T]) SoleValue(ctx context.Context, g auth.Grant, column string) (any, error) {
	model, err := b.sole(ctx, g, column)
	if err != nil {
		return nil, err
	}
	return model.GetAttribute(afterLastDot(column)), nil
}

// Pluck returns one column of every row matching the query.
func (b *Builder[T]) Pluck(ctx context.Context, g auth.Grant, column string) ([]any, error) {
	found, err := b.get(ctx, g, column)
	if err != nil {
		return nil, err
	}
	return found.Pluck(afterLastDot(column)), nil
}

// Count returns the row count of the query, or the count of columns when
// given.
func (b *Builder[T]) Count(ctx context.Context, g auth.Grant, columns ...any) (int64, error) {
	if len(columns) == 0 {
		columns = []any{"*"}
	}
	value, err := b.Aggregate(ctx, g, "count", columns...)
	if err != nil {
		return 0, err
	}
	return toInt64(value), nil
}

// Aggregate runs function (count, sum, min, max, avg) over columns and
// returns the result.
func (b *Builder[T]) Aggregate(ctx context.Context, g auth.Grant, function string, columns ...any) (any, error) {
	prepared, err := b.prepare(g)
	if err != nil {
		return nil, err
	}
	if len(columns) == 0 {
		columns = []any{"*"}
	}
	return prepared.runAggregate(ctx, function, columns)
}

// Exists reports whether the query matches any row.
func (b *Builder[T]) Exists(ctx context.Context, g auth.Grant) (bool, error) {
	count, err := b.Count(ctx, g)
	return count > 0, err
}

// Create returns a new model, filled with attributes and saved.
func (b *Builder[T]) Create(ctx context.Context, g auth.Grant, attributes map[string]any) (*T, error) {
	instance, err := b.create(ctx, g, attributes)
	if err != nil {
		return nil, err
	}
	return instance.Entity, nil
}

// create is Create with the model still in hand. See get.
func (b *Builder[T]) create(ctx context.Context, g auth.Grant, attributes map[string]any) (*Model[T], error) {
	instance, err := b.NewModelInstance(attributes)
	if err != nil {
		return nil, err
	}
	if _, err := instance.Save(ctx, g); err != nil {
		return nil, err
	}
	return instance, nil
}

// ForceCreate returns a new model, filled with attributes via ForceFill and
// saved.
//
// There is no mass-assignment guard to turn off (see the package comment);
// what ForceCreate keeps from Create is ForceFill's behavior: an attribute
// the entity does not declare is carried through as a raw attribute instead
// of dropped.
func (b *Builder[T]) ForceCreate(ctx context.Context, g auth.Grant, attributes map[string]any) (*T, error) {
	instance, err := b.model.NewInstance(nil, false)
	if err != nil {
		return nil, err
	}
	if err := instance.ForceFill(mergeMaps(b.pendingAttributes, attributes)); err != nil {
		return nil, err
	}
	if _, err := instance.Save(ctx, g); err != nil {
		return nil, err
	}
	return instance.Entity, nil
}

// Insert writes values as new rows.
//
// The tenant column is written from the Grant on every row, overwriting
// whatever the caller put there: the tenant comes from the Grant and from
// nowhere else.
func (b *Builder[T]) Insert(ctx context.Context, g auth.Grant, values ...map[string]any) (bool, error) {
	prepared, rows, err := b.prepareWrite(g, values)
	if err != nil {
		return false, err
	}
	return prepared.runInsert(ctx, rows)
}

// InsertGetID inserts values as one new row and returns the value generated
// for sequence.
func (b *Builder[T]) InsertGetID(ctx context.Context, g auth.Grant, values map[string]any, sequence string) (int64, error) {
	prepared, rows, err := b.prepareWrite(g, []map[string]any{values})
	if err != nil {
		return 0, err
	}
	return prepared.runInsertGetID(ctx, rows[0], sequence)
}

// prepareWrite is prepare() for a statement that carries values: the Grant is
// checked, the scopes are applied, and the tenant is written into every row.
func (b *Builder[T]) prepareWrite(g auth.Grant, values []map[string]any) (*Builder[T], []map[string]any, error) {
	prepared, err := b.prepare(g)
	if err != nil {
		return nil, nil, err
	}
	rows := make([]map[string]any, 0, len(values))
	for _, row := range values {
		row = copyMap(row)
		if column := b.model.TenantColumn; column != "" {
			row[column] = auth.Tenant(g)
		}
		rows = append(rows, row)
	}
	return prepared, rows, nil
}

// Update runs an UPDATE with values over the query as it stands, and
// returns the number of rows affected.
func (b *Builder[T]) Update(ctx context.Context, g auth.Grant, values map[string]any) (int64, error) {
	prepared, err := b.prepare(g)
	if err != nil {
		return 0, err
	}
	return prepared.runUpdate(ctx, prepared.addUpdatedAtColumn(values))
}

// addUpdatedAtColumn adds the updated-at timestamp to values when the model
// uses timestamps and the caller did not already set it.
//
// The column is re-keyed to <table>.<column>, so that an update with a join
// names the table it means. The grammars that cannot take a qualified name
// on the left of a SET -- Postgres and SQLite -- strip it again before
// compiling.
func (b *Builder[T]) addUpdatedAtColumn(values map[string]any) map[string]any {
	column := b.model.GetUpdatedAtColumn()
	if !b.model.UsesTimestamps() || column == "" {
		return values
	}

	out := copyMap(values)
	if _, ok := out[column]; !ok {
		if _, qualified := out[b.model.QualifyColumn(column)]; qualified {
			return out
		}
		out[column] = b.model.FreshTimestamp()
	}

	value := out[column]
	delete(out, column)
	out[tableOf(b.query.GetFrom())+"."+column] = value
	return out
}

// tableOf returns the table name from a FROM value, dropping any " as
// alias" suffix.
func tableOf(from any) string {
	name := fmt.Sprint(from)
	if i := strings.Index(strings.ToLower(name), " as "); i >= 0 {
		return strings.TrimSpace(name[i+4:])
	}
	return strings.TrimSpace(name)
}

// Upsert inserts values, updating the columns in update on any row that
// conflicts on uniqueBy. With no update columns given, every column is
// updated.
func (b *Builder[T]) Upsert(ctx context.Context, g auth.Grant, values []map[string]any, uniqueBy, update []string) (int64, error) {
	if len(values) == 0 {
		return 0, nil
	}
	if len(uniqueBy) == 0 {
		return 0, fmt.Errorf("model: the unique columns must not be empty")
	}
	prepared, rows, err := b.prepareWrite(g, values)
	if err != nil {
		return 0, err
	}
	if update == nil {
		update = sortedKeys(rows[0])
	}
	if column := b.model.GetUpdatedAtColumn(); b.model.UsesTimestamps() && column != "" {
		now := b.model.FreshTimestamp()
		for _, row := range rows {
			if _, ok := row[column]; !ok {
				row[column] = now
			}
		}
		if !slices.Contains(update, column) {
			update = append(update, column)
		}
	}
	return prepared.runUpsert(ctx, rows, uniqueBy, update)
}

// Increment adds amount to column, plus any extra columns to set, and
// returns the number of rows affected.
func (b *Builder[T]) Increment(ctx context.Context, g auth.Grant, column string, amount any, extra map[string]any) (int64, error) {
	return b.incrementOrDecrement(ctx, g, column, amount, extra, "+")
}

// Decrement subtracts amount from column, plus any extra columns to set, and
// returns the number of rows affected.
func (b *Builder[T]) Decrement(ctx context.Context, g auth.Grant, column string, amount any, extra map[string]any) (int64, error) {
	return b.incrementOrDecrement(ctx, g, column, amount, extra, "-")
}

// incrementOrDecrement is the shared body of Increment and Decrement.
//
// The amount goes into the SQL rather than into a binding, because it is on
// the right of an assignment to the column itself. That is why a
// non-numeric amount is refused: one that came from a request would
// otherwise be SQL.
func (b *Builder[T]) incrementOrDecrement(ctx context.Context, g auth.Grant, column string, amount any, extra map[string]any, sign string) (int64, error) {
	if amount == nil {
		amount = 1
	}
	if !isNumeric(reflect.ValueOf(amount).Kind()) {
		return 0, fmt.Errorf("model: non-numeric value passed to increment or decrement on %s.%s: the amount is compiled into the statement, so it cannot come from anywhere a value can", b.model.GetTable(), column)
	}

	values := copyMap(extra)
	wrapped := b.model.Grammar.Wrap(column)
	values[column] = query.Raw(fmt.Sprintf("%s %s %v", wrapped, sign, amount))
	return b.Update(ctx, g, values)
}

// Delete removes the rows matching the query, or runs the registered
// onDelete callback instead.
//
// A model that soft deletes has an onDelete callback on its builder, so this
// runs the update the SoftDeletingScope registered instead of a delete.
func (b *Builder[T]) Delete(ctx context.Context, g auth.Grant) (int64, error) {
	if b.onDelete != nil {
		return b.onDelete(ctx, b, g)
	}
	prepared, err := b.prepare(g)
	if err != nil {
		return 0, err
	}
	return prepared.runDelete(ctx)
}

// ForceDelete removes the row, whatever the scope would have done.
//
// It still applies the tenant filter. "Force" is about the soft delete, never
// about the tenant.
func (b *Builder[T]) ForceDelete(ctx context.Context, g auth.Grant) (int64, error) {
	// The held error is read here rather than in prepare, because this is one of
	// the two methods that does not go through prepare. Without this line a
	// builder invalidated by WithTrashed on a model that does not soft delete,
	// or by a scope name nothing registered, still issues the DELETE -- against
	// the contract this file states about itself, which is that nothing is ever
	// executed with an error waiting.
	if b.err != nil {
		return 0, b.err
	}

	tenant := auth.Tenant(g)
	if tenant == "" || !auth.ValidTenant(tenant) {
		return 0, ErrNoTenant
	}
	forced := b.clone()
	if err := forced.scopeToTenant(g, tenant); err != nil {
		return 0, err
	}
	return forced.runDelete(ctx)
}

// OnDelete registers callback as the override Delete runs instead of a
// plain delete.
func (b *Builder[T]) OnDelete(callback func(context.Context, *Builder[T], auth.Grant) (int64, error)) *Builder[T] {
	b.onDelete = callback
	return b
}

// Touch sets column, or the model's updated-at column when none is given, to
// the current time on every row matching the query.
func (b *Builder[T]) Touch(ctx context.Context, g auth.Grant, column ...string) (int64, error) {
	name := b.model.GetUpdatedAtColumn()
	if len(column) > 0 && column[0] != "" {
		name = column[0]
	} else if !b.model.UsesTimestamps() || name == "" {
		return 0, nil
	}
	return b.Update(ctx, g, map[string]any{name: b.model.FreshTimestamp()})
}

// With marks relations to eager load.
func (b *Builder[T]) With(relations ...string) *Builder[T] {
	for _, relation := range relations {
		if relation == "" {
			continue
		}
		b.eagerLoad[relation] = nil
	}
	return b
}

// WithConstraints marks relation to eager load, constrained by callback.
//
// The callback takes a query.Builder rather than a Builder[T], because the
// relation it constrains belongs to another model type and this signature
// has no type parameter to spell that other model's builder with.
func (b *Builder[T]) WithConstraints(relation string, constraints func(*query.Builder)) *Builder[T] {
	b.eagerLoad[relation] = constraints
	return b
}

// WithOnly replaces the eager load list with relations.
func (b *Builder[T]) WithOnly(relations ...string) *Builder[T] {
	b.eagerLoad = map[string]func(*query.Builder){}
	return b.With(relations...)
}

// Without removes relations from the eager load list.
func (b *Builder[T]) Without(relations ...string) *Builder[T] {
	for _, relation := range relations {
		delete(b.eagerLoad, relation)
	}
	return b
}

// GetEagerLoads returns the names marked to eager load, sorted.
func (b *Builder[T]) GetEagerLoads() []string {
	out := make([]string, 0, len(b.eagerLoad))
	for name := range b.eagerLoad {
		out = append(out, name)
	}
	slices.Sort(out)
	return out
}

// WithoutEagerLoads clears the eager load list.
func (b *Builder[T]) WithoutEagerLoads() *Builder[T] {
	b.eagerLoad = map[string]func(*query.Builder){}
	return b
}

// WithoutEagerLoad removes these relations, and the ones nested under them,
// from the eager load list.
func (b *Builder[T]) WithoutEagerLoad(relations ...string) *Builder[T] {
	for name := range b.eagerLoad {
		for _, relation := range relations {
			if name == relation || strings.HasPrefix(name, relation+".") {
				delete(b.eagerLoad, name)
			}
		}
	}
	return b
}

// SetEagerLoads replaces the eager load list with relations.
func (b *Builder[T]) SetEagerLoads(relations ...string) *Builder[T] {
	b.eagerLoad = map[string]func(*query.Builder){}
	return b.With(relations...)
}

// WithoutGlobalScopesExcept removes every registered scope except the named
// ones.
func (b *Builder[T]) WithoutGlobalScopesExcept(identifiers ...string) *Builder[T] {
	for identifier := range b.scopes {
		if !slices.Contains(identifiers, identifier) {
			b.WithoutGlobalScope(identifier)
		}
	}
	return b
}

// FindSole returns the row with the given primary key, and fails unless it
// is the only one.
func (b *Builder[T]) FindSole(ctx context.Context, g auth.Grant, id any, columns ...any) (*T, error) {
	return b.WhereKey(id).Sole(ctx, g, columns...)
}

// TouchQuietly touches the query's rows without firing model events.
func (b *Builder[T]) TouchQuietly(ctx context.Context, g auth.Grant, column ...string) (touched int64, err error) {
	return touched, b.model.WithoutEvents(func() error {
		touched, err = b.Touch(ctx, g, column...)
		return err
	})
}

// EagerLoadRelations loads every top-level relation marked with With, and
// attaches the matches to the model behind each row.
//
// A row reaches its model only when T embeds Model[T], so rows that carry none
// are ErrRowHasNoModel: attaching a relation to nothing and reporting success
// would be a load the next line cannot read back.
func (b *Builder[T]) EagerLoadRelations(ctx context.Context, g auth.Grant, rows Collection[T]) error {
	found, err := rows.modelsOrFail()
	if err != nil {
		return err
	}
	return b.eagerLoadRelations(ctx, g, found)
}

// eagerLoadRelations is EagerLoadRelations over the models, which is what it
// needs: a relation is attached to the model, and the row is only where the
// columns are.
//
// The loading itself is relations.EagerLoadRelation, which is four calls and is
// where eager loading is implemented. Writing it again here would be the second
// implementation of the one thing that turns N queries into two, and the two
// would drift on the case neither author tested -- the morph-to that matches
// its own parents, the relation whose local key is not the primary key.
//
// The models are handed over as refs, which is the same pointer seen through the
// interface a relation consumes: what the relation sets is set on the model the
// caller holds.
func (b *Builder[T]) eagerLoadRelations(ctx context.Context, g auth.Grant, models models[T]) error {
	if len(models) == 0 {
		return nil
	}

	refs := make([]relations.Model, 0, len(models))
	for _, model := range models {
		refs = append(refs, model.Ref())
	}

	for _, name := range b.GetEagerLoads() {
		if strings.Contains(name, ".") {
			// A nested eager load is loaded by the query that fetches its parent,
			// which is where the models it hangs off are hydrated.
			continue
		}
		relation, err := b.GetRelationWithoutConstraints(name)
		if err != nil {
			return err
		}
		if _, err := relations.EagerLoadRelation(ctx, g, refs, name, relation, onRelationQuery(b.eagerLoad[name])); err != nil {
			return err
		}
	}
	return nil
}

// onRelationQuery adapts what With recorded to what the eager loader applies.
//
// WithConstraints takes func(*query.Builder) because a caller narrowing an
// eager load is writing wheres, and the loader hands the relation itself so that
// a constraint could reach more than its query. This is the step between, and it
// is applied after the batch's own `in (...)` for the reason the loader
// documents.
func onRelationQuery(callback func(*query.Builder)) func(relations.Relation) {
	if callback == nil {
		return nil
	}
	return func(rel relations.Relation) { callback(rel.GetQuery().GetQuery()) }
}

// Limit sets the row limit, forwarded to the underlying query.
func (b *Builder[T]) Limit(value int) *Builder[T] {
	b.query.Limit(value)
	return b
}

// Offset sets the row offset, forwarded to the underlying query.
func (b *Builder[T]) Offset(value int) *Builder[T] {
	b.query.Offset(value)
	return b
}

// ForPage sets the limit and offset for page, perPage rows at a time.
func (b *Builder[T]) ForPage(page, perPage int) *Builder[T] {
	b.query.ForPage(page, perPage)
	return b
}

// GetLimit returns the row limit, or nil when none is set.
func (b *Builder[T]) GetLimit() *int { return b.query.GetLimit() }

// GetOffset returns the row offset, or nil when none is set.
func (b *Builder[T]) GetOffset() *int { return b.query.GetOffset() }

// OrderBy adds an ordering, forwarded to the underlying query.
func (b *Builder[T]) OrderBy(column any, direction ...string) *Builder[T] {
	b.query.OrderBy(column, direction...)
	return b
}

// OrderByDesc adds a descending ordering, forwarded to the underlying query.
func (b *Builder[T]) OrderByDesc(column any) *Builder[T] {
	b.query.OrderByDesc(column)
	return b
}

// Select sets the columns to return, forwarded to the underlying query.
func (b *Builder[T]) Select(columns ...any) *Builder[T] {
	b.query.Select(columns...)
	return b
}

// AddSelect adds columns to the ones already selected.
func (b *Builder[T]) AddSelect(columns ...any) *Builder[T] {
	b.query.AddSelect(columns...)
	return b
}

// SelectRaw adds a raw select expression, forwarded to the underlying query.
func (b *Builder[T]) SelectRaw(expression string, bindings ...any) *Builder[T] {
	b.query.SelectRaw(expression, bindings...)
	return b
}

// enforceOrderBy adds an ascending order by the model's key when the query
// has none: chunking without an order is chunking over a set the engine may
// return in a different order each time.
func (b *Builder[T]) enforceOrderBy() {
	if len(b.query.Orders) == 0 && len(b.query.UnionOrders) == 0 {
		b.OrderBy(b.model.GetQualifiedKeyName(), "asc")
	}
}

// DefaultKeyName returns the model's primary key name.
func (b *Builder[T]) DefaultKeyName() string { return b.model.GetKeyName() }

// Chunk walks the query count rows at a time, calling callback for each
// chunk.
//
// The callback stops the walk by returning false, and stops it with a
// reason by returning an error.
func (b *Builder[T]) Chunk(ctx context.Context, g auth.Grant, count int, callback func(Collection[T], int) (bool, error)) error {
	if count < 1 {
		return fmt.Errorf("model: the chunk size should be at least 1")
	}
	b.enforceOrderBy()

	skip := 0
	if offset := b.GetOffset(); offset != nil {
		skip = *offset
	}
	remaining := -1
	if limit := b.GetLimit(); limit != nil {
		remaining = *limit
	}

	for page := 1; ; page++ {
		size := count
		if remaining >= 0 {
			size = min(count, remaining)
		}
		if size == 0 {
			return nil
		}

		results, err := b.clone().Offset((page-1)*count+skip).Limit(size).Get(ctx, g)
		if err != nil {
			return err
		}
		if len(results) == 0 {
			return nil
		}
		if remaining >= 0 {
			remaining = max(remaining-len(results), 0)
		}
		keepGoing, err := callback(results, page)
		if err != nil {
			return err
		}
		if !keepGoing || len(results) != count {
			return nil
		}
	}
}

// Each walks the query count rows at a time, calling callback for every row
// with its overall index.
func (b *Builder[T]) Each(ctx context.Context, g auth.Grant, count int, callback func(*T, int) (bool, error)) error {
	return b.Chunk(ctx, g, count, func(models Collection[T], page int) (bool, error) {
		for i, model := range models {
			keepGoing, err := callback(model, (page-1)*count+i)
			if err != nil || !keepGoing {
				return false, err
			}
		}
		return true, nil
	})
}

// ChunkById walks the query count rows at a time, ordered and paged by the
// key rather than by offset, so that a row inserted between two chunks
// cannot shift the window and hide a row.
func (b *Builder[T]) ChunkById(ctx context.Context, g auth.Grant, count int, callback func(Collection[T], int) (bool, error), column ...string) error {
	if count < 1 {
		return fmt.Errorf("model: the chunk size should be at least 1")
	}
	name := b.DefaultKeyName()
	if len(column) > 0 && column[0] != "" {
		name = column[0]
	}

	var lastID any
	for page := 1; ; page++ {
		q := b.clone()
		if lastID != nil {
			q.Where(b.model.QualifyColumn(name), ">", lastID)
		}
		found, err := q.OrderBy(b.model.QualifyColumn(name), "asc").Limit(count).get(ctx, g)
		if err != nil {
			return err
		}
		if len(found) == 0 {
			return nil
		}
		keepGoing, err := callback(q.results(found), page)
		if err != nil {
			return err
		}
		if !keepGoing {
			return nil
		}
		// The key the next page resumes from is read off the model rather than
		// off the row: a chunk walks whatever T is, and only a T that embeds
		// Model[T] could be asked for a column.
		lastID = found[len(found)-1].GetAttribute(name)
		if lastID == nil {
			return fmt.Errorf("model: chunkById stopped because %s is not in the result", name)
		}
		if len(found) != count {
			return nil
		}
	}
}

// Lazy returns an iterator over the rows, fetched a chunk at a time.
//
// It returns a range-over-func iterator directly. The error the walk
// stopped on is reported through the pointer the caller passes, because an
// iterator has nowhere else to put one.
func (b *Builder[T]) Lazy(ctx context.Context, g auth.Grant, chunkSize int, err *error) func(func(*T) bool) {
	return func(yield func(*T) bool) {
		stopped := false
		chunkErr := b.Chunk(ctx, g, chunkSize, func(models Collection[T], _ int) (bool, error) {
			for _, model := range models {
				if !yield(model) {
					stopped = true
					return false, nil
				}
			}
			return true, nil
		})
		_ = stopped
		if chunkErr != nil && err != nil {
			*err = chunkErr
		}
	}
}

// LazyById returns an iterator over the rows, fetched a chunk at a time and
// paged by the key rather than by offset.
func (b *Builder[T]) LazyById(ctx context.Context, g auth.Grant, chunkSize int, err *error, column ...string) func(func(*T) bool) {
	return func(yield func(*T) bool) {
		chunkErr := b.ChunkById(ctx, g, chunkSize, func(models Collection[T], _ int) (bool, error) {
			for _, model := range models {
				if !yield(model) {
					return false, nil
				}
			}
			return true, nil
		}, column...)
		if chunkErr != nil && err != nil {
			*err = chunkErr
		}
	}
}

// Cursor walks the result one model at a time.
//
// query.Connection hands back the rows it read, so this is a walk over one
// result set rather than a second way to run a query -- and it stays lazy from
// the caller's side, which is what the method is for.
func (b *Builder[T]) Cursor(ctx context.Context, g auth.Grant, err *error) func(func(*T) bool) {
	return func(yield func(*T) bool) {
		rows, getErr := b.Get(ctx, g)
		if getErr != nil {
			if err != nil {
				*err = getErr
			}
			return
		}
		for _, row := range rows {
			if !yield(row) {
				return
			}
		}
	}
}

// cursor is Cursor with the models still in hand, and without the after-query
// callbacks. See get.
func (b *Builder[T]) cursor(ctx context.Context, g auth.Grant, err *error) func(func(*Model[T]) bool) {
	return func(yield func(*Model[T]) bool) {
		found, getErr := b.get(ctx, g)
		if getErr != nil {
			if err != nil {
				*err = getErr
			}
			return
		}
		for _, model := range found {
			if !yield(model) {
				return
			}
		}
	}
}

func uniqueValues(values []any) []any {
	out := make([]any, 0, len(values))
	for _, value := range values {
		if !slices.ContainsFunc(out, func(seen any) bool { return seen == value }) {
			out = append(out, value)
		}
	}
	return out
}

func mergeMaps(maps ...map[string]any) map[string]any {
	out := map[string]any{}
	for _, m := range maps {
		for key, value := range m {
			out[key] = value
		}
	}
	return out
}

func afterLastDot(column string) string {
	if i := strings.LastIndex(column, "."); i >= 0 {
		return column[i+1:]
	}
	return column
}

func toInt64(value any) int64 {
	switch v := value.(type) {
	case int64:
		return v
	case int:
		return int64(v)
	case int32:
		return int64(v)
	case float64:
		return int64(v)
	case float32:
		return int64(v)
	case []byte:
		return parseInt64(string(v))
	case string:
		return parseInt64(v)
	}
	return 0
}

func parseInt64(s string) int64 {
	var out int64
	negative := false
	for i, c := range s {
		if i == 0 && c == '-' {
			negative = true
			continue
		}
		if c < '0' || c > '9' {
			break
		}
		out = out*10 + int64(c-'0')
	}
	if negative {
		return -out
	}
	return out
}
