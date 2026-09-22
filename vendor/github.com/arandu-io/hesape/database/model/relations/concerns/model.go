package concerns

import (
	"context"
	"fmt"
	"iter"
	"strings"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database/query"
	"github.com/arandu-io/hesape/pagination"
)

// Model is what a relation asks of a model.
//
// The surface a relation actually touches is written out as an interface, and it
// is declared here rather than in the model package that implements it,
// because an interface belongs with its consumer. The concrete model embeds the
// concerns in database/model/concerns and satisfies this by doing so.
//
// It lives in this package, and not in relations, because Go forbids a
// subpackage from importing its parent: relations, relations/concerns and every
// relation type have to agree on one Model, and the only place all three can
// see is the leaf. relations aliases it back (type Model = concerns.Model), so
// relations.Model and concerns.Model are the same type.
type Model interface {
	// GetTable answers Model::getTable.
	GetTable() string

	// SetTable answers Model::setTable. A self-join renames the related model's
	// table to the join alias, which is why this is on the interface.
	SetTable(table string)

	// QualifyColumn answers Model::qualifyColumn: the column prefixed with the
	// table, unless it already carries one.
	QualifyColumn(column string) string

	// GetKeyName answers Model::getKeyName.
	GetKeyName() string

	// GetKeyType answers Model::getKeyType: "int" or "string".
	GetKeyType() string

	// GetKey answers Model::getKey.
	GetKey() any

	// GetForeignKey answers Model::getForeignKey: the conventional foreign key
	// pointing at this model, snake_case singular plus the key name.
	GetForeignKey() string

	// GetMorphClass answers Model::getMorphClass.
	//
	// In PHP it defaults to the class name and the morph map is optional. There
	// is no class name at run time here, so it is the alias this model was
	// registered under -- see the relations package doc for why that is the
	// better trade rather than a regrettable one.
	GetMorphClass() string

	// Exists answers Model::$exists. A property in PHP; a method here, because
	// the field name is taken by nothing and the value is read, never assigned,
	// by a relation.
	Exists() bool

	// GetAttribute answers Model::getAttribute.
	GetAttribute(key string) any

	// SetAttribute answers Model::setAttribute.
	SetAttribute(key string, value any)

	// GetAttributes answers Model::getAttributes.
	GetAttributes() map[string]any

	// SetRawAttributes answers Model::setRawAttributes.
	SetRawAttributes(attributes map[string]any, sync bool)

	// UnsetAttribute answers PHP's unset($model->$key), which BelongsToMany
	// performs on every pivot_ column it lifts off the related model.
	UnsetAttribute(key string)

	// GetRelation answers Model::getRelation, with the second result reporting
	// whether the relation was loaded, where PHP throws.
	GetRelation(relation string) (any, bool)

	// SetRelation answers Model::setRelation.
	SetRelation(relation string, value any)

	// RelationLoaded answers Model::relationLoaded.
	RelationLoaded(relation string) bool

	// UnsetRelation answers Model::unsetRelation.
	UnsetRelation(relation string)

	// IsRelation answers Model::isRelation: whether the name is one of this
	// model's relations. Chaperone asks before it writes an inverse nobody
	// declared.
	IsRelation(key string) bool

	// Touches answers Model::touches.
	Touches(relation string) bool

	// Touch answers Model::touch.
	Touch(ctx context.Context, g auth.Grant) error

	// NewInstance answers Model::newInstance.
	NewInstance(attributes map[string]any) Model

	// Fill answers Model::fill: mass assignment, through the guard.
	Fill(attributes map[string]any)

	// ForceFill answers Model::forceFill: mass assignment, past the guard.
	ForceFill(attributes map[string]any)

	// WasRecentlyCreated answers Model::$wasRecentlyCreated, which
	// updateOrCreate reads to tell a create from a find.
	WasRecentlyCreated() bool

	// WithoutEvents answers Model::withoutEvents.
	//
	// Static in PHP, a method here: Go has no late static binding, so the only
	// way to reach the model's own event registry is through the model. It is
	// what the Quietly variants of save and create are made of.
	WithoutEvents(callback func() error) error

	// Delete answers Model::delete, returning rows affected where the PHP
	// returns bool|null.
	Delete(ctx context.Context, g auth.Grant) (int64, error)

	// NewQuery answers Model::newQuery.
	NewQuery() Builder

	// GetCreatedAtColumn answers Model::getCreatedAtColumn.
	GetCreatedAtColumn() string

	// GetUpdatedAtColumn answers Model::getUpdatedAtColumn.
	GetUpdatedAtColumn() string

	// UsesTimestamps answers Model::usesTimestamps.
	UsesTimestamps() bool

	// FreshTimestamp answers Model::freshTimestamp.
	FreshTimestamp() time.Time

	// Save writes the model.
	//
	// It carries the context and the Grant that every write in this collection
	// carries: a relation that could save without one would be a write nobody
	// authorized, on the side where it is easiest to spot and therefore least
	// excusable to leave open.
	Save(ctx context.Context, g auth.Grant) error
}

// Builder is what a relation asks of a model's query builder.
//
// Narrow on purpose: it is the set of methods the sixteen relation types call,
// not the whole typed builder. The concrete builder lives in the model
// package and satisfies this; declaring the contract here is what keeps
// relations from importing it and closing a cycle.
//
// Every method that reaches the database takes a context and an auth.Grant.
// That is not decoration. A relation is a read path, List and Find and Get and
// Paginate are read paths, and a read path without a Policy is a tenant leak
// with a technical name -- so the Grant is in the signature, where forgetting it
// does not compile.
type Builder interface {
	// GetModel answers Builder::getModel.
	GetModel() Model

	// GetQuery answers Builder::getQuery: the base query builder underneath.
	GetQuery() *query.Builder

	// Select answers Builder::select.
	Select(columns ...any) Builder

	// AddSelect answers Builder::addSelect.
	AddSelect(columns ...any) Builder

	// Where answers Builder::where.
	Where(column any, args ...any) Builder

	// WhereIn answers Builder::whereIn.
	WhereIn(column any, values []any) Builder

	// WhereNotNull answers Builder::whereNotNull.
	WhereNotNull(columns ...any) Builder

	// WhereColumn answers Builder::whereColumn.
	WhereColumn(first any, args ...any) Builder

	// WhereKey answers Builder::whereKey.
	WhereKey(ids ...any) Builder

	// Join answers Builder::join.
	Join(table any, first any, args ...any) Builder

	// GroupBy answers Builder::groupBy.
	GroupBy(groups ...any) Builder

	// SelectRaw answers Builder::selectRaw.
	SelectRaw(expression string, bindings ...any) Builder

	// OrderBy answers Builder::orderBy.
	OrderBy(column any, direction ...string) Builder

	// Limit answers Builder::limit.
	Limit(value int) Builder

	// Offset answers Builder::offset.
	//
	// It is here for chunking. BelongsToMany::chunk and
	// HasOneOrManyThrough::chunk hand the prepared query to the builder's own
	// chunk, which walks it with forPage -- an offset and a limit per page.
	Offset(value int) Builder

	// Cursor answers Builder::cursor: the rows streamed one at a time rather
	// than gathered into a slice.
	//
	// The error arrives beside each value, as iter.Seq2 does throughout this
	// collection, because a stream can fail after it has already yielded rows
	// and a signature that returned the error at the end would be read after
	// the caller had acted on half the data.
	Cursor(ctx context.Context, g auth.Grant) iter.Seq2[Model, error]

	// Paginate answers Builder::paginate.
	Paginate(ctx context.Context, g auth.Grant, perPage, page int, opts pagination.Options, columns ...any) (*pagination.LengthAwarePaginator[Model], error)

	// SimplePaginate answers Builder::simplePaginate.
	SimplePaginate(ctx context.Context, g auth.Grant, perPage, page int, opts pagination.Options, columns ...any) (*pagination.Paginator[Model], error)

	// CursorPaginate answers Builder::cursorPaginate.
	CursorPaginate(ctx context.Context, g auth.Grant, perPage int, cursor *pagination.Cursor, opts pagination.Options, columns ...any) (*pagination.CursorPaginator[Model], error)

	// Get answers Builder::get.
	Get(ctx context.Context, g auth.Grant) ([]Model, error)

	// First answers Builder::first. A miss is (nil, nil), where the PHP returns
	// null; ErrNotFound belongs to firstOrFail, which is a different question.
	First(ctx context.Context, g auth.Grant) (Model, error)

	// Find answers Builder::find. A miss is (nil, nil), as it is for First.
	Find(ctx context.Context, g auth.Grant, id any) (Model, error)

	// Insert answers Builder::insert.
	Insert(ctx context.Context, g auth.Grant, values []map[string]any) error

	// Update answers Builder::update, returning rows affected.
	Update(ctx context.Context, g auth.Grant, values map[string]any) (int64, error)

	// Upsert answers Builder::upsert, returning rows affected.
	//
	// It is here because HasOneOrMany::upsert and MorphOneOrMany::upsert stamp
	// the foreign key -- and the morph type -- onto every row and then hand the
	// batch straight to the builder. Doing that one row at a time through Insert
	// would turn one statement into N, which is the opposite of what a caller
	// reaches for upsert to get.
	Upsert(ctx context.Context, g auth.Grant, values []map[string]any, uniqueBy, update []string) (int64, error)

	// Delete answers Builder::delete, returning rows affected.
	Delete(ctx context.Context, g auth.Grant) (int64, error)

	// Clone answers PHP's `clone $builder`.
	Clone() Builder
}

// TenantColumn is the column a tenant-scoped table carries.
//
// One name, not a setting: every query this package emits filters on it, and a
// per-model column name that could be misspelled is a filter that silently
// matches nothing -- or, worse, a filter the grammar drops.
const TenantColumn = "tenant_id"

// Tenanted is how a model says its rows are not partitioned by tenant.
//
// Almost nothing should implement it. A currency table, a country list, a
// shared taxonomy -- rows that are the same for every customer of the system --
// return the empty string and are read unfiltered. Everything else does not
// implement it at all and is filtered on TenantColumn.
//
// The shape is deliberate: opting out is a method somebody wrote, with a name
// that greps, in the model's own file. The alternative -- a column name that
// defaults to empty -- makes the leak the default and the safety the thing that
// has to be remembered.
type Tenanted interface {
	// GetTenantColumn answers the column this model is partitioned by, or the
	// empty string for a table that is shared across tenants.
	GetTenantColumn() string
}

// TenantColumnFor answers the column m is partitioned by.
func TenantColumnFor(m Model) string {
	if t, ok := m.(Tenanted); ok {
		return t.GetTenantColumn()
	}
	return TenantColumn
}

// OwnTenantScoper is how a Builder says it puts the tenant filter on its own
// table itself, on every statement it runs.
//
// ScopeTenant asks before filtering, and a builder that stays silent is
// filtered. That direction is the whole point: the answer that requires no code
// is the safe one, so a builder nobody thought about is scoped rather than
// trusted.
//
// A builder that answers true is promising something the reader cannot see from
// here, so the promise has to be paid for by the builder's own tests: that every
// terminal it offers carries the filter, not just the one the author had in
// mind. What it buys is a statement whose tenant clause appears once -- a second
// identical clause is not wrong, but every reader after has to prove it
// redundant before they can move on, and the one who decides it is the copy to
// delete may delete the other one.
type OwnTenantScoper interface {
	// ScopesOwnTableByTenant reports whether every statement this builder runs
	// already filters its own table by the tenant of the Grant it was given.
	//
	// It says nothing about the tables the query joins: those carry no model to
	// ask and no builder answers for them, which is why ScopeTenant filters them
	// whatever this returns.
	ScopesOwnTableByTenant() bool
}

// RequireTenant answers the tenant carried by g, and refuses a Grant that has
// none.
//
// The zero Grant carries the empty tenant. Filtering on it would compile to
// `tenant_id = ”`, which matches nothing on the read side and writes an
// unreachable row on the write side -- a bug that looks like an empty result
// set and gets debugged as a missing fixture. Refusing is louder and is the
// same answer auth.Grant.Check gives: this is not authorized.
func RequireTenant(g auth.Grant) (string, error) {
	tenant := auth.Tenant(g)
	if tenant == "" {
		return "", fmt.Errorf("%w: this relation was loaded with a grant that carries no tenant. The tenant comes from the Grant (RULE 14), so an empty one means nothing authorized this read", auth.ErrForbidden)
	}
	return tenant, nil
}

// ScopeTenant returns b filtered by the tenant g carries -- on the model's own
// table, and on every table the query joins.
//
// Every relation calls it on the way to the database, on the eager path exactly
// as on the lazy one. The eager path is the one that matters: a with() whose
// parent query is correctly scoped and whose child query is not returns the
// right parents carrying another customer's children, and nothing in the result
// looks wrong.
//
// # What leaked
//
// It used to filter the model's table and nothing else, and a join contributes
// no filter of its own: query.Builder.scoped qualifies the tenant column on the
// table in the from clause, so every table brought in by a join was read whole.
// An audit found the same hole in three relations at once, and the one that
// proved it was BelongsToMany.Get: `select roles.*, role_user.user_id as
// pivot_user_id from roles inner join role_user on roles.id = role_user.role_id
// where role_user.user_id = 1 and roles.tenant_id = 'acme'` matched the pivot
// row (user_id 1, role_id 'admin', tenant_id 'B') -- another customer's grant --
// and handed this customer's user the role, with B's pivot columns hydrated onto
// it. HasOneOrManyThrough.performJoin had it on the intermediate table.
//
// Patching those two would have left the door open for the third relation to
// join a table, so the filter goes on here, where every read already passes: a
// table this query joins is filtered on TenantColumn, whoever joined it.
//
// # Why a where and not a join condition
//
// `on role_user.tenant_id = ?` and `where role_user.tenant_id = ?` select the
// same rows of an inner join, and every join in this package is an inner join.
// The where is the one that can be added here: a join clause hands its bindings
// to the parent's join segment when the join is declared (see addJoinClause), so
// a condition added afterwards would compile into the statement with its value
// sitting in another join's placeholder.
//
// # The own table, and who filters it
//
// The joined tables are filtered here always. The relation's own table is
// filtered here only when the builder does not already do it, which it says by
// implementing OwnTenantScoper -- and the read that reaches a real model does,
// so its statement names the tenant once instead of twice.
//
// The question is asked of the builder rather than answered by a name check on
// the wheres, and that is the part that matters. A query already carrying
// `posts.tenant_id = 'other'` because the caller wrote that where by hand looks
// exactly like a query somebody scoped, and a filter skipped on that evidence is
// a filter the caller chose the value of.
func ScopeTenant(b Builder, m Model, g auth.Grant) (Builder, error) {
	tenant, err := RequireTenant(g)
	if err != nil {
		return nil, err
	}

	b = scopeJoinedTables(b, tenant)

	if scoper, ok := b.(OwnTenantScoper); ok && scoper.ScopesOwnTableByTenant() {
		return b, nil
	}

	column := TenantColumnFor(m)
	if column == "" {
		return b, nil
	}
	return b.Where(m.QualifyColumn(column), tenant), nil
}

// scopeJoinedTables filters every table b joins on TenantColumn.
//
// A derived table -- the `(select ...) as alias` a subquery join compiles to --
// is skipped: it has no tenant column of its own, and the query inside it is
// scoped where it was built, by query.Builder's subquery pass.
//
// A table that already carries the filter is skipped too, so that scoping a
// query twice does not write the clause twice.
//
// Everything else is filtered, without asking whether the table is partitioned:
// there is no model behind a joined table to ask, an intermediate table is the
// one thing in this package that is known to be partitioned -- newPivotStatement
// has always assumed it on the write side -- and the two failures are not
// comparable. A shared table joined here yields a statement the engine refuses
// by name, which is a build that stops; the filter left off yields a read that
// crosses customers and looks right. Tenanted opts a model's own table out, and
// a joined table that has to be opted out belongs beside it.
func scopeJoinedTables(b Builder, tenant string) Builder {
	base := b.GetQuery()
	if base == nil {
		return b
	}

	for _, join := range base.Joins {
		table, ok := joinedTableName(join.Table)
		if !ok {
			continue
		}
		column := table + "." + TenantColumn
		if hasFilterOn(base, column) {
			continue
		}
		b = b.Where(column, tenant)
	}
	return b
}

// joinedTableName answers the name a joined table's columns are qualified by,
// and whether there is one at all.
//
// The alias wins when the join declares one -- `users as arandu_reserved_0` is
// referred to by the alias and the real name resolves to nothing. An expression
// is not a table name and reports false.
func joinedTableName(table any) (string, bool) {
	name, ok := table.(string)
	if !ok {
		return "", false
	}

	name = strings.TrimSpace(name)
	if index := strings.Index(strings.ToLower(name), " as "); index >= 0 {
		name = strings.TrimSpace(name[index+len(" as "):])
	}
	if name == "" {
		return "", false
	}
	return name, true
}

// hasFilterOn reports whether the query already compares column with a value.
func hasFilterOn(q *query.Builder, column string) bool {
	for _, where := range q.Wheres {
		if where.Type == "Basic" && fmt.Sprint(where.Column) == column {
			return true
		}
	}
	return false
}

// ScopeTenantQuery is ScopeTenant for a base query builder.
//
// The pivot table is reached with the base builder rather than a typed one,
// so the intermediate table needs its own scoping call. It needs it just as
// much: a pivot row is what says this customer's user has that customer's role.
func ScopeTenantQuery(q *query.Builder, table string, g auth.Grant) (*query.Builder, error) {
	tenant, err := RequireTenant(g)
	if err != nil {
		return nil, err
	}
	return q.Where(table+"."+TenantColumn, tenant), nil
}
