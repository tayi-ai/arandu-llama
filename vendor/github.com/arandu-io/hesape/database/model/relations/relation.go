package relations

import (
	"context"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database/model/relations/concerns"
	"github.com/arandu-io/hesape/database/query"
)

// Model is what a relation asks of a model.
//
// It is an alias rather than a declaration: the type is defined in
// relations/concerns, because Go forbids a subpackage from importing its
// parent and the traits in there need the same one. Aliasing it back means
// relations.Model is that type, not a second one that looks like it.
type Model = concerns.Model

// Builder is the query a relation narrows and then runs: the interface of
// selects, wheres, joins and orders every relation in this package builds on
// top of, with a context and an auth.Grant on every method that reaches the
// database.
//
// It is an alias rather than a declaration, for the reason Model is: the
// interface is defined in relations/concerns because the traits there need the
// same one and Go forbids a subpackage from importing its parent. Aliasing it
// back means relations.Builder is that interface, not a second one shaped like
// it.
type Builder = concerns.Builder

// ErrMultipleRecordsFound is returned by the singular reads on a relation --
// Sole, and the pivot lookups that expect one row -- when the query matched
// more than one.
//
// It means the relation is not as unique as the caller assumed: a has-one
// pointing at a column with duplicates, or a pivot row that exists twice. The
// wrapping message carries the count and the table.
var ErrMultipleRecordsFound = errors.New("relations: more than one record matched a query that expected one")

// ErrModelNotFound is returned by the failing reads on a relation -- the ones
// that must produce a row -- when the query matched none.
//
// It is the sentinel a handler turns into a 404. The reads that tolerate a miss
// answer nil instead, so seeing this error means the caller asked for a row
// that has to exist. The wrapping message carries the table, and the key when
// the lookup had one.
var ErrModelNotFound = errors.New("relations: no query results for the model")

// Relation is the contract every relation type implements.
//
// The contract is here and the shared body is BaseRelation, the same division
// the query package makes between Grammar and BaseGrammar.
//
// The four methods that matter are AddEagerConstraints, InitRelation, Match and
// GetEager. They are what turns N+1 queries into two, and they are worth
// reading in that order: one query for the parents, one for every child of
// every parent, and a dictionary in memory to put them together.
type Relation interface {
	// AddConstraints answers Relation::addConstraints: the where clauses that
	// narrow the relation to one parent.
	AddConstraints()

	// AddEagerConstraints answers Relation::addEagerConstraints: the same
	// narrowing, widened to every parent at once. This is the method that
	// replaces N queries with one.
	AddEagerConstraints(models []Model) error

	// InitRelation answers Relation::initRelation: it seeds every parent with
	// an empty value, so that a parent with no children answers an empty
	// collection rather than a lazy load.
	InitRelation(models []Model, relation string) []Model

	// Match answers Relation::match: it distributes one flat result set over
	// the parents it belongs to, in memory, with no further queries.
	Match(models []Model, results []Model, relation string) ([]Model, error)

	// GetResults answers Relation::getResults: the relation as the developer
	// reads it -- one model for a has-one, many for a has-many.
	GetResults(ctx context.Context, g auth.Grant) (any, error)

	// GetEager answers Relation::getEager: the results for an eager load.
	GetEager(ctx context.Context, g auth.Grant) ([]Model, error)

	// GetQuery answers Relation::getQuery.
	GetQuery() Builder

	// GetParent answers Relation::getParent.
	GetParent() Model

	// GetRelated answers Relation::getRelated.
	GetRelated() Model
}

// The relation constraints used to be a process-wide flag here, switched off
// around a constructor by NoConstraints so that an eager load could build a
// relation whose query was not already narrowed to one parent.
//
// It is gone, and what replaced it is a second constructor per relation --
// NewHasManyUnconstrained beside NewHasMany. The flag was correct in PHP, where
// one request is one process; here a relation built on another goroutine while
// the flag was down came back unconstrained, which is every parent's children
// in a statement nobody could tell from the right one. Being atomic, the race
// detector could never see it.
//
// The call site says which it wants now, and there is no flag to leave down.

// selfJoinCount answers Relation::$selfJoinCount.
var selfJoinCount struct {
	sync.Mutex
	value int
}

// BaseRelation answers the body of the abstract Relation class: everything it
// implements rather than declares.
//
// A relation embeds it and supplies the five abstract methods. It is exported
// for the same reason query.BaseGrammar is: embedding is how Go says "extends"
// for the half that is code rather than contract.
//
// # There is no virtual dispatch, so the constructor does not call AddConstraints
//
// The PHP constructor ends with $this->addConstraints(), and late binding sends
// it to the subclass. Embedding in Go does not: a call from here would reach
// BaseRelation's own method and quietly build an unconstrained relation. So
// each concrete constructor calls its own AddConstraints as its last statement,
// and that is the one line of the PHP shape that had to move.
type BaseRelation struct {
	// Query is Relation::$query.
	Query Builder

	// Parent is Relation::$parent.
	Parent Model

	// Related is Relation::$related.
	Related Model

	// EagerKeysWereEmpty is Relation::$eagerKeysWereEmpty: set when the parents
	// carried no keys at all, so that GetEager answers an empty collection
	// instead of running `where in ()`.
	EagerKeysWereEmpty bool

	// ExistenceCompareKey is getExistenceCompareKey as the subtype answers it.
	//
	// It is a field for the reason AliasedPivotColumns is one: the method below
	// is called from GetRelationExistenceQuery, which lives on this type, and Go
	// dispatches statically -- so the call would reach this type's own answer and
	// never the override, whatever the relation actually is. The field is what
	// makes the subtype's version run.
	//
	// What it cost while it was missing: a whereHas over a has-many compiled
	// `where exists (select * from posts where users.id = posts.id)`, comparing
	// the parent's key against the child's own key instead of against the foreign
	// key pointing back. That is a well formed query, it runs, and it answers
	// with the rows whose ids happen to line up.
	ExistenceCompareKey func() string
}

// NewBaseRelation answers the shared half of Relation::__construct.
func NewBaseRelation(query Builder, parent Model) BaseRelation {
	return BaseRelation{Query: query, Parent: parent, Related: query.GetModel()}
}

// GetQuery answers Relation::getQuery.
func (r *BaseRelation) GetQuery() Builder { return r.Query }

// GetRelationQuery answers Relation::getRelationQuery. CanBeOneOfMany overrides
// it to point the constraints at the subquery.
func (r *BaseRelation) GetRelationQuery() Builder { return r.Query }

// GetBaseQuery answers Relation::getBaseQuery.
func (r *BaseRelation) GetBaseQuery() *query.Builder { return r.Query.GetQuery() }

// ToBase answers Relation::toBase.
func (r *BaseRelation) ToBase() *query.Builder { return r.Query.GetQuery() }

// GetParent answers Relation::getParent.
func (r *BaseRelation) GetParent() Model { return r.Parent }

// GetRelated answers Relation::getRelated.
func (r *BaseRelation) GetRelated() Model { return r.Related }

// GetQualifiedParentKeyName answers Relation::getQualifiedParentKeyName.
func (r *BaseRelation) GetQualifiedParentKeyName() string {
	return r.Parent.QualifyColumn(r.Parent.GetKeyName())
}

// CreatedAt answers Relation::createdAt.
func (r *BaseRelation) CreatedAt() string { return r.Parent.GetCreatedAtColumn() }

// UpdatedAt answers Relation::updatedAt.
func (r *BaseRelation) UpdatedAt() string { return r.Parent.GetUpdatedAtColumn() }

// RelatedUpdatedAt answers Relation::relatedUpdatedAt.
func (r *BaseRelation) RelatedUpdatedAt() string { return r.Related.GetUpdatedAtColumn() }

// Get answers Relation::get.
//
// It is the one place every relation's reads funnel through, and it is where
// the tenant filter goes on. On a clone, so that reading a relation twice does
// not add the clause twice, and before the Grant reaches the connection, so
// that a read authorized for one customer cannot return another's rows.
func (r *BaseRelation) Get(ctx context.Context, g auth.Grant, columns ...any) ([]Model, error) {
	scoped, err := r.scoped(g)
	if err != nil {
		return nil, err
	}
	if len(columns) > 0 {
		scoped = scoped.Select(columns...)
	}
	return scoped.Get(ctx, g)
}

// First answers the Builder::first that every one-result relation calls.
func (r *BaseRelation) First(ctx context.Context, g auth.Grant) (Model, error) {
	scoped, err := r.scoped(g)
	if err != nil {
		return nil, err
	}
	return scoped.First(ctx, g)
}

// GetEager answers Relation::getEager.
//
// The empty-keys branch is not an optimization. `where in ()` is a syntax error
// on some engines and a match-nothing on others, and the parents that produced
// no keys still have to come back with their relation initialized -- so the
// query is skipped and the empty collection is the answer.
func (r *BaseRelation) GetEager(ctx context.Context, g auth.Grant) ([]Model, error) {
	if r.EagerKeysWereEmpty {
		return []Model{}, nil
	}
	return r.Get(ctx, g)
}

// Sole answers Relation::sole.
//
// It carries two errors where the PHP throws two exceptions: nothing matched,
// and more than one did. Both are the caller's assumption being wrong, and
// both are worth telling apart.
func (r *BaseRelation) Sole(ctx context.Context, g auth.Grant, columns ...any) (Model, error) {
	scoped, err := r.scoped(g)
	if err != nil {
		return nil, err
	}
	if len(columns) > 0 {
		scoped = scoped.Select(columns...)
	}

	results, err := scoped.Limit(2).Get(ctx, g)
	if err != nil {
		return nil, err
	}

	switch len(results) {
	case 0:
		return nil, fmt.Errorf("%w: table %s", ErrModelNotFound, r.Related.GetTable())
	case 1:
		return results[0], nil
	default:
		return nil, fmt.Errorf("%w: %d found on table %s", ErrMultipleRecordsFound, len(results), r.Related.GetTable())
	}
}

// RawUpdate answers Relation::rawUpdate.
func (r *BaseRelation) RawUpdate(ctx context.Context, g auth.Grant, attributes map[string]any) (int64, error) {
	scoped, err := r.scoped(g)
	if err != nil {
		return 0, err
	}
	return scoped.Update(ctx, g, attributes)
}

// Touch answers Relation::touch: it stamps the related rows' updated_at.
func (r *BaseRelation) Touch(ctx context.Context, g auth.Grant) error {
	if !r.Related.UsesTimestamps() {
		return nil
	}
	_, err := r.RawUpdate(ctx, g, map[string]any{
		r.Related.GetUpdatedAtColumn(): r.Related.FreshTimestamp(),
	})
	return err
}

// GetRelationExistenceQuery answers Relation::getRelationExistenceQuery: the
// subquery a whereHas compares against, which matches on column names rather
// than on values.
func (r *BaseRelation) GetRelationExistenceQuery(q Builder, parentQuery Builder, columns ...any) Builder {
	if len(columns) == 0 {
		columns = []any{"*"}
	}
	return q.Select(columns...).WhereColumn(
		r.GetQualifiedParentKeyName(), "=", r.GetExistenceCompareKey(),
	)
}

// GetRelationExistenceCountQuery answers
// Relation::getRelationExistenceCountQuery.
func (r *BaseRelation) GetRelationExistenceCountQuery(q Builder, parentQuery Builder) Builder {
	counted := r.GetRelationExistenceQuery(q, parentQuery, query.Raw("count(*)"))
	counted.GetQuery().SetBindings(nil, "select")
	return counted
}

// GetExistenceCompareKey answers Relation::getExistenceCompareKey.
//
// Each relation overrides it, and sets ExistenceCompareKey in its constructor so
// that the override is what this answers -- see the field. Without one set, this
// answers the related model's key, which is what a relation with no foreign key
// of its own would compare.
func (r *BaseRelation) GetExistenceCompareKey() string {
	if r.ExistenceCompareKey != nil {
		return r.ExistenceCompareKey()
	}
	return r.Related.QualifyColumn(r.Related.GetKeyName())
}

// GetRelationCountHash answers Relation::getRelationCountHash: the alias a
// self-join needs so the same table can appear twice.
func (r *BaseRelation) GetRelationCountHash(incrementJoinCount ...bool) string {
	selfJoinCount.Lock()
	defer selfJoinCount.Unlock()

	if len(incrementJoinCount) == 0 || incrementJoinCount[0] {
		hash := "arandu_reserved_" + strconv.Itoa(selfJoinCount.value)
		selfJoinCount.value++
		return hash
	}
	return "arandu_reserved_" + strconv.Itoa(selfJoinCount.value)
}

// GetKeys answers Relation::getKeys: every parent's key, deduplicated and
// sorted.
//
// Sorted because the PHP sorts, and the PHP sorts because the emitted `in (...)`
// is then stable between requests -- which is what lets a query log, a slow
// query report and a prepared statement cache recognize the same query twice.
func (r *BaseRelation) GetKeys(models []Model, key string) ([]any, error) {
	seen := make(map[string]struct{}, len(models))
	pairs := make([]struct {
		sortKey string
		value   any
	}, 0, len(models))

	for _, model := range models {
		var value any
		if key != "" {
			value = model.GetAttribute(key)
		} else {
			value = model.GetKey()
		}
		if value == nil {
			continue
		}

		dictionaryKey, err := concerns.GetDictionaryKey(value)
		if err != nil {
			return nil, err
		}
		if _, ok := seen[dictionaryKey]; ok {
			continue
		}
		seen[dictionaryKey] = struct{}{}
		pairs = append(pairs, struct {
			sortKey string
			value   any
		}{dictionaryKey, value})
	}

	sort.SliceStable(pairs, func(i, j int) bool { return pairs[i].sortKey < pairs[j].sortKey })

	keys := make([]any, 0, len(pairs))
	for _, pair := range pairs {
		keys = append(keys, pair.value)
	}
	return keys, nil
}

// WhereInEager answers Relation::whereInEager.
//
// The PHP picks between whereIn and whereIntegerInRaw by method name; the
// choice is an optimization for integer keys that the base builder here does
// not offer, so this is whereIn and the flag it sets is the part that matters:
// an eager load with no keys must not run a query at all.
func (r *BaseRelation) WhereInEager(key string, modelKeys []any, q ...Builder) {
	target := r.Query
	if len(q) > 0 && q[0] != nil {
		target = q[0]
	}

	target.WhereIn(key, modelKeys)

	if len(modelKeys) == 0 {
		r.EagerKeysWereEmpty = true
	}
}

// WhereInMethod answers Relation::whereInMethod.
//
// It reports whether the key is the model's own integer primary key, which is
// the condition under which the PHP switches to whereIntegerInRaw. Kept because
// the question it asks is real and a driver-aware builder will want it; the
// answer changes no SQL today.
func (r *BaseRelation) WhereInMethod(model Model, key string) string {
	segments := strings.Split(key, ".")
	last := segments[len(segments)-1]

	if model.GetKeyName() == last && (model.GetKeyType() == "int" || model.GetKeyType() == "integer") {
		return "whereIntegerInRaw"
	}
	return "whereIn"
}

// scoped returns the relation query narrowed to the Grant's tenant.
//
// Every read in this package goes through it. A relation that skipped it would
// be the leak that is hardest to see: the parent query is right, the join is
// right, and the rows that come back belong to somebody else.
func (r *BaseRelation) scoped(g auth.Grant) (Builder, error) {
	return concerns.ScopeTenant(r.Query.Clone(), r.Related, g)
}

// Every relation the PHP declares implements the contract, and the compiler
// checks it here rather than at the first call site. A relation that stopped
// satisfying it -- a signature drifting during a refactor -- would otherwise
// only be caught by the one caller that eager loads it.
var (
	_ Relation = (*HasOne)(nil)
	_ Relation = (*HasMany)(nil)
	_ Relation = (*BelongsTo)(nil)
	_ Relation = (*BelongsToMany)(nil)
	_ Relation = (*MorphOne)(nil)
	_ Relation = (*MorphMany)(nil)
	_ Relation = (*MorphTo)(nil)
	_ Relation = (*MorphToMany)(nil)
	_ Relation = (*HasOneThrough)(nil)
	_ Relation = (*HasManyThrough)(nil)
)
