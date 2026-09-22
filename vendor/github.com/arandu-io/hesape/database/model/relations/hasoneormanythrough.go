package relations

import (
	"context"
	"iter"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database/model/relations/concerns"
	"github.com/arandu-io/hesape/pagination"
)

// ThroughKey is the alias the intermediate key is selected under.
//
// The name is the framework's rather than the user's, and it is a column name
// that comes back on every row of a has-many-through, which makes it the one
// string in this package a reader is guaranteed to meet in a result set.
const ThroughKey = "arandu_through_key"

// HasOneOrManyThrough is a relation that reaches its rows through a third table
// -- a country's posts, through its users.
//
// The intermediate table is joined rather than queried, so it is still two
// queries for an eager load and not three. What makes the eager load work is
// selecting the intermediate table's foreign key alongside the related row,
// under ThroughKey: without it the result set could not say which far parent
// each row came from, because the related table has no column that names it.
type HasOneOrManyThrough struct {
	BaseRelation

	throughParent  Model
	farParent      Model
	firstKey       string
	secondKey      string
	localKey       string
	secondLocalKey string

	// withTrashedParents is what WithTrashedParents sets. PHP registers the
	// soft-delete filter as a named global scope on the builder and removes it
	// by name; relations.Builder has no scope registry, so the filter is applied
	// at read time and this flag is what suppresses it.
	withTrashedParents bool
}

// SoftDeletableThrough is what ThroughParentSoftDeletes asks of the
// intermediate model.
//
// PHP writes $this->throughParent::isSoftDeletable(), a static call on whatever
// class the model happens to be. Go has no late static binding and the Model
// interface a relation works against says nothing about soft deletes -- so the
// question is an optional interface, and a through parent that does not
// implement it is simply not soft deletable.
type SoftDeletableThrough interface {
	// IsSoftDeletable answers Model::isSoftDeletable.
	IsSoftDeletable() bool

	// GetDeletedAtColumn answers SoftDeletes::getDeletedAtColumn.
	GetDeletedAtColumn() string
}

// NewHasOneOrManyThrough answers HasOneOrManyThrough::__construct.
func NewHasOneOrManyThrough(query Builder, farParent, throughParent Model, firstKey, secondKey, localKey, secondLocalKey string) HasOneOrManyThrough {
	return HasOneOrManyThrough{
		BaseRelation:   NewBaseRelation(query, throughParent),
		farParent:      farParent,
		throughParent:  throughParent,
		firstKey:       firstKey,
		secondKey:      secondKey,
		localKey:       localKey,
		secondLocalKey: secondLocalKey,
	}
}

// AddConstraints answers HasOneOrManyThrough::addConstraints.
func (r *HasOneOrManyThrough) AddConstraints() {
	r.performJoin(nil)
	r.addWhereConstraints()
}

// addWhereConstraints is the half of AddConstraints that narrows the query to
// one far parent. The join is the other half, and it is on both paths: an eager
// load reads through the intermediate table for every parent at once.
func (r *HasOneOrManyThrough) addWhereConstraints() {
	r.Query.Where(r.GetQualifiedFirstKeyName(), "=", r.farParent.GetAttribute(r.localKey))
}

// performJoin answers HasOneOrManyThrough::performJoin.
func (r *HasOneOrManyThrough) performJoin(q Builder) {
	if q == nil {
		q = r.Query
	}
	q.Join(r.throughParent.GetTable(), r.GetQualifiedParentKeyName(), "=", r.GetQualifiedFarKeyName())
}

// AddEagerConstraints answers HasOneOrManyThrough::addEagerConstraints.
func (r *HasOneOrManyThrough) AddEagerConstraints(models []Model) error {
	keys, err := r.GetKeys(models, r.localKey)
	if err != nil {
		return err
	}
	r.WhereInEager(r.GetQualifiedFirstKeyName(), keys)
	return nil
}

// buildDictionary answers HasOneOrManyThrough::buildDictionary: keyed by the
// intermediate key that came back under ThroughKey.
func (r *HasOneOrManyThrough) buildDictionary(results []Model) (map[string][]Model, error) {
	dictionary := make(map[string][]Model, len(results))
	for _, result := range results {
		key, err := concerns.GetDictionaryKey(result.GetAttribute(ThroughKey))
		if err != nil {
			return nil, err
		}
		dictionary[key] = append(dictionary[key], result)
	}
	return dictionary, nil
}

// ThroughParentSoftDeletes answers
// HasOneOrManyThrough::throughParentSoftDeletes.
func (r *HasOneOrManyThrough) ThroughParentSoftDeletes() bool {
	deletable, ok := r.throughParent.(SoftDeletableThrough)
	return ok && deletable.IsSoftDeletable()
}

// WithTrashedParents answers HasOneOrManyThrough::withTrashedParents: the rows
// whose intermediate parent was soft deleted come back too.
//
// PHP removes the 'SoftDeletableHasManyThrough' global scope from the query.
// There is no scope registry on the builder a relation holds here, so the
// filter is added by constrainThroughParents on the way to the database and
// this turns it off. The name and the effect are the PHP's; the mechanism is
// the one this collection has.
func (r *HasOneOrManyThrough) WithTrashedParents() *HasOneOrManyThrough {
	r.withTrashedParents = true
	return r
}

// constrainThroughParents is the body of the global scope PHP registers in
// performJoin: a row whose intermediate parent is soft deleted is not reachable
// through it.
//
// It runs on the read rather than at construction because WithTrashedParents is
// called after the constructor, and a filter already compiled into the join
// could not be taken back off.
func (r *HasOneOrManyThrough) constrainThroughParents(q Builder) Builder {
	if r.withTrashedParents || !r.ThroughParentSoftDeletes() {
		return q
	}
	deletable := r.throughParent.(SoftDeletableThrough)
	q.GetQuery().WhereNull(r.throughParent.QualifyColumn(deletable.GetDeletedAtColumn()))
	return q
}

// Get answers HasOneOrManyThrough::get: the select carries the intermediate key
// so that Match has something to key on.
func (r *HasOneOrManyThrough) Get(ctx context.Context, g auth.Grant, columns ...any) ([]Model, error) {
	scoped, err := concerns.ScopeTenant(r.Query.Clone(), r.Related, g)
	if err != nil {
		return nil, err
	}
	return r.constrainThroughParents(scoped).AddSelect(r.shouldSelect(columns)...).Get(ctx, g)
}

// GetEager answers Relation::getEager for a through relation.
func (r *HasOneOrManyThrough) GetEager(ctx context.Context, g auth.Grant) ([]Model, error) {
	if r.EagerKeysWereEmpty {
		return []Model{}, nil
	}
	return r.Get(ctx, g)
}

// First answers the one-row read, with the same select as Get.
func (r *HasOneOrManyThrough) First(ctx context.Context, g auth.Grant) (Model, error) {
	scoped, err := concerns.ScopeTenant(r.Query.Clone(), r.Related, g)
	if err != nil {
		return nil, err
	}
	return r.constrainThroughParents(scoped).AddSelect(r.shouldSelect(nil)...).First(ctx, g)
}

// prepareQueryBuilder answers HasOneOrManyThrough::prepareQueryBuilder: the
// relation's query with its select applied, ready to be walked.
//
// It is scoped to the Grant's tenant here rather than at each call site, which
// is the whole reason it takes one. Every method below reaches the database
// through it, so there is exactly one place a chunked or paginated read could
// have lost the tenant filter, and it is this line.
func (r *HasOneOrManyThrough) prepareQueryBuilder(g auth.Grant) (Builder, error) {
	scoped, err := concerns.ScopeTenant(r.Query.Clone(), r.Related, g)
	if err != nil {
		return nil, err
	}
	return r.constrainThroughParents(scoped).AddSelect(r.shouldSelect(nil)...), nil
}

// walker is the shared body of the thirteen walking methods, given this
// relation's prepared query. A through relation hydrates no pivot.
func (r *HasOneOrManyThrough) walker() chunker {
	return chunker{
		prepare: r.prepareQueryBuilder,
		keyName: func() string { return r.Related.QualifyColumn(r.Related.GetKeyName()) },
	}
}

// Chunk answers HasOneOrManyThrough::chunk: the results a page at a time, so a
// table that does not fit in memory does not have to.
//
// The callback answers false to stop, as the PHP's does, and Chunk answers
// false when it was stopped rather than exhausted.
func (r *HasOneOrManyThrough) Chunk(ctx context.Context, g auth.Grant, count int, callback func(results []Model, page int) bool) (bool, error) {
	return r.walker().chunk(ctx, g, count, callback)
}

// ChunkById answers HasOneOrManyThrough::chunkById: pages taken by comparing
// against the last id seen rather than by an offset.
//
// The empty column and alias take the PHP's defaults, which are the related
// model's qualified key and its key name.
func (r *HasOneOrManyThrough) ChunkById(ctx context.Context, g auth.Grant, count int, callback func(results []Model, page int) bool, column, alias string) (bool, error) {
	return r.walker().orderedChunkByID(ctx, g, count, callback, column, alias, false)
}

// ChunkByIdDesc answers HasOneOrManyThrough::chunkByIdDesc.
func (r *HasOneOrManyThrough) ChunkByIdDesc(ctx context.Context, g auth.Grant, count int, callback func(results []Model, page int) bool, column, alias string) (bool, error) {
	return r.walker().orderedChunkByID(ctx, g, count, callback, column, alias, true)
}

// Each answers HasOneOrManyThrough::each: chunk, one row at a time.
func (r *HasOneOrManyThrough) Each(ctx context.Context, g auth.Grant, callback func(value Model, key int) bool, count int) (bool, error) {
	return r.walker().each(ctx, g, callback, count)
}

// EachById answers HasOneOrManyThrough::eachById.
func (r *HasOneOrManyThrough) EachById(ctx context.Context, g auth.Grant, callback func(value Model, key int) bool, count int, column, alias string) (bool, error) {
	return r.walker().orderedChunkByID(ctx, g, count, func(results []Model, page int) bool {
		for index, value := range results {
			if !callback(value, (page-1)*count+index) {
				return false
			}
		}
		return true
	}, column, alias, false)
}

// Lazy answers HasOneOrManyThrough::lazy: the chunked walk as a sequence.
func (r *HasOneOrManyThrough) Lazy(ctx context.Context, g auth.Grant, chunkSize int) iter.Seq2[Model, error] {
	return r.walker().lazy(ctx, g, chunkSize, false, "", "", false)
}

// LazyById answers HasOneOrManyThrough::lazyById.
func (r *HasOneOrManyThrough) LazyById(ctx context.Context, g auth.Grant, chunkSize int, column, alias string) iter.Seq2[Model, error] {
	return r.walker().lazy(ctx, g, chunkSize, true, column, alias, false)
}

// LazyByIdDesc answers HasOneOrManyThrough::lazyByIdDesc.
func (r *HasOneOrManyThrough) LazyByIdDesc(ctx context.Context, g auth.Grant, chunkSize int, column, alias string) iter.Seq2[Model, error] {
	return r.walker().lazy(ctx, g, chunkSize, true, column, alias, true)
}

// Cursor answers HasOneOrManyThrough::cursor: one statement, streamed.
func (r *HasOneOrManyThrough) Cursor(ctx context.Context, g auth.Grant) iter.Seq2[Model, error] {
	return r.walker().cursor(ctx, g)
}

// Paginate answers HasOneOrManyThrough::paginate.
func (r *HasOneOrManyThrough) Paginate(ctx context.Context, g auth.Grant, perPage, page int, opts pagination.Options, columns ...any) (*pagination.LengthAwarePaginator[Model], error) {
	return r.walker().paginate(ctx, g, perPage, page, opts, columns...)
}

// SimplePaginate answers HasOneOrManyThrough::simplePaginate.
func (r *HasOneOrManyThrough) SimplePaginate(ctx context.Context, g auth.Grant, perPage, page int, opts pagination.Options, columns ...any) (*pagination.Paginator[Model], error) {
	return r.walker().simplePaginate(ctx, g, perPage, page, opts, columns...)
}

// CursorPaginate answers HasOneOrManyThrough::cursorPaginate.
func (r *HasOneOrManyThrough) CursorPaginate(ctx context.Context, g auth.Grant, perPage int, cursor *pagination.Cursor, opts pagination.Options, columns ...any) (*pagination.CursorPaginator[Model], error) {
	return r.walker().cursorPaginate(ctx, g, perPage, cursor, opts, columns...)
}

// shouldSelect answers HasOneOrManyThrough::shouldSelect.
func (r *HasOneOrManyThrough) shouldSelect(columns []any) []any {
	if len(columns) == 0 {
		columns = []any{r.Related.QualifyColumn("*")}
	}
	return append(columns, r.GetQualifiedFirstKeyName()+" as "+ThroughKey)
}

// Limit answers HasOneOrManyThrough::limit.
func (r *HasOneOrManyThrough) Limit(value int) *HasOneOrManyThrough {
	if r.farParent.Exists() {
		r.Query.Limit(value)
	} else {
		r.Query.GetQuery().GroupLimit(value, r.GetQualifiedFirstKeyName())
	}
	return r
}

// Take answers HasOneOrManyThrough::take.
func (r *HasOneOrManyThrough) Take(value int) *HasOneOrManyThrough { return r.Limit(value) }

// GetRelationExistenceQuery answers
// HasOneOrManyThrough::getRelationExistenceQuery.
func (r *HasOneOrManyThrough) GetRelationExistenceQuery(q Builder, parentQuery Builder, columns ...any) Builder {
	if len(columns) == 0 {
		columns = []any{"*"}
	}
	r.performJoin(q)

	return q.Select(columns...).WhereColumn(
		r.GetQualifiedLocalKeyName(), "=", r.GetQualifiedFirstKeyName(),
	)
}

// GetRelationExistenceQueryForThroughSelfRelation answers
// HasOneOrManyThrough::getRelationExistenceQueryForThroughSelfRelation: the
// existence subquery when the intermediate table is the same table the outer
// query already names.
//
// The join has to alias the intermediate table, or the two references to it
// collapse into one and the subquery correlates with itself. The alias is
// GetRelationCountHash, which is what every self-join in this package uses.
func (r *HasOneOrManyThrough) GetRelationExistenceQueryForThroughSelfRelation(q Builder, parentQuery Builder, columns ...any) Builder {
	if len(columns) == 0 {
		columns = []any{"*"}
	}
	hash := r.GetRelationCountHash()

	q.Join(r.throughParent.GetTable()+" as "+hash, hash+"."+r.secondLocalKey, "=", r.GetQualifiedFarKeyName())

	if deletable, ok := r.throughParent.(SoftDeletableThrough); ok && deletable.IsSoftDeletable() {
		q.GetQuery().WhereNull(hash + "." + deletable.GetDeletedAtColumn())
	}

	return q.Select(columns...).WhereColumn(
		stringifyFrom(parentQuery)+"."+r.localKey, "=", hash+"."+r.firstKey,
	)
}

// GetParent answers Relation::getParent. For a through relation the parent of
// the relation is the intermediate model, and the far parent is what declared
// it -- the PHP keeps both, and so does this.
func (r *HasOneOrManyThrough) GetFarParent() Model { return r.farParent }

// GetThroughParent answers HasOneOrManyThrough::$throughParent.
func (r *HasOneOrManyThrough) GetThroughParent() Model { return r.throughParent }

// GetQualifiedParentKeyName answers
// HasOneOrManyThrough::getQualifiedParentKeyName.
func (r *HasOneOrManyThrough) GetQualifiedParentKeyName() string {
	return r.throughParent.QualifyColumn(r.secondLocalKey)
}

// GetQualifiedFarKeyName answers HasOneOrManyThrough::getQualifiedFarKeyName.
func (r *HasOneOrManyThrough) GetQualifiedFarKeyName() string {
	return r.GetQualifiedForeignKeyName()
}

// GetFirstKeyName answers HasOneOrManyThrough::getFirstKeyName.
func (r *HasOneOrManyThrough) GetFirstKeyName() string { return r.firstKey }

// GetQualifiedFirstKeyName answers
// HasOneOrManyThrough::getQualifiedFirstKeyName.
func (r *HasOneOrManyThrough) GetQualifiedFirstKeyName() string {
	return r.throughParent.QualifyColumn(r.firstKey)
}

// GetForeignKeyName answers HasOneOrManyThrough::getForeignKeyName.
func (r *HasOneOrManyThrough) GetForeignKeyName() string { return r.secondKey }

// GetQualifiedForeignKeyName answers
// HasOneOrManyThrough::getQualifiedForeignKeyName.
func (r *HasOneOrManyThrough) GetQualifiedForeignKeyName() string {
	return r.Related.QualifyColumn(r.secondKey)
}

// GetLocalKeyName answers HasOneOrManyThrough::getLocalKeyName.
func (r *HasOneOrManyThrough) GetLocalKeyName() string { return r.localKey }

// GetQualifiedLocalKeyName answers
// HasOneOrManyThrough::getQualifiedLocalKeyName.
func (r *HasOneOrManyThrough) GetQualifiedLocalKeyName() string {
	return r.farParent.QualifyColumn(r.localKey)
}

// GetSecondLocalKeyName answers
// HasOneOrManyThrough::getSecondLocalKeyName.
func (r *HasOneOrManyThrough) GetSecondLocalKeyName() string { return r.secondLocalKey }

// GetParentKey answers the far parent's local key, which is what a through
// relation is narrowed by.
func (r *HasOneOrManyThrough) GetParentKey() any { return r.farParent.GetAttribute(r.localKey) }

// HasManyThrough is the many end of a relation that reaches its rows through an
// intermediate table: a country's posts, where posts belong to users and users
// belong to the country.
//
// There is no column on posts naming the country, so the query joins users and
// filters on the country's key. GetResults answers the matching rows, or an
// empty slice when the far parent has no key yet -- an unsaved parent has no
// children, and the join would otherwise match every row.
type HasManyThrough struct {
	HasOneOrManyThrough
}

// NewHasManyThrough answers HasManyThrough's constructor.
func newHasManyThrough(query Builder, farParent, throughParent Model, firstKey, secondKey, localKey, secondLocalKey string) *HasManyThrough {
	relation := &HasManyThrough{
		HasOneOrManyThrough: NewHasOneOrManyThrough(query, farParent, throughParent, firstKey, secondKey, localKey, secondLocalKey),
	}
	// The join is on both paths; only the where that narrows to one far parent
	// is what "unconstrained" leaves off.
	relation.performJoin(nil)
	return relation
}

// NewHasManyThrough builds the relation and narrows it to its parent.
func NewHasManyThrough(query Builder, farParent, throughParent Model, firstKey, secondKey, localKey, secondLocalKey string) *HasManyThrough {
	relation := newHasManyThrough(query, farParent, throughParent, firstKey, secondKey, localKey, secondLocalKey)
	relation.addWhereConstraints()
	return relation
}

// GetResults answers HasManyThrough::getResults.
func (r *HasManyThrough) GetResults(ctx context.Context, g auth.Grant) (any, error) {
	if r.GetParentKey() == nil {
		return []Model{}, nil
	}
	return r.Get(ctx, g)
}

// InitRelation answers HasManyThrough::initRelation.
func (r *HasManyThrough) InitRelation(models []Model, relation string) []Model {
	for _, model := range models {
		model.SetRelation(relation, []Model{})
	}
	return models
}

// Match answers HasManyThrough::match.
func (r *HasManyThrough) Match(models []Model, results []Model, relation string) ([]Model, error) {
	dictionary, err := r.buildDictionary(results)
	if err != nil {
		return nil, err
	}

	for _, model := range models {
		value := model.GetAttribute(r.localKey)
		if value == nil {
			continue
		}
		key, err := concerns.GetDictionaryKey(value)
		if err != nil {
			return nil, err
		}
		if matched, ok := dictionary[key]; ok {
			model.SetRelation(relation, matched)
		}
	}

	return models, nil
}

// One answers HasManyThrough::one.
func (r *HasManyThrough) One() *HasOneThrough {
	q := r.Query
	q.GetQuery().Joins = nil
	one := NewHasOneThroughUnconstrained(q, r.farParent, r.throughParent, r.firstKey, r.secondKey, r.localKey, r.secondLocalKey)
	return one
}

// HasOneThrough is the singular end of the same join HasManyThrough makes: a
// mechanic's car owner, through the car.
//
// It is HasManyThrough narrowed to the first row. What it adds is a default:
// where the many form answers an empty slice, this one answers whatever
// WithDefault was given -- so a caller reads fields off the result without
// testing for nil first. With no default configured the miss is still nil.
//
// It also compares related models, which is what makes Is and IsNot work
// against a relation that has to be resolved before it can be compared.
type HasOneThrough struct {
	HasOneOrManyThrough
	concerns.SupportsDefaultModels
	concerns.ComparesRelatedModels
}

// NewHasOneThrough answers HasOneThrough's constructor.
func newHasOneThrough(query Builder, farParent, throughParent Model, firstKey, secondKey, localKey, secondLocalKey string) *HasOneThrough {
	relation := &HasOneThrough{
		HasOneOrManyThrough: NewHasOneOrManyThrough(query, farParent, throughParent, firstKey, secondKey, localKey, secondLocalKey),
	}
	// The join is on both paths; only the where that narrows to one far parent
	// is what "unconstrained" leaves off.
	relation.performJoin(nil)
	relation.SupportsDefaultModels = concerns.SupportsDefaultModels{
		NewRelatedInstanceFor: relation.newRelatedInstanceFor,
	}
	relation.ComparesRelatedModels = concerns.ComparesRelatedModels{
		CompareRelated:        relation.Related,
		CompareParentKey:      relation.GetParentKey,
		CompareRelatedKeyFrom: relation.getRelatedKeyFrom,
	}

	return relation
}

// NewHasOneThrough builds the relation and narrows it to its parent.
func NewHasOneThrough(query Builder, farParent, throughParent Model, firstKey, secondKey, localKey, secondLocalKey string) *HasOneThrough {
	relation := newHasOneThrough(query, farParent, throughParent, firstKey, secondKey, localKey, secondLocalKey)
	relation.addWhereConstraints()
	return relation
}

// GetResults answers HasOneThrough::getResults.
func (r *HasOneThrough) GetResults(ctx context.Context, g auth.Grant) (any, error) {
	if r.GetParentKey() == nil {
		return r.GetDefaultFor(r.farParent), nil
	}

	result, err := r.First(ctx, g)
	if err != nil {
		return nil, err
	}
	if result == nil {
		return r.GetDefaultFor(r.farParent), nil
	}
	return result, nil
}

// InitRelation answers HasOneThrough::initRelation.
func (r *HasOneThrough) InitRelation(models []Model, relation string) []Model {
	for _, model := range models {
		model.SetRelation(relation, r.GetDefaultFor(model))
	}
	return models
}

// Match answers HasOneThrough::match.
func (r *HasOneThrough) Match(models []Model, results []Model, relation string) ([]Model, error) {
	dictionary, err := r.buildDictionary(results)
	if err != nil {
		return nil, err
	}

	for _, model := range models {
		value := model.GetAttribute(r.localKey)
		if value == nil {
			continue
		}
		key, err := concerns.GetDictionaryKey(value)
		if err != nil {
			return nil, err
		}
		if matched, ok := dictionary[key]; ok && len(matched) > 0 {
			model.SetRelation(relation, matched[0])
		}
	}

	return models, nil
}

// newRelatedInstanceFor answers HasOneThrough::newRelatedInstanceFor.
func (r *HasOneThrough) newRelatedInstanceFor(parent Model) Model {
	return r.Related.NewInstance(nil)
}

// getRelatedKeyFrom answers HasOneThrough::getRelatedKeyFrom.
func (r *HasOneThrough) getRelatedKeyFrom(model Model) any {
	return model.GetAttribute(r.GetForeignKeyName())
}

// NewHasManyThroughUnconstrained builds the relation without narrowing it to one parent.
//
// It exists because the constraint used to be switched off through a
// process-wide flag, which meant a relation built on another goroutine while
// the flag was down came back unconstrained -- every parent's children, in a
// well-formed query nobody could tell apart from the right one. The call site
// says which it wants now, and there is no flag to leave down.
func NewHasManyThroughUnconstrained(query Builder, farParent, throughParent Model, firstKey, secondKey, localKey, secondLocalKey string) *HasManyThrough {
	return newHasManyThrough(query, farParent, throughParent, firstKey, secondKey, localKey, secondLocalKey)
}

// NewHasOneThroughUnconstrained builds the relation without narrowing it to one parent.
//
// It exists because the constraint used to be switched off through a
// process-wide flag, which meant a relation built on another goroutine while
// the flag was down came back unconstrained -- every parent's children, in a
// well-formed query nobody could tell apart from the right one. The call site
// says which it wants now, and there is no flag to leave down.
func NewHasOneThroughUnconstrained(query Builder, farParent, throughParent Model, firstKey, secondKey, localKey, secondLocalKey string) *HasOneThrough {
	return newHasOneThrough(query, farParent, throughParent, firstKey, secondKey, localKey, secondLocalKey)
}
