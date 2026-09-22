package relations

import (
	"context"
	"errors"
	"iter"
	"sort"
	"strings"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database"
	"github.com/arandu-io/hesape/database/model/relations/concerns"
	"github.com/arandu-io/hesape/database/query"
	"github.com/arandu-io/hesape/pagination"
)

var errNoPivotQuery = errors.New("relations: this pivot was built without a query factory, so it cannot reach the database")

// PivotFactory builds the model a custom pivot class would be in PHP. using()
// takes one.
type PivotFactory func(parent Model, attributes map[string]any, table string, exists bool) Model

// BelongsToMany is many rows on the other table, reached through an intermediate
// table that holds the pairs.
//
// The intermediate table is what makes this relation different from every other
// one: it is joined for reads, written directly for attach and detach, and its
// extra columns come back on the related model under an accessor -- "pivot" by
// default -- as a Pivot model rather than as attributes of the related row.
type BelongsToMany struct {
	BaseRelation
	concerns.InteractsWithPivotTable

	table           string
	foreignPivotKey string
	relatedPivotKey string
	parentKey       string
	relatedKey      string
	relationName    string
	accessor        string

	withTimestamps bool
	pivotCreatedAt string
	pivotUpdatedAt string

	// pivotWheres, pivotWhereIns and pivotWhereNulls are the relation's
	// properties of the same name: the constraints put on the pivot columns,
	// kept so newPivotQuery can replay them on a statement that has no join.
	pivotWheres     []func(*query.Builder)
	pivotWhereIns   []func(*query.Builder)
	pivotWhereNulls []func(*query.Builder)

	using PivotFactory

	// AliasedPivotColumns is the PHP's protected aliasedPivotColumns, as a hook.
	// MorphToMany overrides that method to add the type column; Go dispatches
	// statically, so an override would never be reached from here -- the field
	// is what makes the subtype's version run.
	AliasedPivotColumns func() []any
}

// NewBelongsToMany answers BelongsToMany::__construct.
func newBelongsToMany(query Builder, parent Model, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relationName string) *BelongsToMany {
	relation := &BelongsToMany{
		BaseRelation:    NewBaseRelation(query, parent),
		table:           table,
		foreignPivotKey: foreignPivotKey,
		relatedPivotKey: relatedPivotKey,
		parentKey:       parentKey,
		relatedKey:      relatedKey,
		relationName:    relationName,
		accessor:        "pivot",
	}
	relation.InteractsWithPivotTable = concerns.InteractsWithPivotTable{Host: relation}
	relation.AliasedPivotColumns = relation.aliasedPivotColumns
	relation.ExistenceCompareKey = relation.GetExistenceCompareKey

	// The join is on both paths. Only the where clause that narrows the pivot to
	// one parent is what "unconstrained" leaves off -- an eager load reads the
	// pivot for every parent at once and still needs the join to reach it.
	relation.performJoin(nil)

	return relation
}

// NewBelongsToMany builds the relation and narrows it to its parent.
func NewBelongsToMany(query Builder, parent Model, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relationName string) *BelongsToMany {
	relation := newBelongsToMany(query, parent, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relationName)
	relation.addWhereConstraints()
	return relation
}

// AddConstraints answers BelongsToMany::addConstraints.
func (r *BelongsToMany) AddConstraints() {
	r.performJoin(nil)
	r.addWhereConstraints()
}

// performJoin answers BelongsToMany::performJoin.
func (r *BelongsToMany) performJoin(q Builder) *BelongsToMany {
	if q == nil {
		q = r.Query
	}
	q.Join(r.table, r.GetQualifiedRelatedKeyName(), "=", r.GetQualifiedRelatedPivotKeyName())
	return r
}

// addWhereConstraints answers BelongsToMany::addWhereConstraints.
func (r *BelongsToMany) addWhereConstraints() *BelongsToMany {
	r.Query.Where(r.GetQualifiedForeignPivotKeyName(), "=", r.ParentKeyValue())
	return r
}

// AddEagerConstraints answers BelongsToMany::addEagerConstraints.
func (r *BelongsToMany) AddEagerConstraints(models []Model) error {
	keys, err := r.GetKeys(models, r.parentKey)
	if err != nil {
		return err
	}
	r.WhereInEager(r.GetQualifiedForeignPivotKeyName(), keys)
	return nil
}

// InitRelation answers BelongsToMany::initRelation.
func (r *BelongsToMany) InitRelation(models []Model, relation string) []Model {
	for _, model := range models {
		model.SetRelation(relation, []Model{})
	}
	return models
}

// Match answers BelongsToMany::match.
//
// The dictionary is keyed by the pivot's copy of the parent key, not by
// anything on the related row -- the related row has no idea which parent
// pulled it in, which is why the pivot columns are selected in the first place.
func (r *BelongsToMany) Match(models []Model, results []Model, relation string) ([]Model, error) {
	dictionary, err := r.buildDictionary(results)
	if err != nil {
		return nil, err
	}

	for _, model := range models {
		key, err := concerns.GetDictionaryKey(model.GetAttribute(r.parentKey))
		if err != nil {
			return nil, err
		}
		if matched, ok := dictionary[key]; ok {
			model.SetRelation(relation, matched)
		}
	}

	return models, nil
}

// buildDictionary answers BelongsToMany::buildDictionary.
func (r *BelongsToMany) buildDictionary(results []Model) (map[string][]Model, error) {
	dictionary := make(map[string][]Model, len(results))

	for _, result := range results {
		pivot, ok := result.GetRelation(r.accessor)
		if !ok {
			continue
		}
		row, ok := pivot.(Model)
		if !ok {
			continue
		}

		key, err := concerns.GetDictionaryKey(row.GetAttribute(r.foreignPivotKey))
		if err != nil {
			return nil, err
		}
		dictionary[key] = append(dictionary[key], result)
	}

	return dictionary, nil
}

// GetResults answers BelongsToMany::getResults.
func (r *BelongsToMany) GetResults(ctx context.Context, g auth.Grant) (any, error) {
	if r.ParentKeyValue() == nil {
		return []Model{}, nil
	}
	return r.Get(ctx, g)
}

// Get answers BelongsToMany::get.
//
// The pivot columns are added to the select and then lifted back off the
// related model into a Pivot: they arrive as pivot_user_id and pivot_role_id
// because they would otherwise collide with the related table's own columns,
// and a related model carrying a column from another table is a model whose
// attributes lie about where they came from.
func (r *BelongsToMany) Get(ctx context.Context, g auth.Grant, columns ...any) ([]Model, error) {
	scoped, err := concerns.ScopeTenant(r.Query.Clone(), r.Related, g)
	if err != nil {
		return nil, err
	}

	if len(scoped.GetQuery().GetColumns()) > 0 {
		columns = nil
	}
	scoped.AddSelect(r.shouldSelect(columns)...)

	models, err := scoped.Get(ctx, g)
	if err != nil {
		return nil, err
	}

	r.hydratePivotRelation(models)
	return models, nil
}

// GetEager answers Relation::getEager for this relation: the pivot hydration
// has to happen on the eager path too, or Match has nothing to key on.
func (r *BelongsToMany) GetEager(ctx context.Context, g auth.Grant) ([]Model, error) {
	if r.EagerKeysWereEmpty {
		return []Model{}, nil
	}
	return r.Get(ctx, g)
}

// prepareQueryBuilder answers BelongsToMany::prepareQueryBuilder: the relation's
// query with the aliased pivot columns added to the select.
//
// The tenant filter goes on here, once, for every walking method below. The
// pivot table is what says this customer's user has that customer's role, and a
// paginated read that lost the filter would hand page two of somebody else's
// memberships to a caller who is looking at a page number, not at the SQL.
func (r *BelongsToMany) prepareQueryBuilder(g auth.Grant) (Builder, error) {
	scoped, err := concerns.ScopeTenant(r.Query.Clone(), r.Related, g)
	if err != nil {
		return nil, err
	}
	return scoped.AddSelect(r.shouldSelect(nil)...), nil
}

// walker is the shared body of the thirteen walking methods, given this
// relation's prepared query and its pivot hydration.
//
// The hydration is the one line that differs from the through relation's, and
// it has to run on every page rather than once at the end: a caller that reads
// $model->pivot inside the chunk callback -- which is the ordinary reason to
// chunk a BelongsToMany -- would otherwise find nothing there.
func (r *BelongsToMany) walker() chunker {
	return chunker{
		prepare: r.prepareQueryBuilder,
		hydrate: r.hydratePivotRelation,
		keyName: func() string { return r.Related.QualifyColumn(r.GetRelatedKeyName()) },
	}
}

// Chunk answers BelongsToMany::chunk: the results a page at a time, with the
// pivot hydrated on each page.
func (r *BelongsToMany) Chunk(ctx context.Context, g auth.Grant, count int, callback func(results []Model, page int) bool) (bool, error) {
	return r.walker().chunk(ctx, g, count, callback)
}

// ChunkById answers BelongsToMany::chunkById, which the PHP writes as
// orderedChunkById with descending left false.
func (r *BelongsToMany) ChunkById(ctx context.Context, g auth.Grant, count int, callback func(results []Model, page int) bool, column, alias string) (bool, error) {
	return r.OrderedChunkById(ctx, g, count, callback, column, alias, false)
}

// ChunkByIdDesc answers BelongsToMany::chunkByIdDesc.
func (r *BelongsToMany) ChunkByIdDesc(ctx context.Context, g auth.Grant, count int, callback func(results []Model, page int) bool, column, alias string) (bool, error) {
	return r.OrderedChunkById(ctx, g, count, callback, column, alias, true)
}

// OrderedChunkById answers BelongsToMany::orderedChunkById: pages taken by
// comparing against the last id seen, in the given direction.
//
// The empty column and alias take the PHP's defaults, which are the related
// model's qualified key and the related key's name.
func (r *BelongsToMany) OrderedChunkById(ctx context.Context, g auth.Grant, count int, callback func(results []Model, page int) bool, column, alias string, descending bool) (bool, error) {
	return r.walker().orderedChunkByID(ctx, g, count, callback, column, alias, descending)
}

// Each answers BelongsToMany::each.
func (r *BelongsToMany) Each(ctx context.Context, g auth.Grant, callback func(value Model, key int) bool, count int) (bool, error) {
	return r.walker().each(ctx, g, callback, count)
}

// EachById answers BelongsToMany::eachById.
func (r *BelongsToMany) EachById(ctx context.Context, g auth.Grant, callback func(value Model, key int) bool, count int, column, alias string) (bool, error) {
	return r.OrderedChunkById(ctx, g, count, func(results []Model, page int) bool {
		for index, value := range results {
			if !callback(value, (page-1)*count+index) {
				return false
			}
		}
		return true
	}, column, alias, false)
}

// Lazy answers BelongsToMany::lazy.
func (r *BelongsToMany) Lazy(ctx context.Context, g auth.Grant, chunkSize int) iter.Seq2[Model, error] {
	return r.walker().lazy(ctx, g, chunkSize, false, "", "", false)
}

// LazyById answers BelongsToMany::lazyById.
func (r *BelongsToMany) LazyById(ctx context.Context, g auth.Grant, chunkSize int, column, alias string) iter.Seq2[Model, error] {
	return r.walker().lazy(ctx, g, chunkSize, true, column, alias, false)
}

// LazyByIdDesc answers BelongsToMany::lazyByIdDesc.
func (r *BelongsToMany) LazyByIdDesc(ctx context.Context, g auth.Grant, chunkSize int, column, alias string) iter.Seq2[Model, error] {
	return r.walker().lazy(ctx, g, chunkSize, true, column, alias, true)
}

// Cursor answers BelongsToMany::cursor.
func (r *BelongsToMany) Cursor(ctx context.Context, g auth.Grant) iter.Seq2[Model, error] {
	return r.walker().cursor(ctx, g)
}

// Paginate answers BelongsToMany::paginate.
func (r *BelongsToMany) Paginate(ctx context.Context, g auth.Grant, perPage, page int, opts pagination.Options, columns ...any) (*pagination.LengthAwarePaginator[Model], error) {
	return r.walker().paginate(ctx, g, perPage, page, opts, columns...)
}

// SimplePaginate answers BelongsToMany::simplePaginate.
func (r *BelongsToMany) SimplePaginate(ctx context.Context, g auth.Grant, perPage, page int, opts pagination.Options, columns ...any) (*pagination.Paginator[Model], error) {
	return r.walker().simplePaginate(ctx, g, perPage, page, opts, columns...)
}

// CursorPaginate answers BelongsToMany::cursorPaginate.
func (r *BelongsToMany) CursorPaginate(ctx context.Context, g auth.Grant, perPage int, cursor *pagination.Cursor, opts pagination.Options, columns ...any) (*pagination.CursorPaginator[Model], error) {
	return r.walker().cursorPaginate(ctx, g, perPage, cursor, opts, columns...)
}

// shouldSelect answers BelongsToMany::shouldSelect.
func (r *BelongsToMany) shouldSelect(columns []any) []any {
	if len(columns) == 0 {
		columns = []any{r.Related.QualifyColumn("*")}
	}
	return append(columns, r.AliasedPivotColumns()...)
}

// aliasedPivotColumns answers BelongsToMany::aliasedPivotColumns.
func (r *BelongsToMany) aliasedPivotColumns() []any {
	names := append([]string{r.foreignPivotKey, r.relatedPivotKey}, r.PivotColumns...)

	seen := make(map[string]struct{}, len(names))
	out := make([]any, 0, len(names))
	for _, column := range names {
		aliased := r.QualifyPivotColumn(column) + " as pivot_" + column
		if _, ok := seen[aliased]; ok {
			continue
		}
		seen[aliased] = struct{}{}
		out = append(out, aliased)
	}
	return out
}

// hydratePivotRelation answers BelongsToMany::hydratePivotRelation.
func (r *BelongsToMany) hydratePivotRelation(models []Model) {
	for _, model := range models {
		model.SetRelation(r.accessor, r.NewExistingPivot(r.migratePivotAttributes(model)))
	}
}

// migratePivotAttributes answers BelongsToMany::migratePivotAttributes.
func (r *BelongsToMany) migratePivotAttributes(model Model) map[string]any {
	values := map[string]any{}

	attributes := model.GetAttributes()
	keys := make([]string, 0, len(attributes))
	for key := range attributes {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	for _, key := range keys {
		if !strings.HasPrefix(key, "pivot_") {
			continue
		}
		values[strings.TrimPrefix(key, "pivot_")] = attributes[key]
		model.UnsetAttribute(key)
	}

	return values
}

// Using answers BelongsToMany::using.
//
// The PHP takes a class name and instantiates it. Here it takes the factory,
// because a name is not a type at run time and a registry keyed by string would
// be a second morph map for a problem that does not need one.
func (r *BelongsToMany) Using(factory PivotFactory) *BelongsToMany {
	r.using = factory
	r.InteractsWithPivotTable.Using = factory != nil
	return r
}

// As answers BelongsToMany::as: the accessor the pivot is read under, for a
// relation where "pivot" reads wrong -- $role->subscription rather than
// $role->pivot.
func (r *BelongsToMany) As(accessor string) *BelongsToMany {
	r.accessor = accessor
	return r
}

// WherePivot answers BelongsToMany::wherePivot.
func (r *BelongsToMany) WherePivot(column string, args ...any) *BelongsToMany {
	qualified := r.QualifyPivotColumn(column)
	r.pivotWheres = append(r.pivotWheres, func(q *query.Builder) { q.Where(qualified, args...) })
	r.Query.Where(qualified, args...)
	return r
}

// WherePivotIn answers BelongsToMany::wherePivotIn.
func (r *BelongsToMany) WherePivotIn(column string, values []any) *BelongsToMany {
	qualified := r.QualifyPivotColumn(column)
	r.pivotWhereIns = append(r.pivotWhereIns, func(q *query.Builder) { q.WhereIn(qualified, values) })
	r.Query.WhereIn(qualified, values)
	return r
}

// WherePivotNotIn answers BelongsToMany::wherePivotNotIn.
func (r *BelongsToMany) WherePivotNotIn(column string, values []any) *BelongsToMany {
	qualified := r.QualifyPivotColumn(column)
	r.pivotWhereIns = append(r.pivotWhereIns, func(q *query.Builder) { q.WhereNotIn(qualified, values) })
	r.Query.GetQuery().WhereNotIn(qualified, values)
	return r
}

// WherePivotNull answers BelongsToMany::wherePivotNull.
func (r *BelongsToMany) WherePivotNull(column string) *BelongsToMany {
	qualified := r.QualifyPivotColumn(column)
	r.pivotWhereNulls = append(r.pivotWhereNulls, func(q *query.Builder) { q.WhereNull(qualified) })
	r.Query.GetQuery().WhereNull(qualified)
	return r
}

// WherePivotNotNull answers BelongsToMany::wherePivotNotNull.
func (r *BelongsToMany) WherePivotNotNull(column string) *BelongsToMany {
	qualified := r.QualifyPivotColumn(column)
	r.pivotWhereNulls = append(r.pivotWhereNulls, func(q *query.Builder) { q.WhereNotNull(qualified) })
	r.Query.WhereNotNull(qualified)
	return r
}

// WherePivotBetween answers BelongsToMany::wherePivotBetween.
func (r *BelongsToMany) WherePivotBetween(column string, from, to any) *BelongsToMany {
	qualified := r.QualifyPivotColumn(column)
	r.pivotWheres = append(r.pivotWheres, func(q *query.Builder) { q.WhereBetween(qualified, from, to) })
	r.Query.GetQuery().WhereBetween(qualified, from, to)
	return r
}

// WithPivotValue answers BelongsToMany::withPivotValue: a pivot column that is
// both a filter on read and a default on write.
//
// It carries an error where the PHP throws InvalidArgumentException, for the
// same case: a null value would be a filter that matches nothing and a default
// that writes nothing.
func (r *BelongsToMany) WithPivotValue(column string, value any) (*BelongsToMany, error) {
	if value == nil {
		return r, errors.New("relations: the value provided to withPivotValue may not be null")
	}
	r.PivotValues = append(r.PivotValues, concerns.PivotValue{Column: column, Value: value})
	return r.WherePivot(column, "=", value), nil
}

// OrderByPivot answers BelongsToMany::orderByPivot.
func (r *BelongsToMany) OrderByPivot(column string, direction ...string) *BelongsToMany {
	r.Query.OrderBy(r.QualifyPivotColumn(column), direction...)
	return r
}

// WithTimestamps answers BelongsToMany::withTimestamps.
func (r *BelongsToMany) WithTimestamps(columns ...string) *BelongsToMany {
	r.withTimestamps = true

	if len(columns) > 0 && columns[0] != "" {
		r.pivotCreatedAt = columns[0]
	}
	if len(columns) > 1 && columns[1] != "" {
		r.pivotUpdatedAt = columns[1]
	}

	return r.WithPivot(r.CreatedAt(), r.UpdatedAt())
}

// WithPivot answers InteractsWithPivotTable::withPivot, redeclared so it
// returns the relation rather than the embedded struct and the chain reads like
// the PHP's.
func (r *BelongsToMany) WithPivot(columns ...string) *BelongsToMany {
	r.InteractsWithPivotTable.WithPivot(columns...)
	return r
}

// CreatedAt answers BelongsToMany::createdAt.
func (r *BelongsToMany) CreatedAt() string {
	if r.pivotCreatedAt != "" {
		return r.pivotCreatedAt
	}
	return r.Parent.GetCreatedAtColumn()
}

// UpdatedAt answers BelongsToMany::updatedAt.
func (r *BelongsToMany) UpdatedAt() string {
	if r.pivotUpdatedAt != "" {
		return r.pivotUpdatedAt
	}
	return r.Parent.GetUpdatedAtColumn()
}

// Save answers BelongsToMany::save: save the related model, then attach it.
func (r *BelongsToMany) Save(ctx context.Context, g auth.Grant, model Model, pivotAttributes map[string]any, touch ...bool) (Model, error) {
	if err := model.Save(ctx, g); err != nil {
		return nil, err
	}
	if err := r.Attach(ctx, g, model.GetAttribute(r.relatedKey), pivotAttributes, touch...); err != nil {
		return nil, err
	}
	return model, nil
}

// SaveMany answers BelongsToMany::saveMany.
func (r *BelongsToMany) SaveMany(ctx context.Context, g auth.Grant, models []Model, pivotAttributes map[string]any) ([]Model, error) {
	for _, model := range models {
		if _, err := r.Save(ctx, g, model, pivotAttributes, false); err != nil {
			return nil, err
		}
	}
	if err := r.TouchIfTouching(ctx, g); err != nil {
		return nil, err
	}
	return models, nil
}

// Create answers BelongsToMany::create.
func (r *BelongsToMany) Create(ctx context.Context, g auth.Grant, attributes map[string]any, joining map[string]any, touch ...bool) (Model, error) {
	instance := r.Related.NewInstance(attributes)
	if err := instance.Save(ctx, g); err != nil {
		return nil, err
	}
	if err := r.Attach(ctx, g, instance.GetAttribute(r.relatedKey), joining, touch...); err != nil {
		return nil, err
	}
	return instance, nil
}

// CreateOrFirst answers BelongsToMany::createOrFirst: create the related row
// and attach it, and if somebody else created it first, attach theirs.
//
// It is the same race HasOneOrMany::createOrFirst loses, with one more way to
// lose it, and the PHP body is two try blocks for exactly that reason. The first
// insert can collide on the related table's unique index; the recovery then
// finds the row somebody else wrote and attaches it, and that attach can collide
// on the pivot's own unique index, because the other request is attaching it
// too. The second collision is success -- the row exists and the link exists,
// which is what the caller asked for -- so it is answered with the row rather
// than the error.
//
// Every error that is not a unique violation is returned. A caller that gets a
// row back from this method can rely on the link existing; one that got a row
// back from a swallowed error could not.
func (r *BelongsToMany) CreateOrFirst(ctx context.Context, g auth.Grant, attributes map[string]any, values map[string]any, joining map[string]any, touch ...bool) (Model, error) {
	instance, err := r.Create(ctx, g, merge(attributes, values), joining, touch...)
	if err == nil {
		return instance, nil
	}

	var violation *database.UniqueConstraintViolationException
	if !errors.As(err, &violation) {
		return nil, err
	}

	existing, findErr := r.firstWhere(ctx, g, attributes)
	if findErr != nil {
		return nil, findErr
	}
	if existing == nil {
		// Nothing matches the attributes, so the index that rejected the insert
		// was not the one these attributes name. The original error says more.
		return nil, err
	}

	attachErr := r.Attach(ctx, g, existing.GetAttribute(r.relatedKey), joining, touch...)
	if attachErr == nil {
		return existing, nil
	}
	if errors.As(attachErr, &violation) {
		// The pivot row is already there, which is the state Attach was asked
		// to reach. Another request reached it first.
		return existing, nil
	}
	return nil, attachErr
}

// firstWhere reads the first related row matching the attributes.
//
// The attributes are applied in sorted order: a Go map has no order, and a where
// clause whose columns reorder between runs is a different SQL string every
// time.
//
// It reads through the related model's own query rather than the relation's.
// The relation's query is joined to the pivot, and the row being looked for is
// the one that exists but is not attached yet -- so a lookup through the join
// would answer nothing and the method would loop back to the error it just
// caught. The tenant filter is applied all the same: which query object the read
// went out on changes nothing.
func (r *BelongsToMany) firstWhere(ctx context.Context, g auth.Grant, attributes map[string]any) (Model, error) {
	q := r.Related.NewQuery()
	if q == nil {
		return nil, nil
	}

	scoped, err := concerns.ScopeTenant(q, r.Related, g)
	if err != nil {
		return nil, err
	}

	columns := make([]string, 0, len(attributes))
	for column := range attributes {
		columns = append(columns, column)
	}
	sort.Strings(columns)

	for _, column := range columns {
		scoped = scoped.Where(column, attributes[column])
	}
	return scoped.First(ctx, g)
}

// CreateMany answers BelongsToMany::createMany.
func (r *BelongsToMany) CreateMany(ctx context.Context, g auth.Grant, records []map[string]any, joinings []map[string]any) ([]Model, error) {
	instances := make([]Model, 0, len(records))
	for index, record := range records {
		var joining map[string]any
		if index < len(joinings) {
			joining = joinings[index]
		}
		instance, err := r.Create(ctx, g, record, joining, false)
		if err != nil {
			return nil, err
		}
		instances = append(instances, instance)
	}
	if err := r.TouchIfTouching(ctx, g); err != nil {
		return nil, err
	}
	return instances, nil
}

// TouchIfTouching answers BelongsToMany::touchIfTouching.
func (r *BelongsToMany) TouchIfTouching(ctx context.Context, g auth.Grant) error {
	if r.Parent.Touches(r.relationName) {
		return r.Touch(ctx, g)
	}
	return nil
}

// Touch answers BelongsToMany::touch: it stamps the related rows this relation
// reaches, found through the pivot table rather than through the join.
func (r *BelongsToMany) Touch(ctx context.Context, g auth.Grant) error {
	if !r.Related.UsesTimestamps() {
		return nil
	}

	ids, err := r.AllRelatedIDs(ctx, g)
	if err != nil {
		return err
	}
	if len(ids) == 0 {
		return nil
	}

	scoped, err := concerns.ScopeTenant(r.Related.NewQuery(), r.Related, g)
	if err != nil {
		return err
	}

	_, err = scoped.WhereKey(ids...).Update(ctx, g, map[string]any{
		r.Related.GetUpdatedAtColumn(): r.Related.FreshTimestamp(),
	})
	return err
}

// NewPivotStatement answers BelongsToMany::newPivotStatement.
//
// It is the base query builder over the intermediate table, and it comes back
// already filtered by the Grant's tenant. That filter is not optional and not
// the caller's job: attach, detach, sync and toggle all reach the table through
// here, and a pivot table is shared by every tenant in the system.
func (r *BelongsToMany) NewPivotStatement(g auth.Grant) (*query.Builder, error) {
	return concerns.ScopeTenantQuery(r.Query.GetQuery().NewQuery().From(r.table), r.table, g)
}

// NewPivotQuery answers BelongsToMany::newPivotQuery.
func (r *BelongsToMany) NewPivotQuery(g auth.Grant) (*query.Builder, error) {
	statement, err := r.NewPivotStatement(g)
	if err != nil {
		return nil, err
	}

	for _, constraint := range r.pivotWheres {
		constraint(statement)
	}
	for _, constraint := range r.pivotWhereIns {
		constraint(statement)
	}
	for _, constraint := range r.pivotWhereNulls {
		constraint(statement)
	}

	return statement.Where(r.GetQualifiedForeignPivotKeyName(), r.ParentKeyValue()), nil
}

// NewPivot answers BelongsToMany::newPivot.
func (r *BelongsToMany) NewPivot(attributes map[string]any, exists bool) Model {
	values := make(map[string]any, len(attributes)+len(r.PivotValues))
	for _, value := range r.PivotValues {
		values[value.Column] = value.Value
	}
	for key, value := range attributes {
		values[key] = value
	}

	if r.using != nil {
		return r.using(r.Parent, values, r.table, exists)
	}

	pivot := FromAttributes(r.Parent, values, r.table, exists)
	pivot.NewQueryFor = r.newQueryForTable
	pivot.SetPivotKeys(r.foreignPivotKey, r.relatedPivotKey)
	pivot.SetRelatedModel(r.Related)
	return pivot
}

// newQueryForTable is how a Pivot this relation built reaches the database.
func (r *BelongsToMany) newQueryForTable(table string) Builder {
	q := r.Related.NewQuery()
	q.GetQuery().From(table)
	return q
}

// GetRelationExistenceQuery answers BelongsToMany::getRelationExistenceQuery.
func (r *BelongsToMany) GetRelationExistenceQuery(q Builder, parentQuery Builder, columns ...any) Builder {
	if sameTable(q, parentQuery) {
		return r.getRelationExistenceQueryForSelfJoin(q, parentQuery, columns...)
	}

	r.performJoin(q)
	return r.BaseRelation.GetRelationExistenceQuery(q, parentQuery, columns...)
}

// getRelationExistenceQueryForSelfJoin answers
// BelongsToMany::getRelationExistenceQueryForSelfJoin.
func (r *BelongsToMany) getRelationExistenceQueryForSelfJoin(q Builder, parentQuery Builder, columns ...any) Builder {
	if len(columns) == 0 {
		columns = []any{"*"}
	}

	hash := r.GetRelationCountHash()
	q.Select(columns...).GetQuery().From(r.Related.GetTable() + " as " + hash)
	q.GetModel().SetTable(hash)

	return q.WhereColumn(r.GetQualifiedParentKeyName(), "=", r.GetExistenceCompareKey())
}

// GetExistenceCompareKey answers BelongsToMany::getExistenceCompareKey.
func (r *BelongsToMany) GetExistenceCompareKey() string {
	return r.GetQualifiedForeignPivotKeyName()
}

// Limit answers BelongsToMany::limit.
func (r *BelongsToMany) Limit(value int) *BelongsToMany {
	if r.Parent.Exists() {
		r.Query.Limit(value)
	} else {
		r.Query.GetQuery().GroupLimit(value, r.GetQualifiedForeignPivotKeyName())
	}
	return r
}

// Take answers BelongsToMany::take.
func (r *BelongsToMany) Take(value int) *BelongsToMany { return r.Limit(value) }

// GetPivotClass answers BelongsToMany::getPivotClass.
//
// The PHP answers a class name; the equivalent here is the factory that builds
// the pivot, which is what using() takes. nil means the plain Pivot.
func (r *BelongsToMany) GetPivotClass() PivotFactory { return r.using }

// GetTable answers BelongsToMany::getTable: the intermediate table.
func (r *BelongsToMany) GetTable() string { return r.table }

// GetRelationName answers BelongsToMany::getRelationName.
func (r *BelongsToMany) GetRelationName() string { return r.relationName }

// GetPivotAccessor answers BelongsToMany::getPivotAccessor.
func (r *BelongsToMany) GetPivotAccessor() string { return r.accessor }

// GetPivotColumns answers BelongsToMany::getPivotColumns.
func (r *BelongsToMany) GetPivotColumns() []string { return r.PivotColumns }

// GetForeignPivotKeyName answers BelongsToMany::getForeignPivotKeyName.
func (r *BelongsToMany) GetForeignPivotKeyName() string { return r.foreignPivotKey }

// GetQualifiedForeignPivotKeyName answers the method of the same name.
func (r *BelongsToMany) GetQualifiedForeignPivotKeyName() string {
	return r.QualifyPivotColumn(r.foreignPivotKey)
}

// GetRelatedPivotKeyName answers BelongsToMany::getRelatedPivotKeyName.
func (r *BelongsToMany) GetRelatedPivotKeyName() string { return r.relatedPivotKey }

// GetQualifiedRelatedPivotKeyName answers the method of the same name.
func (r *BelongsToMany) GetQualifiedRelatedPivotKeyName() string {
	return r.QualifyPivotColumn(r.relatedPivotKey)
}

// GetParentKeyName answers BelongsToMany::getParentKeyName.
func (r *BelongsToMany) GetParentKeyName() string { return r.parentKey }

// GetQualifiedParentKeyName answers BelongsToMany::getQualifiedParentKeyName.
func (r *BelongsToMany) GetQualifiedParentKeyName() string {
	return r.Parent.QualifyColumn(r.parentKey)
}

// GetRelatedKeyName answers BelongsToMany::getRelatedKeyName.
func (r *BelongsToMany) GetRelatedKeyName() string { return r.relatedKey }

// GetQualifiedRelatedKeyName answers BelongsToMany::getQualifiedRelatedKeyName.
func (r *BelongsToMany) GetQualifiedRelatedKeyName() string {
	return r.Related.QualifyColumn(r.relatedKey)
}

// QualifyPivotColumn answers BelongsToMany::qualifyPivotColumn.
func (r *BelongsToMany) QualifyPivotColumn(column string) string {
	return qualifyColumn(r.table, column)
}

// ParentKeyValue answers the PHP's $this->parent->{$this->parentKey}, which the
// pivot trait reads through the host contract.
func (r *BelongsToMany) ParentKeyValue() any { return r.Parent.GetAttribute(r.parentKey) }

// NewBelongsToManyUnconstrained builds the relation without narrowing it to one parent.
//
// It exists because the constraint used to be switched off through a
// process-wide flag, which meant a relation built on another goroutine while
// the flag was down came back unconstrained -- every parent's children, in a
// well-formed query nobody could tell apart from the right one. The call site
// says which it wants now, and there is no flag to leave down.
func NewBelongsToManyUnconstrained(query Builder, parent Model, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relationName string) *BelongsToMany {
	return newBelongsToMany(query, parent, table, foreignPivotKey, relatedPivotKey, parentKey, relatedKey, relationName)
}
