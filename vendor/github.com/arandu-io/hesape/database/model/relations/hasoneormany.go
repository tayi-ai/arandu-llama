package relations

import (
	"context"
	"errors"
	"sort"
	"strings"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database"
	"github.com/arandu-io/hesape/database/model/relations/concerns"
	"github.com/arandu-io/hesape/str"
)

// HasOneOrMany is the shared body of HasOne and HasMany.
//
// The relation is a foreign key on the other table pointing back at this one.
// Everything below follows from that: the constraint is `where posts.user_id =
// ?`, the eager constraint is the same clause widened to `in (?, ?, ?)`, and
// the match keys the results by that column.
type HasOneOrMany struct {
	BaseRelation
	concerns.SupportsInverseRelations

	foreignKey string
	localKey   string
}

// NewHasOneOrMany answers HasOneOrMany::__construct.
//
// It does not call AddConstraints. The PHP constructor does, through the
// subclass; Go has no virtual dispatch, so HasOne and HasMany call their own as
// the last statement of theirs.
func NewHasOneOrMany(query Builder, parent Model, foreignKey, localKey string) HasOneOrMany {
	relation := HasOneOrMany{
		BaseRelation: NewBaseRelation(query, parent),
		foreignKey:   foreignKey,
		localKey:     localKey,
	}
	relation.SupportsInverseRelations = concerns.SupportsInverseRelations{
		InverseRelatedModel: func() Model { return relation.Related },
		InverseParent:       func() Model { return relation.Parent },
	}
	relation.SupportsInverseRelations.PossibleInverseRelations = func() []string {
		return concerns.DefaultPossibleInverseRelations(relation.GetForeignKeyName(), relation.Parent, relation.Related)
	}
	return relation
}

// AddConstraints answers HasOneOrMany::addConstraints.
//
// The second clause is not redundant. Without `whereNotNull`, a parent whose
// local key is null would match every child whose foreign key is null -- on
// engines where that comparison holds -- and produce a relation full of other
// people's orphans.
func (r *HasOneOrMany) AddConstraints() {
	q := r.GetRelationQuery()
	q.Where(r.foreignKey, "=", r.GetParentKey())
	q.WhereNotNull(r.foreignKey)
}

// AddEagerConstraints answers HasOneOrMany::addEagerConstraints.
//
// This is one half of what makes with() two queries instead of N: every
// parent's key goes into one `in (...)`, so the children of a hundred parents
// arrive in a single round trip.
func (r *HasOneOrMany) AddEagerConstraints(models []Model) error {
	keys, err := r.GetKeys(models, r.localKey)
	if err != nil {
		return err
	}
	r.WhereInEager(r.foreignKey, keys, r.GetRelationQuery())
	return nil
}

// MatchOne answers HasOneOrMany::matchOne.
func (r *HasOneOrMany) MatchOne(models []Model, results []Model, relation string) ([]Model, error) {
	return r.matchOneOrMany(models, results, relation, "one")
}

// MatchMany answers HasOneOrMany::matchMany.
func (r *HasOneOrMany) MatchMany(models []Model, results []Model, relation string) ([]Model, error) {
	return r.matchOneOrMany(models, results, relation, "many")
}

// matchOneOrMany answers HasOneOrMany::matchOneOrMany.
//
// The other half of the two-query eager load. The results arrive flat; they are
// bucketed once by foreign key and then handed out, so the cost is linear in
// the number of rows rather than quadratic in parents times children -- which
// is what a nested loop over the parents would have cost, and what makes people
// think eager loading is slow.
func (r *HasOneOrMany) matchOneOrMany(models []Model, results []Model, relation string, typ string) ([]Model, error) {
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

		matched, ok := dictionary[key]
		if !ok {
			continue
		}

		if typ == "one" {
			model.SetRelation(relation, matched[0])
			r.ApplyInverseRelationToModel(matched[0], model)
			continue
		}

		model.SetRelation(relation, matched)
		r.ApplyInverseRelationToCollection(matched, model)
	}

	return models, nil
}

// buildDictionary answers HasOneOrMany::buildDictionary.
func (r *HasOneOrMany) buildDictionary(results []Model) (map[string][]Model, error) {
	foreign := r.GetForeignKeyName()

	dictionary := make(map[string][]Model, len(results))
	for _, result := range results {
		key, err := concerns.GetDictionaryKey(result.GetAttribute(foreign))
		if err != nil {
			return nil, err
		}
		dictionary[key] = append(dictionary[key], result)
	}
	return dictionary, nil
}

// Make answers HasOneOrMany::make: an unsaved related model with the foreign
// key already set.
func (r *HasOneOrMany) Make(attributes map[string]any) Model {
	instance := r.Related.NewInstance(attributes)
	r.SetForeignAttributesForCreate(instance)
	return instance
}

// MakeMany answers HasOneOrMany::makeMany.
func (r *HasOneOrMany) MakeMany(records []map[string]any) []Model {
	instances := make([]Model, 0, len(records))
	for _, record := range records {
		instances = append(instances, r.Make(record))
	}
	return instances
}

// Save answers HasOneOrMany::save.
func (r *HasOneOrMany) Save(ctx context.Context, g auth.Grant, model Model) (Model, error) {
	r.SetForeignAttributesForCreate(model)
	if err := model.Save(ctx, g); err != nil {
		return nil, err
	}
	return model, nil
}

// SaveQuietly answers HasOneOrMany::saveQuietly.
func (r *HasOneOrMany) SaveQuietly(ctx context.Context, g auth.Grant, model Model) (Model, error) {
	var saved Model
	err := model.WithoutEvents(func() error {
		var err error
		saved, err = r.Save(ctx, g, model)
		return err
	})
	return saved, err
}

// SaveMany answers HasOneOrMany::saveMany.
func (r *HasOneOrMany) SaveMany(ctx context.Context, g auth.Grant, models []Model) ([]Model, error) {
	for _, model := range models {
		if _, err := r.Save(ctx, g, model); err != nil {
			return nil, err
		}
	}
	return models, nil
}

// SaveManyQuietly answers HasOneOrMany::saveManyQuietly.
func (r *HasOneOrMany) SaveManyQuietly(ctx context.Context, g auth.Grant, models []Model) ([]Model, error) {
	if len(models) == 0 {
		return models, nil
	}
	var saved []Model
	err := models[0].WithoutEvents(func() error {
		var err error
		saved, err = r.SaveMany(ctx, g, models)
		return err
	})
	return saved, err
}

// Create answers HasOneOrMany::create.
func (r *HasOneOrMany) Create(ctx context.Context, g auth.Grant, attributes map[string]any) (Model, error) {
	instance := r.Related.NewInstance(attributes)
	r.SetForeignAttributesForCreate(instance)

	if err := instance.Save(ctx, g); err != nil {
		return nil, err
	}
	r.ApplyInverseRelationToModel(instance)
	return instance, nil
}

// CreateQuietly answers HasOneOrMany::createQuietly.
func (r *HasOneOrMany) CreateQuietly(ctx context.Context, g auth.Grant, attributes map[string]any) (Model, error) {
	var created Model
	err := r.Related.WithoutEvents(func() error {
		var err error
		created, err = r.Create(ctx, g, attributes)
		return err
	})
	return created, err
}

// CreateMany answers HasOneOrMany::createMany.
func (r *HasOneOrMany) CreateMany(ctx context.Context, g auth.Grant, records []map[string]any) ([]Model, error) {
	instances := make([]Model, 0, len(records))
	for _, record := range records {
		instance, err := r.Create(ctx, g, record)
		if err != nil {
			return nil, err
		}
		instances = append(instances, instance)
	}
	return instances, nil
}

// CreateManyQuietly answers HasOneOrMany::createManyQuietly.
func (r *HasOneOrMany) CreateManyQuietly(ctx context.Context, g auth.Grant, records []map[string]any) ([]Model, error) {
	var created []Model
	err := r.Related.WithoutEvents(func() error {
		var err error
		created, err = r.CreateMany(ctx, g, records)
		return err
	})
	return created, err
}

// ForceCreate answers HasOneOrMany::forceCreate: mass assignment allowed.
func (r *HasOneOrMany) ForceCreate(ctx context.Context, g auth.Grant, attributes map[string]any) (Model, error) {
	filled := make(map[string]any, len(attributes)+1)
	for key, value := range attributes {
		filled[key] = value
	}
	filled[r.GetForeignKeyName()] = r.GetParentKey()

	instance := r.Related.NewInstance(nil)
	instance.ForceFill(filled)

	if err := instance.Save(ctx, g); err != nil {
		return nil, err
	}
	return r.ApplyInverseRelationToModel(instance), nil
}

// ForceCreateQuietly answers HasOneOrMany::forceCreateQuietly.
func (r *HasOneOrMany) ForceCreateQuietly(ctx context.Context, g auth.Grant, attributes map[string]any) (Model, error) {
	var created Model
	err := r.Related.WithoutEvents(func() error {
		var err error
		created, err = r.ForceCreate(ctx, g, attributes)
		return err
	})
	return created, err
}

// ForceCreateMany answers HasOneOrMany::forceCreateMany.
func (r *HasOneOrMany) ForceCreateMany(ctx context.Context, g auth.Grant, records []map[string]any) ([]Model, error) {
	instances := make([]Model, 0, len(records))
	for _, record := range records {
		instance, err := r.ForceCreate(ctx, g, record)
		if err != nil {
			return nil, err
		}
		instances = append(instances, instance)
	}
	return instances, nil
}

// ForceCreateManyQuietly answers HasOneOrMany::forceCreateManyQuietly.
func (r *HasOneOrMany) ForceCreateManyQuietly(ctx context.Context, g auth.Grant, records []map[string]any) ([]Model, error) {
	var created []Model
	err := r.Related.WithoutEvents(func() error {
		var err error
		created, err = r.ForceCreateMany(ctx, g, records)
		return err
	})
	return created, err
}

// FindOrNew answers HasOneOrMany::findOrNew.
func (r *HasOneOrMany) FindOrNew(ctx context.Context, g auth.Grant, id any) (Model, error) {
	scoped, err := r.scoped(g)
	if err != nil {
		return nil, err
	}

	instance, err := scoped.Find(ctx, g, id)
	if err != nil {
		return nil, err
	}
	if instance != nil {
		return instance, nil
	}

	instance = r.Related.NewInstance(nil)
	r.SetForeignAttributesForCreate(instance)
	return instance, nil
}

// FirstOrNew answers HasOneOrMany::firstOrNew.
func (r *HasOneOrMany) FirstOrNew(ctx context.Context, g auth.Grant, attributes, values map[string]any) (Model, error) {
	instance, err := r.firstWhere(ctx, g, attributes)
	if err != nil {
		return nil, err
	}
	if instance != nil {
		return instance, nil
	}

	instance = r.Related.NewInstance(merge(attributes, values))
	r.SetForeignAttributesForCreate(instance)
	return instance, nil
}

// FirstOrCreate answers HasOneOrMany::firstOrCreate.
func (r *HasOneOrMany) FirstOrCreate(ctx context.Context, g auth.Grant, attributes, values map[string]any) (Model, error) {
	instance, err := r.firstWhere(ctx, g, attributes)
	if err != nil {
		return nil, err
	}
	if instance != nil {
		return instance, nil
	}
	return r.Create(ctx, g, merge(attributes, values))
}

// CreateOrFirst answers HasOneOrMany::createOrFirst: create the row, and if
// somebody else created it first, read theirs.
//
// It is the race FirstOrCreate cannot win on its own. Two requests that both
// find nothing both go on to insert, and one of them loses to the unique index;
// this catches exactly that loss and answers the row the winner wrote. Every
// other statement error is returned, because a NOT NULL violation answered with
// a silent select is a bug that looks like a missing record.
//
// Two things the PHP does are not here, and neither is available to reach.
// withSavepointIfNeeded belongs to ManagesTransactions, which this package does
// not hold; and useWritePdo names a read/write split that this connection does
// not offer. The retry reads through the same connection the insert used, which
// is what useWritePdo is asking for in the first place.
func (r *HasOneOrMany) CreateOrFirst(ctx context.Context, g auth.Grant, attributes, values map[string]any) (Model, error) {
	instance, err := r.Create(ctx, g, merge(attributes, values))
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
		// The PHP rethrows the violation here. Nothing matched the attributes,
		// so whatever the unique index rejected was not this row, and the
		// original error is the one that says what actually happened.
		return nil, err
	}
	return existing, nil
}

// Upsert answers HasOneOrMany::upsert: insert the rows, updating the ones that
// collide, with the foreign key stamped on every one.
//
// The stamping is the whole point of the override. Without it an upsert through
// a relation writes rows that belong to no parent, and they are invisible to
// the relation that wrote them.
//
// The PHP's first line wraps a single row in an array so that upsert(['a' => 1])
// and upsert([['a' => 1]]) both work. A Go signature says which one it takes, so
// that branch has nothing to decide and is gone -- one of the shapes PHP needs a
// run-time check for and Go gets from the type.
func (r *HasOneOrMany) Upsert(ctx context.Context, g auth.Grant, values []map[string]any, uniqueBy, update []string) (int64, error) {
	scoped, err := r.scoped(g)
	if err != nil {
		return 0, err
	}

	stamped := make([]map[string]any, 0, len(values))
	for _, row := range values {
		stamped = append(stamped, r.stampForeignKey(row))
	}

	return scoped.Upsert(ctx, g, stamped, uniqueBy, update)
}

// stampForeignKey copies row with the foreign key set to the parent's key.
//
// It copies rather than writing through, because the PHP mutates the caller's
// array and Go's caller keeps the same map: an upsert that silently added a
// column to the map the caller still holds would be a surprise on the second
// use of it. MorphOneOrMany wraps this to add the type column.
func (r *HasOneOrMany) stampForeignKey(row map[string]any) map[string]any {
	stamped := make(map[string]any, len(row)+1)
	for key, value := range row {
		stamped[key] = value
	}
	stamped[r.GetForeignKeyName()] = r.GetParentKey()
	return stamped
}

// UpdateOrCreate answers HasOneOrMany::updateOrCreate.
func (r *HasOneOrMany) UpdateOrCreate(ctx context.Context, g auth.Grant, attributes, values map[string]any) (Model, error) {
	instance, err := r.FirstOrCreate(ctx, g, attributes, values)
	if err != nil {
		return nil, err
	}
	if instance.WasRecentlyCreated() {
		return instance, nil
	}

	instance.Fill(values)
	if err := instance.Save(ctx, g); err != nil {
		return nil, err
	}
	return instance, nil
}

// firstWhere answers the PHP's `$this->where($attributes)->first()`.
//
// The attributes are applied in sorted order: a Go map has none, and a where
// clause whose column order changes between runs is a different SQL string
// every time, which defeats every cache that keys on it.
func (r *HasOneOrMany) firstWhere(ctx context.Context, g auth.Grant, attributes map[string]any) (Model, error) {
	scoped, err := r.scoped(g)
	if err != nil {
		return nil, err
	}
	for _, column := range sortedKeys(attributes) {
		scoped = scoped.Where(column, attributes[column])
	}
	return scoped.First(ctx, g)
}

// SetForeignAttributesForCreate answers
// HasOneOrMany::setForeignAttributesForCreate.
func (r *HasOneOrMany) SetForeignAttributesForCreate(model Model) {
	model.SetAttribute(r.GetForeignKeyName(), r.GetParentKey())
	r.ApplyInverseRelationToModel(model)
}

// Limit answers HasOneOrMany::limit.
//
// The branch is the one that makes `->latest()->limit(3)` mean "three per
// parent" on an eager load: with no parent row to narrow to, a plain limit
// would take three rows from the whole result set, so it becomes a per-group
// limit instead.
func (r *HasOneOrMany) Limit(value int) *HasOneOrMany {
	if r.Parent.Exists() {
		r.Query.Limit(value)
	} else {
		r.Query.GetQuery().GroupLimit(value, r.GetExistenceCompareKey())
	}
	return r
}

// Take answers HasOneOrMany::take.
func (r *HasOneOrMany) Take(value int) *HasOneOrMany { return r.Limit(value) }

// GetRelationExistenceQuery answers HasOneOrMany::getRelationExistenceQuery.
func (r *HasOneOrMany) GetRelationExistenceQuery(q Builder, parentQuery Builder, columns ...any) Builder {
	if sameTable(q, parentQuery) {
		return r.GetRelationExistenceQueryForSelfRelation(q, parentQuery, columns...)
	}
	return r.BaseRelation.GetRelationExistenceQuery(q, parentQuery, columns...)
}

// GetRelationExistenceQueryForSelfRelation answers
// HasOneOrMany::getRelationExistenceQueryForSelfRelation.
func (r *HasOneOrMany) GetRelationExistenceQueryForSelfRelation(q Builder, parentQuery Builder, columns ...any) Builder {
	if len(columns) == 0 {
		columns = []any{"*"}
	}
	hash := r.GetRelationCountHash()

	q.GetQuery().From(q.GetModel().GetTable() + " as " + hash)
	q.GetModel().SetTable(hash)

	return q.Select(columns...).WhereColumn(
		r.GetQualifiedParentKeyName(), "=", hash+"."+r.GetForeignKeyName(),
	)
}

// GetExistenceCompareKey answers HasOneOrMany::getExistenceCompareKey.
func (r *HasOneOrMany) GetExistenceCompareKey() string { return r.GetQualifiedForeignKeyName() }

// GetParentKey answers HasOneOrMany::getParentKey.
func (r *HasOneOrMany) GetParentKey() any { return r.Parent.GetAttribute(r.localKey) }

// GetQualifiedParentKeyName answers HasOneOrMany::getQualifiedParentKeyName.
func (r *HasOneOrMany) GetQualifiedParentKeyName() string {
	return r.Parent.QualifyColumn(r.localKey)
}

// GetForeignKeyName answers HasOneOrMany::getForeignKeyName: the plain column,
// with any table prefix cut off.
func (r *HasOneOrMany) GetForeignKeyName() string {
	segments := strings.Split(r.GetQualifiedForeignKeyName(), ".")
	return segments[len(segments)-1]
}

// GetQualifiedForeignKeyName answers
// HasOneOrMany::getQualifiedForeignKeyName.
func (r *HasOneOrMany) GetQualifiedForeignKeyName() string { return r.foreignKey }

// GetLocalKeyName answers HasOneOrMany::getLocalKeyName.
func (r *HasOneOrMany) GetLocalKeyName() string { return r.localKey }

func sameTable(left, right Builder) bool {
	return stringifyFrom(left) != "" && stringifyFrom(left) == stringifyFrom(right)
}

func stringifyFrom(b Builder) string {
	if from, ok := b.GetQuery().GetFrom().(string); ok {
		return from
	}
	return ""
}

func merge(left, right map[string]any) map[string]any {
	out := make(map[string]any, len(left)+len(right))
	for key, value := range left {
		out[key] = value
	}
	for key, value := range right {
		out[key] = value
	}
	return out
}

func sortedKeys(attributes map[string]any) []string {
	keys := make([]string, 0, len(attributes))
	for key := range attributes {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

// ForeignKeyFor answers Model::getForeignKey for a relation guessing its own
// key: the parent's morph alias, snake cased, plus the key name.
//
// It is here rather than on the model because a relation is the only caller,
// and because the alias -- not a class name -- is what names a model in this
// collection.
func ForeignKeyFor(model Model) string {
	return str.Snake(model.GetMorphClass(), "_") + "_" + model.GetKeyName()
}
