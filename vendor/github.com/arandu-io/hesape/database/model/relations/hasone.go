package relations

import (
	"context"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database/model/relations/concerns"
	"github.com/arandu-io/hesape/database/query"
)

// HasOne is one row on the other table, found by a foreign key pointing here.
//
// It is HasMany matching one row instead of many, plus the three shared halves
// that make it comparable, defaultable and reducible to one of many.
type HasOne struct {
	HasOneOrMany
	concerns.SupportsDefaultModels
	concerns.ComparesRelatedModels
	concerns.CanBeOneOfMany
}

// NewHasOne builds a HasOne over query for parent, joining on foreignKey and
// localKey, wires its embedded concerns.SupportsDefaultModels,
// concerns.ComparesRelatedModels and concerns.CanBeOneOfMany to call back
// into the relation itself, applies its constraints, and returns it.
func newHasOne(query Builder, parent Model, foreignKey, localKey string) *HasOne {
	relation := &HasOne{HasOneOrMany: NewHasOneOrMany(query, parent, foreignKey, localKey)}
	relation.ExistenceCompareKey = relation.GetExistenceCompareKey

	relation.SupportsDefaultModels = concerns.SupportsDefaultModels{
		NewRelatedInstanceFor: relation.NewRelatedInstanceFor,
	}
	relation.ComparesRelatedModels = concerns.ComparesRelatedModels{
		CompareRelated:        relation.Related,
		CompareParentKey:      relation.GetParentKey,
		CompareRelatedKeyFrom: relation.getRelatedKeyFrom,
	}
	relation.CanBeOneOfMany = concerns.CanBeOneOfMany{
		OneOfManyQuery:                      relation.GetQuery,
		OneOfManyRelated:                    relation.GetRelated,
		OneOfManyAddConstraints:             relation.AddConstraints,
		AddOneOfManySubQueryConstraints:     relation.AddOneOfManySubQueryConstraints,
		GetOneOfManySubQuerySelectColumns:   relation.GetOneOfManySubQuerySelectColumns,
		AddOneOfManyJoinSubQueryConstraints: relation.AddOneOfManyJoinSubQueryConstraints,
	}

	return relation
}

// NewHasOne builds the relation and narrows it to its parent.
func NewHasOne(query Builder, parent Model, foreignKey, localKey string) *HasOne {
	relation := newHasOne(query, parent, foreignKey, localKey)
	relation.AddConstraints()
	return relation
}

// GetRelationQuery builds the relation's query through the embedded
// CanBeOneOfMany, using r.Query as its base builder. It overrides the method
// promoted from Relation because on a one-of-many relation, the constraints
// belong on the subquery.
func (r *HasOne) GetRelationQuery() Builder {
	return r.CanBeOneOfMany.GetRelationQuery(r.Query)
}

// AddConstraints redeclares HasOneOrMany's constraint logic so that it
// reaches the one-of-many subquery when there is one. Go promotes the
// embedded method but not the overridden GetRelationQuery it calls, which is
// the whole of what "no virtual dispatch" costs.
func (r *HasOne) AddConstraints() {
	q := r.GetRelationQuery()
	q.Where(r.GetQualifiedForeignKeyName(), "=", r.GetParentKey())
	q.WhereNotNull(r.GetQualifiedForeignKeyName())
}

// GetResults returns the first related model, or the relation's default
// model if the parent has no key yet or no related row exists.
func (r *HasOne) GetResults(ctx context.Context, g auth.Grant) (any, error) {
	if r.GetParentKey() == nil {
		return r.GetDefaultFor(r.Parent), nil
	}

	result, err := r.First(ctx, g)
	if err != nil {
		return nil, err
	}
	if result == nil {
		return r.GetDefaultFor(r.Parent), nil
	}
	return result, nil
}

// InitRelation seeds relation on every model in models with that model's
// default related instance.
func (r *HasOne) InitRelation(models []Model, relation string) []Model {
	for _, model := range models {
		model.SetRelation(relation, r.GetDefaultFor(model))
	}
	return models
}

// Match assigns each model in models the one result from results that
// belongs to it, via MatchOne, and stores it under relation.
func (r *HasOne) Match(models []Model, results []Model, relation string) ([]Model, error) {
	return r.MatchOne(models, results, relation)
}

// GetRelationExistenceQuery merges the one-of-many joins into q when the
// relation is one-of-many, then delegates to HasOneOrMany's existence query.
func (r *HasOne) GetRelationExistenceQuery(q Builder, parentQuery Builder, columns ...any) Builder {
	if r.IsOneOfMany() {
		r.MergeOneOfManyJoinsTo(q)
	}
	return r.HasOneOrMany.GetRelationExistenceQuery(q, parentQuery, columns...)
}

// AddOneOfManySubQueryConstraints adds the relation's qualified foreign key
// to q's select list.
func (r *HasOne) AddOneOfManySubQueryConstraints(q Builder, column, aggregate string) {
	q.AddSelect(r.GetQualifiedForeignKeyName())
}

// GetOneOfManySubQuerySelectColumns returns the relation's qualified foreign
// key as the sole column to select in the one-of-many subquery.
func (r *HasOne) GetOneOfManySubQuerySelectColumns() []any {
	return []any{r.GetQualifiedForeignKeyName()}
}

// AddOneOfManyJoinSubQueryConstraints joins the subquery to the related table
// by equating their foreign key columns.
func (r *HasOne) AddOneOfManyJoinSubQueryConstraints(join *query.JoinClause) {
	join.On(
		r.QualifySubSelectColumn(r.GetForeignKeyName()),
		"=",
		r.QualifyRelatedColumn(r.GetForeignKeyName()),
	)
}

// NewRelatedInstanceFor returns a new related model instance with its
// foreign key already set to parent's local key, and the inverse relation
// applied.
func (r *HasOne) NewRelatedInstanceFor(parent Model) Model {
	instance := r.Related.NewInstance(nil)
	instance.SetAttribute(r.GetForeignKeyName(), parent.GetAttribute(r.GetLocalKeyName()))
	r.ApplyInverseRelationToModel(instance, parent)
	return instance
}

// getRelatedKeyFrom returns model's value for the relation's foreign key.
func (r *HasOne) getRelatedKeyFrom(model Model) any {
	return model.GetAttribute(r.GetForeignKeyName())
}

// NewHasOneUnconstrained builds the relation without narrowing it to one parent.
//
// It exists because the constraint used to be switched off through a
// process-wide flag, which meant a relation built on another goroutine while
// the flag was down came back unconstrained -- every parent's children, in a
// well-formed query nobody could tell apart from the right one. The call site
// says which it wants now, and there is no flag to leave down.
func NewHasOneUnconstrained(query Builder, parent Model, foreignKey, localKey string) *HasOne {
	return newHasOne(query, parent, foreignKey, localKey)
}
