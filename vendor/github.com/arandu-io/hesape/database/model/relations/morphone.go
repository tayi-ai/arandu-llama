package relations

import (
	"context"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database/model/relations/concerns"
	"github.com/arandu-io/hesape/database/query"
)

// MorphOne is one row on a polymorphic table: the image of a post or of a user.
type MorphOne struct {
	MorphOneOrMany
	concerns.SupportsDefaultModels
	concerns.ComparesRelatedModels
	concerns.CanBeOneOfMany
}

// NewMorphOne answers MorphOne::__construct.
func newMorphOne(query Builder, parent Model, typ, id, localKey string) *MorphOne {
	relation := &MorphOne{MorphOneOrMany: NewMorphOneOrMany(query, parent, typ, id, localKey)}
	relation.ExistenceCompareKey = relation.GetExistenceCompareKey

	relation.SupportsInverseRelations.PossibleInverseRelations = relation.PossibleInverseRelations
	relation.SupportsDefaultModels = concerns.SupportsDefaultModels{
		NewRelatedInstanceFor: relation.newRelatedInstanceFor,
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

// NewMorphOne builds the relation and narrows it to its parent.
func NewMorphOne(query Builder, parent Model, typ, id, localKey string) *MorphOne {
	relation := newMorphOne(query, parent, typ, id, localKey)
	relation.AddConstraints()
	return relation
}

// GetRelationQuery answers CanBeOneOfMany::getRelationQuery.
func (r *MorphOne) GetRelationQuery() Builder {
	return r.CanBeOneOfMany.GetRelationQuery(r.Query)
}

// AddConstraints answers MorphOneOrMany::addConstraints, redeclared so that it
// reaches this type's GetRelationQuery.
func (r *MorphOne) AddConstraints() {
	q := r.GetRelationQuery()
	q.Where(r.GetQualifiedMorphType(), r.GetMorphClass())
	q.Where(r.GetQualifiedForeignKeyName(), "=", r.GetParentKey())
	q.WhereNotNull(r.GetQualifiedForeignKeyName())
}

// GetResults answers MorphOne::getResults.
func (r *MorphOne) GetResults(ctx context.Context, g auth.Grant) (any, error) {
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

// InitRelation answers MorphOne::initRelation.
func (r *MorphOne) InitRelation(models []Model, relation string) []Model {
	for _, model := range models {
		model.SetRelation(relation, r.GetDefaultFor(model))
	}
	return models
}

// Match answers MorphOne::match.
func (r *MorphOne) Match(models []Model, results []Model, relation string) ([]Model, error) {
	return r.MatchOne(models, results, relation)
}

// GetRelationExistenceQuery answers MorphOne::getRelationExistenceQuery.
func (r *MorphOne) GetRelationExistenceQuery(q Builder, parentQuery Builder, columns ...any) Builder {
	if r.IsOneOfMany() {
		r.MergeOneOfManyJoinsTo(q)
	}
	return r.MorphOneOrMany.GetRelationExistenceQuery(q, parentQuery, columns...)
}

// AddOneOfManySubQueryConstraints answers the method of the same name: the
// subquery has to carry the type column too, or the join would pick the newest
// comment of any commentable with that id.
func (r *MorphOne) AddOneOfManySubQueryConstraints(q Builder, column, aggregate string) {
	q.AddSelect(r.GetQualifiedForeignKeyName(), r.GetQualifiedMorphType())
}

// GetOneOfManySubQuerySelectColumns answers the method of the same name.
func (r *MorphOne) GetOneOfManySubQuerySelectColumns() []any {
	return []any{r.GetQualifiedForeignKeyName(), r.GetQualifiedMorphType()}
}

// AddOneOfManyJoinSubQueryConstraints answers the method of the same name.
func (r *MorphOne) AddOneOfManyJoinSubQueryConstraints(join *query.JoinClause) {
	join.
		On(r.QualifySubSelectColumn(r.GetMorphType()), "=", r.QualifyRelatedColumn(r.GetMorphType())).
		On(r.QualifySubSelectColumn(r.GetForeignKeyName()), "=", r.QualifyRelatedColumn(r.GetForeignKeyName()))
}

// newRelatedInstanceFor answers MorphOne::newRelatedInstanceFor.
func (r *MorphOne) newRelatedInstanceFor(parent Model) Model {
	instance := r.Related.NewInstance(nil)
	instance.SetAttribute(r.GetForeignKeyName(), parent.GetAttribute(r.GetLocalKeyName()))
	instance.SetAttribute(r.GetMorphType(), r.GetMorphClass())
	r.ApplyInverseRelationToModel(instance, parent)
	return instance
}

// getRelatedKeyFrom answers MorphOne::getRelatedKeyFrom.
func (r *MorphOne) getRelatedKeyFrom(model Model) any {
	return model.GetAttribute(r.GetForeignKeyName())
}

// NewMorphOneUnconstrained builds the relation without narrowing it to one parent.
//
// It exists because the constraint used to be switched off through a
// process-wide flag, which meant a relation built on another goroutine while
// the flag was down came back unconstrained -- every parent's children, in a
// well-formed query nobody could tell apart from the right one. The call site
// says which it wants now, and there is no flag to leave down.
func NewMorphOneUnconstrained(query Builder, parent Model, typ, id, localKey string) *MorphOne {
	return newMorphOne(query, parent, typ, id, localKey)
}
