package relations

import (
	"context"

	"github.com/arandu-io/hesape/auth"
)

// HasMany is every row on the other table whose foreign key points here.
type HasMany struct {
	HasOneOrMany
}

// NewHasMany builds a HasMany over query for parent, joining on foreignKey
// and localKey, applies its constraints, and returns it.
func newHasMany(query Builder, parent Model, foreignKey, localKey string) *HasMany {
	relation := &HasMany{HasOneOrMany: NewHasOneOrMany(query, parent, foreignKey, localKey)}
	relation.ExistenceCompareKey = relation.GetExistenceCompareKey
	return relation
}

// NewHasMany builds the relation and narrows it to its parent.
func NewHasMany(query Builder, parent Model, foreignKey, localKey string) *HasMany {
	relation := newHasMany(query, parent, foreignKey, localKey)
	relation.AddConstraints()
	return relation
}

// One returns the same relation read as a single model instead of a slice.
//
// It is built unconstrained, because the new HasOne would otherwise add a
// second copy of the where clause the HasMany already put on the shared query.
func (r *HasMany) One() *HasOne {
	one := NewHasOneUnconstrained(r.Query, r.Parent, r.GetQualifiedForeignKeyName(), r.GetLocalKeyName())
	if inverse := r.GetInverseRelationship(); inverse != "" {
		_ = one.Inverse(inverse)
	}
	return one
}

// GetResults returns every related model for the parent, or an empty slice
// if the parent has no key yet.
func (r *HasMany) GetResults(ctx context.Context, g auth.Grant) (any, error) {
	if r.GetParentKey() == nil {
		return []Model{}, nil
	}
	return r.Get(ctx, g)
}

// InitRelation seeds relation on every model in models with an empty slice.
//
// Seeding every parent with an empty collection is what makes a parent with no
// children answer "none" instead of going back to the database to ask.
func (r *HasMany) InitRelation(models []Model, relation string) []Model {
	for _, model := range models {
		model.SetRelation(relation, []Model{})
	}
	return models
}

// Match assigns each model in models every result in results that belongs to
// it, via MatchMany, and stores the slice under relation.
func (r *HasMany) Match(models []Model, results []Model, relation string) ([]Model, error) {
	return r.MatchMany(models, results, relation)
}

// NewHasManyUnconstrained builds the relation without narrowing it to one parent.
//
// It exists because the constraint used to be switched off through a
// process-wide flag, which meant a relation built on another goroutine while
// the flag was down came back unconstrained -- every parent's children, in a
// well-formed query nobody could tell apart from the right one. The call site
// says which it wants now, and there is no flag to leave down.
func NewHasManyUnconstrained(query Builder, parent Model, foreignKey, localKey string) *HasMany {
	return newHasMany(query, parent, foreignKey, localKey)
}
