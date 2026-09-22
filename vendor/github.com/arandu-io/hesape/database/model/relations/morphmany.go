package relations

import (
	"context"

	"github.com/arandu-io/hesape/auth"
)

// MorphMany is the comments of a post, on a table that also holds the comments
// of a video.
type MorphMany struct {
	MorphOneOrMany
}

// NewMorphMany answers MorphMany::__construct.
func newMorphMany(query Builder, parent Model, typ, id, localKey string) *MorphMany {
	relation := &MorphMany{MorphOneOrMany: NewMorphOneOrMany(query, parent, typ, id, localKey)}
	relation.ExistenceCompareKey = relation.GetExistenceCompareKey
	relation.SupportsInverseRelations.PossibleInverseRelations = relation.PossibleInverseRelations
	return relation
}

// NewMorphMany builds the relation and narrows it to its parent.
func NewMorphMany(query Builder, parent Model, typ, id, localKey string) *MorphMany {
	relation := newMorphMany(query, parent, typ, id, localKey)
	relation.AddConstraints()
	return relation
}

// One answers MorphMany::one.
func (r *MorphMany) One() *MorphOne {
	one := NewMorphOneUnconstrained(r.Query, r.Parent, r.GetQualifiedMorphType(), r.GetQualifiedForeignKeyName(), r.GetLocalKeyName())
	if inverse := r.GetInverseRelationship(); inverse != "" {
		_ = one.Inverse(inverse)
	}
	return one
}

// GetResults answers MorphMany::getResults.
func (r *MorphMany) GetResults(ctx context.Context, g auth.Grant) (any, error) {
	if r.GetParentKey() == nil {
		return []Model{}, nil
	}
	return r.Get(ctx, g)
}

// InitRelation answers MorphMany::initRelation.
func (r *MorphMany) InitRelation(models []Model, relation string) []Model {
	for _, model := range models {
		model.SetRelation(relation, []Model{})
	}
	return models
}

// Match answers MorphMany::match.
func (r *MorphMany) Match(models []Model, results []Model, relation string) ([]Model, error) {
	return r.MatchMany(models, results, relation)
}

// NewMorphManyUnconstrained builds the relation without narrowing it to one parent.
//
// It exists because the constraint used to be switched off through a
// process-wide flag, which meant a relation built on another goroutine while
// the flag was down came back unconstrained -- every parent's children, in a
// well-formed query nobody could tell apart from the right one. The call site
// says which it wants now, and there is no flag to leave down.
func NewMorphManyUnconstrained(query Builder, parent Model, typ, id, localKey string) *MorphMany {
	return newMorphMany(query, parent, typ, id, localKey)
}
