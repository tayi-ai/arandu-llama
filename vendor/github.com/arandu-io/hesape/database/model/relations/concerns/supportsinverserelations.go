package concerns

import (
	"fmt"
	"strings"

	"github.com/arandu-io/hesape/str"
)

// SupportsInverseRelations answers the trait of the same name: chaperone, which
// links every loaded child back to the parent it came from so that reading
// $post->user after $user->posts does not go back to the database.
//
// The PHP registers an afterQuery callback on the builder to do the linking for
// results that arrive later. There is no afterQuery on the builder contract
// here, so the linking happens where the results are known -- in matchOneOrMany
// and in the create paths, which is where the PHP's callback ends up running
// anyway.
type SupportsInverseRelations struct {
	// InverseRelatedModel answers the trait's $this->getModel(): the related
	// model, the one the inverse relation is declared on.
	InverseRelatedModel func() Model

	// InverseParent answers $this->getParent().
	InverseParent func() Model

	// PossibleInverseRelations answers the trait's method of the same name. A
	// relation that has a better guess than the default -- MorphOneOrMany
	// prepends the morph type minus its _type suffix -- sets it.
	PossibleInverseRelations func() []string

	inverseRelationship string
}

// Inverse answers SupportsInverseRelations::inverse, the alias of Chaperone.
func (s *SupportsInverseRelations) Inverse(relation ...string) error {
	return s.Chaperone(relation...)
}

// Chaperone answers SupportsInverseRelations::chaperone.
//
// It carries an error where the PHP throws RelationNotFoundException: naming a
// relation the related model does not declare is a typo that would otherwise
// set an attribute nobody reads.
func (s *SupportsInverseRelations) Chaperone(relation ...string) error {
	name := ""
	if len(relation) > 0 {
		name = relation[0]
	}
	if name == "" {
		name = s.guessInverseRelation()
	}

	if name == "" || s.InverseRelatedModel == nil || !s.InverseRelatedModel().IsRelation(name) {
		return fmt.Errorf("relations: call to undefined relationship [%s] on model [%s]", nameOrNull(name), tableOf(s.InverseRelatedModel))
	}

	s.inverseRelationship = name
	return nil
}

// guessInverseRelation answers SupportsInverseRelations::guessInverseRelation.
func (s *SupportsInverseRelations) guessInverseRelation() string {
	if s.PossibleInverseRelations == nil || s.InverseRelatedModel == nil {
		return ""
	}
	for _, relation := range s.PossibleInverseRelations() {
		if relation != "" && s.InverseRelatedModel().IsRelation(relation) {
			return relation
		}
	}
	return ""
}

// DefaultPossibleInverseRelations answers the trait's own
// getPossibleInverseRelations, for a relation that adds nothing to the list.
//
// The PHP's third candidate is Str::camel(class_basename($parent)). There is no
// class name at run time here, so it is the parent's morph alias -- the name
// the model was registered under, which is the same string the column holds.
func DefaultPossibleInverseRelations(foreignKey string, parent Model, related Model) []string {
	if parent == nil {
		return nil
	}

	keyName := parent.GetKeyName()
	candidates := []string{
		str.Camel(strings.TrimSuffix(foreignKey, keyName)),
		str.Camel(strings.TrimSuffix(parent.GetForeignKey(), keyName)),
		str.Camel(parent.GetMorphClass()),
		"owner",
	}
	if related != nil && parent.GetTable() == related.GetTable() {
		candidates = append(candidates, "parent")
	}

	return unique(nonEmpty(candidates))
}

// ApplyInverseRelationToModel answers the trait's method of the same name.
func (s *SupportsInverseRelations) ApplyInverseRelationToModel(model Model, parent ...Model) Model {
	inverse := s.GetInverseRelationship()
	if inverse == "" || model == nil {
		return model
	}

	owner := s.parentOr(parent...)
	if owner == nil {
		return model
	}

	model.SetRelation(inverse, owner)
	return model
}

// ApplyInverseRelationToCollection answers the trait's method of the same name.
func (s *SupportsInverseRelations) ApplyInverseRelationToCollection(models []Model, parent ...Model) []Model {
	for _, model := range models {
		s.ApplyInverseRelationToModel(model, parent...)
	}
	return models
}

// GetInverseRelationship answers the trait's method of the same name.
func (s *SupportsInverseRelations) GetInverseRelationship() string { return s.inverseRelationship }

// WithoutInverse answers SupportsInverseRelations::withoutInverse.
func (s *SupportsInverseRelations) WithoutInverse() { s.WithoutChaperone() }

// WithoutChaperone answers SupportsInverseRelations::withoutChaperone.
func (s *SupportsInverseRelations) WithoutChaperone() { s.inverseRelationship = "" }

func (s *SupportsInverseRelations) parentOr(parent ...Model) Model {
	if len(parent) > 0 && parent[0] != nil {
		return parent[0]
	}
	if s.InverseParent == nil {
		return nil
	}
	return s.InverseParent()
}

func nameOrNull(name string) string {
	if name == "" {
		return "null"
	}
	return name
}

func tableOf(model func() Model) string {
	if model == nil || model() == nil {
		return "unknown"
	}
	return model().GetTable()
}

func nonEmpty(values []string) []string {
	out := make([]string, 0, len(values))
	for _, value := range values {
		if value != "" {
			out = append(out, value)
		}
	}
	return out
}

func unique(values []string) []string {
	seen := make(map[string]struct{}, len(values))
	out := make([]string, 0, len(values))
	for _, value := range values {
		if _, ok := seen[value]; ok {
			continue
		}
		seen[value] = struct{}{}
		out = append(out, value)
	}
	return out
}
