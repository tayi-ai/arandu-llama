package model

import (
	"context"

	"github.com/arandu-io/hesape/auth"
)

// The four methods a relation asks of a model and this one did not have.
//
// They are here because the relations tree declares the contract it consumes
// -- a narrow interface, in the package that consumes it -- and a model that
// answers all of it but four methods answers none of it. Each is small, and
// each answers a question the model already had the state for.

// UnsetAttribute removes a raw attribute.
//
// It reaches only the attributes a column has no field behind: a struct field
// cannot be removed, and setting it to its zero value would be a different
// thing said with the same word. A pivot row, whose columns are not known until
// run time, is what needs this.
func (m *Model[T]) UnsetAttribute(key string) {
	delete(m.attributes, key)
}

// IsRelation reports whether key names a relation this model declares.
//
// It reads the resolvers rather than the loaded values, so it answers true for
// a relation that has not been loaded yet -- which is the question being asked:
// "is this a relation" and not "is this relation here".
func (m *Model[T]) IsRelation(key string) bool {
	_, declared := m.RelationResolvers[key]
	return declared
}

// Touches reports whether saving this model stamps the owner of relation.
//
// The list is the model's own, set by the application. An empty list means the
// model touches nothing, which is the default: a save that silently stamped a
// parent would be a write the caller did not ask for.
func (m *Model[T]) Touches(relation string) bool {
	for _, touched := range m.touches {
		if touched == relation {
			return true
		}
	}
	return false
}

// GetTouchedRelations returns the relations whose owner is stamped on save.
func (m *Model[T]) GetTouchedRelations() []string { return m.touches }

// SetTouchedRelations replaces them, and returns the model so it can be set at
// construction with everything else.
func (m *Model[T]) SetTouchedRelations(relations []string) *Model[T] {
	m.touches = relations
	return m
}

// Touch stamps the model's updated-at column and saves it.
//
// A model that does not use timestamps, or has no updated-at column, is not an
// error: it is a model there is nothing to stamp on, and the call is a no-op
// rather than a failure. The Builder's Touch is the same idea over a set of
// rows.
func (m *Model[T]) Touch(ctx context.Context, g auth.Grant) error {
	if !m.UsesTimestamps() || m.GetUpdatedAtColumn() == "" {
		return nil
	}
	if err := m.SetAttribute(m.GetUpdatedAtColumn(), m.FreshTimestamp()); err != nil {
		return err
	}
	_, err := m.Save(ctx, g)
	return err
}
