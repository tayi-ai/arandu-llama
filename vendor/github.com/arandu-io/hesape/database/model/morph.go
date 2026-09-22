package model

import (
	"context"
	"reflect"

	"github.com/arandu-io/hesape/auth"
)

// GetMorphClass returns the name a polymorphic column writes down for a row
// of this model: the unaliased type name of T.
//
// A relation can register a short alias for that name instead, through the
// morph map in model/relations, which imports this package and applies
// the alias there.
func (m *Model[T]) GetMorphClass() string { return reflect.TypeFor[T]().Name() }

// MorphLoadable is the loaded value of a polymorphic relation: a row of
// whatever model the morph column named.
//
// Nothing in Go can turn a stored name back into a type, so the value
// identifies itself instead: *Model[R] satisfies this for every R, and the
// morph map is keyed by GetMorphClass.
type MorphLoadable interface {
	// GetMorphClass returns the name this value is stored under in the
	// morph column.
	GetMorphClass() string
	// Load eager loads these relations onto the value.
	Load(ctx context.Context, g auth.Grant, relations ...string) error
	// LoadAggregate loads function over column of each relation onto the
	// value.
	LoadAggregate(ctx context.Context, g auth.Grant, relations []string, column, function string) error
}

// LoadMorph eager loads, on the row a polymorphic relation points at, the
// relations named for that row's own type.
//
// relation is the morph relation on this model; relations maps a morph
// class to what to load on it. A morph class with no entry loads nothing.
func (m *Model[T]) LoadMorph(ctx context.Context, g auth.Grant, relation string, relations map[string][]string) error {
	loadable, ok := m.morphTarget(relation)
	if !ok {
		return nil
	}
	return loadable.Load(ctx, g, relations[loadable.GetMorphClass()]...)
}

// LoadMorphAggregate loads function over column of each relation named for
// the target row's type, on the row a polymorphic relation points at.
func (m *Model[T]) LoadMorphAggregate(ctx context.Context, g auth.Grant, relation string, relations map[string][]string, column, function string) error {
	loadable, ok := m.morphTarget(relation)
	if !ok {
		return nil
	}
	return loadable.LoadAggregate(ctx, g, relations[loadable.GetMorphClass()], column, function)
}

// LoadMorphCount loads the count of each relation named for the target
// row's type, on the row a polymorphic relation points at.
func (m *Model[T]) LoadMorphCount(ctx context.Context, g auth.Grant, relation string, relations map[string][]string) error {
	return m.LoadMorphAggregate(ctx, g, relation, relations, "*", "count")
}

// LoadMorphMax loads the max of column over each relation named for the
// target row's type, on the row a polymorphic relation points at.
func (m *Model[T]) LoadMorphMax(ctx context.Context, g auth.Grant, relation string, relations map[string][]string, column string) error {
	return m.LoadMorphAggregate(ctx, g, relation, relations, column, "max")
}

// LoadMorphMin loads the min of column over each relation named for the
// target row's type, on the row a polymorphic relation points at.
func (m *Model[T]) LoadMorphMin(ctx context.Context, g auth.Grant, relation string, relations map[string][]string, column string) error {
	return m.LoadMorphAggregate(ctx, g, relation, relations, column, "min")
}

// LoadMorphSum loads the sum of column over each relation named for the
// target row's type, on the row a polymorphic relation points at.
func (m *Model[T]) LoadMorphSum(ctx context.Context, g auth.Grant, relation string, relations map[string][]string, column string) error {
	return m.LoadMorphAggregate(ctx, g, relation, relations, column, "sum")
}

// LoadMorphAvg loads the average of column over each relation named for the
// target row's type, on the row a polymorphic relation points at.
func (m *Model[T]) LoadMorphAvg(ctx context.Context, g auth.Grant, relation string, relations map[string][]string, column string) error {
	return m.LoadMorphAggregate(ctx, g, relation, relations, column, "avg")
}

// morphTarget returns the loaded row a morph relation points at, or nothing
// when it points at nothing.
func (m *Model[T]) morphTarget(relation string) (MorphLoadable, bool) {
	value, ok := m.GetRelation(relation)
	if !ok || value == nil {
		return nil, false
	}
	loadable, ok := value.(MorphLoadable)
	return loadable, ok
}

// LoadMorph eager loads relation for every model in the collection, per the
// target row's own type.
//
// A []MorphLoadable cannot become a Collection[R] -- R is only known at run
// time -- so the load runs per row instead: the same rows, more queries. A
// caller on a hot path loads the concrete relation by its type instead,
// where the grouping is the compiler's.
func (c Collection[T]) LoadMorph(ctx context.Context, g auth.Grant, relation string, relations map[string][]string) error {
	for _, model := range c.models() {
		if err := model.LoadMorph(ctx, g, relation, relations); err != nil {
			return err
		}
	}
	return nil
}

// LoadMorphCount loads the count of relation for every model in the
// collection. See LoadMorph for why it is one query per row rather than one
// per class.
func (c Collection[T]) LoadMorphCount(ctx context.Context, g auth.Grant, relation string, relations map[string][]string) error {
	for _, model := range c.models() {
		if err := model.LoadMorphCount(ctx, g, relation, relations); err != nil {
			return err
		}
	}
	return nil
}
