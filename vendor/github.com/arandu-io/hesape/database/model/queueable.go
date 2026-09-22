package model

import (
	"errors"
	"reflect"
	"slices"
)

// ErrMixedQueueableConnections is what GetQueueableConnection returns for a
// queued collection whose models are not all on one connection: it cannot
// be restored, because the job records one connection name.
var ErrMixedQueueableConnections = errors.New("model: queueing collections with multiple model connections is not supported")

// Queueable is what GetQueueableRelations recurses into: a value hanging off a
// loaded relation that can name its own.
//
// It is one method, declared where it is consumed.
type Queueable interface {
	// GetQueueableRelations returns the names of this value's own loaded
	// relations that a queued job restores along with it.
	GetQueueableRelations() []string
}

// GetQueueableID returns what a queued job writes down so it can find this
// row again.
func (m *Model[T]) GetQueueableID() any { return m.GetKey() }

// GetQueueableConnection returns the name of the connection this model
// uses, for a queued job to restore it on.
func (m *Model[T]) GetQueueableConnection() string { return m.GetConnectionName() }

// GetQueueableRelations returns the loaded relations a job restores along
// with the row.
//
// A loaded relation with no registered resolver is skipped, since a
// relation with no resolver cannot be loaded again on the other side of the
// queue.
//
// The order is sorted rather than insertion order: a Go map has none, and a
// job payload that differs between two runs over the same row is a payload
// nobody can diff.
func (m *Model[T]) GetQueueableRelations() []string {
	out := []string{}
	for _, name := range sortedKeys(m.relations) {
		if _, declared := m.RelationResolvers[name]; !declared {
			continue
		}
		out = append(out, name)
		nested, ok := m.relations[name].(Queueable)
		if !ok {
			continue
		}
		for _, child := range nested.GetQueueableRelations() {
			out = append(out, name+"."+child)
		}
	}
	return out
}

// GetQueueableClass returns the type name of the models being queued.
//
// A Collection[T] cannot hold two model types, so there is no mixed-type
// case left to refuse.
//
// It returns the empty string for an empty collection: there is no model to
// take the name from.
func (c Collection[T]) GetQueueableClass() string {
	if c.IsEmpty() {
		return ""
	}
	return reflect.TypeFor[T]().Name()
}

// GetQueueableIDs returns the queueable id of every model.
func (c Collection[T]) GetQueueableIDs() []any {
	if c.IsEmpty() {
		return []any{}
	}
	found := c.models()
	out := make([]any, 0, len(found))
	for _, model := range found {
		out = append(out, model.GetQueueableID())
	}
	return out
}

// GetQueueableRelations returns the relations every model in the collection
// has loaded.
//
// It is the intersection and not the union: a relation loaded on one row
// and not on another cannot be restored for the whole collection.
func (c Collection[T]) GetQueueableRelations() []string {
	if c.IsEmpty() {
		return []string{}
	}
	found := c.models()
	if len(found) == 0 {
		return []string{}
	}
	shared := found[0].GetQueueableRelations()
	for _, model := range found[1:] {
		relations := model.GetQueueableRelations()
		shared = slices.DeleteFunc(shared, func(name string) bool {
			return !slices.Contains(relations, name)
		})
	}
	return shared
}

// GetQueueableConnection returns the connection name shared by every model,
// or ErrMixedQueueableConnections when they disagree. An empty collection
// returns the empty string.
func (c Collection[T]) GetQueueableConnection() (string, error) {
	if c.IsEmpty() {
		return "", nil
	}
	found := c.models()
	if len(found) == 0 {
		return "", nil
	}
	connection := found[0].GetConnectionName()
	for _, model := range found {
		if model.GetConnectionName() != connection {
			return "", ErrMixedQueueableConnections
		}
	}
	return connection, nil
}
