package model

import "reflect"

// Scope is a filter every query on a model carries until somebody removes it by
// name.
type Scope[T any] interface {
	// Apply adds the scope's filter to builder's query.
	Apply(builder *Builder[T], model *Model[T])
}

// ScopeExtender is the optional half of a Scope: it is called when the scope
// implements it, which is how SoftDeletingScope hangs WithTrashed and
// OnlyTrashed off the builder.
type ScopeExtender[T any] interface {
	// Extend adds whatever methods or state this scope hangs off builder.
	Extend(builder *Builder[T])
}

// AddGlobalScope registers scope under identifier: every query on the model
// carries it until WithoutGlobalScope removes it by that name.
//
// The identifier is given rather than derived, because a closure scope (see
// ScopeFunc) has no name of its own.
func (m *Model[T]) AddGlobalScope(identifier string, scope Scope[T]) *Model[T] {
	if m.globalScopes == nil {
		m.globalScopes = map[string]Scope[T]{}
	}
	m.globalScopes[identifier] = scope
	return m
}

// GetGlobalScopes returns every scope registered on the model.
func (m *Model[T]) GetGlobalScopes() map[string]Scope[T] { return cloneScopes(m.globalScopes) }

// HasGlobalScope reports whether a scope is registered under identifier.
func (m *Model[T]) HasGlobalScope(identifier string) bool {
	_, ok := m.globalScopes[identifier]
	return ok
}

// funcScope is a Scope written as a function, for the closure form of
// AddGlobalScope.
type funcScope[T any] func(*Builder[T])

// Apply calls f with builder.
func (f funcScope[T]) Apply(builder *Builder[T], _ *Model[T]) { f(builder) }

// ScopeFunc wraps a function as a Scope, for the closure form of
// addGlobalScope.
func ScopeFunc[T any](apply func(*Builder[T])) Scope[T] { return funcScope[T](apply) }

func cloneScopes[T any](in map[string]Scope[T]) map[string]Scope[T] {
	if in == nil {
		return nil
	}
	out := make(map[string]Scope[T], len(in))
	for identifier, scope := range in {
		out[identifier] = scope
	}
	return out
}

// scopeIdentifier returns the type name of scope, for removing a scope that
// was registered by value rather than by name.
func scopeIdentifier[T any](scope Scope[T]) string {
	t := reflect.TypeOf(scope)
	if t == nil {
		return ""
	}
	if t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	return t.Name()
}
