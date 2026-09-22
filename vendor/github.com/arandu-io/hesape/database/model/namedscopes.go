package model

import "fmt"

// NamedScope is one entry of Model.NamedScopes: a filter the caller applies
// by name, through Scopes or CallNamedScope.
//
// It is a func and not a method because Go cannot look a method up by name,
// which is the same reason RelationResolvers is a map.
type NamedScope[T any] func(builder *Builder[T], parameters ...any) *Builder[T]

// HasNamedScope reports whether the model has a scope registered under this
// name.
//
// A scope is registered under its bare name -- Active, not scopeActive --
// and this is a lookup in that map.
func (m *Model[T]) HasNamedScope(scope string) bool {
	_, ok := m.NamedScopes[scope]
	return ok
}

// CallNamedScope calls the named scope with b and parameters, and returns
// what it returns. It fails the builder when scope is not registered.
//
// The builder is its own argument rather than the first entry of
// parameters: an []any that must have a *Builder[T] in slot zero is a
// signature that documents nothing and fails at run time.
func (m *Model[T]) CallNamedScope(scope string, b *Builder[T], parameters ...any) *Builder[T] {
	apply, ok := m.NamedScopes[scope]
	if !ok {
		return b.fail(fmt.Errorf("%w: %s on %s", ErrNamedScopeNotFound, scope, m.GetTable()))
	}
	return apply(b, parameters...)
}

// HasNamedScope reports whether the builder's model has a scope registered
// under this name.
func (b *Builder[T]) HasNamedScope(scope string) bool {
	return b.model != nil && b.model.HasNamedScope(scope)
}

// Scopes applies the named scopes in order, each one's wheres grouped so
// that an or inside a scope cannot escape it.
//
// This takes only names; CallNamedScope takes the parameters -- a scope
// that needs arguments is called directly rather than listed here.
func (b *Builder[T]) Scopes(scopes ...string) *Builder[T] {
	out := b
	for _, scope := range scopes {
		before := len(out.query.Wheres)
		out = out.model.CallNamedScope(scope, out)
		groupNewWheres(out.query, before)
	}
	return out
}
