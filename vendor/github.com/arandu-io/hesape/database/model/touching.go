package model

import (
	"reflect"
	"slices"
	"sync"
)

// ignoreOnTouch holds the model types for which touch propagation is
// currently suspended.
//
// It is a package variable behind a mutex, the same shape strict.go uses for
// its switches: the state is process-wide rather than per-model, so any
// goroutine calling WithoutTouchingOn or IsIgnoringTouch on any model type
// sees the same list. The entries are types rather than names, because a type
// is what identifies a model in Go.
var ignoreOnTouch struct {
	mu    sync.RWMutex
	types []reflect.Type
}

// WithoutTouching suspends touch propagation for T for the length of
// callback: a relation whose owner would have had its updated_at bumped is
// left alone until callback returns.
//
// It is a package-level generic function rather than a method, because no
// model instance is needed -- only the type identifies which relations to
// suspend.
func WithoutTouching[T any](callback func() error) error {
	return WithoutTouchingOn([]reflect.Type{reflect.TypeFor[T]()}, callback)
}

// WithoutTouchingOn suspends touch propagation for every type in models for
// the length of callback, restoring the previous list via defer once callback
// returns -- even when it returns an error.
//
// It takes reflect.Type values rather than name strings, because Go cannot
// reach a type from a name in a string.
func WithoutTouchingOn(models []reflect.Type, callback func() error) error {
	ignoreOnTouch.mu.Lock()
	ignoreOnTouch.types = append(ignoreOnTouch.types, models...)
	ignoreOnTouch.mu.Unlock()

	defer func() {
		ignoreOnTouch.mu.Lock()
		defer ignoreOnTouch.mu.Unlock()
		ignoreOnTouch.types = slices.DeleteFunc(ignoreOnTouch.types, func(t reflect.Type) bool {
			return slices.Contains(models, t)
		})
	}()

	return callback()
}

// IsIgnoringTouch reports whether touch propagation is currently suspended
// for T.
//
// A model with no updated_at column, or with timestamps switched off,
// reports true without consulting the suspended list, because neither has
// anything a touch could update. Otherwise a type counts as suspended only
// when it is itself in the list -- there is no supertype or subtype
// relationship to walk.
func (m *Model[T]) IsIgnoringTouch() bool {
	if m.GetUpdatedAtColumn() == "" || !m.UsesTimestamps() {
		return true
	}
	ignoreOnTouch.mu.RLock()
	defer ignoreOnTouch.mu.RUnlock()
	return slices.Contains(ignoreOnTouch.types, reflect.TypeFor[T]())
}
