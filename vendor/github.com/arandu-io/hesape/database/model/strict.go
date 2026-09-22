package model

import (
	"errors"
	"fmt"
	"sync"
)

// ErrLazyLoadingViolation is what reading a relation that was never loaded means
// once
// PreventLazyLoading is on. See PreventLazyLoading for what "lazy loading" can
// still mean in a framework that has none.
var ErrLazyLoadingViolation = errors.New("model: attempted to lazy load a relation that is not loaded")

// ErrMassAssignment is what Fill returns once PreventSilentlyDiscardingAttributes
// is on and a key has no column behind it.
var ErrMassAssignment = errors.New("model: attribute does not exist on the model")

// ErrMissingAttribute names reading an attribute a persisted model does not
// have -- a key that is neither a column nor a raw attribute -- once
// PreventAccessingMissingAttributes is on.
//
// Nothing in this package returns it. GetAttribute returns one value and
// has no error to return, so the violation is reported through the callback
// HandleMissingAttributeViolationUsing registered, and this is the error
// that callback has to report or wrap. Without a callback the read is
// simply nil, which is why the switch alone changes nothing.
var ErrMissingAttribute = errors.New("model: attribute does not exist on the model")

// strict holds the three violation switches and their callbacks.
//
// The switches are package functions, and the state they guard is a package
// variable behind a mutex -- the same shape concerns uses for Unguard and
// WithoutEvents. It is process-wide: every model of every type reads the
// one flag.
var strict struct {
	mu sync.RWMutex

	preventLazyLoading                  bool
	preventSilentlyDiscardingAttributes bool
	preventAccessingMissingAttributes   bool

	lazyLoadingViolation        func(model any, key string)
	discardedAttributeViolation func(model any, keys []string)
	missingAttributeViolation   func(model any, key string)
}

// ShouldBeStrict turns the three switches below on or off together.
//
// value is variadic because Go has no default argument; an empty call turns
// them all on.
func ShouldBeStrict(shouldBeStrict ...bool) {
	PreventLazyLoading(shouldBeStrict...)
	PreventSilentlyDiscardingAttributes(shouldBeStrict...)
	PreventAccessingMissingAttributes(shouldBeStrict...)
}

// PreventLazyLoading turns reading an unloaded relation into a reported
// violation.
//
// # What it does not do
//
// It does not stop a query, because there is no query to stop: nothing here
// loads a relation behind the caller, since such a query would carry no
// auth.Grant.
//
// What is left is reading a relation that is not there. With the switch
// off, GetRelation returns false and a caller that ignored the second value
// reads a nil it will blame on the database. With it on, the read is
// reported through the callback HandleLazyLoadingViolationUsing registered.
//
// A model that does not exist yet, and one that was just created, are exempt --
// they have nothing to have loaded.
func PreventLazyLoading(value ...bool) {
	strict.mu.Lock()
	defer strict.mu.Unlock()
	strict.preventLazyLoading = optionalBool(value)
}

// PreventsLazyLoading reports whether PreventLazyLoading is on.
func PreventsLazyLoading() bool {
	strict.mu.RLock()
	defer strict.mu.RUnlock()
	return strict.preventLazyLoading
}

// HandleLazyLoadingViolationUsing registers what to do about a lazy loading
// violation.
//
// GetRelation cannot return an error without becoming a second way to read a
// relation, so the callback is where the violation goes. An application that
// wants it to be loud registers one:
//
//	model.HandleLazyLoadingViolationUsing(func(model any, key string) {
//		panic(fmt.Sprintf("%v: %s was not loaded", model, key))
//	})
//
// A nil callback unregisters it.
func HandleLazyLoadingViolationUsing(callback func(model any, key string)) {
	strict.mu.Lock()
	defer strict.mu.Unlock()
	strict.lazyLoadingViolation = callback
}

// PreventSilentlyDiscardingAttributes changes what Fill does with a key
// that has no column behind it: dropped when this is off, refused with
// ErrMassAssignment when it is on.
func PreventSilentlyDiscardingAttributes(value ...bool) {
	strict.mu.Lock()
	defer strict.mu.Unlock()
	strict.preventSilentlyDiscardingAttributes = optionalBool(value)
}

// PreventsSilentlyDiscardingAttributes reports whether
// PreventSilentlyDiscardingAttributes is on.
func PreventsSilentlyDiscardingAttributes() bool {
	strict.mu.RLock()
	defer strict.mu.RUnlock()
	return strict.preventSilentlyDiscardingAttributes
}

// HandleDiscardedAttributeViolationUsing registers what to do about a
// discarded-attribute violation.
//
// A registered callback replaces the error: Fill reports the keys to it and
// then succeeds.
func HandleDiscardedAttributeViolationUsing(callback func(model any, keys []string)) {
	strict.mu.Lock()
	defer strict.mu.Unlock()
	strict.discardedAttributeViolation = callback
}

// PreventAccessingMissingAttributes turns reading an absent attribute into a
// reported violation.
//
// GetAttribute returns nil for a key that is neither a column nor a raw
// attribute. With this on, the read is reported through the callback
// HandleMissingAttributeViolationUsing registered.
//
// Most of what this would catch the compiler catches first: a row is a struct,
// and found.Entity.Naem does not build. What is left is the read by name --
// GetAttribute takes a string -- which is where a typo still survives to run
// time.
func PreventAccessingMissingAttributes(value ...bool) {
	strict.mu.Lock()
	defer strict.mu.Unlock()
	strict.preventAccessingMissingAttributes = optionalBool(value)
}

// PreventsAccessingMissingAttributes reports whether
// PreventAccessingMissingAttributes is on.
func PreventsAccessingMissingAttributes() bool {
	strict.mu.RLock()
	defer strict.mu.RUnlock()
	return strict.preventAccessingMissingAttributes
}

// HandleMissingAttributeViolationUsing registers what to do about a
// missing-attribute violation.
//
// GetAttribute returns one value and has no error to return, so the
// callback is the whole of the switch; see HandleLazyLoadingViolationUsing.
func HandleMissingAttributeViolationUsing(callback func(model any, key string)) {
	strict.mu.Lock()
	defer strict.mu.Unlock()
	strict.missingAttributeViolation = callback
}

// handleLazyLoadingViolation reports a lazy-loading violation to the
// registered callback, when PreventLazyLoading is on. With no callback
// registered, nothing happens.
//
// Unlike handleMissingAttributeViolation, it has no exempt case for a model
// that does not exist yet or was just created: there is no fallback
// behavior here left to exempt anything from, only the callback.
func (m *Model[T]) handleLazyLoadingViolation(key string) {
	if !PreventsLazyLoading() {
		return
	}
	strict.mu.RLock()
	callback := strict.lazyLoadingViolation
	strict.mu.RUnlock()

	if callback != nil {
		callback(m, key)
	}
}

// handleMissingAttributeViolation reports a missing-attribute violation to
// the registered callback, when PreventAccessingMissingAttributes is on and
// the model exists.
func (m *Model[T]) handleMissingAttributeViolation(key string) {
	if !PreventsAccessingMissingAttributes() || !m.Exists {
		return
	}
	strict.mu.RLock()
	callback := strict.missingAttributeViolation
	strict.mu.RUnlock()

	if callback != nil {
		callback(m, key)
	}
}

// handleDiscardedAttributeViolation returns the error Fill reports for the
// discarded keys, or nil when a callback took the violation instead.
func (m *Model[T]) handleDiscardedAttributeViolation(keys []string) error {
	if len(keys) == 0 || !PreventsSilentlyDiscardingAttributes() {
		return nil
	}
	strict.mu.RLock()
	callback := strict.discardedAttributeViolation
	strict.mu.RUnlock()

	if callback != nil {
		callback(m, keys)
		return nil
	}
	return fmt.Errorf("%w: %s on %s", ErrMassAssignment, keys, m.GetTable())
}

// optionalBool reads a variadic bool as an optional argument defaulting to
// true: no argument means true, and an argument is used as given.
func optionalBool(value []bool) bool {
	if len(value) == 0 {
		return true
	}
	return value[0]
}
