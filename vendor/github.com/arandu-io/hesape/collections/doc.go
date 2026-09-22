// Package collections provides Collection, an ordered list of values
// carrying a large set of transformations, and LazyCollection, the same
// vocabulary over a sequence produced one element at a time.
//
// Collect wraps an existing slice; Make, Empty, Times, Range, Wrap and
// FromJSON build one from scratch. Collection.Lazy, NewLazyCollection and
// RangeLazy are the lazy counterparts, and LazyCollection.Collect goes back.
//
// # Keys
//
// A Collection[T] is a list, so the key of an element is its position, from
// zero to Count()-1. An operation that is only meaningful when the keys are
// keys rather than positions takes a map[K]V instead, as a package function.
// The arr subpackage reads and writes nested maps and slices by "dot" path.
//
// # Methods that cannot be methods
//
// A Go method cannot declare a type parameter, so every operation whose
// result changes the element type -- Map, Pluck, GroupBy, Reduce, Sum and the
// rest -- is a package function taking the collection as its first argument.
// That is the only reason any of them is not a method.
//
// # Callbacks
//
// A callback handed both the element and its position takes
// (value T, key int); one that needs only the element takes (value T). A
// callback that can stop the walk early says so by returning false.
//
// # Defaults and errors
//
// A read that may find nothing returns a second bool result rather than
// taking a default, so the fallback is written at the call site. An operation
// that can be asked for something impossible returns an error:
// ErrInvalidArgument, ErrUnexpectedValue, ErrItemNotFound, or a
// MultipleItemsFoundError carrying the count.
package collections
