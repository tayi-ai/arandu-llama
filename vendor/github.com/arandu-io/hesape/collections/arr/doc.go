// Package arr reads and writes nested maps and slices by "dot" path, and
// carries the list operations that work on untyped data.
//
// A nested structure here is two shapes: a keyed map, usually a
// map[string]any, and a list, a []any or a []T. Get, Set, Add, Has, Forget,
// Pull and Push walk both with a dotted key. DataGet, DataSet, DataFill,
// DataForget and DataHas walk the same paths and also give a meaning to the
// wildcard segments "*", "{first}" and "{last}".
//
// Several names here -- Map, First, Last, Only, Except, Sort, Where, Random,
// Take, Wrap and Pluck -- also exist in the parent package over
// Collection[T]. They are separate operations over separate shapes, and the
// subpackage is what lets both keep the same spelling.
//
// # Ordering
//
// A Go map has no order, so every function whose result is a list -- Divide,
// Query, Undot, ToCssClasses, Flatten over a map -- walks the keys in
// ascending order. The result is then reproducible from the value alone.
//
// # Defaults and errors
//
// Wherever a key may be missing the second result is a bool rather than a
// default passed in, so the fallback is written at the call site. The typed
// readers -- Array, Boolean, Float, Integer and String -- take an optional
// default as a variadic and return ErrInvalidArgument when the value they
// find is of another type.
package arr
