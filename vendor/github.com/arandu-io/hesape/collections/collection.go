package collections

// Collection is an ordered list of values, and the key of an element is its
// index.
//
// It is declared as a slice rather than a struct so that a []T can be handed
// to it and taken back out without a copy, and so that len, range and the
// slices package keep working on it.
type Collection[T any] []T

// Collect wraps items in a Collection. It is the constructor an application
// actually types.
//
// The result shares the backing array with items: mutate one and the other
// sees it, up to the point a method reallocates.
func Collect[T any](items []T) Collection[T] { return Collection[T](items) }

// Make wraps items in a Collection, as Collect does.
func Make[T any](items []T) Collection[T] { return Collection[T](items) }

// Empty returns an empty collection that is not nil, so that appending to it
// and comparing its length behave the same as on a collection that had
// elements.
func Empty[T any]() Collection[T] { return Collection[T]{} }

// Times builds a collection of number elements from the callback.
//
// The callback receives 1, 2, ... number, one-based. A number below one yields
// an empty collection rather than an error, which is what makes Sliding safe
// to write in terms of it.
func Times[T any](number int, callback func(int) T) Collection[T] {
	if number < 1 {
		return Collection[T]{}
	}
	out := make(Collection[T], 0, number)
	for i := 1; i <= number; i++ {
		out = append(out, callback(i))
	}
	return out
}

// Range counts from from to to, inclusive.
//
// A step of zero is treated as one, and a negative step, or a to below from,
// counts downwards.
func Range(from, to, step int) Collection[int] {
	if step == 0 {
		step = 1
	}
	if step < 0 {
		step = -step
	}
	out := Collection[int]{}
	if from <= to {
		for v := from; v <= to; v += step {
			out = append(out, v)
		}
		return out
	}
	for v := from; v >= to; v -= step {
		out = append(out, v)
	}
	return out
}

// Wrap gathers its arguments into a collection: Wrap[int]() is empty, Wrap(1)
// holds one element, and Wrap(s...) holds the slice.
func Wrap[T any](value ...T) Collection[T] {
	if value == nil {
		return Collection[T]{}
	}
	return Collection[T](value)
}

// Unwrap gives back the plain slice underneath the collection.
func Unwrap[T any](value Collection[T]) []T { return []T(value) }

// All returns the elements as a plain slice.
//
// The result is never nil: an empty collection gives an empty slice, so that a
// caller can append to it without checking.
func (c Collection[T]) All() []T {
	if c == nil {
		return []T{}
	}
	return []T(c)
}

// Count reports the number of elements.
func (c Collection[T]) Count() int { return len(c) }

// IsEmpty reports whether the collection holds no elements.
func (c Collection[T]) IsEmpty() bool { return len(c) == 0 }

// IsNotEmpty reports whether the collection holds at least one element.
func (c Collection[T]) IsNotEmpty() bool { return len(c) > 0 }

// Get returns the element at the index.
//
// The second result is false when the index is out of range, so the caller
// supplies the fallback at the call site instead of passing one in.
func (c Collection[T]) Get(key int) (T, bool) {
	if key < 0 || key >= len(c) {
		var zero T
		return zero, false
	}
	return c[key], true
}

// Has reports whether every index given is within the collection. With no
// index it reports false.
func (c Collection[T]) Has(keys ...int) bool {
	if len(keys) == 0 {
		return false
	}
	for _, k := range keys {
		if k < 0 || k >= len(c) {
			return false
		}
	}
	return true
}

// HasAny reports whether at least one index given is within the collection.
func (c Collection[T]) HasAny(keys ...int) bool {
	for _, k := range keys {
		if k >= 0 && k < len(c) {
			return true
		}
	}
	return false
}

// Keys returns the indices of the elements: 0, 1, ... Count()-1.
func (c Collection[T]) Keys() Collection[int] {
	out := make(Collection[int], len(c))
	for i := range c {
		out[i] = i
	}
	return out
}

// Values returns a copy of the elements, detached from the receiver: writing
// to the result does not reach the collection it came from.
func (c Collection[T]) Values() Collection[T] {
	out := make(Collection[T], len(c))
	copy(out, c)
	return out
}

// Collect returns a fresh collection holding the same elements.
func (c Collection[T]) Collect() Collection[T] { return c.Values() }

// Put writes the value at the index and returns the receiver, so that calls
// chain.
//
// Writing past the end grows the collection with zero values up to that
// index. A negative index is ignored.
func (c *Collection[T]) Put(key int, value T) *Collection[T] {
	if key < 0 {
		return c
	}
	for len(*c) <= key {
		var zero T
		*c = append(*c, zero)
	}
	(*c)[key] = value
	return c
}

// Push appends every value and returns the receiver.
func (c *Collection[T]) Push(values ...T) *Collection[T] {
	*c = append(*c, values...)
	return c
}

// Add appends one element and returns the receiver.
func (c *Collection[T]) Add(item T) *Collection[T] {
	*c = append(*c, item)
	return c
}

// Unshift puts the values at the front, in the order given, and returns the
// receiver.
func (c *Collection[T]) Unshift(values ...T) *Collection[T] {
	*c = append(append(make(Collection[T], 0, len(*c)+len(values)), values...), *c...)
	return c
}

// Prepend puts a single value at the front. It is Unshift of one element.
func (c *Collection[T]) Prepend(value T) *Collection[T] { return c.Unshift(value) }

// Pop removes the last count elements and returns them, most recent first.
//
// A count of zero or less, or an empty collection, gives an empty result; a
// count above the length takes everything.
func (c *Collection[T]) Pop(count int) Collection[T] {
	if count <= 0 || len(*c) == 0 {
		return Collection[T]{}
	}
	if count > len(*c) {
		count = len(*c)
	}
	out := make(Collection[T], 0, count)
	for i := 0; i < count; i++ {
		last := len(*c) - 1
		out = append(out, (*c)[last])
		*c = (*c)[:last]
	}
	return out
}

// Shift removes the first count elements and returns them in order. As with
// Pop, a count of zero or less gives an empty result and a count above the
// length takes everything.
func (c *Collection[T]) Shift(count int) Collection[T] {
	if count <= 0 || len(*c) == 0 {
		return Collection[T]{}
	}
	if count > len(*c) {
		count = len(*c)
	}
	out := make(Collection[T], count)
	copy(out, (*c)[:count])
	*c = (*c)[count:]
	return out
}

// Forget removes the elements at the given indices. The survivors close the
// gap and are renumbered.
func (c *Collection[T]) Forget(keys ...int) *Collection[T] {
	drop := make(map[int]struct{}, len(keys))
	for _, k := range keys {
		drop[k] = struct{}{}
	}
	out := make(Collection[T], 0, len(*c))
	for i, v := range *c {
		if _, ok := drop[i]; ok {
			continue
		}
		out = append(out, v)
	}
	*c = out
	return c
}

// Pull reads the element at the index and removes it. The second result is
// false when the index is out of range, and nothing is removed then.
func (c *Collection[T]) Pull(key int) (T, bool) {
	v, ok := c.Get(key)
	if !ok {
		return v, false
	}
	c.Forget(key)
	return v, true
}

// GetOrPut returns the element at the index, or writes the value there and
// returns it when the index is not filled yet.
func (c *Collection[T]) GetOrPut(key int, value T) T {
	if v, ok := c.Get(key); ok {
		return v
	}
	c.Put(key, value)
	return value
}

// Transform replaces every element with the result of the callback, in place.
// Unlike Map it cannot change the element type.
func (c *Collection[T]) Transform(callback func(T) T) *Collection[T] {
	for i, v := range *c {
		(*c)[i] = callback(v)
	}
	return c
}

// Concat appends the source to a copy of the collection and leaves the
// receiver alone, which is where it differs from Push.
func (c Collection[T]) Concat(source []T) Collection[T] {
	out := make(Collection[T], 0, len(c)+len(source))
	out = append(out, c...)
	out = append(out, source...)
	return out
}

// Merge appends items to a copy of the collection. It is Concat.
func (c Collection[T]) Merge(items []T) Collection[T] { return c.Concat(items) }

// Union keeps the receiver's element at every index it has, and appends the
// tail of items past that length.
func (c Collection[T]) Union(items []T) Collection[T] {
	out := make(Collection[T], 0, max(len(c), len(items)))
	out = append(out, c...)
	if len(items) > len(c) {
		out = append(out, items[len(c):]...)
	}
	return out
}

// Reverse returns the elements in the opposite order.
func (c Collection[T]) Reverse() Collection[T] {
	out := make(Collection[T], len(c))
	for i, v := range c {
		out[len(c)-1-i] = v
	}
	return out
}

// CrossJoin returns the cartesian product of the collection with every list
// given, one tuple per combination.
//
// The product is folded from a single empty tuple, so crossing nothing wraps
// each element on its own -- [1,2] becomes [[1],[2]] -- and crossing an empty
// list yields nothing at all.
//
// It is a function and not a method for the reason Chunk is: instantiating
// Collection[Collection[T]] from inside a method of Collection[T] is an
// instantiation cycle, which Go rejects.
func CrossJoin[T any](c Collection[T], lists ...[]T) Collection[Collection[T]] {
	results := Collection[Collection[T]]{Collection[T]{}}
	for _, array := range append([][]T{[]T(c)}, lists...) {
		next := make(Collection[Collection[T]], 0, len(results)*len(array))
		for _, product := range results {
			for _, item := range array {
				combined := make(Collection[T], len(product), len(product)+1)
				copy(combined, product)
				next = append(next, append(combined, item))
			}
		}
		results = next
	}
	return results
}

// Zip pairs the collection position by position with the lists given.
//
// A short input is padded with the zero value of T, and the result is as long
// as the longest input.
//
// It is a function and not a method for the reason CrossJoin is.
func Zip[T any](c Collection[T], items ...[]T) Collection[Collection[T]] {
	longest := len(c)
	for _, list := range items {
		longest = max(longest, len(list))
	}
	out := make(Collection[Collection[T]], longest)
	for i := range out {
		tuple := make(Collection[T], 0, len(items)+1)
		if i < len(c) {
			tuple = append(tuple, c[i])
		} else {
			var zero T
			tuple = append(tuple, zero)
		}
		for _, list := range items {
			if i < len(list) {
				tuple = append(tuple, list[i])
				continue
			}
			var zero T
			tuple = append(tuple, zero)
		}
		out[i] = tuple
	}
	return out
}

// Multiply returns the elements repeated multiplier times, in order.
//
// A multiplier of zero or less gives an empty collection.
func (c Collection[T]) Multiply(multiplier int) Collection[T] {
	if multiplier <= 0 {
		return Collection[T]{}
	}
	out := make(Collection[T], 0, len(c)*multiplier)
	for i := 0; i < multiplier; i++ {
		out = append(out, c...)
	}
	return out
}

// ToBase returns a detached copy of the collection, as Values does.
func (c Collection[T]) ToBase() Collection[T] { return c.Values() }

// Pad grows the collection to size elements, filling with value.
//
// A positive size pads on the right and a negative size on the left. A size
// whose magnitude is not larger than the count returns the elements unchanged.
func (c Collection[T]) Pad(size int, value T) Collection[T] {
	n := size
	left := false
	if n < 0 {
		n = -n
		left = true
	}
	if n <= len(c) {
		return c.Values()
	}
	pad := make(Collection[T], n-len(c))
	for i := range pad {
		pad[i] = value
	}
	if left {
		return append(pad, c...)
	}
	return append(c.Values(), pad...)
}
