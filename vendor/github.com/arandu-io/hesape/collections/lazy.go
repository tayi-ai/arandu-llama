package collections

import (
	"iter"
	"time"
)

// LazyCollection holds an iter.Seq2 rather than a slice, so the elements are
// produced one at a time and a source too large for memory never lands in
// memory.
//
// The sequence is walked afresh on every operation, and nothing between two
// walks is retained; Remember and Eager are the ways to change that.
//
// The keys are positions, as they are for Collection, and an operation that
// drops elements renumbers the survivors from zero. Range over one with
// GetIterator.
//
// The zero value yields nothing, so a LazyCollection is usable before it is
// assigned.
type LazyCollection[T any] struct {
	source iter.Seq2[int, T]
}

// NewLazyCollection builds a LazyCollection over the sequence.
//
// The sequence must be re-walkable, since every operation calls it again. One
// that closes over a single-use resource is not, and Remember is the fix
// there.
func NewLazyCollection[T any](source iter.Seq2[int, T]) LazyCollection[T] {
	return LazyCollection[T]{source: source}
}

// Lazy returns a LazyCollection over the elements of this one.
func (c Collection[T]) Lazy() LazyCollection[T] {
	return LazyCollection[T]{source: func(yield func(int, T) bool) {
		for i, v := range c {
			if !yield(i, v) {
				return
			}
		}
	}}
}

// RangeLazy counts from from to to, producing the integers one at a time and
// never all at once.
//
// As in Range, a step of zero counts by one and a negative step, or a to below
// from, counts downwards: a constructor that cannot fail is worth more than
// the error.
func RangeLazy(from, to, step int) LazyCollection[int] {
	if step == 0 {
		step = 1
	}
	if step < 0 {
		step = -step
	}
	return LazyCollection[int]{source: func(yield func(int, int) bool) {
		i := 0
		if from <= to {
			for v := from; v <= to; v += step {
				if !yield(i, v) {
					return
				}
				i++
			}
			return
		}
		for v := from; v >= to; v -= step {
			if !yield(i, v) {
				return
			}
			i++
		}
	}}
}

// GetIterator returns the underlying sequence, so that range works over it. It
// is the one method every other one here is built on.
//
// The zero value yields an empty sequence rather than nil.
func (l LazyCollection[T]) GetIterator() iter.Seq2[int, T] {
	if l.source == nil {
		return func(func(int, T) bool) {}
	}
	return l.source
}

// All walks the whole sequence and hands back the elements as a slice.
//
// This is the method that gives up on being lazy, and its result is never nil.
func (l LazyCollection[T]) All() []T {
	out := []T{}
	for _, v := range l.GetIterator() {
		out = append(out, v)
	}
	return out
}

// Collect walks the whole sequence and returns the elements as an eager
// Collection.
func (l LazyCollection[T]) Collect() Collection[T] { return Collection[T](l.All()) }

// Eager returns a lazy collection backed by a slice that has already been
// read.
//
// The source is walked once, here, and never again -- which is the point when
// the source is a query or a file and the collection is about to be walked
// several times.
func (l LazyCollection[T]) Eager() LazyCollection[T] {
	return Collection[T](l.All()).Lazy()
}

// Remember caches the elements as they are enumerated.
//
// Unlike Eager it reads nothing up front. The first walk pulls from the source
// and fills a cache as it goes; a second walk serves what the first reached
// from the cache and pulls only past that point. A source that can be consumed
// once survives being walked twice, as long as the second walk does not run
// ahead of the first.
//
// The result is not safe for concurrent use: the cache and the puller are
// shared state behind the closure.
func (l LazyCollection[T]) Remember() LazyCollection[T] {
	type entry struct {
		key   int
		value T
	}
	var (
		cache    []entry
		next     func() (int, T, bool)
		stop     func()
		finished bool
	)
	source := l.GetIterator()
	return LazyCollection[T]{source: func(yield func(int, T) bool) {
		for index := 0; ; index++ {
			if index < len(cache) {
				if !yield(cache[index].key, cache[index].value) {
					return
				}
				continue
			}
			if finished {
				return
			}
			if next == nil {
				next, stop = iter.Pull2(source)
			}
			key, value, ok := next()
			if !ok {
				finished = true
				stop()
				return
			}
			cache = append(cache, entry{key: key, value: value})
			if !yield(key, value) {
				return
			}
		}
	}}
}

// TapEach hands each element to the callback as it goes past, and passes it on
// unchanged.
//
// Nothing runs until the result is walked, which is the whole difference from
// Each.
func (l LazyCollection[T]) TapEach(callback func(value T, key int)) LazyCollection[T] {
	source := l.GetIterator()
	return LazyCollection[T]{source: func(yield func(int, T) bool) {
		for k, v := range source {
			callback(v, k)
			if !yield(k, v) {
				return
			}
		}
	}}
}

// Each walks the elements now, stopping when the callback returns false, and
// returns the collection.
func (l LazyCollection[T]) Each(callback func(value T, key int) bool) LazyCollection[T] {
	for k, v := range l.GetIterator() {
		if !callback(v, k) {
			break
		}
	}
	return l
}

// Filter keeps the elements the callback passes.
//
// As on Collection, a nil callback keeps everything. The survivors are
// renumbered from zero.
func (l LazyCollection[T]) Filter(callback func(value T, key int) bool) LazyCollection[T] {
	source := l.GetIterator()
	return LazyCollection[T]{source: func(yield func(int, T) bool) {
		i := 0
		for k, v := range source {
			if callback != nil && !callback(v, k) {
				continue
			}
			if !yield(i, v) {
				return
			}
			i++
		}
	}}
}

// Reject is Filter with the predicate negated. A nil callback keeps
// everything.
func (l LazyCollection[T]) Reject(callback func(value T, key int) bool) LazyCollection[T] {
	if callback == nil {
		return l.Filter(nil)
	}
	return l.Filter(func(value T, key int) bool { return !callback(value, key) })
}

// Take returns the first limit elements.
//
// A positive limit stops the source as soon as it has enough, which is what
// makes it safe on an endless sequence. A limit of zero yields nothing.
//
// A negative limit takes that many from the end, and for that a ring buffer of
// that size is filled while the whole source goes past -- so it is neither
// lazy nor safe on an endless sequence. When the source is shorter than the
// window every element is yielded in order, as Collection.Take does with the
// same argument.
func (l LazyCollection[T]) Take(limit int) LazyCollection[T] {
	source := l.GetIterator()
	if limit >= 0 {
		return LazyCollection[T]{source: func(yield func(int, T) bool) {
			if limit == 0 {
				return
			}
			taken := 0
			for k, v := range source {
				if !yield(k, v) {
					return
				}
				if taken++; taken == limit {
					return
				}
			}
		}}
	}
	size := -limit
	return LazyCollection[T]{source: func(yield func(int, T) bool) {
		ring := make([]T, 0, size)
		position, seen := 0, 0
		for _, v := range source {
			if len(ring) < size {
				ring = append(ring, v)
			} else {
				ring[position] = v
			}
			position = (position + 1) % size
			seen++
		}
		start := 0
		if seen > size {
			start = position
		}
		for i := 0; i < len(ring); i++ {
			if !yield(i, ring[(start+i)%len(ring)]) {
				return
			}
		}
	}}
}

// TakeWhile returns the leading run of elements passing the test.
func (l LazyCollection[T]) TakeWhile(callback func(value T, key int) bool) LazyCollection[T] {
	return l.TakeUntil(func(value T, key int) bool { return !callback(value, key) })
}

// TakeUntil returns the elements before the first one passing the test.
func (l LazyCollection[T]) TakeUntil(callback func(value T, key int) bool) LazyCollection[T] {
	source := l.GetIterator()
	return LazyCollection[T]{source: func(yield func(int, T) bool) {
		for k, v := range source {
			if callback(v, k) {
				return
			}
			if !yield(k, v) {
				return
			}
		}
	}}
}

// TakeUntilTimeout yields elements until the deadline passes, then nothing.
//
// The deadline is checked after each element is handed on, so one element is
// always produced when the deadline is still ahead at the start. When the
// deadline has already passed nothing is produced at all.
//
// The callback is optional and reports the element the deadline was noticed
// after. When there was no such element it is called with the zero value and
// the key -1.
func (l LazyCollection[T]) TakeUntilTimeout(timeout time.Time, callback func(value T, key int)) LazyCollection[T] {
	source := l.GetIterator()
	return LazyCollection[T]{source: func(yield func(int, T) bool) {
		if !time.Now().Before(timeout) {
			if callback != nil {
				var zero T
				callback(zero, -1)
			}
			return
		}
		for k, v := range source {
			if !yield(k, v) {
				return
			}
			if !time.Now().Before(timeout) {
				if callback != nil {
					callback(v, k)
				}
				return
			}
		}
	}}
}

// Throttle releases at most one element per interval.
//
// The interval is measured from the moment the element was fetched, and
// whatever is left of it is slept off after the consumer is done, so a slow
// consumer is never slowed further.
func (l LazyCollection[T]) Throttle(interval time.Duration) LazyCollection[T] {
	source := l.GetIterator()
	return LazyCollection[T]{source: func(yield func(int, T) bool) {
		for k, v := range source {
			fetchedAt := time.Now()
			if !yield(k, v) {
				return
			}
			if rest := interval - time.Since(fetchedAt); rest > 0 {
				time.Sleep(rest)
			}
		}
	}}
}

// WithHeartbeat runs the callback every time the interval has passed while the
// elements go by.
//
// The clock is read per element, so a sequence that stalls calls the callback
// no more often than the elements arrive. This is not a timer.
func (l LazyCollection[T]) WithHeartbeat(interval time.Duration, callback func()) LazyCollection[T] {
	source := l.GetIterator()
	return LazyCollection[T]{source: func(yield func(int, T) bool) {
		start := time.Now()
		for k, v := range source {
			if now := time.Now(); now.Sub(start) >= interval {
				callback()
				start = now
			}
			if !yield(k, v) {
				return
			}
		}
	}}
}

// Skip yields everything past the first count elements, renumbered from zero.
func (l LazyCollection[T]) Skip(count int) LazyCollection[T] {
	source := l.GetIterator()
	return LazyCollection[T]{source: func(yield func(int, T) bool) {
		seen, i := 0, 0
		for _, v := range source {
			if seen < count {
				seen++
				continue
			}
			if !yield(i, v) {
				return
			}
			i++
		}
	}}
}

// First returns the first element passing the test, a nil callback returning
// the first element.
//
// It stops the source at the first match, so it is safe on an endless
// sequence. The second result is false when nothing matched, as on Collection.
func (l LazyCollection[T]) First(callback func(value T, key int) bool) (T, bool) {
	for k, v := range l.GetIterator() {
		if callback == nil || callback(v, k) {
			return v, true
		}
	}
	var zero T
	return zero, false
}

// Count walks the whole sequence and counts it.
func (l LazyCollection[T]) Count() int {
	n := 0
	for range l.GetIterator() {
		n++
	}
	return n
}

// IsEmpty reports whether the sequence yields no element. It reads one element
// and no more.
func (l LazyCollection[T]) IsEmpty() bool {
	_, ok := l.First(nil)
	return !ok
}

// IsNotEmpty reports whether the sequence yields at least one element.
func (l LazyCollection[T]) IsNotEmpty() bool { return !l.IsEmpty() }

// ChunkLazy yields the elements in runs of size, each run gathered into a
// Collection.
//
// A size below one yields nothing rather than looping forever, and the last
// run is short. Only the run being built is held, so this stays lazy.
//
// Go cannot overload and Chunk is already taken by the eager form, so the
// receiver's type is the suffix. It is a function and not a method because
// instantiating LazyCollection[Collection[T]] from a method of
// LazyCollection[T] is an instantiation cycle, which Go rejects.
func ChunkLazy[T any](l LazyCollection[T], size int) LazyCollection[Collection[T]] {
	source := l.GetIterator()
	return LazyCollection[Collection[T]]{source: func(yield func(int, Collection[T]) bool) {
		if size < 1 {
			return
		}
		chunk := make(Collection[T], 0, size)
		i := 0
		for _, v := range source {
			chunk = append(chunk, v)
			if len(chunk) < size {
				continue
			}
			if !yield(i, chunk) {
				return
			}
			i++
			chunk = make(Collection[T], 0, size)
		}
		if len(chunk) > 0 {
			yield(i, chunk)
		}
	}}
}

// MapLazy runs every element through the callback, one at a time.
//
// Go cannot overload and Map is already taken by the eager form, so the
// receiver's type is the suffix. It is a function and not a method because the
// callback changes the element type.
func MapLazy[T, U any](l LazyCollection[T], callback func(value T, key int) U) LazyCollection[U] {
	source := l.GetIterator()
	return LazyCollection[U]{source: func(yield func(int, U) bool) {
		for k, v := range source {
			if !yield(k, callback(v, k)) {
				return
			}
		}
	}}
}
