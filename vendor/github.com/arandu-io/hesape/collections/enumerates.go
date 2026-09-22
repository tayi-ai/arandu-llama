package collections

import (
	"cmp"
	"crypto/rand"
	"fmt"
	"math"
	"math/big"
	"os"
	"sort"
	"strings"
)

// stableSort sorts s in place with compare, keeping equal elements in the
// order they had. Every sort in this package is stable, and callers rely on
// it: a sort that reordered ties would make repeated passes non-reproducible.
func stableSort[T any](s []T, compare func(a, b T) int) {
	sort.SliceStable(s, func(i, j int) bool { return compare(s[i], s[j]) < 0 })
}

// Filter keeps the elements the callback passes.
//
// A nil callback keeps everything, and the result is empty rather than nil
// when nothing passes.
func (c Collection[T]) Filter(callback func(value T, key int) bool) Collection[T] {
	if callback == nil {
		return c.Values()
	}
	out := make(Collection[T], 0, len(c))
	for i, v := range c {
		if callback(v, i) {
			out = append(out, v)
		}
	}
	return out
}

// Reject is Filter with the predicate negated: it drops the elements the
// callback passes. A nil callback keeps everything.
func (c Collection[T]) Reject(callback func(value T, key int) bool) Collection[T] {
	if callback == nil {
		return c.Values()
	}
	return c.Filter(func(value T, key int) bool { return !callback(value, key) })
}

// First returns the first element passing the test.
//
// A nil callback returns the first element. The second result is false when
// the collection is empty or nothing matches, and the first is then the zero
// value, so the caller writes the fallback at the call site.
func (c Collection[T]) First(callback func(value T, key int) bool) (T, bool) {
	for i, v := range c {
		if callback == nil || callback(v, i) {
			return v, true
		}
	}
	var zero T
	return zero, false
}

// FirstOrFail returns the first element passing the test, or ErrItemNotFound
// when nothing matches.
func (c Collection[T]) FirstOrFail(callback func(value T, key int) bool) (T, error) {
	if value, ok := c.First(callback); ok {
		return value, nil
	}
	var zero T
	return zero, ErrItemNotFound
}

// Last returns the last element passing the test.
//
// A nil callback returns the last element. The second result is false when
// nothing matches, as in First.
func (c Collection[T]) Last(callback func(value T, key int) bool) (T, bool) {
	for i := len(c) - 1; i >= 0; i-- {
		if callback == nil || callback(c[i], i) {
			return c[i], true
		}
	}
	var zero T
	return zero, false
}

// Sole returns the one element passing the test.
//
// It returns ErrItemNotFound when nothing matches, and a
// MultipleItemsFoundError carrying the count when more than one does.
func (c Collection[T]) Sole(callback func(value T, key int) bool) (T, error) {
	matched := c.Filter(callback)
	var zero T
	switch len(matched) {
	case 0:
		return zero, ErrItemNotFound
	case 1:
		return matched[0], nil
	default:
		return zero, &MultipleItemsFoundError{Count: len(matched)}
	}
}

// HasSole reports whether exactly one element passes the test.
//
// A nil callback counts the whole collection instead of filtering it.
func (c Collection[T]) HasSole(callback func(value T, key int) bool) bool {
	if callback == nil {
		return len(c) == 1
	}
	return len(c.Filter(callback)) == 1
}

// HasMany reports whether at least two elements pass the test.
//
// The walk stops at the second match, which is what makes it cheap on a long
// collection. A nil callback reads the length instead.
func (c Collection[T]) HasMany(callback func(value T, key int) bool) bool {
	if callback == nil {
		return len(c) >= 2
	}
	found := 0
	for i, v := range c {
		if !callback(v, i) {
			continue
		}
		if found++; found == 2 {
			return true
		}
	}
	return false
}

// ContainsManyItems reports whether at least two elements pass the test. It is
// HasMany, which is the name to prefer.
func (c Collection[T]) ContainsManyItems(callback func(value T, key int) bool) bool {
	return c.HasMany(callback)
}

// ForPage returns the perPage elements of the one-based page.
//
// A page below one reads from the start: the offset is clamped at zero rather
// than counting backwards. A negative perPage yields nothing.
func (c Collection[T]) ForPage(page, perPage int) Collection[T] {
	offset := (page - 1) * perPage
	if offset < 0 {
		offset = 0
	}
	if perPage < 0 {
		perPage = 0
	}
	return c.Slice(offset, perPage)
}

// PipeThrough feeds the collection through the callbacks in order, each one
// receiving what the previous returned.
//
// Every callback here takes and returns a collection; Pipe is there for the
// shape that returns something else.
func (c Collection[T]) PipeThrough(callbacks ...func(collection Collection[T]) Collection[T]) Collection[T] {
	carry := c
	for _, callback := range callbacks {
		carry = callback(carry)
	}
	return carry
}

// Dump writes the elements to standard output and returns the collection, so
// the call can sit inside a chain.
func (c Collection[T]) Dump() Collection[T] {
	fmt.Fprintf(os.Stdout, "%#v\n", c.All())
	return c
}

// Dd writes the elements to standard output and ends the process with status
// 1. It never returns, which is why it is not chainable.
func (c Collection[T]) Dd() {
	c.Dump()
	os.Exit(1)
}

// Each hands every element to the callback in order.
//
// The callback stops the walk by returning false. It returns the collection so
// the call can be chained.
func (c Collection[T]) Each(callback func(value T, key int) bool) Collection[T] {
	for i, v := range c {
		if !callback(v, i) {
			break
		}
	}
	return c
}

// Every reports whether every element passes the test.
//
// An empty collection returns true: there is no element to fail the test.
func (c Collection[T]) Every(callback func(value T, key int) bool) bool {
	for i, v := range c {
		if !callback(v, i) {
			return false
		}
	}
	return true
}

// Contains reports whether any element passes the test. To search for a value
// rather than run a test, use ContainsStrict.
func (c Collection[T]) Contains(callback func(value T, key int) bool) bool {
	_, ok := c.First(callback)
	return ok
}

// Some reports whether any element passes the test. It is Contains.
func (c Collection[T]) Some(callback func(value T, key int) bool) bool {
	return c.Contains(callback)
}

// DoesntContain reports whether no element passes the test. It is the negation
// of Contains.
func (c Collection[T]) DoesntContain(callback func(value T, key int) bool) bool {
	return !c.Contains(callback)
}

// ContainsOneItem reports whether exactly one element passes the test. A nil
// callback reads the length instead.
func (c Collection[T]) ContainsOneItem(callback func(value T, key int) bool) bool {
	if callback == nil {
		return len(c) == 1
	}
	return len(c.Filter(callback)) == 1
}

// Percentage returns the share of elements passing the test, from 0 to 100,
// rounded to precision decimal places.
//
// The second result is false on an empty collection.
func (c Collection[T]) Percentage(callback func(value T, key int) bool, precision int) (float64, bool) {
	if len(c) == 0 {
		return 0, false
	}
	raw := float64(len(c.Filter(callback))) / float64(len(c)) * 100
	factor := math.Pow(10, float64(precision))
	return math.Round(raw*factor) / factor, true
}

// Partition returns the elements passing the test, then the ones failing it.
//
// Neither half is nil, and the two together hold every element exactly once.
func (c Collection[T]) Partition(callback func(value T, key int) bool) (passed, failed Collection[T]) {
	passed = make(Collection[T], 0, len(c))
	failed = make(Collection[T], 0, len(c))
	for i, v := range c {
		if callback(v, i) {
			passed = append(passed, v)
		} else {
			failed = append(failed, v)
		}
	}
	return passed, failed
}

// Slice returns the run of elements starting at offset.
//
// A negative offset counts from the end, and a negative length stops that many
// elements before the end. Omit length to read to the end; only the first one
// is used, the variadic standing in for an optional argument.
func (c Collection[T]) Slice(offset int, length ...int) Collection[T] {
	start := offset
	if start < 0 {
		start = len(c) + start
		if start < 0 {
			start = 0
		}
	}
	if start > len(c) {
		return Collection[T]{}
	}

	end := len(c)
	if len(length) > 0 {
		n := length[0]
		if n < 0 {
			end = len(c) + n
		} else {
			end = start + n
		}
	}
	if end > len(c) {
		end = len(c)
	}
	if end < start {
		end = start
	}

	out := make(Collection[T], end-start)
	copy(out, c[start:end])
	return out
}

// Take returns the first limit elements. A negative limit takes that many from
// the end instead.
func (c Collection[T]) Take(limit int) Collection[T] {
	if limit < 0 {
		return c.Slice(limit, -limit)
	}
	return c.Slice(0, limit)
}

// Skip returns everything past the first count elements.
func (c Collection[T]) Skip(count int) Collection[T] { return c.Slice(count) }

// TakeWhile returns the leading run of elements passing the test.
func (c Collection[T]) TakeWhile(callback func(value T, key int) bool) Collection[T] {
	for i, v := range c {
		if !callback(v, i) {
			return c.Slice(0, i)
		}
	}
	return c.Values()
}

// TakeUntil returns the elements before the first one passing the test.
func (c Collection[T]) TakeUntil(callback func(value T, key int) bool) Collection[T] {
	return c.TakeWhile(func(value T, key int) bool { return !callback(value, key) })
}

// SkipWhile returns everything from the first element failing the test
// onwards.
func (c Collection[T]) SkipWhile(callback func(value T, key int) bool) Collection[T] {
	for i, v := range c {
		if !callback(v, i) {
			return c.Slice(i)
		}
	}
	return Collection[T]{}
}

// SkipUntil returns everything from the first element passing the test
// onwards.
func (c Collection[T]) SkipUntil(callback func(value T, key int) bool) Collection[T] {
	return c.SkipWhile(func(value T, key int) bool { return !callback(value, key) })
}

// Chunk breaks the collection into runs of size elements.
//
// A size below one yields an empty collection rather than looping forever. The
// last chunk is short when the length does not divide evenly.
func Chunk[T any](c Collection[T], size int) Collection[Collection[T]] {
	if size < 1 {
		return Collection[Collection[T]]{}
	}
	out := make(Collection[Collection[T]], 0, (len(c)+size-1)/size)
	for start := 0; start < len(c); start += size {
		out = append(out, c.Slice(start, size))
	}
	return out
}

// ChunkWhile breaks the collection wherever the callback reports false for an
// element against the chunk built so far, which starts a new chunk.
func ChunkWhile[T any](c Collection[T], callback func(value T, key int, chunk Collection[T]) bool) Collection[Collection[T]] {
	out := Collection[Collection[T]]{}
	var chunk Collection[T]
	for i, v := range c {
		if len(chunk) == 0 || callback(v, i, chunk) {
			chunk = append(chunk, v)
			continue
		}
		out = append(out, chunk)
		chunk = Collection[T]{v}
	}
	if len(chunk) > 0 {
		out = append(out, chunk)
	}
	return out
}

// Split deals the elements into numberOfGroups groups, the earlier groups
// taking the remainder.
//
// It returns ErrInvalidArgument when numberOfGroups is below one.
func Split[T any](c Collection[T], numberOfGroups int) (Collection[Collection[T]], error) {
	if numberOfGroups < 1 {
		return nil, fmt.Errorf("%w: number of groups must be at least 1, got %d", ErrInvalidArgument, numberOfGroups)
	}
	if len(c) == 0 {
		return Collection[Collection[T]]{}, nil
	}

	groups := make(Collection[Collection[T]], 0, numberOfGroups)
	groupSize := len(c) / numberOfGroups
	remain := len(c) % numberOfGroups
	start := 0
	for i := 0; i < numberOfGroups; i++ {
		size := groupSize
		if i < remain {
			size++
		}
		if size == 0 {
			continue
		}
		groups = append(groups, c.Slice(start, size))
		start += size
	}
	return groups, nil
}

// SplitIn returns chunks of the size that fits numberOfGroups groups, the last
// one short.
//
// It returns ErrInvalidArgument when numberOfGroups is below one.
func SplitIn[T any](c Collection[T], numberOfGroups int) (Collection[Collection[T]], error) {
	if numberOfGroups < 1 {
		return nil, fmt.Errorf("%w: number of groups must be at least 1, got %d", ErrInvalidArgument, numberOfGroups)
	}
	size := int(math.Ceil(float64(len(c)) / float64(numberOfGroups)))
	return Chunk(c, size), nil
}

// Sliding returns a sliding window of size elements, advancing step at a time.
//
// It returns ErrInvalidArgument when size or step is below one. A collection
// shorter than the window yields no chunk.
func Sliding[T any](c Collection[T], size, step int) (Collection[Collection[T]], error) {
	switch {
	case size < 1:
		return nil, fmt.Errorf("%w: size value must be at least 1, got %d", ErrInvalidArgument, size)
	case step < 1:
		return nil, fmt.Errorf("%w: step value must be at least 1, got %d", ErrInvalidArgument, step)
	}
	chunks := (len(c)-size)/step + 1
	if len(c) < size {
		chunks = 0
	}
	out := make(Collection[Collection[T]], 0, chunks)
	for i := 0; i < chunks; i++ {
		out = append(out, c.Slice(i*step, size))
	}
	return out, nil
}

// Nth returns every step-th element, starting at offset.
//
// It returns ErrInvalidArgument when step is below one.
func (c Collection[T]) Nth(step, offset int) (Collection[T], error) {
	if step < 1 {
		return nil, fmt.Errorf("%w: step value must be at least 1, got %d", ErrInvalidArgument, step)
	}
	rest := c.Slice(offset)
	out := make(Collection[T], 0, (len(rest)+step-1)/step)
	for i, v := range rest {
		if i%step == 0 {
			out = append(out, v)
		}
	}
	return out, nil
}

// Splice removes a run of elements, optionally putting the replacement in its
// place, and returns what it removed.
//
// It mutates the receiver, which is why it takes a pointer. A nil length cuts
// to the end; a negative length stops that many elements before it.
func (c *Collection[T]) Splice(offset int, length *int, replacement ...T) Collection[T] {
	items := *c
	start := offset
	if start < 0 {
		start = len(items) + start
	}
	if start < 0 {
		start = 0
	}
	if start > len(items) {
		start = len(items)
	}

	end := len(items)
	if length != nil {
		if *length < 0 {
			end = len(items) + *length
		} else {
			end = start + *length
		}
	}
	if end > len(items) {
		end = len(items)
	}
	if end < start {
		end = start
	}

	removed := make(Collection[T], end-start)
	copy(removed, items[start:end])

	kept := make(Collection[T], 0, len(items)-(end-start)+len(replacement))
	kept = append(kept, items[:start]...)
	kept = append(kept, replacement...)
	kept = append(kept, items[end:]...)
	*c = kept

	return removed
}

// Sort orders a copy of the elements with compare, which returns zero for
// equal, the convention of the cmp and slices packages.
//
// The comparison is required, because Go cannot compare an arbitrary T; a nil
// compare returns the copy unsorted. The sort is stable.
func (c Collection[T]) Sort(compare func(a, b T) int) Collection[T] {
	out := c.Values()
	if compare != nil {
		stableSort(out, compare)
	}
	return out
}

// SortDesc is Sort with the comparison reversed.
func (c Collection[T]) SortDesc(compare func(a, b T) int) Collection[T] {
	if compare == nil {
		return c.Values()
	}
	return c.Sort(func(a, b T) int { return -compare(a, b) })
}

// Shuffle returns a copy of the elements in a random order.
//
// It draws from crypto/rand: a shuffle seeded from the clock is a shuffle an
// attacker can replay. When the draw fails the copy is returned as it stands.
func (c Collection[T]) Shuffle() Collection[T] {
	out := c.Values()
	for i := len(out) - 1; i > 0; i-- {
		j, err := rand.Int(rand.Reader, big.NewInt(int64(i+1)))
		if err != nil {
			return out
		}
		k := int(j.Int64())
		out[i], out[k] = out[k], out[i]
	}
	return out
}

// Random returns number elements picked at random.
//
// It returns ErrInvalidArgument when number is negative or exceeds the length.
func (c Collection[T]) Random(number int) (Collection[T], error) {
	if number < 0 || number > len(c) {
		return nil, fmt.Errorf("%w: cannot take %d items from a collection of %d", ErrInvalidArgument, number, len(c))
	}
	return c.Shuffle().Slice(0, number), nil
}

// Implode renders each element with value and joins the results with glue.
//
// A nil value renders each element with fmt.Sprint.
func (c Collection[T]) Implode(glue string, value func(item T) string) string {
	parts := make([]string, len(c))
	for i, v := range c {
		if value == nil {
			parts[i] = fmt.Sprint(v)
			continue
		}
		parts[i] = value(v)
	}
	return strings.Join(parts, glue)
}

// Join is Implode, except that the last element is attached with finalGlue.
//
// An empty finalGlue is Implode, one element is that element, and no element
// is the empty string.
func (c Collection[T]) Join(glue, finalGlue string, value func(item T) string) string {
	if finalGlue == "" {
		return c.Implode(glue, value)
	}
	switch len(c) {
	case 0:
		return ""
	case 1:
		return c.Implode(glue, value)
	}
	return c.Slice(0, len(c)-1).Implode(glue, value) + finalGlue + c.Slice(len(c)-1).Implode(glue, value)
}

// Tap hands the collection to the callback and returns it unchanged.
func (c Collection[T]) Tap(callback func(collection Collection[T])) Collection[T] {
	callback(c)
	return c
}

// When runs callback on the collection when condition holds, and otherwise on
// the other branch.
//
// Either branch may be nil, which returns the collection unchanged.
func (c Collection[T]) When(condition bool, callback, otherwise func(collection Collection[T]) Collection[T]) Collection[T] {
	if condition {
		if callback == nil {
			return c
		}
		return callback(c)
	}
	if otherwise == nil {
		return c
	}
	return otherwise(c)
}

// Unless is When with the condition negated.
func (c Collection[T]) Unless(condition bool, callback, otherwise func(collection Collection[T]) Collection[T]) Collection[T] {
	return c.When(!condition, callback, otherwise)
}

// WhenEmpty is When with the condition that the collection has no elements.
func (c Collection[T]) WhenEmpty(callback, otherwise func(collection Collection[T]) Collection[T]) Collection[T] {
	return c.When(len(c) == 0, callback, otherwise)
}

// WhenNotEmpty is When with the condition that the collection has at least one
// element.
func (c Collection[T]) WhenNotEmpty(callback, otherwise func(collection Collection[T]) Collection[T]) Collection[T] {
	return c.When(len(c) > 0, callback, otherwise)
}

// UnlessEmpty is WhenNotEmpty.
func (c Collection[T]) UnlessEmpty(callback, otherwise func(collection Collection[T]) Collection[T]) Collection[T] {
	return c.WhenNotEmpty(callback, otherwise)
}

// UnlessNotEmpty is WhenEmpty.
func (c Collection[T]) UnlessNotEmpty(callback, otherwise func(collection Collection[T]) Collection[T]) Collection[T] {
	return c.WhenEmpty(callback, otherwise)
}

// Only returns the elements at the given indices, in the collection's order.
//
// An index outside the collection is skipped.
func (c Collection[T]) Only(keys ...int) Collection[T] {
	wanted := make(map[int]struct{}, len(keys))
	for _, k := range keys {
		wanted[k] = struct{}{}
	}
	out := make(Collection[T], 0, len(keys))
	for i, v := range c {
		if _, ok := wanted[i]; ok {
			out = append(out, v)
		}
	}
	return out
}

// Except returns everything but the elements at the given indices.
func (c Collection[T]) Except(keys ...int) Collection[T] {
	unwanted := make(map[int]struct{}, len(keys))
	for _, k := range keys {
		unwanted[k] = struct{}{}
	}
	out := make(Collection[T], 0, len(c))
	for i, v := range c {
		if _, ok := unwanted[i]; !ok {
			out = append(out, v)
		}
	}
	return out
}

// Replace overwrites the elements at the indices items names, and appends the
// ones past the end in ascending index order. A negative index is ignored.
func (c Collection[T]) Replace(items map[int]T) Collection[T] {
	out := c.Values()
	extra := make([]int, 0, len(items))
	for k := range items {
		if k >= 0 && k < len(out) {
			out[k] = items[k]
			continue
		}
		if k >= 0 {
			extra = append(extra, k)
		}
	}
	stableSort(extra, cmp.Compare)
	for _, k := range extra {
		out = append(out, items[k])
	}
	return out
}

// CountBy, GroupBy, KeyBy and the rest that change the element type are
// package functions, because a Go method cannot declare a type parameter.
