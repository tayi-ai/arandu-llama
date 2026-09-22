package collections

import (
	"cmp"
	"fmt"
	"strconv"
	"strings"
)

// Number is the set of types Sum can add. It is written out here rather than
// imported, so that the package carries no dependency for it.
//
// Complex numbers are absent on purpose: admitting them would make the zero
// value of an empty sum harder to explain than it is worth.
type Number interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr |
		~float32 | ~float64
}

// Map returns the result of the callback for every element, in order.
//
// It is a function and not a method because the callback changes the element
// type, and a Go method cannot declare a type parameter.
func Map[T, U any](c Collection[T], callback func(value T, key int) U) Collection[U] {
	out := make([]U, len(c))
	for i, v := range c {
		out[i] = callback(v, i)
	}
	return Collection[U](out)
}

// MapSpread spreads each nested chunk across the callback's arguments.
//
// The position is appended to the chunk before it is spread, so the callback
// receives the chunk's elements and then that position. That is why the
// element type is []any and not []T: the trailing key is an int and the
// elements need not be.
func MapSpread[U any](c Collection[[]any], callback func(values ...any) U) Collection[U] {
	out := make([]U, len(c))
	for i, chunk := range c {
		out[i] = callback(spreadWithKey(chunk, i)...)
	}
	return Collection[U](out)
}

// EachSpread hands each nested chunk to the callback, spread across its
// arguments, and returns the collection unchanged.
//
// As in MapSpread the position is appended to the chunk before it is spread.
// Returning false from the callback stops the walk.
func EachSpread(c Collection[[]any], callback func(values ...any) bool) Collection[[]any] {
	for i, chunk := range c {
		if !callback(spreadWithKey(chunk, i)...) {
			break
		}
	}
	return c
}

// spreadWithKey appends the key to the chunk, on a copy, so that the
// collection's own chunk is not grown by being spread.
func spreadWithKey(chunk []any, key int) []any {
	out := make([]any, 0, len(chunk)+1)
	out = append(out, chunk...)
	return append(out, key)
}

// MapInto builds a U from every element by handing it to into.
//
// The constructor itself is the argument: Go cannot reach a type from a name
// in a string, so there is no class name to pass.
func MapInto[T, U any](c Collection[T], into func(value T) U) Collection[U] {
	out := make([]U, len(c))
	for i, v := range c {
		out[i] = into(v)
	}
	return Collection[U](out)
}

// FlatMap concatenates the slice the callback returns for every element.
func FlatMap[T, U any](c Collection[T], callback func(value T, key int) []U) Collection[U] {
	out := make([]U, 0, len(c))
	for i, v := range c {
		out = append(out, callback(v, i)...)
	}
	return Collection[U](out)
}

// Pluck reads one field off every element.
//
// The field is named with an accessor rather than a string, because Go cannot
// reach a field from a name at run time. To key the result by a second field,
// use MapWithKeys.
func Pluck[T, V any](c Collection[T], value func(item T) V) Collection[V] {
	out := make([]V, len(c))
	for i, v := range c {
		out[i] = value(v)
	}
	return Collection[V](out)
}

// Value reads one field off the first item. The second result is false on an
// empty collection.
func Value[T, V any](c Collection[T], key func(item T) V) (V, bool) {
	if len(c) == 0 {
		var zero V
		return zero, false
	}
	return key(c[0]), true
}

// Reduce folds the elements into a single value, starting from initial and
// carrying the callback's result forward.
func Reduce[T, A any](c Collection[T], callback func(carry A, value T, key int) A, initial A) A {
	carry := initial
	for i, v := range c {
		carry = callback(carry, v, i)
	}
	return carry
}

// ReduceWithKeys is Reduce, under a name that says the key is passed to the
// callback.
func ReduceWithKeys[T, A any](c Collection[T], callback func(carry A, value T, key int) A, initial A) A {
	return Reduce(c, callback, initial)
}

// ReduceSpread is a reduction that carries several accumulators at once.
//
// A reducer that returns a nil slice stops the fold and is reported as
// ErrUnexpectedValue.
func ReduceSpread[T, A any](c Collection[T], callback func(carry []A, value T, key int) []A, initial ...A) ([]A, error) {
	carry := append(make([]A, 0, len(initial)), initial...)
	for i, v := range c {
		carry = callback(carry, v, i)
		if carry == nil {
			return nil, fmt.Errorf("%w: reduceSpread expects the reducer to return a slice", ErrUnexpectedValue)
		}
	}
	return carry, nil
}

// Pipe hands the whole collection to the callback and returns its result.
func Pipe[T, U any](c Collection[T], callback func(collection Collection[T]) U) U {
	return callback(c)
}

// PipeInto builds a U from the whole collection by handing it to into.
//
// As in MapInto the constructor itself is the argument.
func PipeInto[T, U any](c Collection[T], into func(collection Collection[T]) U) U {
	return into(c)
}

// GroupBy gathers the items under the key the callback reads off each one.
//
// Within each group the items keep the order they had in the collection. The
// keys of a Collection[T] are positions, and a group renumbers them, so the
// original positions are not preserved.
func GroupBy[T any, K comparable](c Collection[T], groupBy func(value T, key int) K) map[K]Collection[T] {
	out := make(map[K]Collection[T])
	for i, v := range c {
		k := groupBy(v, i)
		out[k] = append(out[k], v)
	}
	return out
}

// KeyBy keys the items by what the callback reads off each one.
//
// When two items share a key the later one wins.
func KeyBy[T any, K comparable](c Collection[T], keyBy func(value T, key int) K) map[K]T {
	out := make(map[K]T, len(c))
	for i, v := range c {
		out[keyBy(v, i)] = v
	}
	return out
}

// CountBy counts how many items fall under each key the callback reads.
func CountBy[T any, K comparable](c Collection[T], countBy func(value T, key int) K) map[K]int {
	out := make(map[K]int)
	for i, v := range c {
		out[countBy(v, i)]++
	}
	return out
}

// MapWithKeys builds a map from the key and value the callback returns for
// every element. A repeated key keeps the last value.
func MapWithKeys[T any, K comparable, V any](c Collection[T], callback func(value T, key int) (K, V)) map[K]V {
	out := make(map[K]V, len(c))
	for i, v := range c {
		k, val := callback(v, i)
		out[k] = val
	}
	return out
}

// MapToDictionary is MapWithKeys where every key keeps all the values mapped
// onto it, in order, rather than only the last.
func MapToDictionary[T any, K comparable, V any](c Collection[T], callback func(value T, key int) (K, V)) map[K][]V {
	out := make(map[K][]V)
	for i, v := range c {
		k, val := callback(v, i)
		out[k] = append(out[k], val)
	}
	return out
}

// MapToGroups is MapToDictionary with each bucket wrapped in a collection.
func MapToGroups[T any, K comparable, V any](c Collection[T], callback func(value T, key int) (K, V)) map[K]Collection[V] {
	out := make(map[K]Collection[V])
	for k, v := range MapToDictionary(c, callback) {
		out[k] = Collection[V](v)
	}
	return out
}

// Flip maps every value to its position.
//
// When a value repeats, the last position wins.
func Flip[T comparable](c Collection[T]) map[T]int {
	out := make(map[T]int, len(c))
	for i, v := range c {
		out[v] = i
	}
	return out
}

// Combine builds a map, this collection supplying the keys and values the
// values.
//
// Pairing stops at the shorter of the two.
func Combine[K comparable, V any](keys Collection[K], values []V) map[K]V {
	n := min(len(keys), len(values))
	out := make(map[K]V, n)
	for i := 0; i < n; i++ {
		out[keys[i]] = values[i]
	}
	return out
}

// Unique drops the items whose key the callback has already produced.
//
// The first item carrying a key is the one kept, and the order of the
// collection survives.
func Unique[T any, K comparable](c Collection[T], key func(value T, k int) K) Collection[T] {
	seen := make(map[K]struct{}, len(c))
	out := make([]T, 0, len(c))
	for i, v := range c {
		k := key(v, i)
		if _, ok := seen[k]; ok {
			continue
		}
		seen[k] = struct{}{}
		out = append(out, v)
	}
	return Collection[T](out)
}

// UniqueStrict drops the repeated keys exactly as Unique does. Go's == is
// already an identity comparison, so the two collapse into one behaviour and
// this name forwards to Unique.
func UniqueStrict[T any, K comparable](c Collection[T], key func(value T, k int) K) Collection[T] {
	return Unique(c, key)
}

// Duplicates reports the keys that appear more than once, in the order their
// repeats appear. A key seen three times is reported twice.
func Duplicates[T any, K comparable](c Collection[T], key func(value T, k int) K) Collection[K] {
	seen := make(map[K]struct{}, len(c))
	out := make([]K, 0)
	for i, v := range c {
		k := key(v, i)
		if _, ok := seen[k]; ok {
			out = append(out, k)
			continue
		}
		seen[k] = struct{}{}
	}
	return Collection[K](out)
}

// DuplicatesStrict reports the keys that appear more than once, exactly as
// Duplicates does. Go's == is already an identity comparison, so the two
// collapse into one behaviour and this name forwards to Duplicates.
func DuplicatesStrict[T any, K comparable](c Collection[T], key func(value T, k int) K) Collection[K] {
	return Duplicates(c, key)
}

// Sum adds the number the projection reads off every element.
//
// It returns the zero of N on an empty collection. There is no form without
// the projection: over a collection of numbers, pass a function that returns
// its argument.
func Sum[T any, N Number](c Collection[T], value func(item T) N) N {
	var total N
	for _, v := range c {
		total += value(v)
	}
	return total
}

// Avg returns the mean of the number the projection reads off every element.
//
// The second result is false on an empty collection.
func Avg[T any, N Number](c Collection[T], value func(item T) N) (float64, bool) {
	if len(c) == 0 {
		return 0, false
	}
	return float64(Sum(c, value)) / float64(len(c)), true
}

// Average is Avg.
func Average[T any, N Number](c Collection[T], value func(item T) N) (float64, bool) {
	return Avg(c, value)
}

// Median returns the middle of the numbers the projection reads.
//
// With an even count it averages the two middle values. The second result is
// false on an empty collection.
func Median[T any, N Number](c Collection[T], value func(item T) N) (float64, bool) {
	if len(c) == 0 {
		return 0, false
	}
	values := make([]float64, len(c))
	for i, v := range c {
		values[i] = float64(value(v))
	}
	stableSort(values, cmp.Compare)
	middle := len(values) / 2
	if len(values)%2 == 1 {
		return values[middle], true
	}
	return (values[middle-1] + values[middle]) / 2, true
}

// Mode returns the values that occur most often, sorted ascending.
//
// The result is nil on an empty collection.
func Mode[T any, N cmp.Ordered](c Collection[T], value func(item T) N) []N {
	if len(c) == 0 {
		return nil
	}
	counts := make(map[N]int, len(c))
	for _, v := range c {
		counts[value(v)]++
	}
	highest := 0
	for _, n := range counts {
		highest = max(highest, n)
	}
	out := make([]N, 0, len(counts))
	for k, n := range counts {
		if n == highest {
			out = append(out, k)
		}
	}
	stableSort(out, cmp.Compare)
	return out
}

// Min returns the smallest value the projection reads.
//
// The second result is false on an empty collection.
func Min[T any, V cmp.Ordered](c Collection[T], value func(item T) V) (V, bool) {
	var lowest V
	if len(c) == 0 {
		return lowest, false
	}
	lowest = value(c[0])
	for _, v := range c[1:] {
		lowest = min(lowest, value(v))
	}
	return lowest, true
}

// Max returns the largest value the projection reads.
//
// The second result is false on an empty collection.
func Max[T any, V cmp.Ordered](c Collection[T], value func(item T) V) (V, bool) {
	var highest V
	if len(c) == 0 {
		return highest, false
	}
	highest = value(c[0])
	for _, v := range c[1:] {
		highest = max(highest, value(v))
	}
	return highest, true
}

// SortBy orders the elements by the value the callback reads off each one. The
// sort is stable.
func SortBy[T any, V cmp.Ordered](c Collection[T], callback func(value T, key int) V) Collection[T] {
	type keyed struct {
		item T
		key  V
	}
	pairs := make([]keyed, len(c))
	for i, v := range c {
		pairs[i] = keyed{item: v, key: callback(v, i)}
	}
	stableSort(pairs, func(a, b keyed) int { return cmp.Compare(a.key, b.key) })
	out := make([]T, len(pairs))
	for i, p := range pairs {
		out[i] = p.item
	}
	return Collection[T](out)
}

// SortByDesc is SortBy with the order reversed.
func SortByDesc[T any, V cmp.Ordered](c Collection[T], callback func(value T, key int) V) Collection[T] {
	return SortBy(c, callback).Reverse()
}

// Where keeps the items whose projected key compares to value under the
// operator.
//
// The operator is one of "=", "==", "===", "!=", "<>", "!==", ">", ">=", "<"
// and "<=". An unknown operator matches nothing.
func Where[T any, V cmp.Ordered](c Collection[T], key func(item T) V, operator string, value V) Collection[T] {
	return c.Filter(func(item T, _ int) bool {
		return compareOperator(key(item), operator, value)
	})
}

// FirstWhere returns the first item whose projected key satisfies the
// comparison. It is Where followed by First, and takes the same operator set
// Where documents.
//
// The second result is false when nothing matches, so a zero-valued item is
// never mistaken for a hit. The whole collection is filtered before the first
// survivor is taken, so this is a convenience over Where, not a cheaper scan.
func FirstWhere[T any, V cmp.Ordered](c Collection[T], key func(item T) V, operator string, value V) (T, bool) {
	return Where(c, key, operator, value).First(nil)
}

// WhereStrict keeps the items whose projected key equals value. It takes no
// operator: Go's == is already an identity comparison.
func WhereStrict[T any, V comparable](c Collection[T], key func(item T) V, value V) Collection[T] {
	return c.Filter(func(item T, _ int) bool { return key(item) == value })
}

// WhereIn keeps the items whose projected key is one of values.
//
// The order of the collection survives, and values is loaded into a set before
// the scan, so the cost is one pass regardless of how long values is.
func WhereIn[T any, V comparable](c Collection[T], key func(item T) V, values []V) Collection[T] {
	want := make(map[V]struct{}, len(values))
	for _, v := range values {
		want[v] = struct{}{}
	}
	return c.Filter(func(item T, _ int) bool {
		_, ok := want[key(item)]
		return ok
	})
}

// WhereInStrict keeps the items whose projected key is one of values, exactly
// as WhereIn does. Go's == is already an identity comparison, so this name
// forwards to WhereIn.
func WhereInStrict[T any, V comparable](c Collection[T], key func(item T) V, values []V) Collection[T] {
	return WhereIn(c, key, values)
}

// WhereNotIn keeps the items whose projected key is absent from values: the
// complement of WhereIn, and the way to exclude a known set without spelling
// out the negated predicate.
//
// The order of the collection survives, and values is loaded into a set
// before the scan, so the cost is one pass regardless of how long values is.
func WhereNotIn[T any, V comparable](c Collection[T], key func(item T) V, values []V) Collection[T] {
	want := make(map[V]struct{}, len(values))
	for _, v := range values {
		want[v] = struct{}{}
	}
	return c.Filter(func(item T, _ int) bool {
		_, ok := want[key(item)]
		return !ok
	})
}

// WhereNotInStrict keeps the items whose projected key is absent from values,
// exactly as WhereNotIn does. Go's == is already an identity comparison, so
// this name forwards to WhereNotIn.
func WhereNotInStrict[T any, V comparable](c Collection[T], key func(item T) V, values []V) Collection[T] {
	return WhereNotIn(c, key, values)
}

// WhereBetween keeps the items whose projected key falls in the range. Both
// ends are included.
func WhereBetween[T any, V cmp.Ordered](c Collection[T], key func(item T) V, from, to V) Collection[T] {
	return c.Filter(func(item T, _ int) bool {
		v := key(item)
		return v >= from && v <= to
	})
}

// WhereNotBetween keeps the items whose projected key falls outside the
// range: below from, or above to.
//
// Both ends count as inside, so an item sitting exactly on from or on to is
// dropped. That makes this the exact complement of WhereBetween -- every item
// is kept by one of the two and by neither twice.
func WhereNotBetween[T any, V cmp.Ordered](c Collection[T], key func(item T) V, from, to V) Collection[T] {
	return c.Filter(func(item T, _ int) bool {
		v := key(item)
		return v < from || v > to
	})
}

// WhereNull keeps the items whose accessor returns a nil pointer.
//
// The accessor returns a pointer because a pointer is how a field that may be
// absent is spelled in Go.
func WhereNull[T any, V any](c Collection[T], key func(item T) *V) Collection[T] {
	return c.Filter(func(item T, _ int) bool { return key(item) == nil })
}

// WhereNotNull keeps the items whose accessor returns a non-nil pointer, and
// drops those where the field is absent. It is the complement of WhereNull.
//
// The accessor returns a pointer because a pointer is how a field that may be
// absent is spelled in Go.
func WhereNotNull[T any, V any](c Collection[T], key func(item T) *V) Collection[T] {
	return c.Filter(func(item T, _ int) bool { return key(item) != nil })
}

// WhereInstanceOf keeps the items whose dynamic type is U.
func WhereInstanceOf[T, U any](c Collection[T]) Collection[T] {
	return c.Filter(func(item T, _ int) bool {
		_, ok := any(item).(U)
		return ok
	})
}

// Ensure returns the collection unchanged when every item is a U, and
// ErrUnexpectedValue naming the first item that is not.
func Ensure[T, U any](c Collection[T]) (Collection[T], error) {
	for i, v := range c {
		if _, ok := any(v).(U); !ok {
			var want U
			return c, fmt.Errorf("%w: item at %d is %T, expected %T", ErrUnexpectedValue, i, v, want)
		}
	}
	return c, nil
}

// ContainsStrict reports whether any item equals value. It is the search by
// value, where Contains runs a test.
func ContainsStrict[T comparable](c Collection[T], value T) bool {
	for _, v := range c {
		if v == value {
			return true
		}
	}
	return false
}

// DoesntContainStrict reports whether no item equals value. It is the
// negation of ContainsStrict, and exists so the absent case reads as its own
// call rather than as an exclamation mark in front of somebody else's.
func DoesntContainStrict[T comparable](c Collection[T], value T) bool {
	return !ContainsStrict(c, value)
}

// Search returns the index of the first item equal to value. The second result
// is false when there is none.
func Search[T comparable](c Collection[T], value T) (int, bool) {
	for i, v := range c {
		if v == value {
			return i, true
		}
	}
	return 0, false
}

// Before returns the item sitting just before the first one equal to value.
//
// The second result is false when the value is absent or is the first item.
func Before[T comparable](c Collection[T], value T) (T, bool) {
	i, ok := Search(c, value)
	if !ok || i == 0 {
		var zero T
		return zero, false
	}
	return c[i-1], true
}

// After returns the item sitting just after the first one equal to value.
//
// The second result is false when the value is absent or is the last item.
func After[T comparable](c Collection[T], value T) (T, bool) {
	i, ok := Search(c, value)
	if !ok || i == len(c)-1 {
		var zero T
		return zero, false
	}
	return c[i+1], true
}

// Diff keeps the items not present in items.
func Diff[T comparable](c Collection[T], items []T) Collection[T] {
	drop := make(map[T]struct{}, len(items))
	for _, v := range items {
		drop[v] = struct{}{}
	}
	return c.Filter(func(v T, _ int) bool {
		_, ok := drop[v]
		return !ok
	})
}

// DiffUsing is Diff with the match decided by compare instead of by ==.
// compare returns zero for equal, the convention of the cmp and slices
// packages.
//
// Each item is walked against items until one matches, so the cost is the
// product of the two lengths where Diff is linear.
func DiffUsing[T any](c Collection[T], items []T, compare func(a, b T) int) Collection[T] {
	return c.Filter(func(v T, _ int) bool {
		for _, other := range items {
			if compare(v, other) == 0 {
				return false
			}
		}
		return true
	})
}

// Intersect keeps the items also present in items.
func Intersect[T comparable](c Collection[T], items []T) Collection[T] {
	keep := make(map[T]struct{}, len(items))
	for _, v := range items {
		keep[v] = struct{}{}
	}
	return c.Filter(func(v T, _ int) bool {
		_, ok := keep[v]
		return ok
	})
}

// IntersectUsing keeps the items that match something in items, with the
// match decided by compare instead of by ==. compare returns zero for equal,
// the convention of the cmp and slices packages.
//
// It is Intersect for element types that == cannot decide, or should not:
// a struct with a field to ignore, or a name to match without regard to case.
// Each item is walked against items until one matches, so the cost is the
// product of the two lengths where Intersect is linear. Reach for it when ==
// cannot answer the question, not by default.
func IntersectUsing[T any](c Collection[T], items []T, compare func(a, b T) int) Collection[T] {
	return c.Filter(func(v T, _ int) bool {
		for _, other := range items {
			if compare(v, other) == 0 {
				return true
			}
		}
		return false
	})
}

// Collapse concatenates the inner collections into one, in order.
func Collapse[T any](c Collection[Collection[T]]) Collection[T] {
	out := make([]T, 0, len(c))
	for _, inner := range c {
		out = append(out, inner...)
	}
	return Collection[T](out)
}

// Flatten squashes a nested structure into one level.
//
// The depth is optional and unlimited when omitted; only the first one is
// used. A nested value is a []any or a Collection[any]; anything else is a
// leaf.
func Flatten(c Collection[any], depth ...int) Collection[any] {
	limit := -1
	if len(depth) > 0 {
		limit = depth[0]
	}
	return Collection[any](flattenInto(make([]any, 0, len(c)), c, limit))
}

func flattenInto(out []any, items []any, depth int) []any {
	for _, item := range items {
		var nested []any
		switch v := item.(type) {
		case []any:
			nested = v
		case Collection[any]:
			nested = v
		default:
			out = append(out, item)
			continue
		}
		if depth == 0 {
			out = append(out, nested...)
			continue
		}
		out = flattenInto(out, nested, depth-1)
	}
	return out
}

// Select reduces every item to the named keys. A key an item does not have is
// left out rather than filled with the zero value.
func Select[T any](c Collection[map[string]T], keys ...string) Collection[map[string]T] {
	out := make([]map[string]T, len(c))
	for i, item := range c {
		row := make(map[string]T, len(keys))
		for _, k := range keys {
			if v, ok := item[k]; ok {
				row[k] = v
			}
		}
		out[i] = row
	}
	return Collection[map[string]T](out)
}

// Dot flattens the collection into single-level "dot" keys, the top-level key
// of each element being its position.
func Dot(c Collection[any]) map[string]any {
	out := make(map[string]any, len(c))
	for i, v := range c {
		dotInto(out, strconv.Itoa(i), v)
	}
	return out
}

func dotInto(out map[string]any, prefix string, value any) {
	switch v := value.(type) {
	case map[string]any:
		if len(v) == 0 {
			out[prefix] = value
			return
		}
		for k, inner := range v {
			dotInto(out, prefix+"."+k, inner)
		}
	case []any:
		if len(v) == 0 {
			out[prefix] = value
			return
		}
		for i, inner := range v {
			dotInto(out, prefix+"."+strconv.Itoa(i), inner)
		}
	case Collection[any]:
		dotInto(out, prefix, v)
	default:
		out[prefix] = value
	}
}

// Undot expands "dot" keys back into nested maps.
func Undot(m map[string]any) map[string]any {
	out := map[string]any{}
	for k, v := range m {
		setDotted(out, k, v)
	}
	return out
}

func setDotted(m map[string]any, key string, value any) {
	parts := strings.Split(key, ".")
	for _, part := range parts[:len(parts)-1] {
		next, ok := m[part].(map[string]any)
		if !ok {
			next = map[string]any{}
			m[part] = next
		}
		m = next
	}
	m[parts[len(parts)-1]] = value
}

// compareOperator applies the comparison the operator names. An operator the
// list does not name matches nothing.
func compareOperator[V cmp.Ordered](left V, operator string, right V) bool {
	switch operator {
	case "=", "==", "===":
		return left == right
	case "!=", "<>", "!==":
		return left != right
	case ">":
		return left > right
	case ">=":
		return left >= right
	case "<":
		return left < right
	case "<=":
		return left <= right
	default:
		return false
	}
}
