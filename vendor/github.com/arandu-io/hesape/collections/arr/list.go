package arr

import (
	"cmp"
	"crypto/rand"
	"fmt"
	"math/big"
	"reflect"
	"sort"
	"strings"

	"github.com/arandu-io/hesape/collections"
)

// Collapse removes one level of nesting.
//
// Only the elements that are themselves indexable contribute; everything else
// is dropped. The typed form, over a slice of slices, is collections.Collapse.
func Collapse(array []any) []any {
	out := []any{}
	for _, values := range array {
		nested, ok := elements(values)
		if !ok {
			continue
		}
		out = append(out, nested...)
	}
	return out
}

// CrossJoin returns the cartesian product of the lists, one tuple per
// combination.
//
// The product is folded from a single empty tuple, so crossing nothing gives
// one empty tuple, crossing a single list wraps each of its elements on its
// own, and crossing an empty list yields nothing at all.
func CrossJoin[T any](arrays ...[]T) [][]T {
	results := [][]T{{}}
	for _, array := range arrays {
		next := make([][]T, 0, len(results)*len(array))
		for _, product := range results {
			for _, item := range array {
				combined := make([]T, len(product), len(product)+1)
				copy(combined, product)
				next = append(next, append(combined, item))
			}
		}
		results = next
	}
	return results
}

// First returns the first element passing the test.
//
// A nil callback returns the first element. The second result is false when
// the slice is empty or nothing matched.
//
// The miss is reported as a bool and not as a default taken as an argument,
// because an element equal to the zero value of T is a hit and a default
// returned in its place could not be told from one. It also spares the caller
// building a fallback for a call that finds something.
func First[T any](array []T, callback func(value T, key int) bool) (T, bool) {
	for i, v := range array {
		if callback == nil || callback(v, i) {
			return v, true
		}
	}
	var zero T
	return zero, false
}

// Last returns the last element passing the test.
//
// The slice is walked backwards rather than reversed, and the callback still
// sees each element's original position. A nil callback returns the last
// element.
//
// The second result reports the miss, for the reason it does in First: a zero
// value found and nothing found are different answers.
func Last[T any](array []T, callback func(value T, key int) bool) (T, bool) {
	for i := len(array) - 1; i >= 0; i-- {
		if callback == nil || callback(array[i], i) {
			return array[i], true
		}
	}
	var zero T
	return zero, false
}

// Take returns the first limit elements, or the last ones when limit is
// negative.
//
// A limit of zero gives nothing, and a limit larger than the slice gives the
// whole slice. The result is always a fresh slice.
func Take[T any](array []T, limit int) []T {
	if limit < 0 {
		if -limit >= len(array) {
			return append([]T{}, array...)
		}
		return append([]T{}, array[len(array)+limit:]...)
	}
	if limit > len(array) {
		limit = len(array)
	}
	return append([]T{}, array[:limit]...)
}

// Flatten squashes a nested structure into one level.
//
// The depth is optional and unlimited when omitted; a depth of 1 removes one
// level of nesting. Only the first depth is used.
//
// A nested map contributes its values in ascending key order, since a Go map
// holds no order of its own.
func Flatten(array []any, depth ...int) []any {
	limit := -1
	if len(depth) > 0 {
		limit = depth[0]
	}
	return flattenInto([]any{}, array, limit)
}

func flattenInto(out []any, array []any, depth int) []any {
	for _, item := range array {
		nested, ok := elements(item)
		if !ok {
			out = append(out, item)
			continue
		}
		if depth == 1 {
			out = append(out, nested...)
			continue
		}
		out = flattenInto(out, nested, depth-1)
	}
	return out
}

// Every reports whether every element passes the test.
//
// An empty slice passes: there is no element to fail the test.
func Every[T any](array []T, callback func(value T, key int) bool) bool {
	for i, v := range array {
		if !callback(v, i) {
			return false
		}
	}
	return true
}

// Some reports whether at least one element passes the test.
func Some[T any](array []T, callback func(value T, key int) bool) bool {
	for i, v := range array {
		if callback(v, i) {
			return true
		}
	}
	return false
}

// Join glues the elements together, attaching the last one with finalGlue.
//
// An empty finalGlue joins with glue throughout, one element is that element
// and no element is the empty string.
func Join(array []string, glue, finalGlue string) string {
	if finalGlue == "" {
		return strings.Join(array, glue)
	}
	switch len(array) {
	case 0:
		return ""
	case 1:
		return array[0]
	}
	return strings.Join(array[:len(array)-1], glue) + finalGlue + array[len(array)-1]
}

// KeyBy keys the elements by what the callback reads off each one.
//
// The field is named with an accessor rather than a string. When two elements
// share a key the later one wins.
func KeyBy[T any](array []T, keyBy func(value T, key int) string) map[string]T {
	out := make(map[string]T, len(array))
	for i, v := range array {
		out[keyBy(v, i)] = v
	}
	return out
}

// Map returns the result of the callback for every element, in order. The
// callback is handed the element and its position.
func Map[T, U any](array []T, callback func(value T, key int) U) []U {
	out := make([]U, len(array))
	for i, v := range array {
		out[i] = callback(v, i)
	}
	return out
}

// MapWithKeys builds a map from the key and value the callback returns for
// every element. A repeated key keeps the last value.
func MapWithKeys[T, V any](array []T, callback func(value T, key int) (string, V)) map[string]V {
	out := make(map[string]V, len(array))
	for i, v := range array {
		key, value := callback(v, i)
		out[key] = value
	}
	return out
}

// MapSpread spreads each nested chunk across the callback's arguments.
//
// The position is appended to the chunk before it is spread, so the callback
// receives the chunk's elements and then that position. That is easy to miss
// reading the signature, and it is why this takes [][]any rather than a typed
// chunk: the trailing key is an int and the elements need not be.
func MapSpread[U any](array [][]any, callback func(values ...any) U) []U {
	out := make([]U, len(array))
	for i, chunk := range array {
		spread := make([]any, 0, len(chunk)+1)
		spread = append(spread, chunk...)
		spread = append(spread, i)
		out[i] = callback(spread...)
	}
	return out
}

// Prepend puts the value at the front of a fresh slice.
//
// There is no keyed form: a Go map has no order for a key written at the front
// to be visible in, so writing the entry is the whole of it.
func Prepend[T any](array []T, value T) []T {
	return append(append(make([]T, 0, len(array)+1), value), array...)
}

// Pluck reads one field off every element.
//
// With no key the result is a []any; with one it is a map[string]any keyed by
// that second field, which is why the return type is any. Both the value and
// the key are resolved with DataGet, so either may be a "dot" path.
func Pluck(array []any, value string, key ...string) any {
	if len(key) == 0 {
		out := make([]any, 0, len(array))
		for _, item := range array {
			resolved, _ := DataGet(item, value)
			out = append(out, resolved)
		}
		return out
	}
	out := make(map[string]any, len(array))
	for _, item := range array {
		resolved, _ := DataGet(item, value)
		itemKey, _ := DataGet(item, key[0])
		out[keyString(itemKey)] = resolved
	}
	return out
}

// Select reduces every element to the named keys.
//
// A key is read as an index into an indexable element, and failing that as an
// exported struct field of the same name. A key an element does not have is
// left out rather than filled with nil.
func Select(array []any, keys ...string) []map[string]any {
	out := make([]map[string]any, len(array))
	for i, item := range array {
		row := make(map[string]any, len(keys))
		for _, key := range keys {
			if value, ok := index(item, key); ok {
				row[key] = value
				continue
			}
			if value, ok := field(item, key); ok {
				row[key] = value
			}
		}
		out[i] = row
	}
	return out
}

// field reads an exported struct field by name, following a pointer, and
// reports whether there was one.
func field(item any, name string) (any, bool) {
	if item == nil {
		return nil, false
	}
	rv := reflect.ValueOf(item)
	for rv.Kind() == reflect.Ptr {
		if rv.IsNil() {
			return nil, false
		}
		rv = rv.Elem()
	}
	if rv.Kind() != reflect.Struct {
		return nil, false
	}
	found := rv.FieldByName(name)
	if !found.IsValid() || !found.CanInterface() {
		return nil, false
	}
	return found.Interface(), true
}

// Random returns number elements picked at random.
//
// It returns ErrInvalidArgument when more elements are asked for than there
// are. A number of zero or less gives an empty slice rather than an error.
//
// The draw is from crypto/rand: a shuffle seeded from the clock is a shuffle
// an attacker can replay.
func Random[T any](array []T, number int) ([]T, error) {
	if number > len(array) {
		return nil, fmt.Errorf("%w: you requested %d items, but there are only %d items available", ErrInvalidArgument, number, len(array))
	}
	if number <= 0 || len(array) == 0 {
		return []T{}, nil
	}
	return Shuffle(array)[:number], nil
}

// Shuffle returns the elements in a random order.
//
// The result is a new slice and the argument is untouched. When the draw fails
// the copy is returned as it stands.
func Shuffle[T any](array []T) []T {
	out := append(make([]T, 0, len(array)), array...)
	for i := len(out) - 1; i > 0; i-- {
		drawn, err := rand.Int(rand.Reader, big.NewInt(int64(i+1)))
		if err != nil {
			return out
		}
		j := int(drawn.Int64())
		out[i], out[j] = out[j], out[i]
	}
	return out
}

// Sole returns the one element passing the test. A nil callback takes the
// whole slice.
//
// It returns ErrItemNotFound when nothing matches, and a
// collections.MultipleItemsFoundError -- which carries the count and unwraps
// to ErrMultipleItemsFound -- when more than one does.
func Sole[T any](array []T, callback func(value T, key int) bool) (T, error) {
	matched := array
	if callback != nil {
		matched = Where(array, callback)
	}
	var zero T
	switch len(matched) {
	case 0:
		return zero, ErrItemNotFound
	case 1:
		return matched[0], nil
	default:
		return zero, &collections.MultipleItemsFoundError{Count: len(matched)}
	}
}

// Sort orders the elements by the value the callback reads off each one.
//
// The callback is a projection and not a comparator, and it is required: Go
// cannot compare an arbitrary T. The result is a new slice, renumbered from
// zero, and the sort is stable.
//
// A projection is taken rather than a comparator because a comparator over a
// slice is already slices.SortStableFunc, which sorts in place; what this adds
// is deriving the key once per element and leaving the argument untouched. An
// order a projection cannot express -- several keys, a ranking of its own --
// is a comparator, so reach for slices.SortStableFunc over a copy.
func Sort[T any, V cmp.Ordered](array []T, callback func(value T, key int) V) []T {
	type keyed struct {
		item T
		key  V
	}
	pairs := make([]keyed, len(array))
	for i, v := range array {
		pairs[i] = keyed{item: v, key: callback(v, i)}
	}
	sort.SliceStable(pairs, func(i, j int) bool { return cmp.Less(pairs[i].key, pairs[j].key) })
	out := make([]T, len(pairs))
	for i, pair := range pairs {
		out[i] = pair.item
	}
	return out
}

// SortDesc is Sort with the order reversed.
func SortDesc[T any, V cmp.Ordered](array []T, callback func(value T, key int) V) []T {
	sorted := Sort(array, callback)
	for i, j := 0, len(sorted)-1; i < j; i, j = i+1, j-1 {
		sorted[i], sorted[j] = sorted[j], sorted[i]
	}
	return sorted
}

// Where keeps the elements passing the test.
//
// The survivors close the gaps and are renumbered from zero.
func Where[T any](array []T, callback func(value T, key int) bool) []T {
	out := make([]T, 0, len(array))
	for i, v := range array {
		if callback(v, i) {
			out = append(out, v)
		}
	}
	return out
}

// Reject is Where with the test negated.
func Reject[T any](array []T, callback func(value T, key int) bool) []T {
	return Where(array, func(value T, key int) bool { return !callback(value, key) })
}

// Partition returns the elements passing the test, then the ones failing it.
//
// Neither half is nil, and the two together hold every element exactly once.
func Partition[T any](array []T, callback func(value T, key int) bool) (passed, failed []T) {
	passed = make([]T, 0, len(array))
	failed = make([]T, 0, len(array))
	for i, v := range array {
		if callback(v, i) {
			passed = append(passed, v)
			continue
		}
		failed = append(failed, v)
	}
	return passed, failed
}

// WhereNotNull keeps the elements that are not nil.
func WhereNotNull(array []any) []any {
	return Where(array, func(value any, _ int) bool { return value != nil })
}

// ExceptValues keeps everything but the values given.
//
// Comparison is by ==, and the survivors close the gaps and are renumbered
// from zero.
func ExceptValues[T comparable](array []T, values ...T) []T {
	unwanted := make(map[T]struct{}, len(values))
	for _, value := range values {
		unwanted[value] = struct{}{}
	}
	return Where(array, func(value T, _ int) bool {
		_, found := unwanted[value]
		return !found
	})
}

// OnlyValues keeps only the values given. It is the complement of
// ExceptValues.
func OnlyValues[T comparable](array []T, values ...T) []T {
	wanted := make(map[T]struct{}, len(values))
	for _, value := range values {
		wanted[value] = struct{}{}
	}
	return Where(array, func(value T, _ int) bool {
		_, found := wanted[value]
		return found
	})
}

// Wrap turns nil into an empty list, leaves a []any as it is, and puts
// anything else in a list on its own.
//
// A slice or array of another element type is converted rather than wrapped.
// A map is wrapped whole, because a []any cannot carry its keys, which at
// least keeps it reachable.
func Wrap(value any) []any {
	if value == nil {
		return []any{}
	}
	if list, ok := value.([]any); ok {
		return list
	}
	if rv, ok := container(value); ok && rv.Kind() != reflect.Map {
		out := make([]any, rv.Len())
		for i := range out {
			out[i] = rv.Index(i).Interface()
		}
		return out
	}
	return []any{value}
}
