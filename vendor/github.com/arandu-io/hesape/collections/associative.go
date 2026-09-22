package collections

import (
	"cmp"
	"slices"
)

// This file holds the operations that only mean something when the keys are
// keys and not positions. A Collection[T] is a list, so they take a map[K]V
// instead.
//
// A Go map has no order, so an operation whose result depends on key order
// returns a Collection[V], which does have one.

// sortedKeys returns the keys of m in ascending order. Go randomises map
// iteration on purpose, and every function here that produces a list has to be
// reproducible, so nothing in this file walks a map without going through it.
func sortedKeys[K cmp.Ordered, V any](m map[K]V) []K {
	keys := make([]K, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	slices.Sort(keys)
	return keys
}

// DiffAssoc keeps the entries whose key and value together are absent from
// items.
//
// An entry survives when items has no such key, or has it with another value.
// The result is empty rather than nil.
func DiffAssoc[K comparable, V comparable](array, items map[K]V) map[K]V {
	out := make(map[K]V, len(array))
	for k, v := range array {
		if other, ok := items[k]; ok && other == v {
			continue
		}
		out[k] = v
	}
	return out
}

// DiffAssocUsing is DiffAssoc with the keys compared by the callback rather
// than by ==.
//
// The callback compares keys only; values are still compared by ==. It reports
// zero when two keys are the same key.
func DiffAssocUsing[K comparable, V comparable](array, items map[K]V, compare func(a, b K) int) map[K]V {
	out := make(map[K]V, len(array))
	for k, v := range array {
		matched := false
		for otherKey, otherValue := range items {
			if compare(k, otherKey) == 0 && otherValue == v {
				matched = true
				break
			}
		}
		if !matched {
			out[k] = v
		}
	}
	return out
}

// DiffKeys keeps the entries whose key is absent from items, whatever the
// values on either side are.
//
// items may hold a different value type, for exactly that reason: only its
// keys are read.
func DiffKeys[K comparable, V, W any](array map[K]V, items map[K]W) map[K]V {
	out := make(map[K]V, len(array))
	for k, v := range array {
		if _, ok := items[k]; ok {
			continue
		}
		out[k] = v
	}
	return out
}

// DiffKeysUsing is DiffKeys with the keys compared by the callback rather than
// by ==.
func DiffKeysUsing[K comparable, V, W any](array map[K]V, items map[K]W, compare func(a, b K) int) map[K]V {
	out := make(map[K]V, len(array))
	for k, v := range array {
		matched := false
		for otherKey := range items {
			if compare(k, otherKey) == 0 {
				matched = true
				break
			}
		}
		if !matched {
			out[k] = v
		}
	}
	return out
}

// IntersectAssoc keeps the entries items has under the same key with the same
// value.
func IntersectAssoc[K comparable, V comparable](array, items map[K]V) map[K]V {
	out := make(map[K]V, len(array))
	for k, v := range array {
		if other, ok := items[k]; ok && other == v {
			out[k] = v
		}
	}
	return out
}

// IntersectAssocUsing is IntersectAssoc with the keys compared by the callback
// rather than by ==.
func IntersectAssocUsing[K comparable, V comparable](array, items map[K]V, compare func(a, b K) int) map[K]V {
	out := make(map[K]V, len(array))
	for k, v := range array {
		for otherKey, otherValue := range items {
			if compare(k, otherKey) == 0 && otherValue == v {
				out[k] = v
				break
			}
		}
	}
	return out
}

// IntersectByKeys keeps the entries whose key items also has.
//
// The values kept are this map's, never items'.
func IntersectByKeys[K comparable, V, W any](array map[K]V, items map[K]W) map[K]V {
	out := make(map[K]V, len(array))
	for k, v := range array {
		if _, ok := items[k]; ok {
			out[k] = v
		}
	}
	return out
}

// SortKeys returns the values of the map ascending by key.
//
// A Go map cannot carry an order, so the ordering is the result rather than a
// property of it. Keys over the same map gives the keys in the matching order.
func SortKeys[K cmp.Ordered, V any](array map[K]V) Collection[V] {
	keys := sortedKeys(array)
	out := make(Collection[V], len(keys))
	for i, k := range keys {
		out[i] = array[k]
	}
	return out
}

// SortKeysDesc is SortKeys with the order reversed.
func SortKeysDesc[K cmp.Ordered, V any](array map[K]V) Collection[V] {
	return SortKeys(array).Reverse()
}

// SortKeysUsing returns the values ordered by the callback applied to their
// keys.
//
// The keys are put in ascending order before the callback sorts them, so that
// keys the callback calls equal come out in a fixed order instead of the
// random one Go gives map iteration.
func SortKeysUsing[K cmp.Ordered, V any](array map[K]V, compare func(a, b K) int) Collection[V] {
	keys := sortedKeys(array)
	stableSort(keys, compare)
	out := make(Collection[V], len(keys))
	for i, k := range keys {
		out[i] = array[k]
	}
	return out
}

// Keys returns the keys of the map, where the method of the same name on
// Collection[T] returns the positions of a list.
//
// They come out ascending, which is the order SortKeys puts their values in,
// so the two walked together pair each value with the key it came from.
func Keys[K cmp.Ordered, V any](array map[K]V) Collection[K] {
	return Collection[K](sortedKeys(array))
}

// MergeRecursive merges items into array, descending wherever both sides hold
// a map under a key.
//
// Where a key is on both sides and either side is not a map, the two values
// become a []any holding both -- a value already a []any contributes its
// elements rather than itself, so a list grows across repeated merges.
func MergeRecursive(array, items map[string]any) map[string]any {
	out := make(map[string]any, len(array)+len(items))
	for k, v := range array {
		out[k] = v
	}
	for k, v := range items {
		existing, ok := out[k]
		if !ok {
			out[k] = v
			continue
		}
		leftMap, leftIsMap := existing.(map[string]any)
		rightMap, rightIsMap := v.(map[string]any)
		if leftIsMap && rightIsMap {
			out[k] = MergeRecursive(leftMap, rightMap)
			continue
		}
		out[k] = append(asList(existing), asList(v)...)
	}
	return out
}

// asList wraps a value in a []any unless it already is one, so that two values
// found under the same key can be concatenated.
func asList(value any) []any {
	if list, ok := value.([]any); ok {
		return list
	}
	return []any{value}
}

// ReplaceRecursive lets items win at every key, except where both sides hold a
// map or both hold a []any: those descend.
//
// Replacing {"a":[1,2,3]} with {"a":[9]} therefore gives {"a":[9,2,3]} and not
// {"a":[9]}.
func ReplaceRecursive(array, items map[string]any) map[string]any {
	out := make(map[string]any, len(array)+len(items))
	for k, v := range array {
		out[k] = v
	}
	for k, v := range items {
		existing, ok := out[k]
		if !ok {
			out[k] = v
			continue
		}
		out[k] = replaceRecursiveValue(existing, v)
	}
	return out
}

func replaceRecursiveValue(existing, replacement any) any {
	if leftMap, ok := existing.(map[string]any); ok {
		if rightMap, ok := replacement.(map[string]any); ok {
			return ReplaceRecursive(leftMap, rightMap)
		}
		return replacement
	}
	leftList, leftIsList := existing.([]any)
	rightList, rightIsList := replacement.([]any)
	if !leftIsList || !rightIsList {
		return replacement
	}
	out := make([]any, len(leftList))
	copy(out, leftList)
	for i, v := range rightList {
		if i < len(out) {
			out[i] = replaceRecursiveValue(out[i], v)
			continue
		}
		out = append(out, v)
	}
	return out
}

// CollapseWithKeys flattens the maps in the collection into one, keeping their
// keys.
//
// A key held by more than one element ends up with the value of the last.
func CollapseWithKeys[K comparable, V any](c Collection[map[K]V]) map[K]V {
	out := make(map[K]V)
	for _, item := range c {
		for k, v := range item {
			out[k] = v
		}
	}
	return out
}
