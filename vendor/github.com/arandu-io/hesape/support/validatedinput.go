package support

import (
	"sort"

	"github.com/arandu-io/hesape/collections/arr"
)

// ValidatedInput carries the input that passed validation, and nothing else.
//
// The typed readers of the embedded data source -- String, Integer, Float,
// Boolean, Array, Date, Only, Except, Collect, Has, Missing -- read that
// input.
type ValidatedInput struct {
	dataSource
	input map[string]any
}

// NewValidatedInput builds a ValidatedInput over a copy of the given input.
func NewValidatedInput(input map[string]any) *ValidatedInput {
	v := &ValidatedInput{input: map[string]any{}}
	for key, held := range input {
		v.input[key] = held
	}
	v.dataSource = dataSource{
		allFn:  v.All,
		dataFn: func(key string, def any) any { return v.Input(key, def) },
	}
	return v
}

// Merge returns a new ValidatedInput carrying this input with the given items
// written over it. The receiver is left alone.
func (v *ValidatedInput) Merge(items map[string]any) *ValidatedInput {
	merged := v.All()
	for key, held := range items {
		merged[key] = held
	}
	return NewValidatedInput(merged)
}

// All returns the whole input when given no key, and the subset under the
// given dotted keys otherwise.
func (v *ValidatedInput) All(keys ...string) map[string]any {
	if len(keys) == 0 {
		out := make(map[string]any, len(v.input))
		for key, held := range v.input {
			out[key] = held
		}
		return out
	}
	return subsetByKeys(v.input, keys)
}

// Keys returns the top-level keys, sorted. A map has no order of its own, so
// sorted is the only order that can be promised.
func (v *ValidatedInput) Keys() []string {
	keys := make([]string, 0, len(v.input))
	for key := range v.input {
		keys = append(keys, key)
	}
	sort.Strings(keys)
	return keys
}

// Input returns one item by dotted key, falling back to the optional default,
// which is nil when not given. An empty key returns everything.
func (v *ValidatedInput) Input(key string, def ...any) any {
	all := v.All()
	if key == "" {
		return all
	}
	if held, ok := arr.Get(all, key); ok {
		return held
	}
	return firstOr(def, nil)
}

// ToArray returns a copy of the whole input.
func (v *ValidatedInput) ToArray() map[string]any { return v.All() }

// Dump writes the input to standard error through [Dump] and returns the
// ValidatedInput. Given keys, only those are written.
func (v *ValidatedInput) Dump(keys ...string) *ValidatedInput {
	if len(keys) > 0 {
		Dump(v.Only(keys...))
		return v
	}
	Dump(v.All())
	return v
}

// Dd dumps, then ends the process with status 1.
func (v *ValidatedInput) Dd(keys ...string) {
	v.Dump(keys...)
	exit(1)
}
