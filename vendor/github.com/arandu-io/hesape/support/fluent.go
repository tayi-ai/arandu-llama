package support

import (
	"encoding/json"
	"strconv"

	"github.com/arandu-io/hesape/collections/arr"
)

// Fluent is a bag of attributes read and written by dotted key, with the typed
// readers of the embedded data source on top -- String, Integer, Float,
// Boolean, Array, Date, Collect and the rest.
type Fluent struct {
	dataSource
	attributes map[string]any
}

// NewFluent builds a Fluent over a copy of the given attributes, each written
// at the top level. A nil map gives an empty bag.
func NewFluent(attributes map[string]any) *Fluent {
	f := &Fluent{attributes: map[string]any{}}
	f.dataSource = dataSource{
		allFn:  f.All,
		dataFn: func(key string, def any) any { return f.Get(key, def) },
	}
	f.Fill(attributes)
	return f
}

// Make builds a Fluent over the given attributes, the same as [NewFluent].
func Make(attributes map[string]any) *Fluent { return NewFluent(attributes) }

// Get returns the value under a dotted key, falling back to the optional
// default, which is nil when not given. An empty key returns every attribute,
// and the default is not consulted.
func (f *Fluent) Get(key string, def ...any) any {
	if key == "" {
		return f.attributes
	}
	if held, ok := arr.Get(f.attributes, key); ok {
		return held
	}
	return firstOr(def, nil)
}

// Set writes a value under a dotted key, creating the levels that are missing,
// and returns the Fluent.
func (f *Fluent) Set(key string, v any) *Fluent {
	if f.attributes == nil {
		f.attributes = map[string]any{}
	}
	arr.Set(f.attributes, key, v)
	return f
}

// Fill writes every pair at the top level, with no dot reading, and returns
// the Fluent.
func (f *Fluent) Fill(attributes map[string]any) *Fluent {
	if f.attributes == nil {
		f.attributes = map[string]any{}
	}
	for key, v := range attributes {
		f.attributes[key] = v
	}
	return f
}

// Value returns the attribute under the exact key, with no dot reading,
// falling back to the optional default. A default that is a func() any is
// invoked and its result returned.
func (f *Fluent) Value(key string, def ...any) any {
	if v, ok := f.attributes[key]; ok {
		return v
	}
	return value(firstOr(def, nil))
}

// Scope returns the value under the key as a Fluent of its own. A map becomes
// its attributes, a slice is keyed by its decimal index, nil gives an empty
// bag, and anything else becomes a one-attribute bag under the key "0".
func (f *Fluent) Scope(key string, def ...any) *Fluent {
	switch v := f.Get(key, firstOr(def, nil)).(type) {
	case nil:
		return NewFluent(nil)
	case map[string]any:
		return NewFluent(v)
	case []any:
		attributes := make(map[string]any, len(v))
		for i, item := range v {
			attributes[strconv.Itoa(i)] = item
		}
		return NewFluent(attributes)
	default:
		return NewFluent(map[string]any{"0": v})
	}
}

// All returns every attribute when given no key, and the subset under the
// given dotted keys otherwise. A key the bag does not hold comes back as nil.
func (f *Fluent) All(keys ...string) map[string]any {
	if len(keys) == 0 {
		return f.ToArray()
	}
	return subsetByKeys(f.attributes, keys)
}

// GetAttributes returns a copy of every attribute.
func (f *Fluent) GetAttributes() map[string]any { return f.ToArray() }

// ToArray returns a copy of the attributes, so the caller cannot write through
// it into the bag.
func (f *Fluent) ToArray() map[string]any {
	out := make(map[string]any, len(f.attributes))
	for k, v := range f.attributes {
		out[k] = v
	}
	return out
}

// ToJson encodes the attributes as JSON, or returns the error encoding raised.
func (f *Fluent) ToJson() (string, error) {
	raw, err := json.Marshal(f.attributes)
	if err != nil {
		return "", err
	}
	return string(raw), nil
}

// MarshalJSON encodes the attributes, so a Fluent nests inside another encoded
// value. An empty bag encodes as {}.
func (f *Fluent) MarshalJSON() ([]byte, error) {
	if f.attributes == nil {
		return []byte("{}"), nil
	}
	return json.Marshal(f.attributes)
}
