package config

import (
	"fmt"
	"math"
	"strconv"
	"strings"
	"sync"

	"github.com/arandu-io/hesape/collections"
)

// Repository is the dotted-key configuration store: a nested tree of maps and
// lists read and written by keys such as "app.name".
//
// It sits beside [App] rather than replacing it. [App] is what the framework
// reads -- a wrong field is a compile error there -- and this is what an
// application reads when the key is only known at runtime. A first-party
// package that reaches for a string key instead of a struct field is doing it
// wrong; that is the only rule about which of the two to use.
//
// Every method is safe for concurrent use.
//
// Values are copied in and out. [Repository.Get], [Repository.All] and the
// constructor deep-copy any map[string]any and []any they cross.
type Repository struct {
	mu    sync.RWMutex
	items map[string]any
}

// NewRepository creates a configuration repository over the given items.
//
// The items are deep-copied, so the caller's map is not the repository's map.
func NewRepository(items map[string]any) *Repository {
	copied, _ := clone(items).(map[string]any)
	if copied == nil {
		copied = map[string]any{}
	}
	return &Repository{items: copied}
}

// Has reports whether the given configuration value exists.
//
// A key present with a nil value exists -- the question is about the key, not
// the value. An empty repository answers false to every key, including the
// empty one.
func (r *Repository) Has(key string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if len(r.items) == 0 {
		return false
	}
	_, ok := lookup(r.items, key)
	return ok
}

// Get returns the specified configuration value, or the optional default when
// the key is missing.
//
// A default of func() any is called rather than returned. Only the first
// default is read; the rest are ignored.
func (r *Repository) Get(key string, def ...any) any {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if v, ok := lookup(r.items, key); ok {
		return clone(v)
	}
	return value(first(def))
}

// GetMany returns many configuration values at once.
//
// Each entry maps a key to the default used when that key is missing. A nil
// default is called if it is a func() any, as everywhere else.
//
// The result is never nil, and it holds one entry per requested key.
func (r *Repository) GetMany(keys map[string]any) map[string]any {
	r.mu.RLock()
	defer r.mu.RUnlock()

	out := make(map[string]any, len(keys))
	for key, def := range keys {
		if v, ok := lookup(r.items, key); ok {
			out[key] = clone(v)
			continue
		}
		out[key] = value(def)
	}
	return out
}

// String returns the specified string configuration value.
//
// The check is on the type and not a conversion: an int is an error, not "42".
func (r *Repository) String(key string, def ...any) (string, error) {
	v := r.Get(key, def...)
	s, ok := v.(string)
	if !ok {
		return "", typeError(key, "a string", v)
	}
	return s, nil
}

// Integer returns the specified integer configuration value. Every signed and
// unsigned integer kind is accepted.
//
// A value too large for int is an error rather than a truncation.
func (r *Repository) Integer(key string, def ...any) (int, error) {
	v := r.Get(key, def...)
	switch t := v.(type) {
	case int:
		return t, nil
	case int8:
		return int(t), nil
	case int16:
		return int(t), nil
	case int32:
		return int(t), nil
	case int64:
		if t < math.MinInt || t > math.MaxInt {
			return 0, rangeError(key, v)
		}
		return int(t), nil
	case uint:
		if uint64(t) > math.MaxInt {
			return 0, rangeError(key, v)
		}
		return int(t), nil
	case uint8:
		return int(t), nil
	case uint16:
		return int(t), nil
	case uint32:
		if uint64(t) > math.MaxInt {
			return 0, rangeError(key, v)
		}
		return int(t), nil
	case uint64:
		if t > math.MaxInt {
			return 0, rangeError(key, v)
		}
		return int(t), nil
	default:
		return 0, typeError(key, "an integer", v)
	}
}

// Float returns the specified float configuration value.
//
// An integer is not a float: config values are written by hand, and 0 where
// 0.0 was meant is worth a message.
func (r *Repository) Float(key string, def ...any) (float64, error) {
	v := r.Get(key, def...)
	switch t := v.(type) {
	case float64:
		return t, nil
	case float32:
		return float64(t), nil
	default:
		return 0, typeError(key, "a float", v)
	}
}

// Boolean returns the specified boolean configuration value.
//
// It is a type check and not a cast, so "true" and 1 are errors.
func (r *Repository) Boolean(key string, def ...any) (bool, error) {
	v := r.Get(key, def...)
	b, ok := v.(bool)
	if !ok {
		return false, typeError(key, "a boolean", v)
	}
	return b, nil
}

// Array returns the specified array configuration value.
//
// This is the list shape only. A section with string keys is a map[string]any
// and is read with [Repository.Get] or by its sub-keys; asking for it here is
// an error that says so, rather than a silent conversion that would drop the
// keys.
func (r *Repository) Array(key string, def ...any) ([]any, error) {
	v := r.Get(key, def...)
	l, ok := v.([]any)
	if !ok {
		if _, assoc := v.(map[string]any); assoc {
			return nil, fmt.Errorf("configuration value for key [%s] is an array with string keys; read it with Get or by its sub-keys, because a Go slice has no string keys", key)
		}
		return nil, typeError(key, "an array", v)
	}
	return l, nil
}

// Collection returns the specified array configuration value as a collection.
//
// It is [Repository.Array] plus the wrap.
//
// collections.Collection[any] has []any as its underlying type, so a caller
// that only ranges over the result reads it as a list; a caller that wants
// Map, Filter or First has them without converting.
func (r *Repository) Collection(key string, def ...any) (collections.Collection[any], error) {
	list, err := r.Array(key, def...)
	if err != nil {
		return nil, err
	}
	return collections.Collection[any](list), nil
}

// Set sets a given configuration value, creating the intermediate levels the
// dotted key names.
//
// A level that exists but is not a map is replaced by one. Setting several
// keys is one call per key.
func (r *Repository) Set(key string, val any) {
	r.mu.Lock()
	defer r.mu.Unlock()

	assign(r.items, key, clone(val))
}

// Prepend prepends a value onto an array configuration value. A missing key
// starts an empty list.
//
// A key that holds something other than a list is the error.
func (r *Repository) Prepend(key string, val any) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	list, err := listAt(r.items, key)
	if err != nil {
		return err
	}
	assign(r.items, key, append([]any{clone(val)}, list...))
	return nil
}

// Push pushes a value onto an array configuration value.
//
// It is [Repository.Prepend] at the other end, with the same rule about a key
// that is not a list.
func (r *Repository) Push(key string, val any) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	list, err := listAt(r.items, key)
	if err != nil {
		return err
	}
	assign(r.items, key, append(list, clone(val)))
	return nil
}

// All returns all of the configuration items for the application.
//
// The result is a deep copy.
func (r *Repository) All() map[string]any {
	r.mu.RLock()
	defer r.mu.RUnlock()

	all, _ := clone(r.items).(map[string]any)
	return all
}

// lookup resolves a dotted key against items and reports whether it was found.
// Reading and testing for existence are the same walk, and two copies of it
// would be two answers to "does app.name exist".
func lookup(items map[string]any, key string) (any, bool) {
	if v, ok := items[key]; ok {
		return v, true
	}
	if !strings.Contains(key, ".") {
		return nil, false
	}

	var current any = items
	for _, segment := range strings.Split(key, ".") {
		next, ok := index(current, segment)
		if !ok {
			return nil, false
		}
		current = next
	}
	return current, true
}

// index reads one segment out of one level. A map is read by key and a list by
// position.
func index(container any, segment string) (any, bool) {
	switch t := container.(type) {
	case map[string]any:
		v, ok := t[segment]
		return v, ok
	case []any:
		i, err := strconv.Atoi(segment)
		if err != nil || i < 0 || i >= len(t) {
			return nil, false
		}
		return t[i], true
	default:
		return nil, false
	}
}

// assign writes val at a dotted key, creating the intermediate maps it names.
func assign(items map[string]any, key string, val any) {
	segments := strings.Split(key, ".")
	level := items
	for _, segment := range segments[:len(segments)-1] {
		next, ok := level[segment].(map[string]any)
		if !ok {
			next = map[string]any{}
			level[segment] = next
		}
		level = next
	}
	level[segments[len(segments)-1]] = val
}

// listAt returns the list stored at key for Prepend and Push, or the error they
// share. The result is a fresh slice: appending to the stored one could write
// into an array a caller still holds.
func listAt(items map[string]any, key string) ([]any, error) {
	v, ok := lookup(items, key)
	if !ok {
		return []any{}, nil
	}
	l, ok := v.([]any)
	if !ok {
		return nil, typeError(key, "an array", v)
	}
	return append(make([]any, 0, len(l)+1), l...), nil
}

// first returns the single optional default.
func first(def []any) any {
	if len(def) == 0 {
		return nil
	}
	return def[0]
}

// value resolves a default: one that is a func() any is the result of calling
// it, and anything else is itself.
func value(v any) any {
	if f, ok := v.(func() any); ok {
		return f()
	}
	return v
}

// clone deep-copies the containers a configuration tree is made of and returns
// anything else as it stands. Scalars are values in Go already; maps and slices
// are not, and handing out a live one is how a read turns into a write.
func clone(v any) any {
	switch t := v.(type) {
	case map[string]any:
		out := make(map[string]any, len(t))
		for k, item := range t {
			out[k] = clone(item)
		}
		return out
	case []any:
		out := make([]any, len(t))
		for i, item := range t {
			out[i] = clone(item)
		}
		return out
	default:
		return v
	}
}

// typeError is the error the typed readers return when the stored value is not
// the type asked for. It names the key, the wanted type and the type found.
func typeError(key, want string, v any) error {
	return fmt.Errorf("configuration value for key [%s] must be %s, %s given", key, want, phpType(v))
}

// rangeError reports a value that is an integer but does not fit in an int.
func rangeError(key string, v any) error {
	return fmt.Errorf("configuration value for key [%s] is out of range for int: %v", key, v)
}

// phpType names the type of a value for the messages [typeError] builds.
func phpType(v any) string {
	switch v.(type) {
	case nil:
		return "NULL"
	case bool:
		return "boolean"
	case string:
		return "string"
	case float32, float64:
		return "double"
	case int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, uintptr:
		return "integer"
	case []any, map[string]any:
		return "array"
	default:
		return "object"
	}
}
