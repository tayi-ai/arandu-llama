package support

import "reflect"

// Optional wraps a value and hands back nil for anything that value does not
// have, so a caller reading into it does not have to check at every step.
//
// A nil Optional, and one wrapping nil, read the same: every lookup is nil.
type Optional struct {
	value any
}

// NewOptional wraps a value, which may be nil.
func NewOptional(value any) *Optional { return &Optional{value: value} }

// Value returns the value being wrapped.
func (o *Optional) Value() any { return o.value }

// Get returns the member of the value underneath named by key, or nil. A map
// is read by key and a struct by exported field name, with pointers followed;
// anything else is nil.
func (o *Optional) Get(key string) any {
	if o == nil || o.value == nil {
		return nil
	}
	if m, ok := o.value.(map[string]any); ok {
		return m[key]
	}
	rv := reflect.ValueOf(o.value)
	for rv.Kind() == reflect.Pointer {
		if rv.IsNil() {
			return nil
		}
		rv = rv.Elem()
	}
	switch rv.Kind() {
	case reflect.Struct:
		field := rv.FieldByName(key)
		if !field.IsValid() || !field.CanInterface() {
			return nil
		}
		return field.Interface()
	case reflect.Map:
		v := rv.MapIndex(reflect.ValueOf(key))
		if !v.IsValid() {
			return nil
		}
		return v.Interface()
	default:
		return nil
	}
}

// IsSet reports whether [Optional.Get] finds anything non-nil under the key.
func (o *Optional) IsSet(key string) bool { return o.Get(key) != nil }

// OffsetExists reports whether the value underneath is a map[string]any that
// holds the key.
func (o *Optional) OffsetExists(key string) bool {
	if o == nil || o.value == nil {
		return false
	}
	m, ok := o.value.(map[string]any)
	if !ok {
		return false
	}
	_, exists := m[key]
	return exists
}

// OffsetGet returns the value under the key, the same as [Optional.Get].
func (o *Optional) OffsetGet(key string) any { return o.Get(key) }

// OffsetSet writes under the key, but only when the value underneath is a
// map[string]any. Any other value drops the write.
func (o *Optional) OffsetSet(key string, v any) {
	if m, ok := o.value.(map[string]any); ok {
		m[key] = v
	}
}

// OffsetUnset deletes the key, but only when the value underneath is a
// map[string]any. Any other value drops the call.
func (o *Optional) OffsetUnset(key string) {
	if m, ok := o.value.(map[string]any); ok {
		delete(m, key)
	}
}
