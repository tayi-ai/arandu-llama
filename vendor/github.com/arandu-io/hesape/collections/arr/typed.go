package arr

import (
	"fmt"
	"reflect"
)

// The five readers that assert a type -- Array, Boolean, Float, Integer and
// String -- each read a "dot" path through Get and return ErrInvalidArgument
// when what they find is of another type.
//
// Each takes an optional default as a variadic; only the first is used. With
// no default and no value at the key, the read fails rather than yielding the
// zero value.

// Array returns the list at the "dot" path.
//
// A map at that path is not a list and reports ErrInvalidArgument, since a
// []any cannot carry keys; Get returns the map itself.
func Array(array any, key string, def ...[]any) ([]any, error) {
	value, ok := Get(array, key)
	if !ok {
		if len(def) > 0 {
			return def[0], nil
		}
		return nil, typeError(key, "an array", nil)
	}
	if rv, ok := container(value); ok && rv.Kind() != reflect.Map {
		out := make([]any, rv.Len())
		for i := range out {
			out[i] = rv.Index(i).Interface()
		}
		return out, nil
	}
	return nil, typeError(key, "an array", value)
}

// Boolean returns the bool at the "dot" path.
func Boolean(array any, key string, def ...bool) (bool, error) {
	value, ok := Get(array, key)
	if !ok {
		if len(def) > 0 {
			return def[0], nil
		}
		return false, typeError(key, "a boolean", nil)
	}
	if b, ok := value.(bool); ok {
		return b, nil
	}
	return false, typeError(key, "a boolean", value)
}

// Float returns the float at the "dot" path.
//
// A value stored as an int is not a float and fails. Both float widths are
// accepted and returned as a float64.
func Float(array any, key string, def ...float64) (float64, error) {
	value, ok := Get(array, key)
	if !ok {
		if len(def) > 0 {
			return def[0], nil
		}
		return 0, typeError(key, "a float", nil)
	}
	switch number := value.(type) {
	case float64:
		return number, nil
	case float32:
		return float64(number), nil
	}
	return 0, typeError(key, "a float", value)
}

// Integer returns the int at the "dot" path.
//
// A value stored as a float is not an integer and fails. Every signed and
// unsigned integer width is accepted and returned as an int.
func Integer(array any, key string, def ...int) (int, error) {
	value, ok := Get(array, key)
	if !ok {
		if len(def) > 0 {
			return def[0], nil
		}
		return 0, typeError(key, "an integer", nil)
	}
	rv := reflect.ValueOf(value)
	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return int(rv.Int()), nil
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return int(rv.Uint()), nil
	}
	return 0, typeError(key, "an integer", value)
}

// String returns the string at the "dot" path.
func String(array any, key string, def ...string) (string, error) {
	value, ok := Get(array, key)
	if !ok {
		if len(def) > 0 {
			return def[0], nil
		}
		return "", typeError(key, "a string", nil)
	}
	if s, ok := value.(string); ok {
		return s, nil
	}
	return "", typeError(key, "a string", value)
}

// typeError builds the message the typed readers fail with: the key, the type
// that was wanted and the type that was found. An absent value is named nil.
func typeError(key, want string, got any) error {
	if got == nil {
		return fmt.Errorf("%w: array value for key [%s] must be %s, nil found", ErrInvalidArgument, key, want)
	}
	return fmt.Errorf("%w: array value for key [%s] must be %s, %T found", ErrInvalidArgument, key, want, got)
}
