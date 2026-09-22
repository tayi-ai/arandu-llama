package support

import (
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"time"

	"github.com/arandu-io/hesape/collections/arr"
)

// dataSource is the typed reader embedded in [Fluent], [UriQueryString] and
// [ValidatedInput], which promotes every method below onto them.
//
// The two accessors are function fields rather than methods to override: the
// embedding type fills them in at construction, and they are what every read
// here goes through.
type dataSource struct {
	allFn  func(keys ...string) map[string]any
	dataFn func(key string, def any) any
}

// Data returns one value out of the source, by dotted key, falling back to the
// default. It is exported because [Enum] and [Enums] are functions and have to
// reach it from outside the type.
func (d dataSource) Data(key string, def any) any {
	if d.dataFn == nil {
		return value(def)
	}
	return d.dataFn(key, def)
}

func (d dataSource) allData(keys ...string) map[string]any {
	if d.allFn == nil {
		return map[string]any{}
	}
	return d.allFn(keys...)
}

// Exists is [dataSource.Has] under a second name.
func (d dataSource) Exists(keys ...string) bool { return d.Has(keys...) }

// Has reports whether every one of the dotted keys is present. With no key it
// is false.
func (d dataSource) Has(keys ...string) bool {
	if len(keys) == 0 {
		return false
	}
	data := d.allData()
	for _, key := range keys {
		if !arr.Has(data, key) {
			return false
		}
	}
	return true
}

// HasAny reports whether at least one of the dotted keys is present.
func (d dataSource) HasAny(keys ...string) bool {
	return arr.HasAny(d.allData(), keys...)
}

// Missing reports whether any of the dotted keys is absent.
func (d dataSource) Missing(keys ...string) bool { return !d.Has(keys...) }

// Filled reports whether every key holds something that is not an empty
// string. A bool, a list and a map count as filled even when empty.
func (d dataSource) Filled(keys ...string) bool {
	for _, key := range keys {
		if d.isEmptyString(key) {
			return false
		}
	}
	return true
}

// IsNotFilled reports whether every one of the keys is empty.
func (d dataSource) IsNotFilled(keys ...string) bool {
	for _, key := range keys {
		if !d.isEmptyString(key) {
			return false
		}
	}
	return true
}

// AnyFilled reports whether at least one of the keys is filled.
func (d dataSource) AnyFilled(keys ...string) bool {
	for _, key := range keys {
		if d.Filled(key) {
			return true
		}
	}
	return false
}

func (d dataSource) isEmptyString(key string) bool {
	v := d.Data(key, nil)
	switch v.(type) {
	case bool:
		return false
	case []any, map[string]any:
		return false
	}
	return strings.TrimSpace(toString(v)) == ""
}

// WhenHas runs the callback with the value when the key is present, and the
// optional fallback when it is not.
//
// It returns nothing: an embedded struct cannot reach the type that embeds it,
// so there is no receiver to hand back for chaining.
func (d dataSource) WhenHas(key string, callback func(value any), def ...func()) {
	if d.Has(key) {
		held, _ := arr.Get(d.allData(), key)
		callback(held)
		return
	}
	if len(def) > 0 && def[0] != nil {
		def[0]()
	}
}

// WhenFilled runs the callback with the value when the key is filled, and the
// optional fallback when it is not. It returns nothing for the reason given on
// [dataSource.WhenHas].
func (d dataSource) WhenFilled(key string, callback func(value any), def ...func()) {
	if d.Filled(key) {
		held, _ := arr.Get(d.allData(), key)
		callback(held)
		return
	}
	if len(def) > 0 && def[0] != nil {
		def[0]()
	}
}

// WhenMissing runs the callback when the key is absent, and the optional
// fallback when it is present. It returns nothing for the reason given on
// [dataSource.WhenHas].
func (d dataSource) WhenMissing(key string, callback func(value any), def ...func()) {
	if d.Missing(key) {
		held, _ := arr.Get(d.allData(), key)
		callback(held)
		return
	}
	if len(def) > 0 && def[0] != nil {
		def[0]()
	}
}

// Str is [dataSource.String] under a second name.
func (d dataSource) Str(key string, def ...any) string { return d.String(key, def...) }

// String returns the value at the key as a string, falling back to the
// optional default. It is a plain string and not a richer wrapper, so this
// package carries no dependency on the string package.
func (d dataSource) String(key string, def ...any) string {
	return toString(d.Data(key, firstOr(def, nil)))
}

// Boolean returns the value at the key as a bool: "1", "true", "on" and "yes"
// are true, in any case, and everything else is false. The optional default is
// used when the key is absent, and is false when not given.
func (d dataSource) Boolean(key string, def ...bool) bool {
	var fallback any
	if len(def) > 0 {
		fallback = def[0]
	} else {
		fallback = false
	}
	return toBool(d.Data(key, fallback))
}

// Integer returns the value at the key as an int, reading the leading numeric
// run and truncating it, or zero when there is none. The optional default is
// used when the key is absent, and is zero when not given.
func (d dataSource) Integer(key string, def ...int) int {
	var fallback any
	if len(def) > 0 {
		fallback = def[0]
	} else {
		fallback = 0
	}
	return toInt(d.Data(key, fallback))
}

// Float returns the value at the key as a float64, reading the leading numeric
// run, or zero when there is none. The optional default is used when the key
// is absent, and is zero when not given.
func (d dataSource) Float(key string, def ...float64) float64 {
	var fallback any
	if len(def) > 0 {
		fallback = def[0]
	} else {
		fallback = 0.0
	}
	return toFloat(d.Data(key, fallback))
}

// Date returns the value at the key as a time.Time. The variadic argument is
// the layout and then the location name, in that order; with no layout the
// value is read by [Parse].
//
// A key that is not filled gives the zero time and no error. A value that
// cannot be read, and a location name that cannot be loaded, are errors.
func (d dataSource) Date(key string, formatAndLocation ...string) (time.Time, error) {
	if d.IsNotFilled(key) {
		return time.Time{}, nil
	}
	raw := toString(d.Data(key, nil))
	format := firstOr(formatAndLocation, "")
	loc := time.Local
	if len(formatAndLocation) > 1 && formatAndLocation[1] != "" {
		parsed, err := time.LoadLocation(formatAndLocation[1])
		if err != nil {
			return time.Time{}, err
		}
		loc = parsed
	}
	if format == "" {
		parsed, err := Parse(raw)
		if err != nil {
			return time.Time{}, err
		}
		return parsed.In(loc), nil
	}
	return time.ParseInLocation(format, raw, loc)
}

// Array returns the value at the key as a list, wrapping a value that is not
// one. To read several keys at once, use [dataSource.Only]: the two shapes
// cannot share a return type.
func (d dataSource) Array(key string) []any {
	return arr.Wrap(d.Data(key, nil))
}

// Collect is [dataSource.Array] under a second name.
func (d dataSource) Collect(key string) []any { return d.Array(key) }

// Only returns the subset under the given dotted keys, leaving out the keys
// that are absent.
func (d dataSource) Only(keys ...string) map[string]any {
	results := map[string]any{}
	data := d.allData()
	for _, key := range keys {
		if held, ok := arr.Get(data, key); ok {
			arr.Set(results, key, held)
		}
	}
	return results
}

// Except returns everything but the given dotted keys. The data is copied
// first, so the source is left alone.
func (d dataSource) Except(keys ...string) map[string]any {
	results := arr.Except(d.allData())
	arr.Forget(results, keys...)
	return results
}

// Enum returns the value at the key as one of the given cases, or the zero
// value and false when it is not one of them.
//
// The cases are an argument because there is no way to list the values of a
// named string or int type, and it is a function rather than a method because
// a method cannot take a type parameter.
func Enum[T ~string | ~int](d dataSource, key string, cases []T) (T, bool) {
	var zero T
	if d.IsNotFilled(key) {
		return zero, false
	}
	raw := d.Data(key, nil)
	for _, c := range cases {
		if matchesEnumCase(c, raw) {
			return c, true
		}
	}
	return zero, false
}

// Enums returns every value under the key that is one of the given cases. See
// [Enum] on why the cases are an argument.
func Enums[T ~string | ~int](d dataSource, key string, cases []T) []T {
	results := []T{}
	if d.IsNotFilled(key) {
		return results
	}
	for _, raw := range d.Array(key) {
		for _, c := range cases {
			if matchesEnumCase(c, raw) {
				results = append(results, c)
				break
			}
		}
	}
	return results
}

// matchesEnumCase compares a raw value against a case, string against string
// and int against int. The kind is read with reflect because a named type over
// string does not satisfy a type switch on string.
func matchesEnumCase[T ~string | ~int](c T, raw any) bool {
	rv := reflect.ValueOf(c)
	if rv.Kind() == reflect.String {
		return toString(raw) == rv.String()
	}
	return int64(toInt(raw)) == rv.Int()
}

// toString renders any value as a string. nil and false are the empty string,
// true is "1", a float keeps no trailing zeros, and a fmt.Stringer states its
// own.
func toString(v any) string {
	switch typed := v.(type) {
	case nil:
		return ""
	case string:
		return typed
	case bool:
		if typed {
			return "1"
		}
		return ""
	case float64:
		return strconv.FormatFloat(typed, 'f', -1, 64)
	case float32:
		return strconv.FormatFloat(float64(typed), 'f', -1, 32)
	case int:
		return strconv.Itoa(typed)
	case int64:
		return strconv.FormatInt(typed, 10)
	case fmtStringer:
		return typed.String()
	default:
		return fmt.Sprintf("%v", v)
	}
}

type fmtStringer interface{ String() string }

// toBool reads a value for its truth: a bool as itself, a number as whether it
// equals one, and a string as whether it is "1", "true", "on" or "yes" in any
// case. Anything else is false.
func toBool(v any) bool {
	switch typed := v.(type) {
	case nil:
		return false
	case bool:
		return typed
	case int:
		return typed == 1
	case int64:
		return typed == 1
	case float64:
		return typed == 1
	case string:
		switch strings.ToLower(strings.TrimSpace(typed)) {
		case "1", "true", "on", "yes":
			return true
		}
		return false
	default:
		return false
	}
}

// toInt reads the leading numeric run of a value and truncates it, or zero
// when there is none.
func toInt(v any) int { return int(toFloat(v)) }

// toFloat reads a value as a float64: a number as itself, a bool as one or
// zero, and anything else by its leading numeric run.
func toFloat(v any) float64 {
	switch typed := v.(type) {
	case nil:
		return 0
	case bool:
		if typed {
			return 1
		}
		return 0
	case int:
		return float64(typed)
	case int64:
		return float64(typed)
	case float32:
		return float64(typed)
	case float64:
		return typed
	case string:
		return leadingFloat(typed)
	default:
		return leadingFloat(toString(v))
	}
}

func leadingFloat(s string) float64 {
	s = strings.TrimLeft(s, " \t\n\r")
	end := 0
	seenDigit, seenDot, seenExp := false, false, false
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= '0' && c <= '9':
			seenDigit = true
		case (c == '+' || c == '-') && (i == 0 || s[i-1] == 'e' || s[i-1] == 'E'):
		case c == '.' && !seenDot && !seenExp:
			seenDot = true
		case (c == 'e' || c == 'E') && seenDigit && !seenExp:
			seenExp = true
		default:
			i = len(s)
			continue
		}
		end = i + 1
	}
	if !seenDigit {
		return 0
	}
	parsed, err := strconv.ParseFloat(strings.TrimRight(s[:end], "eE+-"), 64)
	if err != nil {
		return 0
	}
	return parsed
}

// value invokes a func() any and returns its result; every other value is
// itself.
func value(v any) any {
	if fn, ok := v.(func() any); ok {
		return fn()
	}
	return v
}

func firstOr[T any](values []T, def T) T {
	if len(values) > 0 {
		return values[0]
	}
	return def
}

// subsetByKeys returns a fresh map holding source's value under each of the
// dotted keys, with a key the source does not hold coming back as nil.
//
// An empty key is skipped rather than written: a map cannot be replaced through
// its own reference, so it names no destination to write to.
func subsetByKeys(source map[string]any, keys []string) map[string]any {
	results := map[string]any{}
	for _, key := range keys {
		if key == "" {
			continue
		}
		held, _ := arr.Get(source, key)
		arr.Set(results, key, held)
	}
	return results
}
