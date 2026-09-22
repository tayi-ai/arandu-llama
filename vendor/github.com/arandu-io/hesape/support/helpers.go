package support

import (
	"fmt"
	"os"
	"reflect"
	"regexp"
	"runtime"
	"sort"
	"strconv"
	"strings"
	"sync"

	"github.com/arandu-io/hesape/support/deferpkg"
)

// exit is os.Exit behind a name a test can replace, so that Dd and
// ValidatedInput.Dd can be exercised without taking the test process down.
var exit = os.Exit

// Append_config renumbers the numeric keys of a map past 9999, so a merge
// appends them instead of writing over what is already there.
//
// A key counts as numeric when its text parses as a number. The numeric keys
// are renumbered in sorted order and every other key is carried through
// untouched.
func Append_config(array map[string]any) map[string]any {
	start := 9999
	out := make(map[string]any, len(array))
	numeric := make([]string, 0, len(array))
	for key, held := range array {
		if _, err := strconv.Atoi(key); err == nil {
			numeric = append(numeric, key)
			continue
		}
		out[key] = held
	}
	sort.Strings(numeric)
	for _, key := range numeric {
		start++
		out[strconv.Itoa(start)] = array[key]
	}
	return out
}

// Blank reports whether there is nothing there at all. A string of spaces is
// blank, a number and a bool never are, an empty slice, map, array or channel
// is, and a nil pointer, interface or func is; a pointer is read through.
func Blank(v any) bool {
	switch typed := v.(type) {
	case nil:
		return true
	case string:
		return strings.TrimSpace(typed) == ""
	case bool, int, int8, int16, int32, int64, uint, uint8, uint16, uint32, uint64, float32, float64:
		return false
	case fmtStringer:
		return strings.TrimSpace(typed.String()) == ""
	}

	rv := reflect.ValueOf(v)
	switch rv.Kind() {
	case reflect.Pointer, reflect.Interface:
		if rv.IsNil() {
			return true
		}
		return Blank(rv.Elem().Interface())
	case reflect.Slice, reflect.Map, reflect.Array, reflect.Chan:
		return rv.Len() == 0
	case reflect.Func:
		return rv.IsNil()
	default:
		return false
	}
}

// Filled reports whether the value is not [Blank].
func Filled(v any) bool { return !Blank(v) }

// Class_basename returns the type name with its package path off. A string is
// read as the name itself, and a pointer type is read through. An untyped nil
// gives the empty string.
func Class_basename(class any) string {
	name, ok := class.(string)
	if !ok {
		rt := reflect.TypeOf(class)
		if rt == nil {
			return ""
		}
		for rt.Kind() == reflect.Pointer {
			rt = rt.Elem()
		}
		name = rt.String()
	}
	name = strings.ReplaceAll(name, `\`, "/")
	if i := strings.LastIndexAny(name, "/."); i >= 0 {
		return name[i+1:]
	}
	return name
}

// htmlEscapes rewrites the five characters that can end an attribute or open a
// tag, and nothing else.
var htmlEscapes = strings.NewReplacer(
	"&", "&amp;",
	"<", "&lt;",
	">", "&gt;",
	`"`, "&quot;",
	"'", "&#039;",
)

// alreadyAnEntity matches an ampersand that already opens an entity, named or
// numeric, which [E] leaves alone when told not to double-encode.
var alreadyAnEntity = regexp.MustCompile(`&(#[0-9]+|#[xX][0-9a-fA-F]+|[a-zA-Z][a-zA-Z0-9]*);`)

// E writes the HTML-special characters of a value as entities, so it cannot
// escape the attribute it is put in.
//
// An [Htmlable] states its own markup and is not escaped again. The variadic
// argument is whether to double-encode and defaults to true; false leaves an
// ampersand that already opens an entity alone.
func E(v any, doubleEncode ...bool) string {
	if htmlable, ok := v.(Htmlable); ok {
		return htmlable.ToHtml()
	}
	raw := toString(v)
	if firstOr(doubleEncode, true) {
		return htmlEscapes.Replace(raw)
	}

	var b strings.Builder
	b.Grow(len(raw))
	for i := 0; i < len(raw); {
		if raw[i] == '&' {
			if match := alreadyAnEntity.FindString(raw[i:]); match != "" && strings.HasPrefix(raw[i:], match) {
				b.WriteString(match)
				i += len(match)
				continue
			}
		}
		b.WriteString(htmlEscapes.Replace(raw[i : i+1]))
		i++
	}
	return b.String()
}

// Laravel_cloud reports whether the hosting platform of that name is running
// this process, which it marks with an environment variable set to "1".
func Laravel_cloud() bool {
	return os.Getenv("LARAVEL_CLOUD") == "1"
}

// Object_get reads a value out of a struct or a map by dotted key. An empty
// key gives the object itself, and a name that cannot be reached gives the
// optional default, which is nil when not given.
//
// Each segment reads an exported struct field or a map key, through
// [Optional], so a nil anywhere along the path stops the walk.
func Object_get(object any, key string, def ...any) any {
	if strings.TrimSpace(key) == "" {
		return object
	}
	current := object
	for _, segment := range strings.Split(key, ".") {
		next := NewOptional(current).Get(segment)
		if next == nil {
			return value(firstOr(def, nil))
		}
		current = next
	}
	return current
}

// Preg_replace_array replaces each match of the pattern with the next value of
// the list, in order. A match past the end of the list is replaced with
// nothing, and a pattern that will not compile leaves the subject alone.
//
// The pattern is a Go regular expression, carrying no delimiters.
func Preg_replace_array(pattern string, replacements []string, subject string) string {
	expression, err := regexp.Compile(pattern)
	if err != nil {
		return subject
	}
	next := 0
	return expression.ReplaceAllStringFunc(subject, func(match string) string {
		if next >= len(replacements) {
			return ""
		}
		replacement := replacements[next]
		next++
		return replacement
	})
}

// Retry runs the callback again while it fails, up to the given number of
// attempts, and returns the last error when the attempts run out.
//
// times is an int count of attempts, or a []int of backoff milliseconds, one
// per retry. The variadic argument is the pause between attempts and the test
// that decides whether to retry, in that order: the pause is an int of
// milliseconds, a func() int or a func(attempt int, err error) int, and
// defaults to none; the test is a func(err error) bool, and by default every
// error is retried. The pause goes through [Usleep], so a test that called
// [Fake] captures it instead of serving it.
func Retry(times any, callback func(attempt int) (any, error), options ...any) (any, error) {
	var backoff []int
	remaining := 0
	switch t := times.(type) {
	case []int:
		backoff = t
		remaining = len(t) + 1
	default:
		remaining = toInt(times)
	}

	sleepMilliseconds := any(0)
	if len(options) > 0 {
		sleepMilliseconds = options[0]
	}
	var when func(err error) bool
	if len(options) > 1 {
		if test, ok := options[1].(func(err error) bool); ok {
			when = test
		}
	}

	attempts := 0
	for {
		attempts++
		remaining--

		result, err := callback(attempts)
		if err == nil {
			return result, nil
		}
		if remaining < 1 || (when != nil && !when(err)) {
			return nil, err
		}

		if attempts-1 < len(backoff) {
			sleepMilliseconds = backoff[attempts-1]
		}
		milliseconds := sleepMilliseconds
		switch fn := sleepMilliseconds.(type) {
		case func(attempt int, err error) int:
			milliseconds = fn(attempts, err)
		case func() int:
			milliseconds = fn()
		}
		if pause := toInt(milliseconds); pause > 0 {
			_ = Usleep(pause * 1000).Goodnight()
		}
	}
}

// Tap hands the value to the callback, then hands the value back. A nil
// callback returns the value untouched.
func Tap[T any](v T, callback func(T)) T {
	if callback != nil {
		callback(v)
	}
	return v
}

// Throw_if returns the error when the condition holds, and nil when it does
// not.
func Throw_if(condition bool, err error) error {
	if condition {
		return err
	}
	return nil
}

// Throw_unless returns the error when the condition does not hold, and nil
// when it does.
func Throw_unless(condition bool, err error) error {
	return Throw_if(!condition, err)
}

// Transform returns the callback's result when the value is [Filled], and the
// optional default when it is [Blank]. With no default, a blank value gives
// the zero R.
func Transform[T, R any](v T, callback func(T) R, def ...R) R {
	if Filled(any(v)) {
		return callback(v)
	}
	var zero R
	return firstOr(def, zero)
}

// Windows_os reports whether the process is running on Windows.
func Windows_os() bool { return runtime.GOOS == "windows" }

// With returns the value passed through the callback. The callback is
// required: a value passed through nothing is the value itself.
func With[T, R any](v T, callback func(T) R) R { return callback(v) }

// Dump writes the values to standard error with %#v, one per line, and returns
// the first of them; with no value it returns nil.
//
// Standard error is where a dump belongs when standard output is the response.
func Dump(values ...any) any {
	for _, v := range values {
		fmt.Fprintf(os.Stderr, "%#v\n", v)
	}
	if len(values) == 0 {
		return nil
	}
	return values[0]
}

// Dd dumps the values and ends the process with status 1.
func Dd(values ...any) {
	Dump(values...)
	exit(1)
}

// Application is the part of the running application that [WithLocale]
// touches. Nothing is resolved from a registry, so the caller hands it in.
type Application interface {
	// GetLocale returns the locale the application is running under.
	GetLocale() string
	// SetLocale sets the locale the application runs under.
	SetLocale(locale string)
}

// WithLocale runs the callback under the given locale and puts the old one
// back afterwards, whatever happens. An empty locale, or a nil application,
// runs the callback as it stands.
func WithLocale(app Application, locale string, callback func() any) any {
	if locale == "" || app == nil {
		return callback()
	}
	original := app.GetLocale()
	defer app.SetLocale(original)
	app.SetLocale(locale)
	return callback()
}

// ErrBadMethodCall is returned when a call is forwarded to a method the target
// does not carry.
type ErrBadMethodCall struct {
	// Class is the type the call was forwarded to.
	Class string
	// Method is the name that was called on it.
	Method string
}

// Error names the type the call was forwarded to and the method called on it.
func (e *ErrBadMethodCall) Error() string {
	return fmt.Sprintf("Call to undefined method %s::%s()", e.Class, e.Method)
}

// ThrowBadMethodCallException builds an [ErrBadMethodCall] naming the object's
// type and the method, and returns it.
func ThrowBadMethodCallException(object any, method string) error {
	return &ErrBadMethodCall{Class: Class_basename(object), Method: method}
}

// ForwardCallTo calls the named method on the object through reflection and
// returns what it returned, one entry per result.
//
// A method the object does not carry, and a non-variadic method handed the
// wrong number of arguments, are both [ErrBadMethodCall]. A nil argument is
// passed as the zero value of the parameter it lands on.
func ForwardCallTo(object any, method string, parameters ...any) ([]any, error) {
	rv := reflect.ValueOf(object)
	if !rv.IsValid() {
		return nil, ThrowBadMethodCallException(object, method)
	}
	fn := rv.MethodByName(method)
	if !fn.IsValid() {
		return nil, ThrowBadMethodCallException(object, method)
	}

	rt := fn.Type()
	if !rt.IsVariadic() && rt.NumIn() != len(parameters) {
		return nil, ThrowBadMethodCallException(object, method)
	}

	in := make([]reflect.Value, 0, len(parameters))
	for i, parameter := range parameters {
		if parameter == nil {
			at := rt.In(min(i, rt.NumIn()-1))
			in = append(in, reflect.Zero(at))
			continue
		}
		in = append(in, reflect.ValueOf(parameter))
	}

	returned := fn.Call(in)
	out := make([]any, 0, len(returned))
	for _, v := range returned {
		out = append(out, v.Interface())
	}
	return out, nil
}

// ForwardDecoratedCallTo is [ForwardCallTo] with one change: a result that is
// the object itself comes back as the decorator, so a chained call keeps
// returning the decorator.
func ForwardDecoratedCallTo(decorator, object any, method string, parameters ...any) ([]any, error) {
	returned, err := ForwardCallTo(object, method, parameters...)
	if err != nil {
		return nil, err
	}
	for i, v := range returned {
		if v == object {
			returned[i] = decorator
		}
	}
	return returned, nil
}

var (
	deferredMu         sync.Mutex
	deferredCollection *deferpkg.DeferredCallbackCollection
)

// DeferredCallbackCollection returns the one collection every deferred
// callback of this process lands in, building it on first use. Nothing is
// resolved from a registry: the collection is this package's own.
func DeferredCallbackCollection() *deferpkg.DeferredCallbackCollection {
	deferredMu.Lock()
	defer deferredMu.Unlock()
	if deferredCollection == nil {
		deferredCollection = deferpkg.NewDeferredCallbackCollection()
	}
	return deferredCollection
}

// Defer puts the callback off until the response has been sent, and returns
// the handle it can be called off by.
//
// The variadic argument is the name and then the always flag, defaulting to an
// empty name and false; an empty name is filled with a random one. To reach
// the collection itself, call [DeferredCallbackCollection].
func Defer(callback func(), options ...any) *deferpkg.DeferredCallback {
	name := ""
	if len(options) > 0 {
		name = toString(options[0])
	}
	always := false
	if len(options) > 1 {
		always = toBool(options[1])
	}
	deferred := deferpkg.NewDeferredCallback(callback, name, always)
	DeferredCallbackCollection().OffsetSet(-1, deferred)
	return deferred
}
