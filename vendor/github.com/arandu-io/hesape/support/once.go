package support

import (
	"crypto/sha256"
	"encoding/hex"
	"runtime"
	"strconv"
	"sync"
)

// onceRepository is the store that keeps the value a call site produced, so
// the second call gives back the first answer. It is reached through
// [Instance]; the name [Once] belongs to the helper that uses it.
type onceRepository struct {
	mu     sync.Mutex
	values map[string]any
}

var (
	onceMu       sync.Mutex
	onceInstance *onceRepository
	onceEnabled  = true
)

// Instance returns the process-wide memoization store, building it on first
// use.
func Instance() *onceRepository {
	onceMu.Lock()
	defer onceMu.Unlock()
	if onceInstance == nil {
		onceInstance = &onceRepository{values: map[string]any{}}
	}
	return onceInstance
}

// Value returns the value of the onceable, computed once and kept under its
// hash. A nil onceable is nil, and while memoization is off with [Disable] the
// callback runs every time.
//
// Nothing is ever evicted: the standard library has no weak map to key the
// store by the object the call was made on, so the object is folded into the
// hash instead and the whole store is dropped with [Flush].
func (o *onceRepository) Value(onceable *Onceable) any {
	if onceable == nil {
		return nil
	}
	onceMu.Lock()
	enabled := onceEnabled
	onceMu.Unlock()
	if !enabled {
		return onceable.Callable()
	}

	o.mu.Lock()
	defer o.mu.Unlock()
	if o.values == nil {
		o.values = map[string]any{}
	}
	if held, ok := o.values[onceable.Hash]; ok {
		return held
	}
	computed := onceable.Callable()
	o.values[onceable.Hash] = computed
	return computed
}

// Enable turns memoization back on.
func Enable() {
	onceMu.Lock()
	defer onceMu.Unlock()
	onceEnabled = true
}

// Disable turns memoization off, so every call runs its callback.
func Disable() {
	onceMu.Lock()
	defer onceMu.Unlock()
	onceEnabled = false
}

// Flush drops the store, so the next call at every site computes again.
func Flush() {
	onceMu.Lock()
	defer onceMu.Unlock()
	onceInstance = nil
}

// Onceable identifies a memoized call: the call site, whatever it was called
// on, and the callback whose value is kept.
type Onceable struct {
	// Hash is the key the value is stored under.
	Hash string
	// Object is whatever the call was made on, and may be nil.
	Object any
	// Callable computes the value the first time it is asked for.
	Callable func() any
}

// NewOnceable builds an [Onceable] from a hash, an object and a callback.
func NewOnceable(hash string, object any, callable func() any) *Onceable {
	return &Onceable{Hash: hash, Object: object, Callable: callable}
}

// TryFromTrace builds an [Onceable] keyed by the call site, or nil when the
// stack cannot be read. skip is how many frames to step over: 0 is the caller
// of TryFromTrace.
//
// The key is the file, the function and the line, and nothing else: a closure
// cannot be asked what it captured, so two calls from one line share a value
// however their captured variables differ.
func TryFromTrace(skip int, callable func() any) *Onceable {
	pc, file, line, ok := runtime.Caller(skip + 1)
	if !ok {
		return nil
	}
	function := ""
	if fn := runtime.FuncForPC(pc); fn != nil {
		function = fn.Name()
	}
	sum := sha256.Sum256([]byte(file + "@" + function + ":" + strconv.Itoa(line)))
	return NewOnceable(hex.EncodeToString(sum[:16]), nil, callable)
}

// Once runs the callback the first time this line of code is reached and gives
// back that value every time after. The value keeps the type it went in with;
// a stored value of another type reads as the zero T.
//
// See [TryFromTrace] for what the value is keyed by, and what that does not
// take into account.
func Once[T any](callback func() T) T {
	onceable := TryFromTrace(1, func() any { return callback() })
	if onceable == nil {
		return callback()
	}
	held := Instance().Value(onceable)
	if typed, ok := held.(T); ok {
		return typed
	}
	var zero T
	return zero
}
