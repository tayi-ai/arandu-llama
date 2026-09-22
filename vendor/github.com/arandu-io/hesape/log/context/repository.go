package context

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"maps"
	"reflect"
	"slices"
	"strconv"
	"sync"

	"github.com/arandu-io/hesape/log/context/events"
)

// Dispatcher is the slice of an event dispatcher this package needs, declared on
// the side that consumes it so that one concrete dispatcher can serve every
// package that fires an event.
type Dispatcher interface {
	// Dispatch fires the event.
	Dispatch(event any)

	// Listen registers a listener that receives every event; Dehydrating and
	// Hydrated wrap it with the type assertion that selects theirs.
	Listen(listener func(event any))
}

// repositoryKey is the context key the repository travels under.
type repositoryKey struct{}

// Repository is the context that crosses a whole request and lands in every log
// line it produces.
//
// A Repository is safe for concurrent use.
type Repository struct {
	mu     sync.RWMutex
	events Dispatcher
	data   map[string]any
	hidden map[string]any
}

// handleUnserializeExceptions is what HandleUnserializeExceptionsUsing sets. It
// is package level, so it is set once for the process, and it is guarded because
// any goroutine may be hydrating.
var (
	handleUnserializeMu         sync.RWMutex
	handleUnserializeExceptions func(err error, key string, value any, hidden bool) any
)

// New returns an empty repository that fires its events on dispatcher.
//
// The dispatcher may be nil: a repository without one still holds context, and
// Dehydrating, Hydrated, Dehydrate and Hydrate simply have nobody to tell.
func New(dispatcher Dispatcher) *Repository {
	return &Repository{
		events: dispatcher,
		data:   map[string]any{},
		hidden: map[string]any{},
	}
}

// Into stores the repository in ctx and returns the new context.
//
// The context is the carrier because one repository shared across the process is
// not safe under concurrency, and one per context is.
func Into(ctx context.Context, repository *Repository) context.Context {
	return context.WithValue(ctx, repositoryKey{}, repository)
}

// For returns the repository ctx carries, or nil when it carries none.
//
// It is the read side of Into. Every method is safe on a nil receiver, so a
// caller may use the result without checking: reading gives the zero value and
// writing goes nowhere.
func For(ctx context.Context) *Repository {
	if ctx == nil {
		return nil
	}
	repository, _ := ctx.Value(repositoryKey{}).(*Repository)
	return repository
}

// Has reports whether the key exists.
//
// A key added with a nil value is here, even though Get returns the default for
// it: presence and value are two different questions.
func (r *Repository) Has(key string) bool {
	if r == nil {
		return false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.data[key]
	return ok
}

// Missing reports whether the key is absent.
func (r *Repository) Missing(key string) bool {
	return !r.Has(key)
}

// HasHidden reports whether the key exists in the hidden half.
func (r *Repository) HasHidden(key string) bool {
	if r == nil {
		return false
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.hidden[key]
	return ok
}

// MissingHidden reports whether the key is absent from the hidden half.
func (r *Repository) MissingHidden(key string) bool {
	return !r.HasHidden(key)
}

// All returns all the context data.
//
// It is a copy, because a live map would be read while another goroutine writes
// it.
func (r *Repository) All() map[string]any {
	if r == nil {
		return map[string]any{}
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return maps.Clone(r.data)
}

// AllHidden returns a copy of the hidden half.
func (r *Repository) AllHidden() map[string]any {
	if r == nil {
		return map[string]any{}
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return maps.Clone(r.hidden)
}

// Get returns the key's value, or the default the variadic carries.
//
// A func() any default is called only when it is needed. A key holding nil takes
// the default too: Has is what tells a missing key from a nil one.
func (r *Repository) Get(key string, def ...any) any {
	if r == nil {
		return defaultValue(def)
	}
	r.mu.RLock()
	found := r.data[key]
	r.mu.RUnlock()

	if found != nil {
		return found
	}
	return defaultValue(def)
}

// GetHidden is Get over the hidden half.
func (r *Repository) GetHidden(key string, def ...any) any {
	if r == nil {
		return defaultValue(def)
	}
	r.mu.RLock()
	found := r.hidden[key]
	r.mu.RUnlock()

	if found != nil {
		return found
	}
	return defaultValue(def)
}

// Pull returns the key's value, and then forgets the key.
func (r *Repository) Pull(key string, def ...any) any {
	found := r.Get(key, def...)
	r.Forget(key)
	return found
}

// PullHidden is Pull over the hidden half.
func (r *Repository) PullHidden(key string, def ...any) any {
	found := r.GetHidden(key, def...)
	r.ForgetHidden(key)
	return found
}

// Only returns the values of the given keys, and nothing else.
//
// A key that is not there is simply absent from the result.
func (r *Repository) Only(keys []string) map[string]any {
	if r == nil {
		return map[string]any{}
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return only(r.data, keys)
}

// OnlyHidden is Only over the hidden half.
func (r *Repository) OnlyHidden(keys []string) map[string]any {
	if r == nil {
		return map[string]any{}
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return only(r.hidden, keys)
}

// Except returns everything but the given keys.
func (r *Repository) Except(keys []string) map[string]any {
	if r == nil {
		return map[string]any{}
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return except(r.data, keys)
}

// ExceptHidden is Except over the hidden half.
func (r *Repository) ExceptHidden(keys []string) map[string]any {
	if r == nil {
		return map[string]any{}
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return except(r.hidden, keys)
}

// Add adds a context value.
//
// key is any because it takes two shapes: a string with a value, or a
// map[string]any on its own, whose entries are merged in. A string key with no
// value adds nil.
func (r *Repository) Add(key any, value ...any) *Repository {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	merge(r.data, key, value)
	return r
}

// AddHidden is Add into the hidden half, which is the half that travels with a
// queued job but never reaches a log line.
func (r *Repository) AddHidden(key any, value ...any) *Repository {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	merge(r.hidden, key, value)
	return r
}

// AddIf adds the value only when the key is not there yet.
func (r *Repository) AddIf(key string, value any) *Repository {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.data[key]; !ok {
		r.data[key] = value
	}
	return r
}

// AddHiddenIf is AddIf over the hidden half.
func (r *Repository) AddHiddenIf(key string, value any) *Repository {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if _, ok := r.hidden[key]; !ok {
		r.hidden[key] = value
	}
	return r
}

// Remember adds the value when the key is missing, and returns the value either
// way.
//
// A func() any value is only called when the key is missing.
func (r *Repository) Remember(key string, value any) any {
	if r == nil {
		return unwrap(value)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if existing, ok := r.data[key]; ok {
		return existing
	}
	resolved := unwrap(value)
	r.data[key] = resolved
	return resolved
}

// RememberHidden is Remember over the hidden half.
func (r *Repository) RememberHidden(key string, value any) any {
	if r == nil {
		return unwrap(value)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	if existing, ok := r.hidden[key]; ok {
		return existing
	}
	resolved := unwrap(value)
	r.hidden[key] = resolved
	return resolved
}

// Forget drops the keys.
//
// A key that is not there is not an error.
func (r *Repository) Forget(key ...string) *Repository {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, k := range key {
		delete(r.data, k)
	}
	return r
}

// ForgetHidden is Forget over the hidden half.
func (r *Repository) ForgetHidden(key ...string) *Repository {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	for _, k := range key {
		delete(r.hidden, k)
	}
	return r
}

// Push appends the values to the key's stack.
//
// A key that holds something which is not a list is not stackable, and the error
// for it matches ErrUnableToPush. A key that holds nothing is stackable, and
// pushing creates the stack.
func (r *Repository) Push(key string, values ...any) (*Repository, error) {
	if r == nil {
		return nil, errUnableToPush(key)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r, push(r.data, key, values, errUnableToPush)
}

// PushHidden is Push over the hidden half.
func (r *Repository) PushHidden(key string, values ...any) (*Repository, error) {
	if r == nil {
		return nil, errUnableToPushHidden(key)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return r, push(r.hidden, key, values, errUnableToPushHidden)
}

// Pop takes the latest value off the key's stack.
//
// A key that is not a stack, and a stack that is empty, are both an error
// matching ErrUnableToPop.
func (r *Repository) Pop(key string) (any, error) {
	if r == nil {
		return nil, errUnableToPop(key)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return pop(r.data, key, errUnableToPop)
}

// PopHidden is Pop over the hidden half.
func (r *Repository) PopHidden(key string) (any, error) {
	if r == nil {
		return nil, errUnableToPopHidden(key)
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	return pop(r.hidden, key, errUnableToPopHidden)
}

// Increment adds to a counter, starting it at zero.
//
// The variadic is the amount, and it defaults to one. A key holding something
// that is not a number restarts from zero.
func (r *Repository) Increment(key string, amount ...int) *Repository {
	if r == nil {
		return nil
	}
	by := 1
	if len(amount) > 0 {
		by = amount[0]
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.data[key] = toInt(r.data[key]) + by
	return r
}

// Decrement is Increment with the amount negated.
func (r *Repository) Decrement(key string, amount ...int) *Repository {
	by := 1
	if len(amount) > 0 {
		by = amount[0]
	}
	return r.Increment(key, -by)
}

// StackContains reports whether the value is in the key's stack.
//
// A key that is not a stack is an error matching ErrNotAStack; a key that holds
// nothing is a stack, and the answer is false. value may be a func(any) bool,
// which is then the test each item is put to. The variadic is strictness:
// strict compares type and value, loose compares two numbers numerically and
// anything else by its printed form.
func (r *Repository) StackContains(key string, value any, strict ...bool) (bool, error) {
	if r == nil {
		return false, errNotAStack(key)
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return stackContains(r.data, key, value, strict)
}

// HiddenStackContains is StackContains over the hidden half.
func (r *Repository) HiddenStackContains(key string, value any, strict ...bool) (bool, error) {
	if r == nil {
		return false, errNotAStack(key)
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return stackContains(r.hidden, key, value, strict)
}

// Scope runs the callback with the given values added, and puts the context back
// the way it was afterwards -- including when the callback fails or panics.
//
// A Go method cannot take a type parameter, so the callback returns only an
// error and a caller who wants a value closes over it.
func (r *Repository) Scope(callback func() error, data map[string]any, hidden map[string]any) error {
	if r == nil {
		return callback()
	}

	r.mu.Lock()
	dataBefore, hiddenBefore := maps.Clone(r.data), maps.Clone(r.hidden)
	maps.Copy(r.data, data)
	maps.Copy(r.hidden, hidden)
	r.mu.Unlock()

	defer func() {
		r.mu.Lock()
		r.data, r.hidden = dataBefore, hiddenBefore
		r.mu.Unlock()
	}()

	return callback()
}

// When runs the callback when the condition holds, and the fallback when it does
// not.
//
// condition is a bool, a func() bool, a func(*Repository) bool, or any value
// read for truthiness. The variadic is the fallback.
func (r *Repository) When(condition any, callback func(*Repository), otherwise ...func(*Repository)) *Repository {
	if truthy(condition, r) {
		if callback != nil {
			callback(r)
		}
		return r
	}
	if len(otherwise) > 0 && otherwise[0] != nil {
		otherwise[0](r)
	}
	return r
}

// IsEmpty reports whether there is nothing visible and nothing hidden.
func (r *Repository) IsEmpty() bool {
	if r == nil {
		return true
	}
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.data) == 0 && len(r.hidden) == 0
}

// Flush drops everything, both halves.
func (r *Repository) Flush() *Repository {
	if r == nil {
		return nil
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.data, r.hidden = map[string]any{}, map[string]any{}
	return r
}

// Dehydrating runs the callback when the context is about to be written down for
// a queued job.
//
// The callback receives the repository the event carries, which is the copy made
// for the dehydration, not this one. Without a dispatcher there is nothing to
// listen on and the call does nothing.
func (r *Repository) Dehydrating(callback func(*Repository)) *Repository {
	if r == nil || r.events == nil || callback == nil {
		return r
	}
	r.events.Listen(func(event any) {
		if dehydrating, ok := event.(events.ContextDehydrating); ok {
			if repository, ok := dehydrating.Context.(*Repository); ok {
				callback(repository)
			}
		}
	})
	return r
}

// Hydrated runs the callback once a dehydrated context has been read back.
func (r *Repository) Hydrated(callback func(*Repository)) *Repository {
	if r == nil || r.events == nil || callback == nil {
		return r
	}
	r.events.Listen(func(event any) {
		if hydrated, ok := event.(events.ContextHydrated); ok {
			if repository, ok := hydrated.Context.(*Repository); ok {
				callback(repository)
			}
		}
	})
	return r
}

// HandleUnserializeExceptionsUsing sets what to do with a value Hydrate cannot
// read back.
//
// It is package level, so it is set once for the process and not per repository.
// The callback returns the value to use in place of the broken one; a nil
// callback restores the default, which is to fail.
func (r *Repository) HandleUnserializeExceptionsUsing(callback func(err error, key string, value any, hidden bool) any) *Repository {
	handleUnserializeMu.Lock()
	handleUnserializeExceptions = callback
	handleUnserializeMu.Unlock()
	return r
}

// Dehydrate writes the context down so it can travel with a queued job.
//
// It dispatches ContextDehydrating with a copy first, so a listener can still
// change what travels, and returns nil when there is nothing to carry, which is
// the signal that no context needs to be attached.
//
// Each value is encoded as JSON, because that is the encoding that survives
// crossing into a process built from different code.
func (r *Repository) Dehydrate() (map[string]any, error) {
	if r == nil {
		return nil, nil
	}

	instance := New(r.events).Add(r.All()).AddHidden(r.AllHidden())
	if r.events != nil {
		r.events.Dispatch(events.ContextDehydrating{Context: instance})
	}
	if instance.IsEmpty() {
		return nil, nil
	}

	data, err := serialize(instance.All())
	if err != nil {
		return nil, err
	}
	hidden, err := serialize(instance.AllHidden())
	if err != nil {
		return nil, err
	}
	return map[string]any{"data": data, "hidden": hidden}, nil
}

// Hydrate reads a dehydrated context back, replacing whatever this repository
// held.
//
// It accepts what Dehydrate produced, and nil, which leaves an empty context. A
// value that will not read back goes to the callback
// HandleUnserializeExceptionsUsing set, and without one it is returned as an
// error.
func (r *Repository) Hydrate(dehydrated map[string]any) error {
	if r == nil {
		return nil
	}

	data, err := unserialize(dehydrated["data"], false)
	if err != nil {
		return err
	}
	hidden, err := unserialize(dehydrated["hidden"], true)
	if err != nil {
		return err
	}

	r.Flush().Add(data).AddHidden(hidden)
	if r.events != nil {
		r.events.Dispatch(events.ContextHydrated{Context: r})
	}
	return nil
}

// only is the body Only and OnlyHidden share.
func only(source map[string]any, keys []string) map[string]any {
	out := make(map[string]any, len(keys))
	for _, key := range keys {
		if found, ok := source[key]; ok {
			out[key] = found
		}
	}
	return out
}

// except is the body Except and ExceptHidden share.
func except(source map[string]any, keys []string) map[string]any {
	out := maps.Clone(source)
	if out == nil {
		return map[string]any{}
	}
	for _, key := range keys {
		delete(out, key)
	}
	return out
}

// merge writes one key and value, or every entry of a map given as the key.
func merge(into map[string]any, key any, value []any) {
	switch typed := key.(type) {
	case map[string]any:
		maps.Copy(into, typed)
	case string:
		if len(value) > 0 {
			into[typed] = value[0]
			return
		}
		into[typed] = nil
	default:
		var single any
		if len(value) > 0 {
			single = value[0]
		}
		into[fmt.Sprint(key)] = single
	}
}

// push is the body Push and PushHidden share, including the stackable check.
func push(into map[string]any, key string, values []any, fail func(string) error) error {
	existing, ok := into[key]
	if ok {
		stack, isList := existing.([]any)
		if !isList {
			return fail(key)
		}
		into[key] = append(slices.Clone(stack), values...)
		return nil
	}
	into[key] = append([]any{}, values...)
	return nil
}

// pop is the body Pop and PopHidden share.
func pop(from map[string]any, key string, fail func(string) error) (any, error) {
	stack, ok := from[key].([]any)
	if !ok || len(stack) == 0 {
		return nil, fail(key)
	}
	last := stack[len(stack)-1]
	from[key] = stack[:len(stack)-1]
	return last, nil
}

// stackContains is the body StackContains and HiddenStackContains share.
func stackContains(source map[string]any, key string, value any, strict []bool) (bool, error) {
	found, ok := source[key]
	if !ok {
		return false, nil
	}
	stack, isList := found.([]any)
	if !isList {
		return false, errNotAStack(key)
	}
	if match, isClosure := value.(func(any) bool); isClosure {
		return slices.ContainsFunc(stack, match), nil
	}

	exact := len(strict) > 0 && strict[0]
	for _, item := range stack {
		if exact && reflect.DeepEqual(item, value) {
			return true, nil
		}
		if !exact && looseEqual(item, value) {
			return true, nil
		}
	}
	return false, nil
}

// serialize encodes every value of the map as JSON.
func serialize(source map[string]any) (map[string]any, error) {
	out := make(map[string]any, len(source))
	for key, value := range source {
		encoded, err := json.Marshal(value)
		if err != nil {
			return nil, fmt.Errorf("log/context: unable to dehydrate key [%s]: %w", key, err)
		}
		out[key] = string(encoded)
	}
	return out, nil
}

// unserialize decodes every value back, sending what will not decode to the
// callback HandleUnserializeExceptionsUsing set.
func unserialize(source any, hidden bool) (map[string]any, error) {
	encoded, ok := source.(map[string]any)
	if !ok {
		return map[string]any{}, nil
	}

	handleUnserializeMu.RLock()
	handle := handleUnserializeExceptions
	handleUnserializeMu.RUnlock()

	out := make(map[string]any, len(encoded))
	for key, value := range encoded {
		text, isText := value.(string)
		if !isText {
			out[key] = value
			continue
		}
		var decoded any
		if err := json.Unmarshal([]byte(text), &decoded); err != nil {
			wrapped := fmt.Errorf("log/context: unable to hydrate key [%s]: %w", key, err)
			if handle == nil {
				return nil, wrapped
			}
			out[key] = handle(wrapped, key, value, hidden)
			continue
		}
		out[key] = decoded
	}
	return out, nil
}

// defaultValue resolves the variadic default: none is nil, and one is unwrapped.
func defaultValue(def []any) any {
	if len(def) == 0 {
		return nil
	}
	return unwrap(def[0])
}

// unwrap calls a func() any, and returns anything else as it is.
func unwrap(value any) any {
	if resolve, ok := value.(func() any); ok {
		return resolve()
	}
	return value
}

// truthy reads a When condition: a bool is itself, a func is called, and
// anything else is false only when it is nil, an empty or "0" string, or a zero
// number.
func truthy(condition any, repository *Repository) bool {
	switch typed := condition.(type) {
	case nil:
		return false
	case bool:
		return typed
	case func(*Repository) bool:
		return typed(repository)
	case func() bool:
		return typed()
	case string:
		return typed != "" && typed != "0"
	case int:
		return typed != 0
	case float64:
		return typed != 0
	}
	return true
}

// toInt reads a value as an int: a number becomes itself, a numeric string
// becomes its number, and anything else becomes zero.
func toInt(value any) int {
	switch typed := value.(type) {
	case int:
		return typed
	case int8:
		return int(typed)
	case int16:
		return int(typed)
	case int32:
		return int(typed)
	case int64:
		return int(typed)
	case uint:
		return int(typed)
	case uint8:
		return int(typed)
	case uint16:
		return int(typed)
	case uint32:
		return int(typed)
	case uint64:
		return int(typed)
	case float32:
		return int(typed)
	case float64:
		return int(typed)
	case string:
		parsed, err := strconv.ParseFloat(typed, 64)
		if err != nil {
			return 0
		}
		return int(parsed)
	}
	return 0
}

// looseEqual is the non-strict comparison: two numbers compare numerically
// whatever their Go types, and everything else compares by the form it prints
// as.
func looseEqual(a, b any) bool {
	if reflect.DeepEqual(a, b) {
		return true
	}
	an, aok := toFloat(a)
	bn, bok := toFloat(b)
	if aok && bok {
		return an == bn
	}
	return fmt.Sprint(a) == fmt.Sprint(b)
}

// toFloat reports the numeric value of a number, and whether it was one.
func toFloat(value any) (float64, bool) {
	switch typed := value.(type) {
	case int:
		return float64(typed), true
	case int8:
		return float64(typed), true
	case int16:
		return float64(typed), true
	case int32:
		return float64(typed), true
	case int64:
		return float64(typed), true
	case uint:
		return float64(typed), true
	case uint8:
		return float64(typed), true
	case uint16:
		return float64(typed), true
	case uint32:
		return float64(typed), true
	case uint64:
		return float64(typed), true
	case float32:
		return float64(typed), true
	case float64:
		return typed, true
	}
	return 0, false
}

// The three stack failures. Each error names the key that caused it and wraps
// the sentinel, so that a caller can tell the three apart with errors.Is instead
// of reading the text.
var (
	// ErrUnableToPush is what Push and PushHidden report for a key that holds
	// something other than a list.
	ErrUnableToPush = errors.New("log/context: unable to push value onto context stack")

	// ErrUnableToPop is what Pop and PopHidden report for a key that is not a
	// stack or is an empty one.
	ErrUnableToPop = errors.New("log/context: unable to pop value from context stack")

	// ErrNotAStack is what StackContains and HiddenStackContains report for a
	// key that holds something other than a list.
	ErrNotAStack = errors.New("log/context: key is not a stack")
)

func errUnableToPush(key string) error {
	return fmt.Errorf("log/context: unable to push value onto context stack for key [%s]: %w", key, ErrUnableToPush)
}

func errUnableToPushHidden(key string) error {
	return fmt.Errorf("log/context: unable to push value onto hidden context stack for key [%s]: %w", key, ErrUnableToPush)
}

func errUnableToPop(key string) error {
	return fmt.Errorf("log/context: unable to pop value from context stack for key [%s]: %w", key, ErrUnableToPop)
}

func errUnableToPopHidden(key string) error {
	return fmt.Errorf("log/context: unable to pop value from hidden context stack for key [%s]: %w", key, ErrUnableToPop)
}

func errNotAStack(key string) error {
	return fmt.Errorf("log/context: given key [%s] is not a stack: %w", key, ErrNotAStack)
}
