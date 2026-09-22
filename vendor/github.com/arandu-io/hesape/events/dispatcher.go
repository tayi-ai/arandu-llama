package events

import (
	"context"
	"fmt"
	"reflect"
	"strings"
	"sync"
	"time"

	"github.com/arandu-io/hesape/str"
)

// Listener is one prepared listener: the closure MakeListener returns.
//
// The two arguments are the event name and the payload the event was dispatched
// with.
type Listener func(event string, payload []any) any

// ClassListener pairs a listener value with the name of the method to call on
// it.
//
//	dispatcher.Listen("invoice.paid", events.ClassListener{Class: notifier, Method: "WhenPaid"})
//
// Class is the listener value itself rather than the name of its type, because
// a type is not addressable by name at run time -- which is also why
// CreateClassListener has nothing to resolve one out of.
type ClassListener struct {
	// Class is the listener. An empty Method calls its Handle method.
	Class any
	// Method is the method to call. Empty means Handle.
	Method string
}

// Subscriber is an object that registers its own listeners.
//
// The map it returns is event name to listeners; a nil map means the subscriber
// registered everything itself through the dispatcher it was handed.
type Subscriber interface {
	Subscribe(dispatcher *Dispatcher) map[string][]any
}

// ShouldQueue marks a listener that is handled on the queue.
//
// An empty interface in Go is satisfied by every value, so the marker carries
// the method that answers it: returning false is the listener declining the
// queue for this event.
type ShouldQueue interface {
	ShouldQueue(event any) bool
}

// The interfaces below are the optional methods a queued listener may declare to
// shape the job the dispatcher pushes for it. Every one of them is asked with
// the event being dispatched; a listener that does not care about the event
// ignores the parameter, and the event is nil when the event was dispatched
// with no payload.
//
// They are named rather than asserted against inline, so that a listener can
// write
//
//	var _ events.ViaConnection = (*Notifier)(nil)
//
// and hear about the wrong signature from the compiler rather than from
// production: an anonymous interface cannot be asserted against at compile time,
// and a listener that got the signature wrong would simply not match -- the job
// would keep the empty connection, go to the default one, and say nothing about
// it.
type (
	// ViaConnection names the queue connection the listener's job is pushed to.
	ViaConnection interface {
		ViaConnection(event any) string
	}

	// ViaQueue names the queue the listener's job is pushed onto.
	ViaQueue interface {
		ViaQueue(event any) string
	}

	// WithDelay is how long the job waits before it becomes available.
	WithDelay interface {
		WithDelay(event any) time.Duration
	}

	// Tries is how many times the job may be attempted.
	Tries interface {
		Tries(event any) int
	}

	// Backoff is how long to wait before retrying after an uncaught failure.
	Backoff interface {
		Backoff(event any) time.Duration
	}

	// RetryUntil is when the job stops being retried.
	RetryUntil interface {
		RetryUntil(event any) time.Time
	}

	// MessageGroup is the group the job belongs to on the queues that have them.
	MessageGroup interface {
		MessageGroup(event any) string
	}

	// Deduplicator is the deduplication ID for the job, asked with the event the
	// job was built from.
	Deduplicator interface {
		Deduplicator(event any) string
	}

	// UniqueID is the key a unique listener is unique by.
	UniqueID interface {
		UniqueID(event any) any
	}

	// UniqueFor is how long the unique lock is held.
	UniqueFor interface {
		UniqueFor(event any) time.Duration
	}
)

// ShouldDispatchAfterCommit marks an event that waits for the transaction.
//
// It carries a method for the same reason ShouldQueue does.
type ShouldDispatchAfterCommit interface {
	ShouldDispatchAfterCommit() bool
}

// ShouldHandleEventsAfterCommit marks a listener that waits for the transaction.
//
// It carries a method for the same reason ShouldQueue does.
type ShouldHandleEventsAfterCommit interface {
	ShouldHandleEventsAfterCommit() bool
}

// Queue is the little of a queue factory the dispatcher needs to put a listener
// on the queue.
//
// It is declared here rather than imported so this package does not depend on
// the queue implementation, which is the same reason DB is declared here.
type Queue interface {
	// Connection returns the named connection. An empty name is the default.
	Connection(name string) Connection
}

// Connection is the two calls the dispatcher makes on a queue connection.
type Connection interface {
	PushOn(queue string, job any) error
	LaterOn(queue string, delay time.Duration, job any) error
}

// TransactionManager is what the dispatcher needs from a database transaction
// manager: somewhere to hang work that only runs if the transaction commits.
type TransactionManager interface {
	AddCallback(callback func())
}

// Dispatcher registers listeners by event name, fires them in registration
// order, and stops early on request.
//
// The outbox in this package is a different thing and they meet in one place:
// Publish hands a stored event to the listeners registered for its name, so the
// relay delivers through this registry.
//
// A Dispatcher is safe for concurrent use: one process serves many requests, and
// a listener registered while another goroutine dispatches would otherwise be a
// data race. Defer is the exception and says so.
type Dispatcher struct {
	mu sync.Mutex

	// listeners are the registered listeners, unprepared, by event name.
	listeners map[string][]any
	// wildcards are the listeners registered against a pattern.
	wildcards map[string][]any
	// wildcardsCache is the prepared wildcard listeners, by event name.
	wildcardsCache map[string][]Listener

	queueResolver              func() Queue
	transactionManagerResolver func() TransactionManager

	deferredEvents  []deferredEvent
	deferringEvents bool
	eventsToDefer   []string
}

// deferredEvent is one dispatch held back by Defer.
type deferredEvent struct {
	event   any
	payload []any
	halt    bool
}

// NewDispatcher returns a dispatcher with nothing registered.
//
// It takes no arguments: a class listener is the listener value itself, so
// there is nothing to resolve one out of.
func NewDispatcher() *Dispatcher {
	return &Dispatcher{
		listeners:      map[string][]any{},
		wildcards:      map[string][]any{},
		wildcardsCache: map[string][]Listener{},
	}
}

// Listen registers an event listener with the dispatcher.
//
//	d.Listen("invoice.paid", func(e Stored) error { ... })
//	d.Listen([]string{"invoice.paid", "invoice.voided"}, listener)
//	d.Listen("invoice.*", func(name string, payload []any) any { ... })
//	d.Listen(func(e InvoicePaid) { ... })
//
// In the last form the closure registers itself against the type of its first
// parameter, read with reflect. When the first parameter is a context.Context
// the second one names the event, because a listener in Go takes the context it
// will need.
//
// Go has no default argument, so listener is variadic and the one-argument call
// is the closure form.
func (d *Dispatcher) Listen(events any, listener ...any) {
	var l any
	if len(listener) > 0 {
		l = listener[0]
	}

	if l == nil {
		if queued, ok := events.(*QueuedClosure); ok {
			names := firstParameterTypes(queued.Closure)
			assertNamesAnEvent(queued.Closure, names)
			for _, name := range names {
				d.Listen(name, queued.Resolve(d))
			}
			return
		}
		if isFunc(events) {
			names := firstParameterTypes(events)
			assertNamesAnEvent(events, names)
			for _, name := range names {
				d.Listen(name, events)
			}
			return
		}
	}

	if queued, ok := l.(*QueuedClosure); ok {
		l = queued.Resolve(d)
	}
	assertListenable(l)

	d.mu.Lock()
	defer d.mu.Unlock()
	for _, event := range eventNames(events) {
		if strings.Contains(event, "*") {
			d.setupWildcardListen(event, l)
			continue
		}
		d.listeners[event] = append(d.listeners[event], l)
	}
}

// setupWildcardListen records a pattern listener and drops the prepared cache.
// The caller holds the lock.
func (d *Dispatcher) setupWildcardListen(event string, listener any) {
	d.wildcards[event] = append(d.wildcards[event], listener)
	d.wildcardsCache = map[string][]Listener{}
}

// HasListeners reports whether a given event has listeners.
func (d *Dispatcher) HasListeners(eventName string) bool {
	d.mu.Lock()
	_, direct := d.listeners[eventName]
	_, pattern := d.wildcards[eventName]
	d.mu.Unlock()
	return direct || pattern || d.HasWildcardListeners(eventName)
}

// HasWildcardListeners reports whether the given event has any wildcard
// listeners.
func (d *Dispatcher) HasWildcardListeners(eventName string) bool {
	d.mu.Lock()
	defer d.mu.Unlock()
	for key := range d.wildcards {
		if str.Is([]string{key}, eventName, false) {
			return true
		}
	}
	return false
}

// Push registers an event and payload to be fired later.
//
// It is Flush that fires it: a listener is registered against the event name
// with "_pushed" appended, and dispatching that name dispatches the real one.
func (d *Dispatcher) Push(event string, payload ...any) {
	d.Listen(event+"_pushed", func(...any) any {
		d.Dispatch(event, payload...)
		return nil
	})
}

// Flush fires the events pushed under the given name.
func (d *Dispatcher) Flush(event string) {
	d.Dispatch(event + "_pushed")
}

// Subscribe registers an event subscriber with the dispatcher.
//
// A string in the returned map is the name of a method on the subscriber;
// anything else is a listener. A subscriber is always the value itself, so
// there is nothing to resolve.
func (d *Dispatcher) Subscribe(subscriber Subscriber) {
	for event, listeners := range subscriber.Subscribe(d) {
		for _, listener := range listeners {
			if method, ok := listener.(string); ok && hasMethod(subscriber, method) {
				d.Listen(event, ClassListener{Class: subscriber, Method: method})
				continue
			}
			d.Listen(event, listener)
		}
	}
}

// Until fires an event and returns the first non-nil response, leaving the
// listeners behind it uncalled.
//
// It is Dispatch with halting turned on. Go has no default argument, so the two
// modes are two methods.
func (d *Dispatcher) Until(event any, payload ...any) any {
	return d.dispatch(event, payload, true)
}

// Dispatch fires an event and calls the listeners.
//
// It returns what each listener returned, in order. A listener that returns
// false stops the ones after it, and a listener that returns nothing
// contributes nil.
//
// When event is not a string it is the event itself: the name is its type and
// the payload is the value.
func (d *Dispatcher) Dispatch(event any, payload ...any) []any {
	responses, _ := d.dispatch(event, payload, false).([]any)
	return responses
}

// dispatch is the one body behind Dispatch and Until, because Go cannot default
// the halt argument.
func (d *Dispatcher) dispatch(event any, payload []any, halt bool) any {
	// A name is a string and an event is anything else.
	_, named := event.(string)
	isEventObject := !named

	parsedEvent, parsedPayload := parseEventAndPayload(event, payload)

	d.mu.Lock()
	deferring := d.deferringEvents && (d.eventsToDefer == nil || contains(d.eventsToDefer, parsedEvent))
	if deferring {
		d.deferredEvents = append(d.deferredEvents, deferredEvent{event: event, payload: payload, halt: halt})
		d.mu.Unlock()
		return nil
	}
	transactions := d.transactionManagerResolver
	d.mu.Unlock()

	// An event that is not meant to be dispatched unless the transaction
	// commits is hung on the transaction manager instead of being fired now.
	if isEventObject && len(parsedPayload) > 0 && transactions != nil {
		if after, ok := parsedPayload[0].(ShouldDispatchAfterCommit); ok && after.ShouldDispatchAfterCommit() {
			if manager := transactions(); manager != nil {
				manager.AddCallback(func() { d.invokeListeners(parsedEvent, parsedPayload, halt) })
				return nil
			}
		}
	}

	return d.invokeListeners(parsedEvent, parsedPayload, halt)
}

// invokeListeners calls every listener for the event, in order.
//
// It does not broadcast: an event that has to leave the process goes through
// the outbox, which is the one delivery this package owns.
func (d *Dispatcher) invokeListeners(event string, payload []any, halt bool) any {
	var responses []any

	for _, listener := range d.GetListeners(event) {
		response := listener(event, payload)

		// A response with halting enabled is the answer, and the listeners
		// after it are not called.
		if halt && response != nil {
			return response
		}

		// A listener that returns false stops the event travelling further
		// down the chain.
		if stop, ok := response.(bool); ok && !stop {
			break
		}

		responses = append(responses, response)
	}

	if halt {
		return nil
	}
	return responses
}

// parseEventAndPayload prepares the event and payload for dispatching.
func parseEventAndPayload(event any, payload []any) (string, []any) {
	if name, ok := event.(string); ok {
		return name, payload
	}
	return typeName(event), []any{event}
}

// GetListeners returns all of the listeners for a given event name, prepared.
//
// Listeners registered against an interface the event implements are not merged
// in: Go has no way to ask which interfaces a name implements, because an
// interface is structural and a name is not a type at run time.
func (d *Dispatcher) GetListeners(eventName string) []Listener {
	d.mu.Lock()
	defer d.mu.Unlock()

	prepared := make([]Listener, 0, len(d.listeners[eventName]))
	for _, listener := range d.listeners[eventName] {
		prepared = append(prepared, d.MakeListener(listener, false))
	}
	return append(prepared, d.getWildcardListeners(eventName)...)
}

// getWildcardListeners returns the pattern listeners for the event, from the
// cache when it is warm. The caller holds the lock.
func (d *Dispatcher) getWildcardListeners(eventName string) []Listener {
	if cached, ok := d.wildcardsCache[eventName]; ok {
		return cached
	}

	var wildcards []Listener
	for key, listeners := range d.wildcards {
		if !str.Is([]string{key}, eventName, false) {
			continue
		}
		for _, listener := range listeners {
			wildcards = append(wildcards, d.MakeListener(listener, true))
		}
	}
	d.wildcardsCache[eventName] = wildcards
	return wildcards
}

// GetRawListeners returns the raw, unprepared listeners.
//
// The map and its slices are copies: a caller that ranged over the live map
// while another goroutine registered a listener would race.
func (d *Dispatcher) GetRawListeners() map[string][]any {
	d.mu.Lock()
	defer d.mu.Unlock()

	out := make(map[string][]any, len(d.listeners))
	for event, listeners := range d.listeners {
		out[event] = append([]any(nil), listeners...)
	}
	return out
}

// MakeListener prepares a registered listener for calling.
//
// Go has no default argument, so the wildcard flag is always written out.
//
// A wildcard listener is handed the event name and the whole payload, because
// it answered a pattern and has to be told which event arrived. Every other
// listener is handed the payload spread across its parameters.
func (d *Dispatcher) MakeListener(listener any, wildcard bool) Listener {
	switch l := listener.(type) {
	case Listener:
		return l
	case func(event string, payload []any) any:
		return l
	case ClassListener:
		return d.CreateClassListener(l, wildcard)
	case *QueuedClosure:
		return l.Resolve(d)
	case string:
		// Listen refuses this already; MakeListener is exported, so it is the
		// other door onto the same rule. See assertListenable.
		panic(refusedClassName(l))
	}

	// A value that is not a function is a class listener: the object whose
	// Handle method answers the event.
	if listener != nil && reflect.TypeOf(listener).Kind() != reflect.Func {
		return d.CreateClassListener(listener, wildcard)
	}

	return func(event string, payload []any) any {
		if wildcard {
			return callFunc(listener, []any{event, payload})
		}
		return callFunc(listener, payload)
	}
}

// CreateClassListener creates a listener out of an object and a method.
//
// The default method is Handle, a listener that wants the queue is pushed onto
// it instead of being called, and a listener that waits for the transaction is
// hung on the transaction manager.
func (d *Dispatcher) CreateClassListener(listener any, wildcard bool) Listener {
	return func(event string, payload []any) any {
		callable := d.createClassCallable(listener)
		if wildcard {
			return callable([]any{event, payload})
		}
		return callable(payload)
	}
}

// createClassCallable turns a class listener into the function that runs it.
func (d *Dispatcher) createClassCallable(listener any) func(args []any) any {
	class, method := listener, ""
	if pair, ok := listener.(ClassListener); ok {
		class, method = pair.Class, pair.Method
	}
	// resolveMethod falls back to Invoke, which is the name a listener answers
	// by when the value itself is not callable. It returns the empty string
	// when the listener answers with nothing, which Listen and MakeListener
	// have already refused.
	method = resolveMethod(class, method)

	if d.handlerShouldBeQueued(class) {
		return func(args []any) any {
			return d.createQueuedHandlerCallable(class, method, args)
		}
	}

	if d.handlerShouldBeDispatchedAfterDatabaseTransactions(class) {
		return func(args []any) any {
			d.resolveTransactionManager().AddCallback(func() {
				callMethod(class, method, args)
			})
			return nil
		}
	}

	return func(args []any) any { return callMethod(class, method, args) }
}

// handlerShouldBeQueued reports whether the listener class goes on the queue.
func (d *Dispatcher) handlerShouldBeQueued(class any) bool {
	_, ok := class.(ShouldQueue)
	d.mu.Lock()
	resolver := d.queueResolver
	d.mu.Unlock()
	return ok && resolver != nil
}

// handlerShouldBeDispatchedAfterDatabaseTransactions reports whether the
// listener waits for the current transaction to commit.
func (d *Dispatcher) handlerShouldBeDispatchedAfterDatabaseTransactions(class any) bool {
	after, ok := class.(ShouldHandleEventsAfterCommit)
	return ok && after.ShouldHandleEventsAfterCommit() && d.resolveTransactionManager() != nil
}

// createQueuedHandlerCallable puts the listener on the queue.
//
// The arguments are copied first. It matters because the job is run later:
// without the copy,
//
//	e := &InvoicePaid{Amount: 100}
//	d.Dispatch(e)
//	e.Amount = 0
//
// left the worker looking at 0. An event is a value at the instant it was
// dispatched, and mutating it afterwards must not change what was dispatched.
func (d *Dispatcher) createQueuedHandlerCallable(class any, method string, args []any) any {
	args = cloneArguments(args)

	if wants, ok := class.(ShouldQueue); ok {
		var event any
		if len(args) > 0 {
			event = args[0]
		}
		if !wants.ShouldQueue(event) {
			return nil
		}
	}
	return d.queueHandler(class, method, args)
}

// cloneArguments copies the arguments a queued listener is called with.
//
// Only a pointer is copied: it is what a later mutation travels through, and
// everything else was already copied on its way into the any. The copy is
// shallow -- a pointer field inside the event still points where it did.
//
// A context.Context is left alone. It is not part of the event -- it is the
// ambient deadline and cancellation of the pass that dispatched -- and copying
// what it points at would produce a context whose parent no longer reaches it.
func cloneArguments(args []any) []any {
	if len(args) == 0 {
		return args
	}

	cloned := make([]any, len(args))
	for i, arg := range args {
		cloned[i] = cloneArgument(arg)
	}
	return cloned
}

// cloneArgument copies one argument. See cloneArguments.
func cloneArgument(arg any) any {
	if arg == nil || isContext(arg) {
		return arg
	}

	v := reflect.ValueOf(arg)
	if v.Kind() != reflect.Pointer || v.IsNil() {
		return arg
	}

	copied := reflect.New(v.Type().Elem())
	copied.Elem().Set(v.Elem())
	return copied.Interface()
}

// queueHandler pushes the listener's job onto the queue.
//
// It takes no unique lock: that would make this package depend on a cache, so
// the job carries what it knows about uniqueness and the queue is what enforces
// it.
func (d *Dispatcher) queueHandler(class any, method string, args []any) any {
	var event any
	if len(args) > 0 {
		event = args[0]
	}
	return d.push(d.propagateListenerOptions(class, event, NewCallQueuedListener(class, method, args)))
}

// push sends a job to the connection and queue it asked for.
//
// It is separate from queueHandler because QueuedClosure needs the same three
// lines, and there is no global helper for it to reach them through.
func (d *Dispatcher) push(job *CallQueuedListener) any {
	queue := d.resolveQueue()
	if queue == nil {
		return nil
	}

	connection := queue.Connection(job.Connection)
	if connection == nil {
		return nil
	}

	if job.Delay <= 0 {
		return errorOrNil(connection.PushOn(job.Queue, job))
	}
	return errorOrNil(connection.LaterOn(job.Queue, job.Delay, job))
}

// propagateListenerOptions copies what the listener declares onto the job.
//
// Each option is an optional interface, and every one of them takes the event.
// The two marker interfaces below stay written inline: they take no argument, so
// there is no signature for a listener to get wrong.
func (d *Dispatcher) propagateListenerOptions(listener, event any, job *CallQueuedListener) *CallQueuedListener {
	if v, ok := listener.(ViaConnection); ok {
		job.Connection = v.ViaConnection(event)
	}
	if v, ok := listener.(ViaQueue); ok {
		job.Queue = v.ViaQueue(event)
	}
	if v, ok := listener.(WithDelay); ok {
		job.Delay = v.WithDelay(event)
	}
	if v, ok := listener.(Tries); ok {
		job.Tries = v.Tries(event)
	}
	if v, ok := listener.(Backoff); ok {
		job.Backoff = v.Backoff(event)
	}
	if v, ok := listener.(RetryUntil); ok {
		job.RetryUntil = v.RetryUntil(event)
	}
	if v, ok := listener.(MessageGroup); ok {
		job.MessageGroup = v.MessageGroup(event)
	}
	if v, ok := listener.(Deduplicator); ok {
		job.WithDeduplicator(func() string { return v.Deduplicator(event) })
	}
	if v, ok := listener.(interface{ ShouldBeUnique() bool }); ok {
		job.shouldBeUnique = v.ShouldBeUnique()
	}
	if v, ok := listener.(interface{ ShouldBeUniqueUntilProcessing() bool }); ok {
		job.shouldBeUniqueUntilProcessing = v.ShouldBeUniqueUntilProcessing()
	}
	if job.shouldBeUnique {
		if v, ok := listener.(UniqueID); ok {
			job.uniqueID = v.UniqueID(event)
		}
		if v, ok := listener.(UniqueFor); ok {
			job.uniqueFor = v.UniqueFor(event)
		}
	}
	if v, ok := listener.(ShouldHandleEventsAfterCommit); ok {
		job.AfterCommit = v.ShouldHandleEventsAfterCommit()
	}
	return job
}

// Forget removes a set of listeners from the dispatcher.
func (d *Dispatcher) Forget(event string) {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.forget(event)
}

// forget is Forget with the lock already held.
func (d *Dispatcher) forget(event string) {
	if strings.Contains(event, "*") {
		delete(d.wildcards, event)
	} else {
		delete(d.listeners, event)
	}

	for key := range d.wildcardsCache {
		if str.Is([]string{event}, key, false) {
			delete(d.wildcardsCache, key)
		}
	}
}

// ForgetPushed forgets all of the pushed listeners.
func (d *Dispatcher) ForgetPushed() {
	d.mu.Lock()
	defer d.mu.Unlock()
	for key := range d.listeners {
		if strings.HasSuffix(key, "_pushed") {
			d.forget(key)
		}
	}
}

// resolveQueue returns the queue from the resolver, or nil when there is none.
func (d *Dispatcher) resolveQueue() Queue {
	d.mu.Lock()
	resolver := d.queueResolver
	d.mu.Unlock()
	if resolver == nil {
		return nil
	}
	return resolver()
}

// SetQueueResolver sets the queue implementation the dispatcher pushes through.
func (d *Dispatcher) SetQueueResolver(resolver func() Queue) *Dispatcher {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.queueResolver = resolver
	return d
}

// resolveTransactionManager returns the transaction manager from the resolver.
func (d *Dispatcher) resolveTransactionManager() TransactionManager {
	d.mu.Lock()
	resolver := d.transactionManagerResolver
	d.mu.Unlock()
	if resolver == nil {
		return nil
	}
	return resolver()
}

// SetTransactionManagerResolver sets the database transaction manager the
// dispatcher hangs after-commit work on.
//
// Leaving it unset is allowed: a missing resolver reads as no transaction
// manager, and an event or listener that asked to wait for a commit is then
// handled at once.
func (d *Dispatcher) SetTransactionManagerResolver(resolver func() TransactionManager) *Dispatcher {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.transactionManagerResolver = resolver
	return d
}

// Defer runs callback while holding events back, then dispatches them.
//
// Passing event names defers only those; passing none defers every event.
//
// A Go method cannot have a type parameter, so the callback's result is any.
//
// This is the one method on the dispatcher that is not safe to call from two
// goroutines at once: deferring is a mode the whole dispatcher is in, and two
// callers would take each other's events.
func (d *Dispatcher) Defer(callback func() any, events ...string) any {
	d.mu.Lock()
	wasDeferring := d.deferringEvents
	previousDeferredEvents := d.deferredEvents
	previousEventsToDefer := d.eventsToDefer

	d.deferringEvents = true
	d.deferredEvents = nil
	d.eventsToDefer = events
	d.mu.Unlock()

	defer func() {
		d.mu.Lock()
		d.deferringEvents = wasDeferring
		d.deferredEvents = previousDeferredEvents
		d.eventsToDefer = previousEventsToDefer
		d.mu.Unlock()
	}()

	result := callback()

	d.mu.Lock()
	d.deferringEvents = false
	held := d.deferredEvents
	d.mu.Unlock()

	for _, held := range held {
		d.dispatch(held.event, held.payload, held.halt)
	}

	return result
}

// Publish is where the outbox meets the registry, handing a stored event to the
// listeners registered for its name.
//
// This is where the mechanism and the surface meet: the outbox stores, the
// relay reads, and the listeners were registered with Listen. A Dispatcher is
// therefore a Publisher, and NewRelay takes one directly.
//
// The payload is the context and the row, in that order, so a listener reads
//
//	d.Listen("invoice.paid", func(ctx context.Context, e events.Stored) error { ... })
//
// A listener that returns an error fails the delivery, which is what puts the
// event back in the outbox for the next pass. Nothing else is treated as a
// failure: a listener that answers with a value is answering, not failing.
//
// Every listener runs, including the ones behind the one that failed, because
// Dispatch is what runs them and Dispatch does not stop for a response. The
// retry then runs all of them again, which is at-least-once arriving where it
// always arrives: a listener is idempotent on Stored.ID. A listener that does
// want to stop the ones behind it returns false.
func (d *Dispatcher) Publish(ctx context.Context, e Stored) error {
	for _, response := range d.Dispatch(e.Name, ctx, e) {
		if err, ok := response.(error); ok && err != nil {
			return err
		}
	}
	return nil
}

// eventNames reads the event names out of what Listen was given.
func eventNames(events any) []string {
	switch e := events.(type) {
	case string:
		return []string{e}
	case []string:
		return e
	case nil:
		return nil
	}
	return []string{typeName(events)}
}

// contains reports whether the list holds the value.
func contains(list []string, value string) bool {
	for _, item := range list {
		if item == value {
			return true
		}
	}
	return false
}

var contextType = reflect.TypeOf((*context.Context)(nil)).Elem()

// firstParameterTypes names the event a closure listens for, from the type of
// its first parameter.
//
// A leading context.Context is skipped: a listener that takes one still listens
// for what comes after it.
func firstParameterTypes(fn any) []string {
	if fn == nil {
		return nil
	}
	t := reflect.TypeOf(fn)
	if t.Kind() != reflect.Func || t.NumIn() == 0 {
		return nil
	}

	first := 0
	if t.In(0) == contextType {
		if t.NumIn() == 1 {
			return nil
		}
		first = 1
	}
	in := t.In(first)
	if in.Kind() == reflect.Interface || in.Kind() == reflect.Slice || in.Kind() == reflect.String {
		// A listener that takes any, a payload slice or a name is the prepared
		// shape rather than a typed event: it names no event of its own.
		return nil
	}
	return []string{typeNameOf(in)}
}

// typeName names an event by its import path and type name, which is the only
// globally unique naming Go offers.
func typeName(v any) string {
	if v == nil {
		return ""
	}
	return typeNameOf(reflect.TypeOf(v))
}

func typeNameOf(t reflect.Type) string {
	for t.Kind() == reflect.Pointer {
		t = t.Elem()
	}
	if t.PkgPath() == "" {
		return t.String()
	}
	return t.PkgPath() + "." + t.Name()
}

// isFunc reports whether the value is a function.
func isFunc(v any) bool {
	return v != nil && reflect.TypeOf(v).Kind() == reflect.Func
}

// assertListenable panics when the listener could never be called.
//
// A listener named by a string cannot be resolved, because a type is not
// addressable by name at run time. So
//
//	d.Listen("invoice.paid", "Notifier")
//
// registered a listener that was then quietly never called: MakeListener took
// the string for a class listener, looked for Handle and then Invoke on a
// string, found neither, and answered nil for every event for the life of the
// process. The same silence swallowed a value with no Handle on it and a
// ClassListener naming a method that is not there.
//
// Registering is the last moment at which the line that was wrong can still be
// named, so a listener that could never be called is refused here. Either it
// resolves or it refuses; what it must not do is accept and not deliver.
func assertListenable(listener any) {
	switch l := listener.(type) {
	case nil:
		panic("events: a nil listener would never be called")
	case string:
		panic(refusedClassName(l))
	case Listener, func(event string, payload []any) any, *QueuedClosure:
		return
	case ClassListener:
		if resolveMethod(l.Class, l.Method) == "" {
			panic(fmt.Sprintf(
				"events: the listener %s has no method %s and no Invoke, so it would never be called",
				typeName(l.Class), quotedMethod(l.Method),
			))
		}
		return
	}

	if isFunc(listener) {
		return
	}
	if resolveMethod(listener, "") == "" {
		panic(fmt.Sprintf(
			"events: the listener %s has no Handle and no Invoke, so it would never be called",
			typeName(listener),
		))
	}
}

// assertNamesAnEvent panics when a closure registered on its own names no event.
//
// A closure registers itself against the type of its first parameter; one with
// nothing to read would register against nothing and then go quietly uncalled,
// so it is refused here for the reason assertListenable refuses.
func assertNamesAnEvent(closure any, names []string) {
	if len(names) == 0 {
		panic(fmt.Sprintf(
			"events: the listener %s names no event: its first parameter is the event it listens for, so pass the event name as well",
			typeName(closure),
		))
	}
}

// refusedClassName is the refusal a class-name listener earns, in one place
// because Listen and MakeListener are two doors onto the same rule.
func refusedClassName(listener string) string {
	return fmt.Sprintf(
		"events: the listener [%s] is a class name, and there is no container to resolve one: register the listener value itself",
		listener,
	)
}

// quotedMethod names the method a ClassListener asked for, or Handle when it
// asked for none.
func quotedMethod(method string) string {
	if method == "" {
		return "Handle"
	}
	return method
}

// resolveMethod returns the method a class listener answers with, or the empty
// string when it answers with none.
//
// The default is Handle, and a listener without the named method falls back to
// Invoke.
func resolveMethod(class any, method string) string {
	if method == "" {
		method = "Handle"
	}
	if hasMethod(class, method) {
		return method
	}
	if hasMethod(class, "Invoke") {
		return "Invoke"
	}
	return ""
}

// hasMethod reports whether the value has an exported method of that name.
func hasMethod(v any, name string) bool {
	if v == nil || name == "" {
		return false
	}
	rv := reflect.ValueOf(v)
	if rv.MethodByName(name).IsValid() {
		return true
	}
	// A method with a pointer receiver is not on the value, and a listener
	// registered by value would silently stop answering.
	if rv.Kind() != reflect.Pointer && rv.CanAddr() {
		return rv.Addr().MethodByName(name).IsValid()
	}
	return false
}

// callMethod calls the named method with the payload spread across its
// parameters.
func callMethod(v any, name string, args []any) any {
	if v == nil {
		return nil
	}
	method := reflect.ValueOf(v).MethodByName(name)
	if !method.IsValid() {
		return nil
	}
	return callValue(method, args)
}

// callFunc calls a listener function with the payload spread across its
// parameters.
func callFunc(fn any, args []any) any {
	switch f := fn.(type) {
	case nil:
		return nil
	case func():
		f()
		return nil
	case func(...any):
		f(args...)
		return nil
	case func(...any) any:
		return f(args...)
	case func(...any) error:
		return errorOrNil(f(args...))
	}

	v := reflect.ValueOf(fn)
	if v.Kind() != reflect.Func {
		return nil
	}
	return callValue(v, args)
}

// callValue spreads args across the function's parameters and calls it.
//
// A reflect.Call with the wrong arity panics, so a missing argument is the zero
// value of its parameter and a surplus one is dropped: a mismatched listener
// stays quiet rather than taking the process down with it.
//
// One binding rule is not positional: a leading context.Context argument is
// dropped when the function does not take one. It is what lets the same
// listener be registered for an in-process dispatch and for an outbox delivery,
// which carries the context of the pass that read the row.
func callValue(v reflect.Value, args []any) any {
	t := v.Type()

	if len(args) > 0 && isContext(args[0]) && !(t.NumIn() > 0 && t.In(0) == contextType) {
		args = args[1:]
	}

	var in []reflect.Value
	if t.IsVariadic() {
		fixed := t.NumIn() - 1
		in = make([]reflect.Value, 0, max(fixed, len(args)))
		for i := range fixed {
			in = append(in, argumentFor(args, i, t.In(i)))
		}
		elem := t.In(fixed).Elem()
		for i := fixed; i < len(args); i++ {
			in = append(in, argumentFor(args, i, elem))
		}
	} else {
		in = make([]reflect.Value, 0, t.NumIn())
		for i := range t.NumIn() {
			in = append(in, argumentFor(args, i, t.In(i)))
		}
	}

	out := v.Call(in)
	if len(out) == 0 {
		return nil
	}
	return out[0].Interface()
}

// argumentFor prepares one argument for the parameter it is about to fill.
func argumentFor(args []any, i int, want reflect.Type) reflect.Value {
	if i >= len(args) || args[i] == nil {
		return reflect.Zero(want)
	}
	v := reflect.ValueOf(args[i])
	if v.Type().AssignableTo(want) {
		return v
	}
	return reflect.Zero(want)
}

func isContext(v any) bool {
	_, ok := v.(context.Context)
	return ok
}

// errorOrNil keeps a nil error out of the response list, where it would read as
// a listener that answered.
func errorOrNil(err error) any {
	if err == nil {
		return nil
	}
	return err
}
