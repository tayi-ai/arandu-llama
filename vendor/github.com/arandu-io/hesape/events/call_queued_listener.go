package events

import "time"

// CallQueuedListener is the job that runs a listener on the queue.
//
// The dispatcher builds one whenever a listener implements ShouldQueue, and the
// worker calls Handle. Everything the listener declared about how it wants to
// be run -- the connection, the queue, the tries, the delay -- is copied onto
// the job before it is pushed, because the worker has the job and not the
// listener.
type CallQueuedListener struct {
	// Class is the listener value itself: a type is not addressable by name at
	// run time, so what travels is the value.
	Class any
	// Method is the listener method to call.
	Method string
	// Data is what the listener is called with.
	Data []any

	// Connection, Queue, Delay, MessageGroup and AfterCommit are where the job
	// goes and when a worker may take it.
	Connection   string
	Queue        string
	Delay        time.Duration
	MessageGroup string
	AfterCommit  bool

	// Tries is how many times the job may be attempted.
	Tries int
	// MaxExceptions is the ceiling on exceptions, whatever the attempts.
	MaxExceptions int
	// Backoff is how long to wait before retrying after an uncaught failure.
	Backoff time.Duration
	// RetryUntil is when the job stops being retried.
	RetryUntil time.Time
	// Timeout is how long the job may run.
	Timeout time.Duration
	// FailOnTimeout says the job fails rather than retries when it times out.
	FailOnTimeout bool
	// ShouldBeEncrypted says the payload is encrypted on the way to the queue.
	ShouldBeEncrypted bool
	// DeleteWhenMissingModels deletes the job when its records are gone.
	DeleteWhenMissingModels bool

	// The four below are unexported because Go cannot have a field and a method
	// of the same name, and the methods are the surface. The dispatcher fills
	// them in this package.
	shouldBeUnique                bool
	shouldBeUniqueUntilProcessing bool
	uniqueID                      any
	uniqueFor                     time.Duration

	deduplicator func() string
}

// NewCallQueuedListener creates the job that will call method on class with
// data.
func NewCallQueuedListener(class any, method string, data []any) *CallQueuedListener {
	return &CallQueuedListener{Class: class, Method: method, Data: data}
}

// Handle calls the listener's method with the data.
//
// It takes nothing: the job holds the listener value itself, so there is
// nothing to resolve and nothing to deserialize.
func (j *CallQueuedListener) Handle() any {
	return callMethod(j.Class, j.Method, j.Data)
}

// ShouldBeUnique reports whether the listener should be unique.
func (j *CallQueuedListener) ShouldBeUnique() bool { return j.shouldBeUnique }

// ShouldBeUniqueUntilProcessing reports whether the listener should be unique
// only until processing begins.
func (j *CallQueuedListener) ShouldBeUniqueUntilProcessing() bool {
	return j.shouldBeUniqueUntilProcessing
}

// UniqueID returns the unique ID for the listener.
func (j *CallQueuedListener) UniqueID() any { return j.uniqueID }

// UniqueFor returns how long the unique lock is held.
func (j *CallQueuedListener) UniqueFor() time.Duration { return j.uniqueFor }

// UniqueVia returns the cache store the unique lock is kept in, or nil when the
// listener does not name one.
//
// The listener's own UniqueVia is called with no arguments, because the
// optional interface that asks for it has to fix one signature.
//
// The result is untyped because this package must not depend on the cache: the
// worker that acquires the lock is the only thing that reads it, and it knows
// the concrete store.
func (j *CallQueuedListener) UniqueVia() any {
	via, ok := j.Class.(interface{ UniqueVia() any })
	if !ok {
		return nil
	}
	return via.UniqueVia()
}

// WithDeduplicator sets the callback that generates the deduplication ID.
//
// The callback is kept as it was given: nothing about this job is serialized.
func (j *CallQueuedListener) WithDeduplicator(deduplicator func() string) *CallQueuedListener {
	j.deduplicator = deduplicator
	return j
}

// Deduplicator returns the deduplication ID for the queue connection to read,
// or the empty string when the job has no deduplicator.
//
// It is a method rather than a field because Go cannot have both under one
// name, which is the same reason the four uniqueness settings are methods.
func (j *CallQueuedListener) Deduplicator() string {
	if j.deduplicator == nil {
		return ""
	}
	return j.deduplicator()
}

// Failed calls the listener's own Failed method, with the event and the
// failure, when the listener has one.
func (j *CallQueuedListener) Failed(err error) {
	if !hasMethod(j.Class, "Failed") {
		return
	}
	callMethod(j.Class, "Failed", append(append([]any(nil), j.Data...), err))
}

// DisplayName is the display name for the queued job: the type of the listener
// value, which is the nearest thing to a name a type has at run time.
func (j *CallQueuedListener) DisplayName() string { return typeName(j.Class) }
