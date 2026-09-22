package events

import "time"

// QueuedClosure is a closure listener that runs on the queue.
//
//	d.Listen("invoice.paid", events.Queueable(func(e Stored) {
//		// ...
//	}).OnQueue("billing").Delay(10*time.Second))
//
// Every setter returns the receiver, so the calls chain.
type QueuedClosure struct {
	// Closure is the underlying function. It is exported because the dispatcher
	// reads it to work out which event the closure listens for.
	Closure any

	// The rest are unexported because Go cannot have both a field and a method
	// of the same name, and the methods are the surface.
	connection   string
	queue        string
	messageGroup string
	deduplicator func() string
	delay        time.Duration

	catchCallbacks []any
}

// Queueable creates a queued closure event listener.
func Queueable(closure any) *QueuedClosure {
	return &QueuedClosure{Closure: closure}
}

// OnConnection sets the desired connection for the job.
func (q *QueuedClosure) OnConnection(connection string) *QueuedClosure {
	q.connection = connection
	return q
}

// OnQueue sets the desired queue for the job.
func (q *QueuedClosure) OnQueue(queue string) *QueuedClosure {
	q.queue = queue
	return q
}

// OnGroup sets the desired job "group". Only some queues support it.
func (q *QueuedClosure) OnGroup(group string) *QueuedClosure {
	q.messageGroup = group
	return q
}

// WithDeduplicator sets the callback that generates the deduplication ID.
//
// The callback is kept as it was given: a job here is a value the worker
// already has the type of, so nothing about it is serialized.
func (q *QueuedClosure) WithDeduplicator(deduplicator func() string) *QueuedClosure {
	q.deduplicator = deduplicator
	return q
}

// Delay sets how long the job waits before it becomes available.
func (q *QueuedClosure) Delay(delay time.Duration) *QueuedClosure {
	q.delay = delay
	return q
}

// Catch registers a callback invoked if the queued listener job fails.
func (q *QueuedClosure) Catch(closure any) *QueuedClosure {
	q.catchCallbacks = append(q.catchCallbacks, closure)
	return q
}

// Resolve returns the listener that puts the closure on the queue.
//
// The queue comes from the dispatcher the listener was registered on, which is
// why Resolve is handed it: there is no global helper to reach one through.
func (q *QueuedClosure) Resolve(d *Dispatcher) Listener {
	return func(_ string, payload []any) any {
		job := NewCallQueuedListener(InvokeQueuedClosure{}, "Handle", []any{
			q.Closure,
			payload,
			q.catchCallbacks,
		})
		job.Connection = q.connection
		job.Queue = q.queue
		job.Delay = q.delay
		job.MessageGroup = q.messageGroup
		job.WithDeduplicator(q.deduplicator)

		return d.push(job)
	}
}
