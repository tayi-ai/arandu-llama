package events

import (
	"time"

	"github.com/arandu-io/hesape/queue/jobs"
)

// WorkerStopReason is why a worker's loop ended.
//
// It is declared here, and re-exported from the queue package, because
// [WorkerStopping] carries it and the queue package is the one that dispatches
// these events -- a type in the queue package would make this package import it
// back.
type WorkerStopReason string

const (
	// Interrupted is a signal: SIGTERM from an orchestrator, SIGINT from a
	// person. The worker finished the job it had and then stopped.
	Interrupted WorkerStopReason = "interrupted"
	// MaxJobsExceeded is the worker's job limit reached.
	MaxJobsExceeded WorkerStopReason = "max_jobs"
	// MaxMemoryExceeded is the worker's memory limit reached.
	MaxMemoryExceeded WorkerStopReason = "memory"
	// MaxTimeExceeded is the worker's time limit reached.
	MaxTimeExceeded WorkerStopReason = "max_time"
	// QueueEmpty is a worker asked to stop when the queue ran dry.
	QueueEmpty WorkerStopReason = "empty"
	// ReceivedRestartSignal is `aru queue:restart`.
	ReceivedRestartSignal WorkerStopReason = "restart_signal"
)

// String is the wire value.
func (r WorkerStopReason) String() string { return string(r) }

// JobQueueing is dispatched before a job is written to its store.
//
// It is the only hook that runs while the push can still be stopped, which is
// what an audit listener wants.
type JobQueueing struct {
	// ConnectionName is the queue connection the job is going to.
	ConnectionName string
	// Queue is the queue it is going onto.
	Queue string
	// Job is the record about to be written.
	Job *jobs.Job
	// Delay is how long until it becomes eligible, and zero means now.
	Delay time.Duration
}

// Payload is the job's arguments, as they were encoded.
//
// The envelope is the record's own fields, and this is the half a handler
// decodes.
func (e JobQueueing) Payload() []byte {
	if e.Job == nil {
		return nil
	}
	return e.Job.Payload
}

// JobQueued is dispatched after a job has been written to its store.
type JobQueued struct {
	// ConnectionName is the queue connection the job went to.
	ConnectionName string
	// Queue is the queue it went onto.
	Queue string
	// ID is the job's identifier in the store. It is always the job's UUID: the
	// id is minted by the application rather than handed back by the store.
	ID string
	// Job is the record that was written.
	Job *jobs.Job
	// Delay is how long until it becomes eligible, and zero means now.
	Delay time.Duration
}

// Payload is the job's arguments, as they were encoded.
func (e JobQueued) Payload() []byte {
	if e.Job == nil {
		return nil
	}
	return e.Job.Payload
}

// JobPopping is dispatched before a worker asks its queue for work.
type JobPopping struct {
	// ConnectionName is the connection being asked.
	ConnectionName string
	// Queue is the queue being drained.
	Queue string
}

// JobPopped is dispatched for each job a worker took off its queue.
type JobPopped struct {
	// ConnectionName is the connection it came off.
	ConnectionName string
	// Job is the job. It is never nil: an empty batch dispatches nothing.
	Job *jobs.Job
}

// JobProcessing is dispatched before a handler runs.
type JobProcessing struct {
	// ConnectionName is the connection the job came off.
	ConnectionName string
	// Job is the job about to run.
	Job *jobs.Job
}

// JobProcessed is dispatched after a handler finished without an error.
type JobProcessed struct {
	// ConnectionName is the connection the job came off.
	ConnectionName string
	// Job is the job that ran.
	Job *jobs.Job
}

// JobAttempted is dispatched after every delivery, whether it worked or not.
//
// It is the one event a listener can use to count deliveries without
// subscribing to two.
type JobAttempted struct {
	// ConnectionName is the connection the job came off.
	ConnectionName string
	// Job is the job that was delivered.
	Job *jobs.Job
	// Exception is why it failed, and nil when it did not.
	Exception error
}

// Successful reports whether the delivery ended without an error.
//
// A released job reports false: one that a middleware put back has not
// succeeded, it has been postponed.
func (e JobAttempted) Successful() bool {
	return e.Exception == nil && (e.Job == nil || !e.Job.IsReleased())
}

// JobExceptionOccurred is dispatched when a handler returned an error, before
// anything has been decided about the job.
type JobExceptionOccurred struct {
	// ConnectionName is the connection the job came off.
	ConnectionName string
	// Job is the job that failed.
	Job *jobs.Job
	// Exception is why.
	Exception error
}

// JobReleasedAfterException is dispatched when a failed job was put back on its
// queue for another try.
//
// It is the counterpart of [JobFailed]: every failed delivery ends in exactly
// one of the two.
type JobReleasedAfterException struct {
	// ConnectionName is the connection the job came off.
	ConnectionName string
	// Job is the job that was released.
	Job *jobs.Job
	// Delay is how long until it is eligible again.
	Delay time.Duration
}

// JobFailed is dispatched when a job gave up and was parked.
//
// This is the one an application alerts on: a job here will not run again until
// somebody retries it.
type JobFailed struct {
	// ConnectionName is the connection the job came off.
	ConnectionName string
	// Job is the job that was parked.
	Job *jobs.Job
	// Exception is why it gave up.
	Exception error
}

// JobTimedOut is dispatched when a handler ran past its timeout.
type JobTimedOut struct {
	// ConnectionName is the connection the job came off.
	ConnectionName string
	// Job is the job that ran too long.
	Job *jobs.Job
}

// JobRetryRequested is dispatched when a person put a failed job back in line.
type JobRetryRequested struct {
	// Job is the job that was retried.
	Job *jobs.Job
}

// Payload is the retried job's arguments.
func (e JobRetryRequested) Payload() []byte {
	if e.Job == nil {
		return nil
	}
	return e.Job.Payload
}

// Looping is dispatched at the top of every pass of a worker's loop.
//
// A listener that returns false stops the worker taking work on this pass,
// which is how an application pauses its own queues without a cache.
type Looping struct {
	// ConnectionName is the connection being drained.
	ConnectionName string
	// Queue is the queue being drained.
	Queue string
}

// QueueBusy is dispatched by `aru queue:monitor` when a queue is over its
// threshold.
//
// It is the event an application turns into a page: a queue that only grows is
// a worker that is not running.
type QueueBusy struct {
	// ConnectionName is the connection.
	ConnectionName string
	// Queue is the queue that is behind.
	Queue string
	// Size is how many jobs are waiting.
	Size int
}

// QueuePaused is dispatched when a queue was paused.
type QueuePaused struct {
	// ConnectionName is the connection.
	ConnectionName string
	// Queue is the queue that was paused.
	Queue string
	// TTL is how long the pause lasts, and zero means until it is resumed.
	TTL time.Duration
}

// QueueResumed is dispatched when a paused queue was resumed.
type QueueResumed struct {
	// ConnectionName is the connection.
	ConnectionName string
	// Queue is the queue that was resumed.
	Queue string
}

// QueueFailedOver is dispatched when a failover queue moved on to the next
// connection.
//
// It fires once per connection that went down, not once per job: a broker that
// is refusing everything would otherwise produce one event per push.
type QueueFailedOver struct {
	// ConnectionName is the connection that failed.
	ConnectionName string
	// Job is the job that could not be pushed onto it.
	Job *jobs.Job
	// Exception is why it failed.
	Exception error
}

// WorkerStarting is dispatched once, before a worker's first pass.
type WorkerStarting struct {
	// ConnectionName is the connection being drained.
	ConnectionName string
	// Queue is the queue being drained.
	Queue string
	// WorkerOptions is the configuration the worker is running under.
	//
	// It is untyped because the options are queue.WorkerOptions and the queue
	// package is what dispatches this event: naming the type here would make
	// this package import the one that imports it. A listener that wants the
	// fields asserts to queue.WorkerOptions, which is one line and is the only
	// thing it can be.
	WorkerOptions any
}

// WorkerStopping is dispatched once, after a worker's loop ended.
//
// Status and Reason are what a supervisor reads to decide whether to start it
// again.
type WorkerStopping struct {
	// Status is the exit status the process will return.
	Status int
	// WorkerOptions is the configuration the worker was running under. It is
	// untyped for the reason given on [WorkerStarting].
	WorkerOptions any
	// Reason is why the loop ended, and empty when nobody said.
	Reason WorkerStopReason
}
