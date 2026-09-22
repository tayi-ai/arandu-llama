package queue

import "github.com/arandu-io/hesape/queue/events"

// WorkerStopReason is why a worker's loop ended.
//
// It is a string type, so its wire value is what a process supervisor greps
// for.
//
// It exists so the thing that restarts the process can tell the two kinds of
// exit apart: a worker that stopped because it hit its job limit is healthy and
// should be started again, and one that stopped because it ran out of memory is
// a bug somebody should see.
//
// It is an alias for the type declared in [github.com/arandu-io/hesape/queue/events],
// because events.WorkerStopping carries it and this package is the one that
// dispatches that event.
type WorkerStopReason = events.WorkerStopReason

const (
	// Interrupted is a signal: SIGTERM from an orchestrator, SIGINT from a
	// person. The worker finished the job it had and then stopped, which is
	// what makes a deploy not lose work.
	Interrupted = events.Interrupted

	// MaxJobsExceeded is WorkerOptions.MaxJobs reached. A worker that stops
	// after n jobs is how a leak in a handler stays survivable.
	MaxJobsExceeded = events.MaxJobsExceeded

	// MaxMemoryExceeded is WorkerOptions.Memory reached.
	MaxMemoryExceeded = events.MaxMemoryExceeded

	// MaxTimeExceeded is WorkerOptions.MaxTime reached.
	MaxTimeExceeded = events.MaxTimeExceeded

	// QueueEmpty is WorkerOptions.StopWhenEmpty with nothing left to do. It is
	// what a batch job in a pipeline waits for.
	QueueEmpty = events.QueueEmpty

	// ReceivedRestartSignal is `aru queue:restart`: a timestamp went into the
	// cache and every worker that reads it stops, so the next deploy's binary
	// is the one running.
	ReceivedRestartSignal = events.ReceivedRestartSignal
)

// Exit statuses a worker returns, and what a supervisor reads.
//
// Twelve for memory is not one of the shell's own statuses, so a supervisor can
// tell "the worker decided to stop" from "something killed it".
const (
	// ExitSuccess is a worker that stopped because it was asked to.
	ExitSuccess = 0
	// ExitError is a worker that stopped because it could not continue.
	ExitError = 1
	// ExitMemoryLimit is a worker that stopped because it was using too much
	// memory. A supervisor that restarts on this one is doing the right thing.
	ExitMemoryLimit = 12
)
