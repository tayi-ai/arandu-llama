package queue

import (
	"context"
	"fmt"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/queue/jobs"
)

// Handler does the work.
//
// The Grant is rebuilt from the job's tenant and action, so a handler reaches
// repositories the same way a service does. There is no unauthorized path into
// the database from a worker, which is the whole point of the Grant existing.
//
// The record names the work and something has to turn that name into code: the
// name is a string and the registry is the [Worker], because Go cannot reach a
// type from a name in a string.
type Handler interface {
	Handle(ctx context.Context, g auth.Grant, j *jobs.Job) error
}

// HandlerFunc adapts a function to Handler.
type HandlerFunc func(ctx context.Context, g auth.Grant, j *jobs.Job) error

// Handle calls f.
func (f HandlerFunc) Handle(ctx context.Context, g auth.Grant, j *jobs.Job) error {
	return f(ctx, g, j)
}

// Queue is what a driver implements.
//
// Every push takes an auth.Grant, so the tenant comes from the Grant and not
// from an argument somebody can get wrong, and Pop takes a count and a lease
// rather than returning one job at a time -- a worker with a concurrency of
// four asks for four.
//
// Pop rather than a channel of jobs: the job has to stay in the store,
// invisible to other workers, until it is settled. A worker that dies mid-job
// must not lose it, and that is not something a channel can offer.
type Queue interface {
	// Push adds a job to the queue named on it.
	//
	// Inside database.Transaction the DatabaseQueue joins it, which is the
	// property that driver exists for: the job is committed by the same
	// transaction as the row it describes, so it cannot refer to a write that
	// rolled back.
	Push(ctx context.Context, g auth.Grant, j jobs.Job) error

	// PushOn adds a job to a named queue, ignoring the one on the job.
	PushOn(ctx context.Context, g auth.Grant, queue string, j jobs.Job) error

	// Later adds a job that becomes eligible after delay.
	Later(ctx context.Context, g auth.Grant, delay time.Duration, j jobs.Job) error

	// Bulk adds many jobs. A driver that can do it in one round trip does; the
	// contract only promises that either all of them arrive or an error says
	// otherwise.
	Bulk(ctx context.Context, g auth.Grant, js []jobs.Job) error

	// Pop takes up to n jobs off a queue and hides them for the lease.
	//
	// Jobs whose lease expires become visible again -- which is what makes a
	// worker crash recoverable and delivery at-least-once.
	//
	// The returned jobs carry Attempts INCLUDING this delivery: a job handed
	// over for the first time has Attempts == 1. A driver that returns the
	// count from before the delivery makes the worker park a job one attempt
	// early, and with a try limit of 2 it parks on the first failure and never
	// retries at all.
	//
	// Each job carries the queue it came off, so the worker settles it by
	// calling Release, Delete or Fail on the job rather than on the driver.
	Pop(ctx context.Context, queue string, n int, lease time.Duration) ([]*jobs.Job, error)

	// Size is how many jobs the queue holds, waiting or in flight.
	Size(ctx context.Context, queue string) (int, error)

	// PendingSize is how many jobs are waiting. It feeds the health check: a
	// queue that only grows is a worker that is not running.
	//
	// A job whose lease is still valid is running, not waiting.
	PendingSize(ctx context.Context, queue string) (int, error)

	// CreationTimeOfOldestPendingJob is when the oldest waiting job became
	// eligible, or the zero time when nothing is waiting.
	//
	// A stopped worker looks exactly like an idle one, and this is what tells
	// them apart. A job scheduled for the future returns a time in the future,
	// so a caller measuring lag with time.Since gets a negative number and can
	// treat it as no lag at all.
	CreationTimeOfOldestPendingJob(ctx context.Context, queue string) (time.Time, error)

	// Clear removes every job on a queue and returns how many went.
	Clear(ctx context.Context, queue string) (int, error)

	// Failed lists the jobs that gave up, most recent failure first.
	Failed(ctx context.Context, limit int) ([]jobs.Job, error)

	// Retry puts a failed job back in line with its attempts reset.
	//
	// Without it the only way out of a dead letter queue is SQL by hand, which
	// is how it becomes a table nobody touches.
	Retry(ctx context.Context, uuid string) error
}

// ErrNoConnection is returned when a connection nobody registered is asked for.
var ErrNoConnection = fmt.Errorf("queue: no such connection")

// Dispatcher is the two things the queue asks of an event dispatcher.
//
// It is declared here rather than imported so the queue does not depend on the
// events package to run, and so a test can watch what a worker announced with
// eight lines and no wiring. events.Dispatcher satisfies it.
//
// Until is on it and not just Dispatch because of one listener: a Looping
// listener that returns false stops the worker taking work, and halting on the
// first answer is the only way to hear it.
type Dispatcher interface {
	// Dispatch sends an event to every listener and returns what they returned.
	Dispatch(event any, payload ...any) []any
	// Until sends an event and stops at the first listener that answers
	// something other than nil.
	Until(event any, payload ...any) any
}
