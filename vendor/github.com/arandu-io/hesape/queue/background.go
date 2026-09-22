package queue

import (
	"context"
	"fmt"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/queue/jobs"
)

// BackgroundQueue runs each job in a separate operating system process.
//
// Work that must not block the request usually belongs on [DeferredQueue] --
// which is this without the process. What is left for this driver is the case
// a goroutine cannot serve: a job that must survive the request's process
// crashing, or one that has to be isolated from it.
//
// It is a wrapper over a command rather than a fork, because Go cannot fork:
// Run is `aru queue:work --once` or whatever the application names it, and the
// job is handed over on its argument list. An application that has not set Run
// gets an error at push, not silence.
type BackgroundQueue struct {
	connection
	// run starts the job in another process and returns once it has started.
	run func(ctx context.Context, j jobs.Job) error
}

// NewBackgroundQueue returns the queue over a way to start a process.
//
// run is given the job and must start a process for it and return; it must not
// wait for the work, because the point of this driver is that the caller does
// not.
func NewBackgroundQueue(run func(ctx context.Context, j jobs.Job) error) *BackgroundQueue {
	q := &BackgroundQueue{run: run}
	q.SetConnectionName("background")
	return q
}

var (
	_ Queue       = (*BackgroundQueue)(nil)
	_ jobs.Driver = (*BackgroundQueue)(nil)
)

// ErrNoBackgroundRunner is returned when a background queue was built without a
// way to start a process.
var ErrNoBackgroundRunner = errNoBackgroundRunner{}

type errNoBackgroundRunner struct{}

func (errNoBackgroundRunner) Error() string {
	return "queue: this background queue has no way to start a process. Pass one to NewBackgroundQueue"
}

// Push starts the job in another process.
//
// The job is authorized first, for the reason [DeferredQueue.Push] gives: a
// refusal has to reach the caller.
func (q *BackgroundQueue) Push(ctx context.Context, g auth.Grant, j jobs.Job) error {
	if err := jobs.Authorized(g, j); err != nil {
		return err
	}
	if q.run == nil {
		return ErrNoBackgroundRunner
	}
	return q.run(ctx, jobs.Prepare(g, j))
}

// PushOn starts the job on a named queue.
func (q *BackgroundQueue) PushOn(ctx context.Context, g auth.Grant, queue string, j jobs.Job) error {
	j.Queue = queue
	return q.Push(ctx, g, j)
}

// PushRaw starts a job whose arguments are already encoded. It answers
// pushRaw().
func (q *BackgroundQueue) PushRaw(ctx context.Context, g auth.Grant, name string, payload []byte, queue string) error {
	j, err := jobs.New(g, queue, name, nil)
	if err != nil {
		return err
	}
	j.Payload = payload
	return q.Push(ctx, g, j)
}

// Later starts the job with its RunAt set, and leaves the waiting to the
// process that runs it.
//
// Sleeping here would hold the caller for the delay, which is the one thing
// this driver exists to avoid.
func (q *BackgroundQueue) Later(ctx context.Context, g auth.Grant, delay time.Duration, j jobs.Job) error {
	j.RunAt = time.Now().UTC().Add(delay)
	return q.Push(ctx, g, j)
}

// Bulk starts each job in order, stopping at the first failure to start.
func (q *BackgroundQueue) Bulk(ctx context.Context, g auth.Grant, js []jobs.Job) error {
	for _, j := range js {
		if err := q.Push(ctx, g, j); err != nil {
			return err
		}
	}
	return nil
}

// Pop returns nothing: the job is already in another process.
func (q *BackgroundQueue) Pop(context.Context, string, int, time.Duration) ([]*jobs.Job, error) {
	return nil, nil
}

// Size is zero. Nothing is stored.
func (q *BackgroundQueue) Size(context.Context, string) (int, error) { return 0, nil }

// PendingSize is zero.
func (q *BackgroundQueue) PendingSize(context.Context, string) (int, error) { return 0, nil }

// DelayedSize is zero. It answers delayedSize().
func (q *BackgroundQueue) DelayedSize(context.Context, string) (int, error) { return 0, nil }

// ReservedSize is zero. It answers reservedSize().
func (q *BackgroundQueue) ReservedSize(context.Context, string) (int, error) { return 0, nil }

// CreationTimeOfOldestPendingJob is the zero time: nothing waits.
func (q *BackgroundQueue) CreationTimeOfOldestPendingJob(context.Context, string) (time.Time, error) {
	return time.Time{}, nil
}

// Clear removes nothing.
func (q *BackgroundQueue) Clear(context.Context, string) (int, error) { return 0, nil }

// Failed lists nothing: a job that failed failed in another process, and its
// error went to that process's log.
func (q *BackgroundQueue) Failed(context.Context, int) ([]jobs.Job, error) { return nil, nil }

// Retry has nothing to retry: the job ran in another process, and this one kept
// nothing to put back.
func (q *BackgroundQueue) Retry(_ context.Context, uuid string) error {
	return fmt.Errorf("%w: %s", ErrNotParked, uuid)
}

// ReleaseJob does nothing. There is no queue to put the job back on.
func (q *BackgroundQueue) ReleaseJob(context.Context, *jobs.Job, time.Duration) error { return nil }

// DeleteJob does nothing.
func (q *BackgroundQueue) DeleteJob(context.Context, *jobs.Job) error { return nil }

// FailJob does nothing.
func (q *BackgroundQueue) FailJob(context.Context, *jobs.Job, error) error { return nil }
