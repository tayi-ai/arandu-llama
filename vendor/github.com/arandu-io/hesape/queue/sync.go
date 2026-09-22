package queue

import (
	"context"
	"fmt"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/queue/jobs"
)

// Handlers resolves a job name to the handler that runs it.
//
// [Worker] implements it, and it is declared as an interface so [SyncQueue] can
// borrow the worker's registry without the two constructors having to be called
// in a circle.
type Handlers interface {
	Handler(name string) (Handler, bool)
}

// SyncQueue runs each job the moment it is pushed, in the caller's goroutine.
//
// It exists for two reasons: a test that pushes a job wants to assert on what
// the job did without starting a worker, and a developer running the
// application locally wants the email to be sent rather than to sit in a
// table.
//
// Nothing is stored, so nothing is retried: a handler that fails fails the push.
// That is the trade the sync connection makes, and it is why it is not a
// production driver.
//
// It borrows the worker's handler registry rather than keeping one of its own,
// so a job name is registered once:
//
//	w := queue.NewWorker(queue.NullQueue{}, queue.WorkerOptions{})
//	w.HandleFunc("invoice.send", sendInvoice)
//	q := queue.NewSyncQueue(w)
//
// The worker over a NullQueue is the registry with nothing to drain, which is
// exactly what the sync connection is.
type SyncQueue struct {
	connection
	handlers Handlers
}

// NewSyncQueue returns the queue.
func NewSyncQueue(h Handlers) *SyncQueue {
	q := &SyncQueue{handlers: h}
	q.SetConnectionName(syncConnection)
	return q
}

var (
	_ Queue       = (*SyncQueue)(nil)
	_ jobs.Driver = (*SyncQueue)(nil)
)

const syncConnection = "sync"

// Push runs the job.
func (q *SyncQueue) Push(ctx context.Context, g auth.Grant, j jobs.Job) error {
	if err := jobs.Authorized(g, j); err != nil {
		return err
	}
	j = jobs.Prepare(g, j)
	j.Attempts = 1

	handler, known := q.handlers.Handler(j.Name)
	if !known {
		return fmt.Errorf("queue: no handler registered for %s", j.Name)
	}
	popped := jobs.Popped(q, syncConnection, j)
	return handler.Handle(ctx, jobs.GrantFor(popped), popped)
}

// PushRaw runs a job whose arguments are already encoded. It answers pushRaw().
func (q *SyncQueue) PushRaw(ctx context.Context, g auth.Grant, name string, payload []byte, queue string) error {
	j, err := jobs.New(g, queue, name, nil)
	if err != nil {
		return err
	}
	j.Payload = payload
	return q.Push(ctx, g, j)
}

// DelayedSize is zero: a sync job has already run. It answers delayedSize().
func (q *SyncQueue) DelayedSize(context.Context, string) (int, error) { return 0, nil }

// ReservedSize is zero. It answers reservedSize().
func (q *SyncQueue) ReservedSize(context.Context, string) (int, error) { return 0, nil }

// PushOn runs the job, recording the queue it was meant for.
func (q *SyncQueue) PushOn(ctx context.Context, g auth.Grant, queue string, j jobs.Job) error {
	j.Queue = queue
	return q.Push(ctx, g, j)
}

// Later runs the job now.
//
// The delay is dropped rather than slept through: the sync connection is what a
// test and a laptop use, and neither wants the process to stop for an hour
// because the job was scheduled for one.
func (q *SyncQueue) Later(ctx context.Context, g auth.Grant, _ time.Duration, j jobs.Job) error {
	return q.Push(ctx, g, j)
}

// Bulk runs the jobs in order, stopping at the first failure.
func (q *SyncQueue) Bulk(ctx context.Context, g auth.Grant, js []jobs.Job) error {
	for _, j := range js {
		if err := q.Push(ctx, g, j); err != nil {
			return err
		}
	}
	return nil
}

// Pop returns nothing: a sync job has already run by the time Push returns.
func (q *SyncQueue) Pop(context.Context, string, int, time.Duration) ([]*jobs.Job, error) {
	return nil, nil
}

// Size is zero. Nothing is ever stored.
func (q *SyncQueue) Size(context.Context, string) (int, error) { return 0, nil }

// PendingSize is zero.
func (q *SyncQueue) PendingSize(context.Context, string) (int, error) { return 0, nil }

// CreationTimeOfOldestPendingJob is the zero time: nothing waits.
func (q *SyncQueue) CreationTimeOfOldestPendingJob(context.Context, string) (time.Time, error) {
	return time.Time{}, nil
}

// Clear removes nothing.
func (q *SyncQueue) Clear(context.Context, string) (int, error) { return 0, nil }

// Failed lists nothing: a job that failed here failed the push, and the caller
// already has the error.
func (q *SyncQueue) Failed(context.Context, int) ([]jobs.Job, error) { return nil, nil }

// Retry has nothing to retry: this driver keeps no job to put back.
func (q *SyncQueue) Retry(_ context.Context, uuid string) error {
	return fmt.Errorf("%w: %s", ErrNotParked, uuid)
}

// ReleaseJob does nothing. There is no queue to put the job back on, and a
// handler that releases a sync job is told so by the error it gets back from
// its own Push.
func (q *SyncQueue) ReleaseJob(context.Context, *jobs.Job, time.Duration) error { return nil }

// DeleteJob does nothing.
func (q *SyncQueue) DeleteJob(context.Context, *jobs.Job) error { return nil }

// FailJob does nothing: the error travels back out of Push.
func (q *SyncQueue) FailJob(context.Context, *jobs.Job, error) error { return nil }
