package queue

import (
	"context"
	"fmt"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/queue/jobs"
)

// DeferredQueue runs each job after the response has been sent, in this
// process.
//
// It is [SyncQueue] with the work put off: nothing is stored, so nothing is
// retried and a restart loses whatever was in flight -- but the request does
// not wait for the work, which is the one thing SyncQueue cannot offer.
//
// What it is for is the work that is genuinely optional: a webhook nobody
// waits for, a cache warm. Anything that must survive a deploy belongs on
// [DatabaseQueue], and the difference between the two is exactly "may this be
// lost".
//
// It is a distinct type rather than an option on SyncQueue because the choice
// is a wiring decision made once, and an option would be a second way to spell
// a connection name.
type DeferredQueue struct {
	connection
	sync *SyncQueue
	// deferrer is what runs the callback later. It is a field so a test can
	// make "later" mean "now".
	deferrer func(func())
}

// NewDeferredQueue returns the queue over the same handler registry a worker
// uses.
//
// The callback runs on its own goroutine, which is what "after the response"
// means here: the handler that pushed has already returned by the time the
// response is written.
func NewDeferredQueue(h Handlers) *DeferredQueue {
	q := &DeferredQueue{sync: NewSyncQueue(h), deferrer: func(fn func()) { go fn() }}
	q.SetConnectionName("deferred")
	return q
}

// DeferUsing replaces how the work is put off, and returns the queue so the
// call chains.
//
// It is what a test calls with func(fn func()) { fn() } to make the queue
// synchronous, which is the only way to assert on what a deferred job did
// without sleeping.
func (q *DeferredQueue) DeferUsing(defer_ func(func())) *DeferredQueue {
	q.deferrer = defer_
	return q
}

var (
	_ Queue       = (*DeferredQueue)(nil)
	_ jobs.Driver = (*DeferredQueue)(nil)
)

// Push runs the job after the caller has moved on.
//
// The job is authorized before the deferral, not inside it: an unauthorized
// push has to fail the caller, and a failure that happens on another goroutine
// after the response is a failure nobody sees.
func (q *DeferredQueue) Push(ctx context.Context, g auth.Grant, j jobs.Job) error {
	if err := jobs.Authorized(g, j); err != nil {
		return err
	}
	j = jobs.Prepare(g, j)

	// The context is deliberately not the caller's: it is cancelled when the
	// response is written, and every deferred job would be cancelled with it.
	q.deferrer(func() { _ = q.sync.Push(context.WithoutCancel(ctx), g, j) })
	return nil
}

// PushOn runs the job on a named queue, after the caller has moved on.
func (q *DeferredQueue) PushOn(ctx context.Context, g auth.Grant, queue string, j jobs.Job) error {
	j.Queue = queue
	return q.Push(ctx, g, j)
}

// PushRaw runs a job whose arguments are already encoded. It answers pushRaw().
func (q *DeferredQueue) PushRaw(ctx context.Context, g auth.Grant, name string, payload []byte, queue string) error {
	j, err := jobs.New(g, queue, name, nil)
	if err != nil {
		return err
	}
	j.Payload = payload
	return q.Push(ctx, g, j)
}

// Later runs the job after the caller has moved on, and after delay.
//
// The delay is honoured here where [SyncQueue] drops it, because the work is
// already off the request's path and sleeping costs nobody anything.
func (q *DeferredQueue) Later(ctx context.Context, g auth.Grant, delay time.Duration, j jobs.Job) error {
	if err := jobs.Authorized(g, j); err != nil {
		return err
	}
	j = jobs.Prepare(g, j)

	uncancelled := context.WithoutCancel(ctx)
	q.deferrer(func() {
		if delay > 0 {
			timer := time.NewTimer(delay)
			defer timer.Stop()
			<-timer.C
		}
		_ = q.sync.Push(uncancelled, g, j)
	})
	return nil
}

// Bulk defers each job in order.
func (q *DeferredQueue) Bulk(ctx context.Context, g auth.Grant, js []jobs.Job) error {
	for _, j := range js {
		if err := q.Push(ctx, g, j); err != nil {
			return err
		}
	}
	return nil
}

// Pop returns nothing: a deferred job runs in this process, not off a queue.
func (q *DeferredQueue) Pop(context.Context, string, int, time.Duration) ([]*jobs.Job, error) {
	return nil, nil
}

// Size is zero. Nothing is stored.
func (q *DeferredQueue) Size(context.Context, string) (int, error) { return 0, nil }

// PendingSize is zero.
func (q *DeferredQueue) PendingSize(context.Context, string) (int, error) { return 0, nil }

// DelayedSize is zero. It answers delayedSize().
func (q *DeferredQueue) DelayedSize(context.Context, string) (int, error) { return 0, nil }

// ReservedSize is zero. It answers reservedSize().
func (q *DeferredQueue) ReservedSize(context.Context, string) (int, error) { return 0, nil }

// CreationTimeOfOldestPendingJob is the zero time: nothing waits.
func (q *DeferredQueue) CreationTimeOfOldestPendingJob(context.Context, string) (time.Time, error) {
	return time.Time{}, nil
}

// Clear removes nothing.
func (q *DeferredQueue) Clear(context.Context, string) (int, error) { return 0, nil }

// Failed lists nothing. A deferred job that failed failed on a goroutine
// nobody is holding, and its error went to the log.
func (q *DeferredQueue) Failed(context.Context, int) ([]jobs.Job, error) { return nil, nil }

// Retry has nothing to retry: the job ran on a goroutine nobody is holding, and
// this driver kept nothing to put back.
func (q *DeferredQueue) Retry(_ context.Context, uuid string) error {
	return fmt.Errorf("%w: %s", ErrNotParked, uuid)
}

// ReleaseJob does nothing. There is no queue to put the job back on.
func (q *DeferredQueue) ReleaseJob(context.Context, *jobs.Job, time.Duration) error { return nil }

// DeleteJob does nothing.
func (q *DeferredQueue) DeleteJob(context.Context, *jobs.Job) error { return nil }

// FailJob does nothing: nobody is left to hand the error to.
func (q *DeferredQueue) FailJob(context.Context, *jobs.Job, error) error { return nil }
