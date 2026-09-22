package queue

import (
	"context"
	"fmt"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/queue/jobs"
)

// NullQueue accepts every job and keeps none of them.
//
// It is what a test that does not care about the queue wires, and what the sync
// connection's registry-only worker is built over -- see [SyncQueue].
//
// It is a value and not a pointer, because it has no state and a nil pointer
// that silently swallows every job is a worse mistake than the one this type
// exists to make cheap.
type NullQueue struct{}

var (
	_ Queue       = NullQueue{}
	_ jobs.Driver = NullQueue{}
)

// GetConnectionName is "null".
//
// It answers getConnectionName() without the embedded connection struct the
// other drivers use, because this one is a value with no state and giving it a
// settable name would be the one field on it.
func (NullQueue) GetConnectionName() string { return "null" }

// Push discards the job.
func (NullQueue) Push(context.Context, auth.Grant, jobs.Job) error { return nil }

// PushRaw discards the job. It answers pushRaw().
func (NullQueue) PushRaw(context.Context, auth.Grant, string, []byte, string) error { return nil }

// DelayedSize is zero. It answers delayedSize().
func (NullQueue) DelayedSize(context.Context, string) (int, error) { return 0, nil }

// ReservedSize is zero. It answers reservedSize().
func (NullQueue) ReservedSize(context.Context, string) (int, error) { return 0, nil }

// PushOn discards the job.
func (NullQueue) PushOn(context.Context, auth.Grant, string, jobs.Job) error { return nil }

// Later discards the job.
func (NullQueue) Later(context.Context, auth.Grant, time.Duration, jobs.Job) error { return nil }

// Bulk discards the jobs.
func (NullQueue) Bulk(context.Context, auth.Grant, []jobs.Job) error { return nil }

// Pop returns nothing.
func (NullQueue) Pop(context.Context, string, int, time.Duration) ([]*jobs.Job, error) {
	return nil, nil
}

// Size is zero.
func (NullQueue) Size(context.Context, string) (int, error) { return 0, nil }

// PendingSize is zero.
func (NullQueue) PendingSize(context.Context, string) (int, error) { return 0, nil }

// CreationTimeOfOldestPendingJob is the zero time.
func (NullQueue) CreationTimeOfOldestPendingJob(context.Context, string) (time.Time, error) {
	return time.Time{}, nil
}

// Clear removes nothing.
func (NullQueue) Clear(context.Context, string) (int, error) { return 0, nil }

// Failed lists nothing.
func (NullQueue) Failed(context.Context, int) ([]jobs.Job, error) { return nil, nil }

// Retry has nothing to retry, and says so.
//
// This is the one place a null driver does not answer with nil. Nil means "the
// job is back in line", and a caller holding a dead letter record acts on that
// by forgetting the record -- so a silent success here is the one way to lose
// work through a queue that was never meant to hold any.
func (NullQueue) Retry(_ context.Context, uuid string) error {
	return fmt.Errorf("%w: %s", ErrNotParked, uuid)
}

// ReleaseJob does nothing.
func (NullQueue) ReleaseJob(context.Context, *jobs.Job, time.Duration) error { return nil }

// DeleteJob does nothing.
func (NullQueue) DeleteJob(context.Context, *jobs.Job) error { return nil }

// FailJob does nothing.
func (NullQueue) FailJob(context.Context, *jobs.Job, error) error { return nil }
