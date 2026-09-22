package middleware

import (
	"context"

	"github.com/arandu-io/hesape/bus"
	"github.com/arandu-io/hesape/queue/jobs"
)

// SkipIfBatchCancelled drops a job whose batch was cancelled.
//
// Cancelling a batch cannot recall the jobs already on the queue, so they are
// delivered,
// they ask, and they skip -- which is what makes cancelling mean anything.
//
// A job that belongs to no batch is handed on and the store is never consulted,
// so this middleware is safe to put on a worker that also runs jobs pushed on
// their own.
type SkipIfBatchCancelled struct {
	batches bus.BatchRepository
}

var _ Middleware = (*SkipIfBatchCancelled)(nil)

// NewSkipIfBatchCancelled returns the middleware over the repository the
// batches live in.
func NewSkipIfBatchCancelled(batches bus.BatchRepository) *SkipIfBatchCancelled {
	return &SkipIfBatchCancelled{batches: batches}
}

// Handle asks the batch, and skips the job if it was cancelled.
func (m *SkipIfBatchCancelled) Handle(ctx context.Context, j *jobs.Job, next func(context.Context) error) error {
	batchable, err := bus.Batched(j.Payload, nil)
	if err != nil {
		// A payload the bus did not write is a job in no batch. It runs.
		return next(ctx)
	}

	worth, err := batchable.Batching(ctx, jobs.GrantFor(j), m.batches)
	if err != nil {
		return err
	}
	if !worth {
		return nil
	}
	return next(ctx)
}
