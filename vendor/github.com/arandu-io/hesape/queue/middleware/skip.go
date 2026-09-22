package middleware

import (
	"context"

	"github.com/arandu-io/hesape/queue/jobs"
)

// Skip drops a job without running it.
//
// The job is not released and not failed: the worker reads a middleware that
// returned without touching the job as "handled", so the job is deleted.
// Skipping means the work is not wanted, not that it should be tried later --
// for later, release it.
//
//	w := queue.NewWorker(q, queue.WorkerOptions{
//		Middleware: []middleware.Middleware{middleware.Skip{}.Unless(featureOn)},
//	})
//
// [Skip.When] and [Skip.Unless] are methods on the zero value, so
// `middleware.Skip{}.When(cond)` reads as one phrase.
type Skip struct {
	skip bool
}

var _ Middleware = Skip{}

// When skips the job when cond is true.
func (Skip) When(cond bool) Skip { return Skip{skip: cond} }

// Unless skips the job unless cond is true.
func (Skip) Unless(cond bool) Skip { return Skip{skip: !cond} }

// Handle drops the job, or hands it on.
func (m Skip) Handle(ctx context.Context, _ *jobs.Job, next func(context.Context) error) error {
	if m.skip {
		return nil
	}
	return next(ctx)
}
