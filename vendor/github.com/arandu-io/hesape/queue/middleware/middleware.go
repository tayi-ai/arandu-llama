package middleware

import (
	"context"

	"github.com/arandu-io/hesape/queue/jobs"
)

// Middleware wraps the handling of one job.
//
// Call next to hand the job on. Not calling it is how a middleware says the
// work should not happen: release the job to try again later, delete it to drop
// it, or return without touching it -- which the worker reads as "handled", and
// the job is deleted.
type Middleware interface {
	Handle(ctx context.Context, j *jobs.Job, next func(context.Context) error) error
}

// Func adapts a function to Middleware.
type Func func(ctx context.Context, j *jobs.Job, next func(context.Context) error) error

// Handle calls f.
func (f Func) Handle(ctx context.Context, j *jobs.Job, next func(context.Context) error) error {
	return f(ctx, j, next)
}

// key names a per-job resource -- a lock, a counter -- inside its tenant.
//
// The tenant is first and it is not optional: a lock named after the job alone
// would let one customer's import block every other customer's, and a
// counter named after the job alone would let one customer spend everybody's
// budget.
func key(j *jobs.Job, name string) string {
	if name == "" {
		name = j.Name
	}
	return j.TenantID + ":" + name
}
