package middleware

import (
	"context"

	"github.com/arandu-io/hesape/queue/jobs"
)

// FailOnException parks a job on a failure that retrying cannot fix.
//
// The default schedule retries everything the same number of times, which is
// right for a timeout and
// wrong for a payload that will never parse: five deliveries of a job that
// cannot succeed is five copies of the same error, an hour apart.
//
//	m := middleware.NewFailOnException(func(err error) bool {
//		return errors.Is(err, database.ErrNotFound)
//	})
//
// The failure is still returned to the worker, so the log line and the recorded
// job are the same as any other failure. The job is already parked by then, and
// the worker does not park it twice.
type FailOnException struct {
	when func(error) bool
}

var _ Middleware = (*FailOnException)(nil)

// NewFailOnException returns the middleware.
//
// The set of failures it parks on is errors.Is inside the predicate, written
// where the compiler can read it.
func NewFailOnException(when func(error) bool) *FailOnException {
	return &FailOnException{when: when}
}

// Handle runs the job and parks it on a failure the predicate claims.
func (m *FailOnException) Handle(ctx context.Context, j *jobs.Job, next func(context.Context) error) error {
	err := next(ctx)
	if err == nil {
		return nil
	}
	if m.when != nil && m.when(err) {
		if failErr := j.Fail(ctx, err); failErr != nil {
			return failErr
		}
	}
	return err
}
