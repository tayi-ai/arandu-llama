package middleware

import (
	"context"
	"time"

	"github.com/arandu-io/hesape/cache"
	"github.com/arandu-io/hesape/queue/jobs"
)

// ThrottlesExceptions stops hammering a dependency that is already failing.
//
// It counts failures rather than attempts: once a job has failed more than the
// limit
// allows inside the window, the ones that follow are released without being
// run at all, so a third-party API that is down gets a pause instead of the
// whole queue retrying against it.
//
//	m := middleware.NewThrottlesExceptions(limiter, cache.PerMinute(10)).
//		RetryAfter(5 * time.Minute)
//
// A failure it catches is not returned to the worker: the job is released and
// tries again, which is the point -- the worker's own attempt counter is for
// jobs that are wrong, and this is for dependencies that are down.
type ThrottlesExceptions struct {
	limiter    *cache.RateLimiter
	limit      cache.Limit
	retryAfter time.Duration
	backoff    time.Duration
	byJob      bool
	by         string
	prefix     string
	when       func(error) bool
	deleteWhen func(error) bool
	failWhen   func(error) bool
	report     func(error) bool
}

var _ Middleware = (*ThrottlesExceptions)(nil)

// NewThrottlesExceptions returns the middleware.
func NewThrottlesExceptions(limiter *cache.RateLimiter, limit cache.Limit) *ThrottlesExceptions {
	return &ThrottlesExceptions{limiter: limiter, limit: limit, retryAfter: time.Minute}
}

// RetryAfter is how long a job waits after a failure this middleware caught.
func (m *ThrottlesExceptions) RetryAfter(d time.Duration) *ThrottlesExceptions {
	m.retryAfter = d
	return m
}

// ByJob counts each job separately instead of counting the whole name
// together.
//
// It answers byJob(). Use it when the dependency is per-record rather than
// shared: one broken invoice should not throttle the other nine hundred.
func (m *ThrottlesExceptions) ByJob() *ThrottlesExceptions {
	m.byJob = true
	return m
}

// By counts against a key of your own instead of the job's name.
//
// It answers by(). Use it when the thing that is failing is neither the job nor
// the record but something they share: an account, a region, a tenant of the
// third party.
func (m *ThrottlesExceptions) By(key string) *ThrottlesExceptions {
	m.by = key
	return m
}

// WithPrefix puts a prefix in front of the counter's key.
//
// It answers withPrefix(). Two middlewares over the same job name would
// otherwise share one counter, and the failures of one would throttle the
// other.
func (m *ThrottlesExceptions) WithPrefix(prefix string) *ThrottlesExceptions {
	m.prefix = prefix
	return m
}

// Backoff is how long the jobs that arrive while the limit is spent wait before
// they are eligible again.
//
// It answers backoff(). It is the pause on the closed circuit, where RetryAfter
// is the wait after the failure that closed it: the first says how long the
// dependency gets to recover, the second how soon the job that noticed tries
// again.
func (m *ThrottlesExceptions) Backoff(d time.Duration) *ThrottlesExceptions {
	m.backoff = d
	return m
}

// When narrows which failures are counted. A failure it says no to is returned
// to the worker untouched, which parks the job on the usual schedule.
func (m *ThrottlesExceptions) When(f func(error) bool) *ThrottlesExceptions {
	m.when = f
	return m
}

// DeleteWhen drops the job instead of releasing it, for the failures the
// predicate claims.
//
// It answers deleteWhen(). It is for the failure that means the work is
// pointless rather than early: the record was deleted while the job waited, and
// releasing it would put it back for four more rounds of the same answer.
func (m *ThrottlesExceptions) DeleteWhen(f func(error) bool) *ThrottlesExceptions {
	m.deleteWhen = f
	return m
}

// FailWhen parks the job instead of releasing it, for the failures the
// predicate claims.
//
// It answers failWhen(). It is DeleteWhen's louder sibling: the work is
// pointless and somebody should see that it was.
func (m *ThrottlesExceptions) FailWhen(f func(error) bool) *ThrottlesExceptions {
	m.failWhen = f
	return m
}

// Report narrows which failures are worth reporting.
//
// A dependency that is down produces one failure per job, and reporting every
// one of them is how the report becomes noise: the predicate is what says "only
// the first", or "only the ones that are not timeouts". A nil predicate reports
// everything.
//
// It does not report anything itself. What it decides is what
// [ThrottlesExceptions.ShouldReport] answers, and the caller's error reporter
// is what acts on it -- this package has no business knowing where a report
// goes.
func (m *ThrottlesExceptions) Report(f func(error) bool) *ThrottlesExceptions {
	m.report = f
	return m
}

// ShouldReport reports whether a failure is worth reporting.
//
// It is the read half of [ThrottlesExceptions.Report]. This package does not
// report anything itself, so the caller asks and acts.
func (m *ThrottlesExceptions) ShouldReport(err error) bool {
	return m.report == nil || m.report(err)
}

// Handle refuses while the failures are piling up, and counts the ones it sees.
func (m *ThrottlesExceptions) Handle(ctx context.Context, j *jobs.Job, next func(context.Context) error) error {
	name := m.limit.Key
	switch {
	case m.by != "":
		name = m.by
	case m.byJob:
		name = j.UUID
	}
	if m.prefix != "" {
		name = m.prefix + ":" + name
	}
	limit := m.limit.By(key(j, name))

	// Reading the counter without spending it: Attempt counts and answers, and
	// Release gives the count straight back. cache.RateLimiter has no method
	// for asking without spending, and adding one for a caller in this package
	// would be a second way to ask a counter a question.
	//
	// The window this counter measures is failures, not deliveries, which is
	// why nothing is spent here and Hit is called below only when something
	// actually failed.
	res, err := m.limiter.Attempt(ctx, limit)
	if err != nil {
		return err
	}
	if err := m.limiter.Release(ctx, limit); err != nil {
		return err
	}
	if !res.OK {
		wait := res.RetryAfter + retryPadding
		if m.backoff > 0 {
			wait = m.backoff
		}
		return j.Release(ctx, wait)
	}

	runErr := next(ctx)
	if runErr == nil {
		return nil
	}
	// Delete and fail are checked before the counter: a failure that means the
	// work is pointless is not evidence
	// that the dependency is down, and counting it would throttle jobs that
	// would have succeeded.
	if m.deleteWhen != nil && m.deleteWhen(runErr) {
		return j.Delete(ctx)
	}
	if m.failWhen != nil && m.failWhen(runErr) {
		return j.Fail(ctx, runErr)
	}
	if m.when != nil && !m.when(runErr) {
		return runErr
	}
	if _, err := m.limiter.Hit(ctx, limit); err != nil {
		return err
	}
	return j.Release(ctx, m.retryAfter)
}
