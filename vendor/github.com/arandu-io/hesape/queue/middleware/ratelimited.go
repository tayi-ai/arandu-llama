package middleware

import (
	"context"
	"time"

	"github.com/arandu-io/hesape/cache"
	"github.com/arandu-io/hesape/queue/jobs"
)

// retryPadding is added to the wait a released job is given.
//
// It is there so a job does not come back at the exact instant the window rolls
// and spend the first attempt of the next one racing every other job that was
// released with it.
const retryPadding = 3 * time.Second

// RateLimited runs a job only while the budget for it lasts.
//
// The job over budget is released and comes back when the window rolls, so the
// work is delayed rather than dropped.
//
//	m := middleware.NewRateLimited(limiter, cache.PerMinute(30))
//
// The limit's key is scoped by tenant, so one customer cannot spend another's
// budget. An empty key means the job's name, which is the "thirty of these a
// minute, per customer" case.
//
// There is no RESP-specific variant: the limiter counts in whatever cache.Store
// it was built over.
type RateLimited struct {
	limiter      *cache.RateLimiter
	limit        cache.Limit
	releaseAfter time.Duration
	release      bool
}

var _ Middleware = (*RateLimited)(nil)

// NewRateLimited returns the middleware.
func NewRateLimited(limiter *cache.RateLimiter, limit cache.Limit) *RateLimited {
	return &RateLimited{limiter: limiter, limit: limit, release: true}
}

// ReleaseAfter fixes how long a job over budget waits, instead of asking the
// limiter when the window rolls.
func (m *RateLimited) ReleaseAfter(d time.Duration) *RateLimited {
	m.releaseAfter = d
	return m
}

// DontRelease drops the job over budget instead of putting it back.
func (m *RateLimited) DontRelease() *RateLimited {
	m.release = false
	return m
}

// Handle spends one attempt, and runs the job if it fit.
func (m *RateLimited) Handle(ctx context.Context, j *jobs.Job, next func(context.Context) error) error {
	limit := m.limit.By(key(j, m.limit.Key))

	// A limiter that cannot answer refuses rather than letting the job
	// through. The HTTP throttle makes the opposite call -- a rate limiter that
	// is down must not become an outage -- and the difference is what is on the
	// other side: a request that is refused is one person waiting, and a job
	// that runs unlimited is the third-party API the limit was protecting.
	res, err := m.limiter.Attempt(ctx, limit)
	if err != nil {
		return err
	}
	if res.OK {
		return next(ctx)
	}
	if !m.release {
		return nil
	}

	wait := m.releaseAfter
	if wait <= 0 {
		wait = res.RetryAfter + retryPadding
	}
	return j.Release(ctx, wait)
}
