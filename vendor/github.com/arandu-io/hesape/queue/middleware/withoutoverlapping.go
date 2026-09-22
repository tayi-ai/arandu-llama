package middleware

import (
	"context"
	"time"

	"github.com/arandu-io/hesape/cache"
	"github.com/arandu-io/hesape/queue/jobs"
)

// defaultLockExpiry is how long the overlap lock lives when nobody sized it.
//
// It matches the worker's default lease, because those two are the same
// duration seen from two sides: the lease is how long the job is invisible to
// other workers, and the lock is how long the work it does is protected. A
// lock shorter than the lease lets a second worker in while the first is still
// going, which is the thing this middleware exists to prevent.
const defaultLockExpiry = 5 * time.Minute

// WithoutOverlapping runs at most one job with a given key at a time.
//
// The usual key is the id of the thing being worked on -- an account, an
// invoice, a report --
// so two jobs about the same row never run at once while two jobs about
// different rows run in parallel.
//
//	m := middleware.NewWithoutOverlapping(locks, accountID).ReleaseAfter(10 * time.Second)
//
// A job that cannot take the lock is released and tries again later. Call
// [WithoutOverlapping.DontRelease] to drop it instead, which is right when the
// work is "make this true" rather than "do this once".
type WithoutOverlapping struct {
	locks        *cache.Locks
	key          string
	prefix       string
	shared       bool
	expiresAfter time.Duration
	releaseAfter time.Duration
	release      bool
}

var _ Middleware = (*WithoutOverlapping)(nil)

// NewWithoutOverlapping returns the middleware.
//
// An empty key means the job's name, which is the "one of these at a time"
// case.
func NewWithoutOverlapping(locks *cache.Locks, key string) *WithoutOverlapping {
	return &WithoutOverlapping{
		locks:        locks,
		key:          key,
		prefix:       "queue-overlap:",
		expiresAfter: defaultLockExpiry,
		release:      true,
	}
}

// ReleaseAfter sets how long a job waits before trying for the lock again.
func (m *WithoutOverlapping) ReleaseAfter(d time.Duration) *WithoutOverlapping {
	m.releaseAfter = d
	return m
}

// DontRelease drops the job instead of putting it back when the lock is taken.
func (m *WithoutOverlapping) DontRelease() *WithoutOverlapping {
	m.release = false
	return m
}

// ExpireAfter sets the lock's ttl, which is the deadlock protection: a worker
// that dies holding it releases it when the ttl passes.
func (m *WithoutOverlapping) ExpireAfter(d time.Duration) *WithoutOverlapping {
	m.expiresAfter = d
	return m
}

// WithPrefix names the lock's namespace.
func (m *WithoutOverlapping) WithPrefix(prefix string) *WithoutOverlapping {
	m.prefix = prefix
	return m
}

// Shared makes jobs with different names share the lock, so long as they share
// the key.
func (m *WithoutOverlapping) Shared() *WithoutOverlapping {
	m.shared = true
	return m
}

// GetLockKey is the name of the lock this job takes.
func (m *WithoutOverlapping) GetLockKey(j *jobs.Job) string {
	if m.shared {
		return m.prefix + j.TenantID + ":" + m.key
	}
	return m.prefix + key(j, "") + ":" + m.key
}

// Handle takes the lock, runs the job, and gives the lock back.
func (m *WithoutOverlapping) Handle(ctx context.Context, j *jobs.Job, next func(context.Context) error) error {
	lock := m.locks.Lock(m.GetLockKey(j), m.expiresAfter)

	taken, err := lock.Get(ctx, next)
	if err != nil {
		return err
	}
	if taken {
		return nil
	}
	if !m.release {
		return nil
	}
	return j.Release(ctx, m.releaseAfter)
}
