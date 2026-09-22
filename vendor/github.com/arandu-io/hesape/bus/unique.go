package bus

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/cache"
)

// UniqueJob is a job that must not be queued twice while one is still in
// flight.
//
// The case it exists for is the one every application meets: a button that
// recalculates a report, pressed four times
// because the first press showed no feedback. Four jobs, four identical
// reports, four times the work -- and none of it wrong enough to notice.
type UniqueJob struct {
	// Key is what "the same job" means. It is per tenant already, because the
	// cache namespaces every key by the tenant on the Grant, so it should name
	// the work and not the customer: "report.monthly:2026-08".
	Key string
	// TTL is how long the claim lasts if nothing releases it. It is required,
	// and it is the deadlock protection: a worker that dies holding a claim
	// blocks that job until the claim expires and there is no other way out.
	// Size it above the longest run of the job.
	TTL time.Duration
	// Queue, Name and Payload are the job itself.
	Queue   string
	Name    string
	Payload any
}

// ErrNoKey is returned when a unique job has no key to be unique by.
var ErrNoKey = errors.New("bus: a unique job needs a key, and an empty one names every job at once")

// PushUnique pushes the job unless one with the same key is already in flight,
// and reports whether it pushed.
//
// False is not an error. "Somebody already queued this" is the answer the
// caller asked for, and a handler that treats it as a failure turns the fourth
// button press into a red banner.
//
// The claim is taken before the push and given back by ReleaseUnique, which the
// handler calls when the work is done. It is an atomic add in the cache rather
// than a cache.Lock, and that is not a second kind of lock: a cache.Lock is held
// by a handle in one process's memory, and the process that releases this claim
// is not the one that took it. What crosses that gap is a key that is either
// there or not.
//
// If the claim is taken and the push then fails, the claim is given back, so a
// queue that was briefly down does not silence the job for a whole TTL.
func PushUnique(ctx context.Context, g auth.Grant, c *cache.Repository, q Queue, j UniqueJob) (bool, error) {
	if j.Key == "" {
		return false, ErrNoKey
	}
	if j.Name == "" {
		return false, ErrNoName
	}
	if j.TTL <= 0 {
		return false, fmt.Errorf("%w: the unique job %q has none, and a worker that dies holding the claim would silence it forever", cache.ErrNoTTL, j.Key)
	}
	if c == nil || q == nil {
		return false, fmt.Errorf("bus: pushing the unique job %q needs a cache and a queue", j.Key)
	}

	payload, err := encode(j.Name, j.Payload)
	if err != nil {
		return false, err
	}

	taken, err := c.Add(ctx, g, uniqueKey(j.Key), j.Name, j.TTL)
	if err != nil {
		return false, err
	}
	if !taken {
		return false, nil
	}

	if err := q.Push(ctx, g, j.Queue, j.Name, payload); err != nil {
		// The release runs on a context that is not cancelled with the
		// caller's: a request abandoned at exactly this moment would otherwise
		// leave the claim standing with no job behind it.
		_ = ReleaseUnique(context.WithoutCancel(ctx), g, c, j.Key)
		return false, fmt.Errorf("bus: pushing the unique job %s: %w", j.Name, err)
	}
	return true, nil
}

// ReleaseUnique gives the claim back, so the job can be queued again.
//
// The handler calls it when the work is done, on the failing path as much as on
// the succeeding one -- a job that failed and left its claim standing is a job
// nobody can retry until the TTL runs out. Releasing a claim that has already
// expired is not an error: the caller wanted it gone and it is.
func ReleaseUnique(ctx context.Context, g auth.Grant, c *cache.Repository, key string) error {
	if key == "" {
		return ErrNoKey
	}
	if c == nil {
		return errors.New("bus: releasing a unique job needs a cache")
	}
	return c.Forget(ctx, g, uniqueKey(key))
}

// uniqueKey namespaces the claim, so it cannot collide with an application's
// own cache entry of the same name.
func uniqueKey(key string) string { return UniqueLock{}.GetKey(UniqueJob{Key: key}) }

// UniqueLock is the claim a unique job holds while it is in flight.
//
// PushUnique is Acquire and a push in one call, and is what an application
// uses; this is the pair underneath, for the handler that has to take or give
// back a claim on its own -- a job released early because the work turned out
// to be a no-op, a claim taken by a scheduler before it decides what to queue.
type UniqueLock struct {
	cache *cache.Repository
}

// NewUniqueLock returns the lock over a cache.
func NewUniqueLock(c *cache.Repository) UniqueLock { return UniqueLock{cache: c} }

// Acquire takes the claim, and reports whether it got it.
//
// False is not an error: "somebody already queued this" is the answer the
// caller asked for.
//
// It is an atomic add in the cache rather than a cache.Lock, and that is not a
// second kind of lock: a cache.Lock is held by a handle in one process's
// memory, and the process that releases this claim is not the one that took it.
// What crosses that gap is a key that is either there or not.
func (u UniqueLock) Acquire(ctx context.Context, g auth.Grant, j UniqueJob) (bool, error) {
	if j.Key == "" {
		return false, ErrNoKey
	}
	if j.TTL <= 0 {
		return false, fmt.Errorf("%w: the unique job %q has none, and a worker that dies holding the claim would silence it forever", cache.ErrNoTTL, j.Key)
	}
	if u.cache == nil {
		return false, fmt.Errorf("bus: acquiring the claim on %q needs a cache", j.Key)
	}
	return u.cache.Add(ctx, g, u.GetKey(j), j.Name, j.TTL)
}

// Release gives the claim back, so the job can be queued again.
//
// Releasing a claim that has already expired is not an error: the caller wanted
// it gone and it is.
func (u UniqueLock) Release(ctx context.Context, g auth.Grant, j UniqueJob) error {
	if j.Key == "" {
		return ErrNoKey
	}
	if u.cache == nil {
		return errors.New("bus: releasing a claim needs a cache")
	}
	return u.cache.Forget(ctx, g, u.GetKey(j))
}

// GetKey is the cache key a job's claim is held under.
//
// It is namespaced so that a claim cannot collide with an application's own
// cache entry of the same name. It is already per tenant without saying so:
// the cache prefixes every key with the tenant on the Grant.
func (UniqueLock) GetKey(j UniqueJob) string { return "bus:unique:" + j.Key }
