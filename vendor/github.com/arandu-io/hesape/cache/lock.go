package cache

import (
	"context"
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"time"
)

// defaultSleepBetweenBlockedAttempts is how long Block waits before trying the
// lock again.
const defaultSleepBetweenBlockedAttempts = 250 * time.Millisecond

// Locks issues distributed locks over a store.
//
// It exists for the scheduler: with N replicas, a task scheduled every minute
// runs N times a minute unless exactly one of them wins a lock first. Same for
// the outbox relay, which would otherwise publish every event N times, and for
// a console command that must not run twice.
//
// This is the only kind of lock in the collection: one issuer, one handle, one
// key per name. Nothing here declares an interface for it, and that is not an
// omission -- Go satisfies an interface by shape, so a caller that wants the
// lock behind one declares it where it is used, and a declaration here would be
// a second name for the same thing with nothing keeping the two in step.
//
// It is not a consensus lock. It is correct while the store is up and one node
// answers, and it fails the way every such lock fails: a partition longer than
// the ttl can let two holders exist. The mitigation is the one a scheduler
// needs anyway -- tasks are idempotent, because at-least-once is what a
// distributed scheduler delivers.
type Locks struct {
	store Locking
}

// NewLocks returns the issuer.
//
// It takes a Locking and not a Store, so a backend that cannot hold a lock
// cannot be wired in as one that can.
func NewLocks(s Locking) *Locks { return &Locks{store: s} }

// Lock names a lock. It does not touch the store: Acquire does.
//
// The ttl is required and it is the deadlock protection: a process that dies
// holding the lock releases it when the ttl expires, and there is no other way
// out. Size it above the longest run of the work it guards, or a second worker
// starts while the first is still going.
func (l *Locks) Lock(name string, ttl time.Duration) *Lock {
	return &Lock{store: l.store, name: name, ttl: ttl}
}

// RestoreLock returns a handle on a lock owner already holds.
//
// It answers the restoreLock() of HasCacheLock. It is how one process takes a
// lock and another gives it back: the owner string is the whole handle, so a
// job that acquires a lock can hand its owner to the worker that releases it.
//
// The returned handle believes it holds the lock. Ask IsOwnedByCurrentProcess
// if it matters whether that is still true.
func (l *Locks) RestoreLock(name, owner string) *Lock {
	return &Lock{store: l.store, name: name, owner: owner, held: true}
}

// Lock is one named lock, held or not.
//
// A Lock is not safe for concurrent use, and it does not need to be: it is the
// handle one goroutine holds while it does the work the lock protects.
type Lock struct {
	store Locking
	name  string
	ttl   time.Duration

	// owner proves this holder still owns the lock. It is minted at Acquire,
	// so a handle that was never acquired has nothing to release, and it is
	// minted fresh on every Acquire -- reusing one would let a holder whose
	// lock expired release the lock its successor now owns.
	owner string

	// held is what Held reports: this handle took the lock and has not given it
	// back. It is not the same question as "the lock has not expired", which
	// nothing can answer usefully.
	held bool

	// sleep is how long Block waits between attempts.
	sleep time.Duration
}

// Acquire takes the lock, or returns ErrLocked.
//
// It answers Lock::acquire(), which returns a bool. This returns an error
// because the two failures are different and a bool folds them together: a lock
// somebody else holds is ErrLocked and is ordinary, and a store that did not
// answer is the store's error and is not.
func (lk *Lock) Acquire(ctx context.Context) error {
	if lk.ttl <= 0 {
		return fmt.Errorf("%w: the lock %q has none, and a process that dies holding it would block the work forever", ErrNoTTL, lk.name)
	}
	if lk.name == "" {
		return fmt.Errorf("cache: a lock needs a name, and %q names every lock at once", lk.name)
	}

	owner, err := newLockOwner()
	if err != nil {
		return err
	}

	ok, err := lk.store.AcquireLock(ctx, lockKey(lk.name), owner, lk.ttl)
	if err != nil {
		return err
	}
	if !ok {
		return ErrLocked
	}
	lk.owner = owner
	lk.held = true
	return nil
}

// Release gives the lock back, and only if this holder still owns it.
//
// It answers Lock::release(). Releasing a lock that was never acquired, or that
// has already expired, is not an error: the caller wanted it gone and it is.
func (lk *Lock) Release(ctx context.Context) error {
	if lk == nil || lk.owner == "" {
		return nil
	}
	owner := lk.owner
	lk.owner = ""
	lk.held = false
	return lk.store.ReleaseLock(ctx, lockKey(lk.name), owner)
}

// ForceRelease releases the lock whoever holds it.
//
// It answers the forceRelease() of CacheLock and ArrayLock. It is the recovery
// hatch, not a tool: a caller reaching for it routinely has two holders running
// at once and does not know it. A store that cannot say who holds a lock cannot
// take it away from them, and returns ErrUnsupported.
func (lk *Lock) ForceRelease(ctx context.Context) error {
	lk.owner = ""
	lk.held = false

	reader, ok := lk.store.(CurrentOwner)
	if !ok {
		return fmt.Errorf("%w: releasing a lock held by somebody else", ErrUnsupported)
	}
	owner, err := reader.CurrentOwner(ctx, lockKey(lk.name))
	if err != nil {
		return err
	}
	if owner == "" {
		return nil
	}
	return lk.store.ReleaseLock(ctx, lockKey(lk.name), owner)
}

// Get takes the lock, runs fn if it got it, and releases it.
//
// It answers Lock::get(), including the shape of the answer: the bool says
// whether the lock was taken, and fn ran exactly when it is true. A lock
// somebody else holds is (false, nil) and not an error -- "another replica is
// doing it" is the expected outcome, not a fault.
//
// Pass a nil fn to acquire without running anything, which is what
// $lock->get() with no callback does.
func (lk *Lock) Get(ctx context.Context, fn func(context.Context) error) (bool, error) {
	switch err := lk.Acquire(ctx); {
	case err == nil:
	case isLocked(err):
		return false, nil
	default:
		return false, err
	}

	if fn == nil {
		return true, nil
	}
	defer func() { _ = lk.Release(context.WithoutCancel(ctx)) }()
	return true, fn(ctx)
}

// Block waits up to wait for the lock, then runs fn and releases it.
//
// A wait that runs out is ErrLockTimeout; a cancelled context is the context's
// error. Pass a nil fn to wait for the lock and keep it.
//
// It polls. The interval is a quarter second unless
// BetweenBlockedAttemptsSleepFor says otherwise, and it is a poll rather than a
// subscription because a lock that can be waited on properly is a feature of
// one backend and this has to work on all of them.
func (lk *Lock) Block(ctx context.Context, wait time.Duration, fn func(context.Context) error) error {
	sleep := lk.sleep
	if sleep <= 0 {
		sleep = defaultSleepBetweenBlockedAttempts
	}
	deadline := time.Now().Add(wait)

	for {
		switch err := lk.Acquire(ctx); {
		case err == nil:
			if fn == nil {
				return nil
			}
			defer func() { _ = lk.Release(context.WithoutCancel(ctx)) }()
			return fn(ctx)
		case isLocked(err):
			// Held. Wait, unless waiting would take us past the deadline.
			// The check is before the sleep rather than after, so a wait of
			// zero never sleeps at all.
			if time.Now().Add(sleep).After(deadline) {
				return fmt.Errorf("%w: %q was still held after %s", ErrLockTimeout, lk.name, wait)
			}
		default:
			return err
		}

		timer := time.NewTimer(sleep)
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
		}
	}
}

// Run acquires the lock, runs fn, and releases it.
//
// This is the shape the scheduler, the relay and an isolated command use, and
// having it here is what stops the acquire-defer-release sequence from being
// written slightly wrong in each of them. A lock that is already held returns
// ErrLocked and fn does not run -- which for a scheduled task means "another
// replica is doing it", not an error to report.
//
// The release runs on a context that is not cancelled with the caller's, so a
// request that is abandoned while fn is finishing still gives the lock back
// instead of leaving it to expire.
//
// It is Get with the refusal surfaced as ErrLocked instead of a false, which is
// what a caller that has nothing else to do wants.
func (lk *Lock) Run(ctx context.Context, fn func(context.Context) error) error {
	if err := lk.Acquire(ctx); err != nil {
		return err
	}
	defer func() { _ = lk.Release(context.WithoutCancel(ctx)) }()

	return fn(ctx)
}

// Owner returns the token written into the store for this lock.
//
// It answers Lock::owner(). It is empty until Acquire, because that is when the
// token is minted; hand it to Locks.RestoreLock to release the lock from
// somewhere else.
func (lk *Lock) Owner() string { return lk.owner }

// IsOwnedByCurrentProcess reports whether the store still says this handle
// holds the lock.
//
// It answers Lock::isOwnedByCurrentProcess(). Unlike Held it asks the store, so
// it says "not expired and not taken by anybody else", which is the question
// worth asking before doing something that assumed the lock was held.
func (lk *Lock) IsOwnedByCurrentProcess(ctx context.Context) (bool, error) {
	return lk.IsOwnedBy(ctx, lk.owner)
}

// IsOwnedBy reports whether owner holds the lock right now.
//
// It answers Lock::isOwnedBy(). A store that cannot say who holds a lock
// returns ErrUnsupported rather than guessing.
//
// The answer is true at the moment it is given and may be false at the moment
// it is acted on. Nothing can fix that, which is why Release checks ownership
// in the store rather than trusting a check made here.
func (lk *Lock) IsOwnedBy(ctx context.Context, owner string) (bool, error) {
	reader, ok := lk.store.(CurrentOwner)
	if !ok {
		return false, fmt.Errorf("%w: reading a lock's owner", ErrUnsupported)
	}
	current, err := reader.CurrentOwner(ctx, lockKey(lk.name))
	if err != nil {
		return false, err
	}
	return current != "" && current == owner, nil
}

// BetweenBlockedAttemptsSleepFor sets how long Block waits between attempts.
//
// It returns the lock, so it can be written in one line:
//
//	lock.BetweenBlockedAttemptsSleepFor(50 * time.Millisecond).Block(ctx, time.Second, fn)
func (lk *Lock) BetweenBlockedAttemptsSleepFor(d time.Duration) *Lock {
	lk.sleep = d
	return lk
}

// Held reports whether this handle currently believes it holds the lock.
//
// It answers from the token and does not ask the store, so it says "the lock
// was taken and not released here", not "the lock has not expired". Nothing can
// answer the second question usefully: it would be true at the moment of the
// answer and false at the moment the caller acted on it. Ask
// IsOwnedByCurrentProcess when the store's opinion is what is wanted.
func (lk *Lock) Held() bool { return lk != nil && lk.held }

// Name returns the lock's name. It answers the $name every Lock subclass reads.
func (lk *Lock) Name() string { return lk.name }

// isLocked is the one place that decides an error means "somebody else has it".
func isLocked(err error) bool { return err == ErrLocked }

// newLockOwner mints the token that proves ownership.
func newLockOwner() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", fmt.Errorf("cache: reading random bytes for a lock token: %w", err)
	}
	return hex.EncodeToString(b[:]), nil
}

// lockKey namespaces locks, and does not prefix them by tenant.
//
// That is deliberate, and it is the sentence that has to survive somebody
// making this consistent with the cache keys next door, which are all prefixed.
// A lock is taken to make one thing happen once across every replica of the
// instance. Prefixed by tenant it would be a different lock per tenant, so N
// replicas would each win a lock and each run the scheduled task -- for a
// different tenant, at the same minute, which is the failure the lock exists to
// prevent rather than an improvement on it. The same goes for the outbox relay
// and for a console command that must not run twice.
//
// A lock that really is about one tenant is written by putting the tenant in
// the name passed to Locks.Lock, which is what Repository.WithoutOverlapping
// does. The name is a string for that reason.
func lockKey(name string) string { return "lock:" + name }
