package cache

import (
	"context"
	"errors"
	"time"
)

// Store is what a cache backend implements.
//
// It is deliberately below the application's vocabulary: it moves bytes under
// keys that are already built, it has never heard of a Grant or a tenant, and
// it does not know what a namespace is. Everything that decides WHICH key a
// value belongs under lives in Repository, in one place, where it can be got
// right once -- rather than in every backend, where the third one gets the
// tenant separator subtly different and two customers share an entry.
//
// A Store is safe for concurrent use.
type Store interface {
	// Get returns the stored bytes, or ErrNotFound when the key is absent or
	// expired.
	//
	// ErrNotFound specifically, and not the backend's own not-found error:
	// callers branch on it to mean "compute it", and a Store that returns
	// something else makes swapping the backend change the behaviour of the
	// application.
	Get(ctx context.Context, key string) ([]byte, error)

	// Put stores value under key for ttl, replacing whatever was there.
	//
	// ttl is always positive -- Repository refuses a non-positive one before it
	// gets here -- so a Store never has to decide what "no expiry" means.
	Put(ctx context.Context, key string, value []byte, ttl time.Duration) error

	// Add stores value only if the key is absent, and reports whether it did.
	//
	// It is the atomic half of the interface and the reason Repository.Add
	// exists: a get-then-put written at the call site is a race with a nice
	// name. A Store that cannot do it atomically has not implemented it.
	Add(ctx context.Context, key string, value []byte, ttl time.Duration) (bool, error)

	// Forget removes a key. Removing what is not there is not an error: the
	// caller wanted the key gone, and it is.
	Forget(ctx context.Context, key string) error

	// Increment adds delta to the integer stored under key and returns the new
	// value. An absent key starts at zero, so the first call returns delta.
	//
	// The expiry is set when the key is created and an existing expiry is left
	// alone. That is what makes a fixed window fixed: refreshing the ttl on
	// every hit would keep a counter alive for as long as traffic continues,
	// which is a window that never closes.
	//
	// The value is written as decimal text, which is also what JSON makes of an
	// integer -- so a counter written here reads back through Get[int64].
	Increment(ctx context.Context, key string, delta int64, ttl time.Duration) (int64, error)

	// Flush removes every key beginning with prefix.
	//
	// It takes a prefix and not nothing, because the only caller is
	// Repository.Flush and a repository owns one tenant's slice of one
	// namespace. A cache:clear that emptied the store would clear every other
	// tenant on the way past.
	Flush(ctx context.Context, prefix string) error
}

// Locking is the optional half of a Store: the two calls a distributed lock
// needs.
//
// It is a second interface rather than four more methods on Store, because a
// backend can be a perfectly good cache without being able to hold a lock, and
// because the lock is what the scheduler and the outbox relay depend on -- they
// take a Locking, and cannot be handed a Store that merely happens to compile.
//
// The token is what makes the release safe: without it, a holder whose lock
// already expired would release the lock a different holder now owns.
type Locking interface {
	// AcquireLock stores token under key for ttl if the key is free, and
	// reports whether it took it. It is Store.Add with a name that says what it
	// is for.
	AcquireLock(ctx context.Context, key, token string, ttl time.Duration) (bool, error)

	// ReleaseLock removes key only if it still holds token. Releasing a lock
	// that expired, or that somebody else now holds, is not an error and must
	// not delete anything.
	ReleaseLock(ctx context.Context, key, token string) error
}

// CanFlushLocks is the optional third half of a Store: emptying the lock space
// wholesale.
//
// It is a separate interface because a backend that keeps its locks in the same
// key space as its entries cannot flush the first without flushing the second,
// and must not pretend it can.
type CanFlushLocks interface {
	// FlushLocks removes every lock the store holds, held or not.
	FlushLocks(ctx context.Context) error
}

// CurrentOwner is the optional read half of Locking: who holds a lock now.
//
// It is what Lock.IsOwnedBy, Lock.IsOwnedByCurrentProcess and Lock.ForceRelease
// are built on.
//
// It is a third interface rather than a method on Locking, because Locking is
// implemented outside this module -- hesape/redis is one -- and widening an
// interface an adapter already satisfies breaks it at the next build.
type CurrentOwner interface {
	// CurrentOwner returns the token written into the store for key, or the
	// empty string when the lock is free or has expired.
	CurrentOwner(ctx context.Context, key string) (string, error)
}

// Taggable is the optional tag half of a Store.
//
// Every Store in this package satisfies it for free: a tag is an ordinary entry
// holding a generation id, so tagging needs Get, Put and Forget and nothing a
// Store does not already have. The interface exists so Repository.SupportsTags
// has something to ask.
type Taggable interface {
	Store
}

// ErrNotFound is returned when a key is absent.
//
// It is distinct from a stored empty value on purpose: "not cached" and "cached
// as empty" lead to different code, and conflating them is how a cache stampede
// starts.
var ErrNotFound = errors.New("cache: not found")

// ErrNoTenant is returned when a Grant carries no tenant, or one that cannot be
// part of a key.
//
// An error and not a fallback bucket: a cache key without a tenant is one
// request away from serving one customer's data to another.
var ErrNoTenant = errors.New("cache: the operation needs a tenant, and the Grant carries none")

// ErrNoTTL is returned when something that must expire was given no expiry.
var ErrNoTTL = errors.New("cache: an entry needs a time to live, or it is a second copy of the truth that nobody knows exists")

// ErrLocked is returned when the lock is held by somebody else.
var ErrLocked = errors.New("cache: the lock is held")

// ErrLockTimeout is returned by Lock.Block when the wait ran out before the
// lock came free.
//
// It is distinct from ErrLocked, which says "not now"; this one says "not
// within the time you were willing to wait", and a caller that retries on the
// first should not retry on the second.
var ErrLockTimeout = errors.New("cache: waiting for the lock timed out")

// ErrUnsupported is returned when a call needs something of the store that this
// store does not implement: flushing locks, reading a lock's owner.
//
// It is an error rather than a panic because the wiring that chose the store is
// usually not the code making the call.
var ErrUnsupported = errors.New("cache: this store does not support the operation")
