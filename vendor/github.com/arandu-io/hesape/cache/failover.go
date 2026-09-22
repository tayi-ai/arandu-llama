package cache

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"sync"
	"time"

	"github.com/arandu-io/hesape/cache/events"
)

// FailoverStore tries each store in turn and answers with the first one that
// worked.
//
// It is for the cache that must not be a single point of failure: put the fast store first and the durable one
// after it, and a cache that goes down becomes a slower cache instead of an
// outage.
//
// What it is not is replication. A write goes to the first store that accepts
// it and to no other, so after a failover the second store answers with whatever
// it happened to have -- which for a value written while the first store was up
// is nothing. The guarantee is availability of the cache, never agreement
// between the copies, and a caller that needs the second is not describing a
// cache.
//
// It fires CacheFailedOver when a store refuses an operation, once per store per
// spell of failure: a cache that has been down for an hour produces one event
// and not one per request, which is what makes the event worth listening to.
type FailoverStore struct {
	stores []Store
	names  []string
	events Dispatcher

	mu sync.Mutex
	// failing is which stores refused the last operation. It is what keeps
	// CacheFailedOver from firing on every request while a store is down.
	failing []string
}

var (
	_ Store    = (*FailoverStore)(nil)
	_ Locking  = (*FailoverStore)(nil)
	_ Taggable = (*FailoverStore)(nil)
)

// ErrAllStoresFailed is returned when every store in a failover set refused.
//
// It is only reached when the set is empty: with stores in it, the last one's
// own error is returned instead, because "the cache is down" is less use than
// "the connection was refused".
var ErrAllStoresFailed = errors.New("cache: every store in the failover set failed")

// NewFailoverStore returns a store over an ordered set of others.
//
// It answers FailoverStore::__construct(). The order is the priority: the first
// store is asked first, always, so a store that has come back is used again
// without anything having to notice that it did.
//
// names are what goes into CacheFailedOver, one per store and in the same order.
// A shorter list leaves the extra stores unnamed rather than refusing to build,
// because a store that cannot be named is still a store that can answer.
func NewFailoverStore(d Dispatcher, names []string, stores ...Store) *FailoverStore {
	return &FailoverStore{stores: stores, names: slices.Clone(names), events: d}
}

// Get returns the stored bytes from the first store that answered.
//
// A miss is an answer: ErrNotFound from the first store is the result, and the
// second store is not asked. Falling through on a miss would make every miss
// cost a round trip to every store, and would answer with whatever stale value
// the second one still had.
func (s *FailoverStore) Get(ctx context.Context, key string) ([]byte, error) {
	return attemptOnAllStores(s, func(store Store) ([]byte, error) {
		return store.Get(ctx, key)
	})
}

// Many returns the stored bytes for several keys from the first store that
// answered.
func (s *FailoverStore) Many(ctx context.Context, keys []string) (map[string][]byte, error) {
	return attemptOnAllStores(s, func(store Store) (map[string][]byte, error) {
		if batch, ok := store.(interface {
			Many(context.Context, []string) (map[string][]byte, error)
		}); ok {
			return batch.Many(ctx, keys)
		}
		out := make(map[string][]byte, len(keys))
		for _, key := range keys {
			value, err := store.Get(ctx, key)
			switch {
			case err == nil:
				out[key] = value
			case errors.Is(err, ErrNotFound):
				out[key] = nil
			default:
				return nil, err
			}
		}
		return out, nil
	})
}

// Put stores value in the first store that accepted it.
func (s *FailoverStore) Put(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	_, err := attemptOnAllStores(s, func(store Store) (struct{}, error) {
		return struct{}{}, store.Put(ctx, key, value, ttl)
	})
	return err
}

// PutMany stores the values in the first store that accepted them.
func (s *FailoverStore) PutMany(ctx context.Context, values map[string][]byte, ttl time.Duration) error {
	_, err := attemptOnAllStores(s, func(store Store) (struct{}, error) {
		if batch, ok := store.(interface {
			PutMany(context.Context, map[string][]byte, time.Duration) error
		}); ok {
			return struct{}{}, batch.PutMany(ctx, values, ttl)
		}
		for key, value := range values {
			if err := store.Put(ctx, key, value, ttl); err != nil {
				return struct{}{}, err
			}
		}
		return struct{}{}, nil
	})
	return err
}

// Add stores value in the first store that accepted it, if the key was absent
// there.
func (s *FailoverStore) Add(ctx context.Context, key string, value []byte, ttl time.Duration) (bool, error) {
	return attemptOnAllStores(s, func(store Store) (bool, error) {
		return store.Add(ctx, key, value, ttl)
	})
}

// Increment counts in the first store that accepted it.
//
// The counter lives in one store, not in all of them, so a failover in the
// middle of a window restarts the count. That is the trade the whole type is:
// availability over agreement.
func (s *FailoverStore) Increment(ctx context.Context, key string, delta int64, ttl time.Duration) (int64, error) {
	return attemptOnAllStores(s, func(store Store) (int64, error) {
		return store.Increment(ctx, key, delta, ttl)
	})
}

// Decrement counts down in the first store that accepted it.
func (s *FailoverStore) Decrement(ctx context.Context, key string, delta int64, ttl time.Duration) (int64, error) {
	return s.Increment(ctx, key, -delta, ttl)
}

// Forever stores a value with no expiry in the first store that accepted it.
func (s *FailoverStore) Forever(ctx context.Context, key string, value []byte) error {
	return s.Put(ctx, key, value, foreverTTL)
}

// Touch gives an entry a new expiry in the first store that accepted it.
func (s *FailoverStore) Touch(ctx context.Context, key string, ttl time.Duration) (bool, error) {
	return attemptOnAllStores(s, func(store Store) (bool, error) {
		touchable, ok := store.(interface {
			Touch(context.Context, string, time.Duration) (bool, error)
		})
		if !ok {
			return false, nil
		}
		return touchable.Touch(ctx, key, ttl)
	})
}

// Forget removes a key from the first store that accepted the removal.
//
// It is not removed from the others, which is the sharpest edge on this type: a
// value invalidated while the first store is up is still in the second, and a
// failover after that serves it. Give entries a ttl short enough that the stale
// window is one you can defend.
func (s *FailoverStore) Forget(ctx context.Context, key string) error {
	_, err := attemptOnAllStores(s, func(store Store) (struct{}, error) {
		return struct{}{}, store.Forget(ctx, key)
	})
	return err
}

// Flush empties the prefix in the first store that accepted it.
func (s *FailoverStore) Flush(ctx context.Context, prefix string) error {
	_, err := attemptOnAllStores(s, func(store Store) (struct{}, error) {
		return struct{}{}, store.Flush(ctx, prefix)
	})
	return err
}

// FlushStaleTags removes the tag entries nothing points at any more.
//
// It answers FailoverStore::flushStaleTags(): the first store that knows how is
// asked, and the rest are left alone. A store that cannot prune tags has none to
// prune -- the tag generations in this package are ordinary entries with a ttl,
// and the only backend that keeps a set beside them is the RESP one.
func (s *FailoverStore) FlushStaleTags(ctx context.Context) error {
	for _, store := range s.stores {
		flusher, ok := store.(interface{ FlushStaleTags(context.Context) error })
		if !ok {
			continue
		}
		return flusher.FlushStaleTags(ctx)
	}
	return nil
}

// GetPrefix is the prefix of the first store that has one.
func (s *FailoverStore) GetPrefix() string {
	prefix, err := attemptOnAllStores(s, func(store Store) (string, error) {
		prefixed, ok := store.(interface{ GetPrefix() string })
		if !ok {
			return "", nil
		}
		return prefixed.GetPrefix(), nil
	})
	if err != nil {
		return ""
	}
	return prefix
}

// AcquireLock takes the lock in the first store that could hold it.
//
// A lock held in one store means nothing in another, so a failover between
// acquiring and releasing leaves the lock to expire where it was taken -- and
// lets a second holder take it where it was not. A lock that must be exact needs
// a store that is not a failover set.
func (s *FailoverStore) AcquireLock(ctx context.Context, key, token string, ttl time.Duration) (bool, error) {
	return attemptOnAllStores(s, func(store Store) (bool, error) {
		locking, ok := store.(Locking)
		if !ok {
			return false, fmt.Errorf("%w: holding a lock", ErrUnsupported)
		}
		return locking.AcquireLock(ctx, key, token, ttl)
	})
}

// ReleaseLock releases the lock in the first store that could hold it.
func (s *FailoverStore) ReleaseLock(ctx context.Context, key, token string) error {
	_, err := attemptOnAllStores(s, func(store Store) (struct{}, error) {
		locking, ok := store.(Locking)
		if !ok {
			return struct{}{}, fmt.Errorf("%w: holding a lock", ErrUnsupported)
		}
		return struct{}{}, locking.ReleaseLock(ctx, key, token)
	})
	return err
}

// CurrentOwner asks the first store that can say who holds the lock.
func (s *FailoverStore) CurrentOwner(ctx context.Context, key string) (string, error) {
	return attemptOnAllStores(s, func(store Store) (string, error) {
		reader, ok := store.(CurrentOwner)
		if !ok {
			return "", fmt.Errorf("%w: reading a lock's owner", ErrUnsupported)
		}
		return reader.CurrentOwner(ctx, key)
	})
}

// Lock returns a handle on a named lock. It does not touch any store.
func (s *FailoverStore) Lock(name string, ttl time.Duration, owner string) *Lock {
	return &Lock{store: s, name: name, ttl: ttl, owner: owner, held: owner != ""}
}

// RestoreLock returns a handle on a lock owner already holds.
func (s *FailoverStore) RestoreLock(name, owner string) *Lock { return s.Lock(name, 0, owner) }

// attemptOnAllStores runs fn against each store until one of them answers.
//
// The bookkeeping is the reason it is one function: the set of stores that
// failed this time replaces the set that failed last time, only the newly
// failing ones get an event, and the replacement happens even when every store
// fails, so the record stays straight.
//
// It is a function and not a method because it is generic over what fn returns,
// and a method cannot take a type parameter in Go.
func attemptOnAllStores[T any](s *FailoverStore, fn func(Store) (T, error)) (T, error) {
	var zero T
	var lastErr error
	var failed []string

	defer func() {
		s.mu.Lock()
		s.failing = failed
		s.mu.Unlock()
	}()

	for i, store := range s.stores {
		value, err := fn(store)
		if err == nil || errors.Is(err, ErrNotFound) {
			// A miss is an answer. The store was reachable and said the key is
			// not there, which is the truth for the whole set: the next store's
			// copy of a key this one does not have is older, not better.
			return value, err
		}

		lastErr = err
		name := s.name(i)
		failed = append(failed, name)

		s.mu.Lock()
		alreadyFailing := slices.Contains(s.failing, name)
		s.mu.Unlock()

		if !alreadyFailing && s.events != nil {
			s.events.Dispatch(events.NewCacheFailedOver(name, err))
		}
	}

	if lastErr == nil {
		return zero, ErrAllStoresFailed
	}
	return zero, lastErr
}

// name is what the store at position i is called in the events, and an empty
// string when nobody said.
func (s *FailoverStore) name(i int) string {
	if i < len(s.names) {
		return s.names[i]
	}
	return ""
}
