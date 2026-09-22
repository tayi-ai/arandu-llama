package cache

import (
	"context"
	"errors"
	"sync"
	"time"
)

// MemoizedStore remembers, for the life of one request, what the store
// underneath already answered.
//
// The problem it solves is the one nobody notices until they count: a request that asks the cache for the same
// feature flag in the controller, in the policy and twice in the view makes four
// round trips for one value. This makes it one, and the other three are a map
// lookup.
//
// It is a per-request object, not a second cache. Build one at the top of a
// request or a job, throw it away at the bottom. One kept alive for the life of
// the process is a cache with no expiry at all: nothing here ever forgets what
// it read, which is exactly what makes it safe for the length of one request and
// wrong for anything longer.
//
// Every write forgets what it remembered about that key first, so a caller that
// writes and then reads sees what it wrote.
//
// It wraps a Store rather than a Repository: a Repository method takes an
// auth.Grant and a Store has none to give it, so a Store that delegated to a
// Repository could not be written. What is memoized is the bytes under a fully
// built key.
type MemoizedStore struct {
	name  string
	store Store

	mu sync.Mutex
	// memo holds one entry per key that has been read. A miss is remembered as
	// a nil value rather than as an absence, so a key that is not there is
	// looked for once too.
	memo map[string][]byte
}

var (
	_ Store    = (*MemoizedStore)(nil)
	_ Taggable = (*MemoizedStore)(nil)
)

// NewMemoizedStore returns a store that remembers what it read.
//
// The name is the name of the store underneath. It goes into the events, so a
// listener can tell which cache a hit came from.
func NewMemoizedStore(name string, store Store) *MemoizedStore {
	return &MemoizedStore{name: name, store: store, memo: map[string][]byte{}}
}

// Name is the name of the store underneath.
func (s *MemoizedStore) Name() string { return s.name }

// GetPrefix is the prefix of the store underneath.
//
// It answers MemoizedStore::getPrefix(), which forwards for the same reason: the
// memo is keyed on what the store underneath would be asked, so the two have to
// agree on what that is.
func (s *MemoizedStore) GetPrefix() string {
	if prefixed, ok := s.store.(interface{ GetPrefix() string }); ok {
		return prefixed.GetPrefix()
	}
	return ""
}

// Get returns the stored bytes, asking the store underneath at most once per
// key.
//
// It answers MemoizedStore::get(). The second call for a key that was not there
// is still a miss and still costs nothing, which is the half people forget: a
// cache miss asked for four times is four round trips, and they are the
// expensive ones.
func (s *MemoizedStore) Get(ctx context.Context, key string) ([]byte, error) {
	s.mu.Lock()
	value, remembered := s.memo[key]
	s.mu.Unlock()

	if remembered {
		if value == nil {
			return nil, ErrNotFound
		}
		return copyBytes(value), nil
	}

	value, err := s.store.Get(ctx, key)
	switch {
	case err == nil:
	case errors.Is(err, ErrNotFound):
		s.remember(key, nil)
		return nil, ErrNotFound
	default:
		// A store that failed is not a store that answered. Remembering the
		// failure would turn one bad moment into a bad request.
		return nil, err
	}

	s.remember(key, value)
	return copyBytes(value), nil
}

// Many returns the stored bytes for several keys, asking the store underneath
// only for the ones it has not seen.
func (s *MemoizedStore) Many(ctx context.Context, keys []string) (map[string][]byte, error) {
	out := make(map[string][]byte, len(keys))
	var missing []string

	s.mu.Lock()
	for _, key := range keys {
		if value, remembered := s.memo[key]; remembered {
			out[key] = copyBytes(value)
			continue
		}
		missing = append(missing, key)
	}
	s.mu.Unlock()

	if len(missing) == 0 {
		return out, nil
	}

	retrieved, err := s.many(ctx, missing)
	if err != nil {
		return nil, err
	}
	for key, value := range retrieved {
		s.remember(key, value)
		out[key] = copyBytes(value)
	}
	return out, nil
}

// many asks the store underneath, using its own batch read when it has one.
func (s *MemoizedStore) many(ctx context.Context, keys []string) (map[string][]byte, error) {
	if batch, ok := s.store.(interface {
		Many(context.Context, []string) (map[string][]byte, error)
	}); ok {
		return batch.Many(ctx, keys)
	}

	out := make(map[string][]byte, len(keys))
	for _, key := range keys {
		value, err := s.store.Get(ctx, key)
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
}

// Put forgets what it remembered about the key and writes it through.
func (s *MemoizedStore) Put(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	s.forget(key)
	return s.store.Put(ctx, key, value, ttl)
}

// PutMany forgets them all and writes them through.
func (s *MemoizedStore) PutMany(ctx context.Context, values map[string][]byte, ttl time.Duration) error {
	for key := range values {
		s.forget(key)
	}

	if batch, ok := s.store.(interface {
		PutMany(context.Context, map[string][]byte, time.Duration) error
	}); ok {
		return batch.PutMany(ctx, values, ttl)
	}
	for key, value := range values {
		if err := s.store.Put(ctx, key, value, ttl); err != nil {
			return err
		}
	}
	return nil
}

// Add forgets what it remembered about the key and adds it through.
func (s *MemoizedStore) Add(ctx context.Context, key string, value []byte, ttl time.Duration) (bool, error) {
	s.forget(key)
	return s.store.Add(ctx, key, value, ttl)
}

// Increment forgets what it remembered about the counter and increments it
// through.
//
// It answers MemoizedStore::increment(). A memoized counter would be a counter
// that stopped counting, which is why the forget comes first.
func (s *MemoizedStore) Increment(ctx context.Context, key string, delta int64, ttl time.Duration) (int64, error) {
	s.forget(key)
	return s.store.Increment(ctx, key, delta, ttl)
}

// Decrement forgets what it remembered about the counter and decrements it
// through. It answers MemoizedStore::decrement().
func (s *MemoizedStore) Decrement(ctx context.Context, key string, delta int64, ttl time.Duration) (int64, error) {
	return s.Increment(ctx, key, -delta, ttl)
}

// Forever forgets what it remembered about the key and writes it through.
func (s *MemoizedStore) Forever(ctx context.Context, key string, value []byte) error {
	return s.Put(ctx, key, value, foreverTTL)
}

// Touch forgets what it remembered about the key and touches it through.
func (s *MemoizedStore) Touch(ctx context.Context, key string, ttl time.Duration) (bool, error) {
	s.forget(key)
	if touchable, ok := s.store.(interface {
		Touch(context.Context, string, time.Duration) (bool, error)
	}); ok {
		return touchable.Touch(ctx, key, ttl)
	}
	return false, nil
}

// Forget forgets what it remembered about the key and removes it.
func (s *MemoizedStore) Forget(ctx context.Context, key string) error {
	s.forget(key)
	return s.store.Forget(ctx, key)
}

// Flush forgets everything it remembered and flushes the store underneath.
//
// It answers MemoizedStore::flush(). Everything, and not only the prefix being
// flushed: a memo that kept part of what a flush removed would serve it after it
// was gone, and a request-scoped map is cheap to rebuild.
func (s *MemoizedStore) Flush(ctx context.Context, prefix string) error {
	s.mu.Lock()
	clear(s.memo)
	s.mu.Unlock()
	return s.store.Flush(ctx, prefix)
}

// AcquireLock takes the lock in the store underneath.
//
// It answers MemoizedStore::lock(), including the refusal: a store that cannot
// hold a lock is asked for one and says so. Nothing about a lock is memoized --
// remembering that a lock was free would be a lock that is always free.
func (s *MemoizedStore) AcquireLock(ctx context.Context, key, token string, ttl time.Duration) (bool, error) {
	locking, ok := s.store.(Locking)
	if !ok {
		return false, ErrUnsupported
	}
	return locking.AcquireLock(ctx, key, token, ttl)
}

// ReleaseLock releases the lock in the store underneath.
func (s *MemoizedStore) ReleaseLock(ctx context.Context, key, token string) error {
	locking, ok := s.store.(Locking)
	if !ok {
		return ErrUnsupported
	}
	return locking.ReleaseLock(ctx, key, token)
}

// CurrentOwner asks the store underneath who holds the lock.
func (s *MemoizedStore) CurrentOwner(ctx context.Context, key string) (string, error) {
	reader, ok := s.store.(CurrentOwner)
	if !ok {
		return "", ErrUnsupported
	}
	return reader.CurrentOwner(ctx, key)
}

// GetStore returns the store underneath.
func (s *MemoizedStore) GetStore() Store { return s.store }

// remember records what a key answered.
func (s *MemoizedStore) remember(key string, value []byte) {
	s.mu.Lock()
	s.memo[key] = copyBytes(value)
	s.mu.Unlock()
}

// forget drops what a key answered.
func (s *MemoizedStore) forget(key string) {
	s.mu.Lock()
	delete(s.memo, key)
	s.mu.Unlock()
}

// copyBytes keeps the memo and the caller from sharing a slice: a caller that
// mutates what it was given must not rewrite what is remembered.
func copyBytes(b []byte) []byte {
	if b == nil {
		return nil
	}
	out := make([]byte, len(b))
	copy(out, b)
	return out
}
