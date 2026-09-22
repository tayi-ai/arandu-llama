package cache

import (
	"context"
	"fmt"
	"strconv"
	"strings"
	"sync"
	"time"
)

// ArrayStore is the in-process cache: a map, a mutex, and expiry.
//
// It is the default store, and it is the right one for development, for tests,
// and for a single instance that only caches what it can recompute. It is the
// wrong one for anything else, and the reason is not performance: with N
// replicas there are N caches, so a Forget on one replica leaves the other N-1
// serving the old value until the ttl runs out. When that matters, the store is
// the RESP one in hesape/redis -- a separate module, registered through
// CacheManager.Extend -- and nothing else in the application changes.
//
// It is safe for concurrent use, and it holds a copy of every value it is
// given: a caller that reuses a buffer after a Put cannot rewrite what is
// cached, and a caller that mutates what Get returned cannot either.
type ArrayStore struct {
	mu sync.Mutex

	// entries is the cache. Expired entries are dropped when they are next
	// looked at rather than by a goroutine: a background sweeper in a library
	// is a goroutine the application never asked for and cannot stop, and the
	// memory it saves is memory that a process this size was not short of.
	// Flush and Forget still remove them for real.
	entries map[string]entry

	// locks is separate from entries on purpose. A lock is not tenant-scoped
	// and a cache entry is, so they live in different key spaces -- and
	// Repository.Flush, which empties one tenant's namespace, must not be able
	// to release a lock the scheduler is holding.
	locks map[string]entry
}

type entry struct {
	value   []byte
	expires time.Time
}

// NewArrayStore returns an empty in-process store.
func NewArrayStore() *ArrayStore {
	return &ArrayStore{
		entries: make(map[string]entry),
		locks:   make(map[string]entry),
	}
}

var (
	_ Store         = (*ArrayStore)(nil)
	_ Locking       = (*ArrayStore)(nil)
	_ CanFlushLocks = (*ArrayStore)(nil)
	_ CurrentOwner  = (*ArrayStore)(nil)
	_ Taggable      = (*ArrayStore)(nil)
)

// Entry is one stored value and when it stops being one.
//
// It exists only for All: nothing in the ordinary path of the cache needs to
// see an expiry.
type Entry struct {
	// Value is the stored bytes. It is a copy; mutating it changes nothing.
	Value []byte

	// ExpiresAt is when the entry stops being live.
	ExpiresAt time.Time
}

// All returns every live entry and its expiry.
//
// Expired entries are dropped on the way out rather than returned with a past
// deadline, so what All shows is what Get would find.
//
// It is a snapshot: the map and every value in it are copies, so a caller that
// ranges over the result while another goroutine writes cannot see a torn read
// and cannot rewrite the cache by assigning into what it was given.
func (s *ArrayStore) All() map[string]Entry {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := make(map[string]Entry, len(s.entries))
	for key := range s.entries {
		e, ok := s.live(s.entries, key)
		if !ok {
			continue
		}
		value := make([]byte, len(e.value))
		copy(value, e.value)
		out[key] = Entry{Value: value, ExpiresAt: e.expires}
	}
	return out
}

// GetPrefix returns the prefix this store puts in front of every key.
//
// It is the empty string: the store is a map in this process, nothing else is
// sharing it, and the prefixing that matters -- tenant and namespace -- happens
// in Repository, where it can be got right once.
func (s *ArrayStore) GetPrefix() string { return "" }

// Many returns the stored bytes for several keys at once.
//
// Keys that are absent or expired are present in the result with a nil value,
// so the caller can tell "I asked for six and got six" without holding the
// slice it asked with.
func (s *ArrayStore) Many(_ context.Context, keys []string) (map[string][]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	out := make(map[string][]byte, len(keys))
	for _, key := range keys {
		e, ok := s.live(s.entries, key)
		if !ok {
			out[key] = nil
			continue
		}
		value := make([]byte, len(e.value))
		copy(value, e.value)
		out[key] = value
	}
	return out, nil
}

// PutMany stores several values under one ttl.
//
// It is a loop rather than a transaction: a store that is a map has nothing to
// roll back to.
func (s *ArrayStore) PutMany(_ context.Context, values map[string][]byte, ttl time.Duration) error {
	if ttl <= 0 {
		return ErrNoTTL
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	for key, value := range values {
		s.entries[key] = newEntry(value, ttl)
	}
	return nil
}

// Get returns a copy of the stored bytes, or ErrNotFound.
func (s *ArrayStore) Get(_ context.Context, key string) ([]byte, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	e, ok := s.live(s.entries, key)
	if !ok {
		return nil, ErrNotFound
	}
	out := make([]byte, len(e.value))
	copy(out, e.value)
	return out, nil
}

// Put stores value for ttl.
func (s *ArrayStore) Put(_ context.Context, key string, value []byte, ttl time.Duration) error {
	if ttl <= 0 {
		return ErrNoTTL
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	s.entries[key] = newEntry(value, ttl)
	return nil
}

// Add stores value if the key is free, and reports whether it did.
func (s *ArrayStore) Add(_ context.Context, key string, value []byte, ttl time.Duration) (bool, error) {
	if ttl <= 0 {
		return false, ErrNoTTL
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.live(s.entries, key); ok {
		return false, nil
	}
	s.entries[key] = newEntry(value, ttl)
	return true, nil
}

// Forget removes a key, present or not.
func (s *ArrayStore) Forget(_ context.Context, key string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.entries, key)
	return nil
}

// Increment adds delta to the counter under key and returns the new value.
//
// The expiry belongs to the counter and not to the increment: a key that is
// already there keeps the deadline it was created with, which is what makes a
// fixed window fixed.
func (s *ArrayStore) Increment(_ context.Context, key string, delta int64, ttl time.Duration) (int64, error) {
	if ttl <= 0 {
		return 0, ErrNoTTL
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	n := int64(0)
	expires := time.Now().Add(ttl)
	if e, ok := s.live(s.entries, key); ok {
		parsed, err := strconv.ParseInt(string(e.value), 10, 64)
		if err != nil {
			return 0, errNotACounter(key)
		}
		n = parsed
		expires = e.expires
	}

	n += delta
	s.entries[key] = entry{value: []byte(strconv.FormatInt(n, 10)), expires: expires}
	return n, nil
}

// Decrement subtracts delta from the counter under key and returns the new
// value. It is Increment with the sign turned round.
func (s *ArrayStore) Decrement(ctx context.Context, key string, delta int64, ttl time.Duration) (int64, error) {
	return s.Increment(ctx, key, -delta, ttl)
}

// Forever stores a value with no expiry the caller has to think about.
//
// A Store here is promised a positive ttl, so "never" is the longest deadline
// the package is willing to write down -- see Repository.Forever for what that
// is and why it is a number and not a special case.
func (s *ArrayStore) Forever(ctx context.Context, key string, value []byte) error {
	return s.Put(ctx, key, value, foreverTTL)
}

// Touch gives a live entry a new expiry and reports whether there was one.
//
// An absent or expired key is false and is not created: touching what is not
// there would turn a cache miss into a cache entry holding nothing.
func (s *ArrayStore) Touch(_ context.Context, key string, ttl time.Duration) (bool, error) {
	if ttl <= 0 {
		return false, ErrNoTTL
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	e, ok := s.live(s.entries, key)
	if !ok {
		return false, nil
	}
	e.expires = time.Now().Add(ttl)
	s.entries[key] = e
	return true, nil
}

// FlushLocks releases every lock this store holds.
//
// It never refuses: locks live in their own map, precisely so that a
// Repository.Flush emptying one tenant's namespace cannot release the lock the
// scheduler is holding.
func (s *ArrayStore) FlushLocks(_ context.Context) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	clear(s.locks)
	return nil
}

// HasSeparateLockStore reports whether locks live apart from entries, and the
// answer is always yes: see FlushLocks.
func (s *ArrayStore) HasSeparateLockStore() bool { return true }

// CurrentOwner returns the token holding the lock, or the empty string.
//
// A lock that expired is free, and free is the empty string rather than the
// token of whoever held it last -- otherwise IsOwnedBy would keep saying yes to
// a holder that lost the lock.
func (s *ArrayStore) CurrentOwner(_ context.Context, key string) (string, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	e, ok := s.live(s.locks, key)
	if !ok {
		return "", nil
	}
	return string(e.value), nil
}

// ForceReleaseLock releases a lock whoever holds it.
//
// It is the recovery hatch, not a tool: a caller that uses it routinely has two
// holders running at once and does not know it.
func (s *ArrayStore) ForceReleaseLock(_ context.Context, key string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	delete(s.locks, key)
	return nil
}

// Lock returns a handle on a named lock. It does not touch the store.
//
// Pass an empty owner to have one minted at Acquire; pass one to restore a
// handle on a lock this process already took, which is what RestoreLock does.
func (s *ArrayStore) Lock(name string, ttl time.Duration, owner string) *Lock {
	return &Lock{store: s, name: name, ttl: ttl, owner: owner, held: owner != ""}
}

// RestoreLock returns a handle on a lock owner already holds.
//
// It is what lets one process take a lock and another part of it release the
// same lock: the owner string is the whole handle.
func (s *ArrayStore) RestoreLock(name, owner string) *Lock {
	return s.Lock(name, 0, owner)
}

// Flush removes every key beginning with prefix.
func (s *ArrayStore) Flush(_ context.Context, prefix string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	for key := range s.entries {
		if strings.HasPrefix(key, prefix) {
			delete(s.entries, key)
		}
	}
	return nil
}

// AcquireLock takes the lock if it is free.
func (s *ArrayStore) AcquireLock(_ context.Context, key, token string, ttl time.Duration) (bool, error) {
	if ttl <= 0 {
		return false, ErrNoTTL
	}
	s.mu.Lock()
	defer s.mu.Unlock()

	if _, ok := s.live(s.locks, key); ok {
		return false, nil
	}
	s.locks[key] = newEntry([]byte(token), ttl)
	return true, nil
}

// ReleaseLock releases the lock only if token still holds it.
func (s *ArrayStore) ReleaseLock(_ context.Context, key, token string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	e, ok := s.live(s.locks, key)
	if !ok || string(e.value) != token {
		// Already expired, or held by somebody else. Deleting it here is
		// exactly the bug the token exists to prevent.
		return nil
	}
	delete(s.locks, key)
	return nil
}

// live reads an entry and drops it if it has expired. The caller holds the
// mutex.
func (s *ArrayStore) live(from map[string]entry, key string) (entry, bool) {
	e, ok := from[key]
	if !ok {
		return entry{}, false
	}
	if !time.Now().Before(e.expires) {
		delete(from, key)
		return entry{}, false
	}
	return e, true
}

// errNotACounter is what Increment says about a key holding something else.
//
// It refuses rather than overwriting, because the alternative is a counter that
// silently restarts from zero the first time somebody caches a struct under the
// key a rate limiter is using -- and the symptom of that is a limit that stops
// limiting.
func errNotACounter(key string) error {
	return fmt.Errorf("cache: %q holds a value that is not a counter, and Increment would have to overwrite it to answer", key)
}

func newEntry(value []byte, ttl time.Duration) entry {
	stored := make([]byte, len(value))
	copy(stored, value)
	return entry{value: stored, expires: time.Now().Add(ttl)}
}
