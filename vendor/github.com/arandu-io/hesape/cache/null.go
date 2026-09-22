package cache

import (
	"context"
	"time"
)

// NullStore is the store that keeps nothing.
//
// It is what CACHE_STORE=null wires, and it is the honest way to turn caching off: every read is a miss and every
// write goes nowhere, so the application takes exactly the path it takes on a
// cold cache, every time. That is worth more in a test than a mock, because it
// is the same code the production wiring runs.
//
// Put, Forever, Increment and Touch return no error. A caller checks the error
// of every write, and a null store that reported a failure on each of them
// would take the application down rather than turning the cache off. What a
// reader sees is unchanged: ErrNotFound, always.
//
// It holds a lock as NoLock does: it says yes to everybody. See NoLock for why
// that is the right answer here and the wrong one anywhere else.
type NullStore struct{}

// NewNullStore returns the store that keeps nothing.
func NewNullStore() *NullStore { return &NullStore{} }

var (
	_ Store        = (*NullStore)(nil)
	_ Locking      = (*NullStore)(nil)
	_ CurrentOwner = (*NullStore)(nil)
	_ Taggable     = (*NullStore)(nil)
)

// Get is always a miss. It answers NullStore::get(), which returns null.
func (s *NullStore) Get(context.Context, string) ([]byte, error) { return nil, ErrNotFound }

// Many is a miss for every key.
func (s *NullStore) Many(_ context.Context, keys []string) (map[string][]byte, error) {
	out := make(map[string][]byte, len(keys))
	for _, key := range keys {
		out[key] = nil
	}
	return out, nil
}

// Put discards the value. It answers NullStore::put().
func (s *NullStore) Put(context.Context, string, []byte, time.Duration) error { return nil }

// PutMany discards them all. It answers the putMany() of RetrievesMultipleKeys.
func (s *NullStore) PutMany(context.Context, map[string][]byte, time.Duration) error { return nil }

// Add never adds, and says so. It answers the add() a null store does not have:
// nothing is there, and nothing was put there either.
func (s *NullStore) Add(context.Context, string, []byte, time.Duration) (bool, error) {
	return false, nil
}

// Forever discards the value. It answers NullStore::forever().
func (s *NullStore) Forever(context.Context, string, []byte) error { return nil }

// Increment counts nothing and returns zero. It answers
// NullStore::increment(), which returns false.
func (s *NullStore) Increment(context.Context, string, int64, time.Duration) (int64, error) {
	return 0, nil
}

// Decrement counts nothing and returns zero. It answers
// NullStore::decrement().
func (s *NullStore) Decrement(context.Context, string, int64, time.Duration) (int64, error) {
	return 0, nil
}

// Touch reports that there was nothing to touch. It answers NullStore::touch().
func (s *NullStore) Touch(context.Context, string, time.Duration) (bool, error) {
	return false, nil
}

// Forget removes nothing, successfully. It answers NullStore::forget(), which
// returns true.
func (s *NullStore) Forget(context.Context, string) error { return nil }

// Flush removes nothing, successfully. It answers NullStore::flush(), which
// returns true.
func (s *NullStore) Flush(context.Context, string) error { return nil }

// GetPrefix is the empty string. It answers NullStore::getPrefix().
func (s *NullStore) GetPrefix() string { return "" }

// AcquireLock always succeeds. It is NoLock's acquire: see Lock for what that
// costs.
func (s *NullStore) AcquireLock(context.Context, string, string, time.Duration) (bool, error) {
	return true, nil
}

// ReleaseLock always succeeds, having released nothing.
func (s *NullStore) ReleaseLock(context.Context, string, string) error { return nil }

// CurrentOwner is nobody, because no lock was ever written down.
//
// It answers NoLock::getCurrentOwner(), which returns the asking handle's own
// owner -- so isOwnedByCurrentProcess is true. Here the store cannot know who
// is asking, and an empty owner is the closest honest answer: a lock nothing
// wrote down is a lock nobody holds.
func (s *NullStore) CurrentOwner(context.Context, string) (string, error) { return "", nil }

// Lock returns a handle that will take the lock, because NullStore gives it to
// everybody.
func (s *NullStore) Lock(name string, ttl time.Duration, owner string) *Lock {
	return &Lock{store: s, name: name, ttl: ttl, owner: owner, held: owner != ""}
}

// RestoreLock returns a handle on a lock owner already holds.
func (s *NullStore) RestoreLock(name, owner string) *Lock { return s.Lock(name, 0, owner) }

// NoLock is the lock that is always free.
//
// Acquire says yes, release says yes, and nothing is written anywhere. It is
// what the null store hands out, and it is
// correct there for the same reason the null store is correct -- caching is off,
// so there is nothing to serialize access to.
//
// It is wrong everywhere else, and loudly: a scheduler holding a NoLock runs its
// task on every replica at once. If a lock matters, the store has to be one that
// can hold it.
type NoLock struct{ *Lock }

// NewNoLock returns a lock that everybody gets.
//
// The ttl is not a deadlock protection here, because there is no lock to dead:
// it is carried so a handle taken from a NoLock reads the same as one taken from
// a real store.
func NewNoLock(name string, ttl time.Duration, owner string) *NoLock {
	return &NoLock{Lock: (&NullStore{}).Lock(name, ttl, owner)}
}
