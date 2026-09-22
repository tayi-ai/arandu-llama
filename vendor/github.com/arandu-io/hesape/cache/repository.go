package cache

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"sort"
	"strconv"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/cache/events"
)

// foreverTTL is the deadline Forever, RememberForever and Sear write.
//
// A Store here is promised a positive ttl -- that promise is what lets every
// backend skip deciding what zero means -- so forever is the longest deadline
// the package is willing to write down instead of a special case that every
// adapter would implement slightly differently.
//
// A century, and not the largest duration Go can hold, because an adapter that
// turns a ttl into a unix timestamp for a server to store must be able to hold
// the answer.
const foreverTTL = 100 * 365 * 24 * time.Hour

// defaultCacheTime is how long an item is stored for when no ttl is given.
const defaultCacheTime = time.Hour

// Repository is the cache an application calls.
//
// It is the tenant-scoped, JSON-encoding, expiry-enforcing layer over a Store --
// the only one of the two that an application ever holds. Every method takes an
// auth.Grant, and the tenant comes from it: not from the request, not from a
// field somebody filled in.
//
// It does not run a policy. Holding a Grant is already proof that one ran:
// Grant has no exported fields and cannot be written as a literal, so a caller
// with one has been past Authorize or has asked for a SystemGrant by name. What
// the Repository takes from it is the tenant, which is the part a cache key
// cannot be built without.
type Repository struct {
	store     Store
	namespace string

	// name is the name the manager knows this store by. It is empty on a
	// repository built with New, and it is what GetName -- and therefore every
	// event -- reports.
	name string

	// defaultTTL is carried into every TaggedCache that Tags builds. Nothing on
	// this type takes an optional ttl, so that is the whole of its job here.
	defaultTTL time.Duration

	// tags is nil on an untagged repository. When it is not, every key this
	// repository builds is prefixed by the tag set's generation, so resetting
	// the generation orphans every entry at once. See TaggedCache.
	tags *TagSet

	// events is nil until SetEventDispatcher, and a nil one fires nothing,
	// which keeps a cache built without a bus from needing one.
	events Dispatcher
}

// New returns a repository over a store, under the default namespace.
//
// Use Namespace to separate caches of different kinds: "user",
// "invoice-total". Two namespaces over one store cannot collide, and clearing
// one leaves the other alone.
func New(s Store) *Repository {
	return &Repository{store: s, namespace: "cache", defaultTTL: defaultCacheTime}
}

// Namespace returns a repository over the same store, under another namespace.
//
// It derives rather than mutates, so a repository handed to two modules cannot
// have its namespace changed underneath one of them.
//
// The name is limited to what a tenant is limited to -- lowercase letters,
// digits, - and _ -- for exactly the same reason: it is a segment of a key, and
// a segment carrying a colon can name another namespace's entry. An invalid name is
// reported by every call the repository makes, not by this constructor: this is
// wiring, and wiring that panics takes the process down at boot for a typo.
func (r *Repository) Namespace(name string) *Repository {
	out := *r
	out.namespace = name
	return &out
}

// Has reports whether a key is present and unexpired.
//
// It is a read, and it fires the same events a read fires. A hit rate computed
// from CacheHit and CacheMissed counts these, which is one more reason not to
// ask before reading.
//
// Whoever is about to read the value should call Get instead: asking and then
// reading is two round trips and a race, and the answer to "is it there" is
// already in Get's error.
//
// A key cached as null is present: ErrNotFound is what reports an absence, so
// "not cached" and "cached as nothing" stay different -- which is the whole
// reason ErrNotFound exists.
func (r *Repository) Has(ctx context.Context, g auth.Grant, key string) (bool, error) {
	if _, err := Get[any](ctx, r, g, key); err != nil {
		if errors.Is(err, ErrNotFound) {
			return false, nil
		}
		return false, err
	}
	return true, nil
}

// Missing reports whether a key is absent. It is the negation of Has and
// nothing else -- it exists because "if !has" reads worse than "if missing" at
// the call site.
func (r *Repository) Missing(ctx context.Context, g auth.Grant, key string) (bool, error) {
	has, err := r.Has(ctx, g, key)
	if err != nil {
		return false, err
	}
	return !has, nil
}

// Put stores a value for ttl, replacing whatever was there.
//
// A ttl that has already passed forgets the key rather than writing it, because
// an entry that expires the moment it is written and an entry that is not there
// are the same entry, and the shorter of the two ways to say it is the one the
// caller wrote. Refusing the write instead would leave the previous value in
// place: the caller would have told the cache to stop serving something and the
// cache would go on serving it.
//
// The ttl is required rather than optional, because an entry with no expiry is
// a second copy of the truth that nobody knows exists. Call Forever when that
// is what you mean, so it is written down at the call site.
func (r *Repository) Put(ctx context.Context, g auth.Grant, key string, value any, ttl time.Duration) error {
	if ttl <= 0 {
		return r.Forget(ctx, g, key)
	}
	r.eventWriting(key, value, ttl)
	err := r.put(ctx, g, key, value, ttl)
	r.eventWritten(key, value, ttl, err)
	return err
}

// put is the write itself, without the events.
//
// It exists so PutMany can fire WritingManyKeys once for the batch instead of
// WritingKey once per key.
func (r *Repository) put(ctx context.Context, g auth.Grant, key string, value any, ttl time.Duration) error {
	full, err := r.key(ctx, g, key)
	if err != nil {
		return err
	}
	raw, err := encode(value)
	if err != nil {
		return err
	}
	if ttl <= 0 {
		return ErrNoTTL
	}
	return r.store.Put(ctx, full, raw, ttl)
}

// Set is Put under another name.
func (r *Repository) Set(ctx context.Context, g auth.Grant, key string, value any, ttl time.Duration) error {
	return r.Put(ctx, g, key, value, ttl)
}

// PutMany stores several values under one ttl.
//
// It is a loop, and it is honest about being one: it is not atomic, and a
// failure part-way through leaves the entries it already wrote in place. That
// is the right failure for a cache -- the alternative would be a transaction
// across a store that may not have one -- and it is why the method exists here
// rather than being written slightly differently in each module.
//
// A ttl that has already passed forgets the keys, which is the batch spelling
// of what Put does.
func (r *Repository) PutMany(ctx context.Context, g auth.Grant, values map[string]any, ttl time.Duration) error {
	keys := make([]string, 0, len(values))
	written := make([]any, 0, len(values))
	for key, value := range values {
		keys = append(keys, key)
		written = append(written, value)
	}
	sort.Sort(byKey{keys: keys, values: written})

	if ttl <= 0 {
		return r.DeleteMultiple(ctx, g, keys...)
	}

	if r.events != nil {
		r.event(events.NewWritingManyKeys(r.GetName(), keys, written, seconds(ttl), r.tagNames()))
	}

	// One WritingManyKeys for the batch, then one KeyWritten per key. Put is
	// not called: it would fire a WritingKey per key on top, and a listener
	// counting writes would count each of them twice.
	for i, key := range keys {
		err := r.put(ctx, g, key, written[i], ttl)
		r.eventWritten(key, written[i], ttl, err)
		if err != nil {
			return err
		}
	}
	return nil
}

// byKey sorts the keys of a PutMany and keeps the values alongside them.
//
// A Go map has no iteration order, and a batch that landed in a different order
// on every call would make the events -- and any test over them --
// non-deterministic. Sorted is the one order every caller can agree on.
type byKey struct {
	keys   []string
	values []any
}

func (b byKey) Len() int           { return len(b.keys) }
func (b byKey) Less(i, j int) bool { return b.keys[i] < b.keys[j] }
func (b byKey) Swap(i, j int) {
	b.keys[i], b.keys[j] = b.keys[j], b.keys[i]
	b.values[i], b.values[j] = b.values[j], b.values[i]
}

// SetMultiple is PutMany under another name.
func (r *Repository) SetMultiple(ctx context.Context, g auth.Grant, values map[string]any, ttl time.Duration) error {
	return r.PutMany(ctx, g, values, ttl)
}

// Add stores a value only if the key is absent, and reports whether it did.
//
// It is the atomic primitive underneath "only one of them may do this": the
// first caller gets true, everybody else gets false, and no two callers get
// true. A Get followed by a Put is the same code with a race in the middle.
//
// A ttl that has already passed adds nothing and reports false -- an entry that
// is expired before it is written was not added.
//
// The trap that leaves behind is worth knowing: a lock built on Add with a ttl
// that is computed and comes out zero never acquires, and never says why. The
// caller is holding the ttl, and the false is the answer to the question it
// asked.
func (r *Repository) Add(ctx context.Context, g auth.Grant, key string, value any, ttl time.Duration) (bool, error) {
	full, err := r.key(ctx, g, key)
	if err != nil {
		return false, err
	}
	raw, err := encode(value)
	if err != nil {
		return false, err
	}
	if ttl <= 0 {
		return false, nil
	}
	return r.store.Add(ctx, full, raw, ttl)
}

// Forever stores a value with no expiry the caller has to think about.
//
// "No expiry" is written down as a century, because a Store is promised a
// positive ttl and a special case for zero is a thing every backend would get
// subtly different.
//
// Reach for it rarely. An entry that never expires is a second copy of the
// truth, and the day it diverges from the first nothing notices; Put with a ttl
// you can defend is almost always what was meant.
func (r *Repository) Forever(ctx context.Context, g auth.Grant, key string, value any) error {
	return r.Put(ctx, g, key, value, foreverTTL)
}

// Increment adds delta to a counter and returns the new value.
//
// The ttl is the counter's whole life: it is set when the counter is created
// and is not refreshed by later increments, so a counter created at the top of
// an hour is gone an hour later however busy it was.
//
// An absent counter starts at zero.
//
// A ttl that has already passed is ErrNoTTL, and this is the one of the three
// writes that keeps its error rather than reinterpreting the ttl. Forgetting
// the key would make Increment(k, 1, 0) delete a counter while reporting that
// it raised one, and treating it as forever would mint exactly the immortal
// counter this signature takes a ttl to prevent.
func (r *Repository) Increment(ctx context.Context, g auth.Grant, key string, delta int64, ttl time.Duration) (int64, error) {
	full, err := r.key(ctx, g, key)
	if err != nil {
		return 0, err
	}
	if ttl <= 0 {
		return 0, ErrNoTTL
	}
	return r.store.Increment(ctx, full, delta, ttl)
}

// Decrement subtracts delta from a counter and returns the new value. It is
// Increment with the sign turned round.
func (r *Repository) Decrement(ctx context.Context, g auth.Grant, key string, delta int64, ttl time.Duration) (int64, error) {
	return r.Increment(ctx, g, key, -delta, ttl)
}

// Forget removes a key. Removing what is not there is not an error.
func (r *Repository) Forget(ctx context.Context, g auth.Grant, key string) error {
	full, err := r.key(ctx, g, key)
	if err != nil {
		return err
	}
	r.eventForgetting(key)
	err = r.store.Forget(ctx, full)
	r.eventForgotten(key, err)
	return err
}

// Delete is Forget under another name.
func (r *Repository) Delete(ctx context.Context, g auth.Grant, key string) error {
	return r.Forget(ctx, g, key)
}

// DeleteMultiple removes several keys.
//
// It stops at the first error, so the caller gets the error itself rather than
// a bool that says something went wrong without saying what.
func (r *Repository) DeleteMultiple(ctx context.Context, g auth.Grant, keys ...string) error {
	for _, key := range keys {
		if err := r.Forget(ctx, g, key); err != nil {
			return err
		}
	}
	return nil
}

// Flush removes every entry of this tenant in this namespace.
//
// It is not "empty the cache". A cache:clear that emptied the store would clear
// every other tenant on the way past, and in a SaaS that is an outage caused by
// a support request.
func (r *Repository) Flush(ctx context.Context, g auth.Grant) error {
	prefix, err := r.prefix(g)
	if err != nil {
		return err
	}

	if r.events != nil {
		r.event(events.NewCacheFlushing(r.GetName(), r.tagNames()))
	}
	err = r.store.Flush(ctx, prefix)
	if r.events != nil {
		if err != nil {
			r.event(events.NewCacheFlushFailed(r.GetName(), r.tagNames()))
		} else {
			r.event(events.NewCacheFlushed(r.GetName(), r.tagNames()))
		}
	}
	return err
}

// Clear is Flush under another name.
func (r *Repository) Clear(ctx context.Context, g auth.Grant) error {
	return r.Flush(ctx, g)
}

// FlushLocks releases every lock the store holds.
//
// A store that cannot flush its locks returns ErrUnsupported rather than
// pretending. Ask SupportsFlushingLocks first if the answer matters.
//
// It takes no Grant because a lock has no tenant: a scheduler lock covers the
// whole instance.
func (r *Repository) FlushLocks(ctx context.Context) error {
	store, ok := r.store.(CanFlushLocks)
	if !ok {
		return fmt.Errorf("%w: flushing locks", ErrUnsupported)
	}

	if r.events != nil {
		r.event(events.NewCacheLocksFlushing(r.GetName()))
	}
	err := store.FlushLocks(ctx)
	if r.events != nil {
		if err != nil {
			r.event(events.NewCacheLocksFlushFailed(r.GetName()))
		} else {
			r.event(events.NewCacheLocksFlushed(r.GetName()))
		}
	}
	return err
}

// Touch gives a live entry a new expiry and reports whether there was one.
//
// A ttl of zero means forever. An absent key is false and is not created:
// touching a miss would turn it into an entry holding nothing.
func (r *Repository) Touch(ctx context.Context, g auth.Grant, key string, ttl time.Duration) (bool, error) {
	full, err := r.key(ctx, g, key)
	if err != nil {
		return false, err
	}
	if ttl <= 0 {
		ttl = foreverTTL
	}

	if store, ok := r.store.(interface {
		Touch(context.Context, string, time.Duration) (bool, error)
	}); ok {
		return store.Touch(ctx, full, ttl)
	}

	// A store that cannot move a deadline can still be told the value again.
	// Not atomic, and it does not need to be: the value it rewrites is the one
	// it just read, so the worst a race costs is a concurrent write being
	// replaced by the value it replaced.
	raw, err := r.store.Get(ctx, full)
	if errors.Is(err, ErrNotFound) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	if err := r.store.Put(ctx, full, raw, ttl); err != nil {
		return false, err
	}
	return true, nil
}

// WithoutOverlapping runs fn while holding a lock, so two of them cannot run at
// once.
//
// lockFor is how long the lock survives a process that dies holding it;
// waitFor is how long this caller is willing to queue before giving up with
// ErrLockTimeout.
//
// The store must be able to hold a lock. One that cannot returns
// ErrUnsupported.
func (r *Repository) WithoutOverlapping(ctx context.Context, g auth.Grant, key string, fn func(context.Context) error, lockFor, waitFor time.Duration) error {
	store, ok := r.store.(Locking)
	if !ok {
		return fmt.Errorf("%w: holding a lock", ErrUnsupported)
	}
	full, err := r.key(ctx, g, key)
	if err != nil {
		return err
	}
	if lockFor <= 0 {
		lockFor = foreverTTL
	}
	return (&Lock{store: store, name: full, ttl: lockFor}).Block(ctx, waitFor, fn)
}

// Tags begins a tagged operation on this repository.
//
// The returned TaggedCache is the whole Repository surface again, with every
// key prefixed by the tag set's generation -- so TaggedCache.Flush is a single
// write per tag that orphans every entry carrying it, rather than a scan.
//
// Every Store can tag: a tag is an ordinary entry holding a generation id, so
// tagging needs Get, Put and Forget and nothing a Store does not already have.
// The error is for the one thing that is still wrong, which is asking for no
// tags at all.
func (r *Repository) Tags(names ...string) (*TaggedCache, error) {
	if len(names) == 0 {
		return nil, errors.New("cache: tags() needs at least one tag, and none of them tags everything")
	}
	tagged := *r
	tagged.tags = NewTagSet(r.store, names)
	return &TaggedCache{Repository: &tagged}, nil
}

// GetName is the name this cache is known by: the name of the store the
// manager built this repository for. It is what goes into every event, so a
// listener can tell which cache a hit came from.
//
// A repository built with New rather than by a CacheManager has no store name,
// and answers with its namespace instead: that is the thing that distinguishes
// one repository over a store from another, and an empty name in an event is
// worth less than an imperfect one.
func (r *Repository) GetName() string {
	if r.name != "" {
		return r.name
	}
	return r.namespace
}

// SetName returns a repository known by another name. It is the name a
// CacheManager stamps on the repositories it builds, and it is a method rather
// than a constructor argument because a repository is derived from another one
// rather than rebuilt.
func (r *Repository) SetName(name string) *Repository {
	out := *r
	out.name = name
	return &out
}

// SupportsTags reports whether this store can carry tags.
//
// Every Store in this package satisfies Taggable, so the answer is yes for all
// of them; a store registered from elsewhere answers for itself.
func (r *Repository) SupportsTags() bool {
	_, ok := r.store.(Taggable)
	return ok
}

// SupportsFlushingLocks reports whether FlushLocks will work.
func (r *Repository) SupportsFlushingLocks() bool {
	_, ok := r.store.(CanFlushLocks)
	return ok
}

// GetDefaultCacheTime is how long an item is stored for when nobody says.
//
// Nothing on this type takes an optional ttl -- Put requires one -- so its one
// job is to be carried into the TaggedCache that Tags builds.
func (r *Repository) GetDefaultCacheTime() time.Duration { return r.defaultTTL }

// SetDefaultCacheTime returns a repository with another default.
//
// It derives rather than mutates, for the reason Namespace derives: a
// repository handed to two modules must not change underneath one of them.
func (r *Repository) SetDefaultCacheTime(ttl time.Duration) *Repository {
	out := *r
	out.defaultTTL = ttl
	return &out
}

// GetStore returns the store underneath.
//
// It is the hatch for a caller that needs something of the backend the
// Repository does not offer -- and reaching through it skips the tenant prefix,
// so whatever is done there is done to every tenant at once.
func (r *Repository) GetStore() Store { return r.store }

// SetStore returns a repository over another store.
//
// It derives rather than mutates, for the reason SetDefaultCacheTime derives.
func (r *Repository) SetStore(s Store) *Repository {
	out := *r
	out.store = s
	return &out
}

// Get reads a value.
//
// It returns ErrNotFound when the key is absent, which the caller is expected
// to treat as "compute it", not as an error to propagate -- and usually the
// caller should be Remember instead.
//
// It is a function and not a method because a method cannot take a type
// parameter in Go. Every generic entry point in this package has the repository
// as its first argument for that reason, and it is the same reason
// slices.SortFunc is not a method.
func Get[T any](ctx context.Context, r *Repository, g auth.Grant, key string) (T, error) {
	var zero T
	full, err := r.key(ctx, g, key)
	if err != nil {
		return zero, err
	}

	r.eventRetrieving(key)
	return get[T](ctx, r, full, key)
}

// get is the read itself, with the hit and miss events but without
// RetrievingKey.
//
// It exists so Many fires RetrievingManyKeys once for the batch instead of a
// RetrievingKey per key.
func get[T any](ctx context.Context, r *Repository, full, key string) (T, error) {
	var zero T

	raw, err := r.store.Get(ctx, full)
	if err != nil {
		// A miss and a store that did not answer are different here -- the error
		// says which -- but CacheMissed fires for both, so a hit rate computed
		// from CacheHit and CacheMissed still adds up.
		r.eventMissed(key)
		return zero, err
	}

	value, err := decode[T](raw)
	if err != nil {
		r.eventMissed(key)
		return zero, err
	}
	r.eventHit(key, value)
	return value, nil
}

// Many reads several values in one call.
//
// Every key asked for is in the result, including the part that surprises
// people: a miss comes back as the zero value of T.
//
// That means Many cannot tell a miss from a cached zero. Get can, and is what
// to reach for when the difference matters.
func Many[T any](ctx context.Context, r *Repository, g auth.Grant, keys ...string) (map[string]T, error) {
	if r.events != nil {
		r.event(events.NewRetrievingManyKeys(r.GetName(), keys, r.tagNames()))
	}

	out := make(map[string]T, len(keys))
	for _, key := range keys {
		full, err := r.key(ctx, g, key)
		if err != nil {
			return nil, err
		}
		v, err := get[T](ctx, r, full, key)
		switch {
		case err == nil:
			out[key] = v
		case errors.Is(err, ErrNotFound):
			// Every key asked for comes back, so a miss needs something to
			// carry, and the zero value of T is what it carries.
			//
			// That means a missing counter and a counter cached at 0 read the
			// same here. A caller that must distinguish them reaches for Get,
			// which reports ErrNotFound.
			var zero T
			out[key] = zero
		default:
			return nil, err
		}
	}
	return out, nil
}

// GetMultiple is Many under another name.
func GetMultiple[T any](ctx context.Context, r *Repository, g auth.Grant, keys []string) (map[string]T, error) {
	return Many[T](ctx, r, g, keys...)
}

// Pull reads a value and forgets it.
//
// It is the one-shot read: a flash value, a token that may be spent once. The
// forget happens after a successful read, so a miss leaves nothing to delete
// and a decode failure leaves the entry for whoever can read it.
func Pull[T any](ctx context.Context, r *Repository, g auth.Grant, key string) (T, error) {
	v, err := Get[T](ctx, r, g, key)
	if err != nil {
		return v, err
	}
	if err := r.Forget(ctx, g, key); err != nil {
		return v, err
	}
	return v, nil
}

// String reads a value that must be a string.
//
// A value that is stored but is not a string is an error naming the key and
// what was there; a key that is absent is ErrNotFound, so a caller can still
// tell "wrong type" from "not cached".
func (r *Repository) String(ctx context.Context, g auth.Grant, key string) (string, error) {
	v, err := Get[any](ctx, r, g, key)
	if err != nil {
		return "", err
	}
	s, ok := v.(string)
	if !ok {
		return "", wrongType(key, "a string", v)
	}
	return s, nil
}

// Integer reads a value that must be an integer.
//
// A numeric string is accepted: "42" is 42. A number with a fractional part is
// not, and neither is anything else.
func (r *Repository) Integer(ctx context.Context, g auth.Grant, key string) (int, error) {
	v, err := Get[any](ctx, r, g, key)
	if err != nil {
		return 0, err
	}
	switch n := v.(type) {
	case json.Number:
		parsed, err := strconv.ParseInt(n.String(), 10, 0)
		if err != nil {
			return 0, wrongType(key, "an integer", v)
		}
		return int(parsed), nil
	case string:
		parsed, err := strconv.ParseInt(n, 10, 0)
		if err != nil {
			return 0, wrongType(key, "an integer", v)
		}
		return int(parsed), nil
	default:
		return 0, wrongType(key, "an integer", v)
	}
}

// Float reads a value that must be a float.
//
// A numeric string is accepted and so is a whole number: 42 is 42.0.
func (r *Repository) Float(ctx context.Context, g auth.Grant, key string) (float64, error) {
	v, err := Get[any](ctx, r, g, key)
	if err != nil {
		return 0, err
	}
	switch n := v.(type) {
	case json.Number:
		parsed, err := strconv.ParseFloat(n.String(), 64)
		if err != nil {
			return 0, wrongType(key, "a float", v)
		}
		return parsed, nil
	case string:
		parsed, err := strconv.ParseFloat(n, 64)
		if err != nil {
			return 0, wrongType(key, "a float", v)
		}
		return parsed, nil
	default:
		return 0, wrongType(key, "a float", v)
	}
}

// Boolean reads a value that must be a boolean.
//
// Strictly a boolean: 1 and "true" are refused, because a cache that quietly
// agreed that 1 means true is a cache that quietly agrees that 2 does.
func (r *Repository) Boolean(ctx context.Context, g auth.Grant, key string) (bool, error) {
	v, err := Get[any](ctx, r, g, key)
	if err != nil {
		return false, err
	}
	b, ok := v.(bool)
	if !ok {
		return false, wrongType(key, "a boolean", v)
	}
	return b, nil
}

// Array reads a value that must be a list.
//
// This is the list only. A cached object is read with Get[map[string]any] and a
// cached struct with Get[T], both of which say more about the value than a bare
// list does.
func (r *Repository) Array(ctx context.Context, g auth.Grant, key string) ([]any, error) {
	v, err := Get[any](ctx, r, g, key)
	if err != nil {
		return nil, err
	}
	list, ok := v.([]any)
	if !ok {
		return nil, wrongType(key, "an array", v)
	}
	return list, nil
}

// Remember returns the cached value, computing and storing it on a miss.
//
// This is the shape that belongs in a service, and having it here is what keeps
// the get-check-compute-put sequence from being written slightly differently in
// every module.
//
// A store that is unreachable is not an outage: anything other than a missing
// tenant falls through and computes, and a failure to store is not a failure to
// answer. The cost of that choice is load, and the alternative is a cache being
// down taking the application with it.
//
// A missing tenant is the exception, and it is deliberate: it is a bug in the
// caller, not a cache miss, and computing anyway would hide it until the day
// the value is wrong.
//
// A computation that came back with nothing is cached, and it is deliberate: a
// miss is ErrNotFound and a stored nil is a value, so a callback returning nil
// twice runs once rather than twice. Keeping "not cached" apart from "cached as
// nothing" is the whole reason ErrNotFound exists -- see Has. What it buys is
// negative caching, which is usually the answer worth caching: "this customer
// has no plan" costs a query to produce and is asked on every request.
//
// The cost is that a callback returning the zero value for a reason it expects
// to change soon is remembered for the whole ttl. Return an error instead when
// that is a failure, because an error is not cached.
//
// There is no stampede protection here: N requests missing the same key all
// compute. Add is the primitive to build it on when something needs it, and it
// is not built in because the fix has a cost -- everybody waiting on one
// computation -- that only some callers want to pay.
func Remember[T any](ctx context.Context, r *Repository, g auth.Grant, key string, ttl time.Duration, compute func(context.Context) (T, error)) (T, error) {
	var zero T

	// Everything the caller can get wrong is checked before anything is
	// computed, because these are the failures Remember must not swallow: a
	// missing tenant or a missing ttl would otherwise turn it into a function
	// that computes every time, which is the bug that looks like a slow
	// database.
	if _, err := r.key(ctx, g, key); err != nil {
		return zero, err
	}
	if ttl <= 0 {
		return zero, ErrNoTTL
	}

	if v, err := Get[T](ctx, r, g, key); err == nil {
		return v, nil
	}

	computed, err := compute(ctx)
	if err != nil {
		return zero, err
	}

	settled, err := roundTrip[T](computed)
	if err != nil {
		return zero, err
	}

	// A failure to store is not a failure to answer.
	_ = r.Put(ctx, g, key, computed, ttl)
	return settled, nil
}

// RememberForever returns the cached value, computing and storing it with no
// expiry on a miss.
//
// See Forever for what "no expiry" means here, and for why it is worth a second
// thought before reaching for it.
func RememberForever[T any](ctx context.Context, r *Repository, g auth.Grant, key string, compute func(context.Context) (T, error)) (T, error) {
	return Remember(ctx, r, g, key, foreverTTL, compute)
}

// Sear is RememberForever, under its shorter name.
func Sear[T any](ctx context.Context, r *Repository, g auth.Grant, key string, compute func(context.Context) (T, error)) (T, error) {
	return RememberForever(ctx, r, g, key, compute)
}

// Flexible returns the cached value, and refreshes it in the background once it
// has gone stale.
//
// fresh is how long the value is served without a thought; stale is how long it
// is still served while a refresh runs behind it. A reader arriving after fresh
// has passed gets the old value immediately and pays nothing; one arriving
// after stale has passed computes.
//
// This is the answer to a cache stampede on a value that is expensive and not
// exact -- a dashboard total, a count of things. It is not the answer for a
// value that must be right, because between fresh and stale it is by
// construction not.
//
// The refresh takes a lock so that N readers arriving in the stale window
// produce one recomputation and not N; a store that cannot hold a lock refreshes
// without one, which is worse but is not wrong. It runs on a context detached
// from the caller's, so the request that triggered it can return.
func Flexible[T any](ctx context.Context, r *Repository, g auth.Grant, key string, fresh, stale time.Duration, compute func(context.Context) (T, error)) (T, error) {
	var zero T

	if _, err := r.key(ctx, g, key); err != nil {
		return zero, err
	}
	if fresh <= 0 || stale <= 0 {
		return zero, ErrNoTTL
	}
	if fresh > stale {
		return zero, fmt.Errorf("cache: the flexible entry %q is fresh for %s and kept for %s, so it is stale before it is written", key, fresh, stale)
	}

	createdKey := flexibleCreatedKey(key)

	value, valueErr := Get[T](ctx, r, g, key)
	created, createdErr := Get[int64](ctx, r, g, createdKey)

	// Either half missing is a miss: the value without its timestamp cannot be
	// aged, and the timestamp without its value has nothing to serve.
	if valueErr != nil || createdErr != nil {
		if valueErr != nil && !errors.Is(valueErr, ErrNotFound) {
			return zero, valueErr
		}
		return flexibleRefresh(ctx, r, g, key, createdKey, stale, compute)
	}

	if time.Unix(created, 0).Add(fresh).After(time.Now()) {
		return value, nil
	}

	// Stale. Serve what there is and put the recomputation behind the caller.
	go func() {
		background := context.WithoutCancel(ctx)
		refresh := func(ctx context.Context) error {
			// Somebody else may have refreshed between the read and the lock.
			// The timestamp says so, and re-reading it is what stops N stale
			// readers from each doing the work once the lock frees.
			if now, err := Get[int64](ctx, r, g, createdKey); err == nil && now != created {
				return nil
			}
			_, err := flexibleRefresh(ctx, r, g, key, createdKey, stale, compute)
			return err
		}

		store, ok := r.store.(Locking)
		if !ok {
			_ = refresh(background)
			return
		}
		full, err := r.key(background, g, flexibleLockKey(key))
		if err != nil {
			return
		}
		_ = (&Lock{store: store, name: full, ttl: stale}).Run(background, refresh)
	}()

	return value, nil
}

// flexibleRefresh computes the value and writes it next to its timestamp.
func flexibleRefresh[T any](ctx context.Context, r *Repository, g auth.Grant, key, createdKey string, stale time.Duration, compute func(context.Context) (T, error)) (T, error) {
	var zero T

	computed, err := compute(ctx)
	if err != nil {
		return zero, err
	}
	settled, err := roundTrip[T](computed)
	if err != nil {
		return zero, err
	}

	// The value first. A timestamp written before the value it dates would let
	// the next reader decide a value it cannot see is fresh.
	if err := r.Put(ctx, g, key, computed, stale); err == nil {
		_ = r.Put(ctx, g, createdKey, time.Now().Unix(), stale)
	}
	return settled, nil
}

// flexibleCreatedKey names the companion entry holding when the value was
// computed. The tenant and the namespace are already in front of it, so the
// prefix does not repeat them.
func flexibleCreatedKey(key string) string { return "flexible:created:" + key }

// flexibleLockKey names the lock one refresher holds.
func flexibleLockKey(key string) string { return "flexible:" + key }

// ItemKey formats the key an item is really stored under.
//
// It is not the identity: the tenant, the namespace and -- on a tagged
// repository -- the tag generation are all in front of the caller's key, and
// this is where a caller finds out what that came to.
//
// Something eventually has to look in the store and find out where an entry
// went, and this is the one honest way to ask. It takes a context because a
// tagged repository reads, and sometimes mints, the tag generations before it
// knows the answer.
func (r *Repository) ItemKey(ctx context.Context, g auth.Grant, key string) (string, error) {
	return r.key(ctx, g, key)
}

// key builds the full key: cache:<tenant>:<namespace>:<key>
//
// The tenant sits in the key rather than in a separate store, because stores
// are a fixed small number and tenants are not.
//
// The leading segment is what keeps the three key spaces in one store disjoint:
// entries under "cache:", locks under "lock:", counters under "ratelimit:". It
// is not decoration -- without it, a tenant named "ratelimit" would own the
// keys the rate limiter counts in, and it is a perfectly legal tenant name.
//
// The separator is a colon, so a tenant or a namespace containing one could
// name another's entry: tenant "acme:user" with namespace "session" builds the
// same string as tenant "acme" with namespace "user" and a key beginning
// "session:". Both are checked. The application's key is not, because it is the
// last segment and nothing follows it to be impersonated.
//
// It takes a context because a tagged repository has to read -- and sometimes
// write -- the tag generations before it knows what the key is. An untagged one
// does not touch the store here.
func (r *Repository) key(ctx context.Context, g auth.Grant, key string) (string, error) {
	prefix, err := r.prefix(g)
	if err != nil {
		return "", err
	}
	if key == "" {
		return "", errors.New("cache: an empty key is not a key")
	}
	if r.tags != nil {
		tagged, err := r.tags.taggedItemKey(ctx, g, key)
		if err != nil {
			return "", err
		}
		return prefix + tagged, nil
	}
	return prefix + key, nil
}

// prefix builds everything up to the caller's key, trailing colon included.
func (r *Repository) prefix(g auth.Grant) (string, error) {
	tenant := auth.Tenant(g)
	if tenant == "" {
		return "", ErrNoTenant
	}
	if !auth.ValidTenant(tenant) {
		return "", fmt.Errorf("%w: %q cannot be part of a key", ErrNoTenant, tenant)
	}
	if !auth.ValidTenant(r.namespace) {
		return "", fmt.Errorf("cache: %q is not a namespace: it is a segment of a key, so it is limited to lowercase letters, digits, - and _, up to 64 characters", r.namespace)
	}
	return "cache:" + tenant + ":" + r.namespace + ":", nil
}

// encode is the one codec. JSON, because a cached value crosses processes and
// versions of the binary, and a Go-specific encoding makes the entry written by
// the old replica unreadable by the new one during a rollout.
func encode(v any) ([]byte, error) {
	raw, err := json.Marshal(v)
	if err != nil {
		return nil, fmt.Errorf("cache: encoding the value: %w", err)
	}
	return raw, nil
}

// decode is encode's other half.
//
// Numbers come back as json.Number and not float64 when the target is any, so
// String, Integer and Float can tell 1 from 1.0 and from "1" -- a distinction a
// float64 has already thrown away.
func decode[T any](raw []byte) (T, error) {
	var v T
	dec := json.NewDecoder(bytes.NewReader(raw))
	dec.UseNumber()
	if err := dec.Decode(&v); err != nil {
		return v, fmt.Errorf("cache: decoding the value: %w", err)
	}
	return v, nil
}

// roundTrip sends a computed value through the codec, so a hit and a miss leave
// the caller with exactly the same value.
//
// Otherwise the first request after a deploy behaves differently from the rest,
// which is the worst kind of bug to reproduce. json turns a time.Time into a
// string and back into a time.Time; it turns a map[K]V with a non-string K into
// an error, and the caller should learn that on the miss, not on the hit.
func roundTrip[T any](v T) (T, error) {
	var zero T
	raw, err := encode(v)
	if err != nil {
		return zero, err
	}
	return decode[T](raw)
}

// wrongType is what String, Integer, Float, Boolean and Array say about a value
// that is cached but is not what was asked for.
//
// It names the key, because the caller has usually built it from an id and
// cannot see it in the source; and it names what was found, because "expected a
// string" without "found a number" sends people to read the writer instead of
// the value.
func wrongType(key, want string, got any) error {
	return fmt.Errorf("cache: the value for %q must be %s, and it is %s", key, want, describe(got))
}

// describe names a decoded JSON value the way somebody reading a log would.
func describe(v any) string {
	switch t := v.(type) {
	case nil:
		return "null"
	case bool:
		return "a boolean"
	case json.Number:
		return "the number " + t.String()
	case string:
		return "the string " + strconv.Quote(t)
	case []any:
		return "an array"
	case map[string]any:
		return "an object"
	default:
		return fmt.Sprintf("%T", v)
	}
}
