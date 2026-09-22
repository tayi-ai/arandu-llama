package cache

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strconv"
	"sync"
	"time"
)

// Limit is how many attempts are allowed, over what period, against what key.
//
// It is a value and it is built by composition -- PerMinute(5).By(ip) -- so a
// route declares its limit once, as data, and the same value is what the
// middleware counts against and what the error message is written from.
//
// It lives here, next to the RateLimiter that counts against it, and not in a
// ratelimiting subpackage: a subpackage is a real boundary, and one struct on
// the far side of one is an import for no gain.
//
// # It is three fields, and it carries no callbacks
//
// It carried two: one that decided, from the status a handler wrote, whether
// an attempt counted, and one that wrote the refusal itself. Nothing in the
// collection read either of them. A field that is documented as behaviour and
// is never read is worse than an absent one -- writing a limit that counts
// only failures compiled, ran, and counted everything, and the reason was
// invisible to whoever wired it.
//
// They do not come back as working code either, because each already has one
// spelling. The refusal is written by the Refuse passed to the throttle
// middleware, which is one place for every refusal a request layer makes; a
// second one carried on the limit would be two. And giving an attempt back
// once the work is known to have succeeded is Release, which is a call the
// caller makes when it knows the answer, not a callback the counter invokes.
type Limit struct {
	// Key identifies the caller being limited: an IP, a session id, an account.
	// It is not a tenant: rate limiting happens before authentication on the
	// routes where it matters most, and there is no Grant yet.
	Key string

	// MaxAttempts is how many are allowed within Decay.
	MaxAttempts int

	// Decay is the length of the window.
	Decay time.Duration
}

// PerSecond, PerMinute, PerHour and PerDay build the limits anybody writes.
//
// They are four functions and not one with a unit argument, because the call
// site reads as the sentence somebody says out loud: "five per minute".
//
// A window several units long is PerMinutes, so the common call stays one
// argument long.
func PerSecond(max int) Limit { return Limit{MaxAttempts: max, Decay: time.Second} }

// PerMinute is a limit over one minute.
func PerMinute(max int) Limit { return Limit{MaxAttempts: max, Decay: time.Minute} }

// PerMinutes is a limit over several minutes.
//
// Mind the argument order: the window comes first and the count second, which
// is the opposite of every other constructor here.
func PerMinutes(decayMinutes, max int) Limit {
	return Limit{MaxAttempts: max, Decay: time.Duration(decayMinutes) * time.Minute}
}

// PerHour is a limit over one hour.
func PerHour(max int) Limit { return Limit{MaxAttempts: max, Decay: time.Hour} }

// PerDay is a limit over one day.
func PerDay(max int) Limit { return Limit{MaxAttempts: max, Decay: 24 * time.Hour} }

// None is the limit that allows everything.
//
// There is no separate Unlimited or GlobalLimit type: they would be two more
// ways to spell the same value.
//
// It is for the branch that has to return a limit and has nothing to limit --
// a named limiter that decides, per caller, that this one is exempt. A route
// with no limit on it is written by not putting one there.
func None() Limit { return Limit{MaxAttempts: math.MaxInt32, Decay: time.Minute} }

// By returns the same limit against a key.
func (l Limit) By(key string) Limit {
	l.Key = key
	return l
}

// FallbackKey is a key for this limit that no other limit in the same set can
// collide with.
//
// It answers Limit::fallbackKey(). It exists for one situation, and Limiter is
// the only caller: a named limiter that returns several limits which all ended
// up with the same key -- "sixty a minute and a thousand an hour, both by IP" --
// would have the two counting in the same counter, and the hourly one would be
// spent in a minute. Appending the shape of the limit separates them.
func (l Limit) FallbackKey() string {
	prefix := ""
	if l.Key != "" {
		prefix = l.Key + ":"
	}
	return prefix + "attempts:" + strconv.Itoa(l.MaxAttempts) +
		":decay:" + strconv.FormatInt(int64(l.Decay/time.Second), 10)
}

// RateLimiter counts attempts against limits.
//
// It replaces two implementations that disagreed: an in-memory one in the HTTP
// middleware, which counted per process -- so N replicas allowed N times the
// limit, on the one endpoint where that gap is worth exploiting -- and a second
// one in the former distributed adapter. There is one now, and which store it counts in is
// wiring.
//
// It does not decide what happens when the store is unreachable. It reports the
// error and the caller chooses: the HTTP throttle middleware fails open,
// because a rate limiter that is down must not become an outage, and a sign-in
// throttle may well choose the opposite. That decision was previously buried
// inside the limiter, where the middleware could not see it and could not
// change it.
//
// # No method here takes a Grant, and none should
//
// Every Repository method in this package takes one, and takes the tenant out
// of it. This type is the exception on purpose, and making it consistent with
// its neighbour would break the routes it exists to protect: a rate limit runs
// before authentication precisely where it matters most -- the sign-in form,
// the password reset, the public API key check -- and at that point in the
// request nobody has been identified, so there is no Grant to take a tenant
// from. A signature that demanded one would be a signature no caller on those
// routes could satisfy, and the limit would come off the routes rather than the
// requirement coming off the signature.
//
// What a caller wants a tenant in the key for, it already has: Limit.Key is a
// string, so a limit that really is per tenant is written by putting the tenant
// in the key it is built with.
type RateLimiter struct {
	store Store

	mu sync.Mutex
	// limiters is what For registered, by name. It is a map read on every
	// request and written at wiring time, so it is behind a mutex: a route
	// registered from a goroutine while another serves a request would
	// otherwise be a data race with no symptom until production.
	limiters map[string]LimitResolver
}

// LimitResolver decides, per caller, what the limit is.
//
// It receives the context, which is where the request is, and it returns a set
// rather than one limit: "sixty a minute and a thousand an hour" is two limits
// against the same caller, and both are counted. A resolver with nothing to
// limit returns None, or nothing at all.
type LimitResolver func(ctx context.Context) []Limit

// NewRateLimiter returns the limiter.
func NewRateLimiter(s Store) *RateLimiter {
	return &RateLimiter{store: s, limiters: map[string]LimitResolver{}}
}

// For registers a named limiter and returns the rate limiter.
//
// It answers RateLimiter::for(). It is what a route refers to by name, so the
// limit for "login" is written once, in the wiring, rather than repeated at
// every route that needs it.
func (rl *RateLimiter) For(name string, resolver LimitResolver) *RateLimiter {
	rl.mu.Lock()
	defer rl.mu.Unlock()

	if rl.limiters == nil {
		rl.limiters = map[string]LimitResolver{}
	}
	rl.limiters[name] = resolver
	return rl
}

// Limiter returns the named limiter, or nil.
//
// It answers RateLimiter::limiter(), including the part of the body that is the
// reason it wraps rather than returns the closure: two limits that resolved to
// the same key would count in the same counter, and the longer window would be
// spent by the shorter one. The duplicates are given FallbackKey instead, which
// separates them by shape.
func (rl *RateLimiter) Limiter(name string) LimitResolver {
	rl.mu.Lock()
	resolver, ok := rl.limiters[name]
	rl.mu.Unlock()

	if !ok || resolver == nil {
		return nil
	}

	return func(ctx context.Context) []Limit {
		limits := resolver(ctx)
		if len(limits) < 2 {
			return limits
		}

		seen := make(map[string]int, len(limits))
		for _, limit := range limits {
			seen[limit.Key]++
		}
		out := make([]Limit, 0, len(limits))
		for _, limit := range limits {
			if seen[limit.Key] > 1 {
				limit.Key = limit.FallbackKey()
			}
			out = append(out, limit)
		}
		return out
	}
}

// Result is what one Attempt did.
//
// It is a struct and not three return values because the three are read
// together: the throttle middleware writes all of them into response headers,
// and a signature that returned them loose would be four values with an error.
type Result struct {
	// OK says whether this attempt was within the limit. An attempt that was
	// not is still counted -- the flood is the thing being measured.
	OK bool

	// Attempts is how many have been counted in this window, including this
	// one.
	Attempts int

	// Remaining is how many are left, never below zero.
	Remaining int

	// RetryAfter is how long until the window rolls. It is zero when OK.
	RetryAfter time.Duration
}

// Attempt counts one attempt and says whether it fits.
//
// This is the call a throttle makes, and it counts before it answers on
// purpose. A limiter that answered first and counted afterwards would let
// everything that arrived in between through, and the endpoint where that
// matters -- sign-in -- is the one where the budget has to be taken before the
// password is checked, not after.
func (rl *RateLimiter) Attempt(ctx context.Context, l Limit) (Result, error) {
	attempts, err := rl.Hit(ctx, l)
	if err != nil {
		return Result{}, err
	}

	remaining := l.MaxAttempts - attempts
	if remaining < 0 {
		remaining = 0
	}
	if attempts > l.MaxAttempts {
		return Result{Attempts: attempts, Remaining: 0, RetryAfter: rl.AvailableIn(l)}, nil
	}
	return Result{OK: true, Attempts: attempts, Remaining: remaining}, nil
}

// Hit counts one attempt and returns how many there have been in this window.
//
// It is the primitive Attempt is built on, and it is exported because a caller
// that already knows it is going to refuse -- a sign-in that failed, an upload
// that was rejected -- wants to count without asking a question it has already
// answered.
func (rl *RateLimiter) Hit(ctx context.Context, l Limit) (int, error) {
	key, err := rl.key(l)
	if err != nil {
		return 0, err
	}
	n, err := rl.store.Increment(ctx, key, 1, l.Decay)
	if err != nil {
		return 0, err
	}
	return int(n), nil
}

// Release gives one attempt back.
//
// It is what a success does to a failure counter: five wrong passwords lock the
// account for a minute, and the right one on the second try should not leave
// four attempts standing against somebody who is who they say they are. Use
// Clear to give all of them back.
//
// A counter that has already expired stays absent rather than being recreated
// at minus one, and a counter driven below zero is clamped back -- both are the
// same rule, which is that a window nobody has attempted anything in holds
// nothing.
func (rl *RateLimiter) Release(ctx context.Context, l Limit) error {
	key, err := rl.key(l)
	if err != nil {
		return err
	}

	switch _, err := rl.store.Get(ctx, key); {
	case errors.Is(err, ErrNotFound):
		return nil
	case err != nil:
		return err
	}

	n, err := rl.store.Increment(ctx, key, -1, l.Decay)
	if err != nil {
		return err
	}
	if n < 0 {
		// Somebody else released between the read and the decrement. Put the
		// counter back on the floor rather than leaving a negative one, which
		// would hand out free attempts for the rest of the window.
		if _, err := rl.store.Increment(ctx, key, -n, l.Decay); err != nil {
			return err
		}
	}
	return nil
}

// Attempts is how many have been counted in the current window.
//
// A window nobody has attempted anything in is zero and not an error: a
// counter that is not there is a counter at zero.
func (rl *RateLimiter) Attempts(ctx context.Context, l Limit) (int, error) {
	key, err := rl.key(l)
	if err != nil {
		return 0, err
	}
	switch raw, err := rl.store.Get(ctx, key); {
	case errors.Is(err, ErrNotFound):
		return 0, nil
	case err != nil:
		return 0, err
	default:
		n, perr := strconv.Atoi(string(raw))
		if perr != nil {
			return 0, errNotACounter(key)
		}
		return n, nil
	}
}

// TooManyAttempts reports whether the budget for this window is spent.
//
// It asks without counting, which is the difference from Attempt: use this to
// decide whether to offer something, and Attempt to actually spend one.
//
// There is no separate lockout timer to expire: the window is in the counter's
// key, so a counter that outlived its window is not the counter being asked
// about.
func (rl *RateLimiter) TooManyAttempts(ctx context.Context, l Limit) (bool, error) {
	attempts, err := rl.Attempts(ctx, l)
	if err != nil {
		return false, err
	}
	return attempts >= l.MaxAttempts, nil
}

// Remaining is how many attempts are left in this window, never below zero.
func (rl *RateLimiter) Remaining(ctx context.Context, l Limit) (int, error) {
	attempts, err := rl.Attempts(ctx, l)
	if err != nil {
		return 0, err
	}
	return max(0, l.MaxAttempts-attempts), nil
}

// RetriesLeft is Remaining.
func (rl *RateLimiter) RetriesLeft(ctx context.Context, l Limit) (int, error) {
	return rl.Remaining(ctx, l)
}

// ResetAttempts forgets the attempts in the current window.
//
// It forgets all of it: the window is in the counter's key, so there is no
// second thing to forget.
func (rl *RateLimiter) ResetAttempts(ctx context.Context, l Limit) error {
	return rl.Clear(ctx, l)
}

// Clear forgets the attempts in the current window.
func (rl *RateLimiter) Clear(ctx context.Context, l Limit) error {
	key, err := rl.key(l)
	if err != nil {
		return err
	}
	return rl.store.Forget(ctx, key)
}

// AvailableIn is how long until the current window rolls.
//
// It is arithmetic and not a round trip, which is what the bucketed key buys:
// the window a counter belongs to is in its name, so when it ends is known
// without asking the store how long the key has left.
func (rl *RateLimiter) AvailableIn(l Limit) time.Duration {
	if l.Decay <= 0 {
		return 0
	}
	now := time.Now().UnixNano()
	return time.Duration(int64(l.Decay) - now%int64(l.Decay))
}

// key buckets the counter by window, so a new window is a new key and expiry is
// the only cleanup there is.
//
// The window is fixed rather than sliding: one increment, no script, no sorted
// set. A sliding window would be more precise and would need either a server
// script -- which is what keeps every RESP backend interchangeable -- or a set
// per key. Precision is not what a rate limit is for; stopping a flood is, and
// a fixed window stops it.
//
// The known cost is the boundary: a caller can spend the whole budget at the end
// of one window and the whole budget at the start of the next. For sign-in
// throttling that is two bursts instead of one, which is not the attack anybody
// is worried about.
func (rl *RateLimiter) key(l Limit) (string, error) {
	if l.Key == "" {
		return "", errors.New("cache: a rate limit needs a key, and an empty one limits every caller together")
	}
	if l.Decay <= 0 {
		return "", fmt.Errorf("%w: the rate limit on %q has no window, and a limit with no window never resets", ErrNoTTL, l.Key)
	}
	if l.MaxAttempts <= 0 {
		return "", fmt.Errorf("cache: the rate limit on %q allows %d attempts, which is a closed door and not a limit", l.Key, l.MaxAttempts)
	}
	bucket := time.Now().UnixNano() / int64(l.Decay)
	return "ratelimit:" + rl.CleanRateLimiterKey(l.Key) + ":" + strconv.FormatInt(bucket, 10), nil
}
