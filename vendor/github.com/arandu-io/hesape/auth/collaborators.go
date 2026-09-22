package auth

import (
	"context"
	"net/http"
	"time"
)

// This file holds what the guards need from the rest of the framework: the
// session the user id is kept in, the cookie jar the recaller goes into, the
// event dispatcher, the request, and the timebox that makes a failed sign-in
// take as long as a successful one.
//
// They are interfaces, declared here in the consuming package, for the reason
// contracts.go gives for Hasher: the root of auth imports nothing but the
// standard library, because everything that scopes itself by tenant imports it
// to read Tenant off a Grant. The signatures below are the ones hesape/session,
// hesape/cookie, hesape/events and hesape/http already have, method for method,
// so those packages satisfy these structurally without either side importing
// the other.
//
// Each is the part of a collaborator that authentication touches, never the
// whole of it.

// Session is the part of a session store the guards use.
//
// SessionGuard writes the authenticated user's id under [SessionGuard.GetName]
// and regenerates the session id on the way in, which is the defence against a
// session fixed by whoever handed the browser its first cookie.
//
// The four signatures are hesape/session.Store's, so *session.Store is a
// Session with nothing in between.
type Session interface {
	// Get is the value under key, or the first def when there is none.
	Get(key string, def ...any) any

	// Put writes value under key.
	Put(key string, value any)

	// Remove forgets the key and returns what was there.
	Remove(key string) any

	// Regenerate gives the session a new id, destroying the old one when destroy
	// is true. It takes a context because the store it writes through is a
	// database or a cache.
	Regenerate(ctx context.Context, destroy bool) error
}

// CookieJar is the part of a cookie jar that SessionGuard uses to carry the
// "remember me" cookie.
//
// It is five methods and no more: Make and Queue put the recaller on the
// response, Unqueue and Forget take it off on the way out, and HasQueued is how
// [SessionGuard.LogoutOtherDevices] knows a recaller was already queued during
// this request and has to be reissued with the new password hash.
//
// The signatures are hesape/cookie.CookieJar's, which is why they carry the
// full attribute list rather than the three arguments SessionGuard passes. The
// cookie is a *net/http.Cookie: net/http is the standard library, so naming it
// here costs the root of auth no dependency.
type CookieJar interface {
	// Make builds a cookie with these attributes.
	Make(name, value string, minutes int, path, domain string, secure *bool, httpOnly, raw bool, sameSite http.SameSite) *http.Cookie

	// Queue puts a cookie on the response being built.
	Queue(cookie *http.Cookie)

	// Unqueue takes a queued cookie back off it.
	Unqueue(name, path string)

	// Forget builds the cookie that tells the browser to drop it.
	Forget(name, path, domain string) *http.Cookie

	// HasQueued reports that a cookie of this name is already queued.
	HasQueued(key, path string) bool
}

// Dispatcher is the part of an event dispatcher that SessionGuard uses: it
// fires the eight authentication events and it registers the listener
// [SessionGuard.Attempting] takes.
//
// The signatures are hesape/events.Dispatcher's. An event is dispatched as a
// value and a listener is registered against a value of the same type, which is
// how that dispatcher names an event that is not a string.
type Dispatcher interface {
	// Dispatch fires an event and returns whatever the listeners answered.
	Dispatch(event any, payload ...any) []any

	// Listen registers listeners for an event.
	Listen(events any, listener ...any)
}

// Request is the part of the HTTP request the guards read.
//
// SessionGuard reads the recaller cookie and the HTTP Basic pair; TokenGuard
// reads the bearer token and the Basic password; RequestGuard hands the whole
// thing to the callback it was built with, and Query and Input are here for that
// callback, because no guard reads either. Nothing here reads a tenant, and
// nothing here may: the tenant is on the Grant a policy minted, never on the
// request.
//
// Query, Input, BearerToken and Cookie are hesape/http.Request's signatures.
// GetUser, GetPassword and Context are not on it yet:
//
//   - GetUser and GetPassword are the HTTP Basic pair. hesape/http.Request has
//     a User method, but that one is the authenticated user and not the Basic
//     username, so these two carry names of their own and the pair is never
//     confused with it.
//   - Context is the request's own. The Guard contract gives User and ID no
//     context.Context, and both of them read the user store, so the guard takes
//     it from here. Cancel the request and the lookup it started stops.
type Request interface {
	// Query is the value of a query string key, or the first def.
	Query(key string, def ...string) string

	// Input is the value of a key from the query string or the body.
	Input(key string, def ...any) any

	// BearerToken is the token on the Authorization: Bearer header.
	BearerToken() string

	// Cookie is the value of a request cookie, or the first def.
	Cookie(name string, def ...string) string

	// GetUser is the HTTP Basic username.
	GetUser() string

	// GetPassword is the HTTP Basic password.
	GetPassword() string

	// Context is the request's context. See the type's doc for why it is here.
	Context() context.Context
}

// Timebox runs a callback and does not come back before the given number of
// microseconds has passed.
//
// It is why [SessionGuard.Attempt] with an address nobody has registered takes
// as long as one with the wrong password. Without it, the time an attempt took
// says whether the account exists, and that is a list of your customers
// readable by anyone with a stopwatch.
//
// hesape/support has the full implementation. This is an interface rather than
// that type because the root of auth imports nothing but the standard library,
// and [NewTimebox] is what SessionGuard falls back to.
type Timebox interface {
	// Call runs the callback and then waits out the remainder of microseconds.
	// The callback is handed the timebox, so that it can ask to return early. An
	// error it returns is held until the box has been waited out, and then
	// returned.
	Call(callback func(timebox Timebox) (any, error), microseconds int) (any, error)

	// ReturnEarly asks the box not to wait out the remainder.
	ReturnEarly() Timebox

	// DontReturnEarly undoes that, and the box waits again.
	DontReturnEarly() Timebox
}

// NewTimebox returns the timebox [NewSessionGuard] falls back to when it is
// handed none.
//
// The returned timebox does not reset itself between calls: once something has
// asked it to return early it keeps returning early, so a guard is given a
// fresh one per request.
func NewTimebox() Timebox {
	return &plainTimebox{}
}

// plainTimebox is [NewTimebox]'s Timebox, waiting out the remainder with
// time.Sleep.
type plainTimebox struct {
	// earlyReturn records that something asked not to wait out the remainder.
	earlyReturn bool
}

// Call runs the callback and then waits out the remainder of microseconds.
func (t *plainTimebox) Call(callback func(timebox Timebox) (any, error), microseconds int) (any, error) {
	start := time.Now()

	result, err := callback(t)

	remainder := time.Duration(microseconds)*time.Microsecond - time.Since(start)

	if !t.earlyReturn && remainder > 0 {
		time.Sleep(remainder)
	}

	if err != nil {
		return nil, err
	}
	return result, nil
}

// ReturnEarly asks the box not to wait out the remainder.
func (t *plainTimebox) ReturnEarly() Timebox {
	t.earlyReturn = true
	return t
}

// DontReturnEarly undoes that, and the box waits again.
func (t *plainTimebox) DontReturnEarly() Timebox {
	t.earlyReturn = false
	return t
}
