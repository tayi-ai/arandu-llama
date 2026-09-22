package session

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha256"
	"encoding/base64"
	"errors"
	"net/http"
	"strings"
	"sync"
	"time"
)

// CookieName is the cookie the framework reads and writes. It is fixed on
// purpose: a configurable cookie name buys nothing and breaks the CSRF binding
// when two parts of a project disagree about it.
const CookieName = "arandu_session"

// Errors returned by Store.
var (
	// ErrNoSession means the request carries no session cookie, or the cookie
	// signature does not match the application key.
	ErrNoSession = errors.New("session: no session")
	// ErrExpired means the session id is well formed but the handler no longer
	// holds it -- expired, or destroyed by a logout elsewhere.
	ErrExpired = errors.New("session: expired")
	// ErrConfirmationNotStored means the handler accepted the password
	// confirmation stamp and did not keep it, so no window would ever be
	// satisfied and the password screen would ask again immediately.
	//
	// It is a defect in the handler, not in the request: the fix is to carry
	// Record.PasswordConfirmedAt in whatever shape that handler stores. A caller
	// that receives it must report a failure rather than redirect, because
	// redirecting is the loop.
	ErrConfirmationNotStored = errors.New("session: the handler did not keep the password confirmation stamp")
)

// errNoSubjectScope is the one refusal behind every bulk sign-out, stated once
// so the store and the handlers cannot drift apart on it.
//
// Without an id and a tenant there is no question to ask: "every session of
// subject 1" with no tenant reaches every customer, and "every session of
// the empty subject" of one tenant is every session nobody has signed in on
// -- the guests. Found by audit: RecordStore.DestroyOthers refused both, and the
// in-memory handler's DestroyIndex, which is exported and reachable without the
// store, happily deleted every guest session of a tenant while the Redis handler
// answered the same call with an error. Two handlers that disagree is a bug that
// only appears when the application switches to shared storage.
var errNoSubjectScope = errors.New("session: signing out the other sessions needs a subject with an id and a tenant")

// RememberLifetime is how long a session started with Remember(true) lives.
//
// Longer than a working session, and deliberately not unlimited. The cookie is a
// bearer credential sitting on a device that gets shared, lost, resold and
// borrowed, so "stay signed in" has to end on its own: an unlimited lifetime
// means a laptop sold years later still opens the account. A month is long
// enough for the box to be worth ticking -- that is the whole point of it --
// and short enough that a device which left the person's hands stops working
// inside a billing cycle, where somebody notices.
//
// A store configured with a longer ttl than this keeps its own: see
// RecordStore.lifetime. Remember must never make a session shorter.
const RememberLifetime = 30 * 24 * time.Hour

// PasswordConfirmationWindow is how long typing the password again counts for.
//
// Three hours, a constant for the same reason the sign-in throttle's ceiling
// is one: a window somebody can widen from the environment is a window
// somebody widens the afternoon it is inconvenient, and nobody narrows it
// again.
//
// The number is chosen from both ends. Long enough that somebody spending an
// afternoon in the sensitive part of an application types their password once
// rather than at every step, because a check people route around is not a check.
// Short enough that a machine left unlocked overnight asks whoever sits down at
// it in the morning -- which is the situation the confirmation exists for, and
// the one a session lifetime alone never catches.
//
// It is read by the middleware that guards a sensitive action and by anything
// else asking Record.PasswordConfirmedWithin, so the whole application agrees on
// one answer to "recently".
const PasswordConfirmationWindow = 3 * time.Hour

// Record is one session as it is stored: what the application keeps in it, plus
// the four things the store itself has to remember.
//
// The payload is generic because this package has no business knowing who is
// signed in -- that is auth's, and auth builds the Record it hands to Start. The
// rest is not payload and must not be pushed into it:
//
//   - Tenant and SubjectID are the index. They are what makes "sign this account
//     out everywhere" answerable at all, and a handler cannot derive them from an
//     opaque payload.
//   - Remembered and PasswordConfirmedAt decide how long the record and its
//     cookie live. Start writes both, over whatever the caller left there.
type Record[T any] struct {
	// Payload is whatever the application keeps in the session.
	Payload T

	// Tenant is the customer this session belongs to. It comes from the Grant or
	// from the session, never from the request.
	Tenant string

	// SubjectID is the account signed in on this session, empty for a guest.
	//
	// It is a string rather than an auth type because a handler stores it as an
	// index key and this package does not import auth: session is below it.
	SubjectID string

	// Remembered says whether this session was started with the remember-me box
	// ticked, and therefore lives for RememberLifetime instead of the store's
	// configured ttl.
	//
	// It is on the record and not recomputed, because the store rewrites the
	// record when the password is confirmed and has to write back the same
	// lifetime it started with. Without it, confirming a password on a remembered
	// session silently cut it down to the plain ttl -- somebody who ticked the box
	// and then confirmed a payment was signed out that evening.
	//
	// Only Start and Regenerate set it, from the Remember option, and they
	// overwrite whatever the caller put here: a field set by hand on the way in
	// would be a second way to ask for a longer session, and there is one.
	//
	// A policy may also read it: a session nobody has authenticated for a month
	// is the right moment to ask for the password again before a destructive
	// action.
	Remembered bool

	// PasswordConfirmedAt is when the subject last typed their password again on
	// an already open session. Zero means never, and reads as not confirmed.
	// Only Confirm sets it.
	PasswordConfirmedAt time.Time
}

// PasswordConfirmedWithin reports whether the password was typed again on
// this session less than window ago.
//
// It answers false whenever it cannot prove otherwise, which is the whole
// argument for having it be a method rather than a comparison at the call site:
//
//   - No stamp is not confirmed. A session written by an older binary, or by a
//     handler that does not carry the field yet, has the zero time -- and the
//     reading that costs somebody one password screen is the correct one, while
//     the reading that treats an absent stamp as recent waves every session that
//     survived a deploy straight past the check.
//   - A stamp in the future is not confirmed either. It is a clock that moved or
//     a record that was tampered with, and neither is proof that a person was
//     there.
//   - A window of zero or less is not confirmed, so "no window configured" cannot
//     read as "always confirmed".
func (rec Record[T]) PasswordConfirmedWithin(window time.Duration) bool {
	if window <= 0 || rec.PasswordConfirmedAt.IsZero() {
		return false
	}
	elapsed := time.Since(rec.PasswordConfirmedAt)
	return elapsed >= 0 && elapsed < window
}

// Handler stores the record behind a session id.
//
// Read, Write and Destroy are keyed by the id alone. That is the separation
// between one session and another: the id is unguessable and it is the entire
// key, so an implementation has nothing to scope and no second term to get
// wrong. An implementation must not add a customer prefix to it -- the tenant is
// a field of the Record the read returns, so a prefixed key would have to name
// the tenant before the read that discovers it, and two customers whose ids
// differ are already disjoint.
//
// DestroyIndex is the exception, and it is the shape of the rule: it is the one
// method that asks by attribute instead of by id, so it is the one method that
// takes the tenant, and it refuses an empty one.
//
// The core ships ArrayHandler only, which is enough for development and for a
// single instance. A Redis-backed handler provides shared storage and active
// invalidation across application instances.
type Handler[T any] interface {
	// Read returns the record, or ErrExpired when the handler does not hold the
	// id -- expired, evicted, or destroyed by a logout elsewhere.
	//
	// ErrExpired specifically, not the handler's own not-found error. Callers
	// branch on it to send somebody back to the login page, and a handler that
	// returns something else makes swapping the store change the behaviour of the
	// application. Found by audit: the distributed handler returned its own not-found error,
	// so an expired session in Redis fell through to the generic error path that a
	// single-instance deployment never reached.
	Read(ctx context.Context, id string) (Record[T], error)

	// Write stores the record under id for ttl.
	//
	// It must store every exported field of the Record it is given. A field it
	// silently drops reads back as the zero value, which is a decision the
	// application makes on stale information -- and only in the deployment that
	// uses that handler, never in a test against ArrayHandler. RecordStore.Confirm
	// checks the one field whose loss is otherwise undiagnosable; the rest are on
	// the implementation.
	Write(ctx context.Context, id string, rec Record[T], ttl time.Duration) error

	// Destroy removes the session, if present.
	Destroy(ctx context.Context, id string) error

	// DestroyIndex removes every session belonging to one subject of one tenant,
	// except keepID. An empty keepID keeps none of them.
	//
	// It is what a password change and a password reset need, and until it existed
	// they could not do it: a reset that leaves the other sessions open leaves
	// whoever forced the reset signed in on their own machine, which is the exact
	// person the reset is aimed at. Deleting by id one at a time was not an option
	// -- nothing knew which ids belonged to the account.
	//
	// The tenant is part of the question, not a filter applied afterwards. Two
	// tenants may both hold a subject called "1", and signing one of them out
	// must not touch the other.
	//
	// It is not an error for the subject to have no sessions. It IS an error for
	// the tenant or the subject id to be empty: neither names a subject, and an
	// implementation that treats the empty id as one signs out every session
	// nobody has signed in on. Both refusals are made here as well as in
	// RecordStore.DestroyOthers, because this interface is exported and an
	// implementation is reachable without the store.
	DestroyIndex(ctx context.Context, tenant, subjectID, keepID string) error
}

// RecordStore issues and validates sessions whose payload is one typed value.
//
// The cookie value is the session id plus an HMAC of it. The signature is
// checked before the handler is touched, so a forged cookie never reaches the
// store -- and, in a distributed store, never costs a network round trip.
//
// # Why this is not called Store
//
// [Store] is one session, loaded for one request, holding a bag of keys.
// This is the type that mints an id, signs the cookie, reads a [Record]
// back and ends the session, and it is generic over the payload, which
// [Store] is not.
//
// Both are here because both are used: [Store] is what the flash, the old
// input and the CSRF token live in; this is what auth builds a session with
// when the payload is a struct and the application would rather the
// compiler checked it. They share [Handler] and [Record]; nothing else is
// duplicated between them.
type RecordStore[T any] struct {
	appKey  []byte
	ttl     time.Duration
	secure  bool
	handler Handler[T]
}

// NewRecordStore returns a [RecordStore] that mints and signs session ids,
// generic over the payload.
//
// Pass secure=false only in development: without the Secure attribute the
// cookie travels over plain HTTP.
func NewRecordStore[T any](appKey []byte, ttl time.Duration, secure bool, h Handler[T]) *RecordStore[T] {
	if h == nil {
		h = NewArrayHandler[T]()
	}
	return &RecordStore[T]{appKey: appKey, ttl: ttl, secure: secure, handler: h}
}

// Option adjusts how a session is started.
//
// A variadic option and not a second constructor: StartFor beside Start would be
// two functions that both start a session, and the next thing anybody needs -- a
// session for a device, for an impersonation, for a longer window -- adds a
// third. One function that takes options widens; a second name forks.
// Every existing call to Start and Regenerate passes none and behaves as it did.
type Option func(*settings)

// settings is what the options add up to. Unexported, so the set of things a
// caller can ask for is the set of exported options.
type settings struct {
	remember bool
}

// Remember asks for a session that survives closing the browser, for
// RememberLifetime instead of the store's ttl, as an option.
//
// It lengthens the session itself rather than adding a second, long-lived
// credential, so there is one to protect and revoke instead of two.
//
// It takes the answer rather than being a flag, so the call site is the form
// field and there is no branch around it:
//
//	store.Regenerate(ctx, w, old, rec, session.Remember(r.PostFormValue("remember") != ""))
//
// The sign-in screen the starter kit publishes has drawn that checkbox from the
// beginning, and nothing could read it: there was no shape in this API through
// which a longer session could be asked for, so the box was decoration in every
// project the kit created.
func Remember(on bool) Option {
	return func(s *settings) { s.remember = on }
}

// lifetime is how long a session lives, and therefore both what the handler is
// told and what the cookie's MaxAge says. The two are computed here exactly once
// so they cannot disagree: a cookie that outlives its record is a session that
// looks signed in and is not, and the person sees a login screen with no
// explanation on their next click.
func (s *RecordStore[T]) lifetime(remembered bool) time.Duration {
	if !remembered {
		return s.ttl
	}
	if s.ttl > RememberLifetime {
		// An application that already configured a longer session keeps it.
		// Otherwise ticking "remember me" shortened the session, which is the one
		// thing the box must never do.
		return s.ttl
	}
	return RememberLifetime
}

// Start creates a session for the record and writes the cookie.
//
// With no options it is what it has always been: a session for the store's
// configured ttl. See Remember for the only thing there is to ask for.
func (s *RecordStore[T]) Start(ctx context.Context, w http.ResponseWriter, rec Record[T], opts ...Option) (string, error) {
	var set settings
	for _, opt := range opts {
		opt(&set)
	}

	id, err := newID()
	if err != nil {
		return "", err
	}

	// Both fields are written here, over whatever the caller left in the record.
	// Remembered is the option's answer and nothing else's, and a session that has
	// just been created has never had its password confirmed on it -- inheriting a
	// stamp through Regenerate would hand a fresh session the confirmation the old
	// one earned.
	rec.Remembered = set.remember
	rec.PasswordConfirmedAt = time.Time{}

	life := s.lifetime(rec.Remembered)
	if err := s.handler.Write(ctx, id, rec, life); err != nil {
		return "", err
	}
	s.writeCookie(w, id, life)
	return id, nil
}

// Regenerate issues a new session id for the same subject and destroys the
// old one. There is no CSRF token to mint here -- [CSRF] keeps none -- so
// this is [Store.Migrate]'s half of it, with the record written again under
// the new id.
//
// It MUST be called on login: keeping the pre-login id is session fixation, the
// bug that lets an attacker plant a known id and inherit the session after the
// victim authenticates.
// The options are the same as Start's, and Regenerate takes them because login
// is where remember-me is answered: the sign-in handler calls Regenerate, not
// Start, so an option only Start accepted would be unreachable from the one
// screen that has the checkbox on it.
func (s *RecordStore[T]) Regenerate(ctx context.Context, w http.ResponseWriter, oldID string, rec Record[T], opts ...Option) (string, error) {
	id, err := s.Start(ctx, w, rec, opts...)
	if err != nil {
		return "", err
	}
	if oldID != "" && oldID != id {
		// A failure to delete the old id must not fail the login: the new
		// session is already valid and the old one expires on its own.
		_ = s.handler.Destroy(ctx, oldID)
	}
	return id, nil
}

// All returns the record bound to the request's session cookie: the whole
// of what this session holds, in one read, because there is nothing here to
// fetch by name.
func (s *RecordStore[T]) All(ctx context.Context, r *http.Request) (Record[T], error) {
	id := s.ID(r)
	if id == "" {
		return Record[T]{}, ErrNoSession
	}
	rec, err := s.handler.Read(ctx, id)
	if err != nil {
		return Record[T]{}, err
	}
	return rec, nil
}

// Confirm records on the request's session that the subject has just typed
// their password again. It writes the whole record, reads it back and
// rewrites the cookie, for the reasons below.
//
// It is the write half of a step-up check: a sensitive action asks
// Record.PasswordConfirmedWithin, sends the person to a password screen when the
// answer is no, and calls this once they get it right. Without the stamp the only
// two designs available were asking for the password on every sensitive action,
// which people route around, and asking once and never again, which is not a
// check.
//
// It rewrites the record, and rewriting it restarts the record's clock, so the
// cookie is rewritten with the same lifetime in the same breath. The session
// therefore gets a full lifetime back -- earned by proving who is holding it,
// which is the same proof that started it.
//
// It returns ErrConfirmationNotStored when the handler accepted the stamp and
// did not keep it. A caller must report that rather than redirect: redirecting
// sends the person back to the screen they just got right.
func (s *RecordStore[T]) Confirm(ctx context.Context, w http.ResponseWriter, r *http.Request) error {
	id := s.ID(r)
	if id == "" {
		return ErrNoSession
	}
	rec, err := s.handler.Read(ctx, id)
	if err != nil {
		return err
	}

	rec.PasswordConfirmedAt = time.Now()
	life := s.lifetime(rec.Remembered)
	if err := s.handler.Write(ctx, id, rec, life); err != nil {
		return err
	}

	// The stamp is read back before the confirmation is reported as done, because
	// a handler that stores its own wire shape can drop a field it does not know
	// about and the write still succeeds. That failure is invisible from here and
	// not survivable by the person in front of it: the sensitive action asks
	// PasswordConfirmedWithin, gets no, sends them to the password screen, they
	// type it correctly, and land on the password screen again -- forever, with
	// nothing in the logs. Found by audit on the Redis handler, whose stored shape
	// does not carry this field and cannot yet, because the field is newer than
	// the framework version that module requires.
	//
	// One extra round trip, on an operation that happens once per sensitive
	// action and never on a page load. The alternative was a loop that reads as
	// "my password is wrong".
	written, err := s.handler.Read(ctx, id)
	if err != nil {
		return err
	}
	if written.PasswordConfirmedAt.IsZero() {
		return ErrConfirmationNotStored
	}

	s.writeCookie(w, id, life)
	return nil
}

// DestroyOthers signs the subject out of every session except keepID. It
// deletes the records, so they end on their next request whether or not
// they were remembered.
//
// Pass the id of the session doing the asking to keep the person signed in where
// they are -- a password change from the account screen -- and pass an empty
// keepID to end all of them, which is what a password reset from an e-mail link
// wants: there is no session to keep, and the one session that must stop working
// belongs to whoever forced the reset.
//
// It does not touch the cookie. The kept session's cookie is still valid and
// every other browser is holding a cookie whose record is gone, which is a
// session that stops at the next request -- there is no way to reach into those
// browsers and no need to.
//
// The tenant comes from the record, which came from the Grant or the
// session, never from the request. A record with no tenant is refused
// rather than turned into a query that matches an id across every customer.
func (s *RecordStore[T]) DestroyOthers(ctx context.Context, rec Record[T], keepID string) error {
	if rec.SubjectID == "" || rec.Tenant == "" {
		return errNoSubjectScope
	}
	return s.handler.DestroyIndex(ctx, rec.Tenant, rec.SubjectID, keepID)
}

// Invalidate removes the session and clears the session cookie. It is what
// a sign-out calls. What it does NOT clear is the address a guard
// remembered before sending somebody to the sign-in screen: that cookie
// belongs to the package that writes it, and signing out has to clear it
// too -- a shared machine changes hands at exactly that moment, and an
// address remembered before a sign-out is one nobody wants afterwards. See
// http, and the report on this package's move.
func (s *RecordStore[T]) Invalidate(ctx context.Context, w http.ResponseWriter, id string) error {
	if id != "" {
		if err := s.handler.Destroy(ctx, id); err != nil {
			return err
		}
	}
	http.SetCookie(w, &http.Cookie{
		Name:     CookieName,
		Value:    "",
		Path:     "/",
		MaxAge:   -1,
		HttpOnly: true,
		Secure:   s.secure,
		SameSite: http.SameSiteLaxMode,
	})
	return nil
}

// ID returns the session id when the cookie signature is valid, and the
// empty string otherwise, read off the request rather than off a loaded
// session. It is the value to hand to [CSRF], which binds its token to this
// id.
func (s *RecordStore[T]) ID(r *http.Request) string {
	c, err := r.Cookie(CookieName)
	if err != nil {
		return ""
	}
	id, sig, ok := strings.Cut(c.Value, ".")
	if !ok || id == "" {
		return ""
	}
	if !hmac.Equal([]byte(s.sign(id)), []byte(sig)) {
		return ""
	}
	return id
}

// writeCookie takes the lifetime rather than reading s.ttl, because the record
// was written with that same value: a remembered session whose cookie still said
// one hour was a session the browser threw away while the handler held it, and
// the person was signed out an hour after ticking a box that promised a month.
func (s *RecordStore[T]) writeCookie(w http.ResponseWriter, id string, lifetime time.Duration) {
	http.SetCookie(w, &http.Cookie{
		Name:     CookieName,
		Value:    id + "." + s.sign(id),
		Path:     "/",
		MaxAge:   int(lifetime.Seconds()),
		HttpOnly: true,
		Secure:   s.secure,
		SameSite: http.SameSiteLaxMode,
	})
}

func (s *RecordStore[T]) sign(id string) string {
	m := hmac.New(sha256.New, s.appKey)
	m.Write([]byte(CookieName))
	m.Write([]byte{0})
	m.Write([]byte(id))
	return base64.RawURLEncoding.EncodeToString(m.Sum(nil))
}

func newID() (string, error) {
	b := make([]byte, 32)
	if _, err := rand.Read(b); err != nil {
		return "", err
	}
	return base64.RawURLEncoding.EncodeToString(b), nil
}

// ArrayHandler keeps sessions in the process memory.
//
// It is the right choice for development and for a single instance. Behind more
// than one pod it silently logs people out on every deploy and on every request
// routed elsewhere -- use a Redis-backed handler for shared storage.
type ArrayHandler[T any] struct {
	mu      sync.RWMutex
	entries map[string]arrayEntry[T]
}

type arrayEntry[T any] struct {
	record  Record[T]
	expires time.Time
}

// NewArrayHandler returns an empty in-memory session handler, for [Record]
// instead of bytes.
//
// It takes no lifetime, because the ttl arrives with each write rather than
// with the handler.
func NewArrayHandler[T any]() *ArrayHandler[T] {
	return &ArrayHandler[T]{entries: map[string]arrayEntry[T]{}}
}

// Read returns the record, or ErrExpired when the id is unknown.
// [SessionHandler] answers the empty string for a missing session and for
// an expired one alike; this one names the case instead, because a caller
// holding a typed record has no empty string to read as absence.
func (h *ArrayHandler[T]) Read(ctx context.Context, id string) (Record[T], error) {
	h.mu.RLock()
	e, ok := h.entries[id]
	h.mu.RUnlock()
	if !ok {
		return Record[T]{}, ErrExpired
	}
	if time.Now().After(e.expires) {
		h.mu.Lock()
		delete(h.entries, id)
		h.mu.Unlock()
		return Record[T]{}, ErrExpired
	}
	return e.record, nil
}

// Write stores the record under id for the given ttl.
func (h *ArrayHandler[T]) Write(ctx context.Context, id string, rec Record[T], ttl time.Duration) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.entries[id] = arrayEntry[T]{record: rec, expires: time.Now().Add(ttl)}
	return nil
}

// Destroy removes the session, if present.
func (h *ArrayHandler[T]) Destroy(ctx context.Context, id string) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.entries, id)
	return nil
}

// DestroyIndex is what [RecordStore.DestroyOthers] needs from a handler: it
// removes every session of one subject of one tenant except keepID.
//
// It is a scan of the map, and it stays a scan: a second map keyed by subject
// would have to be kept in step with expiry, with Destroy and with eviction, and
// getting that wrong leaves a password reset believing it signed somebody out.
// This handler holds one instance's sessions, and walking them costs less than
// the round trip the caller just made. A distributed handler cannot scan and
// carries a shared index instead.
func (h *ArrayHandler[T]) DestroyIndex(ctx context.Context, tenant, subjectID, keepID string) error {
	// The same refusal RecordStore.DestroyOthers makes, repeated here because this
	// method is exported and a caller can reach it without the store. It used to
	// loop with whatever it was given: tenant "t1" and an empty subject id matched
	// every session that had no subject on it, which is every guest, and the Redis
	// handler refused the identical call. See errNoSubjectScope.
	if tenant == "" || subjectID == "" {
		return errNoSubjectScope
	}

	h.mu.Lock()
	defer h.mu.Unlock()
	for id, e := range h.entries {
		if id == keepID {
			continue
		}
		if e.record.Tenant != tenant || e.record.SubjectID != subjectID {
			continue
		}
		delete(h.entries, id)
	}
	return nil
}
