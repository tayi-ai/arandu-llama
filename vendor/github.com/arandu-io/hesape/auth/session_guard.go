package auth

import (
	"context"
	"crypto/hmac"
	"crypto/rand"
	"crypto/sha1"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"net/http"
)

const (
	// defaultRememberDuration is how long the "remember me" cookie is good for
	// when nobody sets a duration: 576000 minutes, the 400 days a browser will
	// keep a cookie for.
	defaultRememberDuration = 576000

	// defaultTimeboxDuration is the minimum an attempt is held open for when
	// none is configured, in microseconds.
	defaultTimeboxDuration = 200000

	// defaultHashKey is the key [SessionGuard.HashPasswordForCookie] falls back
	// to when the application has none.
	defaultHashKey = "base-key-for-password-hash-mac"

	// rememberTokenLength is how many characters a remember token has.
	rememberTokenLength = 60

	// sessionGuardClass is the type name [SessionGuard.GetName] and
	// [SessionGuard.GetRecallerName] hash into their keys, so that two guards of
	// different types cannot read each other's session entry. Go has no
	// subclassing, so the name is fixed here rather than read off the receiver.
	//
	// It is this package's own import path, and it must be: a session written by
	// some other framework's guard is not one this guard should silently adopt.
	sessionGuardClass = "github.com/arandu-io/hesape/auth.SessionGuard"

	// rememberTokenAlphabet is the set of characters a remember token is drawn
	// from.
	rememberTokenAlphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
)

var (
	// ErrCookieJarNotSet reports that the guard was asked for the cookie jar and
	// nobody set one.
	ErrCookieJarNotSet = errors.New("auth: cookie jar has not been set")

	// ErrHasherNotSet is what [SessionGuard.LogoutOtherDevices] answers when the
	// guard has no hasher. The hasher is a field, and an unset one is a wiring
	// mistake rather than a wrong password.
	ErrHasherNotSet = errors.New("auth: hasher has not been set")

	// ErrPasswordMismatch reports that the password given to
	// [SessionGuard.LogoutOtherDevices] is not the account's current password.
	ErrPasswordMismatch = errors.New("auth: the given password does not match the current password")

	// ErrInvalidBasicCredentials reports that the HTTP Basic credentials on the
	// request did not check out.
	//
	// It carries no WWW-Authenticate challenge: the response is the caller's, and
	// a middleware that gets this error writes 401 and the Basic realm.
	ErrInvalidBasicCredentials = errors.New("auth: invalid credentials")
)

// SessionGuard is the guard behind a browser session, and the one a form login
// goes through.
//
// It is the whole of the sign-in flow -- the attempt, the timebox that hides
// whether the account exists, the session id that is regenerated on the way in,
// the "remember me" cookie, and the logout that cycles the token so the cookie
// left in another browser stops working.
//
// It is stateful and it is per request: it caches the user it resolved, the
// request it reads cookies from, and whether logout was called. Build one per
// request, and do not share it between goroutines.
type SessionGuard struct {
	GuardHelpers

	// Name is the guard's name in the authentication configuration, typically
	// "web". Set it through [NewSessionGuard] and leave it alone, because
	// [SessionGuard.GetName] and [SessionGuard.GetRecallerName] are built from
	// it.
	Name string

	// Hasher checks and rewrites the account's password hash.
	//
	// [SessionGuard.LogoutOtherDevices] is the only method that reads it, and it
	// answers [ErrHasherNotSet] when it is nil.
	Hasher Hasher

	// lastAttempted is the user the last attempt retrieved, whether or not the
	// password matched.
	lastAttempted Authenticatable

	// viaRemember records that the user was resolved from the "remember me"
	// cookie rather than from the session.
	viaRemember bool

	// rememberDuration is how long the "remember me" cookie is good for, in
	// minutes.
	rememberDuration int

	// session is where the authenticated user's identifier is kept.
	session Session

	// cookie makes and queues the cookies the guard writes.
	cookie CookieJar

	// request is what the guard reads cookies and Basic credentials from.
	request Request

	// events is the dispatcher the guard fires on, and a nil one fires nothing.
	events Dispatcher

	// timebox holds every attempt open for a minimum time, so that a refusal
	// cannot be timed.
	timebox Timebox

	// timeboxDuration is that minimum, in microseconds.
	timeboxDuration int

	// rehashOnLogin asks the provider to upgrade a weakly hashed password once
	// the plain one is in hand.
	rehashOnLogin bool

	// hashKey is the application key [SessionGuard.HashPasswordForCookie] signs
	// with.
	hashKey string

	// loggedOut records that logout was called, after which the guard resolves
	// nobody.
	loggedOut bool

	// recallAttempted records that the recaller cookie already had its one
	// chance on this request.
	recallAttempted bool
}

var (
	_ StatefulGuard     = (*SessionGuard)(nil)
	_ SupportsBasicAuth = (*SessionGuard)(nil)
)

// NewSessionGuard returns a guard called name, reading its accounts from
// provider and keeping the signed-in identifier in session.
//
// Every argument is passed, and two are filled in when they arrive empty: a nil
// timebox becomes [NewTimebox], and a timeboxDuration of 0 becomes 200000
// microseconds -- a zero there would switch off the wait that keeps a failed
// attempt from being timed, and nobody would see it go. rehashOnLogin gets no
// such treatment, because false is a real setting.
//
// The cookie jar and the event dispatcher are not arguments. Set them with
// [SessionGuard.SetCookieJar] and [SessionGuard.SetDispatcher], which is what
// [AuthManager] does.
func NewSessionGuard(
	name string,
	provider UserProvider,
	session Session,
	request Request,
	timebox Timebox,
	rehashOnLogin bool,
	timeboxDuration int,
	hashKey string,
) *SessionGuard {
	if timebox == nil {
		timebox = NewTimebox()
	}
	if timeboxDuration == 0 {
		timeboxDuration = defaultTimeboxDuration
	}

	guard := &SessionGuard{
		Name:             name,
		session:          session,
		request:          request,
		timebox:          timebox,
		rehashOnLogin:    rehashOnLogin,
		timeboxDuration:  timeboxDuration,
		hashKey:          hashKey,
		rememberDuration: defaultRememberDuration,
	}
	guard.provider = provider
	guard.resolveUser = guard.User

	return guard
}

// User is the authenticated user, or nil.
//
// It resolves once per request and caches: the id in the session names the
// user, and when it names nobody the recaller cookie gets one chance to. A user
// that came back from the cookie is signed in properly on the spot -- the
// session is written and the Login event fires with remember true -- which is
// why a person who ticked the box never sees the sign-in form again.
//
// The Guard contract gives it no context.Context, so the lookup runs on the
// request's context. A provider that fails is a provider that found nobody, for
// the same reason: there is nowhere to put the error, and a guard that cannot
// read the user store has no user.
func (g *SessionGuard) User() Authenticatable {
	if g.loggedOut {
		return nil
	}

	// If we have already retrieved the user for the current request we can just
	// return it back immediately.
	if g.user != nil {
		return g.user
	}

	ctx := g.context()

	// First we will try to load the user using the identifier in the session if
	// one exists.
	if id := g.session.Get(g.GetName()); id != nil {
		g.user = g.retrieveByID(ctx, id)

		if g.user != nil {
			g.fireAuthenticatedEvent(g.user)
		}
	}

	// If the user is null, but we read a "recaller" cookie we can attempt to
	// pull the user data on that cookie, which serves as a remember cookie.
	if g.user == nil {
		if recaller := g.recaller(); recaller != nil {
			g.user = g.userFromRecaller(ctx, recaller)

			if g.user != nil {
				g.updateSession(ctx, g.user.GetAuthIdentifier())

				g.fireLoginEvent(g.user, true)
			}
		}
	}

	return g.user
}

// userFromRecaller is the user the "remember me" cookie names, at most once per
// request.
func (g *SessionGuard) userFromRecaller(ctx context.Context, recaller *Recaller) Authenticatable {
	if !recaller.Valid() || g.recallAttempted {
		return nil
	}

	g.recallAttempted = true

	user, err := g.provider.RetrieveByToken(ctx, recaller.ID(), recaller.Token())
	if err != nil {
		user = nil
	}

	g.viaRemember = user != nil

	return user
}

// recaller is the recaller cookie on this request, or nil.
func (g *SessionGuard) recaller() *Recaller {
	if g.request == nil {
		return nil
	}

	if value := g.request.Cookie(g.GetRecallerName()); value != "" {
		return NewRecaller(value)
	}
	return nil
}

// ID is the authenticated user's identifier.
//
// It falls back to the id in the session, so a request that has not resolved
// the user yet still knows who it is about.
func (g *SessionGuard) ID() any {
	if g.loggedOut {
		return nil
	}

	if user := g.User(); user != nil {
		return user.GetAuthIdentifier()
	}
	return g.session.Get(g.GetName())
}

// Once signs in for this request only, with no session and no cookie.
func (g *SessionGuard) Once(ctx context.Context, credentials map[string]any) bool {
	g.fireAttemptEvent(credentials, false)

	if g.Validate(ctx, credentials) {
		g.rehashPasswordIfRequired(ctx, g.lastAttempted, credentials)

		g.SetUser(g.lastAttempted)

		return true
	}

	g.fireFailedEvent(g.lastAttempted, credentials)

	return false
}

// OnceUsingID signs the given id in for this request only.
//
// It answers nil when no such user exists.
func (g *SessionGuard) OnceUsingID(ctx context.Context, id any) Authenticatable {
	if user := g.retrieveByID(ctx, id); user != nil {
		g.SetUser(user)

		return user
	}
	return nil
}

// Validate reports whether these credentials are good, without signing anybody
// in.
//
// It runs inside the timebox, and only a match asks to return early. That is
// the whole point: an address nobody registered and an address with the wrong
// password take the same time to refuse, so the clock cannot be read as a list
// of customers.
func (g *SessionGuard) Validate(ctx context.Context, credentials map[string]any) bool {
	validated, _ := g.timebox.Call(func(timebox Timebox) (any, error) {
		_, validated := g.validateCredentials(ctx, credentials)

		if validated {
			timebox.ReturnEarly()
		}

		return validated, nil
	}, g.timeboxDuration)

	ok, _ := validated.(bool)
	return ok
}

// Basic signs in from the HTTP Basic header, with a session, unless somebody is
// signed in already.
//
// It returns nil when the request may carry on and
// [ErrInvalidBasicCredentials] when it may not.
func (g *SessionGuard) Basic(ctx context.Context, field string, extraConditions map[string]any) error {
	if g.Check() {
		return nil
	}

	// If a username is set on the HTTP basic request, we will return out without
	// interrupting the request lifecycle.
	if g.attemptBasic(ctx, g.GetRequest(), field, extraConditions) {
		return nil
	}

	return g.failedBasicResponse()
}

// OnceBasic is a stateless HTTP Basic sign-in: no session, no cookie.
func (g *SessionGuard) OnceBasic(ctx context.Context, field string, extraConditions map[string]any) error {
	credentials := g.basicCredentials(g.GetRequest(), field)

	if !g.Once(ctx, mergeCredentials(credentials, extraConditions)) {
		return g.failedBasicResponse()
	}
	return nil
}

// attemptBasic signs in from the request's Basic header, and reports whether it
// worked.
func (g *SessionGuard) attemptBasic(ctx context.Context, request Request, field string, extraConditions map[string]any) bool {
	if request == nil || request.GetUser() == "" {
		return false
	}

	return g.Attempt(ctx, mergeCredentials(g.basicCredentials(request, field), extraConditions), false)
}

// basicCredentials reads the Basic header as a credentials map.
func (g *SessionGuard) basicCredentials(request Request, field string) map[string]any {
	if request == nil {
		return map[string]any{field: "", "password": ""}
	}
	return map[string]any{field: request.GetUser(), "password": request.GetPassword()}
}

// failedBasicResponse is what a refused Basic sign-in answers with.
func (g *SessionGuard) failedBasicResponse() error {
	return ErrInvalidBasicCredentials
}

// Attempt is the sign-in a login form calls.
//
// Everything happens inside the timebox, and only success returns early, so a
// refusal takes the configured minimum however early it was decided.
func (g *SessionGuard) Attempt(ctx context.Context, credentials map[string]any, remember bool) bool {
	attempted, _ := g.timebox.Call(func(timebox Timebox) (any, error) {
		g.fireAttemptEvent(credentials, remember)

		user, validated := g.validateCredentials(ctx, credentials)

		// If an implementation of Authenticatable was returned, we ask the
		// provider to validate it against the credentials, and if they are in
		// fact valid we log the user in and return true.
		if validated {
			g.rehashPasswordIfRequired(ctx, user, credentials)

			g.Login(ctx, user, remember)

			timebox.ReturnEarly()

			return true, nil
		}

		// If the attempt fails we fire an event, so that the person may be told
		// about an attempt to reach their account from somewhere they do not
		// recognise.
		g.fireFailedEvent(user, credentials)

		return false, nil
	}, g.timeboxDuration)

	ok, _ := attempted.(bool)
	return ok
}

// AttemptWhen is [SessionGuard.Attempt], with callbacks that get a veto after
// the password matched.
//
// It is where "this account is suspended" belongs -- a check that must not
// change the answer to "is this the right password", and must not tell the
// person guessing which of the two failed.
//
// A single callback is a slice of one, and a nil entry is skipped.
func (g *SessionGuard) AttemptWhen(ctx context.Context, credentials map[string]any, callbacks []func(user Authenticatable, guard *SessionGuard) bool, remember bool) bool {
	attempted, _ := g.timebox.Call(func(timebox Timebox) (any, error) {
		g.fireAttemptEvent(credentials, remember)

		user, validated := g.validateCredentials(ctx, credentials)

		// This does the same thing as Attempt, and also runs the callbacks once
		// the user has been retrieved and validated. If one of them says no, the
		// person is not signed in.
		if validated && g.shouldLogin(callbacks, user) {
			g.rehashPasswordIfRequired(ctx, user, credentials)

			g.Login(ctx, user, remember)

			timebox.ReturnEarly()

			return true, nil
		}

		g.fireFailedEvent(user, credentials)

		return false, nil
	}, g.timeboxDuration)

	ok, _ := attempted.(bool)
	return ok
}

// validateCredentials looks up and checks credentials through the shared
// verifier path, records the account attempted, and fires [Validated] on a
// match.
func (g *SessionGuard) validateCredentials(ctx context.Context, credentials map[string]any) (Authenticatable, bool) {
	user, err := verifyCredentials(ctx, g.provider, credentials)
	g.lastAttempted = user
	if err != nil {
		return user, false
	}

	g.fireValidatedEvent(user)
	return user, true
}

// shouldLogin runs the callbacks and reports whether every one of them allowed
// the sign-in.
func (g *SessionGuard) shouldLogin(callbacks []func(user Authenticatable, guard *SessionGuard) bool, user Authenticatable) bool {
	for _, callback := range callbacks {
		if callback == nil {
			continue
		}
		if !callback(user, g) {
			return false
		}
	}
	return true
}

// rehashPasswordIfRequired upgrades a hash made with weaker parameters, now
// that the plain password is in hand.
//
// It returns nothing: the sign-in already succeeded, and a hash that could not
// be rewritten is not a reason to refuse it.
func (g *SessionGuard) rehashPasswordIfRequired(ctx context.Context, user Authenticatable, credentials map[string]any) {
	if g.rehashOnLogin && user != nil {
		_ = g.provider.RehashPasswordIfRequired(ctx, user, credentials, false)
	}
}

// LoginUsingID signs the given id in, queueing the "remember me" cookie when
// remember is true.
//
// It answers nil when no such user exists.
func (g *SessionGuard) LoginUsingID(ctx context.Context, id any, remember bool) Authenticatable {
	if user := g.retrieveByID(ctx, id); user != nil {
		g.Login(ctx, user, remember)

		return user
	}
	return nil
}

// Login puts the user in the session, and in the cookie if they asked to be
// remembered.
//
// The session id is regenerated here, which is what stops a session handed to
// the browser before sign-in from being one after it.
func (g *SessionGuard) Login(ctx context.Context, user Authenticatable, remember bool) {
	g.updateSession(ctx, user.GetAuthIdentifier())

	// If the user should be permanently "remembered" we queue a cookie holding
	// the identifier, the remember token and a MAC of the password hash.
	if remember {
		g.ensureRememberTokenIsSet(ctx, user)

		// The contract returns nothing, so a cookie jar that was never set
		// cannot be reported from here. It surfaces at the next call to
		// GetCookieJar by somebody who can say so.
		_ = g.queueRecallerCookie(user)
	}

	g.fireLoginEvent(user, remember)

	g.SetUser(user)
}

// updateSession writes the identifier to the session and regenerates the
// session id.
func (g *SessionGuard) updateSession(ctx context.Context, id any) {
	g.session.Put(g.GetName(), id)

	_ = g.session.Regenerate(ctx, true)
}

// ensureRememberTokenIsSet gives the user a remember token when they have none.
func (g *SessionGuard) ensureRememberTokenIsSet(ctx context.Context, user Authenticatable) {
	if user.GetRememberToken() == "" {
		g.cycleRememberToken(ctx, user)
	}
}

// queueRecallerCookie queues the "remember me" cookie for this user: the
// identifier, the remember token and a MAC of the password hash.
func (g *SessionGuard) queueRecallerCookie(user Authenticatable) error {
	jar, err := g.GetCookieJar()
	if err != nil {
		return err
	}

	recaller, err := g.createRecaller(
		fmt.Sprint(user.GetAuthIdentifier()) + "|" +
			user.GetRememberToken() + "|" +
			g.HashPasswordForCookie(user.GetAuthPassword()),
	)
	if err != nil {
		return err
	}

	jar.Queue(recaller)

	return nil
}

// createRecaller builds the "remember me" cookie holding value.
func (g *SessionGuard) createRecaller(value string) (*http.Cookie, error) {
	jar, err := g.GetCookieJar()
	if err != nil {
		return nil, err
	}

	// The jar takes the whole attribute list, so the defaults are written out:
	// the jar's path and domain, its secure setting, http only, not raw, and the
	// jar's SameSite.
	return jar.Make(g.GetRecallerName(), value, g.getRememberDuration(), "", "", nil, true, false, http.SameSiteDefaultMode), nil
}

// HashPasswordForCookie is a MAC of the password hash, for the third segment of
// the recaller.
//
// It is what lets AuthenticateSession notice that the password changed since
// the cookie was written, without the cookie carrying the hash itself.
func (g *SessionGuard) HashPasswordForCookie(passwordHash string) string {
	key := g.hashKey
	if key == "" {
		key = defaultHashKey
	}

	mac := hmac.New(sha256.New, []byte(key))
	mac.Write([]byte(passwordHash))

	return hex.EncodeToString(mac.Sum(nil))
}

// Logout signs the user out of this application, everywhere the cookie reached.
//
// It cycles the remember token, so the recaller cookie sitting in another
// browser stops naming anybody.
func (g *SessionGuard) Logout(ctx context.Context) {
	user := g.User()

	_ = g.clearUserDataFromStorage()

	if g.user != nil && user != nil && user.GetRememberToken() != "" {
		g.cycleRememberToken(ctx, user)
	}

	if g.events != nil {
		g.events.Dispatch(Logout{Guard: g.Name, User: user})
	}

	// Once the event has fired we clear the user out of memory: they are no
	// longer signed in and must not be available from here.
	g.user = nil

	g.loggedOut = true
}

// LogoutCurrentDevice signs the user out of this browser only.
//
// It does not cycle the remember token, so the cookie in the other browser
// keeps working. That is the difference from [SessionGuard.Logout], and it is
// the whole method -- and the reason it takes no context: not cycling the token
// is not writing to the user store.
func (g *SessionGuard) LogoutCurrentDevice() {
	user := g.User()

	_ = g.clearUserDataFromStorage()

	if g.events != nil {
		g.events.Dispatch(CurrentDeviceLogout{Guard: g.Name, User: user})
	}

	g.user = nil

	g.loggedOut = true
}

// clearUserDataFromStorage removes the session entry and forgets the recaller
// cookie.
func (g *SessionGuard) clearUserDataFromStorage() error {
	g.session.Remove(g.GetName())

	jar, err := g.GetCookieJar()
	if err != nil {
		return err
	}

	jar.Unqueue(g.GetRecallerName(), "")

	if g.recaller() != nil {
		jar.Queue(jar.Forget(g.GetRecallerName(), "", ""))
	}

	return nil
}

// cycleRememberToken writes a new remember token, on the user and in the store.
func (g *SessionGuard) cycleRememberToken(ctx context.Context, user Authenticatable) {
	token := randomToken(rememberTokenLength)

	user.SetRememberToken(token)

	_ = g.provider.UpdateRememberToken(ctx, user, token)
}

// LogoutOtherDevices invalidates this person's other sessions, keeping this
// one.
//
// It works by rehashing the password, which changes the MAC every other
// session's recaller carries -- so the AuthenticateSession middleware turns
// them away on their next request. The application must be using that
// middleware for this to mean anything.
//
// The returned user is always nil when the password matched. A password that
// does not match is [ErrPasswordMismatch].
func (g *SessionGuard) LogoutOtherDevices(ctx context.Context, password string) (Authenticatable, error) {
	if g.User() == nil {
		return nil, nil
	}

	result, err := g.rehashUserPasswordForDeviceLogout(ctx, password)
	if err != nil {
		return nil, err
	}

	// The cookie jar is asked only when there is no recaller on the request,
	// which is why a guard with no jar and a recaller in hand does not fail
	// here.
	reissue := g.recaller() != nil
	if !reissue {
		jar, err := g.GetCookieJar()
		if err != nil {
			return nil, err
		}
		reissue = jar.HasQueued(g.GetRecallerName(), "")
	}

	if reissue {
		if err := g.queueRecallerCookie(g.User()); err != nil {
			return nil, err
		}
	}

	g.fireOtherDeviceLogoutEvent(g.User())

	return result, nil
}

// rehashUserPasswordForDeviceLogout checks the given password against the
// account's and asks the provider to rehash it.
func (g *SessionGuard) rehashUserPasswordForDeviceLogout(ctx context.Context, password string) (Authenticatable, error) {
	user := g.User()

	if g.Hasher == nil {
		return nil, ErrHasherNotSet
	}

	if !g.Hasher.Check(password, user.GetAuthPassword()) {
		return nil, ErrPasswordMismatch
	}

	if err := g.provider.RehashPasswordIfRequired(ctx, user, map[string]any{"password": password}, true); err != nil {
		return nil, err
	}

	return nil, nil
}

// Attempting registers a listener for the [Attempting] event.
//
// It takes any: what a listener may be is the dispatcher's business, not the
// guard's.
func (g *SessionGuard) Attempting(callback any) {
	if g.events != nil {
		g.events.Listen(Attempting{}, callback)
	}
}

// fireAttemptEvent dispatches [Attempting].
func (g *SessionGuard) fireAttemptEvent(credentials map[string]any, remember bool) {
	if g.events != nil {
		g.events.Dispatch(Attempting{Guard: g.Name, Credentials: credentials, Remember: remember})
	}
}

// fireValidatedEvent dispatches [Validated].
func (g *SessionGuard) fireValidatedEvent(user Authenticatable) {
	if g.events != nil {
		g.events.Dispatch(Validated{Guard: g.Name, User: user})
	}
}

// fireLoginEvent dispatches [Login].
func (g *SessionGuard) fireLoginEvent(user Authenticatable, remember bool) {
	if g.events != nil {
		g.events.Dispatch(Login{Guard: g.Name, User: user, Remember: remember})
	}
}

// fireAuthenticatedEvent dispatches [Authenticated].
func (g *SessionGuard) fireAuthenticatedEvent(user Authenticatable) {
	if g.events != nil {
		g.events.Dispatch(Authenticated{Guard: g.Name, User: user})
	}
}

// fireOtherDeviceLogoutEvent dispatches [OtherDeviceLogout].
func (g *SessionGuard) fireOtherDeviceLogoutEvent(user Authenticatable) {
	if g.events != nil {
		g.events.Dispatch(OtherDeviceLogout{Guard: g.Name, User: user})
	}
}

// fireFailedEvent dispatches [Failed].
func (g *SessionGuard) fireFailedEvent(user Authenticatable, credentials map[string]any) {
	if g.events != nil {
		g.events.Dispatch(Failed{Guard: g.Name, User: user, Credentials: credentials})
	}
}

// GetLastAttempted is the user the last attempt retrieved, whether or not the
// password matched.
func (g *SessionGuard) GetLastAttempted() Authenticatable {
	return g.lastAttempted
}

// GetName is the session key the user id is kept under.
func (g *SessionGuard) GetName() string {
	return "login_" + g.Name + "_" + classHash(sessionGuardClass)
}

// GetRecallerName is the name of the "remember me" cookie.
func (g *SessionGuard) GetRecallerName() string {
	return "remember_" + g.Name + "_" + classHash(sessionGuardClass)
}

// ViaRemember reports that this session came from the cookie, not from a
// password typed in this browser session.
//
// A destructive action is the right moment to read it and ask for the password
// again.
func (g *SessionGuard) ViaRemember() bool {
	return g.viaRemember
}

// getRememberDuration is how many minutes the "remember me" cookie is good for.
func (g *SessionGuard) getRememberDuration() int {
	return g.rememberDuration
}

// SetRememberDuration sets how many minutes the "remember me" cookie is good
// for, and returns the guard.
func (g *SessionGuard) SetRememberDuration(minutes int) *SessionGuard {
	g.rememberDuration = minutes

	return g
}

// GetCookieJar is the cookie jar the guard writes through.
//
// It answers [ErrCookieJarNotSet] when nobody set one.
func (g *SessionGuard) GetCookieJar() (CookieJar, error) {
	if g.cookie == nil {
		return nil, ErrCookieJarNotSet
	}
	return g.cookie, nil
}

// SetCookieJar sets the cookie jar the guard writes through.
func (g *SessionGuard) SetCookieJar(cookie CookieJar) {
	g.cookie = cookie
}

// GetDispatcher is the event dispatcher the guard fires on, nil when none was
// set.
func (g *SessionGuard) GetDispatcher() Dispatcher {
	return g.events
}

// SetDispatcher sets the event dispatcher the guard fires on.
func (g *SessionGuard) SetDispatcher(events Dispatcher) {
	g.events = events
}

// GetSession is the session the guard keeps the user identifier in.
func (g *SessionGuard) GetSession() Session {
	return g.session
}

// GetUser is the user already resolved, without resolving one.
func (g *SessionGuard) GetUser() Authenticatable {
	return g.user
}

// SetUser sets the authenticated user.
//
// It undoes a logout, and it fires [Authenticated]. It returns nothing, because
// the Guard contract's SetUser returns nothing.
func (g *SessionGuard) SetUser(user Authenticatable) {
	g.user = user

	g.loggedOut = false

	g.fireAuthenticatedEvent(user)
}

// GetRequest is the request the guard reads cookies and Basic credentials from.
//
// There is no fallback: a request arrives at a handler and is passed in, never
// read out of a global, so a guard that was given none has none and this
// answers nil.
func (g *SessionGuard) GetRequest() Request {
	return g.request
}

// SetRequest sets the request the guard reads from, and returns the guard.
func (g *SessionGuard) SetRequest(request Request) *SessionGuard {
	g.request = request

	return g
}

// GetTimebox is the timebox every attempt is held open by.
func (g *SessionGuard) GetTimebox() Timebox {
	return g.timebox
}

// context is the context the guard runs its lookups on: the request's, so that
// cancelling the request stops them. See [Request] for why it is read from
// there and not taken as an argument.
func (g *SessionGuard) context() context.Context {
	if g.request != nil {
		if ctx := g.request.Context(); ctx != nil {
			return ctx
		}
	}
	return context.Background()
}

// retrieveByID asks the provider for the user with this id, reading an error as
// nobody.
func (g *SessionGuard) retrieveByID(ctx context.Context, id any) Authenticatable {
	user, err := g.provider.RetrieveByID(ctx, id)
	if err != nil {
		return nil
	}
	return user
}

// mergeCredentials merges the maps the two Basic methods build: the extra
// conditions win.
func mergeCredentials(credentials, extra map[string]any) map[string]any {
	merged := make(map[string]any, len(credentials)+len(extra))
	for key, value := range credentials {
		merged[key] = value
	}
	for key, value := range extra {
		merged[key] = value
	}
	return merged
}

// classHash is the SHA-1 of the guard's type name, as it appears in the session
// and cookie keys.
func classHash(class string) string {
	sum := sha1.Sum([]byte(class))
	return hex.EncodeToString(sum[:])
}

// randomToken is a token of the given length, drawn from the operating system's
// randomness.
func randomToken(length int) string {
	bytes := make([]byte, length)
	if _, err := rand.Read(bytes); err != nil {
		// crypto/rand does not fail on any platform this runs on, and a
		// remember token that is not random is worse than no token at all.
		panic("auth: the operating system refused randomness for a remember token: " + err.Error())
	}

	token := make([]byte, length)
	for i, b := range bytes {
		token[i] = rememberTokenAlphabet[int(b)%len(rememberTokenAlphabet)]
	}
	return string(token)
}

// Attempting announces that somebody is trying to sign in, before anything has
// been looked up.
//
// The eight events below live here rather than in the auth/events subpackage
// because the root of auth imports nothing outside the standard library (see
// doc.go) and a guard cannot fire an event it cannot name. A listener registers
// against the value:
//
//	dispatcher.Listen(auth.Login{}, func(e auth.Login) { ... })
type Attempting struct {
	// Guard is the name of the guard.
	Guard string

	// Credentials are the credentials the attempt was made with, and they hold
	// the plain password: a listener that logs this map writes a password to the
	// log.
	Credentials map[string]any

	// Remember indicates whether the person asked to be remembered.
	Remember bool
}

// Authenticated announces that a user was resolved for this request.
type Authenticated struct {
	// Guard is the name of the guard.
	Guard string

	// User is the authenticated user.
	User Authenticatable
}

// Validated announces that the credentials were good, before anybody was signed
// in.
type Validated struct {
	// Guard is the name of the guard.
	Guard string

	// User is the user the credentials named.
	User Authenticatable
}

// Login announces that somebody signed in.
type Login struct {
	// Guard is the name of the guard.
	Guard string

	// User is the user who signed in.
	User Authenticatable

	// Remember indicates whether the person asked to be remembered.
	Remember bool
}

// Logout announces that somebody signed out, everywhere.
type Logout struct {
	// Guard is the name of the guard.
	Guard string

	// User is the user who signed out.
	User Authenticatable
}

// CurrentDeviceLogout announces that somebody signed out of this browser only.
type CurrentDeviceLogout struct {
	// Guard is the name of the guard.
	Guard string

	// User is the user who signed out.
	User Authenticatable
}

// OtherDeviceLogout announces that somebody invalidated their other sessions.
type OtherDeviceLogout struct {
	// Guard is the name of the guard.
	Guard string

	// User is the user whose other sessions ended.
	User Authenticatable
}

// Failed announces an attempt that did not work.
//
// User is the account the credentials named, and it is nil when they named
// nobody. It is the event a "somebody tried to sign in to your account" notice
// listens for.
type Failed struct {
	// Guard is the name of the guard.
	Guard string

	// User is the account the credentials named, nil when they matched none.
	User Authenticatable

	// Credentials are the credentials the attempt was made with. See
	// [Attempting.Credentials]: they hold the plain password.
	Credentials map[string]any
}
