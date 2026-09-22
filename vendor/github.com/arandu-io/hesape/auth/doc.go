// Package auth answers who is acting, and proves that somebody decided what
// they may do.
//
// The root of this package is the authorization half: who is acting (Subject),
// what they mean to do (Action), the Policy that decides, and the Grant that
// proves a decision happened. Nothing reaches a repository without one, which
// is the thesis of the framework stated as a type -- see Grant.
//
// It also holds the sign-in throttle, because the counter that stops a leaked
// password list from being tried against an account belongs next to the
// decision it protects, not in a route limiter that knows nothing about
// identities.
//
// # Credential verification
//
// CredentialVerifier uses the same provider and timebox path as SessionGuard
// to return the account behind valid credentials without owning a request,
// cookie jar, or session. It is the seam for a flow that must decide what
// follows a valid password before it creates authenticated identity.
//
// # The guards
//
// Three guards live here, with the helpers they embed and the pieces they read.
// SessionGuard is the browser session a login form starts; TokenGuard reads an
// API token off the Authorization header; RequestGuard hands the request to a
// callback that resolves the user itself. Alongside them are Recaller, which parses the
// "remember me" cookie, GenericUser, MustVerifyEmailTrait, the authentication
// error, and AuthManager, which builds a guard from configuration.
//
// The eight events the session guard fires are in session_guard.go, and the
// interfaces the guards need from the session, the cookie jar, the event
// dispatcher, the request and the timebox are in collaborators.go.
//
// The root imports nothing but the standard library, deliberately. Everything
// that scopes itself by tenant -- the database, the cache, the filesystem, the
// scheduler -- imports this package to read Tenant off a Grant, so a dependency
// here would be a dependency everywhere. That is why the collaborators are
// interfaces declared here, in the consuming package, with the signatures
// hesape/session, hesape/cookie, hesape/events and hesape/http already have:
// they satisfy them structurally, and nothing imports anything.
//
// # What is elsewhere
//
// The user providers that read a database are hesape/auth/users; the middleware
// is hesape/auth/middleware; the Gate is hesape/auth/access; the listener that
// asks a fresh registration to confirm its address is hesape/auth/listeners.
//
// The reset link and the verification link are neither here nor in a subpackage
// of this one. Both are signed rather than stored: the application mints a token
// over the account it is for and the address it was mailed to, and nothing is
// written when the message goes out. A second flow here that kept a token in a
// table would be a second answer to the same question, and the two do not
// revoke alike.
package auth
