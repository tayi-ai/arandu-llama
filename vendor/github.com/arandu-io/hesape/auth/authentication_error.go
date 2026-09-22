package auth

import "sync"

// defaultAuthenticationMessage is what an error built with no message says.
const defaultAuthenticationMessage = "Unauthenticated."

// AuthenticationError reports that nobody is signed in, on a path that requires
// somebody to be.
//
// It carries the guards that were asked, so a middleware can say which ones,
// and the path to send the browser to -- which is the whole reason it is a type
// rather than a bare 401: a person who followed a link gets the sign-in page,
// an API client gets the status.
//
// It is not the authorization failure. That one is [ErrForbidden], and the
// difference is the difference between 401 and 403: this says the request had
// no identity, ErrForbidden says the identity it had may not do that.
type AuthenticationError struct {
	// Message is the sentence the error answers with.
	Message string

	// guards are the guards that were checked.
	guards []string

	// redirectTo is where the browser should be sent, when the error names
	// somewhere itself.
	redirectTo string
}

// NewAuthenticationError returns the error.
//
// An empty message becomes "Unauthenticated.", a nil slice means no guard was
// named, and an empty redirect falls back to [RedirectUsing].
func NewAuthenticationError(message string, guards []string, redirectTo string) *AuthenticationError {
	if message == "" {
		message = defaultAuthenticationMessage
	}
	return &AuthenticationError{Message: message, guards: guards, redirectTo: redirectTo}
}

// Error is the sentence the error answers with.
func (e *AuthenticationError) Error() string {
	if e.Message == "" {
		return defaultAuthenticationMessage
	}
	return e.Message
}

// Guards are the guards that were checked.
func (e *AuthenticationError) Guards() []string {
	return e.guards
}

// RedirectTo is where the browser should be sent, or "" when the answer is the
// status code and not a page.
//
// It falls back to the callback registered with [RedirectUsing].
func (e *AuthenticationError) RedirectTo(request Request) string {
	if e.redirectTo != "" {
		return e.redirectTo
	}

	redirectMu.RLock()
	callback := redirectToCallback
	redirectMu.RUnlock()

	if callback != nil {
		return callback(request)
	}
	return ""
}

// redirectToCallback is the application-wide fallback [RedirectUsing] sets.
//
// It is written once at boot and read on every failed request, so it is behind
// a lock: a Go server answers requests concurrently.
var (
	redirectMu         sync.RWMutex
	redirectToCallback func(request Request) string
)

// RedirectUsing sets the callback that builds the redirect path when the error
// carries none.
//
// Call it once, where the application boots. It is not a per-request setting.
func RedirectUsing(callback func(request Request) string) {
	redirectMu.Lock()
	defer redirectMu.Unlock()
	redirectToCallback = callback
}
