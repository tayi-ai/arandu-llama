package auth

import "context"

// RequestGuard is a guard that is one callback.
//
// It is what [AuthManager.ViaRequest] registers, and it is the seam for a way
// of proving identity that this package knows nothing about -- a signed header,
// a mutual TLS certificate, a token service somewhere else. The callback gets
// the request and the provider and answers with a user or nil.
type RequestGuard struct {
	GuardHelpers

	// callback is the guard, as a function.
	callback func(request Request, provider UserProvider) Authenticatable

	// request is what the callback is handed.
	request Request
}

var _ Guard = (*RequestGuard)(nil)

// NewRequestGuard returns a guard that is the callback.
//
// The provider may be nil: a callback that needs no user provider is given
// none.
func NewRequestGuard(callback func(request Request, provider UserProvider) Authenticatable, request Request, provider UserProvider) *RequestGuard {
	guard := &RequestGuard{
		callback: callback,
		request:  request,
	}
	guard.provider = provider
	guard.resolveUser = guard.User

	return guard
}

// User is whatever the callback says, resolved once per request.
func (g *RequestGuard) User() Authenticatable {
	// If we have already retrieved the user for the current request we can just
	// return it back immediately.
	if g.user != nil {
		return g.user
	}

	if g.callback == nil {
		return nil
	}

	g.user = g.callback(g.request, g.GetProvider())

	return g.user
}

// Validate runs the callback against the request in the credentials, without
// touching this guard's resolved user.
//
// It builds a second guard for exactly that reason. The credentials carry the
// request under the key "request".
//
// The context is the Guard contract's, and this method does no I/O of its own:
// whatever the callback does runs on the context the request carries.
func (g *RequestGuard) Validate(ctx context.Context, credentials map[string]any) bool {
	request, _ := credentials["request"].(Request)

	return NewRequestGuard(g.callback, request, g.GetProvider()).User() != nil
}

// SetRequest sets the request the callback is handed, and returns the guard.
func (g *RequestGuard) SetRequest(request Request) *RequestGuard {
	g.request = request

	return g
}
