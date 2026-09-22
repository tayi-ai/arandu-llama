package auth

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
)

// TokenGuard is the API token guard.
//
// It reads a token off the Authorization header and asks the provider for the
// user whose stored token column matches. There is no session, no cookie and
// nothing to log out of: every request proves itself again.
//
// It is the simplest thing that works and it is not a token system. A token it
// accepts never expires, carries no scope, is never rotated and cannot be
// revoked: either the stored column matches or it does not, for as long as the
// row lives. All four belong to whatever issues the tokens, and until something
// issues them, wiring this guard as an application's way in means handing out
// credentials that cannot be taken back.
type TokenGuard struct {
	GuardHelpers

	// request is what the token is read from.
	request Request

	// inputKey is the credential key Validate reads the token under.
	inputKey string

	// storageKey is the column it is kept in.
	storageKey string

	// hash says the stored token is a SHA-256 of the one on the wire.
	hash bool
}

var _ Guard = (*TokenGuard)(nil)

// NewTokenGuard returns a guard that reads its token off the request's
// Authorization header.
//
// An empty inputKey or storageKey becomes "api_token".
func NewTokenGuard(provider UserProvider, request Request, inputKey, storageKey string, hash bool) *TokenGuard {
	if inputKey == "" {
		inputKey = "api_token"
	}
	if storageKey == "" {
		storageKey = "api_token"
	}

	guard := &TokenGuard{
		request:    request,
		inputKey:   inputKey,
		storageKey: storageKey,
		hash:       hash,
	}
	guard.provider = provider
	guard.resolveUser = guard.User

	return guard
}

// User is the user the request's token names, or nil.
//
// It resolves once per request and caches, including the nil: a request with a
// token nobody holds does not go back to the store on every call.
func (g *TokenGuard) User() Authenticatable {
	// If we have already retrieved the user for the current request we can just
	// return it back immediately.
	if g.user != nil {
		return g.user
	}

	var user Authenticatable

	if token := g.GetTokenForRequest(); token != "" {
		value := token
		if g.hash {
			sum := sha256.Sum256([]byte(token))
			value = hex.EncodeToString(sum[:])
		}

		if found, err := g.provider.RetrieveByCredentials(g.context(), map[string]any{g.storageKey: value}); err == nil {
			user = found
		}
	}

	g.user = user

	return g.user
}

// GetTokenForRequest is the token on the Authorization header, or empty.
//
// The bearer token first, then the HTTP Basic password, which is how a token is
// handed to a tool that only knows how to do Basic auth.
//
// The query string and the request body are not read, so a token sent in either
// is not seen and the request stays a guest. A URL reaches the server log, the
// proxy log, the browser history and the Referer header of every link followed
// from the page, and none of those is a place a credential can be taken back
// out of; a header reaches none of them. Reading the body instead would not
// help: a request merges its query string into its input, so a body reader is a
// second URL reader.
func (g *TokenGuard) GetTokenForRequest() string {
	if g.request == nil {
		return ""
	}

	token := g.request.BearerToken()

	if token == "" {
		token = g.request.GetPassword()
	}

	return token
}

// Validate reports whether this token is good, without resolving the guard's
// user.
func (g *TokenGuard) Validate(ctx context.Context, credentials map[string]any) bool {
	token, ok := credentials[g.inputKey]
	if !ok || isEmptyCredential(token) {
		return false
	}

	user, err := g.provider.RetrieveByCredentials(ctx, map[string]any{g.storageKey: token})

	return err == nil && user != nil
}

// SetRequest sets the request the token is read from, and returns the guard.
func (g *TokenGuard) SetRequest(request Request) *TokenGuard {
	g.request = request

	return g
}

// context is the context the lookup runs on. See [Request] for why the guard
// reads it off the request.
func (g *TokenGuard) context() context.Context {
	if g.request != nil {
		if ctx := g.request.Context(); ctx != nil {
			return ctx
		}
	}
	return context.Background()
}

// isEmptyCredential reports that a credential carries nothing: a nil value, or
// a string with nothing in it.
func isEmptyCredential(value any) bool {
	if value == nil {
		return true
	}
	if text, ok := value.(string); ok {
		return text == ""
	}
	return false
}
