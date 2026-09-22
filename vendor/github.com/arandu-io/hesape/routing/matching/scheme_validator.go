package matching

import "net/http"

// SchemeValidator validates that the incoming request's scheme satisfies the
// route's restriction. A route with no restriction passes; an http-only route
// rejects https, and a secure route rejects http.
type SchemeValidator struct{}

// Matches reports whether req's scheme satisfies route's restriction.
func (v SchemeValidator) Matches(route Route, req *http.Request) bool {
	secure := req != nil && req.TLS != nil
	if route.HttpOnly() {
		return !secure
	}
	if route.Secure() {
		return secure
	}
	return true
}
