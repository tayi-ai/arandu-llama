package matching

import "net/http"

// Route is what a validator needs to read off a route.
//
// It is an interface and not the route type because hesape/routing imports
// this package to answer Route.GetValidators, and naming the route type here
// would be a cycle. hesape/routing.Route satisfies it.
type Route interface {
	// GetDomain returns the host the route is declared for, or empty.
	GetDomain() string
	// Methods returns the HTTP verbs the route answers.
	Methods() []string
	// HttpOnly reports whether the route rejects https.
	HttpOnly() bool
	// Secure reports whether the route rejects http.
	Secure() bool
	// URI returns the route's pattern.
	URI() string
	// Matches reports whether the path matches the route pattern. There is no
	// compiled regex to ask: the pattern is matched by the same code the mux
	// match uses.
	Matches(req *http.Request, includeMethod bool) bool
}

// ValidatorInterface is one question asked of a route and a request: does this
// route answer it? This package has the four implementations that matter --
// [UriValidator], [MethodValidator], [SchemeValidator] and [HostValidator] --
// and a route answers a request only when every one of them says so.
//
// hesape/routing.Route.GetValidators returns the four, which is the set a
// caller ranges over. Nothing here dispatches a request: the mux has already
// picked a route by method and path, and these exist for the callers that
// still have to ask -- a middleware, a URL generator, a custom dispatcher.
type ValidatorInterface interface {
	// Matches reports whether route answers req.
	Matches(route Route, req *http.Request) bool
}
