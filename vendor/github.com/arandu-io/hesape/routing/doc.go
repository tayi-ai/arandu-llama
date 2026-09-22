// Package routing decides which handler answers a request, and what the
// address of a handler is called.
//
// It is a thin shell over http.ServeMux, which since Go 1.22 already matches
// methods and path parameters. What this package adds is everything the
// standard mux has no opinion about: groups with an inherited prefix, name
// and middleware; a route table that survives registration so a view can
// build a URL from a name instead of a string literal; the seven REST routes
// of a resource controller; and the two middlewares that decide whether a
// request reaches a route at all.
//
// # One handler type
//
// A route dispatches to an http.Handler. There is one registration path and
// one handler type on it -- Get, Post, Put, Patch, Delete, Match, Any and
// Fallback all take the same thing -- because the shape this replaced had two:
// Get took an http.HandlerFunc and a controller arrived through a second
// method, Action, whose job was to build a handler out of a controller method
// and translate what it returned. That second path was never a second way to
// route. It was a way to construct a handler, and everything it touched -- the
// request context, the renderer, the flash, a rejected form -- belongs to the
// request layer above this one. Constructing the handler happens there now, and
// a controller action reaches a route the same way any other handler does:
//
//	r.Get("/dashboard", adapt(dashboard.Index)).Name("dashboard")
//
// where adapt is the adapter that layer supplies. This package neither declares
// it nor names one, because the type it builds is the one this package must not
// import.
//
// Resource takes the adaptation as an argument for the same reason, which is
// what lets this package register controller routes without importing the type
// a controller receives. See Adapter.
//
// # Names, not paths
//
// A hardcoded "/invoices/"+id compiles and keeps compiling after the route
// moves. Routes.Route does not: an unknown name or a wrong number of
// parameters is an error the caller sees, not a 404 the person sees.
//
// Composing middleware is hesape/pipeline, generic over what it wraps, and
// this package uses it rather than declaring a second one. The request
// context belongs to hesape/http, which owns what an answer looks like; the
// half of the signed-URL story that belongs here is SignedRoute and
// middleware.ValidateSignature, over the Signer in hesape/encryption.
package routing
