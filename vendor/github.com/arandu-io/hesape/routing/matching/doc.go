// Package matching holds the four checks that decide whether a route answers
// a request: its host, its method, its scheme and its path.
//
// Route.GetValidators returns the four, and hesape/routing uses them for the
// questions the mux does not answer: the mux has already picked a route by
// method and path, and a middleware, a URL generator or a custom dispatcher
// still has to know whether a route answers a host and a scheme.
//
// There is no compiled route here, so UriValidator asks the route to match
// its own pattern instead, and HostValidator compares the declared domain to
// the request host.
package matching
