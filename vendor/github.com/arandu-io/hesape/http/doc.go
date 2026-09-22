// Package http is the request and response layer a controller action works
// with: the incoming Request, the Response types a handler can build, and
// the Context an action is called with.
//
// Sub-packages:
//
//	http/client     is an HTTP client for calling other services.
//	http/exceptions holds the exception types a response can carry.
//	http/middleware is the standard middleware every application wires.
//	http/resources  transforms data into response shapes.
//	http/testing    helps a test build requests and inspect responses.
//
// # The tenant never comes from the request
//
// There is no method on Request that reads a tenant id out of a path
// parameter, a query string, a header or a body field, and adding one would
// be the most direct route to a cross-tenant leak. The tenant is on the
// auth.Grant the policy mints, and the repository reads it from there.
//
// If a method here seems to offer tenant access -- Server, Header, Input --
// it does not. They read what the browser sent; the tenant is what the Grant
// authorises, never what the request carries.
//
// # Net/http and the package name
//
// The package is named http because it is the HTTP layer, and the directory
// is the component name. When it collides with net/http, the cost of the
// alias is on the caller:
//
//	import (
//	    "net/http"
//	    hhttp "github.com/arandu-io/hesape/http"
//	)
//
// Inside this package, net/http is imported as stdhttp.
package http
