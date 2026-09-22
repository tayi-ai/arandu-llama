package routing

import (
	"net/http"
	"strings"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/routing/matching"
)

// Scheme declares the scheme the route answers on, "http" or "https". It is
// stored as one key of the route's action map. HttpOnly, HttpsOnly and Secure
// are the readers, and this is what writes what they read.
func (rt *Route) Scheme(scheme string) *Route {
	if rt == nil {
		return nil
	}
	rt.action["scheme"] = strings.ToLower(scheme)
	return rt
}

// HttpOnly reports whether the route answers only plain http, which a route
// generating a URL for a local development host declares.
func (rt *Route) HttpOnly() bool {
	if rt == nil {
		return false
	}
	scheme, _ := rt.action["scheme"].(string)
	return scheme == "http"
}

// HttpsOnly is an alias for Secure.
func (rt *Route) HttpsOnly() bool { return rt.Secure() }

// Secure reports whether the route answers only https, which is what makes
// its generated URL https regardless of how the current request arrived.
func (rt *Route) Secure() bool {
	if rt == nil {
		return false
	}
	scheme, _ := rt.action["scheme"].(string)
	return scheme == "https"
}

// GetOptionalParameterNames returns the names of the route's optional path
// parameters. The values are nil: the map is a set, and the question asked
// of it is only whether a name is in it.
func (rt *Route) GetOptionalParameterNames() map[string]any {
	out := map[string]any{}
	if rt == nil {
		return out
	}
	rest := rt.Pattern
	for {
		start := strings.IndexByte(rest, '{')
		if start < 0 {
			return out
		}
		end := strings.IndexByte(rest[start:], '}')
		if end < 0 {
			return out
		}
		end += start
		inner := rest[start+1 : end]
		if strings.HasSuffix(inner, "?") {
			name := strings.TrimSuffix(inner, "?")
			if i := strings.IndexByte(name, ':'); i >= 0 {
				name = name[:i]
			}
			out[name] = nil
		}
		rest = rest[end+1:]
	}
}

// OriginalParameters returns every path parameter as it arrived, before any
// binding replaced an id with the record it names -- which is what a binder
// compares against to know it changed something.
//
// The request is the argument, so a call against an unbound route is not
// expressible.
func (rt *Route) OriginalParameters(req *http.Request) map[string]string {
	out := map[string]string{}
	if rt == nil || req == nil {
		return out
	}
	for _, name := range rt.ParameterNames() {
		if v := req.PathValue(name); v != "" {
			out[name] = v
		}
	}
	return out
}

// SetRouter attaches the router whose matched listeners the route fires and
// whose binders it resolves through.
func (rt *Route) SetRouter(router *Router) *Route {
	if rt == nil {
		return nil
	}
	if router == nil {
		rt.router = nil
		return rt
	}
	rt.router = router.root
	return rt
}

// Can declares which authorization action this route requires.
//
//	r.Delete("/invoices/{id}", destroy).Name("invoices.destroy").Can("invoice.delete")
//
// It declares and it does not decide, and that difference is the whole of what
// it is for. Nothing in the dispatch reads it: the handler still asks
// auth.Authorize, the policy still answers, and the repository still refuses
// without the Grant that answer produced. A route carrying this and no
// authorization is exactly as open as one carrying neither -- which is why it
// cannot become a second way to authorize, and must not be read as one.
//
// What it is for is the catalogue. A permissions screen has to list what may be
// granted, and the honest list is the router: actions written down beside it
// disagree with the code the first time somebody adds a route. Requirements
// reads it back.
//
// One route requires one action, because Grant.Check compares one. Declaring a
// second replaces the first rather than adding to it.
//
// It used to take a string and a list of models, and to append them into the
// action map -- which nothing ever read. The type is the action the policy is
// asked about, and the reader is Requirements: what a route requires is now
// answerable rather than merely recorded.
//
// The declaration is carried to the sibling rows of a single registration --
// the PATCH of an update, the later verbs of a Match -- because they are one
// route to everyone but the mux.
func (rt *Route) Can(action auth.Action) *Route {
	if rt == nil {
		return nil
	}
	rt.can = action
	for _, sibling := range rt.siblings {
		sibling.can = action
	}
	return rt
}

// RequiredAction returns the action Can declared, or empty.
//
// It is not called Action because that name is the controller action string,
// and the two are different questions: one is which method answers, this is
// what the caller has to be allowed to do.
func (rt *Route) RequiredAction() auth.Action {
	if rt == nil {
		return ""
	}
	return rt.can
}

// Block stops two requests of the same session from running this route at
// once -- a double-submitted form, mostly -- by holding a lock for
// lockSeconds and waiting up to waitSeconds to take it.
func (rt *Route) Block(lockSeconds, waitSeconds *int) *Route {
	if rt == nil {
		return nil
	}
	rt.lockSeconds = lockSeconds
	rt.waitSeconds = waitSeconds
	return rt
}

// WithoutBlocking clears any lock Block set.
func (rt *Route) WithoutBlocking() *Route { return rt.Block(nil, nil) }

// LocksFor returns the lock duration Block set. nil means the route takes no
// lock.
func (rt *Route) LocksFor() *int {
	if rt == nil {
		return nil
	}
	return rt.lockSeconds
}

// WaitsFor returns the wait duration Block set. nil means the route takes no
// lock.
func (rt *Route) WaitsFor() *int {
	if rt == nil {
		return nil
	}
	return rt.waitSeconds
}

// routeValidators are the four checks. They hold no state, so one set serves
// every route.
var routeValidators = []matching.ValidatorInterface{
	matching.UriValidator{},
	matching.MethodValidator{},
	matching.SchemeValidator{},
	matching.HostValidator{},
}

// GetValidators returns the four checks that decide whether the route
// answers a request: path, method, scheme and host.
func (rt *Route) GetValidators() []matching.ValidatorInterface {
	return append([]matching.ValidatorInterface(nil), routeValidators...)
}
