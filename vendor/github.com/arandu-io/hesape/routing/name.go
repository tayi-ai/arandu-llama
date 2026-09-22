package routing

import (
	"fmt"
	"net/http"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/pipeline"
)

// Route is one registered route.
//
// It is metadata and the handler the mux dispatches to: what route
// introspection prints, what the error page shows for the pattern that
// matched, what a URL is generated from, and the http.Handler that answers.
// The handler is stored here rather than handed to the mux wrapped and frozen
// at registration, so a where constraint, a middleware added after
// registration and a route model binding can all take effect on a route that
// is already registered -- which is the shape every fluent call returns.
type Route struct {
	// Method is the HTTP method the route answers, or ANY for a route
	// registered without one.
	Method string
	// Pattern is the full path, prefixes of every enclosing group included.
	Pattern string
	// Module is the module that registered the route, for grouping in route
	// introspection. It is empty for a route registered outside a module.
	Module string

	name       string
	namePrefix string
	table      *Routes

	// siblings are the extra rows a single registration produced: the PATCH of
	// a resource update, the second and later methods of Match. They carry the
	// name for display and are deliberately not indexed by it, so generating a
	// URL from that name has one answer rather than two.
	siblings []*Route

	// handler is what the route dispatches to once matching, where constraints
	// and the route-in-context are settled. It is set at registration.
	handler http.Handler
	// mws are the middleware wrapping the handler, group middleware first. It
	// is read at request time, so Middleware called after registration takes
	// effect on a route already in the mux.
	mws []pipeline.Middleware[http.Handler]
	// excluded are middleware removed from mws at request time, by identity.
	// WithoutMiddleware appends here.
	excluded []pipeline.Middleware[http.Handler]
	// wheres are the regular-expression constraints keyed by parameter name.
	// A request whose path value fails the regex is answered 404 before the
	// handler, which is what stops /users/{id} from reaching the database for
	// /users/abc. See Where and WhereNumber.
	wheres map[string]*regexp.Regexp
	// whereOrder keeps wheres deterministic for matching and display.
	whereOrder []string
	// defaults are the fixed parameter values the route carries, set with
	// Defaults. Redirect and View use them to hand a destination or a view
	// name to the handler.
	defaults map[string]any
	// action describes the handler for display and for binding: "uses" is the
	// action string ("PostController@show" or "Closure"), "controller" is its
	// controller part. Set with Uses.
	action map[string]any
	// domain is the host the route is declared for, if any. http.ServeMux does
	// not route by host, so this is for URL generation and matching only.
	domain string
	// prefix is the path prefix the route was registered under, recorded so
	// GetPrefix can answer it.
	prefix string
	// withTrashed allows soft-deleted records through implicit model binding.
	// The implicit binding (hesape/routing, in the binding files) reads it.
	withTrashed bool
	// scopedBindings is whether the route enforces scoped implicit bindings.
	// nil = unset, true = ScopeBindings, false = WithoutScopedBindings.
	scopedBindings *bool
	// missingHandler answers when an implicit model binding resolves nothing.
	missingHandler http.Handler
	// router is the root router, for firing matched callbacks.
	router *Router
	// bindingFields are the {param:field} qualifiers parsed off the URI, used
	// by the route model binding to resolve a model by a column other than id.
	bindingFields map[string]string
	// fallback marks a route registered with Fallback, so matching treats it
	// as the last resort.
	fallback bool
	// lockSeconds and waitSeconds are what Block set: how long the route holds
	// the session lock, and how long it waits to take it. nil means no lock at
	// all.
	lockSeconds *int
	waitSeconds *int
	// can is the action this route requires, as Can declared it. It is
	// metadata: nothing in the dispatch reads it, and the policy is still what
	// decides. See Can.
	can auth.Action
	// deprecatedSince and sunset are what Deprecated set: the instant the
	// route was deprecated, and the instant it stops answering. Both are zero
	// on a route that is not deprecated, and Deprecated refuses to set one
	// without the other.
	deprecatedSince time.Time
	sunset          time.Time
}

// triState returns a pointer to b, the shape scopedBindings stores so the
// difference between "unset" and "false" survives.
func triState(b bool) *bool { return &b }

// RouteName returns the name given with Name, or empty.
//
// The exported field used to be called Name and nothing ever wrote to it -- a
// promise the code did not keep. Now Name(...) writes it and this reads it.
func (r *Route) RouteName() string {
	if r == nil {
		return ""
	}
	return r.name
}

// Name gives the route a name, so a URL can be generated from it instead of
// written by hand.
//
//	r.Get("/", home).Name("home")
//	routing.Resource(r, "invoices", InvoiceController{}, adapt) // names them all
//
// It returns the route so the call chains, and the declaration reads as one
// line.
//
// The name of every enclosing group is prepended, joined with a dot:
// Group{Name: "admin"} around .Name("users") gives "admin.users". The dot is
// joined here rather than left for the caller to remember, because a
// forgotten dot produces "adminusers", which is a name that works everywhere
// until somebody reads it.
func (r *Route) Name(name string) *Route {
	if r == nil {
		return nil
	}
	full := joinName(r.namePrefix, name)
	r.name = full
	for _, s := range r.siblings {
		s.name = full
	}
	if r.table != nil {
		r.table.register(full, r)
	}
	return r
}

// Routes is the table of registered routes, and the index by name.
type Routes struct {
	mu     sync.RWMutex
	all    []*Route
	byName map[string]*Route
}

// NewRoutes returns an empty table.
//
// A Router builds its own; this is for the caller that has to hold one before
// there is a router -- a test, and the URL generator the view layer is handed
// at boot.
func NewRoutes() *Routes { return &Routes{byName: map[string]*Route{}} }

// Route builds the path of a named route, filling the parameters in order.
//
//	Route("home")                  -> "/"
//	Route("invoices.show", "42")   -> "/invoices/42"
//
// A hardcoded "/invoices/"+id compiles and keeps compiling after the route
// moves. This does not: an unknown name or a wrong number of parameters is an
// error the caller sees, not a 404 the person sees.
//
// It returns an error rather than panicking, because a URL is often built from
// data -- and a panic in a template renderer takes the whole page down to
// report something a broken link would have said better.
func (t *Routes) Route(name string, params ...string) (string, error) {
	t.mu.RLock()
	route, known := t.byName[name]
	t.mu.RUnlock()

	if !known {
		return "", fmt.Errorf("routing: no route named %q. Name it with .Name(%q), or run `aru routes` to see what exists", name, name)
	}

	// "{$}" is not a parameter. It is the anchor that stops a pattern ending in
	// a slash from matching everything below it, which is what "GET /{$}" means
	// and what registering the root route does by default. Reading it as a
	// parameter made Route("home") return an error for the one route every
	// application has.
	out := strings.TrimSuffix(route.Pattern, "{$}")
	if out == "" {
		out = "/"
	}

	var missing []string
	for _, segment := range strings.Split(out, "/") {
		if !strings.HasPrefix(segment, "{") || !strings.HasSuffix(segment, "}") {
			continue
		}
		if len(params) == 0 {
			missing = append(missing, segment)
			continue
		}
		out = strings.Replace(out, segment, params[0], 1)
		params = params[1:]
	}

	if len(missing) > 0 {
		return "", fmt.Errorf("routing: route %q needs %s and got none", name, strings.Join(missing, ", "))
	}
	if len(params) > 0 {
		return "", fmt.Errorf("routing: route %q takes fewer parameters than the %d given", name, len(params))
	}
	return out, nil
}

// Must is Route for the places that cannot handle an error -- a template
// helper, mostly. It returns the message as the href, so a broken link says
// what is wrong instead of pointing at "/".
func (t *Routes) Must(name string, params ...string) string {
	out, err := t.Route(name, params...)
	if err != nil {
		return "#" + err.Error()
	}
	return out
}

// All returns the routes in registration order, for route introspection.
func (t *Routes) All() []*Route {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return append([]*Route(nil), t.all...)
}

func (t *Routes) add(r *Route) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.all = append(t.all, r)
}

// register indexes a route by name.
//
// Registering the same name twice panics rather than replacing. Two routes with
// one name means every URL built from it goes to one of them, chosen by
// registration order -- and finding that out at boot beats finding it out from
// a link that quietly points at the wrong page.
func (t *Routes) register(name string, r *Route) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if existing, taken := t.byName[name]; taken && existing != r {
		panic(fmt.Sprintf("routing: two routes named %q: %s %s and %s %s",
			name, existing.Method, existing.Pattern, r.Method, r.Pattern))
	}
	t.byName[name] = r
}

// joinName joins a group's name prefix to a route's name with a dot, and
// tolerates a prefix that already ends in one.
func joinName(prefix, name string) string {
	prefix = strings.TrimSuffix(prefix, ".")
	switch {
	case prefix == "":
		return name
	case name == "":
		return prefix
	default:
		return prefix + "." + name
	}
}
