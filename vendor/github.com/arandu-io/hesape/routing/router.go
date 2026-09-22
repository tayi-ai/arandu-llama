package routing

import (
	"net/http"
	"regexp"
	"strings"

	"github.com/arandu-io/hesape/pipeline"
)

// anyMethod is what the table shows for a route registered without a method.
//
// http.ServeMux treats a pattern with no method as matching every method, and
// the table has to print something; ANY is that word.
const anyMethod = "ANY"

// Router is a thin shell over http.ServeMux.
//
// The mux already handles methods and path parameters. This exists for groups,
// per-group middleware and route metadata -- the metadata is what lets the CLI
// print the routes and what a view uses to build a URL from a name.
//
// It holds no request state. An earlier version carried the view renderer and
// the flash so that it could build a request context itself, which is what made
// registering a controller a second registration path; both belong to the layer
// that owns the request context, and neither is here.
//
// Configuration that is router-level rather than per-sub-router -- the global
// where patterns, the matched listeners, the middleware aliases and groups,
// the explicit binders and the view renderer -- lives on the root and is
// reached through [Router.root], so a sub-router created by Group reads the
// same map the kernel populated.
type Router struct {
	mux    *http.ServeMux
	prefix string
	name   string
	module string
	mws    []pipeline.Middleware[http.Handler]
	table  *Routes

	// root is the router the kernel built, shared with every sub-router so
	// the configuration below is one set and not one per group.
	root *Router

	// patterns are the global where constraints, applied to every route at
	// registration. Pattern and Patterns set them.
	patterns map[string]string
	// matchedCallbacks fire after a route matches and before its middleware.
	// Matched registers them.
	matchedCallbacks []func(*Route, *http.Request)
	// middlewareAliases name a middleware for route introspection and for
	// resolution by the kernel; AliasMiddleware sets them.
	middlewareAliases map[string]pipeline.Middleware[http.Handler]
	// middlewareGroups are named bundles of middleware the kernel composes;
	// MiddlewareGroup sets them.
	middlewareGroups map[string][]pipeline.Middleware[http.Handler]
	// binders are the explicit route parameter resolvers; Bind and Model set
	// them, SubstituteBindings applies them. Uses RouteBindings from
	// bindings.go, shared with URL generation.
	binders *RouteBindings
	// viewRenderer renders the view a View route answers with; SetViewRenderer
	// wires it and View uses it. The concrete is hesape/view.
	viewRenderer ViewRenderer
	// implicitBindingCallback runs instead of the built-in implicit binding
	// when SubstituteImplicitBindingsUsing set one.
	implicitBindingCallback func(rt *Route, req *http.Request) error

	// groupStack is the attributes of every enclosing group, outermost first.
	// A sub-router is created per Group call, so each carries the stack it
	// was created under and nothing has to be popped.
	groupStack []map[string]any
}

// Group is what a sub-router adds to everything registered under it.
//
// It is one struct rather than three arguments because the three are always
// decided together, and because a positional signature makes the common case --
// a prefix and nothing else -- indistinguishable at the call site from the case
// that also names its routes.
//
//	admin := r.Group(routing.Group{
//		Prefix:     "/admin",
//		Name:       "admin",
//		Middleware: []pipeline.Middleware[http.Handler]{auth.RequireRole("admin")},
//	})
type Group struct {
	// Prefix is prepended to the path of every route registered under it.
	Prefix string
	// Name is prepended, dot-joined, to the name of every route registered
	// under it. See Route.Name.
	Name string
	// Middleware wraps every route registered under it, outside any middleware
	// the route gives itself, and inside the middleware of any enclosing group.
	Middleware []pipeline.Middleware[http.Handler]
}

// NewRouter returns an empty router.
func NewRouter() *Router {
	r := &Router{mux: http.NewServeMux(), table: NewRoutes()}
	r.root = r
	r.patterns = map[string]string{}
	r.middlewareAliases = map[string]pipeline.Middleware[http.Handler]{}
	r.middlewareGroups = map[string][]pipeline.Middleware[http.Handler]{}
	r.binders = NewRouteBindings()
	return r
}

// Group returns a sub-router with the prefix and name appended and the
// middleware inherited. The route table is shared with the parent, and so is
// the root configuration -- patterns, matched listeners, aliases, groups,
// binders and the view renderer -- so a sub-router reads what the kernel set.
func (r *Router) Group(g Group) *Router {
	return &Router{
		mux:    r.mux,
		prefix: joinPath(r.prefix, g.Prefix),
		name:   joinName(r.name, g.Name),
		module: r.module,
		mws:    append(append([]pipeline.Middleware[http.Handler]{}, r.mws...), g.Middleware...),
		table:  r.table,
		root:   r.root,
		groupStack: append(append([]map[string]any{}, r.groupStack...), map[string]any{
			"prefix":     g.Prefix,
			"as":         g.Name,
			"middleware": g.Middleware,
		}),
	}
}

// ForModule returns a sub-router that tags its routes with the module name,
// for grouping in route introspection. The kernel calls it for each module.
func (r *Router) ForModule(name string) *Router {
	g := r.Group(Group{})
	g.module = name
	return g
}

// Get registers a GET route.
func (r *Router) Get(pattern string, h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	return r.handle(http.MethodGet, pattern, h, mws...)
}

// Post registers a POST route.
func (r *Router) Post(pattern string, h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	return r.handle(http.MethodPost, pattern, h, mws...)
}

// Put registers a PUT route.
func (r *Router) Put(pattern string, h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	return r.handle(http.MethodPut, pattern, h, mws...)
}

// Patch registers a PATCH route.
func (r *Router) Patch(pattern string, h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	return r.handle(http.MethodPatch, pattern, h, mws...)
}

// Delete registers a DELETE route.
func (r *Router) Delete(pattern string, h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	return r.handle(http.MethodDelete, pattern, h, mws...)
}

// Match registers one handler under several methods.
//
//	r.Match([]string{"GET", "POST"}, "/search", search)
//
// It returns the route of the first method, and that is the row a name is
// registered against: the others are in the table, they carry the name for
// display, and generating a URL from it has one answer rather than one per
// verb. They all point at the same pattern, so there is nothing to choose
// between.
//
// An empty list of methods panics. It would otherwise register a route that
// answers nothing, at boot, silently -- and the symptom is a 404 on a path
// that the route table says exists.
func (r *Router) Match(methods []string, pattern string, h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	if len(methods) == 0 {
		panic("routing: Match was given no methods. Name at least one, or use Any")
	}
	var first *Route
	for _, method := range methods {
		route := r.handle(strings.ToUpper(method), pattern, h, mws...)
		if first == nil {
			first = route
			continue
		}
		first.siblings = append(first.siblings, route)
	}
	return first
}

// Any registers a route that answers every method.
//
// It is one registration and one row in the table, because that is what the mux
// does with a pattern that names no method. Reach for it for a webhook endpoint
// whose sender is not specific about the verb, and for nothing else: a route
// that also answers POST is a route where the CSRF check and the policy were
// written for a request shape that is not the one arriving.
func (r *Router) Any(pattern string, h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	return r.handle("", pattern, h, mws...)
}

// Fallback registers what answers everything under this router that no other
// route matched.
//
// On the root router that is the whole application, which is where a custom 404
// page goes. On a group it is that group's subtree: a group under /api can
// answer its own misses with JSON while the rest of the site answers with a
// page.
//
// It is registered as the subtree pattern of the group -- "/" at the root,
// "/api/" under a group prefixed /api -- which http.ServeMux already treats as
// the least specific match, so every other route in the subtree still wins.
func (r *Router) Fallback(h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	pattern := r.prefix
	if !strings.HasSuffix(pattern, "/") {
		pattern += "/"
	}
	route := r.register("", pattern, h, mws...)
	route.fallback = true
	return route
}

// Table returns the route table, for URL generation and route introspection.
func (r *Router) Table() *Routes { return r.table }

// Routes returns the registered routes, in registration order.
func (r *Router) Routes() []*Route { return r.table.All() }

// ServeHTTP dispatches to the underlying mux.
func (r *Router) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	r.mux.ServeHTTP(w, req)
}

// handle registers a pattern relative to the router's prefix.
func (r *Router) handle(method, pattern string, h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	return r.register(method, joinPath(r.prefix, pattern), h, mws...)
}

// register registers an already-joined pattern. An empty method matches every
// method, which is what the mux does with a pattern that names none.
//
// The route itself is what the mux dispatches to (Route.ServeHTTP), not a
// handler wrapped and frozen here. That lets Where, Middleware, WithoutMiddleware,
// the route-in-context and a matched listener all act on a route already in the
// mux, which is the shape every fluent call returns.
func (r *Router) register(method, full string, h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	key := full
	label := anyMethod
	if method != "" {
		key = method + " " + full
		label = method
	}

	route := &Route{
		Method:     label,
		Pattern:    full,
		Module:     r.module,
		namePrefix: r.name,
		prefix:     r.prefix,
		table:      r.table,
		router:     r.root,
		handler:    h,
		mws:        append(append([]pipeline.Middleware[http.Handler]{}, r.mws...), mws...),
		wheres:     map[string]*regexp.Regexp{},
		defaults:   map[string]any{},
		action:     map[string]any{"uses": "Closure", "controller": "Closure"},
	}
	// Global where patterns apply to every route. They are compiled once, at
	// registration.
	for name, expr := range r.root.patterns {
		if re, err := regexp.Compile(expr); err == nil {
			route.wheres[name] = re
			route.whereOrder = append(route.whereOrder, name)
		}
	}
	r.mux.Handle(key, route)
	r.table.add(route)
	return route
}

func joinPath(a, b string) string {
	a = strings.TrimSuffix(a, "/")
	if b == "" || b == "/" {
		if a == "" {
			return "/"
		}
		return a
	}
	if !strings.HasPrefix(b, "/") {
		b = "/" + b
	}
	return a + b
}
