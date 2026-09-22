package routing

import (
	"net/http"

	"github.com/arandu-io/hesape/pipeline"
)

// RouteRegistrar collects group attributes fluently and applies them to the
// routes registered through it: a builder that holds prefix, name,
// middleware, domain, namespace and where, and that hands them to the router
// when a route is registered.
//
// The Router methods Prefix, Name, Middleware, Domain and Namespace return
// one, so the call chains:
//
//	r.Prefix("/api").Name("api.").Middleware(auth.RequireLogin).Get("/users", h)
//
// The attributes it carries are the same fields Group carries, plus a where map
// for the Where* constraints the registrar forwards to each route. It does not
// hold a group stack: a registrar is one layer, and nesting is the router's
// Group.
type RouteRegistrar struct {
	router     *Router
	prefix     string
	name       string
	middleware []pipeline.Middleware[http.Handler]
	domain     string
	namespace  string
	wheres     map[string]string
}

// Prefix returns a registrar that prepends prefix to every route registered
// through it.
func (r *Router) Prefix(prefix string) *RouteRegistrar {
	return &RouteRegistrar{router: r, prefix: prefix}
}

// Name returns a registrar that prepends name to every route's name. The name
// is dot-joined, as Group.Name is.
func (r *Router) Name(name string) *RouteRegistrar {
	return &RouteRegistrar{router: r, name: name}
}

// Middleware returns a registrar that wraps every route in the given
// middleware.
func (r *Router) Middleware(mws ...pipeline.Middleware[http.Handler]) *RouteRegistrar {
	return &RouteRegistrar{router: r, middleware: mws}
}

// Domain returns a registrar that sets the domain on every route. The mux does
// not route by host, so this is for URL generation and matching.
func (r *Router) Domain(domain string) *RouteRegistrar {
	return &RouteRegistrar{router: r, domain: domain}
}

// Namespace returns a registrar that sets the namespace on every route. Go has
// no string-based controller resolution, so the namespace is recorded for
// display and for the URL generator, and does not change which handler answers.
func (r *Router) Namespace(ns string) *RouteRegistrar {
	return &RouteRegistrar{router: r, namespace: ns}
}

// Attribute sets one attribute on the registrar, for the caller that builds
// it piecemeal.
func (reg *RouteRegistrar) Attribute(key string, value any) *RouteRegistrar {
	switch key {
	case "prefix":
		if s, ok := value.(string); ok {
			reg.prefix = s
		}
	case "name", "as":
		if s, ok := value.(string); ok {
			reg.name = s
		}
	case "middleware":
		if mws, ok := value.([]pipeline.Middleware[http.Handler]); ok {
			reg.middleware = append(reg.middleware, mws...)
		}
	case "domain":
		if s, ok := value.(string); ok {
			reg.domain = s
		}
	case "namespace":
		if s, ok := value.(string); ok {
			reg.namespace = s
		}
	case "where":
		if m, ok := value.(map[string]string); ok {
			if reg.wheres == nil {
				reg.wheres = map[string]string{}
			}
			for k, v := range m {
				reg.wheres[k] = v
			}
		}
	}
	return reg
}

// Prefix sets the prefix on the registrar and returns it.
func (reg *RouteRegistrar) Prefix(prefix string) *RouteRegistrar {
	reg.prefix = prefix
	return reg
}

// Name sets the name prefix on the registrar and returns it.
func (reg *RouteRegistrar) Name(name string) *RouteRegistrar {
	reg.name = name
	return reg
}

// Middleware appends middleware to the registrar and returns it.
func (reg *RouteRegistrar) Middleware(mws ...pipeline.Middleware[http.Handler]) *RouteRegistrar {
	reg.middleware = append(reg.middleware, mws...)
	return reg
}

// Domain sets the domain on the registrar and returns it.
func (reg *RouteRegistrar) Domain(domain string) *RouteRegistrar {
	reg.domain = domain
	return reg
}

// Namespace sets the namespace on the registrar and returns it.
func (reg *RouteRegistrar) Namespace(ns string) *RouteRegistrar {
	reg.namespace = ns
	return reg
}

// Where sets a where constraint on the registrar and returns it. Every route
// registered through the registrar inherits it.
func (reg *RouteRegistrar) Where(name, pattern string) *RouteRegistrar {
	if reg.wheres == nil {
		reg.wheres = map[string]string{}
	}
	reg.wheres[name] = pattern
	return reg
}

// WhereNumber constrains the given parameters to digits.
func (reg *RouteRegistrar) WhereNumber(params ...string) *RouteRegistrar {
	return reg.assignExpr(params, `[0-9]+`)
}

// WhereAlpha constrains the given parameters to ASCII letters.
func (reg *RouteRegistrar) WhereAlpha(params ...string) *RouteRegistrar {
	return reg.assignExpr(params, `[a-zA-Z]+`)
}

// WhereAlphaNumeric constrains the given parameters to ASCII letters and digits.
func (reg *RouteRegistrar) WhereAlphaNumeric(params ...string) *RouteRegistrar {
	return reg.assignExpr(params, `[a-zA-Z0-9]+`)
}

// WhereUuid constrains the given parameters to the UUID format.
func (reg *RouteRegistrar) WhereUuid(params ...string) *RouteRegistrar {
	return reg.assignExpr(params, `[\da-fA-F]{8}-[\da-fA-F]{4}-[\da-fA-F]{4}-[\da-fA-F]{4}-[\da-fA-F]{12}`)
}

// WhereUlid constrains the given parameters to the ULID alphabet.
func (reg *RouteRegistrar) WhereUlid(params ...string) *RouteRegistrar {
	return reg.assignExpr(params, `[0-7][0-9a-hjkmnp-tv-zA-HJKMNP-TV-Z]{25}`)
}

// WhereIn constrains the given parameter to one of the listed values.
func (reg *RouteRegistrar) WhereIn(param string, values ...string) *RouteRegistrar {
	return reg.Where(param, joinValues(values, "|"))
}

func (reg *RouteRegistrar) assignExpr(params []string, expr string) *RouteRegistrar {
	if reg.wheres == nil {
		reg.wheres = map[string]string{}
	}
	for _, p := range params {
		reg.wheres[p] = expr
	}
	return reg
}

// Group creates a sub-router with the registrar's attributes and runs fn
// against it. It is the registrar's group, which forwards to Router.Group.
func (reg *RouteRegistrar) Group(fn func(*Router)) {
	g := Group{
		Prefix:     reg.prefix,
		Name:       reg.name,
		Middleware: reg.middleware,
	}
	sub := reg.router.Group(g)
	fn(sub)
}

// Get registers a GET route with the registrar's attributes applied.
func (reg *RouteRegistrar) Get(pattern string, h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	return reg.register(http.MethodGet, pattern, h, mws...)
}

// Post registers a POST route with the registrar's attributes applied.
func (reg *RouteRegistrar) Post(pattern string, h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	return reg.register(http.MethodPost, pattern, h, mws...)
}

// Put registers a PUT route with the registrar's attributes applied.
func (reg *RouteRegistrar) Put(pattern string, h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	return reg.register(http.MethodPut, pattern, h, mws...)
}

// Patch registers a PATCH route with the registrar's attributes applied.
func (reg *RouteRegistrar) Patch(pattern string, h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	return reg.register(http.MethodPatch, pattern, h, mws...)
}

// Delete registers a DELETE route with the registrar's attributes applied.
func (reg *RouteRegistrar) Delete(pattern string, h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	return reg.register(http.MethodDelete, pattern, h, mws...)
}

// Options registers an OPTIONS route with the registrar's attributes applied.
func (reg *RouteRegistrar) Options(pattern string, h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	return reg.register(http.MethodOptions, pattern, h, mws...)
}

// Any registers a route answering every method with the registrar's attributes.
func (reg *RouteRegistrar) Any(pattern string, h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	return reg.register("", pattern, h, mws...)
}

// Match registers one handler under several methods, with the registrar's
// attributes applied.
func (reg *RouteRegistrar) Match(methods []string, pattern string, h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	if len(methods) == 0 {
		panic("routing: Match was given no methods. Name at least one, or use Any")
	}
	var first *Route
	for _, m := range methods {
		rt := reg.register(m, pattern, h, mws...)
		if first == nil {
			first = rt
			continue
		}
		first.siblings = append(first.siblings, rt)
	}
	return first
}

// register applies the registrar's attributes and delegates to the router. The
// route is created through the router's handle, so the prefix, name and
// middleware the registrar carries become the route's, and the where
// constraints are attached after.
func (reg *RouteRegistrar) register(method, pattern string, h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	sub := reg.router.Group(Group{
		Prefix:     reg.prefix,
		Name:       reg.name,
		Middleware: reg.middleware,
	})
	rt := sub.handle(method, pattern, h, mws...)
	if reg.domain != "" {
		rt.Domain(reg.domain)
	}
	for name, expr := range reg.wheres {
		rt.Where(name, expr)
	}
	return rt
}

// joinValues joins values with sep, for WhereIn.
func joinValues(values []string, sep string) string {
	out := ""
	for i, v := range values {
		if i > 0 {
			out += sep
		}
		out += v
	}
	return out
}
