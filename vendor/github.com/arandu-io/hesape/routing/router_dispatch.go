package routing

import (
	"context"
	"net/http"
	"regexp"
	"strings"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/pipeline"
)

// Options registers an OPTIONS route.
func (r *Router) Options(pattern string, h http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	return r.handle(http.MethodOptions, pattern, h, mws...)
}

// Redirect registers a route that redirects to destination with status. The
// handler it installs writes the Location header and the status.
//
// It is registered under Any so that a link or form of any verb that lands
// here is redirected; the status is the caller's to set (302 for Redirect,
// 301 for PermanentRedirect).
func (r *Router) Redirect(uri, destination string, status ...int) *Route {
	code := http.StatusFound
	if len(status) > 0 {
		code = status[0]
	}
	h := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		http.Redirect(w, req, destination, code)
	})
	route := r.Any(uri, h)
	route.action["uses"] = "RedirectController"
	route.action["controller"] = "RedirectController"
	route.defaults["destination"] = destination
	route.defaults["status"] = code
	return route
}

// PermanentRedirect registers a 301 redirect route.
func (r *Router) PermanentRedirect(uri, destination string) *Route {
	return r.Redirect(uri, destination, http.StatusMovedPermanently)
}

// View registers a GET route that renders a view.
//
// The view renderer is the one SetViewRenderer wired; without it, the route
// answers 500, which is the honest answer to "the view layer was not given to
// the router". The handler it installs calls the renderer, and the data map is
// carried in the route's defaults -- the same place Redirect carries its
// destination, which keeps the two redirect-and-view routes shaped alike.
func (r *Router) View(pattern, view string, data ...map[string]any) *Route {
	d := map[string]any{}
	if len(data) > 0 {
		d = data[0]
	}
	status := http.StatusOK
	h := http.HandlerFunc(func(w http.ResponseWriter, req *http.Request) {
		vr := r.root.viewRenderer
		if vr == nil {
			http.Error(w, "routing: no view renderer wired", http.StatusInternalServerError)
			return
		}
		if err := vr.Render(w, view, d, status, nil); err != nil {
			http.Error(w, err.Error(), http.StatusInternalServerError)
		}
	})
	route := r.Get(pattern, h)
	route.action["uses"] = "ViewController"
	route.action["controller"] = "ViewController"
	route.defaults["view"] = view
	route.defaults["data"] = d
	route.defaults["status"] = status
	return route
}

// GetRoutes returns the route collection, for URL generation and route
// introspection.
func (r *Router) GetRoutes() *Routes { return r.table }

// Has reports whether a route with the given name is registered.
func (r *Router) Has(name string) bool { return r.table.HasNamedRoute(name) }

// HasMiddlewareGroup reports whether a middleware group with the given name
// exists.
func (r *Router) HasMiddlewareGroup(name string) bool {
	_, ok := r.root.middlewareGroups[name]
	return ok
}

// AliasMiddleware registers a short name for a middleware, for route
// introspection and for the kernel's resolution.
func (r *Router) AliasMiddleware(name string, mw pipeline.Middleware[http.Handler]) *Router {
	r.root.middlewareAliases[name] = mw
	return r
}

// GetMiddleware returns the middleware aliases.
func (r *Router) GetMiddleware() map[string]pipeline.Middleware[http.Handler] {
	out := make(map[string]pipeline.Middleware[http.Handler], len(r.root.middlewareAliases))
	for k, v := range r.root.middlewareAliases {
		out[k] = v
	}
	return out
}

// MiddlewareGroup registers a named bundle of middleware the kernel composes.
func (r *Router) MiddlewareGroup(name string, mws ...pipeline.Middleware[http.Handler]) *Router {
	r.root.middlewareGroups[name] = append(r.root.middlewareGroups[name], mws...)
	return r
}

// GetMiddlewareGroups returns the middleware groups.
func (r *Router) GetMiddlewareGroups() map[string][]pipeline.Middleware[http.Handler] {
	out := make(map[string][]pipeline.Middleware[http.Handler], len(r.root.middlewareGroups))
	for k, v := range r.root.middlewareGroups {
		out[k] = append([]pipeline.Middleware[http.Handler]{}, v...)
	}
	return out
}

// PrependMiddlewareToGroup adds a middleware to the start of a group, if it is
// not already there.
func (r *Router) PrependMiddlewareToGroup(name string, mw pipeline.Middleware[http.Handler]) *Router {
	group := r.root.middlewareGroups[name]
	for _, existing := range group {
		if middlewarePointer(existing) == middlewarePointer(mw) {
			return r
		}
	}
	r.root.middlewareGroups[name] = append([]pipeline.Middleware[http.Handler]{mw}, group...)
	return r
}

// PushMiddlewareToGroup adds a middleware to the end of a group, creating the
// group if it does not exist, if it is not already there.
func (r *Router) PushMiddlewareToGroup(name string, mw pipeline.Middleware[http.Handler]) *Router {
	group := r.root.middlewareGroups[name]
	for _, existing := range group {
		if middlewarePointer(existing) == middlewarePointer(mw) {
			return r
		}
	}
	r.root.middlewareGroups[name] = append(group, mw)
	return r
}

// RemoveMiddlewareFromGroup removes a middleware from a group by identity.
func (r *Router) RemoveMiddlewareFromGroup(name string, mw pipeline.Middleware[http.Handler]) *Router {
	group := r.root.middlewareGroups[name]
	target := middlewarePointer(mw)
	out := group[:0]
	for _, existing := range group {
		if middlewarePointer(existing) == target {
			continue
		}
		out = append(out, existing)
	}
	r.root.middlewareGroups[name] = out
	return r
}

// FlushMiddlewareGroups empties the middleware groups.
func (r *Router) FlushMiddlewareGroups() *Router {
	r.root.middlewareGroups = map[string][]pipeline.Middleware[http.Handler]{}
	return r
}

// Pattern sets a global where pattern, applied to every route at registration.
func (r *Router) Pattern(key, pattern string) {
	r.root.patterns[key] = pattern
}

// Patterns sets several global where patterns at once.
func (r *Router) Patterns(patterns map[string]string) {
	for k, v := range patterns {
		r.root.patterns[k] = v
	}
}

// GetPatterns returns the global where patterns.
func (r *Router) GetPatterns() map[string]string {
	out := make(map[string]string, len(r.root.patterns))
	for k, v := range r.root.patterns {
		out[k] = v
	}
	return out
}

// Matched registers a callback that fires after a route matches, before its
// middleware.
func (r *Router) Matched(cb func(*Route, *http.Request)) {
	r.root.matchedCallbacks = append(r.root.matchedCallbacks, cb)
}

// Current returns the route the request matched, read from the request
// context. nil means the request did not go through a registered route.
func (r *Router) Current(req *http.Request) *Route {
	return RouteFromContext(req.Context())
}

// CurrentRouteName returns the name of the current route, or empty.
func (r *Router) CurrentRouteName(req *http.Request) string {
	if rt := r.Current(req); rt != nil {
		return rt.RouteName()
	}
	return ""
}

// CurrentRouteAction returns the action string of the current route, or empty.
func (r *Router) CurrentRouteAction(req *http.Request) string {
	if rt := r.Current(req); rt != nil {
		return rt.GetActionName()
	}
	return ""
}

// Is reports whether the current route's name matches any of the patterns,
// with * as a glob. It is the alias for CurrentRouteNamed.
func (r *Router) Is(req *http.Request, patterns ...string) bool {
	return r.CurrentRouteNamed(req, patterns...)
}

// CurrentRouteNamed reports whether the current route's name matches any of
// the patterns.
func (r *Router) CurrentRouteNamed(req *http.Request, patterns ...string) bool {
	if rt := r.Current(req); rt == nil {
		return false
	} else {
		return rt.Named(patterns...)
	}
}

// Uses reports whether the current route's action matches any of the patterns.
func (r *Router) Uses(req *http.Request, patterns ...string) bool {
	action := r.CurrentRouteAction(req)
	if action == "" {
		return false
	}
	for _, p := range patterns {
		if wildcardMatch(p, action) {
			return true
		}
	}
	return false
}

// CurrentRouteUses reports whether the current route's action equals action.
func (r *Router) CurrentRouteUses(req *http.Request, action string) bool {
	return r.CurrentRouteAction(req) == action
}

// GetCurrentRoute returns the route the request matched. The request is the
// argument because a router shared by every goroutine cannot hold the one in
// flight.
func (r *Router) GetCurrentRoute(req *http.Request) *Route { return r.Current(req) }

// Input returns a parameter of the current route, or def.
func (r *Router) Input(req *http.Request, key string, def ...any) any {
	if rt := r.Current(req); rt != nil {
		return rt.Parameter(req, key, def...)
	}
	if len(def) > 0 {
		return def[0]
	}
	return nil
}

// Bind registers a binder for a parameter key: the binder resolves the raw
// path value into the record the handler expects.
//
// The binder receives the auth.Grant and MUST filter by auth.Tenant(g). A
// binder that looks a record up by id alone delivers another customer's row
// through the URL, which is the most direct form of the cross-tenant leak.
func (r *Router) Bind(key string, binder BindingFunc) {
	r.root.binders.Bind(key, binder)
}

// Model registers a model binding for a parameter key: the parameter resolves
// through the record type's own ResolveRouteBinding, under the Grant.
//
// There is no container to resolve a class out of, so this takes a zero value
// of the type instead -- which is the same thing a container would have
// handed the binder.
func (r *Router) Model(key string, class UrlRoutable, callback ...MissingModelFunc) {
	r.root.binders.Model(key, class, callback...)
}

// GetBindingCallback returns the binder registered for key, or nil.
func (r *Router) GetBindingCallback(key string) BindingFunc {
	return r.root.binders.GetBindingCallback(key)
}

// SubstituteBindings resolves the explicit binders for the route's parameters
// and installs what they returned as the value the handler reads.
//
// The Grant is a parameter and not something read out of the context, because
// a parameter cannot be forgotten. Every binding is scoped by auth.Tenant(g),
// and a Grant with no tenant is refused before anything is looked up.
func (r *Router) SubstituteBindings(ctx context.Context, g auth.Grant, rt *Route, req *http.Request) error {
	if rt == nil || req == nil || r.root.binders == nil {
		return nil
	}
	if err := requireTenant(g); err != nil {
		return err
	}
	for _, name := range rt.ParameterNames() {
		binder := r.GetBindingCallback(name)
		if binder == nil {
			continue
		}
		value := rt.OriginalParameter(req, name)
		if value == "" {
			continue
		}
		resolved, err := binder(ctx, g, value, rt)
		if err != nil {
			return err
		}
		if resolved != nil {
			rt.SetParameter(req, name, resolved)
		}
	}
	return nil
}

// SubstituteImplicitBindings runs the implicit binding -- every parameter
// with a registered model, resolved through it, scoped where the route asked
// for scoping -- or the callback SubstituteImplicitBindingsUsing installed in
// its place.
func (r *Router) SubstituteImplicitBindings(ctx context.Context, g auth.Grant, rt *Route, req *http.Request) error {
	if callback := r.root.implicitBindingCallback; callback != nil {
		return callback(rt, req)
	}
	return NewImplicitRouteBinding(r.root.binders).ResolveForRoute(ctx, g, rt, req)
}

// SubstituteImplicitBindingsUsing installs callback in place of the default
// implicit binding.
func (r *Router) SubstituteImplicitBindingsUsing(callback func(rt *Route, req *http.Request) error) *Router {
	r.root.implicitBindingCallback = callback
	return r
}

// RespondWithRoute dispatches the named route directly, bypassing the mux's
// match: it binds the request to the route and runs it. Used by a caller
// that has a route name and a request that did not arrive through the mux.
func (r *Router) RespondWithRoute(req *http.Request, name string, w http.ResponseWriter) error {
	rt := r.table.GetByName(name)
	if rt == nil {
		return errRouteNotFound(name)
	}
	req = rt.Bind(req)
	rt.ToResponse(w, req)
	return nil
}

// Dispatch runs the request through the mux, which is what ServeHTTP does. It
// is here for the surface that reads it as a method; the mux already holds the
// routes, and dispatch is its job.
func (r *Router) Dispatch(w http.ResponseWriter, req *http.Request) {
	r.ServeHTTP(w, req)
}

// DispatchToRoute is an alias for Dispatch: there is no separate global
// pipeline here to skip, since the mux is both the pipeline and the match.
func (r *Router) DispatchToRoute(w http.ResponseWriter, req *http.Request) {
	r.ServeHTTP(w, req)
}

// SetViewRenderer wires the view renderer the kernel built, so View routes can
// render. Without it, a View route answers 500.
func (r *Router) SetViewRenderer(vr ViewRenderer) {
	r.root.viewRenderer = vr
}

// GetViewRenderer returns the view renderer, or nil.
func (r *Router) GetViewRenderer() ViewRenderer { return r.root.viewRenderer }

// AddRoute creates a route and puts it in the table, which is what every verb
// method goes through.
func (r *Router) AddRoute(methods []string, uri string, action http.Handler, mws ...pipeline.Middleware[http.Handler]) *Route {
	if len(methods) == 0 {
		return r.Any(uri, action, mws...)
	}
	return r.Match(methods, uri, action, mws...)
}

// NewRoute builds a route bound to this router without registering it, for a
// caller assembling a table of its own.
func (r *Router) NewRoute(methods []string, uri string, action http.Handler) *Route {
	method := anyMethod
	if len(methods) > 0 {
		method = strings.ToUpper(methods[0])
	}
	route := &Route{
		Method:     method,
		Pattern:    joinPath(r.prefix, uri),
		Module:     r.module,
		namePrefix: r.name,
		prefix:     r.prefix,
		table:      r.table,
		handler:    action,
		mws:        append([]pipeline.Middleware[http.Handler]{}, r.mws...),
		wheres:     map[string]*regexp.Regexp{},
		defaults:   map[string]any{},
		action:     map[string]any{"uses": "Closure", "controller": "Closure"},
	}
	for _, m := range methods[1:] {
		route.siblings = append(route.siblings, &Route{Method: strings.ToUpper(m), Pattern: route.Pattern, handler: action})
	}
	return route.SetRouter(r)
}

// GatherRouteMiddleware resolves the route's full middleware list, excluded
// middleware removed.
func (r *Router) GatherRouteMiddleware(route *Route) []pipeline.Middleware[http.Handler] {
	return r.ResolveMiddleware(route.GatherMiddleware(), route.ExcludedMiddleware())
}

// ResolveMiddleware removes the excluded middleware from the list and drops
// duplicates. A middleware here is already the function to run, so there is
// no name or priority to resolve first.
func (r *Router) ResolveMiddleware(middleware []pipeline.Middleware[http.Handler], excluded ...[]pipeline.Middleware[http.Handler]) []pipeline.Middleware[http.Handler] {
	var drop []uintptr
	if len(excluded) > 0 {
		for _, mw := range excluded[0] {
			drop = append(drop, middlewarePointer(mw))
		}
	}

	out := make([]pipeline.Middleware[http.Handler], 0, len(middleware))
	for _, mw := range middleware {
		if containsPointer(drop, middlewarePointer(mw)) {
			continue
		}
		out = append(out, mw)
	}
	return UniqueMiddleware(out)
}

// UniqueMiddleware drops repeats while keeping the first of each, so a
// middleware a group and a route both attached runs once.
func UniqueMiddleware(middleware []pipeline.Middleware[http.Handler]) []pipeline.Middleware[http.Handler] {
	seen := make(map[uintptr]bool, len(middleware))
	out := make([]pipeline.Middleware[http.Handler], 0, len(middleware))
	for _, mw := range middleware {
		p := middlewarePointer(mw)
		if seen[p] {
			continue
		}
		seen[p] = true
		out = append(out, mw)
	}
	return out
}

// HasGroupStack reports whether any route group is currently open.
func (r *Router) HasGroupStack() bool { return len(r.groupStack) > 0 }

// GetGroupStack returns the attributes of every enclosing group, outermost
// first.
func (r *Router) GetGroupStack() []map[string]any {
	return append([]map[string]any(nil), r.groupStack...)
}

// GetLastGroupPrefix returns the prefix of the innermost open group, or empty.
func (r *Router) GetLastGroupPrefix() string {
	if !r.HasGroupStack() {
		return ""
	}
	last := r.groupStack[len(r.groupStack)-1]
	prefix, _ := last["prefix"].(string)
	return prefix
}

// MergeWithLastGroup merges attributes with the innermost open group's own,
// through Merge.
func (r *Router) MergeWithLastGroup(attributes map[string]any, prependExistingPrefix ...bool) map[string]any {
	var last map[string]any
	if r.HasGroupStack() {
		last = r.groupStack[len(r.groupStack)-1]
	}
	return Merge(attributes, last, prependExistingPrefix...)
}

// errRouteNotFound is the error RespondWithRoute returns for an unknown name.
type routeNotFoundError string

func (e routeNotFoundError) Error() string {
	return "routing: no route named " + string(e)
}

func errRouteNotFound(name string) error { return routeNotFoundError(name) }
