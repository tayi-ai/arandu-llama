package routing

import (
	"context"
	"net/http"
	"reflect"
	"regexp"
	"strings"

	"github.com/arandu-io/hesape/pipeline"
)

// routeContextKey is the context key under which the matched route is stored,
// so a middleware below the matcher -- and the route model binding in
// particular -- can read which route answered without a second match.
type routeContextKey struct{}

// overridesKey is the context key under which a per-request map of parameter
// overrides is stored. ServeHTTP installs a fresh map for every request, so
// SetParameter can mutate it in place rather than return a new request: a
// binding that replaces an id with a loaded record does so by writing here, and
// the handler reads it back through Parameter.
type overridesKey struct{}

type paramOverrides map[string]any

// RouteFromContext returns the route the request matched, or nil if it did
// not go through a registered route (a direct mux pattern, or a handler that
// never set it). It is the read side of the contract the route model binding
// depends on.
func RouteFromContext(ctx context.Context) *Route {
	if v, ok := ctx.Value(routeContextKey{}).(*Route); ok {
		return v
	}
	return nil
}

// ContextWithRoute returns ctx with route stored, for a handler or middleware
// that needs to install the matched route itself -- a custom dispatcher, or a
// test. ServeHTTP already does this for every registered route.
func ContextWithRoute(ctx context.Context, route *Route) context.Context {
	return context.WithValue(ctx, routeContextKey{}, route)
}

// ServeHTTP is the mux's entry into the route. It is the lazy half of the
// registration model: where the constraints, the matched listeners, the
// route-in-context and the middleware chain are applied here, at request time,
// so a fluent call after registration takes effect on a route already in the
// mux.
//
// Where constraints are part of matching, so they run first and answer 404 on
// failure before anything else; then the route is installed in the context
// and the matched listeners fire; then the middleware chain runs the handler.
func (rt *Route) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if rt == nil || rt.handler == nil {
		http.NotFound(w, req)
		return
	}

	// Where constraints are matching: a value that fails the regex is a request
	// this route does not answer, and 404 here keeps it from reaching the
	// handler -- which is the whole point of WhereNumber on /users/{id}.
	for _, name := range rt.whereOrder {
		re := rt.wheres[name]
		if re == nil {
			continue
		}
		val := req.PathValue(name)
		if val == "" {
			// An absent optional parameter has nothing to match; let it through.
			continue
		}
		if !re.MatchString(val) {
			http.NotFound(w, req)
			return
		}
	}

	// A deprecated route says so on everything it answers. It goes before the
	// middleware and the handler, both of which may write a status and freeze
	// the header block.
	rt.writeDeprecationHeaders(w)

	// Install the route and a fresh override map in the context. The override
	// map is what lets SubstituteBindings replace an id with a loaded record
	// without replacing the request.
	ctx := req.Context()
	ctx = ContextWithRoute(ctx, rt)
	ctx = context.WithValue(ctx, overridesKey{}, &paramOverrides{})

	// Matched listeners fire after matching, before middleware.
	if rt.router != nil {
		for _, cb := range rt.router.matchedCallbacks {
			cb(rt, req)
		}
	}

	chain := pipeline.Chain[http.Handler](rt.handler, rt.effectiveMiddleware()...)
	chain.ServeHTTP(w, req.WithContext(ctx))
}

// effectiveMiddleware returns the middleware chain that should run, with
// WithoutMiddleware applied. It is read at request time so a post-registration
// Middleware or WithoutMiddleware takes effect.
func (rt *Route) effectiveMiddleware() []pipeline.Middleware[http.Handler] {
	if len(rt.excluded) == 0 {
		return rt.mws
	}
	excluded := make([]uintptr, 0, len(rt.excluded))
	for _, mw := range rt.excluded {
		excluded = append(excluded, middlewarePointer(mw))
	}
	out := make([]pipeline.Middleware[http.Handler], 0, len(rt.mws))
	for _, mw := range rt.mws {
		if containsPointer(excluded, middlewarePointer(mw)) {
			continue
		}
		out = append(out, mw)
	}
	return out
}

// Methods returns the HTTP verbs the route answers. A Match registration
// produces one route per verb, linked through siblings; Methods gathers them.
func (rt *Route) Methods() []string {
	if rt == nil {
		return nil
	}
	out := []string{rt.Method}
	for _, s := range rt.siblings {
		out = append(out, s.Method)
	}
	return out
}

// URI returns the path pattern the route answers, prefixes of every enclosing
// group included.
func (rt *Route) URI() string {
	if rt == nil {
		return ""
	}
	return rt.Pattern
}

// GetURI is an alias for URI for the vocabulary that reads it as a getter.
func (rt *Route) GetURI() string { return rt.URI() }

// SetURI sets the pattern the route answers. It is here for the binding and
// the URL generator, which can rewrite a route's pattern after a group merge;
// the mux is not re-registered, so a pattern change after registration does not
// move what the mux dispatches.
func (rt *Route) SetURI(uri string) *Route {
	if rt != nil {
		rt.Pattern = uri
	}
	return rt
}

// Domain returns the host the route is declared for, or empty if none.
// http.ServeMux does not route by host, so this is for URL generation and for
// Matches.
func (rt *Route) Domain(domain string) *Route {
	if rt == nil {
		return nil
	}
	rt.domain = domain
	return rt
}

// GetDomain returns the host declared with Domain.
func (rt *Route) GetDomain() string {
	if rt == nil {
		return ""
	}
	return rt.domain
}

// GetPrefix returns the prefix the route was registered under.
func (rt *Route) GetPrefix() string {
	if rt == nil {
		return ""
	}
	return rt.prefix
}

// Prefix prepends a prefix to the route's pattern. The mux is not re-registered,
// so this is for the URL generator and the group merge, not for moving a
// route after it has been wired.
func (rt *Route) Prefix(prefix string) *Route {
	if rt == nil {
		return nil
	}
	rt.prefix = prefix
	return rt
}

// GetName returns the name given with Name, or empty. RouteName is the other
// name for it.
func (rt *Route) GetName() string { return rt.RouteName() }

// Named reports whether the route's name matches any of the patterns, using *
// as a glob wildcard. A route with no name matches nothing.
func (rt *Route) Named(patterns ...string) bool {
	if rt == nil || rt.name == "" {
		return false
	}
	for _, p := range patterns {
		if wildcardMatch(p, rt.name) {
			return true
		}
	}
	return false
}

// ParameterNames returns the names of the path parameters in the pattern, in
// order. Optional markers (?) and wildcard markers (...) are stripped.
func (rt *Route) ParameterNames() []string {
	if rt == nil {
		return nil
	}
	return compileParameterNames(rt.Pattern)
}

// HasParameters reports whether the route declares any path parameter. Unlike
// Parameter and its siblings, it does not need a request.
func (rt *Route) HasParameters() bool {
	return len(rt.ParameterNames()) > 0
}

// HasParameter reports whether name is one of the route's path parameters.
func (rt *Route) HasParameter(name string) bool {
	for _, p := range rt.ParameterNames() {
		if p == name {
			return true
		}
	}
	return false
}

// Parameter returns the value of a path parameter by name, with the per-request
// override a binding installed taking precedence over the raw path value. A
// missing parameter returns def.
func (rt *Route) Parameter(req *http.Request, name string, def ...any) any {
	if req != nil {
		if overrides, ok := req.Context().Value(overridesKey{}).(*paramOverrides); ok {
			if v, set := (*overrides)[name]; set {
				return v
			}
		}
		if v := req.PathValue(name); v != "" {
			return v
		}
	}
	if len(def) > 0 {
		return def[0]
	}
	return nil
}

// Parameters returns every path parameter for the request, overrides included.
// The map is a fresh copy, so a caller may mutate it.
func (rt *Route) Parameters(req *http.Request) map[string]any {
	out := map[string]any{}
	if rt == nil || req == nil {
		return out
	}
	for _, name := range rt.ParameterNames() {
		if v := req.PathValue(name); v != "" {
			out[name] = v
		}
	}
	if overrides, ok := req.Context().Value(overridesKey{}).(*paramOverrides); ok {
		for k, v := range *overrides {
			out[k] = v
		}
	}
	return out
}

// ParametersWithoutNulls returns Parameters with nil values dropped.
func (rt *Route) ParametersWithoutNulls(req *http.Request) map[string]any {
	all := rt.Parameters(req)
	for k, v := range all {
		if v == nil {
			delete(all, k)
		}
	}
	return all
}

// OriginalParameter returns the raw path value for name, before any override a
// binding installed. It is what a binding compares against to know whether it
// changed anything.
func (rt *Route) OriginalParameter(req *http.Request, name string, def ...string) string {
	if req != nil {
		if v := req.PathValue(name); v != "" {
			return v
		}
	}
	if len(def) > 0 {
		return def[0]
	}
	return ""
}

// SetParameter installs a per-request override for name, so a binding that
// resolves an id into a record hands the record to the handler through
// Parameter. It mutates the override map ServeHTTP placed in the context, so
// it does not need to return a new request.
func (rt *Route) SetParameter(req *http.Request, name string, value any) {
	if req == nil {
		return
	}
	if overrides, ok := req.Context().Value(overridesKey{}).(*paramOverrides); ok {
		(*overrides)[name] = value
	}
}

// ForgetParameter removes a per-request override, leaving the raw path value
// the next Parameter call reads.
func (rt *Route) ForgetParameter(req *http.Request, name string) {
	if req == nil {
		return
	}
	if overrides, ok := req.Context().Value(overridesKey{}).(*paramOverrides); ok {
		delete(*overrides, name)
	}
}

// Bind associates the route with a request: it installs the route in the
// request context and returns the bound parameters. The binding state lives
// in the request, and ServeHTTP already does this for every registered route,
// so Bind is for a caller that dispatches a route outside the mux
// (RespondWithRoute).
func (rt *Route) Bind(req *http.Request) *http.Request {
	if req == nil {
		return nil
	}
	ctx := ContextWithRoute(req.Context(), rt)
	ctx = context.WithValue(ctx, overridesKey{}, &paramOverrides{})
	return req.WithContext(ctx)
}

// ParentOfParameter returns the value of the parameter that precedes name in
// the parameter list, or empty if name is first or absent. It is what scoped
// binding uses to resolve a child against its parent.
func (rt *Route) ParentOfParameter(req *http.Request, name string) any {
	names := rt.ParameterNames()
	for i, n := range names {
		if n == name && i > 0 {
			return rt.Parameter(req, names[i-1])
		}
	}
	return nil
}

// Matches reports whether the route answers the request. Method is checked
// against the route's verb and its siblings; the path is checked against the
// pattern, with a parameter segment matching any non-slash. A fallback route
// only matches if nothing more specific did, which the caller is expected to
// have tried first.
func (rt *Route) Matches(req *http.Request, includeMethod bool) bool {
	if rt == nil || req == nil {
		return false
	}
	if includeMethod && !rt.methodMatches(req.Method) {
		return false
	}
	return rt.pathMatches(req.URL.Path)
}

func (rt *Route) methodMatches(method string) bool {
	if rt.Method == anyMethod {
		return true
	}
	if rt.Method == method {
		return true
	}
	for _, s := range rt.siblings {
		if s.Method == method {
			return true
		}
	}
	return false
}

// pathMatches reports whether path matches the route pattern, with a {name}
// segment matching any non-slash segment and a {name...} segment matching the
// rest of the path. A trailing-slash pattern matches its subtree, which is the
// fallback shape.
func (rt *Route) pathMatches(path string) bool {
	pattern := strings.TrimSuffix(rt.Pattern, "{$}")
	if pattern == "" {
		pattern = "/"
	}
	if strings.HasSuffix(pattern, "/") && !strings.Contains(pattern, "{") {
		// A subtree pattern matches itself and everything below it.
		return path == strings.TrimSuffix(pattern, "/") || strings.HasPrefix(path, pattern)
	}
	pSegs := strings.Split(strings.Trim(pattern, "/"), "/")
	rSegs := strings.Split(strings.Trim(path, "/"), "/")
	if len(pSegs) != len(rSegs) && !hasWildcard(pSegs) {
		return false
	}
	for i, seg := range pSegs {
		if strings.HasPrefix(seg, "{") && strings.HasSuffix(seg, "}") {
			inner := strings.Trim(seg, "{}")
			if strings.HasSuffix(inner, "...") {
				// Matches the rest of the path, however many segments remain.
				return true
			}
			if i >= len(rSegs) || rSegs[i] == "" {
				return false
			}
			continue
		}
		if i >= len(rSegs) || seg != rSegs[i] {
			return false
		}
	}
	return true
}

func hasWildcard(segs []string) bool {
	for _, s := range segs {
		if strings.HasSuffix(strings.Trim(s, "{}"), "...") {
			return true
		}
	}
	return false
}

// Run runs the route's handler without re-running middleware: middleware is
// the router's job, run is the action's. The route-in-context is installed so
// Parameter and the binding still work.
func (rt *Route) Run(w http.ResponseWriter, req *http.Request) {
	if rt == nil || rt.handler == nil {
		http.NotFound(w, req)
		return
	}
	// The deprecation headers are the route's, not the middleware's, so they
	// go out on this path too.
	rt.writeDeprecationHeaders(w)
	req = rt.Bind(req)
	rt.handler.ServeHTTP(w, req)
}

// ToResponse runs the route's full chain -- middleware and handler -- and is
// what RespondWithRoute calls when it dispatches a named route directly.
func (rt *Route) ToResponse(w http.ResponseWriter, req *http.Request) {
	rt.ServeHTTP(w, req)
}

// GetController returns the handler the route dispatches to: the http.Handler
// the registration was given, which is already the thing the mux calls. It
// returns nil for a route registered without one (which should not happen,
// but the nil is the honest answer).
func (rt *Route) GetController() http.Handler {
	if rt == nil {
		return nil
	}
	return rt.handler
}

// GetControllerClass returns the controller half of the action string -- the
// part before @ in "PostController@show" -- or "Closure" for a closure route.
func (rt *Route) GetControllerClass() string {
	uses, _ := rt.action["uses"].(string)
	if i := strings.Index(uses, "@"); i >= 0 {
		return uses[:i]
	}
	return uses
}

// GetActionName returns the action string the route was registered with, or
// "Closure". It is what CurrentRouteAction returns for the route.
func (rt *Route) GetActionName() string {
	if rt == nil {
		return "Closure"
	}
	if c, ok := rt.action["controller"].(string); ok && c != "" {
		return c
	}
	return "Closure"
}

// GetActionMethod returns the method half of the action string -- the part
// after @ in "PostController@show" -- or the whole string for a closure.
func (rt *Route) GetActionMethod() string {
	uses, _ := rt.action["uses"].(string)
	if i := strings.LastIndex(uses, "@"); i >= 0 {
		return uses[i+1:]
	}
	return uses
}

// GetAction returns the action descriptor, or one of its keys, as a map whose
// keys are uses, controller and anything else set through Uses.
func (rt *Route) GetAction(key ...string) any {
	if rt == nil || rt.action == nil {
		return nil
	}
	if len(key) == 0 {
		return rt.action
	}
	return rt.action[key[0]]
}

// SetAction sets the action descriptor and returns the route, for the group
// merge that carries domain and middleware in the same map.
func (rt *Route) SetAction(action map[string]any) *Route {
	if rt != nil {
		rt.action = action
		if d, ok := action["domain"].(string); ok {
			rt.domain = d
		}
	}
	return rt
}

// Uses sets the action string -- "PostController@show" or "Closure" -- and
// returns the route. It is the fluent form of declaring what a route answers,
// and it is what GetActionName and GetActionMethod read.
func (rt *Route) Uses(action string) *Route {
	if rt == nil {
		return nil
	}
	rt.action["uses"] = action
	rt.action["controller"] = action
	return rt
}

// Action is an alias for Uses that takes the action string, for whichever
// call site reads better with it.
func (rt *Route) Action(action string) *Route { return rt.Uses(action) }

// Middleware appends middleware to the route, to run after the group's
// middleware and after anything already attached. Because the chain is read
// at request time, a Middleware call after registration -- which is the shape
// every fluent call returns -- takes effect on a route already in the mux.
//
// Per-route middleware set at registration, through the varargs of Get and its
// siblings, and through this method, is one form: both reach the same slice.
func (rt *Route) Middleware(mws ...pipeline.Middleware[http.Handler]) *Route {
	if rt == nil {
		return nil
	}
	rt.mws = append(rt.mws, mws...)
	return rt
}

// GatherMiddleware returns the full middleware list for the route, group
// middleware first. It is what the dispatcher reads.
func (rt *Route) GatherMiddleware() []pipeline.Middleware[http.Handler] {
	if rt == nil {
		return nil
	}
	return rt.effectiveMiddleware()
}

// WithoutMiddleware records middleware to remove from the route's chain at
// request time. A middleware removed here is one a group attached that this
// route specifically does not want, and the removal is by identity.
func (rt *Route) WithoutMiddleware(mws ...pipeline.Middleware[http.Handler]) *Route {
	if rt == nil {
		return nil
	}
	rt.excluded = append(rt.excluded, mws...)
	return rt
}

// ExcludedMiddleware returns the middleware WithoutMiddleware removed.
func (rt *Route) ExcludedMiddleware() []pipeline.Middleware[http.Handler] {
	if rt == nil {
		return nil
	}
	return append([]pipeline.Middleware[http.Handler]{}, rt.excluded...)
}

// WithTrashed allows soft-deleted records through implicit model binding on
// this route. The implicit binding (hesape/routing, in the binding files)
// reads it; this is the declaration.
func (rt *Route) WithTrashed(with ...bool) *Route {
	if rt == nil {
		return nil
	}
	rt.withTrashed = len(with) == 0 || with[0]
	return rt
}

// AllowsTrashedBindings reports whether WithTrashed was set.
func (rt *Route) AllowsTrashedBindings() bool {
	return rt != nil && rt.withTrashed
}

// ScopeBindings marks the route as enforcing scoped implicit bindings -- a
// child resolved against its parent parameter. The implicit binding reads it.
func (rt *Route) ScopeBindings() *Route {
	if rt == nil {
		return nil
	}
	t := true
	rt.scopedBindings = &t
	return rt
}

// WithoutScopedBindings marks the route as not enforcing scoped bindings.
func (rt *Route) WithoutScopedBindings() *Route {
	if rt == nil {
		return nil
	}
	f := false
	rt.scopedBindings = &f
	return rt
}

// EnforcesScopedBindings reports whether ScopeBindings was set.
func (rt *Route) EnforcesScopedBindings() bool {
	return rt != nil && rt.scopedBindings != nil && *rt.scopedBindings
}

// PreventsScopedBindings reports whether WithoutScopedBindings was set, which
// is distinct from leaving it unset.
func (rt *Route) PreventsScopedBindings() bool {
	return rt != nil && rt.scopedBindings != nil && !*rt.scopedBindings
}

// Defaults sets a default value for a parameter, and returns the route. The
// redirect and view routes use it to carry a destination or a view name in
// the same place a bound parameter would live.
func (rt *Route) Defaults(key string, value any) *Route {
	if rt == nil {
		return nil
	}
	rt.defaults[key] = value
	return rt
}

// SetDefaults replaces the defaults map.
func (rt *Route) SetDefaults(defaults map[string]any) *Route {
	if rt == nil {
		return nil
	}
	rt.defaults = defaults
	return rt
}

// GetDefaults returns the defaults map, or one key of it.
func (rt *Route) GetDefaults(key ...string) any {
	if rt == nil {
		return nil
	}
	if len(key) == 0 {
		return rt.defaults
	}
	return rt.defaults[key[0]]
}

// Missing sets the handler that answers when an implicit model binding
// resolves nothing. The implicit binding (hesape/routing, in the binding
// files) calls it; this is the declaration.
func (rt *Route) Missing(h http.Handler) *Route {
	if rt == nil {
		return nil
	}
	rt.missingHandler = h
	return rt
}

// GetMissing returns the handler Missing set, or nil.
func (rt *Route) GetMissing() http.Handler {
	if rt == nil {
		return nil
	}
	return rt.missingHandler
}

// SetFallback marks the route as a fallback. Fallback does this; SetFallback
// is for the deserializer.
func (rt *Route) SetFallback(fallback bool) *Route {
	if rt != nil {
		rt.fallback = fallback
	}
	return rt
}

// IsFallback reports whether the route is a fallback route.
func (rt *Route) IsFallback() bool {
	return rt != nil && rt.fallback
}

// BindingFields returns the {param:field} qualifiers parsed off the URI, keyed
// by parameter. The route model binding uses them to resolve a model by a
// column other than id.
func (rt *Route) BindingFields() map[string]string {
	if rt == nil {
		return nil
	}
	return rt.bindingFields
}

// SetBindingFields sets the binding fields, for the group merge and the
// deserializer.
func (rt *Route) SetBindingFields(fields map[string]string) *Route {
	if rt != nil {
		rt.bindingFields = fields
	}
	return rt
}

// BindingFieldFor returns the binding field for a parameter, or empty.
func (rt *Route) BindingFieldFor(param string) string {
	if rt == nil || rt.bindingFields == nil {
		return ""
	}
	return rt.bindingFields[param]
}

// Where sets a regular-expression constraint on a path parameter. A request
// whose value for that parameter fails the regex is answered 404 before the
// handler, which is what stops /users/{id} from reaching the database for
// /users/abc.
//
// Go's ServeMux does not take a regex in a pattern, so the constraint is
// enforced here, at dispatch, rather than in the mux's matcher. A failing
// value does not fall through to a sibling route -- ServeMux already picked
// this one -- but the goal the constraint exists for, keeping a bad value out
// of the database, holds.
func (rt *Route) Where(name, pattern string) *Route {
	if rt == nil {
		return nil
	}
	return rt.whereMap(map[string]string{name: pattern})
}

// WhereMap sets several where constraints at once.
func (rt *Route) WhereMap(wheres map[string]string) *Route {
	if rt == nil {
		return nil
	}
	return rt.whereMap(wheres)
}

// SetWheres replaces the where constraints, given as a map.
func (rt *Route) SetWheres(wheres map[string]string) *Route {
	if rt == nil {
		return nil
	}
	rt.wheres = map[string]*regexp.Regexp{}
	rt.whereOrder = rt.whereOrder[:0]
	return rt.whereMap(wheres)
}

// GetWheres returns the where constraints as the string patterns the caller
// set.
func (rt *Route) GetWheres() map[string]string {
	if rt == nil {
		return nil
	}
	out := map[string]string{}
	for name, re := range rt.wheres {
		out[name] = re.String()
	}
	return out
}

func (rt *Route) whereMap(wheres map[string]string) *Route {
	for name, expr := range wheres {
		re, err := regexp.Compile(expr)
		if err != nil {
			continue
		}
		if _, ok := rt.wheres[name]; !ok {
			rt.whereOrder = append(rt.whereOrder, name)
		}
		rt.wheres[name] = re
	}
	return rt
}

// WhereNumber constrains the given parameters to digits.
func (rt *Route) WhereNumber(params ...string) *Route {
	return rt.assignExpr(params, `[0-9]+`)
}

// WhereAlpha constrains the given parameters to ASCII letters.
func (rt *Route) WhereAlpha(params ...string) *Route {
	return rt.assignExpr(params, `[a-zA-Z]+`)
}

// WhereAlphaNumeric constrains the given parameters to ASCII letters and digits.
func (rt *Route) WhereAlphaNumeric(params ...string) *Route {
	return rt.assignExpr(params, `[a-zA-Z0-9]+`)
}

// WhereUlid constrains the given parameters to the ULID alphabet.
func (rt *Route) WhereUlid(params ...string) *Route {
	return rt.assignExpr(params, `[0-7][0-9a-hjkmnp-tv-zA-HJKMNP-TV-Z]{25}`)
}

// WhereUuid constrains the given parameters to the UUID format.
func (rt *Route) WhereUuid(params ...string) *Route {
	return rt.assignExpr(params, `[\da-fA-F]{8}-[\da-fA-F]{4}-[\da-fA-F]{4}-[\da-fA-F]{4}-[\da-fA-F]{12}`)
}

// WhereIn constrains the given parameters to one of the listed values.
func (rt *Route) WhereIn(param string, values ...string) *Route {
	if rt == nil {
		return nil
	}
	return rt.Where(param, strings.Join(values, "|"))
}

func (rt *Route) assignExpr(params []string, expr string) *Route {
	if rt == nil {
		return nil
	}
	m := map[string]string{}
	for _, p := range params {
		m[p] = expr
	}
	return rt.whereMap(m)
}

// compileParameterNames extracts {name} segments from a pattern, stripping
// optional (?) and wildcard (...) markers. It is what ParameterNames and
// HasParameter read.
func compileParameterNames(pattern string) []string {
	var out []string
	for _, seg := range strings.Split(pattern, "/") {
		if !strings.HasPrefix(seg, "{") || !strings.HasSuffix(seg, "}") {
			continue
		}
		inner := strings.Trim(seg, "{}")
		inner = strings.TrimSuffix(inner, "...")
		inner = strings.TrimSuffix(inner, "?")
		if inner == "" {
			continue
		}
		out = append(out, inner)
	}
	return out
}

// middlewarePointer returns a stable identity for a middleware func, so
// WithoutMiddleware can compare by it. Two distinct funcs compare unequal even
// when they close over the same value, which is the right answer: removing a
// middleware means removing that func, not any func like it.
func middlewarePointer(mw pipeline.Middleware[http.Handler]) uintptr {
	return reflect.ValueOf(mw).Pointer()
}

func containsPointer(haystack []uintptr, needle uintptr) bool {
	for _, p := range haystack {
		if p == needle {
			return true
		}
	}
	return false
}

// wildcardMatch reports whether s matches a pattern with * as a glob. It is
// here rather than in a support package because routing is the only place a
// route name is compared to a pattern.
func wildcardMatch(pattern, s string) bool {
	if pattern == s {
		return true
	}
	parts := strings.Split(pattern, "*")
	if len(parts) == 1 {
		return false
	}
	if !strings.HasPrefix(s, parts[0]) {
		return false
	}
	s = s[len(parts[0]):]
	for i := 1; i < len(parts)-1; i++ {
		idx := strings.Index(s, parts[i])
		if idx < 0 {
			return false
		}
		s = s[idx+len(parts[i]):]
	}
	return strings.HasSuffix(s, parts[len(parts)-1])
}

// compileWhere compiles a where pattern once, for the global patterns applied
// at registration.
func compileWhere(expr string) (*regexp.Regexp, error) { return regexp.Compile(expr) }
