package routing

import (
	"crypto/hmac"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"errors"
	"net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"time"
)

// UrlGenerator builds every address an application needs: the route named
// "invoices.show", the asset at "/css/app.css", the absolute address of the
// page being viewed.
//
// It is the single answer to "where is this thing". A view that concatenates
// "/invoices/" with an id compiles and keeps compiling after the route moves;
// a call through UrlGenerator does not, and the developer finds out at build
// time rather than from a broken link.
//
// Parameters are a map here. The positional shape -- route("invoices.show",
// 42), matched against the pattern in order -- is Routes.Route, because a Go
// map has no order to match against.
type UrlGenerator struct {
	routes    *Routes
	request   *http.Request
	assetRoot string

	forcedRoot   string
	forceScheme  string
	cachedRoot   string
	cachedScheme string

	rootNamespace             string
	sessionResolver           func() SessionStore
	keyResolver               func() []string
	missingNamedRouteResolver func(name string, parameters map[string]any, absolute bool) (string, error)
	formatHostUsing           func(root string, route *Route) string
	formatPathUsing           func(path string, route *Route) string

	routeGenerator *RouteUrlGenerator
}

// SessionStore is what UrlGenerator reads the previous URL and the intended URL from.
//
// Declared here rather than importing hesape/session, so that a package that only
// generates URLs does not depend on the session implementation.
type SessionStore interface {
	Get(key string, def ...any) any
	Put(key string, value any)
	Pull(key string, def ...any) any
	PreviousURL() string
}

// NewUrlGenerator returns a generator backed by routes.
//
// The request may be nil for a generator that only builds relative URLs: a
// CLI command that prints a list of links, or a test that has no incoming
// request to read the scheme and host from.
func NewUrlGenerator(routes *Routes, request *http.Request, assetRoot ...string) *UrlGenerator {
	g := &UrlGenerator{routes: routes}
	if len(assetRoot) > 0 {
		g.assetRoot = assetRoot[0]
	}
	g.SetRequest(request)
	return g
}

// Full returns the full URL of the current request, query string included.
func (g *UrlGenerator) Full() string {
	if g.request == nil {
		return ""
	}
	full := g.requestURL()
	if q := g.request.URL.RawQuery; q != "" {
		full += "?" + q
	}
	return full
}

// Current returns the URL of the current request without its query string.
func (g *UrlGenerator) Current() string {
	if g.request == nil {
		return ""
	}
	return g.To(g.request.URL.Path, nil, nil)
}

// Previous reads the Referer header first, then the session, then the
// fallback, then "/".
func (g *UrlGenerator) Previous(fallback ...string) string {
	referrer := ""
	if g.request != nil {
		referrer = g.request.Header.Get("Referer")
	}

	u := ""
	if referrer != "" {
		u = g.To(referrer, nil, nil)
	} else {
		u = g.getPreviousUrlFromSession()
	}

	if u != "" {
		return u
	}
	if len(fallback) > 0 && fallback[0] != "" {
		return g.To(fallback[0], nil, nil)
	}
	return g.To("/", nil, nil)
}

// PreviousPath returns the path part of Previous, with the application root
// and any query string removed.
func (g *UrlGenerator) PreviousPath(fallback ...string) string {
	previous := g.Previous(fallback...)
	if i := strings.IndexByte(previous, '?'); i >= 0 {
		previous = previous[:i]
	}
	previous = strings.TrimSuffix(previous, "/")
	previous = strings.TrimPrefix(previous, g.To("/", nil, nil))
	if previous == "" {
		return "/"
	}
	return previous
}

// getPreviousUrlFromSession reads the previous URL from the session store.
func (g *UrlGenerator) getPreviousUrlFromSession() string {
	s := g.getSession()
	if s == nil {
		return ""
	}
	return s.PreviousURL()
}

// To returns the absolute URL for a path.
//
//	To("invoices/42", nil, nil)  // "https://example.com/invoices/42"
//
// A path that is already a valid URL is returned unchanged. Anything in extra
// is appended as further path segments.
func (g *UrlGenerator) To(path string, extra []any, secure *bool) string {
	// A path that is already a URL is returned as is, which is convenient
	// because the caller does not always know whether it is one.
	if g.IsValidUrl(path) {
		return path
	}

	segments := make([]string, 0, len(extra))
	for _, p := range extra {
		segments = append(segments, rawURLEncode(formatParameterValue(p, "")))
	}
	tail := strings.Join(segments, "/")

	root := g.FormatRoot(g.FormatScheme(secure))

	p, query := extractQueryString(path)

	return g.Format(root, "/"+strings.Trim(p+"/"+tail, "/")) + query
}

// Query returns the absolute URL for a path with the given query parameters
// merged onto any the path already carried.
func (g *UrlGenerator) Query(path string, query map[string]any, extra []any, secure *bool) string {
	p, existing := extractQueryString(path)

	merged := url.Values{}
	if existing != "" {
		if parsed, err := url.ParseQuery(strings.TrimPrefix(existing, "?")); err == nil {
			for k, v := range parsed {
				merged[k] = v
			}
		}
	}
	for k, v := range query {
		merged.Set(k, formatParameterValue(v, ""))
	}

	return strings.TrimRight(g.To(p+"?"+merged.Encode(), extra, secure), "?")
}

// Secure returns the absolute https URL for a path.
func (g *UrlGenerator) Secure(path string, parameters ...any) string {
	t := true
	return g.To(path, parameters, &t)
}

// Asset returns the URL of an application asset.
func (g *UrlGenerator) Asset(path string, secure *bool) string {
	if g.IsValidUrl(path) {
		return path
	}
	root := g.assetRoot
	if root == "" {
		root = g.FormatRoot(g.FormatScheme(secure))
	}
	return strings.TrimSuffix(g.removeIndex(root), "/") + "/" + strings.Trim(path, "/")
}

// SecureAsset returns the https URL of an application asset.
func (g *UrlGenerator) SecureAsset(path string) string {
	t := true
	return g.Asset(path, &t)
}

// AssetFrom returns the URL of an asset served from a custom root, such as a
// CDN.
func (g *UrlGenerator) AssetFrom(root, path string, secure *bool) string {
	r := g.FormatRoot(g.FormatScheme(secure), root)
	return g.removeIndex(r) + "/" + strings.Trim(path, "/")
}

// removeIndex strips a legacy front-controller filename from an asset root,
// which a root copied from an old deployment might still carry.
func (g *UrlGenerator) removeIndex(root string) string {
	const i = "index.php"
	if strings.Contains(root, i) {
		return strings.ReplaceAll(root, "/"+i, "")
	}
	return root
}

// FormatScheme returns the scheme with its "://" suffix.
func (g *UrlGenerator) FormatScheme(secure *bool) string {
	if secure != nil {
		if *secure {
			return "https://"
		}
		return "http://"
	}
	if g.cachedScheme == "" {
		switch {
		case g.forceScheme != "":
			g.cachedScheme = g.forceScheme
		case g.request != nil && g.request.TLS != nil:
			g.cachedScheme = "https://"
		default:
			g.cachedScheme = "http://"
		}
	}
	return g.cachedScheme
}

// SignedRoute returns the URL of a named route carrying an HMAC over the
// address it was issued for.
//
// The signature covers the whole address, expiry included, so appending
// anything to a signed link invalidates it -- a route behind
// middleware.ValidateSignature cannot be handed a value its author did not
// sign. A zero expiration means the link does not run out.
//
// It returns an error for reserved parameter names, an unknown route, or no
// signing key.
func (g *UrlGenerator) SignedRoute(name string, parameters map[string]any, expiration time.Time, absolute bool) (string, error) {
	if err := g.ensureSignedRouteParametersAreNotReserved(parameters); err != nil {
		return "", err
	}

	params := make(map[string]any, len(parameters)+2)
	for k, v := range parameters {
		params[k] = v
	}
	if !expiration.IsZero() {
		params["expires"] = strconv.FormatInt(g.availableAt(expiration), 10)
	}

	keys := g.resolveKeys()
	if len(keys) == 0 {
		return "", errors.New("routing: no signing key. Wire one with SetKeyResolver before building a signed URL")
	}

	unsigned, err := g.Route(name, params, absolute)
	if err != nil {
		return "", err
	}

	mac := hmac.New(sha256.New, []byte(keys[0]))
	mac.Write([]byte(unsigned))
	params["signature"] = hex.EncodeToString(mac.Sum(nil))

	return g.Route(name, params, absolute)
}

// ensureSignedRouteParametersAreNotReserved refuses "signature" and "expires"
// as caller-supplied parameter names, since SignedRoute sets them itself.
func (g *UrlGenerator) ensureSignedRouteParametersAreNotReserved(parameters map[string]any) error {
	if _, taken := parameters["signature"]; taken {
		return errors.New(`routing: "signature" is a reserved parameter when generating signed routes. Rename the route parameter`)
	}
	if _, taken := parameters["expires"]; taken {
		return errors.New(`routing: "expires" is a reserved parameter when generating signed routes. Rename the route parameter`)
	}
	return nil
}

// TemporarySignedRoute is SignedRoute with the expiration and parameters
// reordered.
func (g *UrlGenerator) TemporarySignedRoute(name string, expiration time.Time, parameters map[string]any, absolute bool) (string, error) {
	return g.SignedRoute(name, parameters, expiration, absolute)
}

// availableAt returns the unix timestamp a moment becomes due.
func (g *UrlGenerator) availableAt(at time.Time) int64 { return at.Unix() }

// HasValidSignature reports whether the request carries a signature that is
// both correct and unexpired.
func (g *UrlGenerator) HasValidSignature(request *http.Request, absolute bool, ignoreQuery ...string) bool {
	return g.HasCorrectSignature(request, absolute, ignoreQuery...) && g.SignatureHasNotExpired(request)
}

// HasValidRelativeSignature is HasValidSignature checked against the
// relative address.
func (g *UrlGenerator) HasValidRelativeSignature(request *http.Request, ignoreQuery ...string) bool {
	return g.HasValidSignature(request, false, ignoreQuery...)
}

// HasCorrectSignature reports whether the signature on the request matches
// the address the request arrived on.
//
// Every key the resolver returns is tried, so a key being rotated out still
// opens the links it signed.
func (g *UrlGenerator) HasCorrectSignature(request *http.Request, absolute bool, ignoreQuery ...string) bool {
	if request == nil || request.URL == nil {
		return false
	}

	base := "/" + strings.TrimPrefix(request.URL.Path, "/")
	if absolute {
		base = g.requestURLOf(request)
	}

	ignored := make(map[string]bool, len(ignoreQuery)+1)
	ignored["signature"] = true
	for _, name := range ignoreQuery {
		ignored[name] = true
	}

	kept := make([]string, 0, 4)
	for _, pair := range strings.Split(request.URL.RawQuery, "&") {
		if pair == "" {
			continue
		}
		name := pair
		if i := strings.IndexByte(pair, '='); i >= 0 {
			name = pair[:i]
		}
		if ignored[name] {
			continue
		}
		kept = append(kept, pair)
	}

	original := strings.TrimSuffix(base+"?"+strings.Join(kept, "&"), "?")
	signature := request.URL.Query().Get("signature")

	for _, key := range g.resolveKeys() {
		mac := hmac.New(sha256.New, []byte(key))
		mac.Write([]byte(original))
		expected := hex.EncodeToString(mac.Sum(nil))
		if subtle.ConstantTimeCompare([]byte(expected), []byte(signature)) == 1 {
			return true
		}
	}
	return false
}

// SignatureHasNotExpired reports whether the request's signature has not
// expired. A link with no expiry never runs out; one whose expiry is in the
// past has.
func (g *UrlGenerator) SignatureHasNotExpired(request *http.Request) bool {
	if request == nil || request.URL == nil {
		return true
	}
	expires := request.URL.Query().Get("expires")
	if expires == "" {
		return true
	}
	at, err := strconv.ParseInt(expires, 10, 64)
	if err != nil {
		// An expiry that is not a timestamp is not an expiry that has passed,
		// but it is not one that can be trusted either.
		return false
	}
	return time.Now().Unix() <= at
}

// Route returns the URL of a named route.
func (g *UrlGenerator) Route(name string, parameters map[string]any, absolute bool) (string, error) {
	if route := g.routes.GetByName(name); route != nil {
		return g.ToRoute(route, parameters, absolute)
	}

	if g.missingNamedRouteResolver != nil {
		if u, err := g.missingNamedRouteResolver(name, parameters, absolute); err == nil && u != "" {
			return u, nil
		}
	}

	return "", &RouteNotFoundError{Name: name}
}

// ToRoute builds the URL of a route already in hand.
func (g *UrlGenerator) ToRoute(route *Route, parameters map[string]any, absolute bool) (string, error) {
	return g.routeUrl().To(route, parameters, absolute)
}

// Action returns the URL of a controller action, written as
// "InvoiceController@show".
func (g *UrlGenerator) Action(action string, parameters map[string]any, absolute bool) (string, error) {
	action = g.formatAction(action)
	route := g.routes.GetByAction(action)
	if route == nil {
		return "", &RouteNotFoundError{Name: action}
	}
	return g.ToRoute(route, parameters, absolute)
}

// formatAction qualifies action with the root controller namespace, unless
// it is already absolute.
func (g *UrlGenerator) formatAction(action string) string {
	if g.rootNamespace != "" && !strings.HasPrefix(action, `\`) {
		return g.rootNamespace + `\` + action
	}
	return strings.Trim(action, `\`)
}

// FormatParameters replaces every value that knows its own route key -- a
// record passed where an id was expected -- with that key.
//
// This takes the map shape a route's parameters have. The list shape is
// applied inline by To, over the same conversion.
func (g *UrlGenerator) FormatParameters(parameters map[string]any) map[string]any {
	out := make(map[string]any, len(parameters))
	for key, parameter := range parameters {
		if routable, ok := parameter.(UrlRoutable); ok {
			out[key] = routable.GetRouteKey()
			continue
		}
		out[key] = parameter
	}
	return out
}

// extractQueryString splits a path from its query string.
func extractQueryString(path string) (string, string) {
	if i := strings.IndexByte(path, '?'); i >= 0 {
		return path[:i], path[i:]
	}
	return path, ""
}

// FormatRoot returns the base URL -- scheme and host -- for the request, or
// for the root given.
func (g *UrlGenerator) FormatRoot(scheme string, root ...string) string {
	r := ""
	if len(root) > 0 && root[0] != "" {
		r = root[0]
	} else {
		if g.cachedRoot == "" {
			if g.forcedRoot != "" {
				g.cachedRoot = g.forcedRoot
			} else if g.request != nil {
				g.cachedRoot = g.requestURLRoot()
			}
		}
		r = g.cachedRoot
	}

	switch {
	case strings.HasPrefix(r, "http://"):
		return scheme + strings.TrimPrefix(r, "http://")
	case strings.HasPrefix(r, "https://"):
		return scheme + strings.TrimPrefix(r, "https://")
	default:
		return r
	}
}

// Format joins a root and a path into one URL, through the host and path
// formatters when they are set.
func (g *UrlGenerator) Format(root, path string, route ...*Route) string {
	path = "/" + strings.Trim(path, "/")

	var rt *Route
	if len(route) > 0 {
		rt = route[0]
	}

	if g.formatHostUsing != nil {
		root = g.formatHostUsing(root, rt)
	}
	if g.formatPathUsing != nil {
		path = g.formatPathUsing(path, rt)
	}

	return strings.Trim(root+path, "/")
}

// IsValidUrl reports whether path is already a complete URL rather than one
// this generator should build.
func (g *UrlGenerator) IsValidUrl(path string) bool {
	for _, prefix := range []string{"#", "//", "http://", "https://", "mailto:", "tel:", "sms:"} {
		if strings.HasPrefix(path, prefix) {
			return true
		}
	}
	u, err := url.Parse(path)
	return err == nil && u.Scheme != "" && u.Host != ""
}

// routeUrl returns the route URL generator, building it on first use.
func (g *UrlGenerator) routeUrl() *RouteUrlGenerator {
	if g.routeGenerator == nil {
		g.routeGenerator = NewRouteUrlGenerator(g, g.request)
	}
	return g.routeGenerator
}

// Defaults sets default values for named route parameters -- the locale
// segment every URL carries, typically.
func (g *UrlGenerator) Defaults(defaults map[string]any) {
	g.routeUrl().Defaults(defaults)
}

// GetDefaultParameters returns the default route parameters Defaults set.
func (g *UrlGenerator) GetDefaultParameters() map[string]any {
	return g.routeUrl().DefaultParameters
}

// ForceScheme forces every generated URL to a scheme. An empty scheme clears
// the force.
func (g *UrlGenerator) ForceScheme(scheme string) {
	g.cachedScheme = ""
	if scheme == "" {
		g.forceScheme = ""
		return
	}
	g.forceScheme = scheme + "://"
}

// ForceHttps is ForceScheme("https").
func (g *UrlGenerator) ForceHttps(force ...bool) {
	if len(force) == 0 || force[0] {
		g.ForceScheme("https")
	}
}

// UseOrigin sets the origin every generated URL is built from.
func (g *UrlGenerator) UseOrigin(root string) {
	g.ForceRootUrl(root)
}

// ForceRootUrl sets the origin every generated URL is built from.
//
// Deprecated: use UseOrigin.
func (g *UrlGenerator) ForceRootUrl(root string) {
	if root == "" {
		g.forcedRoot = ""
	} else {
		g.forcedRoot = strings.TrimRight(root, "/")
	}
	g.cachedRoot = ""
}

// UseAssetOrigin sets the origin every generated asset URL is built from --
// a CDN host, typically.
func (g *UrlGenerator) UseAssetOrigin(root string) {
	if root == "" {
		g.assetRoot = ""
		return
	}
	g.assetRoot = strings.TrimRight(root, "/")
}

// FormatHostUsing sets the callback Format uses to rewrite the host of a
// generated URL.
func (g *UrlGenerator) FormatHostUsing(callback func(root string, route *Route) string) *UrlGenerator {
	g.formatHostUsing = callback
	return g
}

// FormatPathUsing sets the callback Format uses to rewrite the path of a
// generated URL.
func (g *UrlGenerator) FormatPathUsing(callback func(path string, route *Route) string) *UrlGenerator {
	g.formatPathUsing = callback
	return g
}

// PathFormatter returns the path formatter in use, or one that changes
// nothing.
func (g *UrlGenerator) PathFormatter() func(path string, route *Route) string {
	if g.formatPathUsing != nil {
		return g.formatPathUsing
	}
	return func(path string, _ *Route) string { return path }
}

// GetRequest returns the current request, or nil.
func (g *UrlGenerator) GetRequest() *http.Request { return g.request }

// SetRequest replaces the request, clears the cached scheme and root, and
// carries the route defaults over to the new route URL generator.
func (g *UrlGenerator) SetRequest(request *http.Request) {
	g.request = request
	g.cachedRoot = ""
	g.cachedScheme = ""

	var defaults map[string]any
	if g.routeGenerator != nil {
		defaults = g.routeGenerator.DefaultParameters
	}
	g.routeGenerator = nil
	if len(defaults) > 0 {
		g.Defaults(defaults)
	}
}

// SetRoutes replaces the route table URLs are generated against.
func (g *UrlGenerator) SetRoutes(routes *Routes) *UrlGenerator {
	g.routes = routes
	return g
}

// getSession resolves the session store, or nil.
func (g *UrlGenerator) getSession() SessionStore {
	if g.sessionResolver == nil {
		return nil
	}
	return g.sessionResolver()
}

// SetSessionResolver sets the callback that resolves the session store.
func (g *UrlGenerator) SetSessionResolver(sessionResolver func() SessionStore) *UrlGenerator {
	g.sessionResolver = sessionResolver
	return g
}

// SetKeyResolver sets the callback that resolves the signing keys. It
// returns every key a signature may have been made with, newest first:
// signing uses the first, verifying tries them all, which is what makes a
// key rotation something other than an outage.
func (g *UrlGenerator) SetKeyResolver(keyResolver func() []string) *UrlGenerator {
	g.keyResolver = keyResolver
	return g
}

// WithKeyResolver returns a copy of the generator signing with a different
// key, which is how a link is issued for a tenant whose key is not the
// application's.
func (g *UrlGenerator) WithKeyResolver(keyResolver func() []string) *UrlGenerator {
	clone := *g
	clone.routeGenerator = nil
	return clone.SetKeyResolver(keyResolver)
}

// resolveKeys returns the signing keys, or none.
func (g *UrlGenerator) resolveKeys() []string {
	if g.keyResolver == nil {
		return nil
	}
	keys := make([]string, 0, 2)
	for _, key := range g.keyResolver() {
		if key != "" {
			keys = append(keys, key)
		}
	}
	return keys
}

// ResolveMissingNamedRoutesUsing sets the callback that gets a chance to
// answer a name this table does not know -- a route registered by another
// service, in front of the same domain.
func (g *UrlGenerator) ResolveMissingNamedRoutesUsing(resolver func(name string, parameters map[string]any, absolute bool) (string, error)) *UrlGenerator {
	g.missingNamedRouteResolver = resolver
	return g
}

// GetRootControllerNamespace returns the root controller namespace.
func (g *UrlGenerator) GetRootControllerNamespace() string { return g.rootNamespace }

// SetRootControllerNamespace sets the root controller namespace formatAction
// qualifies an action name with.
func (g *UrlGenerator) SetRootControllerNamespace(rootNamespace string) *UrlGenerator {
	g.rootNamespace = rootNamespace
	return g
}

// requestURLRoot returns the request's scheme and host, no path.
func (g *UrlGenerator) requestURLRoot() string {
	if g.request == nil {
		return ""
	}
	return g.FormatScheme(nil) + hostOf(g.request)
}

// requestURL returns the request's scheme, host and path, no query.
func (g *UrlGenerator) requestURL() string { return g.requestURLOf(g.request) }

func (g *UrlGenerator) requestURLOf(request *http.Request) string {
	if request == nil || request.URL == nil {
		return ""
	}
	scheme := "http://"
	if request.TLS != nil {
		scheme = "https://"
	}
	if g.forceScheme != "" {
		scheme = g.forceScheme
	}
	return strings.TrimSuffix(scheme+hostOf(request)+request.URL.Path, "/")
}

// hostOf returns the request host, port included, which is what a URL root
// needs. It is the one piece of the request this package reads directly.
func hostOf(request *http.Request) string {
	if request == nil {
		return ""
	}
	if request.Host != "" {
		return request.Host
	}
	if request.URL != nil {
		return request.URL.Host
	}
	return ""
}

// formatParameterValue turns one parameter into the string that goes in a URL.
//
// A value that knows its own route key answers with it. There is no dynamic
// property access here, so a bound record contributes its route key
// whichever field the route declared.
func formatParameterValue(value any, field string) string {
	_ = field
	switch v := value.(type) {
	case nil:
		return ""
	case string:
		return v
	case UrlRoutable:
		return v.GetRouteKey()
	case bool:
		if v {
			return "1"
		}
		return "0"
	case int:
		return strconv.Itoa(v)
	case int64:
		return strconv.FormatInt(v, 10)
	case float64:
		return strconv.FormatFloat(v, 'f', -1, 64)
	case fmtStringer:
		return v.String()
	default:
		return ""
	}
}

// fmtStringer is fmt.Stringer, named here so formatParameterValue does not
// pull the whole of fmt into a type switch.
type fmtStringer interface{ String() string }

// sortedKeys returns the keys of m in order, so a URL built twice from the
// same parameters is the same string -- which is what a signature over it
// depends on.
func sortedKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}

// RouteNotFoundError is returned when a named route or action is not in the
// table.
type RouteNotFoundError struct {
	Name string
}

func (e *RouteNotFoundError) Error() string {
	return "routing: route [" + e.Name + "] not defined"
}
