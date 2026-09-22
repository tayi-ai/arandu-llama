package http

import (
	"context"
	"crypto/sha1"
	"encoding/hex"
	"encoding/json"
	"io"
	"net"
	"net/url"
	"regexp"
	"strings"

	stdhttp "net/http"

	"github.com/arandu-io/hesape/session"
)

// Route is the minimal view of the matched route the request needs.
//
// The concrete type is hesape/routing.Route. The interface is declared here
// so that hesape/http does not import hesape/routing, which would be a
// cycle: routing builds the request, and the request needs the route's name
// and parameters, not the router itself.
type Route interface {
	RouteName() string
	Parameter(name string, def ...any) any
	Parameters() map[string]any
	Methods() []string
	Domain() string
	URI() string
}

// Request wraps a *net/http.Request and provides the methods a controller
// action reaches for: the input the body and the query string carried, the
// headers, the cookies, the files, the session that survived the redirect,
// and the route that matched.
//
// The tenant NEVER comes from the request. There is no method here that
// reads a tenant id out of a path parameter, a query string, a header or a
// body field, and adding one would be the most direct route to a
// cross-tenant leak. The tenant is on the auth.Grant the policy mints, and
// the repository reads it from there. If a method here seems to offer
// tenant access, it does not.
//
// A note on names: method names use the initial uppercase Go requires.
// Input is Input, never Get. Old is Old, never Previous. Initialisms are
// upper case: FullURL, IsJSON. Where a failure can happen, the method
// returns (T, error).
type Request struct {
	// request is the standard library request this wraps. It is unexported
	// so that Request's own methods are the only surface; a caller that
	// needs the raw request reaches it through RawRequest.
	request *stdhttp.Request

	// session is the session store the middleware set. Old, Flash and Flush
	// read it; when it is nil, Old returns the default and Flash panics.
	session *session.Store

	// json is the decoded JSON body, cached after the first read. The body of
	// a *http.Request is a one-shot reader; caching it here is what lets Input
	// and Json be called more than once.
	json       map[string]any
	jsonParsed bool

	// userResolver is the closure the auth middleware installs, which
	// returns the signed-in user.
	userResolver func(guard string) any

	// routeResolver is the closure the router installs, which returns the
	// matched route.
	routeResolver func() Route

	// precognitive is whether this request was marked as a precognitive
	// validation request.
	precognitive bool

	// cachedAccept and acceptableContentTypes cache the Accept header and
	// its parsed form.
	cachedAccept           string
	acceptableContentTypes []string
}

// NewRequest wraps a *net/http.Request. It is the constructor a handler or a
// test calls; the router calls it once per request.
func NewRequest(r *stdhttp.Request) *Request {
	return &Request{request: r}
}

// RawRequest returns the *net/http.Request this wraps, for the rare case
// that needs something Request's own methods do not cover. It is not the
// common path: the methods on Request are what a controller should reach
// for.
func (r *Request) RawRequest() *stdhttp.Request { return r.request }

// CreateFrom is a new Request built from another, sharing its query, body,
// headers, session and resolvers. It is what a middleware that needs a
// modified copy of the request uses.
func CreateFrom(from *Request) *Request {
	out := &Request{
		request:       from.request,
		session:       from.session,
		userResolver:  from.userResolver,
		routeResolver: from.routeResolver,
		precognitive:  from.precognitive,
	}
	return out
}

// Method is the HTTP method, upper case.
func (r *Request) Method() string { return r.request.Method }

// IsMethod reports a case-insensitive method match.
func (r *Request) IsMethod(method string) bool {
	return strings.EqualFold(r.request.Method, method)
}

// Path is the path without leading or trailing slashes, "/" when the path
// is empty.
func (r *Request) Path() string {
	pattern := strings.Trim(r.request.URL.Path, "/")
	if pattern == "" {
		return "/"
	}
	return pattern
}

// DecodedPath is the path, URL-decoded.
func (r *Request) DecodedPath() string {
	decoded, err := url.PathUnescape(r.Path())
	if err != nil {
		return r.Path()
	}
	return decoded
}

// Segment is the nth segment of the path (1-based), or default when the
// index is out of range.
func (r *Request) Segment(index int, def ...string) string {
	segments := r.Segments()
	if index < 1 || index > len(segments) {
		if len(def) > 0 {
			return def[0]
		}
		return ""
	}
	return segments[index-1]
}

// Segments is the non-empty parts of the decoded path, split on "/".
func (r *Request) Segments() []string {
	decoded := r.DecodedPath()
	parts := strings.Split(decoded, "/")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

// URL is the request URL without the query string. It is the address a
// link inside the page uses, so that the application keeps working behind a
// proxy, on a staging host and on somebody's laptop.
//
// *http.Request carries scheme and host in URL when it was parsed from a
// full target, and in Host when it arrived over the wire. Both are checked,
// so a test that builds a request from "http://example.com/path" and a
// server that builds one from "/path" both get the right answer.
func (r *Request) URL() string {
	scheme := r.request.URL.Scheme
	host := r.request.URL.Host
	if host == "" {
		host = r.request.Host
	}
	if scheme == "" {
		scheme = "http"
		if r.request.TLS != nil {
			scheme = "https"
		}
	}
	path := r.request.URL.Path
	uri := scheme + "://" + host + path
	return strings.TrimRight(uri, "/")
}

// FullURL is the full URL, scheme and host included, with the query string.
func (r *Request) FullURL() string {
	scheme := r.request.URL.Scheme
	if scheme == "" {
		scheme = "http"
		if r.request.TLS != nil {
			scheme = "https"
		}
	}
	return scheme + "://" + r.request.Host + r.request.URL.RequestURI()
}

// FullURLWithQuery is the full URL with the given query parameters merged
// into the existing ones.
func (r *Request) FullURLWithQuery(query map[string]string) string {
	values := r.request.URL.Query()
	for k, v := range query {
		values.Set(k, v)
	}
	base := r.URL()
	encoded := values.Encode()
	if encoded == "" {
		return base
	}
	separator := "?"
	if r.request.URL.Path == "/" && r.request.URL.RawQuery == "" {
		separator = "/?"
	}
	return base + separator + encoded
}

// FullURLWithoutQuery is the full URL without the given query parameters.
func (r *Request) FullURLWithoutQuery(keys ...string) string {
	values := r.request.URL.Query()
	for _, key := range keys {
		values.Del(key)
	}
	base := r.URL()
	encoded := values.Encode()
	if encoded == "" {
		return base
	}
	return base + "?" + encoded
}

// Root is the root URL, scheme and host, without a trailing slash.
func (r *Request) Root() string {
	return strings.TrimRight(r.SchemeAndHttpHost(), "/")
}

// Is reports whether the decoded path matches any of the patterns. "*" is
// the only wildcard.
func (r *Request) Is(patterns ...string) bool {
	decoded := r.DecodedPath()
	for _, pattern := range patterns {
		if strIs(pattern, decoded) {
			return true
		}
	}
	return false
}

// RouteIs reports whether the route name matches any of the patterns.
// Returns false when no route is set.
func (r *Request) RouteIs(patterns ...string) bool {
	route := r.matchedRoute()
	if route == nil {
		return false
	}
	name := route.RouteName()
	for _, pattern := range patterns {
		if strIs(pattern, name) {
			return true
		}
	}
	return false
}

// FullURLIs reports whether the full URL matches any of the patterns. "*"
// is the only wildcard.
func (r *Request) FullURLIs(patterns ...string) bool {
	full := r.FullURL()
	for _, pattern := range patterns {
		if strIs(pattern, full) {
			return true
		}
	}
	return false
}

// Host is the host name, without the port.
func (r *Request) Host() string {
	host := r.request.Host
	if h, _, err := net.SplitHostPort(host); err == nil {
		return h
	}
	return host
}

// HTTPHost is the host as the browser sent it, port included when the
// browser sent one.
func (r *Request) HTTPHost() string { return r.request.Host }

// SchemeAndHttpHost is scheme + "://" + host.
func (r *Request) SchemeAndHttpHost() string {
	scheme := "http"
	if r.request.TLS != nil {
		scheme = "https"
	}
	return scheme + "://" + r.request.Host
}

// Secure reports whether the request was over TLS.
func (r *Request) Secure() bool { return r.request.TLS != nil }

// Ajax reports whether the request was made by XMLHttpRequest. htmx sends
// this header too; see WantsJSON for why htmx is deliberately excluded
// there.
func (r *Request) Ajax() bool {
	return r.request.Header.Get("X-Requested-With") == "XMLHttpRequest"
}

// PJAX reports whether the request carries the X-PJAX header.
func (r *Request) PJAX() bool {
	return r.request.Header.Get("X-PJAX") != ""
}

// PreferSafeContent reports whether the Prefer header asks for safe
// content.
func (r *Request) PreferSafeContent() bool {
	prefer := r.request.Header.Get("Prefer")
	for _, part := range strings.Split(prefer, ",") {
		token := strings.TrimSpace(part)
		if i := strings.Index(token, ";"); i >= 0 {
			token = strings.TrimSpace(token[:i])
		}
		if strings.EqualFold(token, "safe") {
			return true
		}
	}
	return false
}

// IP is the client address, without the port. Behind a proxy it is
// whatever the trust-proxies middleware wrote onto RemoteAddr.
func (r *Request) IP() string {
	addr := r.request.RemoteAddr
	if host, _, err := net.SplitHostPort(addr); err == nil {
		return host
	}
	return addr
}

// forwardedForKey is the context key the trust-proxies middleware writes the
// chain under. It is an unexported type so nothing outside can collide with it
// or overwrite it.
type forwardedForKey struct{}

// WithForwardedFor returns a context carrying the chain of client addresses
// the trust-proxies middleware worked out, nearest hop first.
//
// The middleware calls it, because it is the only thing that knows which
// proxies are ours: the chain cannot be recovered from the request afterwards,
// which is why [Request.IPs] could not answer with one. It is exported for
// that, and for the test that drives a request past the middleware.
//
// Only the first entry is verified. Everything to the left of it was written
// by whoever that is and can say anything at all, which is why [Request.IP] --
// what a rate limit, a throttle or a log line is keyed by -- is the first and
// never the last.
func WithForwardedFor(parent context.Context, chain []string) context.Context {
	return context.WithValue(parent, forwardedForKey{}, chain)
}

// ForwardedForFrom returns the chain of client addresses on a request context,
// or nil when no trusted proxy put one there.
func ForwardedForFrom(ctx context.Context) []string {
	chain, _ := ctx.Value(forwardedForKey{}).([]string)
	return chain
}

// IPs is the chain of client addresses, nearest hop first. With no proxy in
// front, this is a list of one.
//
// It is the X-Forwarded-For entries plus the address the request actually
// came from, with our own proxies removed, reversed so that IPs()[0] is
// IP(). It used to return the one address and no chain, so behind a proxy
// sending "X-Forwarded-For: 1.1.1.1, 2.2.2.2" it answered with a single
// element instead of two: the doc said "the chain of client addresses" and
// there was never a chain.
//
// Only the first entry is trustworthy. See [WithForwardedFor].
func (r *Request) IPs() []string {
	if chain := ForwardedForFrom(r.request.Context()); len(chain) > 0 {
		out := make([]string, len(chain))
		copy(out, chain)
		return out
	}
	ip := r.IP()
	if ip == "" {
		return nil
	}
	return []string{ip}
}

// UserAgent is the User-Agent header, or empty.
func (r *Request) UserAgent() string {
	return r.request.Header.Get("User-Agent")
}

// BearerToken is the token of an Authorization: Bearer header, or empty.
// The scheme is compared case-insensitively (RFC 9110) and the value is
// trimmed.
func (r *Request) BearerToken() string {
	header := r.request.Header.Get("Authorization")
	position := strings.Index(strings.ToLower(header), "bearer ")
	if position < 0 {
		return ""
	}
	token := header[position+7:]
	if comma := strings.Index(token, ","); comma >= 0 {
		token = token[:comma]
	}
	return strings.TrimSpace(token)
}

// Header is the first value of a header, or default when the header is
// absent. The name is canonicalised by net/http.
func (r *Request) Header(name string, def ...string) string {
	value := r.request.Header.Get(name)
	if value != "" {
		return value
	}
	if len(def) > 0 {
		return def[0]
	}
	return ""
}

// HasHeader reports whether a header is present, including when its value
// is empty.
func (r *Request) HasHeader(name string) bool {
	return len(r.request.Header.Values(name)) > 0
}

// Server is a server variable, or default. The common CGI-style keys are
// mapped from the request; an unknown HTTP_* key falls back to the
// corresponding header.
func (r *Request) Server(key string, def ...string) string {
	value := r.serverVar(key)
	if value != "" {
		return value
	}
	if len(def) > 0 {
		return def[0]
	}
	return ""
}

func (r *Request) serverVar(key string) string {
	switch key {
	case "REQUEST_METHOD":
		return r.request.Method
	case "REQUEST_URI":
		return r.request.URL.RequestURI()
	case "PATH_INFO":
		return r.request.URL.Path
	case "QUERY_STRING":
		return r.request.URL.RawQuery
	case "SERVER_NAME":
		return r.Host()
	case "SERVER_PORT":
		if _, port, err := net.SplitHostPort(r.request.Host); err == nil {
			return port
		}
		return ""
	case "SERVER_PROTOCOL":
		return r.request.Proto
	case "REMOTE_ADDR":
		return r.request.RemoteAddr
	case "HTTPS":
		if r.Secure() {
			return "on"
		}
		return ""
	case "CONTENT_TYPE":
		return r.request.Header.Get("Content-Type")
	case "CONTENT_LENGTH":
		return r.request.Header.Get("Content-Length")
	}
	if strings.HasPrefix(key, "HTTP_") {
		headerName := strings.ReplaceAll(key[5:], "_", "-")
		return r.request.Header.Get(headerName)
	}
	return ""
}

// Cookie is the value of a cookie, or default when the browser sent none.
func (r *Request) Cookie(name string, def ...string) string {
	cookie, err := r.request.Cookie(name)
	if err == nil {
		return cookie.Value
	}
	if len(def) > 0 {
		return def[0]
	}
	return ""
}

// HasCookie reports whether a cookie is present.
func (r *Request) HasCookie(name string) bool {
	_, err := r.request.Cookie(name)
	return err == nil
}

// Keys is the keys of all input and files, merged.
func (r *Request) Keys() []string {
	input := r.inputMap()
	files := r.allFilesMap()
	seen := make(map[string]bool, len(input)+len(files))
	out := make([]string, 0, len(input)+len(files))
	for k := range input {
		if !seen[k] {
			seen[k] = true
			out = append(out, k)
		}
	}
	for k := range files {
		if !seen[k] {
			seen[k] = true
			out = append(out, k)
		}
	}
	return out
}

// Merge merges the given input into the request's input source, overwriting
// existing keys. Returns the request, so calls can chain.
func (r *Request) Merge(input map[string]any) *Request {
	source := r.inputSource()
	for k, v := range input {
		source[k] = v
	}
	r.replaceInputSource(source)
	return r
}

// MergeIfMissing merges the given input only for keys that are missing from
// the request.
func (r *Request) MergeIfMissing(input map[string]any) *Request {
	source := r.inputSource()
	for k, v := range input {
		if _, ok := source[k]; !ok {
			source[k] = v
		}
	}
	r.replaceInputSource(source)
	return r
}

// Replace replaces the input source entirely.
func (r *Request) Replace(input map[string]any) *Request {
	r.replaceInputSource(input)
	return r
}

// replaceInputSource writes the merged input back to the request's form or
// JSON body, depending on which source Input reads.
func (r *Request) replaceInputSource(source map[string]any) {
	if r.IsJSON() {
		r.json = source
		r.jsonParsed = true
		return
	}
	if r.request.Method == "GET" || r.request.Method == "HEAD" {
		values := make(url.Values, len(source))
		for k, v := range source {
			values.Set(k, stringify(v))
		}
		r.request.URL.RawQuery = values.Encode()
		return
	}
	_ = r.request.ParseForm()
	values := make(url.Values, len(source))
	for k, v := range source {
		values.Set(k, stringify(v))
	}
	r.request.PostForm = values
	r.request.Form = values
}

// ToArray is all input and files as a map.
func (r *Request) ToArray() map[string]any { return r.All() }

// Json is the decoded JSON body. With a key, returns the value at the
// dotted path; without, returns the whole payload.
func (r *Request) Json(key string, def ...any) any {
	payload := r.jsonPayload()
	if key == "" {
		return payload
	}
	if len(def) > 0 {
		return dataGet(payload, key, def[0])
	}
	return dataGet(payload, key, nil)
}

// SetJson replaces the cached JSON payload.
func (r *Request) SetJson(payload map[string]any) *Request {
	r.json = payload
	r.jsonParsed = true
	return r
}

// jsonPayload reads and caches the JSON body. The body reader is restored so
// that downstream code can still read it.
func (r *Request) jsonPayload() map[string]any {
	if r.jsonParsed {
		return r.json
	}
	r.jsonParsed = true
	body, err := io.ReadAll(r.request.Body)
	if err != nil || len(body) == 0 {
		r.json = map[string]any{}
		r.request.Body = io.NopCloser(strings.NewReader(""))
		return r.json
	}
	r.request.Body = io.NopCloser(strings.NewReader(string(body)))
	var parsed map[string]any
	if err := json.Unmarshal(body, &parsed); err != nil {
		r.json = map[string]any{}
		return r.json
	}
	r.json = parsed
	return r.json
}

// Fingerprint is a unique hash of the route methods, domain, URI and client
// IP. Panics when no route is set.
func (r *Request) Fingerprint() string {
	route := r.matchedRoute()
	if route == nil {
		panic("http: unable to generate fingerprint: route unavailable")
	}
	parts := append(route.Methods(), route.Domain(), route.URI(), r.IP())
	return sha1Hex(strings.Join(parts, "|"))
}

func sha1Hex(s string) string {
	h := sha1.Sum([]byte(s))
	return hex.EncodeToString(h[:])
}

// User is the authenticated user, via the resolver the auth middleware
// installed. Returns nil when no resolver is set.
func (r *Request) User(guard ...string) any {
	if r.userResolver == nil {
		return nil
	}
	g := ""
	if len(guard) > 0 {
		g = guard[0]
	}
	return r.userResolver(g)
}

// SetUserResolver sets the resolver User calls.
func (r *Request) SetUserResolver(resolver func(guard string) any) *Request {
	r.userResolver = resolver
	return r
}

// GetUserResolver is the installed resolver, or a no-op when none was set.
func (r *Request) GetUserResolver() func(guard string) any {
	if r.userResolver == nil {
		return func(string) any { return nil }
	}
	return r.userResolver
}

// Route is the matched route, or a specific parameter from it. With no
// arguments, returns the Route. With a string argument, returns the
// parameter of that name from the route. With a default, returns the
// default when the parameter is absent.
//
// Returns nil when no resolver is set.
func (r *Request) Route(args ...any) any {
	route := r.matchedRoute()
	if len(args) == 0 {
		return route
	}
	if route == nil {
		if len(args) > 1 {
			return args[1]
		}
		return nil
	}
	param, _ := args[0].(string)
	return route.Parameter(param, args[1:]...)
}

// matchedRoute returns the matched route, or nil when no resolver is set.
func (r *Request) matchedRoute() Route {
	if r.routeResolver == nil {
		return nil
	}
	return r.routeResolver()
}

// SetRouteResolver sets the resolver Route calls.
func (r *Request) SetRouteResolver(resolver func() Route) *Request {
	r.routeResolver = resolver
	return r
}

// GetRouteResolver is the installed resolver, or a no-op when none was set.
func (r *Request) GetRouteResolver() func() Route {
	if r.routeResolver == nil {
		return func() Route { return nil }
	}
	return r.routeResolver
}

// HasSession reports whether a session store was set.
func (r *Request) HasSession() bool { return r.session != nil }

// Session is the session store. Panics when none is set.
func (r *Request) Session() *session.Store {
	if r.session == nil {
		panic("http: session store not set on request")
	}
	return r.session
}

// SetSession sets the session store.
func (r *Request) SetSession(s *session.Store) *Request {
	r.session = s
	return r
}

// SetPrecognitive marks the request as precognitive, which IsPrecognitive
// then reports.
func (r *Request) SetPrecognitive() *Request {
	r.precognitive = true
	return r
}

// IsPrecognitive reports whether the request was marked as a precognitive
// validation request.
func (r *Request) IsPrecognitive() bool { return r.precognitive }

// IsAttemptingPrecognition reports whether the Precognition header is
// "true".
func (r *Request) IsAttemptingPrecognition() bool {
	return r.request.Header.Get("Precognition") == "true"
}

// FilterPrecognitiveRules returns only the rules whose attribute matches one
// of the Precognition-Validate-Only header's patterns, when that header is
// present; otherwise it returns the rules unchanged.
func (r *Request) FilterPrecognitiveRules(rules map[string]any) map[string]any {
	if !r.HasHeader("Precognition-Validate-Only") {
		return rules
	}
	validateOnly := strings.Split(r.request.Header.Get("Precognition-Validate-Only"), ",")
	out := make(map[string]any, len(rules))
	for attribute, rule := range rules {
		if r.shouldValidatePrecognitiveAttribute(attribute, validateOnly) {
			out[attribute] = rule
		}
	}
	return out
}

func (r *Request) shouldValidatePrecognitiveAttribute(attribute string, patterns []string) bool {
	for _, pattern := range patterns {
		pattern = strings.TrimSpace(pattern)
		regex := "^" + strings.ReplaceAll(regexp.QuoteMeta(pattern), `\*`, `[^.]+`) + "$"
		if matched, _ := regexp.MatchString(regex, attribute); matched {
			return true
		}
	}
	return false
}

// Prefetch reports whether the request is a browser prefetch.
func (r *Request) Prefetch() bool {
	return strings.EqualFold(r.serverVar("HTTP_X_MOZ"), "prefetch") ||
		strings.EqualFold(r.request.Header.Get("Purpose"), "prefetch") ||
		strings.EqualFold(r.request.Header.Get("Sec-Purpose"), "prefetch")
}

// Instance is the request itself.
func (r *Request) Instance() *Request { return r }

// URI is the full URL string.
func (r *Request) URI() string { return r.FullURL() }

// Capture returns NewRequest called on the given *http.Request.
func Capture(r *stdhttp.Request) *Request { return NewRequest(r) }

// CreateFromBase returns NewRequest called on the given *http.Request.
func CreateFromBase(r *stdhttp.Request) *Request { return NewRequest(r) }

// Duplicate is a copy of the request with optional overrides for query,
// post, cookies and server. Nil arguments keep the original.
func (r *Request) Duplicate(query, post, cookies, server map[string]any) *Request {
	out := CreateFrom(r)
	if query != nil {
		values := make(url.Values, len(query))
		for k, v := range query {
			values.Set(k, stringify(v))
		}
		out.request.URL.RawQuery = values.Encode()
	}
	if post != nil {
		values := make(url.Values, len(post))
		for k, v := range post {
			values.Set(k, stringify(v))
		}
		out.request.PostForm = values
		out.request.Form = values
	}
	return out
}

// SetRequestLocale stores the locale on the request for the view layer to
// read. It is a string because Go has no locale type, and the view layer
// reads it through the same string.
func (r *Request) SetRequestLocale(locale string) *Request {
	if r.request == nil {
		return r
	}
	// Store on the request context as a simple value. The view layer reads it
	// from there; a struct field would not survive a clone.
	ctx := context.WithValue(r.request.Context(), requestLocaleKey{}, locale)
	r.request = r.request.WithContext(ctx)
	return r
}

// SetDefaultRequestLocale stores the default locale on the request, for the
// view layer to fall back to.
func (r *Request) SetDefaultRequestLocale(locale string) *Request {
	if r.request == nil {
		return r
	}
	ctx := context.WithValue(r.request.Context(), defaultRequestLocaleKey{}, locale)
	r.request = r.request.WithContext(ctx)
	return r
}

// GetSession is the session store. Panics when none is set. It is an alias
// for Session.
func (r *Request) GetSession() *session.Store { return r.Session() }

// SetLaravelSession is an alias for SetSession.
func (r *Request) SetLaravelSession(s *session.Store) *Request { return r.SetSession(s) }

// requestLocaleKey and defaultRequestLocaleKey are the context keys for the
// locale the view layer reads.
type requestLocaleKey struct{}
type defaultRequestLocaleKey struct{}
