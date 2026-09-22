package routing

import (
	"net/http"
	"net/url"
	"strconv"
	"strings"

	"github.com/arandu-io/hesape/routing/exceptions"
)

// RouteUrlGenerator is the half of UrlGenerator that turns a route and a bag
// of parameters into an address.
//
// The work it does that a string concatenation does not: it fills named
// placeholders, falls back to the defaults, drops optional segments nobody
// filled, puts what is left over on the query string, and refuses -- rather
// than emitting a URL with a literal "{invoice}" in it -- when a required
// placeholder has no value.
type RouteUrlGenerator struct {
	url     *UrlGenerator
	request *http.Request

	// DefaultParameters are the named defaults every route inherits, keyed by
	// parameter name or by "name:field" when the route binds by a field.
	DefaultParameters map[string]any

	// DontEncode are the characters restored after encoding, so a URL keeps
	// its separators.
	DontEncode map[string]string
}

// dontEncode is the default DontEncode table.
var dontEncode = map[string]string{
	"%2F": "/",
	"%40": "@",
	"%3A": ":",
	"%3B": ";",
	"%2C": ",",
	"%3D": "=",
	"%2B": "+",
	"%21": "!",
	"%2A": "*",
	"%7C": "|",
	"%3F": "?",
	"%26": "&",
	"%23": "#",
	"%25": "%",
}

// NewRouteUrlGenerator returns a generator for u and request.
func NewRouteUrlGenerator(u *UrlGenerator, request *http.Request) *RouteUrlGenerator {
	table := make(map[string]string, len(dontEncode))
	for k, v := range dontEncode {
		table[k] = v
	}
	return &RouteUrlGenerator{
		url:               u,
		request:           request,
		DefaultParameters: map[string]any{},
		DontEncode:        table,
	}
}

// To builds the URL of a route.
//
// It returns an error when a placeholder the caller left unfilled would 404
// the reader; saying so at the call site is the whole reason a named route
// beats a string.
func (g *RouteUrlGenerator) To(route *Route, parameters map[string]any, absolute bool) (string, error) {
	named, query := g.formatParameters(route, parameters)

	domain := g.getRouteDomain(route, named)

	// Build the whole address -- root, path and query -- and only then look for
	// placeholders nobody filled, so the message can name all of them at once.
	root := g.replaceRootParameters(route, domain, named)
	path := g.replaceRouteParameters(route.URI(), named)

	uri := g.addQueryString(g.url.Format(root, path, route), query)

	if missing := unfilledParameters(uri); len(missing) > 0 {
		return "", exceptions.ForMissingParameters(route, missing...)
	}

	uri = strings.NewReplacer(replacerPairs(g.DontEncode)...).Replace(rawURLEncode(uri))

	if !absolute {
		return "/" + strings.TrimLeft(stripOrigin(uri), "/"), nil
	}

	return uri, nil
}

// getRouteDomain returns the route's domain, formatted, or empty when the
// route declares none.
func (g *RouteUrlGenerator) getRouteDomain(route *Route, parameters map[string]string) string {
	if route.GetDomain() == "" {
		return ""
	}
	return g.formatDomain(route, parameters)
}

// formatDomain prepends the route's scheme to its domain and adds the port.
func (g *RouteUrlGenerator) formatDomain(route *Route, parameters map[string]string) string {
	return g.addPortToDomain(g.getRouteScheme(route) + route.GetDomain())
}

// getRouteScheme returns the scheme the route is restricted to, or the
// generator's own.
func (g *RouteUrlGenerator) getRouteScheme(route *Route) string {
	switch {
	case route.HttpOnly():
		return "http://"
	case route.HttpsOnly():
		return "https://"
	default:
		return g.url.FormatScheme(nil)
	}
}

// addPortToDomain appends the request's port to domain, unless it is the
// scheme's default.
func (g *RouteUrlGenerator) addPortToDomain(domain string) string {
	if g.request == nil {
		return domain
	}
	secure := g.request.TLS != nil
	port := requestPort(g.request, secure)
	if (secure && port == 443) || (!secure && port == 80) {
		return domain
	}
	return domain + ":" + strconv.Itoa(port)
}

// formatParameters returns the value for each of the route's own parameters,
// in the route's order, and everything left over for the query string. A Go
// map has no order to match positional values against; Routes.Route is where
// that shape lives.
func (g *RouteUrlGenerator) formatParameters(route *Route, parameters map[string]any) (map[string]string, map[string]any) {
	given := g.url.FormatParameters(parameters)

	named := make(map[string]string, len(given))
	optional := route.GetOptionalParameterNames()

	for _, name := range route.ParameterNames() {
		field := route.BindingFieldFor(name)

		if value, ok := given[name]; ok {
			named[name] = formatParameterValue(value, field)
			delete(given, name)
			continue
		}

		key := name
		if field != "" {
			key = name + ":" + field
		}
		if value, ok := g.DefaultParameters[key]; ok {
			named[name] = formatParameterValue(value, field)
			continue
		}
		if value, ok := g.DefaultParameters[name]; ok {
			named[name] = formatParameterValue(value, field)
			continue
		}
		if _, ok := optional[name]; ok {
			// An optional placeholder nobody filled is dropped, not missing.
			named[name] = ""
		}
	}

	// Whatever is left named something the route does not have is a query
	// parameter, which is how ?page=2 reaches a paginated index.
	return named, given
}

// replaceRootParameters fills the placeholders of the URL root -- scheme and
// domain -- with the route's parameters.
func (g *RouteUrlGenerator) replaceRootParameters(route *Route, domain string, parameters map[string]string) string {
	scheme := g.getRouteScheme(route)
	return g.replaceRouteParameters(g.url.FormatRoot(scheme, domain), parameters)
}

// replaceRouteParameters fills the named placeholders and drops the optional
// ones left empty.
func (g *RouteUrlGenerator) replaceRouteParameters(path string, parameters map[string]string) string {
	var b strings.Builder
	rest := path

	for {
		start := strings.IndexByte(rest, '{')
		if start < 0 {
			b.WriteString(rest)
			break
		}
		end := strings.IndexByte(rest[start:], '}')
		if end < 0 {
			b.WriteString(rest)
			break
		}
		end += start

		b.WriteString(rest[:start])

		inner := rest[start+1 : end]
		name := strings.TrimSuffix(strings.TrimSuffix(inner, "?"), "...")
		if i := strings.IndexByte(name, ':'); i >= 0 {
			name = name[:i]
		}

		value, filled := parameters[name]
		switch {
		case filled && value != "":
			b.WriteString(value)
		case name == "$":
			// "{$}" anchors a pattern that would otherwise match a subtree. It
			// is not a parameter, and it is not part of the address either.
		case filled || strings.HasSuffix(inner, "?"):
			// An optional placeholder with nothing in it leaves a bare slash,
			// which the trim below removes.
		default:
			b.WriteString(rest[start : end+1])
		}

		rest = rest[end+1:]
	}

	out := b.String()
	for strings.Contains(out, "//") {
		out = strings.ReplaceAll(out, "//", "/")
	}
	return strings.Trim(out, "/")
}

// addQueryString appends the query string built from parameters. A fragment
// is moved to the end, because a query string after it is a query string
// nobody reads.
func (g *RouteUrlGenerator) addQueryString(uri string, parameters map[string]any) string {
	fragment := ""
	if i := strings.IndexByte(uri, '#'); i >= 0 {
		fragment = uri[i+1:]
		uri = uri[:i]
	}

	uri += g.getRouteQueryString(parameters)

	if fragment == "" {
		return uri
	}
	return uri + "#" + fragment
}

// getRouteQueryString builds the query string from whatever parameters were
// left over after the named placeholders were filled.
func (g *RouteUrlGenerator) getRouteQueryString(parameters map[string]any) string {
	if len(parameters) == 0 {
		return ""
	}
	values := url.Values{}
	for _, key := range sortedKeys(parameters) {
		values.Set(key, formatParameterValue(parameters[key], ""))
	}
	query := values.Encode()
	if query == "" {
		return ""
	}
	return "?" + query
}

// Defaults merges defaults into DefaultParameters.
func (g *RouteUrlGenerator) Defaults(defaults map[string]any) {
	if g.DefaultParameters == nil {
		g.DefaultParameters = map[string]any{}
	}
	for k, v := range defaults {
		g.DefaultParameters[k] = v
	}
}

// unfilledParameters returns the placeholders still standing in a built URL.
func unfilledParameters(uri string) []string {
	var out []string
	rest := uri
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
		out = append(out, rest[start+1:end])
		rest = rest[end+1:]
	}
}

// stripOrigin removes the scheme and host from a built URL: everything up to
// the first slash or question mark, plus a leading "//".
func stripOrigin(uri string) string {
	if strings.HasPrefix(uri, "//") {
		uri = uri[2:]
	}
	for i := 0; i < len(uri); i++ {
		if uri[i] == '/' || uri[i] == '?' {
			return uri[i:]
		}
	}
	return ""
}

// rawURLEncode percent-escapes every byte outside the unreserved set. The
// separators are put back by DontEncode.
func rawURLEncode(s string) string {
	const upper = "0123456789ABCDEF"
	var b strings.Builder
	b.Grow(len(s))
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch {
		case c >= 'A' && c <= 'Z', c >= 'a' && c <= 'z', c >= '0' && c <= '9',
			c == '-', c == '_', c == '.', c == '~':
			b.WriteByte(c)
		default:
			b.WriteByte('%')
			b.WriteByte(upper[c>>4])
			b.WriteByte(upper[c&15])
		}
	}
	return b.String()
}

// replacerPairs flattens a replacement table for strings.NewReplacer.
func replacerPairs(table map[string]string) []string {
	pairs := make([]string, 0, len(table)*2)
	for from, to := range table {
		pairs = append(pairs, from, to)
	}
	return pairs
}

// requestPort returns the port the request arrived on, defaulting to the one
// the scheme implies.
func requestPort(request *http.Request, secure bool) int {
	host := ""
	if request.Host != "" {
		host = request.Host
	} else if request.URL != nil {
		host = request.URL.Host
	}
	if i := strings.LastIndexByte(host, ':'); i >= 0 && !strings.Contains(host[i:], "]") {
		if port, err := strconv.Atoi(host[i+1:]); err == nil {
			return port
		}
	}
	if secure {
		return 443
	}
	return 80
}
