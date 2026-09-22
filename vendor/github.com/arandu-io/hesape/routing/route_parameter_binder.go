package routing

import "net/http"

// RouteParameterBinder reads a route's parameters off a request.
//
// The mux has already matched the path, so the path values are read back
// through PathValue; the host half and the defaults are this type's work.
//
// # It reads, it does not resolve
//
// What comes back here is what arrived in the URL: strings. Turning a string
// into a record is ImplicitRouteBinding's job, and that is where the Grant
// and the tenant filter are. Nothing in this file reaches a database, and
// nothing in it should: an id read off a path is untrusted input until a
// policy has been consulted about it.
type RouteParameterBinder struct {
	route *Route
}

// NewRouteParameterBinder builds a binder for route.
func NewRouteParameterBinder(route *Route) *RouteParameterBinder {
	return &RouteParameterBinder{route: route}
}

// Parameters returns every parameter of the route for this request, host
// parameters and defaults included.
func (b *RouteParameterBinder) Parameters(request *http.Request) map[string]any {
	parameters := b.bindPathParameters(request)

	if b.route != nil && b.route.GetDomain() != "" {
		parameters = b.bindHostParameters(request, parameters)
	}

	return b.replaceDefaults(parameters)
}

// bindPathParameters reads the route's path parameters off the request.
func (b *RouteParameterBinder) bindPathParameters(request *http.Request) map[string]any {
	out := map[string]any{}
	if b.route == nil || request == nil {
		return out
	}
	for _, name := range b.route.ParameterNames() {
		if value := request.PathValue(name); value != "" {
			out[name] = value
		}
	}
	return out
}

// bindHostParameters reads the route's host parameters off the request. A
// route declared for "{tenant}.example.com" takes its first segment from the
// host, and a path parameter of the same name wins.
func (b *RouteParameterBinder) bindHostParameters(request *http.Request, parameters map[string]any) map[string]any {
	if request == nil {
		return parameters
	}

	host := hostOf(request)
	if i := indexByte(host, ':'); i >= 0 {
		host = host[:i]
	}

	matched := matchHostPattern(b.route.GetDomain(), host)
	out := make(map[string]any, len(matched)+len(parameters))
	for k, v := range matched {
		out[k] = v
	}
	for k, v := range parameters {
		out[k] = v
	}
	return out
}

// replaceDefaults fills in the route's defaults. A parameter with no value
// takes the route's default, and a default with no parameter is added --
// which is how a redirect route carries its destination in the same place a
// bound value would live.
func (b *RouteParameterBinder) replaceDefaults(parameters map[string]any) map[string]any {
	if b.route == nil {
		return parameters
	}
	defaults, _ := b.route.GetDefaults().(map[string]any)

	for key, value := range parameters {
		if value == nil {
			parameters[key] = defaults[key]
		}
	}
	for key, value := range defaults {
		if _, set := parameters[key]; !set {
			parameters[key] = value
		}
	}
	return parameters
}

// matchHostPattern matches a host against a domain pattern with {name}
// placeholders, returning what each placeholder captured.
func matchHostPattern(pattern, host string) map[string]string {
	out := map[string]string{}
	patternParts := splitByte(pattern, '.')
	hostParts := splitByte(host, '.')
	if len(patternParts) != len(hostParts) {
		return out
	}
	for i, part := range patternParts {
		if len(part) >= 2 && part[0] == '{' && part[len(part)-1] == '}' {
			name := part[1 : len(part)-1]
			if j := indexByte(name, ':'); j >= 0 {
				name = name[:j]
			}
			out[trimSuffixByte(name, '?')] = hostParts[i]
			continue
		}
		if part != hostParts[i] {
			return map[string]string{}
		}
	}
	return out
}

func indexByte(s string, c byte) int {
	for i := 0; i < len(s); i++ {
		if s[i] == c {
			return i
		}
	}
	return -1
}

func splitByte(s string, c byte) []string {
	var out []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i] == c {
			out = append(out, s[start:i])
			start = i + 1
		}
	}
	return append(out, s[start:])
}

func trimSuffixByte(s string, c byte) string {
	if len(s) > 0 && s[len(s)-1] == c {
		return s[:len(s)-1]
	}
	return s
}
