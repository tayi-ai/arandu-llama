package routing

import (
	"strings"
)

// RouteUri is the parsed URI of a route: the cleaned path and the binding
// fields a {param:field} qualifier declared.
//
// The parse is what pulls {post:slug} out of the pattern and leaves {post}
// in its place, so the mux sees a pattern it can match and the route model
// binding knows which column to resolve by.
type RouteUri struct {
	URI           string
	BindingFields map[string]string
}

// Parse extracts the {param:field} qualifiers into BindingFields and leaves
// {param} (or {param?}) in the URI.
//
// It is a method on the zero RouteUri rather than a package function, so
// that a package-level Parse is not left saying nothing about what it
// parses.
func (RouteUri) Parse(uri string) RouteUri {
	fields := map[string]string{}
	out := uri

	// Walk the string for {name:field} or {name:field?} segments.
	i := 0
	for i < len(out) {
		start := strings.IndexByte(out[i:], '{')
		if start < 0 {
			break
		}
		start += i
		end := strings.IndexByte(out[start:], '}')
		if end < 0 {
			break
		}
		end += start
		inner := out[start+1 : end]
		colon := strings.IndexByte(inner, ':')
		if colon < 0 {
			i = end + 1
			continue
		}
		name := inner[:colon]
		field := inner[colon+1:]
		fields[name] = field

		optional := strings.HasSuffix(inner, "?")
		replacement := "{" + name + "}"
		if optional {
			replacement = "{" + name + "?}"
		}
		out = out[:start] + replacement + out[end+1:]
		i = start + len(replacement)
	}
	return RouteUri{URI: out, BindingFields: fields}
}

// SetBindingFieldsFromUri parses the route's pattern, applies the binding
// fields to the route, and returns the cleaned URI. It is what the group merge
// and the URL generator call when a pattern carries {param:field}.
func (rt *Route) SetBindingFieldsFromUri() string {
	if rt == nil {
		return ""
	}
	parsed := RouteUri{}.Parse(rt.Pattern)
	if len(parsed.BindingFields) > 0 {
		if rt.bindingFields == nil {
			rt.bindingFields = map[string]string{}
		}
		for k, v := range parsed.BindingFields {
			rt.bindingFields[k] = v
		}
		rt.Pattern = parsed.URI
	}
	return rt.Pattern
}

// parseControllerCallback splits a "Controller@method" string into its two
// halves. It is what GetControllerClass and GetActionMethod read.
func parseControllerCallback(uses string) (string, string) {
	if i := strings.Index(uses, "@"); i >= 0 {
		return uses[:i], uses[i+1:]
	}
	return uses, ""
}
