package http

import (
	"strings"
)

// IsJSON reports whether the Content-Type header indicates JSON. The check
// is for "/json" or "+json" anywhere in the content type, so
// "application/json", "application/vnd.api+json" and "application/problem+json"
// all match.
func (r *Request) IsJSON() bool {
	contentType := r.request.Header.Get("Content-Type")
	return strings.Contains(contentType, "/json") || strings.Contains(contentType, "+json")
}

// ExpectsJSON reports whether the request probably expects a JSON response.
// True when the request is AJAX (not PJAX) and accepts any content type, or
// when it explicitly wants JSON.
func (r *Request) ExpectsJSON() bool {
	return (r.Ajax() && !r.PJAX() && r.AcceptsAnyContentType()) || r.WantsJSON()
}

// WantsJSON reports whether the first acceptable content type is JSON.
func (r *Request) WantsJSON() bool {
	acceptable := r.GetAcceptableContentTypes()
	if len(acceptable) == 0 {
		return false
	}
	first := strings.ToLower(acceptable[0])
	return strings.Contains(first, "/json") || strings.Contains(first, "+json")
}

// Accepts reports whether the request accepts any of the given content
// types. An empty Accept header accepts everything.
func (r *Request) Accepts(contentTypes ...string) bool {
	accepts := r.GetAcceptableContentTypes()
	if len(accepts) == 0 {
		return true
	}
	for _, accept := range accepts {
		accept = strings.TrimSpace(accept)
		if i := strings.Index(accept, ";"); i >= 0 {
			accept = strings.TrimSpace(accept[:i])
		}
		if accept == "*/*" || accept == "*" {
			return true
		}
		for _, contentType := range contentTypes {
			acceptLower := strings.ToLower(accept)
			typeLower := strings.ToLower(contentType)
			if matchesType(acceptLower, typeLower) || acceptLower == strings.SplitN(typeLower, "/", 2)[0]+"/*" {
				return true
			}
		}
	}
	return false
}

// AcceptsAnyContentType reports whether the request accepts any content type
// (empty Accept, "*/*" or "*").
func (r *Request) AcceptsAnyContentType() bool {
	acceptable := r.GetAcceptableContentTypes()
	if len(acceptable) == 0 {
		return true
	}
	first := acceptable[0]
	return first == "*/*" || first == "*"
}

// AcceptsJSON reports whether the request accepts "application/json".
func (r *Request) AcceptsJSON() bool { return r.Accepts("application/json") }

// AcceptsHTML reports whether the request accepts "text/html".
func (r *Request) AcceptsHTML() bool { return r.Accepts("text/html") }

// Format is the response format the request prefers, derived from the
// Accept header. Returns the default ("html") when nothing matches.
func (r *Request) Format(def ...string) string {
	defaultFormat := "html"
	if len(def) > 0 {
		defaultFormat = def[0]
	}
	for _, accept := range r.GetAcceptableContentTypes() {
		accept = strings.TrimSpace(accept)
		if i := strings.Index(accept, ";"); i >= 0 {
			accept = strings.TrimSpace(accept[:i])
		}
		if format := mimeToFormat(accept); format != "" {
			return format
		}
	}
	return defaultFormat
}

// Prefers is the most suitable content type among the given ones, based on
// content negotiation. Returns empty when none match.
func (r *Request) Prefers(contentTypes ...string) string {
	accepts := r.GetAcceptableContentTypes()
	for _, accept := range accepts {
		accept = strings.TrimSpace(accept)
		if i := strings.Index(accept, ";"); i >= 0 {
			accept = strings.TrimSpace(accept[:i])
		}
		if accept == "*/*" || accept == "*" {
			if len(contentTypes) > 0 {
				return contentTypes[0]
			}
		}
		for _, contentType := range contentTypes {
			typeLower := strings.ToLower(contentType)
			acceptLower := strings.ToLower(accept)
			// The arguments are the other way round from Accepts: here the
			// caller's type is the plain one and the Accept header is where
			// the vendor suffix may be.
			if matchesType(typeLower, acceptLower) || acceptLower == strings.SplitN(typeLower, "/", 2)[0]+"/*" {
				return contentType
			}
		}
	}
	return ""
}

// GetAcceptableContentTypes is the content types the Accept header lists, in
// priority order, without quality parameters.
func (r *Request) GetAcceptableContentTypes() []string {
	accept := r.request.Header.Get("Accept")
	if accept == r.cachedAccept && r.acceptableContentTypes != nil {
		return r.acceptableContentTypes
	}
	r.cachedAccept = accept
	r.acceptableContentTypes = parseAcceptHeader(accept)
	return r.acceptableContentTypes
}

// MatchesType reports whether two content types match, accounting for the
// "+suffix" vendor syntax. Accepts and Prefers both rely on it.
func MatchesType(actual, typ string) bool { return matchesType(actual, typ) }

// matchesType checks whether two content types match.
//
// The direction is not symmetric: actual is the plain type and typ is where
// the vendor suffix may be, so matchesType("application/json",
// "application/vnd.api+json") is true and the same pair the other way round
// is false.
//
// This was previously implemented backwards -- the suffix was taken off typ
// and looked for on the end of actual's subtype -- and since actual's
// subtype is a bare "json" at that point, strings.HasSuffix("json", "+json")
// is false for every input there is. Every vendor type failed to match, so
// an API answering application/vnd.api+json saw Accepts and Prefers refuse
// it.
func matchesType(actual, typ string) bool {
	if actual == typ {
		return true
	}
	slash := strings.Index(actual, "/")
	if slash < 0 {
		return false
	}

	// The match is a search rather than a plain prefix comparison:
	// "application/" then at least one character (which is why
	// "application/+json" does not match) then "+json", found anywhere in
	// typ.
	prefix, suffix := actual[:slash+1], "+"+actual[slash+1:]
	for i := 0; i+len(prefix) < len(typ); i++ {
		if strings.HasPrefix(typ[i:], prefix) &&
			strings.Contains(typ[i+len(prefix)+1:], suffix) {
			return true
		}
	}
	return false
}

// parseAcceptHeader splits the Accept header into ordered content types,
// stripping quality parameters and sorting by quality (q=) value descending.
// The result is the list GetAcceptableContentTypes returns.
func parseAcceptHeader(header string) []string {
	if header == "" {
		return nil
	}
	type entry struct {
		quality float64
		value   string
		index   int
	}
	var entries []entry
	for i, part := range strings.Split(header, ",") {
		part = strings.TrimSpace(part)
		if part == "" {
			continue
		}
		quality := 1.0
		if semi := strings.Index(part, ";"); semi >= 0 {
			params := part[semi+1:]
			part = strings.TrimSpace(part[:semi])
			for _, param := range strings.Split(params, ";") {
				param = strings.TrimSpace(param)
				if strings.HasPrefix(param, "q=") {
					quality = parseFloatDefault(param[2:], 1.0)
				}
			}
		}
		entries = append(entries, entry{quality: quality, value: part, index: i})
	}
	// Sort by quality descending, preserving input order for equal quality.
	for i := 1; i < len(entries); i++ {
		for j := i; j > 0 && (entries[j].quality > entries[j-1].quality ||
			(entries[j].quality == entries[j-1].quality && entries[j].index < entries[j-1].index)); j-- {
			entries[j], entries[j-1] = entries[j-1], entries[j]
		}
	}
	out := make([]string, 0, len(entries))
	for _, e := range entries {
		out = append(out, e.value)
	}
	return out
}

func parseFloatDefault(s string, def float64) float64 {
	s = strings.TrimSpace(s)
	n, ok := parseFloat(s)
	if !ok {
		return def
	}
	return n
}

// mimeToFormat maps a content type to a format name (html, json, xml, txt,
// css, js, etc.).
func mimeToFormat(contentType string) string {
	contentType = strings.ToLower(strings.TrimSpace(contentType))
	if i := strings.Index(contentType, ";"); i >= 0 {
		contentType = strings.TrimSpace(contentType[:i])
	}
	switch contentType {
	case "text/html":
		return "html"
	case "application/json", "application/x-json":
		return "json"
	case "application/xml", "text/xml":
		return "xml"
	case "text/plain":
		return "txt"
	case "text/css":
		return "css"
	case "application/javascript", "text/javascript":
		return "js"
	case "application/rss+xml":
		return "rss"
	case "application/atom+xml":
		return "atom"
	case "application/yaml", "text/yaml":
		return "yaml"
	case "application/pdf":
		return "pdf"
	case "image/png", "image/jpeg", "image/gif", "image/webp", "image/svg+xml":
		return "image"
	}
	// Vendor +json or +xml
	if strings.Contains(contentType, "+json") {
		return "json"
	}
	if strings.Contains(contentType, "+xml") {
		return "xml"
	}
	return ""
}

// parseFloat is a minimal float parser that accepts the digits, dot and sign
// the q= parameter carries. It avoids importing strconv here, keeping the
// content file free of standard library imports beyond strings.
func parseFloat(s string) (float64, bool) {
	if s == "" {
		return 0, false
	}
	negative := false
	if s[0] == '-' {
		negative = true
		s = s[1:]
	} else if s[0] == '+' {
		s = s[1:]
	}
	var result float64
	var fraction float64
	hasFraction := false
	for _, c := range s {
		if c == '.' {
			if hasFraction {
				return 0, false
			}
			hasFraction = true
			continue
		}
		if c < '0' || c > '9' {
			return 0, false
		}
		digit := float64(c - '0')
		if hasFraction {
			fraction = fraction*10 + digit
		} else {
			result = result*10 + digit
		}
	}
	for fraction > 1 {
		fraction /= 10
	}
	if hasFraction {
		result += fraction
	}
	if negative {
		result = -result
	}
	return result, true
}
