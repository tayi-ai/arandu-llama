package str

import (
	"regexp"
	"strings"
)

// Is reports whether the value matches any one of the patterns, where * stands
// for any run of characters, including none. Every other rune is literal, so a
// pattern is safe to build from configuration without escaping anything.
//
// Is([]string{"admin/*"}, "admin/users", false) is true, and so is
// Is([]string{"*"}, anything, false).
//
// It is not path.Match: * crosses a slash, because what this matches is a route
// name, an ability or a host, not a file name.
func Is(patterns []string, value string, ignoreCase bool) bool {
	if ignoreCase {
		value = strings.ToLower(value)
	}
	for _, pattern := range patterns {
		if pattern == "*" {
			return true
		}
		if ignoreCase {
			pattern = strings.ToLower(pattern)
		}
		if pattern == value {
			return true
		}
		if globMatch(pattern, value) {
			return true
		}
	}
	return false
}

// globMatch is the pattern with its asterisks read as ".*" and everything else
// read literally, anchored at both ends.
func globMatch(pattern, value string) bool {
	parts := strings.Split(pattern, "*")
	if len(parts) == 1 {
		return false
	}
	if !strings.HasPrefix(value, parts[0]) {
		return false
	}
	rest := value[len(parts[0]):]
	for _, part := range parts[1 : len(parts)-1] {
		i := strings.Index(rest, part)
		if i < 0 {
			return false
		}
		rest = rest[i+len(part):]
	}
	return strings.HasSuffix(rest, parts[len(parts)-1])
}

// Match returns the first capture group of the first match, or the whole match
// when the pattern captures nothing, or the empty string when the pattern does
// not match at all.
//
// The pattern arrives compiled, so a case-insensitive one is written
// regexp.MustCompile(`(?i)name`).
func Match(pattern *regexp.Regexp, subject string) string {
	m := pattern.FindStringSubmatch(subject)
	if m == nil {
		return ""
	}
	if len(m) > 1 {
		return m[1]
	}
	return m[0]
}

// MatchAll returns the first capture group of every match, or every whole match
// when the pattern captures nothing. The slice is empty rather than nil when
// nothing matched.
func MatchAll(pattern *regexp.Regexp, subject string) []string {
	ms := pattern.FindAllStringSubmatch(subject, -1)
	out := make([]string, 0, len(ms))
	for _, m := range ms {
		if len(m) > 1 {
			out = append(out, m[1])
			continue
		}
		out = append(out, m[0])
	}
	return out
}

// IsMatch reports whether the value matches any one of the patterns.
func IsMatch(patterns []*regexp.Regexp, value string) bool {
	for _, pattern := range patterns {
		if pattern.MatchString(value) {
			return true
		}
	}
	return false
}

// ReplaceMatches replaces at most limit matches of the pattern, and every one
// of them when limit is negative.
//
// The replacement expands a capture group as $1 or ${name}. A limit of zero
// replaces nothing.
func ReplaceMatches(pattern *regexp.Regexp, replace, subject string, limit int) string {
	if limit < 0 {
		return pattern.ReplaceAllString(subject, replace)
	}
	if limit == 0 {
		return subject
	}
	done := 0
	return pattern.ReplaceAllStringFunc(subject, func(match string) string {
		if done >= limit {
			return match
		}
		done++
		idx := pattern.FindStringSubmatchIndex(match)
		return string(pattern.ExpandString(nil, replace, match, idx))
	})
}
