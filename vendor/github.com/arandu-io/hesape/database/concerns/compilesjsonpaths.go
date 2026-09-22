package concerns

import (
	"regexp"
	"strings"
)

// Wrapper is the grammar method that wraps a value for its dialect, passed
// in as a callback.
//
// It is a function rather than an interface because it is one method, and
// an interface of one method that every grammar would satisfy anyway is
// ceremony.
type Wrapper func(value any) string

// WrapJSONFieldAndPath splits `options->language` into the wrapped column
// and the JSON path, so a grammar can put them either side of whatever its
// engine spells the extraction function.
//
// The path comes back with a leading ", " already on it, so a grammar
// concatenating the two gets a ready-made argument list.
func WrapJSONFieldAndPath(wrap Wrapper, column string) (field, path string) {
	parts := strings.SplitN(column, "->", 2)

	field = wrap(parts[0])

	if len(parts) > 1 {
		path = ", " + WrapJSONPath(parts[1], "->")
	}
	return field, path
}

// jsonPathQuote matches an apostrophe, escaped or not, which becomes the
// doubled apostrophe SQL wants inside a literal.
var jsonPathQuote = regexp.MustCompile(`\\*'`)

// WrapJSONPath turns `language->code` into the quoted
// '$."language"."code"' a JSON function takes.
func WrapJSONPath(value, delimiter string) string {
	value = jsonPathQuote.ReplaceAllString(value, "''")

	segments := strings.Split(value, delimiter)
	wrapped := make([]string, 0, len(segments))
	for _, segment := range segments {
		wrapped = append(wrapped, WrapJSONPathSegment(segment))
	}
	jsonPath := strings.Join(wrapped, ".")

	dot := "."
	if strings.HasPrefix(jsonPath, "[") {
		// An array subscript attaches to the root directly: '$[0]', never
		// '$.[0]'.
		dot = ""
	}
	return "'$" + dot + jsonPath + "'"
}

// jsonPathSubscript matches one or more array subscripts at the end of a
// segment.
var jsonPathSubscript = regexp.MustCompile(`(\[[^\]]+\])+$`)

// WrapJSONPathSegment quotes the key and leaves the array subscript outside
// the quotes, because '$."tags"[0]' selects an element and '$."tags[0]"'
// selects a key that does not exist.
func WrapJSONPathSegment(segment string) string {
	if parts := jsonPathSubscript.FindString(segment); parts != "" {
		key := strings.TrimSuffix(segment, parts)
		if key != "" {
			return `"` + key + `"` + parts
		}
		return parts
	}
	return `"` + segment + `"`
}
