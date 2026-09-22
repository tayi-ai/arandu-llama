package concerns

import (
	"regexp"
	"strings"
)

// searchPathToken matches anything that is not whitespace, a comma or a
// quote.
var searchPathToken = regexp.MustCompile(`[^\s,"']+`)

// ParseSearchPath reads Postgres's search_path option, which arrives as either a
// string or a list.
//
//	"public,reporting"        ->  ["public", "reporting"]
//	`"public", 'reporting'`   ->  ["public", "reporting"]
//	[]string{"public"}        ->  ["public"]
//
// It takes any because the value comes out of a configuration file where either
// shape is legal. A value of any other type answers an empty list.
func ParseSearchPath(searchPath any) []string {
	var parts []string

	switch v := searchPath.(type) {
	case nil:
		return []string{}
	case string:
		parts = searchPathToken.FindAllString(v, -1)
	case []string:
		parts = v
	case []any:
		for _, item := range v {
			if s, ok := item.(string); ok {
				parts = append(parts, s)
			}
		}
	default:
		return []string{}
	}

	out := make([]string, 0, len(parts))
	for _, schema := range parts {
		// Quotes are trimmed off each entry after the split, because the
		// quoted list form arrives with them still attached.
		out = append(out, strings.Trim(schema, `'"`))
	}
	return out
}
