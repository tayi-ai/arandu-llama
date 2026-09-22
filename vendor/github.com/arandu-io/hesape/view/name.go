package view

import "strings"

const hintPathDelimiter = "::"

// ViewName normalizes a view name: dots become directory separators, and
// namespaced views (package::view) keep the delimiter.
type ViewName struct{}

// Normalize returns a view name in dot notation.
//
//	ViewName{}.Normalize("posts/index")       -> "posts.index"
//	ViewName{}.Normalize("errors::404")       -> "errors::404"
//	ViewName{}.Normalize("errors/404")         -> "errors.404"
func (ViewName) Normalize(name string) string {
	if strings.Contains(name, hintPathDelimiter) {
		// package::partial.name -> package::partial/name
		at := strings.Index(name, hintPathDelimiter)
		ns := name[:at]
		rest := name[at+len(hintPathDelimiter):]
		return ns + hintPathDelimiter + strings.ReplaceAll(rest, "/", ".")
	}
	return strings.ReplaceAll(name, "/", ".")
}
