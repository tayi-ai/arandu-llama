package matching

import "net/http"

// UriValidator validates the request path against the route's pattern.
//
// There is no compiled route here -- the pattern is matched by the same code
// the table's match uses -- so the validator asks the route, with the method
// left out: the method is MethodValidator's question.
type UriValidator struct{}

// Matches reports whether req's path matches route's pattern.
func (v UriValidator) Matches(route Route, req *http.Request) bool {
	if req == nil {
		return false
	}
	return route.Matches(req, false)
}
