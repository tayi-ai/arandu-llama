package matching

import (
	"net/http"
	"strings"
)

// MethodValidator validates that the incoming request's method is one of the
// route's accepted methods.
type MethodValidator struct{}

// Matches reports whether req's method is one route accepts.
func (v MethodValidator) Matches(route Route, req *http.Request) bool {
	if req == nil {
		return false
	}
	methods := route.Methods()
	if len(methods) == 0 {
		// A route registered without a method answers every method, which is
		// what the mux does with a pattern that names none.
		return true
	}
	for _, m := range methods {
		if strings.EqualFold(m, req.Method) || m == "ANY" {
			return true
		}
	}
	return false
}
