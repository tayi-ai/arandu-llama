package matching

import (
	"net/http"
	"strings"
)

// HostValidator validates that the incoming request's host matches the
// route's domain constraint. When the route declares no domain, every host
// matches.
type HostValidator struct{}

// Matches reports whether route's domain constraint accepts req's host.
func (v HostValidator) Matches(route Route, req *http.Request) bool {
	domain := route.GetDomain()
	if domain == "" {
		return true
	}
	if req == nil {
		return false
	}
	return strings.EqualFold(domain, hostOf(req))
}

// hostOf returns the request host without its port.
func hostOf(req *http.Request) string {
	host := req.Host
	if host == "" && req.URL != nil {
		host = req.URL.Host
	}
	if i := strings.LastIndexByte(host, ':'); i >= 0 && !strings.Contains(host[i:], "]") {
		host = host[:i]
	}
	return host
}
