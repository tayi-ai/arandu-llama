package routing

import (
	"fmt"
	"net/http"
	"strconv"
	"time"
)

// Deprecated marks the route as deprecated since the given instant and due to
// stop answering at sunset. It returns the route so the call chains, the same
// shape Name returns.
//
//	r.Get("/v1/invoices", h).Name("v1.invoices").
//		Deprecated(announced, announced.AddDate(1, 0, 0))
//
// Every response the route produces then carries two header fields: the
// Deprecation field of RFC 9745, holding the instant the route was deprecated,
// and the Sunset field of RFC 8594, holding the instant it stops answering. A
// client that reads either one learns the route is going away from the route
// itself, rather than from an announcement somebody has to remember to send.
// Route introspection prints the sunset date alongside the route.
//
// Both instants are required, and a sunset before the deprecation date panics.
// The two fields would otherwise contradict each other on the wire -- RFC 9745
// forbids exactly that pair -- and a client that catches a server contradicting
// itself is a client that stops reading both fields. It panics at registration,
// where the rest of this package refuses a declaration it cannot honour, so the
// contradiction is found at boot rather than by whoever was relying on the
// dates.
//
// The deprecation date is an argument rather than the moment of this call
// because the moment of this call is process start: it would move forward on
// every deploy, and a client tracking when the route was deprecated would watch
// the answer change without anything having been decided.
func (rt *Route) Deprecated(since, sunset time.Time) *Route {
	if rt == nil {
		return nil
	}
	if since.IsZero() || sunset.IsZero() {
		panic("routing: Deprecated needs both the deprecation date and the sunset date")
	}
	if sunset.Before(since) {
		panic(fmt.Sprintf("routing: %s %s sunsets at %s, before it was deprecated at %s",
			rt.Method, rt.Pattern,
			sunset.UTC().Format(time.RFC3339), since.UTC().Format(time.RFC3339)))
	}
	rt.deprecatedSince = since
	rt.sunset = sunset
	for _, s := range rt.siblings {
		s.deprecatedSince = since
		s.sunset = sunset
	}
	return rt
}

// IsDeprecated reports whether Deprecated was called on the route.
func (rt *Route) IsDeprecated() bool {
	return rt != nil && !rt.sunset.IsZero()
}

// GetDeprecation returns the two instants Deprecated set: when the route was
// deprecated, and when it stops answering. Both are zero on a route that is
// not deprecated.
func (rt *Route) GetDeprecation() (since, sunset time.Time) {
	if rt == nil {
		return time.Time{}, time.Time{}
	}
	return rt.deprecatedSince, rt.sunset
}

// writeDeprecationHeaders puts the Deprecation and Sunset fields on the
// response of a deprecated route, and does nothing for any other route.
//
// It runs before the handler, because a handler that writes a status freezes
// the header block and anything set afterwards never leaves. The route is the
// only thing that knows these dates, so the headers are written here rather
// than left to a middleware every deprecated route would have to remember to
// attach.
func (rt *Route) writeDeprecationHeaders(w http.ResponseWriter) {
	if !rt.IsDeprecated() || w == nil {
		return
	}
	// Deprecation is an item structured field holding a Date, which is an
	// at-sign followed by the instant in seconds. Sunset is an HTTP-date.
	w.Header().Set("Deprecation", "@"+strconv.FormatInt(rt.deprecatedSince.Unix(), 10))
	w.Header().Set("Sunset", rt.sunset.UTC().Format(http.TimeFormat))
}
