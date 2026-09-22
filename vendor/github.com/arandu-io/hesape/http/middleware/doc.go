// Package middleware holds the standard middleware every application
// wires: CORS, security headers, body-size limits, host and proxy trust,
// cache validation, and the method a form could not send for itself.
//
// A middleware here is a func(http.Handler) http.Handler, because that is
// the shape net/http composes. Configuration that a bootstrap step sets
// once -- which proxies or hosts to trust, which requests to skip -- lives
// in state.go, as package-level functions rather than fields on an
// instance.
//
// # There is no preload-hint middleware
//
// A Link: rel=preload header naming preloaded assets would need a build
// manifest listing them. There is no such manifest here: assets are
// embedded with go:embed and served content-addressed, with the hash
// already in the path.
//
// An earlier version existed anyway, as a middleware that took a limit and
// passed the request through untouched, with a doc comment claiming it
// added Link headers. That is worse than not having it: nothing fails, the
// header never appears, and the reason is invisible to whoever wired it in.
// It was removed, and the reasoning kept here instead.
//
// If a preload hint is wanted later, it is one header on a response whose
// asset URLs the renderer already knows, and it is not middleware shaped.
// It is also not motivated by HTTP/2 server push: Chrome removed push in
// 2022.
//
// # There was a second TrustProxies, and it trusted everything
//
// middleware.go carried TrustProxiesHTTP alongside [TrustProxies], with no
// callers and no tests. It is deleted: keeping two implementations of the
// same middleware is a problem on its own, and this pair was worse than
// that. The one that stays takes []netip.Prefix, and the one that went took
// []string and compared them against r.RemoteAddr -- which carries the
// port, so a trusted "10.0.0.1" never matched "10.0.0.1:54321" and every
// proxy was untrusted. It also accepted "*", which trusts any peer at all:
// with it set, any client on the internet could send X-Forwarded-For and be
// recorded as whatever address it liked. And it assigned the whole header
// value to RemoteAddr, so a chained "a, b, c" became the remote address
// verbatim.
//
// A middleware that is wrong in the trusting direction is the kind nobody
// notices, because everything keeps working.
package middleware
