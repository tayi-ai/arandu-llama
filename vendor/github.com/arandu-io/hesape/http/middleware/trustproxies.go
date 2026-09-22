package middleware

import (
	"net/http"
	"net/netip"
	"strings"

	hhttp "github.com/arandu-io/hesape/http"
)

// TrustProxies rewrites the request with what the proxy in front says, and only
// when the request actually came from that proxy.
//
// Behind a load balancer every request arrives from the balancer, so RemoteAddr
// is the balancer on every one of them. Anything keyed by address -- a rate
// limit, a throttle on sign-in, a line in a log -- then treats the whole
// internet as one visitor: the first person to be throttled throttles everybody.
// This is what puts the real address back.
//
// It is a whitelist and never a heuristic. X-Forwarded-For is a header, which
// means it is a string the client chose, and a middleware that believes it
// unconditionally has handed every visitor the ability to be any address they
// like -- including somebody else's, which is how a throttle becomes a way to
// lock another account out. Nothing is read unless the peer is inside one of the
// trusted prefixes, so with an empty list this middleware does nothing at all,
// which is the right behaviour for a process listening on the internet directly.
//
//	middleware.TrustProxies([]netip.Prefix{netip.MustParsePrefix("10.0.0.0/8")})
//
// # Which hop is the visitor
//
// X-Forwarded-For is appended to by each proxy, so it reads
// client, proxy1, proxy2 and the rightmost entry is the nearest hop. It is
// walked from the right, skipping entries that are themselves trusted, and the
// first one that is not is the visitor: everything to the left of it was written
// by whoever that is, and is worth nothing. A chain that is trusted all the way
// down leaves the leftmost entry, which is the only candidate left.
//
// # What it does not take
//
// X-Forwarded-Host. The Host is what routes the request and what a signed URL is
// checked against, and rewriting it from a header is how a cache is poisoned and
// how a password-reset link is sent pointing at somebody else's server. A
// deployment that terminates on a different name configures that name; it does
// not read it off the request.
//
// It writes RemoteAddr without a port -- the proxy's port belonged to a
// connection this request did not arrive on -- which hhttp.Context.IP is written
// to accept.
func TrustProxies(trusted []netip.Prefix) hhttp.Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			peer, ok := addrOf(r.RemoteAddr)
			if !ok || !within(trusted, peer) {
				next.ServeHTTP(w, r)
				return
			}

			if chain := forwardedFor(r.Header.Values("X-Forwarded-For"), peer, trusted); len(chain) > 0 {
				// Mutating the request rather than cloning it: the server made
				// this one for this request and nothing else holds it, and a
				// clone per request to change one field is a copy of every
				// header on every request.
				r.RemoteAddr = chain[0].String()

				// The whole chain goes on the context, because this is the only
				// place that knows which hops are ours and hhttp.Request.IPs
				// has no way to work it out later. Attaching it does mean a
				// context value per proxied request, which is why it is one
				// slice and not one value per hop.
				addresses := make([]string, len(chain))
				for i, addr := range chain {
					addresses[i] = addr.String()
				}
				r = r.WithContext(hhttp.WithForwardedFor(r.Context(), addresses))
			}
			if scheme := forwardedProto(r.Header.Get("X-Forwarded-Proto")); scheme != "" {
				// URL.Scheme is empty on a server-side request, so this is a
				// field being filled rather than one being overwritten. It is
				// what Context.FullURL reads.
				r.URL.Scheme = scheme
			}

			next.ServeHTTP(w, r)
		})
	}
}

// forwardedFor works the X-Forwarded-For chain out, nearest hop first: the
// visitor is the first entry and the rest is what the header claimed came
// before them.
//
// The header may arrive more than once, and the values concatenate in order, so
// they are read as one list. An entry that does not parse ends the walk: past it
// the chain cannot be reasoned about, and the nearest thing that is certainly
// real is the hop to its right. Nothing is returned in that case, and the peer
// stands.
//
// The peer completes the chain, our own proxies come out of it -- they are
// infrastructure and not visitors -- and what is left is reversed, nearest
// hop first. It used to return the visitor alone, which is all this
// middleware needed and half of what hhttp.Request.IPs needed.
func forwardedFor(values []string, peer netip.Addr, trusted []netip.Prefix) []netip.Addr {
	var chain []netip.Addr
	for _, value := range values {
		for _, part := range strings.Split(value, ",") {
			addr, ok := addrOf(strings.TrimSpace(part))
			if !ok {
				return nil
			}
			chain = append(chain, addr)
		}
	}
	if len(chain) == 0 {
		return nil
	}
	chain = append(chain, peer)

	// Walk from the nearest hop outwards, dropping every entry that is one of
	// ours wherever it sits -- a hop inside the trusted prefixes is
	// infrastructure and not a visitor, and a client that writes one of our
	// addresses into the header does not become one by saying so.
	var out []netip.Addr
	for i := len(chain) - 1; i >= 0; i-- {
		if !within(trusted, chain[i]) {
			out = append(out, chain[i])
		}
	}
	if len(out) == 0 {
		// Trusted the whole way down. The leftmost entry is the only candidate
		// left, and it is the one the first proxy wrote.
		return []netip.Addr{chain[0]}
	}
	return out
}

// forwardedProto reads the scheme the browser used, or empty.
//
// The leftmost value is the client-facing hop, for the same reason the rightmost
// address is the nearest one. Anything that is not one of the two schemes a
// browser speaks is dropped rather than passed along: it ends up in a link.
func forwardedProto(value string) string {
	if value == "" {
		return ""
	}
	if comma := strings.IndexByte(value, ','); comma >= 0 {
		value = value[:comma]
	}
	switch scheme := strings.ToLower(strings.TrimSpace(value)); scheme {
	case "http", "https":
		return scheme
	default:
		return ""
	}
}

// addrOf parses an address that may or may not carry a port.
//
// RemoteAddr always has one and a forwarded entry usually does not, and an
// IPv4-mapped IPv6 address is unmapped so that 10.0.0.1 and ::ffff:10.0.0.1
// answer the same question the same way.
func addrOf(raw string) (netip.Addr, bool) {
	if raw == "" {
		return netip.Addr{}, false
	}
	if ap, err := netip.ParseAddrPort(raw); err == nil {
		return ap.Addr().Unmap(), true
	}
	addr, err := netip.ParseAddr(raw)
	if err != nil {
		return netip.Addr{}, false
	}
	return addr.Unmap(), true
}

// within reports whether an address is inside any of the prefixes.
func within(prefixes []netip.Prefix, addr netip.Addr) bool {
	for _, prefix := range prefixes {
		if prefix.Contains(addr) {
			return true
		}
	}
	return false
}
