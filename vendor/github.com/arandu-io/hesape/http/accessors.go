package http

import (
	"net"
	stdhttp "net/http"
	"strings"

	"github.com/arandu-io/hesape/auth"
)

// Header reads a request header, canonicalising the name like net/http does.
func (c *Context) Header(name string) string { return c.Request.Header.Get(name) }

// Path is the path of the request, without the query string.
func (c *Context) Path() string { return c.Request.URL.Path }

// Method is the HTTP method, upper case: "GET", "POST".
func (c *Context) Method() string { return c.Request.Method }

// FullURL is the address this request was made to, scheme and host included.
//
// It is what a mail template needs, and what nothing that answers HTML needs: a
// link inside a page is a path, so that the application keeps working behind a
// proxy, on a staging host and on somebody's laptop.
//
// The scheme is the one the browser used, which behind a proxy is not the one
// this process is listening on. middleware.TrustProxies is what records it; with
// no proxy in front, r.TLS answers it directly. A request whose scheme neither
// of those two can prove is reported as http, because guessing https for a
// listener that is plain is how a mail link becomes unreachable.
func (c *Context) FullURL() string {
	scheme := c.Request.URL.Scheme
	if scheme == "" {
		scheme = "http"
		if c.Request.TLS != nil {
			scheme = "https"
		}
	}
	return scheme + "://" + c.Request.Host + c.Request.URL.RequestURI()
}

// IP is the address the request came from.
//
// Behind a proxy it is whatever middleware.TrustProxies wrote onto RemoteAddr,
// and with no proxy it is the peer. Read it and do not parse a forwarding header
// here: which hop to believe is a deployment fact, it is decided once in that
// middleware, and a second reading of X-Forwarded-For is a second answer -- the
// one an attacker gets to choose.
//
// It answers the address without the port, and tolerates a RemoteAddr that has
// none, because TrustProxies writes a bare address: a port that belonged to a
// connection this request did not arrive on is worse than no port.
func (c *Context) IP() string {
	addr := c.Request.RemoteAddr
	if host, _, err := net.SplitHostPort(addr); err == nil {
		return host
	}
	return addr
}

// BearerToken is the token of an Authorization: Bearer header, or empty.
//
// The scheme is compared case-insensitively because RFC 9110 says it is, and
// the value is trimmed because a client that sends two spaces is a client, not
// an attack.
func (c *Context) BearerToken() string {
	const scheme = "bearer "

	v := c.Request.Header.Get("Authorization")
	if len(v) < len(scheme) || !strings.EqualFold(v[:len(scheme)], scheme) {
		return ""
	}
	return strings.TrimSpace(v[len(scheme):])
}

// Cookie is the value of a cookie, or empty when the browser sent none.
//
// Empty rather than (string, bool): every caller of the two-value form in the
// framework threw the bool away, and a cookie that is present and empty means
// the same thing to all of them as a cookie that is absent.
func (c *Context) Cookie(name string) string {
	got, err := c.Request.Cookie(name)
	if err != nil {
		return ""
	}
	return got.Value
}

// WithCookie writes a cookie on the answer and returns the Context, so that the
// line that sets one reads as one line:
//
//	return ctx.WithCookie(pref).Redirect("/settings")
//
// It is a plain stdhttp.Cookie and not a struct of this package's own: the standard
// library's has every attribute there is, and a wrapper would have to be taught
// each new one.
func (c *Context) WithCookie(cookie *stdhttp.Cookie) *Context {
	if cookie != nil {
		stdhttp.SetCookie(c.Response, cookie)
	}
	return c
}

// IsHTMX reports whether htmx made this request.
//
// It is the question that decides whether an answer is a fragment or a page, and
// it is asked in one place per decision: Redirect and Refuse already ask it for
// themselves, so a handler needs this only when the two shapes differ in what
// they render.
func (c *Context) IsHTMX() bool { return c.Request.Header.Get("HX-Request") == "true" }

// WantsJSON reports whether this request asked for JSON rather than a page.
//
// htmx is deliberately excluded even though it sends X-Requested-With: it swaps
// HTML, so answering it with JSON puts a JSON document inside a div.
//
// hesape/exception states the same rule as the default of its Config.RenderJSONWhen,
// because it must answer a failure on a request that never reached a Context.
// The two are the same three lines and must not drift; folding them into one is
// waiting on the layer that wires the handler.
func (c *Context) WantsJSON() bool {
	r := c.Request
	if r.Header.Get("HX-Request") == "true" {
		return false
	}
	if strings.Contains(r.Header.Get("Accept"), "application/json") {
		return true
	}
	return r.Header.Get("X-Requested-With") == "XMLHttpRequest"
}

// User is who is signed in, and whether anybody is.
//
// It reads the subject the authentication middleware put on the context, and it
// is the only thing a controller may ask about identity. It is deliberately not
// a Grant: a Grant is minted by a policy for one action on one thing, and a
// controller that could produce one would be a controller that authorises
// itself.
func (c *Context) User() (auth.Subject, bool) { return auth.SubjectFrom(c.Ctx()) }
