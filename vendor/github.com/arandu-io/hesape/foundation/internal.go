package foundation

import (
	"crypto/subtle"
	"net/http"
	"strings"

	"github.com/arandu-io/hesape/log"
	"github.com/arandu-io/hesape/pipeline"
)

// internalPrefix is what this framework mounts for itself: the health probe,
// the debug console, the development reload.
const internalPrefix = "/_arandu/"

// exceptInternal runs an application's middleware everywhere except on the
// framework's own routes.
//
// The framework's endpoints answer to the framework: the health probe must not
// be rate limited by the application it reports on, the debug console is gated
// by its own secret, and the reload is a question the page asks about the
// process rather than a request the visitor made. None of them touch the
// database, hold a session, or write anything.
//
// The prefix is the boundary because it already is one everywhere else -- one
// name for what belongs to the framework, checked in one place.
//
// It was not. The development reload asks once a second which process is
// answering, and that ran through the rate limit an application mounts for its
// own visitors -- 60 of a 300-per-minute budget per open tab, shared, because
// the key falls back to the address for a request with no session. Ordinary
// browsing with two tabs open answered "too many requests: wait 32 seconds", on
// a page nobody had hammered.
func exceptInternal(mw pipeline.Middleware[http.Handler]) pipeline.Middleware[http.Handler] {
	return func(next http.Handler) http.Handler {
		wrapped := mw(next)
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if strings.HasPrefix(r.URL.Path, internalPrefix) {
				next.ServeHTTP(w, r)
				return
			}
			wrapped.ServeHTTP(w, r)
		})
	}
}

// requireTracingSecret gates the console outside development.
//
// It answers 404 rather than 401, because a 401 confirms that the console is
// there. Somebody scanning for /_arandu/debug learns nothing from a 404 that
// they did not already know.
//
// The comparison is constant-time: a byte-by-byte one leaks the secret to
// anybody willing to measure, and this secret is the whole gate.
//
// The distinction it draws is between "a tracing secret is configured" and "the
// request carries it", and that used to be missing. The recorder exists
// whenever the secret is set, so the console routes were mounted in production,
// and the secret was checked only by the middleware that decides whether to
// RECORD. Anyone could then read the buffer: SQL with bound arguments, dumps,
// event payloads, across every tenant, with no session and no header. Found by
// audit, reproduced over a real socket.
func requireTracingSecret(secret string, next http.HandlerFunc) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		// An empty secret cannot authorize anything. It is also the zero value
		// of the configuration, so treating it as "no gate" would open the
		// console for every application that never set one.
		if secret == "" {
			http.NotFound(w, r)
			return
		}
		if subtle.ConstantTimeCompare([]byte(r.Header.Get(log.TracingHeader)), []byte(secret)) != 1 {
			http.NotFound(w, r)
			return
		}
		next(w, r)
	}
}
