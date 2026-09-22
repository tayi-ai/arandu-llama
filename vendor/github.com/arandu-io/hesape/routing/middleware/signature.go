package middleware

import (
	"errors"
	"net/http"

	"github.com/arandu-io/hesape/encryption"
	"github.com/arandu-io/hesape/pipeline"
	"github.com/arandu-io/hesape/routing"
)

// ValidateSignature refuses a request whose URL this application did not sign,
// or signed for a different address, or signed too long ago.
//
//	signed := r.Group(routing.Group{
//		Prefix:     "/unsubscribe",
//		Middleware: []pipeline.Middleware[http.Handler]{
//			middleware.ValidateSignature(signer, http.Refuse),
//		},
//	})
//
// It is the other half of routing.SignedRoute, and it is a middleware rather
// than a line in each handler for the reason every guard is: a check that has
// to be remembered is a check that will be missing on the route added next
// month, and a signed link that nobody verifies is an unauthenticated endpoint
// with a long URL.
//
// # Two answers, because they are two different things to a person
//
// An expired link answers 403 with a sentence saying it has run out, which is
// something the person can act on: ask for another one. Everything else --
// no signature, a forged one, one issued for a different address -- answers 403
// saying the link is not valid, and deliberately says no more: which of the
// three it was is information about the key, and it goes in the log rather than
// on the page.
//
// 403 and not 401: nothing about the account is wrong and signing in changes
// nothing.
func ValidateSignature(s *encryption.Signer, refuse Refuse) pipeline.Middleware[http.Handler] {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			err := routing.VerifySignature(s, r)
			switch {
			case err == nil:
				next.ServeHTTP(w, r)
			case errors.Is(err, encryption.ErrExpired):
				refuse(w, r, http.StatusForbidden,
					"this link has expired: ask for a new one")
			default:
				refuse(w, r, http.StatusForbidden,
					"this link is not valid")
			}
		})
	}
}
