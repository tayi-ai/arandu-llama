package routing

import (
	"fmt"
	"net/http"
	"net/url"
	"time"

	"github.com/arandu-io/hesape/encryption"
)

// signatureParam is the query parameter a signed URL carries the token in.
const signatureParam = "signature"

// signaturePurpose is what every signed route URL is signed with.
//
// The Signer binds the purpose into the MAC, so a token issued for a route
// cannot be replayed as a password-reset token even though both are signed with
// the application key. See encryption.Signer.
const signaturePurpose = "routing.signed"

// SignedRoute builds the URL of a named route with a signature on it, valid for
// ttl.
//
//	SignedRoute(r.Table(), signer, "unsubscribe", 30*24*time.Hour, list.ID)
//
// It is what an unsubscribe link and a verification link are made of. The
// alternative -- a table of tokens -- costs a write, a read, a cleanup job and a
// decision about what happens when the row is gone; this costs a signature, and
// a link that has run out says so.
//
// The signature covers the whole address, not just the parameters: appending
// anything to the query string of a signed link invalidates it, so a route
// behind ValidateSignature cannot be handed a value its author did not sign.
//
// It signs and it does not verify: middleware.ValidateSignature is the other
// half, and the Signer itself is in hesape/encryption, which is where a leaf
// primitive belongs. Building a signed URL is routing's, because the address
// being signed is a route.
func SignedRoute(t *Routes, s *encryption.Signer, name string, ttl time.Duration, params ...string) (string, error) {
	path, err := t.Route(name, params...)
	if err != nil {
		return "", err
	}
	u := &url.URL{Path: path}
	token := s.Sign(signaturePurpose, signaturePayload(u), ttl)

	q := u.Query()
	q.Set(signatureParam, token)
	u.RawQuery = q.Encode()
	return u.String(), nil
}

// VerifySignature reports whether a request arrived on a URL this application
// signed, and that the signature has not run out.
//
// It returns an error that unwraps to encryption.ErrSignature, and to
// encryption.ErrExpired when the only thing wrong is the time -- which is the
// one failure a person can act on, because "ask for another link" is a
// different sentence from "this link is not ours".
//
// It is exported alongside the middleware because a handler that is reached
// some other way -- a webhook that also accepts an unsigned call from an
// authenticated operator -- needs the same check, and a second reading of the
// query parameter is a second place for the canonicalisation to be missing.
func VerifySignature(s *encryption.Signer, r *http.Request) error {
	token := r.URL.Query().Get(signatureParam)
	if token == "" {
		return fmt.Errorf("%w: the link carries no signature", encryption.ErrSignature)
	}
	signed, err := s.Verify(signaturePurpose, token)
	if err != nil {
		return err
	}
	// The token proves that this application signed the address inside it. It
	// does not prove that the address inside it is the one being requested, and
	// without this comparison a signature issued for one route would open every
	// route the middleware guards.
	//
	// A plain comparison and not hmac.Equal: both sides are already
	// authenticated by the step above, and neither is a secret.
	if signed != signaturePayload(r.URL) {
		return fmt.Errorf("%w: it was issued for a different address", encryption.ErrSignature)
	}
	return nil
}

// signaturePayload is the string a signed URL signs: the path, plus every query
// parameter except the signature itself, in a canonical order.
//
// Sorted, because url.Values.Encode sorts by key and both halves have to agree
// on one byte string; without the signature, because it cannot cover itself.
func signaturePayload(u *url.URL) string {
	q := u.Query()
	q.Del(signatureParam)
	if len(q) == 0 {
		return u.Path
	}
	return u.Path + "?" + q.Encode()
}
