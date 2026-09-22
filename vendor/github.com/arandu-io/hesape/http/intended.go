package http

import (
	stdhttp "net/http"
	"time"

	"github.com/arandu-io/hesape/encryption"
)

// IntendedCookieName carries the address somebody was going to, between the
// request a guard turned away and the sign-in that follows it.
//
// Fixed, for the reason session.CookieName is fixed: the guard writes it and the
// sign-in handler reads it, and two parts of a project that disagree about the
// name is a person who signs in and lands on the front page with no explanation
// -- which is the thing this exists to remove.
//
// It is a cookie and not a row: there is no session yet at the moment the guard
// fires, which is exactly why it fires, so there is nowhere on the server to put
// it that is keyed to this browser.
const IntendedCookieName = "arandu_intended"

// IntendedLifetime is how long the address is worth keeping.
//
// Long enough to find a password in a manager and type it, and short enough that
// it does not survive to an unrelated sign-in: on a shared machine, an address
// kept for a day sends the next person who signs in to the page the previous one
// was refused.
const IntendedLifetime = 10 * time.Minute

// intendedPurpose binds the signature, so a token minted for a verification
// link cannot be pasted in here and turned into a redirect. See encryption.Signer.
const intendedPurpose = "session.intended"

// Intended is where somebody was going when a guard turned them away.
//
// It lives with the rest of what answers a request rather than with the
// session: the value it carries is an address, the whole of its correctness
// is [LocalPath], and hesape/session never validates a URL.
//
// It holds no state of its own -- the address is in a signed cookie in the
// browser -- so one value serves every request, and it is wired once at boot
// beside the Flash and the session store, over the same application key.
type Intended struct {
	signer *encryption.Signer
	secure bool
}

// NewIntended returns an Intended over the application key.
//
// The same key as the session, the flash and the signed links, because they are
// the same secret: an attacker who has it does not need four. Pass secure=false
// only in development -- without the Secure attribute the cookie travels over
// plain HTTP, and it carries where somebody was going.
func NewIntended(appKey []byte, secure bool) *Intended {
	return &Intended{signer: encryption.NewSigner(appKey), secure: secure}
}

// Remember records where this request was going, so that the sign-in screen it
// is about to be sent to can finish the journey.
//
// The guard is the only thing that knows what the person was reaching for,
// and by the time they have typed a password that request is gone. Without
// it every sign-in ends at the front page, and somebody who followed a link
// to one invoice has to find it again.
//
// # Why it is signed rather than merely same-site
//
// The value decides where a browser goes immediately after authenticating, so
// whoever can write it can choose where every person in the application lands.
// SameSite=Strict stops another SITE from setting it, and does not stop another
// HOST on the same registrable domain: a cookie set on ".example.com" by
// anything holding a subdomain -- a customer's CNAME, a forgotten staging box, a
// vendor's status page -- arrives here indistinguishable from ours. An HMAC does
// stop it, because the attacker does not have the application key, and it comes
// with the expiry signed into the same bytes so a stale address cannot be
// replayed by keeping the cookie alive.
//
// The destination is checked for being local anyway, on the way in and on the
// way out, because a signature only proves that WE wrote the value.
//
// # What it declines to remember
//
// Only a GET, because the address is replayed by a browser navigation: a POST
// remembered here is a form submission turned into a link, which either answers
// 405 or, on a route that also accepts GET, performs something the person did
// not ask for a second time.
//
// Only a whole page, never an HTMX fragment. A hx-get that is refused would
// otherwise be remembered as "/inbox/rows?page=2", and after signing in the
// person lands on that partial: no layout, no navigation, and it reads as a
// broken deploy. A boosted navigation is a page and is kept -- hx-boost is how
// most links in this stack are followed, so dropping it would drop nearly
// everything.
//
// It writes nothing when there is nothing worth writing, so a caller has no
// branch to get wrong.
func (i *Intended) Remember(w stdhttp.ResponseWriter, r *stdhttp.Request) {
	if r.Method != stdhttp.MethodGet {
		return
	}
	if r.Header.Get("HX-Request") == "true" && r.Header.Get("HX-Boosted") != "true" {
		return
	}
	to, ok := LocalPath(r.URL.RequestURI())
	if !ok {
		return
	}

	stdhttp.SetCookie(w, &stdhttp.Cookie{
		Name:     IntendedCookieName,
		Value:    i.signer.Sign(intendedPurpose, to, IntendedLifetime),
		Path:     "/",
		MaxAge:   int(IntendedLifetime.Seconds()),
		HttpOnly: true,
		Secure:   i.secure,
		// Lax and not Strict: the whole point is to be readable on the request
		// that follows a sign-in, and every one of those is same-site anyway.
		// The signature, not the attribute, is what makes the value trustworthy.
		SameSite: stdhttp.SameSiteLaxMode,
	})
}

// Take returns the address Remember stored, and clears it.
//
// It is meant to be the whole of a sign-in handler's last line:
//
//	return ctx.Redirect(intended.Take(ctx.Response, ctx.Request, "/"))
//
// The fallback is a parameter rather than a constant because it is the one part
// that genuinely differs -- a blog sends people to the front page, an
// application to its dashboard -- and taking it here means no caller has to
// write the branch for "there was nowhere in particular".
//
// # It answers the fallback for anything it cannot prove
//
// A forged or foreign signature, an expired one, a value that is not a local
// address: all of them are somebody's redirect that is not ours, and none of
// them is worth telling the person about at the moment they have just signed in.
// The check that the address is local is what keeps this from being an open
// redirect -- "sign in and then continue to https://evil.example/login" is the
// oldest phishing link there is, and it belongs here rather than in each
// project's handler precisely because every project would otherwise have to
// remember it.
//
// It is consumed, not read: the cookie is cleared whatever it said, so the
// address is used once. An address left standing is one a person meets again at
// the next sign-in from this browser, weeks later, with no idea why.
func (i *Intended) Take(w stdhttp.ResponseWriter, r *stdhttp.Request, fallback string) string {
	c, err := r.Cookie(IntendedCookieName)
	if err != nil {
		// No cookie, so nothing to clear: writing the clearing header anyway
		// would put a Set-Cookie on every sign-in in the application, including
		// the ones nobody was ever turned away from.
		return fallback
	}

	i.Clear(w)

	payload, err := i.signer.Verify(intendedPurpose, c.Value)
	if err != nil {
		return fallback
	}
	// Checked again, having already been checked before it was stored. The
	// signature proves this application wrote it, and says nothing about what
	// this application wrote a year and three refactors ago -- and this is the
	// side that turns a string into a Location header.
	to, ok := LocalPath(payload)
	if !ok {
		return fallback
	}
	return to
}

// Clear tells the browser to drop the address.
//
// One function and not two copies of a cookie literal, because the two copies
// have to agree on Path: a clearing cookie written for a different path does not
// replace the one that is there, and the address quietly survives the thing that
// was supposed to spend it.
//
// [Intended.Take] calls it, which spends the address. It is exported for the
// other caller, which is signing out: the address outlives a session by up to
// [IntendedLifetime], and a shared machine changes hands at exactly that moment
// -- the next person to sign in would be carried to the page the previous one
// was refused. A sign-out handler calls it beside session.Invalidate.
func (i *Intended) Clear(w stdhttp.ResponseWriter) {
	stdhttp.SetCookie(w, &stdhttp.Cookie{
		Name:     IntendedCookieName,
		Value:    "",
		Path:     "/",
		MaxAge:   -1,
		HttpOnly: true,
		Secure:   i.secure,
		SameSite: stdhttp.SameSiteLaxMode,
	})
}
