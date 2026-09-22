package session

import (
	"net/http"
	"net/url"
	"sort"
	"strings"
	"time"

	"github.com/arandu-io/hesape/encryption"
)

// FlashCookieName carries the messages and the typed input of a rejected
// request, between the request that was rejected and the page it is sent back
// to.
//
// Fixed, for the reason CookieName is fixed: one half of the framework writes it
// and another half reads it, and two parts of a project that disagree about the
// name is a form that comes back with nothing on it -- which is the exact
// failure this exists to remove.
const FlashCookieName = "arandu_flash"

// FlashLifetime is how long the messages are worth keeping.
//
// The gap it has to cover is one redirect, which is milliseconds. Five minutes
// is slack for a browser that was closed mid-navigation and reopened, and short
// enough that a message cannot resurface on a page nobody submitted -- which is
// the failure that makes flashed state untrustworthy. The cookie is cleared on
// the read anyway; this is what covers the read that never happens.
const FlashLifetime = 5 * time.Minute

// MaxFlashBytes is the budget for the signed cookie value.
//
// A browser drops a cookie over about 4096 bytes SILENTLY -- no error, no
// warning, nothing in the log. Past that limit the flash is discarded by the
// browser and the screen says nothing again, which is the bug this whole path
// exists to fix, arrived at by a different route. So the budget is enforced
// here, where something can be done about it: see Flash.Write for what is given
// up first.
const MaxFlashBytes = 3500

// flashPurpose binds the signature, so a token minted to verify an e-mail
// address cannot be pasted in here and turned into messages on a page. See
// encryption.Signer.
const flashPurpose = "session.flash"

// maxFlashMessage is the longest single message worth carrying, in bytes.
//
// Messages are sentences written by the framework or by a rule set, not typed by
// a visitor, so this is not a limit anybody meets in normal use. It exists so
// that the trimming in Write terminates with something rather than with a cookie
// the browser throws away: one field carrying one message of at most this length
// always fits in MaxFlashBytes.
const maxFlashMessage = 512

// The two halves of the payload. A single url.Values carries both, with the key
// prefixed by which half it belongs to, because url.Values.Encode sorts its keys
// -- so the same errors and the same input always produce the same bytes, and a
// test can assert on the cookie instead of on a decoded copy of it.
const (
	flashErrorPrefix = "e:"
	flashOldPrefix   = "o:"
)

// neverFlashed are the field names whose VALUE never goes back in the browser.
//
// A name belongs on one of these two lists when putting its value back on the
// page would redisplay a credential, a one-time code, or a request token that is
// already stale by the time the redirect lands. The two ways of being wrong cost
// different amounts, and that is what decides how wide the rules are: a name
// caught by mistake comes back as an empty box somebody retypes, while a name
// missed comes back as a secret written into the HTML and into the old input a
// debug dump prints. So the rules are written to over-catch.
//
// This half holds the BARE names, and only those. An exact match is the one rule
// that says "the whole field is called this", which is what a bare name needs:
// matching the end of the name instead would take laptop for otp. A name with a
// qualifier in front of it belongs in neverFlashedSuffix, which is why
// current_password and new_password are not repeated here -- a name covered
// twice is a name no test can prove either rule still catches.
//
// secret is here although it is an ordinary English word, and the cost is real:
// a checkbox called is_secret comes back unticked. That is the cheap side of the
// trade above, and it is written down so the next reader knows the name was
// weighed rather than copied in.
//
// What is dropped is the value alone. The MESSAGES for these fields are flashed
// like any other -- an empty password box that does not say why it was rejected
// is the failure being fixed here, not the fix.
var neverFlashed = map[string]bool{
	"password":              true,
	"passwords":             true,
	"password_confirmation": true,
	"token":                 true,
	"tokens":                true,
	"otp":                   true,
	"secret":                true,
	"secrets":               true,

	// The single-use codes. Each one is worth the whole account by itself and
	// is spent once, so a single redrawn form that hands one back is the whole
	// of the damage -- and the screens carrying them are forms like any other,
	// rejected for a stale request token or for an empty box beside them.
	//
	// They are named one at a time rather than caught by a rule on the end of
	// the name, for the reason written on neverFlashedSuffix.
	//
	// code is bare because that is what a prompt asking for one calls its
	// field when it asks for nothing else. The cost is a checkout that redraws
	// its coupon box empty, which is the trade secret is on this list for.
	// The plurals are here because a screen that issues a set names the set.
	"code":              true,
	"recovery_code":     true,
	"recovery_codes":    true,
	"backup_code":       true,
	"backup_codes":      true,
	"two_factor_code":   true,
	"otp_code":          true,
	"mfa_code":          true,
	"verification_code": true,
	"confirmation_code": true,
}

// neverFlashedSuffix catches the same secrets behind a qualifier, which is the
// part of a field name whoever writes the form invents freely.
//
// current_password, owner_password and admin_password are one secret with three
// prefixes; access_token, api_token and refresh_token are one token with three.
// An exact list that names two of any such group looks complete while letting
// the third through, and it is a list somebody has to remember to extend. The
// CSRF field is called _token, so this rule covers it as well as csrf_token.
//
// Every entry is anchored on the underscore, and that anchor is what keeps a
// rule this wide off ordinary fields: otp without it matches laptop and desktop.
//
// # Why _code is not here
//
// Every rule above works because the word it ends on IS the credential:
// anything whose name ends in _password is a password, whoever wrote the form.
// code is not such a word. What decides whether a field called something_code
// holds a credential is the word in FRONT of it, and on that side the ordinary
// ones are the majority -- postal_code, country_code, area_code, status_code,
// currency_code, promo_code are all values somebody typed and expects to get
// back. A rule that empties a postcode box is a rule that gets removed within
// the week, and removing it would take the recovery code out with it.
//
// So the credential-headed compounds are named one at a time on both lists:
// the unqualified form in neverFlashed, and the qualified form here, the same
// pairing password and _password already are. That is a longer list than a
// suffix would be, and it is the length that makes it safe to keep.
var neverFlashedSuffix = []string{
	"_password",
	"_password_confirmation",
	"_token",
	"_otp",
	"_secret",
	"_recovery_code",
	"_recovery_codes",
	"_backup_code",
	"_backup_codes",
	"_two_factor_code",
	"_otp_code",
	"_mfa_code",
	"_verification_code",
	"_confirmation_code",
}

// Flash carries the messages and the input of a rejected request across the one
// redirect that follows it.
//
// It is a signed one-shot cookie rather than session state on purpose. The
// forms that need it most -- sign in, sign up, password reset -- are
// submitted by somebody who has no session at all, which is the same reason
// CSRF.Issue("") already accepts an empty session id. A flash on the session
// would work everywhere except on the three screens it was built for.
//
// It is not a bag of arbitrary messages either. What it carries is what a
// rejected request produced: the errors, and what was typed. A general-purpose
// "flash anything" would be a second way to get data onto a page, next to the
// controller filling a struct.
type Flash struct {
	signer *encryption.Signer
	secure bool
}

// NewFlash returns a Flash over the application key.
//
// The same key as the session, the CSRF token and the signed links, because
// they are the same secret: an attacker who has it does not need four. Pass
// secure=false only in development -- without the Secure attribute the cookie
// travels over plain HTTP, and it carries what somebody typed into a form.
func NewFlash(appKey []byte, secure bool) *Flash {
	return &Flash{signer: encryption.NewSigner(appKey), secure: secure}
}

// Write puts the messages and what was typed in the browser, for the page
// the rejected request is about to be redirected to, on a signed cookie
// instead of the session. A request with no session -- sign in, sign up,
// password reset -- would get nothing from session state, which is why
// this is a cookie.
//
// errs is map[string][]string rather than validation.Errors so that this package
// keeps importing nothing above it. validation.Errors IS that type and assigns
// to it directly.
//
// # What it gives up when it does not fit
//
// Everything below MaxFlashBytes is kept. Over it, in this order:
//
//  1. old input, longest value first. A field redisplayed empty is an
//     inconvenience; a field with no message on it is the bug.
//  2. all but the first message of each field. The first one is the one that
//     names what to change.
//  3. fields, from the end of the sorted order, until it fits.
//
// It writes nothing when there is nothing to write, so no caller needs the
// branch.
func (f *Flash) Write(w http.ResponseWriter, errs map[string][]string, old url.Values) {
	kept := redactFlash(old)
	if len(errs) == 0 && len(kept) == 0 {
		return
	}

	http.SetCookie(w, &http.Cookie{
		Name:     FlashCookieName,
		Value:    f.fit(errs, kept),
		Path:     "/",
		MaxAge:   int(FlashLifetime.Seconds()),
		HttpOnly: true,
		Secure:   f.secure,
		// Lax and not Strict, for the reason the intended address is Lax: the
		// whole point is to be readable on the navigation that follows, and the
		// signature rather than the attribute is what makes the value ours.
		SameSite: http.SameSiteLaxMode,
	})
}

// Take returns what Write stored and clears it, so it survives exactly one
// request.
//
// Cleared on the read rather than counted down like [Store.AgeFlashData],
// which keeps the data for the request that reads it AND ages it afterwards -- a
// second-request window in which a reload shows the message again. A message
// that outlives its redirect appears on a page nobody submitted, and the person
// reading it has no way to tell it is stale.
//
// ok is false, and nothing is cleared, when this request is not a page somebody
// is reading: an asset fetch, a POST, an HTMX fragment or a JSON call would
// spend the flash on a response that draws none of it, and the person would land
// on the form with nothing on it. That is the same failure as not having a flash
// at all, reached through a stylesheet.
//
// ok is false, and the cookie IS cleared, when the value is there and cannot be
// trusted -- forged, minted for another purpose, expired, or unreadable. There
// is nothing to show and no reason to keep it.
func (f *Flash) Take(w http.ResponseWriter, r *http.Request) (map[string][]string, url.Values, bool) {
	if !isPageRead(r) {
		return nil, nil, false
	}
	c, err := r.Cookie(FlashCookieName)
	if err != nil {
		// No cookie, so nothing to clear: writing the clearing header anyway
		// would put a Set-Cookie on every page view in the application.
		return nil, nil, false
	}

	f.clearFlash(w)

	payload, err := f.signer.Verify(flashPurpose, c.Value)
	if err != nil {
		return nil, nil, false
	}
	raw, err := url.ParseQuery(payload)
	if err != nil {
		return nil, nil, false
	}

	errs, old := decodeFlash(raw)
	if len(errs) == 0 && len(old) == 0 {
		return nil, nil, false
	}
	return errs, old, true
}

// isPageRead reports whether this request is a whole page somebody is about to
// read, which is the only kind of request worth spending a one-shot message on.
//
// A GET, because the flash is drawn by the page the redirect landed on and every
// redirect from a rejected form is followed with a GET. And an Accept that asks
// for HTML, which is what separates the navigation from the stylesheet, the
// image and the HTMX fragment that the same page fires on arrival: XHR sends
// */*, and a browser navigating sends text/html.
func isPageRead(r *http.Request) bool {
	if r.Method != http.MethodGet {
		return false
	}
	return strings.Contains(r.Header.Get("Accept"), "text/html")
}

// clearFlash tells the browser to drop the messages.
//
// One function and not a cookie literal at each call site, because the two
// copies have to agree on Path: a clearing cookie written for a different path
// does not replace the one that is there, and the message quietly survives the
// read that was supposed to spend it.
func (f *Flash) clearFlash(w http.ResponseWriter) {
	http.SetCookie(w, &http.Cookie{
		Name:     FlashCookieName,
		Value:    "",
		Path:     "/",
		MaxAge:   -1,
		HttpOnly: true,
		Secure:   f.secure,
		SameSite: http.SameSiteLaxMode,
	})
}

// redactFlash returns the input with every secret's value removed.
func redactFlash(old url.Values) url.Values {
	if len(old) == 0 {
		return nil
	}
	kept := make(url.Values, len(old))
	for field, values := range old {
		if IsSecretField(field) {
			continue
		}
		kept[field] = values
	}
	if len(kept) == 0 {
		return nil
	}
	return kept
}

// redactInput returns the input with every secret's value removed, at any
// depth.
//
// It answers for map[string]any what redactFlash answers for url.Values, and
// both ask IsSecretField -- one list, two shapes. The shapes differ because the
// two carriers do: a cookie is flat, and what a form hands the session is not.
//
// The depth is not caution. A JSON body arrives nested, Only builds a nested
// map out of a dotted key, and GetOldInput reads back by dot -- so a value
// under user.password is written and read exactly like one under password, and
// a rule that only looked at the top level would hand it back.
func redactInput(input map[string]any) map[string]any {
	out := make(map[string]any, len(input))
	for field, value := range input {
		if IsSecretField(field) {
			continue
		}
		out[field] = redactInputValue(value)
	}
	return out
}

// redactInputValue descends into the two shapes that hold further fields, and
// returns everything else as it stands. A list is rebuilt rather than reused so
// that redacting cannot write through into the caller's map.
func redactInputValue(value any) any {
	switch typed := value.(type) {
	case map[string]any:
		return redactInput(typed)
	case []any:
		list := make([]any, len(typed))
		for i, item := range typed {
			list[i] = redactInputValue(item)
		}
		return list
	}
	return value
}

// IsSecretField reports whether a field's value never goes back in the browser:
// the bare names matched whole, and the same secrets behind a qualifier matched
// on the end of the name. The name is compared lowercased.
//
// It is exported because the flash cookie is not the only place a rejected
// request's input is dropped, and the alternative is a second list answering
// this same question. Two such lists diverge by being edited once: the one that
// gained token, otp and the qualified names would have been the only one that
// had them.
func IsSecretField(field string) bool {
	name := strings.ToLower(field)
	if neverFlashed[name] {
		return true
	}
	for _, suffix := range neverFlashedSuffix {
		if strings.HasSuffix(name, suffix) {
			return true
		}
	}
	return false
}

// fit encodes and signs the flash, giving up what Write documents until the
// cookie is inside MaxFlashBytes.
func (f *Flash) fit(errs map[string][]string, old url.Values) string {
	value := f.encode(errs, old)
	if len(value) <= MaxFlashBytes {
		return value
	}

	// 1. Old input, longest value first. Messages matter more than redisplay.
	for len(old) > 0 {
		delete(old, longestFlashField(old))
		value = f.encode(errs, old)
		if len(value) <= MaxFlashBytes {
			return value
		}
	}

	// 2. One message per field. The first is the one that names what to change.
	first := make(map[string][]string, len(errs))
	for field, msgs := range errs {
		if len(msgs) > 0 {
			first[field] = msgs[:1]
		}
	}
	value = f.encode(first, nil)
	if len(value) <= MaxFlashBytes {
		return value
	}

	// 3. Fields, from the end of the sorted order. Sorted rather than map order
	// so that which messages survive is the same on every run -- a flash that
	// keeps a different half each time is one nobody can reproduce a report of.
	for len(first) > 1 {
		names := sortedFlashFields(first)
		delete(first, names[len(names)-1])
		value = f.encode(first, nil)
		if len(value) <= MaxFlashBytes {
			return value
		}
	}
	// One field carrying one message of at most maxFlashMessage bytes, which
	// encode has already capped it to. That always fits.
	return value
}

// encode writes the two halves into one signed payload.
//
// url.Values.Encode sorts its keys, so the same input produces the same bytes
// every time: the cookie is stable across runs, which is what lets the trimming
// above be tested and a report of it be reproduced.
func (f *Flash) encode(errs map[string][]string, old url.Values) string {
	payload := url.Values{}
	for field, msgs := range errs {
		for _, msg := range msgs {
			if len(msg) > maxFlashMessage {
				msg = msg[:maxFlashMessage]
			}
			payload.Add(flashErrorPrefix+field, msg)
		}
	}
	for field, values := range old {
		for _, value := range values {
			payload.Add(flashOldPrefix+field, value)
		}
	}
	return f.signer.Sign(flashPurpose, payload.Encode(), FlashLifetime)
}

// decodeFlash splits the payload back into the two halves.
//
// A key with neither prefix is dropped rather than guessed at. The only way one
// arrives is a payload this version did not write, and inventing a half for it
// would put an unnamed value on a page.
func decodeFlash(raw url.Values) (map[string][]string, url.Values) {
	var errs map[string][]string
	var old url.Values

	for key, values := range raw {
		switch {
		case strings.HasPrefix(key, flashErrorPrefix):
			if errs == nil {
				errs = map[string][]string{}
			}
			errs[strings.TrimPrefix(key, flashErrorPrefix)] = values
		case strings.HasPrefix(key, flashOldPrefix):
			if old == nil {
				old = url.Values{}
			}
			old[strings.TrimPrefix(key, flashOldPrefix)] = values
		}
	}

	// The secrets are dropped again on the way out. They were dropped on the way
	// in, and this side costs nothing: a cookie minted by an older binary, or by
	// a half of the framework that grows a second Write, cannot put a password
	// back into a form through here.
	return errs, redactFlash(old)
}

// longestFlashField is the field whose values cost the most bytes, which is what
// step 1 of the trimming gives up first.
//
// Ties are broken by name so the choice is the same on every run.
func longestFlashField(old url.Values) string {
	worst, size := "", -1
	for field, values := range old {
		total := len(field)
		for _, v := range values {
			total += len(v)
		}
		if total > size || (total == size && field < worst) {
			worst, size = field, total
		}
	}
	return worst
}

// sortedFlashFields returns the field names in a stable order.
func sortedFlashFields(errs map[string][]string) []string {
	out := make([]string, 0, len(errs))
	for field := range errs {
		out = append(out, field)
	}
	sort.Strings(out)
	return out
}
