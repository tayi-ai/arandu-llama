package middleware

import (
	"mime"
	"net/http"
	"slices"
	"strings"

	hhttp "github.com/arandu-io/hesape/http"
)

// methodField is the hidden field a form names its real method in.
const methodField = "_method"

// spoofableMethods are the three the field may name: the state-changing methods
// a browser will not send from a form. GET and POST are left out because a form
// sends them for real, and HEAD, OPTIONS, TRACE and CONNECT are left out because
// no form means them.
var spoofableMethods = []string{http.MethodPut, http.MethodPatch, http.MethodDelete}

// formMediaType is the only body this reads.
const formMediaType = "application/x-www-form-urlencoded"

// OverrideMethod rewrites a POST whose form carries a hidden _method field into
// the PUT, PATCH or DELETE that field names.
//
// A browser sends GET and POST from a form and nothing else, so a form that
// means DELETE posts and says so in a hidden field. Writing the field is only
// half of it: until something reads it the request is still a POST, and a route
// registered for DELETE never matches. That is why this is a middleware over the
// whole server and not a check inside a handler -- the method is what picks the
// route, so the rewrite has to land before the router looks at the request.
//
// # It goes after the CSRF check, not before
//
// The field is a value out of a body nothing has authenticated yet. Put this
// first and the sender chooses which method the CSRF middleware sees: a POST
// arrives, becomes a DELETE, and is checked as a DELETE. Today that changes no
// outcome, because every method the check treats as unsafe is checked the same
// way -- but the exemption that would make it change one is a single line, and
// whoever adds it is reading a list of methods this middleware writes into.
// Behind the check, the request it decides about is the POST the client really
// sent, and the rewrite is something that happens to a request already accepted.
//
// Nothing is lost by waiting: the router runs after both.
//
// # What it leaves alone
//
//   - Anything that is not a POST. Rewriting the method of a GET is the CSRF
//     path: a GET is safe, so the check waves it through, and a rewrite would
//     hand the router a state-changing method nobody validated a token for --
//     from a link, which is all it takes.
//   - Any value that is not PUT, PATCH or DELETE. An unknown one is ignored and
//     not refused: what arrived is a valid POST that a route may well answer,
//     and a middleware that returns 400 for a field it did not recognise fails
//     requests it was only ever asked to translate.
//   - Any body that is not [formMediaType]. A JSON body is not a form and is not
//     parsed here. multipart/form-data is a form and is still not covered:
//     ParseForm does not read a multipart body, and reading it would mean
//     buffering every upload, files and all, in a middleware that usually finds
//     no field at all. A multipart form that needs another method sends it to a
//     route registered for POST.
//
// The field is read out of the body and never out of the query string. A form
// puts it in the body; ?_method=DELETE is something a link can carry, and a link
// is what the first exclusion above exists to refuse.
//
// It calls ParseForm rather than reading the body itself, and that is what keeps
// the body from going missing: the parse fills PostForm on the request, so the
// handler's own ParseForm, FormValue or PostFormValue finds the values already
// there instead of going back to a body that has been read. A handler that reads
// r.Body directly sees what it sees after any form parse, which is nothing -- so
// a handler wanting the raw bytes of a urlencoded body reads them before this
// runs, or does not use a form content type.
//
// A body ParseForm cannot parse is not refused either. It is left as the POST it
// came in as, for whatever the router matches to answer: there is no method to
// read out of it, and answering a malformed body is a decision about an
// application, not about a translation.
func OverrideMethod() hhttp.Middleware {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			if r.Method != http.MethodPost || !isFormBody(r) {
				next.ServeHTTP(w, r)
				return
			}

			if err := r.ParseForm(); err != nil {
				next.ServeHTTP(w, r)
				return
			}

			// Uppercased before the comparison because a method is uppercase on
			// the wire, and the builder that writes the field uppercases it
			// already: a hand-written form saying "delete" means the method
			// spelled DELETE, and assigning it verbatim would name one that
			// matches no route.
			if want := strings.ToUpper(r.PostForm.Get(methodField)); slices.Contains(spoofableMethods, want) {
				r.Method = want
			}

			next.ServeHTTP(w, r)
		})
	}
}

// isFormBody reports whether the request carries a urlencoded form.
//
// The header is parsed rather than compared, because a browser sends the
// parameters with it -- "application/x-www-form-urlencoded; charset=UTF-8" is
// the ordinary case -- and a string comparison would miss every one of those and
// silently leave the method alone.
func isFormBody(r *http.Request) bool {
	media, _, err := mime.ParseMediaType(r.Header.Get("Content-Type"))
	return err == nil && media == formMediaType
}
