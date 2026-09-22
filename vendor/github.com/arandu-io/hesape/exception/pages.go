package exception

import (
	"context"
	"html/template"
	"net/http"
)

// PageData is what a status page is given.
//
// It is a struct and not a map, because every view in this collection takes a
// typed struct: a map renders blank on a typo, and blank is what nobody
// debugs. An application that provides its own errors/404 declares its page
// data as this type.
type PageData struct {
	// Status is the HTTP status being answered.
	Status int
	// Title is the short name of the status: "Not Found", "Page Expired".
	Title string
	// Message is the sentence written for the person reading it.
	Message string
	// RequestID ties the page to the log line for this request. Empty when
	// nothing upstream assigned one.
	RequestID string
}

// Views is the application's own error pages, when it has any.
//
// It is declared here rather than imported because the view layer sits above
// this package and importing it back would be a cycle -- the same reason
// http.Renderer is declared where it is consumed. The view package satisfies
// it without knowing this package exists.
//
// Has is what makes the fallback safe: this package asks before it renders, so
// an application that has an errors/404 gets its own page and one that has not
// gets the built-in one, with no guessing from a failed render.
type Views interface {
	// Has reports whether a view of that name is registered.
	Has(name string) bool
	// Render draws it with the status already decided.
	Render(ctx context.Context, w http.ResponseWriter, status int, name string, data any) error
}

// statusTitle is the short name of a status, and it deliberately falls back to
// the standard text rather than to "Error": a page headed "418" with no words
// is worse than one headed "I'm a teapot".
func statusTitle(status int) string {
	// 419 is in no RFC, so the standard library has no text for it, and the
	// fallback below would head the page with the number alone.
	if status == StatusPageExpired {
		return "Page Expired"
	}
	if t := http.StatusText(status); t != "" {
		return t
	}
	return "Error"
}

// statusMessage is the sentence shown when the caller did not write one.
//
// They live here rather than in the skeleton because an application that has
// not been touched must still answer its own 404, and because they have to
// render when the view build is what broke.
//
// The tone follows 00-meta/DOC-brand.md: say what happened, say what to do, no
// apology and no exclamation mark.
func statusMessage(status int) string {
	switch status {
	case http.StatusUnauthorized:
		return "You need to sign in to see this page."
	case http.StatusForbidden:
		return "This area is not open to your account."
	case http.StatusNotFound:
		return "There is nothing at this address."
	case http.StatusMethodNotAllowed:
		return "This address does not answer that method."
	case StatusPageExpired:
		return "This page expired. Reload it and submit the form again."
	case http.StatusTooManyRequests:
		return "Too many requests. Wait a moment and try again."
	case http.StatusInternalServerError:
		return "Something went wrong on our side. The failure was recorded."
	case http.StatusServiceUnavailable:
		return "The application is unavailable. Try again shortly."
	}
	if status >= 500 {
		return "Something went wrong on our side. The failure was recorded."
	}
	return "This request could not be answered."
}

// statusPage writes the built-in page for a status.
//
// One template and a table of sentences, not eight files: eight files is eight
// places for the markup to drift, and the only thing that differs between a
// 403 and a 503 is two strings.
func statusPage(w http.ResponseWriter, d PageData) {
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	// A status page is about this request and this account. A shared cache must
	// never hand it to somebody else, and a 404 that got cached is a page that
	// stays missing after it is created.
	w.Header().Set("Cache-Control", "no-store, private")
	w.WriteHeader(d.Status)
	_ = statusTmpl.Execute(w, d)
}

var statusTmpl = template.Must(template.New("status").Parse(statusHTML))

const statusHTML = `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{.Status}} {{.Title}}</title>
<style>
:root{color-scheme:light dark;--bg:#fff;--fg:#111827;--dim:#6b7280;--line:#e5e7eb}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--dim:#8b949e;--line:#30363d}}
*{box-sizing:border-box}
body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;
background:var(--bg);color:var(--fg);
font:16px/1.6 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:34rem;padding:2rem}
.status{font-size:.8125rem;letter-spacing:.12em;text-transform:uppercase;color:var(--dim)}
h1{margin:.5rem 0 0;font-size:1.5rem;font-weight:600}
p{margin:.75rem 0 0}
.id{margin-top:1.5rem;padding-top:1rem;border-top:1px solid var(--line);
font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:var(--dim)}
</style></head><body>
<main>
  <div class="status">{{.Status}}</div>
  <h1>{{.Title}}</h1>
  <p>{{.Message}}</p>
  {{if .RequestID}}<div class="id">request_id {{.RequestID}}</div>{{end}}
</main>
</body></html>`
