package exception

import (
	"fmt"
	"html/template"
	"net/http"
	"strings"
	"time"

	"github.com/arandu-io/hesape/log"
)

// viewData is what the debug page is drawn from.
//
// It is unexported, and so are the two methods that render it: the page is not
// something an application draws, it is what the Handler falls back to when
// nothing claimed the error and Config.Dev is true.
type viewData struct {
	// Title is the headline: what happened, in the words of whatever failed.
	//
	// It used to be the Go type of the panic value, so the biggest text on the
	// page read "*errors.errorString" -- true, and useless. The product's whole
	// claim is a debug page that names the probable cause; the headline is the
	// first thing it says. Found by audit.
	Title string
	// Kind is the Go type, kept as the subtitle: it matters when the message is
	// generic, and it is never what somebody reads first.
	Kind      string
	Message   string
	RequestID string
	Method    string
	Path      string
	Frames    []StackFrame
	Queries   []log.QueryRecord
	Dumps     []log.DumpRecord
	Events    []log.EventRecord
	External  []log.ExternalRecord
	NPlusOne  map[string]int
	Headers   map[string][]string
	Elapsed   time.Duration
	QueryTime time.Duration
	Hints     []string
	Editor    string
}

// headline turns whatever failed into one line somebody can read.
//
// The first line only, and bounded: a panic value carrying a stack trace or a
// serialized payload would otherwise push everything else off the screen.
func headline(v any) string {
	text := strings.TrimSpace(fmt.Sprint(v))
	if i := strings.IndexByte(text, '\n'); i >= 0 {
		text = strings.TrimSpace(text[:i])
	}
	if text == "" {
		// Something failed that prints as nothing. The type is all there is, and
		// it beats an empty headline.
		return fmt.Sprintf("%T", v)
	}
	const limit = 140
	if len(text) > limit {
		return text[:limit] + "..."
	}
	return text
}

// renderDebug draws the full debug page.
//
// frames is passed in rather than captured here because the two callers stand
// in different places: Recover captures at the point it caught the panic, which
// is the only place the panicking line is still on the stack. A returned error
// carries no stack at all, and the frames it gets show where the Handler was
// called -- see docs/31, section on what this cannot do.
//
// status is passed in for the same kind of reason: the page is the same page
// whatever the failure classified as, and the status is the failure's answer.
// It was 500 here and the caller had no say, so a 404 drawn by the debug
// displayer went out as a 500.
func (h *Handler) renderDebug(w http.ResponseWriter, r *http.Request, status int, value any, frames []StackFrame) {
	d := viewData{
		Title:   headline(value),
		Kind:    fmt.Sprintf("%T", value),
		Message: fmt.Sprint(value),
		Method:  r.Method,
		Path:    r.URL.Path,
		Frames:  frames,
		Headers: redact(r.Header),
		Editor:  h.cfg.Editor,
	}
	if col := log.FromContext(r.Context()); col != nil {
		d.RequestID = col.RequestID
		d.Queries, d.Dumps, d.Events, d.External = col.Queries(), col.Dumps(), col.Events(), col.External()
		d.NPlusOne = col.SuspectedNPlusOne(nPlusOneThreshold)
		d.Elapsed = time.Since(col.Start)
		d.QueryTime = col.QueryTime()
	}
	d.Hints = hints(d)
	if h.cfg.Diagnose != nil {
		d.Hints = append(d.Hints, h.cfg.Diagnose(r.Context())...)
	}

	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.WriteHeader(status)
	_ = debugTmpl.Execute(w, d)
}

// renderDump draws the dump page, for the DumpDie flow. It answers 200: the
// request was aborted on purpose, not by a failure.
func (h *Handler) renderDump(w http.ResponseWriter, r *http.Request) {
	d := viewData{Title: "Dump", Kind: "dump", Method: r.Method, Path: r.URL.Path, Editor: h.cfg.Editor}
	if col := log.FromContext(r.Context()); col != nil {
		d.RequestID, d.Dumps, d.Queries = col.RequestID, col.Dumps(), col.Queries()
		d.Elapsed = time.Since(col.Start)
		d.QueryTime = col.QueryTime()
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	_ = debugTmpl.Execute(w, d)
}

// redact hides sensitive headers even in development: a screenshot of an error
// page leaks a session cookie with absurd ease.
func redact(h http.Header) map[string][]string {
	sensitive := map[string]bool{
		"Cookie": true, "Set-Cookie": true, "Authorization": true,
		"X-Csrf-Token": true, "X-Arandu-Trace": true, "Proxy-Authorization": true,
	}
	out := map[string][]string{}
	for k, v := range h {
		if sensitive[http.CanonicalHeaderKey(k)] {
			out[k] = []string{"[redacted]"}
			continue
		}
		out[k] = v
	}
	return out
}

var debugTmpl = template.Must(template.New("debug").Funcs(template.FuncMap{
	// The return type must be template.URL: html/template only trusts a short
	// list of schemes in an href, and it rewrites everything else to
	// "#ZgotmplZ". Without this, every "open in editor" link on the page is
	// silently dead, because vscode:// and zed:// are not on that list.
	//
	// It forwards to log.EditorLink, which is where the one implementation lives:
	// the debug console needs the same links, and two copies is two places to add
	// the next editor. The wrapper this package used to export was that second
	// copy.
	"editorLink": func(editor, file string, line int) template.URL {
		return template.URL(log.EditorLink(editor, file, line))
	},
	"isSlow": func(d time.Duration) bool { return d >= slowQuery },
}).Parse(debugHTML))

const debugHTML = `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>{{.Title}} — arandu</title>
<style>
:root{color-scheme:dark;--bg:#0d1117;--panel:#161b22;--line:#30363d;--fg:#e6edf3;--dim:#8b949e;--red:#f85149;--amber:#d29922;--accent:#58a6ff}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);font:14px/1.6 ui-monospace,SFMono-Regular,Menlo,monospace}
header{background:var(--panel);border-bottom:1px solid var(--line);padding:20px 28px}
h1{margin:0;font-size:16px;color:var(--red)}h1 span{color:var(--dim);font-weight:400}
.msg{margin-top:8px;font-size:18px;color:var(--fg)}
.meta{margin-top:10px;color:var(--dim);font-size:12px}
main{padding:20px 28px;max-width:1200px}
section{margin-bottom:28px}
h2{font-size:12px;text-transform:uppercase;letter-spacing:.08em;color:var(--dim);border-bottom:1px solid var(--line);padding-bottom:6px}
.hint{background:rgba(210,153,34,.12);border-left:3px solid var(--amber);padding:10px 14px;margin:8px 0;border-radius:0 4px 4px 0}
.frame{border:1px solid var(--line);border-radius:6px;margin-bottom:10px;overflow:hidden}
.frame.vendor{opacity:.45}
.frame>summary{padding:8px 12px;background:var(--panel);cursor:pointer;list-style:none}
.frame .fn{color:var(--accent)}.frame .loc{color:var(--dim);font-size:12px}
.frame a{color:var(--dim)}
pre{margin:0;padding:12px;overflow-x:auto;background:#010409}
table{width:100%;border-collapse:collapse;font-size:13px}
td,th{text-align:left;padding:6px 10px;border-bottom:1px solid var(--line);vertical-align:top}
th{color:var(--dim);font-weight:500}
.slow{color:var(--amber)}
.err{color:var(--red)}
</style></head><body>
<header>
  <h1>{{.Title}} <span>{{if .Kind}}{{.Kind}} — {{end}}arandu debug (development only)</span></h1>
  <div class="msg">{{.Message}}</div>
  <div class="meta">{{.Method}} {{.Path}} · request_id {{.RequestID}} · {{.Elapsed}} total · {{.QueryTime}} in SQL · {{len .Queries}} queries</div>
</header>
<main>

{{if .Hints}}<section><h2>Diagnosis</h2>
  {{range .Hints}}<div class="hint">{{.}}</div>{{end}}
</section>{{end}}

{{if .Frames}}<section><h2>Stack</h2>
  {{range .Frames}}
  <details class="frame {{if not .IsApp}}vendor{{end}}" {{if .IsApp}}open{{end}}>
    <summary><span class="fn">{{.Func}}</span><br>
      <span class="loc">{{.File}}:{{.Line}}</span>
      {{with editorLink $.Editor .File .Line}}<a href="{{.}}">open in editor</a>{{end}}
    </summary>
    {{if .Snippet}}<pre>{{range .Snippet}}{{.}}
{{end}}</pre>{{end}}
  </details>
  {{end}}
</section>{{end}}

{{if .Queries}}<section><h2>Queries</h2><table>
  <tr><th>SQL</th><th>Time</th><th>Rows</th><th>Origin</th></tr>
  {{range .Queries}}<tr>
    <td class="{{if .Err}}err{{end}}">{{.SQL}}{{if .Err}}<br>{{.Err}}{{end}}</td>
    <td class="{{if isSlow .Duration}}slow{{end}}">{{.Duration}}</td>
    <td>{{.Rows}}</td>
    <td class="loc">{{.Caller.File}}:{{.Caller.Line}}</td>
  </tr>{{end}}
</table></section>{{end}}

{{if .Dumps}}<section><h2>Dumps</h2><table>
  <tr><th>Label</th><th>Value</th><th>Origin</th><th>At</th></tr>
  {{range .Dumps}}<tr><td>{{.Label}}</td><td>{{printf "%+v" .Value}}</td>
  <td class="loc">{{.Caller.File}}:{{.Caller.Line}}</td><td>{{.At}}</td></tr>{{end}}
</table></section>{{end}}

{{if .Events}}<section><h2>Events</h2><table>
  <tr><th>Name</th><th>Payload</th><th>At</th></tr>
  {{range .Events}}<tr><td>{{.Name}}</td><td>{{printf "%+v" .Payload}}</td><td>{{.At}}</td></tr>{{end}}
</table></section>{{end}}

{{if .External}}<section><h2>Outbound calls</h2><table>
  <tr><th>Method</th><th>URL</th><th>Status</th><th>Time</th></tr>
  {{range .External}}<tr><td>{{.Method}}</td><td>{{.URL}}</td><td>{{.Status}}</td><td>{{.Duration}}</td></tr>{{end}}
</table></section>{{end}}

<section><h2>Request headers</h2><table>
  {{range $k, $v := .Headers}}<tr><th>{{$k}}</th><td>{{range $v}}{{.}} {{end}}</td></tr>{{end}}
</table></section>

</main></body></html>`
