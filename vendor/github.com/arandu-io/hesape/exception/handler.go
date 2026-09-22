package exception

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"net/http"
	"reflect"
	"strconv"
	"strings"
	"sync"

	"github.com/arandu-io/hesape/log"
)

// Config is everything the Handler needs from the application.
//
// The zero value is a working production handler: no debug page, built-in
// status pages, everything reported. Development turns Dev on, which is the
// only switch that decides whether the inside of the process is visible.
type Config struct {
	// Dev enables the debug page. It must be false anywhere the application is
	// reachable by somebody who is not running it.
	Dev bool

	// Editor is the target of the "open in IDE" links: vscode, cursor, goland
	// or zed.
	Editor string

	// AppModule is the module path of the application, used to tell app frames
	// from collection and stdlib frames.
	AppModule string

	// Diagnose collects what the registered modules have to say about the state
	// of the system right now. Pass the kernel's own.
	//
	// It exists because the most useful hint is often about something that
	// happened outside this request: the outbox has been stuck for four minutes,
	// the scheduler last ran an hour ago. A page that only looks at the request
	// cannot see any of it, and that is exactly the state where somebody is
	// staring at an error wondering what changed.
	Diagnose func(ctx context.Context) []string

	// Views is the application's own error pages, when it has any. Nil means the
	// built-in ones answer.
	Views Views

	// DontReport are the errors never written to the log.
	//
	// "Do not log 404" is the wrong shape: a 404 from a bad link and a 404 from a
	// repository that lost a row are the same status and different news.
	DontReport []error

	// RenderJSONWhen decides whether a failure is answered as JSON rather than
	// as a page. Nil means the default: the request asked for JSON.
	RenderJSONWhen func(r *http.Request) bool

	// Console says the process is running a command rather than serving
	// requests, which is what RunningInConsole reports.
	//
	// A Go binary knows which of the two it started as, so the kernel says it
	// here instead.
	Console bool
}

// Handler decides what a failed request answers.
//
// Everything an application registers on it -- Reportable, Renderable, Map,
// Ignore, Level, ThrottleUsing, BuildContextUsing -- is registered once at boot
// and read on every request, so the handler is safe for concurrent use.
type Handler struct {
	cfg Config

	mu sync.Mutex

	// dontReport are the error sentinels never written to the log, on top of
	// the ones the configuration named.
	dontReport []error
	// dontReportCallbacks inspect an error to decide whether to report it.
	dontReportCallbacks []func(error) bool
	// reportCallbacks are what Reportable registered.
	reportCallbacks []*ReportableHandler
	// renderCallbacks are what Renderable registered.
	renderCallbacks []any
	// levels maps an error sentinel to the level it is logged at.
	levels []leveled
	// throttleCallbacks decide how often an error may be reported.
	throttleCallbacks []any
	// throttles is the state behind them: how many times a key was reported in
	// the window that is open.
	throttles map[string]*throttleWindow
	// contextCallbacks build the fields that go on the log line.
	contextCallbacks []func(err error, context map[string]any) map[string]any
	// exceptionMap is what Map registered.
	exceptionMap []mapping
	// dontFlash are the input attributes never carried back to a form.
	dontFlash []string
	// withoutDuplicates says an error is reported at most once.
	withoutDuplicates bool
	// reported is the set of errors already reported, for withoutDuplicates.
	reported map[error]bool
	// finalizeResponse is what RespondUsing registered.
	finalizeResponse func(w http.ResponseWriter, r *http.Request, err error)
	// shouldRenderJSONWhenCallback is what ShouldRenderJSONWhen registered.
	shouldRenderJSONWhenCallback func(r *http.Request, err error) bool
	// handlers is the legacy handler stack, which Error and PushError fill.
	handlers []ErrorHandler
	// environment is what Register was told the application is running as.
	environment string
}

// leveled is one entry of the levels map, which
// Go has no run-time equivalent of, so the key is the sentinel errors.Is
// compares against.
type leveled struct {
	target error
	level  slog.Level
}

// mapping is one registered exception mapping.
type mapping struct {
	from error
	to   func(error) error
}

// NewHandler builds a Handler. Nothing is resolved from a registry: the
// configuration arrives as a value.
func NewHandler(cfg Config) *Handler {
	return &Handler{
		cfg:       cfg,
		reported:  map[error]bool{},
		throttles: map[string]*throttleWindow{},
	}
}

// Report writes the failure to the log, unless something
// silenced it.
//
// The level comes from the status and not from a knob: below 500 the
// application answered on purpose and it is a warning, 500 and above nobody
// meant it and it is an error. A framework where every 404 arrives at ERROR is
// a framework whose alerts get switched off. Level overrides that for a
// sentinel that deserves a different one.
func (h *Handler) Report(ctx context.Context, err error) {
	if err == nil {
		return
	}
	err = h.mapException(err)
	if h.shouldntReport(err) {
		return
	}
	h.reportThrowable(ctx, err)
}

// ShouldReport reports whether the error would be written to
// the log.
func (h *Handler) ShouldReport(err error) bool { return !h.shouldntReport(err) }

// reportThrowable reports through the error's own Report method, then through
// the registered callbacks, and finally to the log.
func (h *Handler) reportThrowable(ctx context.Context, err error) {
	h.mu.Lock()
	if h.withoutDuplicates && isComparable(err) {
		h.reported[err] = true
	}
	callbacks := append([]*ReportableHandler(nil), h.reportCallbacks...)
	h.mu.Unlock()

	// An error that knows how to report itself does, and returning true means it
	// is done.
	if own, ok := err.(interface{ Report() bool }); ok && own.Report() {
		return
	}

	for _, callback := range callbacks {
		if callback.Handles(err) && !callback.invoke(err) {
			return
		}
	}

	status, known := classify(err)
	fields := []any{"status", statusOr500(status, known), "error", err}
	for key, value := range h.buildExceptionContext(err) {
		fields = append(fields, key, value)
	}

	l := log.For(ctx)
	level, named := h.mapLogLevel(err)
	switch {
	case named:
		l.Log(ctx, level, "request failed", fields...)
	case known && status < http.StatusInternalServerError:
		l.Warn("request refused", fields...)
	default:
		l.Error("request failed", fields...)
	}
}

// shouldntReport reports whether the error is in the "do not report" list.
func (h *Handler) shouldntReport(err error) bool {
	for _, ignored := range h.cfg.DontReport {
		if errors.Is(err, ignored) {
			return true
		}
	}

	h.mu.Lock()
	if h.withoutDuplicates && isComparable(err) && h.reported[err] {
		h.mu.Unlock()
		return true
	}
	ignoredList := append([]error(nil), h.dontReport...)
	callbacks := append([]func(error) bool(nil), h.dontReportCallbacks...)
	h.mu.Unlock()

	for _, ignored := range ignoredList {
		if errors.Is(err, ignored) {
			return true
		}
	}
	for _, callback := range callbacks {
		if callback(err) {
			return true
		}
	}

	return h.throttled(err)
}

// mapLogLevel is the level Level named for this error, and whether one was.
func (h *Handler) mapLogLevel(err error) (slog.Level, bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, entry := range h.levels {
		if errors.Is(err, entry.target) {
			return entry.level, true
		}
	}
	return slog.LevelError, false
}

// buildExceptionContext creates the fields that go on the log line.
func (h *Handler) buildExceptionContext(err error) map[string]any {
	context := map[string]any{}
	if own, ok := err.(interface{ Context() map[string]any }); ok {
		for key, value := range own.Context() {
			context[key] = value
		}
	}

	h.mu.Lock()
	callbacks := append([]func(error, map[string]any) map[string]any(nil), h.contextCallbacks...)
	h.mu.Unlock()

	for _, callback := range callbacks {
		for key, value := range callback(err, context) {
			context[key] = value
		}
	}
	return context
}

// mapException maps the error through a registered mapper, if one matches.
func (h *Handler) mapException(err error) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, m := range h.exceptionMap {
		if errors.Is(err, m.from) {
			return m.to(err)
		}
	}
	return err
}

func isComparable(err error) bool {
	t := reflect.TypeOf(err)
	return t != nil && t.Comparable()
}

// Render writes the answer for a failed request instead of returning one.
//
// It answers an error a handler returned.
//
// This is the path that did not exist: every error leaving a controller became
// a panic, and every panic became 500, so an authorization refusal and a
// database being down were the same page. Now the error says what it is --
// through Abort, or through the sentinel table in classify -- and what it says
// is the answer.
//
// It does not report. Report and Render are the two halves the type doc
// separates, and the caller calls both. This used to report on the way in, so
// an application written that way logged every failure twice.
func (h *Handler) Render(w http.ResponseWriter, r *http.Request, err error) {
	if err == nil {
		return
	}

	err = h.mapException(err)

	// RespondUsing prepares the response, so it runs before anything has been
	// written: a header set after WriteHeader never leaves the process, which
	// made the callback a no-op for the one thing it is mostly used for. A
	// callback that answered the request itself is the answer, and nothing else
	// runs.
	if h.finalizeRenderedResponse(w, r, err) {
		return
	}

	// A callback that answered is the answer, and nothing else runs.
	if h.renderViaCallbacks(w, r, err) {
		return
	}

	// A returned error carries no stack: nothing captured one where it was
	// created. The frames from here show where the handler gave up, which is
	// worth more than an empty Stack section and less than the truth, and the
	// page does not claim otherwise.
	h.answer(w, r, err, err, Capture(3, h.cfg.AppModule))
}

// renderViaCallbacks tries to answer through the callbacks Renderable
// registered, and reports whether one of them did.
func (h *Handler) renderViaCallbacks(w http.ResponseWriter, r *http.Request, err error) bool {
	h.mu.Lock()
	callbacks := append([]any(nil), h.renderCallbacks...)
	h.mu.Unlock()

	for _, callback := range callbacks {
		if !handles(callback, err) {
			continue
		}
		if answered, _ := callHandler(callback, err, w, r).(bool); answered {
			return true
		}
	}
	return false
}

// finalizeRenderedResponse gives RespondUsing the response before it is
// written, and reports whether the callback answered it itself.
func (h *Handler) finalizeRenderedResponse(w http.ResponseWriter, r *http.Request, err error) bool {
	h.mu.Lock()
	finalize := h.finalizeResponse
	h.mu.Unlock()
	if finalize == nil {
		return false
	}

	p := &probe{ResponseWriter: w}
	finalize(p, r, err)
	return p.wrote
}

// RenderForConsole is the answer to a failure
// outside a request -- a command, a job, a scheduled task.
//
// The same classification, none of the HTML. A status is printed when the error
// claimed one, because a command that hits a 403 from a policy should say so
// rather than print a stack about a Grant.
func (h *Handler) RenderForConsole(w io.Writer, err error) {
	if err == nil {
		return
	}
	if status, known := classify(err); known {
		fmt.Fprintf(w, "error: %s (%d)\n", messageFor(err, status), status)
		return
	}
	fmt.Fprintf(w, "error: %s\n", err.Error())
}

// answer writes the response. value is what failed -- an error, or whatever was
// panicked with -- and err is the same thing when it happened to be an error.
func (h *Handler) answer(w http.ResponseWriter, r *http.Request, value any, err error, frames []StackFrame) {
	status, known := classify(err)
	applyErrorHeaders(w, err)

	if h.shouldReturnJSON(r, err) {
		WriteProblem(w, r, statusOr500(status, known), messageFor(err, status))
		return
	}

	// Nobody claimed it, so it is a defect rather than an answer, and in
	// development a defect is worth a page that says where it came from.
	if !known && h.cfg.Dev {
		h.renderDebug(w, r, statusOr500(status, known), value, frames)
		return
	}

	h.renderStatus(w, r, statusOr500(status, known), messageFor(err, status))
}

// renderStatus draws the page for a status: the application's own if it has
// one, and the built-in one otherwise.
func (h *Handler) renderStatus(w http.ResponseWriter, r *http.Request, status int, message string) {
	d := PageData{
		Status:    status,
		Title:     statusTitle(status),
		Message:   message,
		RequestID: requestID(w, r),
	}

	name := "errors/" + strconv.Itoa(status)
	if h.cfg.Views != nil && h.cfg.Views.Has(name) {
		// The probe is what makes the fallback safe. A view that failed before
		// writing anything can still be replaced by the built-in page; one that
		// failed halfway through cannot, and appending a second document to it
		// would turn a broken page into two broken pages.
		p := &probe{ResponseWriter: w}
		err := h.cfg.Views.Render(r.Context(), p, status, name, d)
		if err == nil {
			return
		}
		log.For(r.Context()).Error("the application's error page failed to render", "view", name, "error", err)
		if p.wrote {
			return
		}
	}

	statusPage(w, d)
}

// shouldReturnJSON decides whether the failure is answered as JSON.
//
// ShouldRenderJSONWhen wins over Config.RenderJSONWhen, which wins over the
// default, and the order is the one that matters: the callback is registered at
// boot by an application that has already read its own configuration.
func (h *Handler) shouldReturnJSON(r *http.Request, err error) bool {
	h.mu.Lock()
	callback := h.shouldRenderJSONWhenCallback
	h.mu.Unlock()

	if callback != nil {
		return callback(r, err)
	}
	if h.cfg.RenderJSONWhen != nil {
		return h.cfg.RenderJSONWhen(r)
	}
	return wantsJSON(r)
}

// wantsJSON is the default of Config.RenderJSONWhen: the request said it wanted
// JSON, either by asking for it or by being an XHR that is not htmx.
//
// htmx is deliberately excluded. It sends X-Requested-With and it swaps HTML;
// answering it with JSON would put a JSON document inside a div.
func wantsJSON(r *http.Request) bool {
	if r.Header.Get("HX-Request") == "true" {
		return false
	}
	if strings.Contains(r.Header.Get("Accept"), "application/json") {
		return true
	}
	return r.Header.Get("X-Requested-With") == "XMLHttpRequest"
}

// messageFor is the one rule about what leaves the process.
//
// An *HTTPError message was written by the developer for the person reading it,
// so it is shown. Anything else -- a driver's text, a policy's reason, the
// contents of a nil dereference -- is replaced by the standard sentence for the
// status. There is no environment flag on this: an error string is not written
// for a stranger even in development, and development already has the debug
// page, which shows everything.
func messageFor(err error, status int) string {
	var he *HTTPError
	if errors.As(err, &he) && he.Message != "" {
		return he.Message
	}
	return statusMessage(statusOr500(status, status != 0))
}

// applyErrorHeaders copies onto the response what the error asked to carry.
//
// Nothing here did it, so a 429 went out with no Retry-After and a 401 with no
// WWW-Authenticate -- a status the client could read and no instruction it
// could obey. It runs before anything is written, because a header set after
// WriteHeader is a header nobody receives.
func applyErrorHeaders(w http.ResponseWriter, err error) {
	var he *HTTPError
	if !errors.As(err, &he) || len(he.Headers) == 0 {
		return
	}
	for key, values := range he.Headers {
		for _, value := range values {
			w.Header().Add(key, value)
		}
	}
}

func statusOr500(status int, known bool) int {
	if !known || status == 0 {
		return http.StatusInternalServerError
	}
	return status
}

// requestID is the thread from this page back to the log line.
//
// The Collector has it in development; in production there is no Collector and
// the header set by the observability middleware is the only copy.
func requestID(w http.ResponseWriter, r *http.Request) string {
	if col := log.FromContext(r.Context()); col != nil && col.RequestID != "" {
		return col.RequestID
	}
	return w.Header().Get("X-Request-ID")
}

// probe records whether anything reached the client.
type probe struct {
	http.ResponseWriter
	wrote bool
}

func (p *probe) WriteHeader(status int) {
	p.wrote = true
	p.ResponseWriter.WriteHeader(status)
}

func (p *probe) Write(b []byte) (int, error) {
	p.wrote = true
	return p.ResponseWriter.Write(b)
}
