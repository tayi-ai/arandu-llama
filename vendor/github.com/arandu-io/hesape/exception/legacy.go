package exception

import (
	"net/http"

	"github.com/arandu-io/hesape/pipeline"
)

// ErrorHandler is one callback on the handler stack.
type ErrorHandler func(err error, status int, fromConsole bool) any

// Error registers an application error handler, in front
// of the ones already registered.
func (h *Handler) Error(callback ErrorHandler) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.handlers = append([]ErrorHandler{callback}, h.handlers...)
}

// PushError registers an application error handler at
// the bottom of the stack.
func (h *Handler) PushError(callback ErrorHandler) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.handlers = append(h.handlers, callback)
}

// Missing registers a 404 error handler.
//
// There are no classes here, so the filter is the status the error classified
// as.
func (h *Handler) Missing(callback func(err error) any) {
	h.Error(func(err error, status int, _ bool) any {
		if status != http.StatusNotFound {
			return nil
		}
		return callback(err)
	})
}

// Fatal registers an error handler for fatal failures.
//
// The same failure in Go is a panic, or an error no part of the collection
// claimed, and that is what this fires for.
func (h *Handler) Fatal(callback func(err error) any) {
	h.Error(func(err error, _ int, _ bool) any {
		if _, known := classify(err); known {
			return nil
		}
		return callback(err)
	})
}

// HandleException handles an exception for the
// application.
//
// It runs the handler stack and, when none of them answered, hands the failure
// to the displayer.
func (h *Handler) HandleException(w http.ResponseWriter, r *http.Request, err error) any {
	if response := h.callCustomHandlers(err, false); response != nil {
		return response
	}

	h.displayException(w, r, err)
	return nil
}

// HandleConsole handles an error raised by a command.
//
// It runs the same handler stack with fromConsole set. The failure that
// nothing answered is written by RenderForConsole.
func (h *Handler) HandleConsole(err error) any {
	return h.callCustomHandlers(err, true)
}

// callCustomHandlers runs the handler stack until one of them answers.
func (h *Handler) callCustomHandlers(err error, fromConsole bool) (response any) {
	h.mu.Lock()
	handlers := append([]ErrorHandler(nil), h.handlers...)
	h.mu.Unlock()

	status, known := classify(err)
	if !known {
		status = http.StatusInternalServerError
	}

	for _, handler := range handlers {
		answer := func() (answer any) {
			defer func() {
				if v := recover(); v != nil {
					answer = h.formatException(v)
				}
			}()
			return handler(err, status, fromConsole)
		}()

		if answer != nil {
			return answer
		}
	}
	return nil
}

// formatException is what a handler that failed leaves behind.
func (h *Handler) formatException(v any) string {
	if h.cfg.Dev {
		return "Error in exception handler: " + headline(v)
	}
	return "Error in exception handler."
}

// displayException shows the failure with the displayer for the current mode.
func (h *Handler) displayException(w http.ResponseWriter, r *http.Request, err error) {
	h.Displayer().Display(w, r, err)
}

// Displayer is the choice made before anything is drawn: the debug displayer
// when the application is in debug mode, the plain one otherwise.
//
// There is only one of each here, so there is nothing to inject and the choice
// is the same choice.
func (h *Handler) Displayer() Displayer {
	if h.cfg.Dev {
		return &DebugDisplayer{handler: h}
	}
	return &PlainDisplayer{handler: h}
}

// RunningInConsole reports whether the process is running a command rather than
// serving requests.
//
// A Go binary knows which of the two it started as, so the kernel says so in
// Config.Console and this reads it.
func (h *Handler) RunningInConsole() bool { return h.cfg.Console }

// SetDebug sets the debug level for the handler.
//
// It is the same switch as Config.Dev, and it must be false anywhere the
// application is reachable by somebody who is not running it.
func (h *Handler) SetDebug(debug bool) {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.cfg.Dev = debug
}

// Register installs the exception handling for the
// environment, and returns the middleware that catches what escapes.
//
// Go has none of the three -- there are no warnings to promote to exceptions,
// no hook for what escaped, and no hook at shutdown. What is left is
// recover(), and Recover is the middleware that calls it, so this is where a
// kernel gets it from.
func (h *Handler) Register(environment string) pipeline.Middleware[http.Handler] {
	h.mu.Lock()
	h.environment = environment
	h.mu.Unlock()
	return Recover(h)
}

// runningInTests reports whether Register was told the environment is
// "testing".
func (h *Handler) runningInTests() bool {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.environment == "testing"
}

// Displayer draws a failure.
type Displayer interface {
	Display(w http.ResponseWriter, r *http.Request, err error)
}

// PlainDisplayer draws the status page: the standard sentence for the status
// and nothing about the inside of the process.
//
// It is what answers anywhere the application is reachable by somebody who is
// not running it, which is why it is the one that leaks nothing.
type PlainDisplayer struct{ handler *Handler }

// Display draws the status page for the failure.
//
// This draws the same page the status path draws, with the sentence for the
// status, and it copies the headers too: the sentence said there were none to
// copy, and an *HTTPError has carried them since -- Retry-After on a 429 is
// the difference between a client that backs off and one that hammers.
func (d *PlainDisplayer) Display(w http.ResponseWriter, r *http.Request, err error) {
	status, known := classify(err)
	applyErrorHeaders(w, err)
	d.handler.renderStatus(w, r, statusOr500(status, known), messageFor(err, status))
}

// DebugDisplayer draws the debug page: the stack with source, the queries with
// their timing, the dumps, the events, and the hints that name the probable
// cause.
//
// One debug page, named for what it is.
type DebugDisplayer struct{ handler *Handler }

// Display draws the debug page for the failure.
//
// It used to answer 500 always, which made a 404 in development a 500 to every
// client and to every test written against it -- the page is the same page
// either way, and the status is the error's answer rather than the page's.
func (d *DebugDisplayer) Display(w http.ResponseWriter, r *http.Request, err error) {
	status, known := classify(err)
	applyErrorHeaders(w, err)
	d.handler.renderDebug(w, r, statusOr500(status, known), err, Capture(3, d.handler.cfg.AppModule))
}
