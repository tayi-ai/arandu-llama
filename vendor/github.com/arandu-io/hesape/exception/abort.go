package exception

import (
	"errors"
	"net/http"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/session"
)

// StatusPageExpired is 419, which is not in any RFC.
//
// 403 would be the standard answer and it is the wrong one: it says the
// account may not do this, when the account may, and the form is simply old.
const StatusPageExpired = 419

// HTTPError is an error that names the answer it wants.
//
// It is what Abort returns and the only thing an application uses to choose a
// status. The Message is shown to whoever made the request, in every
// environment, so it is written for them: it is the developer's own sentence.
type HTTPError struct {
	// Status is the HTTP status to answer with.
	Status int
	// Message is the sentence the person sees. Empty means the standard text
	// for the status.
	Message string
	// Err is the cause, when there was one. It is reported and never shown.
	Err error
	// Headers are what the answer carries besides the status: the Retry-After
	// of a 429, the WWW-Authenticate of a 401.
	//
	// Nothing here had it, so a 429 went out with no Retry-After and a client had
	// nothing to obey. Abort does not take them -- the common failure has none --
	// so an answer that carries headers is written as the value it is:
	//
	//	&exception.HTTPError{
	//		Status:  http.StatusTooManyRequests,
	//		Headers: http.Header{"Retry-After": {"30"}},
	//	}
	Headers http.Header
}

// Error is what makes an *HTTPError satisfy the error interface.
//
// It carries the status because this string ends up in a log line, where the
// number is the first thing anybody looks for.
func (e *HTTPError) Error() string {
	msg := e.Message
	if msg == "" {
		msg = statusMessage(e.Status)
	}
	out := statusTitle(e.Status) + ": " + msg
	if e.Err != nil {
		out += ": " + e.Err.Error()
	}
	return out
}

// Unwrap exposes the cause, so errors.Is and errors.As reach through it.
func (e *HTTPError) Unwrap() error { return e.Err }

// Abort builds a failure as a value rather than raising one.
//
//	return exception.Abort(http.StatusNotFound, "no invoice with that number")
//
// There is no method on the request context, which is where the audit found the
// previous attempt at this -- a helper nothing could reach is a helper that does
// not exist. An error is reachable from every handler, from a service three
// calls down, and from a job.
//
// An empty message means the standard sentence for the status, so the common
// case is exception.Abort(404, "").
//
// To carry a cause, wrap it: fmt.Errorf("loading invoice: %w", exception.Abort(...))
// keeps the status -- StatusOf walks the chain -- and puts the context in the
// log without putting it on the page.
func Abort(status int, message string) error {
	return &HTTPError{Status: status, Message: message}
}

// AbortIf is the abort_if() helper.
//
//	if err := exception.AbortIf(invoice.Locked, http.StatusConflict, "this invoice is closed"); err != nil {
//		return err
//	}
//
// Nothing here throws, so the caller returns the error -- which is why this
// reads as one line and not as the same if statement written twice.
func AbortIf(condition bool, status int, message string) error {
	if !condition {
		return nil
	}
	return Abort(status, message)
}

// AbortUnless is the abort_unless() helper.
//
//	if err := exception.AbortUnless(invoice != nil, http.StatusNotFound, ""); err != nil {
//		return err
//	}
func AbortUnless(condition bool, status int, message string) error {
	return AbortIf(!condition, status, message)
}

// StatusOf reads an error chain and answers two things at once: the HTTP status
// the error asks for, and whether it asked at all.
//
// It is what the routing layer calls with whatever a controller action
// returned. False means nobody claimed the error, which is a 500 and, in
// development, the debug page.
func StatusOf(err error) (int, bool) { return classify(err) }

// classify is the closed table of what the collection's own errors mean.
//
// Closed is the point. Every entry here is a sentinel this collection
// declares, so the list cannot grow with an application's own errors -- those
// say what they want with Abort, which is the one mechanism, and cannot end up
// with two ways to mean 403.
//
// Order matters: an *HTTPError anywhere in the chain wins, because it is the
// explicit statement and the sentinel below it may be the cause it wrapped.
func classify(err error) (int, bool) {
	if err == nil {
		return 0, false
	}

	var he *HTTPError
	if errors.As(err, &he) {
		return he.Status, true
	}

	switch {
	// A policy refused, or a repository was reached without a Grant. Both are
	// auth.ErrForbidden, and both are 403 rather than 404: this collection does
	// not hide the existence of a resource behind a status, because the tenant
	// filter has already decided what exists at all.
	case errors.Is(err, auth.ErrForbidden):
		return http.StatusForbidden, true
	case errors.Is(err, session.ErrTokenMismatch):
		return StatusPageExpired, true
	}

	return 0, false
}
