package exceptions

import (
	"fmt"
	"net/http"
	"strings"
)

// BackedEnumCaseNotFoundError is the error when a parameter value does not
// match any case of the expected backed enum.
type BackedEnumCaseNotFoundError struct {
	Class string
	Value string
}

func (e *BackedEnumCaseNotFoundError) Error() string {
	return fmt.Sprintf("routing: no case of %q matched %q", e.Class, e.Value)
}

// InvalidSignatureError is returned when a signed URL fails verification.
type InvalidSignatureError struct {
	Message string
}

func (e *InvalidSignatureError) Error() string {
	return "routing: invalid signature: " + e.Message
}

// MissingRateLimiterError is returned when a named rate limiter has not
// been registered.
type MissingRateLimiterError struct {
	Name string
}

func (e *MissingRateLimiterError) Error() string {
	return "routing: rate limiter " + e.Name + " has not been registered"
}

// ForLimiter builds a MissingRateLimiterError naming limiter.
func ForLimiter(limiter string) *MissingRateLimiterError {
	return &MissingRateLimiterError{Name: limiter}
}

// ForLimiterAndUser builds a MissingRateLimiterError naming both the limiter
// and the model whose property named it, so the message says which of the two
// halves is missing.
func ForLimiterAndUser(limiter, model string) *MissingRateLimiterError {
	return &MissingRateLimiterError{Name: model + "::" + limiter}
}

// StreamedResponseError wraps an error thrown during streamed response
// generation.
type StreamedResponseError struct {
	Err error
}

func (e *StreamedResponseError) Error() string {
	return "routing: streamed response failed: " + e.Err.Error()
}

func (e *StreamedResponseError) Unwrap() error { return e.Err }

// GetInnerException returns the error raised while the body was already being
// written, which is the one that says what actually went wrong.
func (e *StreamedResponseError) GetInnerException() error { return e.Err }

// Render answers an empty 200 body: the response has already begun, so there
// is nothing left to say. It writes nothing and reports whether the header
// could still be sent.
func (e *StreamedResponseError) Render(w http.ResponseWriter) error {
	if w == nil {
		return nil
	}
	_, err := w.Write(nil)
	return err
}

// URLGenerationError is returned when a route URL cannot be built because
// one or more required parameters are missing.
type URLGenerationError struct {
	// Route is the name of the route the URL was being built for.
	Route string
	// URI is the pattern of that route, so the message shows which
	// placeholders exist next to the ones that were not filled.
	URI string
	// Parameters are the placeholders left unfilled.
	Parameters []string
}

func (e *URLGenerationError) Error() string {
	label := "parameters"
	if len(e.Parameters) == 1 {
		label = "parameter"
	}
	msg := fmt.Sprintf("routing: missing required %s for [Route: %s] [URI: %s]", label, e.Route, e.URI)
	if len(e.Parameters) > 0 {
		msg += fmt.Sprintf(" [Missing %s: %s]", label, strings.Join(e.Parameters, ", "))
	}
	return msg + "."
}

// GeneratedRoute is what ForMissingParameters needs from a route: its name and
// its URI, the two halves of the message.
//
// It is an interface and not the route type because routing imports this
// package to raise the error, and naming the route type here would be a cycle.
// hesape/routing.Route satisfies it.
type GeneratedRoute interface {
	// GetName returns the route's name.
	GetName() string
	// URI returns the route's pattern.
	URI() string
}

// ForMissingParameters builds the error a named URL raises when the caller
// gave fewer parameters than the pattern has placeholders -- which is the
// failure Routes.Route exists to turn into a message instead of a 404 the
// reader sees.
func ForMissingParameters(route GeneratedRoute, parameters ...string) *URLGenerationError {
	e := &URLGenerationError{Parameters: parameters}
	if route != nil {
		e.Route = route.GetName()
		e.URI = route.URI()
	}
	return e
}
