package exception

import (
	"errors"
	"log/slog"
	"net/http"
	"reflect"
	"time"

	"github.com/arandu-io/hesape/session"
)

// ReportableHandler is one callback registered with Reportable.
//
// It is returned so the registration can be continued: Stop says that
// reporting ends with this callback rather than falling through to the log.
type ReportableHandler struct {
	callback   any
	shouldStop bool
}

// Stop makes report handling stop after invoking this
// callback.
func (r *ReportableHandler) Stop() *ReportableHandler {
	r.shouldStop = true
	return r
}

// Handles reports whether the callback handles the given
// error.
//
// A Go closure carries the same information in the type of its first
// parameter, and it is read the same way: a callback written func(*HTTPError)
// bool handles whatever errors.As can pull an *HTTPError out of, and one
// written func(error) bool handles everything.
func (r *ReportableHandler) Handles(err error) bool { return handles(r.callback, err) }

// invoke calls the callback and reports whether reporting continues.
func (r *ReportableHandler) invoke(err error) bool {
	if answered, ok := callHandler(r.callback, err).(bool); ok && !answered {
		return false
	}
	return !r.shouldStop
}

// Reportable registers a reportable callback.
//
//	h.Reportable(func(err *QueryError) bool {
//		telemetry.Record(err)
//		return true
//	}).Stop()
//
// A callback that returns false stops the reporting there.
func (h *Handler) Reportable(reportUsing any) *ReportableHandler {
	handler := &ReportableHandler{callback: reportUsing}

	h.mu.Lock()
	defer h.mu.Unlock()
	h.reportCallbacks = append(h.reportCallbacks, handler)
	return handler
}

// Renderable registers a renderable callback.
//
//	h.Renderable(func(err *PaymentDeclined, w http.ResponseWriter, r *http.Request) bool {
//		...
//		return true
//	})
func (h *Handler) Renderable(renderUsing any) *Handler {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.renderCallbacks = append(h.renderCallbacks, renderUsing)
	return h
}

// Map registers a new exception mapping.
//
//	h.Map(sql.ErrNoRows, func(err error) error {
//		return exception.Abort(http.StatusNotFound, "")
//	})
//
// There are no class names in Go, so the key is the sentinel and the match is
// errors.Is, which is how this collection asks "is this that error" everywhere
// else.
func (h *Handler) Map(from error, to func(err error) error) *Handler {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.exceptionMap = append(h.exceptionMap, mapping{from: from, to: to})
	return h
}

// Ignore stops the given errors from being reported.
func (h *Handler) Ignore(errs ...error) *Handler {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, err := range errs {
		if !containsError(h.dontReport, err) {
			h.dontReport = append(h.dontReport, err)
		}
	}
	return h
}

// DontReport is Ignore under its other name.
//
// It is the alias of Ignore.
func (h *Handler) DontReport(errs ...error) *Handler { return h.Ignore(errs...) }

// DontReportWhen registers a callback that
// decides whether an error is reported.
func (h *Handler) DontReportWhen(dontReportWhen func(err error) bool) *Handler {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.dontReportCallbacks = append(h.dontReportCallbacks, dontReportWhen)
	return h
}

// StopIgnoring removes the given errors from the
// list of ignored ones.
func (h *Handler) StopIgnoring(errs ...error) *Handler {
	h.mu.Lock()
	defer h.mu.Unlock()

	kept := h.dontReport[:0]
	for _, ignored := range h.dontReport {
		if !containsError(errs, ignored) {
			kept = append(kept, ignored)
		}
	}
	h.dontReport = kept
	return h
}

// DontReportDuplicates makes an error reported
// at most once.
//
// Go has no weak map and no exception instance: the key is the error value,
// which is the same identity for the pointer errors a Go program raises, and
// an error value that cannot be a map key is simply always reported.
func (h *Handler) DontReportDuplicates() *Handler {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.withoutDuplicates = true
	return h
}

// Level sets the log level for the given error.
//
// Naming the same error twice replaces the level. This appended, and
// mapLogLevel reads from the front, so the first call won and the second was a
// line that did nothing. The entry keeps its place in the list.
func (h *Handler) Level(target error, level slog.Level) *Handler {
	h.mu.Lock()
	defer h.mu.Unlock()

	for i, entry := range h.levels {
		if sameError(entry.target, target) {
			h.levels[i].level = level
			return h
		}
	}
	h.levels = append(h.levels, leveled{target: target, level: level})
	return h
}

// sameError reports whether two registered targets are the same key.
//
// The nearest thing here is the identity of the sentinel, and == on two
// interfaces panics when their dynamic type is the same and not comparable, so
// it is guarded: an error that cannot be compared is never the same key as
// anything, which leaves it registered twice and the first one winning.
func sameError(a, b error) bool {
	return isComparable(a) && isComparable(b) && a == b
}

// ShouldRenderJSONWhen registers the
// callback that decides whether a failure is answered as JSON.
func (h *Handler) ShouldRenderJSONWhen(callback func(r *http.Request, err error) bool) *Handler {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.shouldRenderJSONWhenCallback = callback
	return h
}

// RespondUsing registers the callback that
// prepares the final response.
//
// Nothing here returns a response: the callback is given the writer the answer
// was written to, and it adds what it wants -- a header, a trace id -- after
// the fact.
func (h *Handler) RespondUsing(callback func(w http.ResponseWriter, r *http.Request, err error)) *Handler {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.finalizeResponse = callback
	return h
}

// BuildContextUsing registers a callback that
// builds the fields logged with an error.
func (h *Handler) BuildContextUsing(contextCallback func(err error, context map[string]any) map[string]any) *Handler {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.contextCallbacks = append(h.contextCallbacks, contextCallback)
	return h
}

// DontFlash keeps the given attributes from ever being carried back
// to a form after a validation failure.
func (h *Handler) DontFlash(attributes ...string) *Handler {
	h.mu.Lock()
	defer h.mu.Unlock()
	for _, attribute := range attributes {
		if !containsString(h.dontFlash, attribute) {
			h.dontFlash = append(h.dontFlash, attribute)
		}
	}
	return h
}

// FlashableInput is the input with every value that never goes back to a form
// removed: the secrets session.IsSecretField names, and whatever DontFlash was
// told on top of them.
//
// The secrets are not listed here. This package carried its own three exact
// names -- current_password, password and password_confirmation -- which let
// token, otp, secret and every qualified name through. One question is answered
// in one place, and it is the place the flash cookie already asks.
//
// This package answers failures, it does not route, so the removal is a method
// of its own and the redirect belongs to whoever builds it -- reading the
// property from another package.
func (h *Handler) FlashableInput(input map[string]any) map[string]any {
	h.mu.Lock()
	hidden := append([]string(nil), h.dontFlash...)
	h.mu.Unlock()

	out := make(map[string]any, len(input))
	for key, value := range input {
		if session.IsSecretField(key) || containsString(hidden, key) {
			continue
		}
		out[key] = value
	}
	return out
}

// Throttle is how often an error may be reported.
//
// Two replicas therefore throttle separately, which is the honest cost of not
// depending on a shared store from here.
//
// The zero value reports nothing, because zero attempts is zero attempts.
// Unlimited is the one that throttles nothing.
type Throttle struct {
	// Key groups the errors that share a budget. Empty means one budget per error
	// type.
	Key string
	// MaxAttempts is how many reports fit in the window.
	//
	// Zero is zero: the error is never reported. Unlimited is how a callback says
	// it wants no budget.
	MaxAttempts int
	// Decay is how long the window lasts. Zero means a minute.
	Decay time.Duration
}

// Unlimited is the throttle that reports the error every time.
//
// It is what a ThrottleUsing callback returns when it has looked at the error
// and decided it wants no budget on it, and it is what the handler falls back to
// when no callback answered at all.
var Unlimited = Throttle{MaxAttempts: -1}

// throttleWindow is one open budget: when it started and how much of it is left.
type throttleWindow struct {
	started time.Time
	count   int
}

// ThrottleUsing registers a callback that decides
// how often an error may be reported.
//
//	h.ThrottleUsing(func(err *QueryError) exception.Throttle {
//		return exception.Throttle{MaxAttempts: 10, Decay: time.Minute}
//	})
//
// The callback is matched to the error by the type of its first parameter, the
// same way Reportable's is.
func (h *Handler) ThrottleUsing(throttleUsing any) *Handler {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.throttleCallbacks = append(h.throttleCallbacks, throttleUsing)
	return h
}

// throttled reports whether this error has used up its budget.
//
// Reading a budget of zero as "no answer" is what made Throttle{MaxAttempts:
// 0} report every time instead of never.
func (h *Handler) throttled(err error) bool {
	h.mu.Lock()
	callbacks := append([]any(nil), h.throttleCallbacks...)
	h.mu.Unlock()

	throttle := Unlimited
	for _, callback := range callbacks {
		if !handles(callback, err) {
			continue
		}
		if answer, ok := callHandler(callback, err).(Throttle); ok {
			throttle = answer
			break
		}
	}
	if throttle.MaxAttempts < 0 {
		return false
	}
	if throttle.MaxAttempts == 0 {
		// Zero attempts fit in the window, so this one does not.
		return true
	}

	key := throttle.Key
	if key == "" {
		key = reflect.TypeOf(err).String()
	}
	decay := throttle.Decay
	if decay <= 0 {
		decay = time.Minute
	}

	h.mu.Lock()
	defer h.mu.Unlock()
	window, ok := h.throttles[key]
	if !ok || time.Since(window.started) > decay {
		h.throttles[key] = &throttleWindow{started: time.Now(), count: 1}
		return false
	}
	window.count++
	return window.count > throttle.MaxAttempts
}

// containsError reports whether the list already names the error.
//
// It compares with errors.Is, and that is not a matter of taste: == on two
// interface values panics at run time when their dynamic type is the same and
// not comparable, which is what an error carrying several causes looks like --
// a struct with a slice in it. Two of those handed to Ignore took the process
// down, while reportThrowable next door had guarded against the same thing all
// along. errors.Is tests comparability before it compares, and it is how the
// rest of this package asks whether one error is another.
func containsError(list []error, target error) bool {
	for _, item := range list {
		if errors.Is(target, item) {
			return true
		}
	}
	return false
}

func containsString(list []string, target string) bool {
	for _, item := range list {
		if item == target {
			return true
		}
	}
	return false
}

var errorType = reflect.TypeOf((*error)(nil)).Elem()

// firstParameterType is the type of a callback's first parameter.
func firstParameterType(fn any) reflect.Type {
	if fn == nil {
		return nil
	}
	t := reflect.TypeOf(fn)
	if t.Kind() != reflect.Func || t.NumIn() == 0 {
		return nil
	}
	return t.In(0)
}

// handles reports whether err is what the callback's first parameter names.
func handles(fn any, err error) bool {
	want := firstParameterType(fn)
	switch {
	case want == nil:
		return false
	case want == errorType:
		return true
	case want.Kind() == reflect.Interface, want.Implements(errorType):
		target := reflect.New(want)
		return errors.As(err, target.Interface())
	default:
		return false
	}
}

// callHandler calls the callback with the error narrowed to the type its first
// parameter names, and the rest of the arguments after it.
//
// An argument the callback did not ask for is dropped and one it asked for and
// did not get is the zero value, because a reflect.Call with the wrong arity
// panics and a mistyped callback should not take the process down while it is
// already answering a failure.
func callHandler(fn any, err error, extra ...any) any {
	v := reflect.ValueOf(fn)
	if v.Kind() != reflect.Func {
		return nil
	}
	t := v.Type()

	in := make([]reflect.Value, 0, t.NumIn())
	for i := range t.NumIn() {
		if i == 0 {
			in = append(in, errorAs(err, t.In(0)))
			continue
		}
		in = append(in, argumentFor(extra, i-1, t.In(i)))
	}

	out := v.Call(in)
	if len(out) == 0 {
		return nil
	}
	return out[0].Interface()
}

// errorAs narrows the error to the type the parameter names.
func errorAs(err error, want reflect.Type) reflect.Value {
	if want == errorType {
		if err == nil {
			return reflect.Zero(want)
		}
		return reflect.ValueOf(err)
	}
	if want.Kind() == reflect.Interface || want.Implements(errorType) {
		target := reflect.New(want)
		if errors.As(err, target.Interface()) {
			return target.Elem()
		}
	}
	return reflect.Zero(want)
}

// argumentFor prepares one argument for the parameter it is about to fill.
func argumentFor(args []any, i int, want reflect.Type) reflect.Value {
	if i >= len(args) || args[i] == nil {
		return reflect.Zero(want)
	}
	v := reflect.ValueOf(args[i])
	if v.Type().AssignableTo(want) {
		return v
	}
	return reflect.Zero(want)
}
