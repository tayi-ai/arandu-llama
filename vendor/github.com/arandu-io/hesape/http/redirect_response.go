package http

import (
	"mime/multipart"
	stdhttp "net/http"
	"net/url"
	"strings"

	"github.com/arandu-io/hesape/session"
	"github.com/arandu-io/hesape/support"
	"github.com/arandu-io/hesape/validation"
)

// RedirectResponse is a Response whose whole content is a Location header,
// plus the four things a redirect carries across the request boundary --
// flashed data, old input, errors and cookies.
//
// It embeds Response, so its own methods and Response's are on the same
// fields.
//
// The tenant is never flashed and never read back: WithInput carries what
// the browser typed, and what a repository is allowed to see comes from the
// auth.Grant a policy minted, not from a value that survived a redirect.
type RedirectResponse struct {
	Response

	// targetURL is where the browser is being sent.
	targetURL string
	// request is the request WithInput and OnlyInput read the input off.
	request *Request
	// session is the session store With, WithInput and WithErrors flash
	// into.
	session *session.Store
}

// NewRedirectResponse builds a RedirectResponse.
//
// The variadic arguments are status (302) and headers (none).
func NewRedirectResponse(to string, args ...any) *RedirectResponse {
	status := stdhttp.StatusFound
	if len(args) > 0 {
		if value, ok := args[0].(int); ok {
			status = value
		}
	}
	r := &RedirectResponse{
		Response: Response{
			status:          status,
			headers:         stdhttp.Header{},
			protocolVersion: "1.0",
		},
	}
	if len(args) > 1 {
		switch headers := args[1].(type) {
		case stdhttp.Header:
			r.WithHeaders(headers)
		case map[string]string:
			for key, value := range headers {
				r.Response.headers.Set(key, value)
			}
		}
	}
	return r.SetTargetUrl(to)
}

// GetTargetUrl is where the browser is being sent.
func (r *RedirectResponse) GetTargetUrl() string { return r.targetURL }

// SetTargetUrl sets where the browser is being sent. WithFragment and
// EnforceSameOrigin both call it.
func (r *RedirectResponse) SetTargetUrl(to string) *RedirectResponse {
	r.targetURL = to
	r.Response.headers.Set("Location", to)
	return r
}

// With flashes a piece of data to the session, so the page that follows the
// redirect can read it once.
//
// The key may be a map or a single key: a map flashes every pair in it, and
// a single key flashes with value. The value is variadic, defaulting to nil.
func (r *RedirectResponse) With(key any, value ...any) *RedirectResponse {
	if r.session == nil {
		return r
	}
	var single any
	if len(value) > 0 {
		single = value[0]
	}
	switch typed := key.(type) {
	case map[string]any:
		for k, v := range typed {
			r.session.Flash(k, v)
		}
	case map[string]string:
		for k, v := range typed {
			r.session.Flash(k, v)
		}
	default:
		r.session.Flash(stringify(key), single)
	}
	return r
}

// WithCookies adds several cookies at once.
func (r *RedirectResponse) WithCookies(cookies []*stdhttp.Cookie) *RedirectResponse {
	for _, cookie := range cookies {
		r.Response.WithCookie(cookie)
	}
	return r
}

// WithInput flashes the input to the session so the form that was rejected
// comes back with the boxes filled.
//
// It is one half of a pair with [Request.Old], which reads what this writes.
// If the two diverge, the form comes back blank, which is the bug the pair
// exists to not have.
//
// Two things never reach the session: an uploaded file, which is not
// something to put back in a text box, and a secret field, which the session
// store drops whatever the caller passes.
//
// The variadic argument is the optional input to flash; with none, the
// request's own input is used.
func (r *RedirectResponse) WithInput(input ...map[string]any) *RedirectResponse {
	if r.session == nil {
		return r
	}
	payload := map[string]any{}
	if len(input) > 0 && input[0] != nil {
		payload = input[0]
	} else if r.request != nil {
		payload, _ = r.request.Input("").(map[string]any)
	}
	r.session.FlashInput(removeFilesFromInput(payload))
	return r
}

// removeFilesFromInput strips uploaded files out of the input: a file is not
// something to put back in a text box, and it is not something to keep in a
// session either.
func removeFilesFromInput(input map[string]any) map[string]any {
	out := make(map[string]any, len(input))
	for key, value := range input {
		if isUploadedValue(value) {
			continue
		}
		switch typed := value.(type) {
		case map[string]any:
			out[key] = removeFilesFromInput(typed)
		case []any:
			list := make([]any, 0, len(typed))
			for _, item := range typed {
				if isUploadedValue(item) {
					continue
				}
				list = append(list, item)
			}
			out[key] = list
		default:
			out[key] = value
		}
	}
	return out
}

// isUploadedValue reports whether a value is a file that arrived on the
// request, in either of the two shapes [Request.AllFiles] produces.
func isUploadedValue(value any) bool {
	switch value.(type) {
	case *multipart.FileHeader, []*multipart.FileHeader, *UploadedFile, []*UploadedFile:
		return true
	}
	return false
}

// OnlyInput flashes only the named keys.
func (r *RedirectResponse) OnlyInput(keys ...string) *RedirectResponse {
	if r.request == nil {
		return r
	}
	return r.WithInput(r.request.Only(keys...))
}

// ExceptInput flashes everything except the named keys.
//
// The keys are the form's own reason to drop a field. A password, a one-time
// code or a request token does not need naming here: the session store drops
// the secret fields whatever the caller passes.
func (r *RedirectResponse) ExceptInput(keys ...string) *RedirectResponse {
	if r.request == nil {
		return r
	}
	return r.WithInput(r.request.Except(keys...))
}

// WithErrors flashes a container of messages into the session's error bag,
// so the page that follows draws them beside the fields they belong to.
//
// provider takes any of several shapes: a MessageProvider, a *MessageBag, a
// map, a slice of strings, a single string, or validation.Errors, which is
// what this module's validator produces. The variadic key names the bag,
// defaulting to "default" -- which is what lets two forms on one page keep
// their messages apart.
func (r *RedirectResponse) WithErrors(provider any, key ...string) *RedirectResponse {
	if r.session == nil {
		return r
	}
	bagName := support.DefaultErrorBag
	if len(key) > 0 && key[0] != "" {
		bagName = key[0]
	}

	errors, _ := r.session.Get("errors", support.NewViewErrorBag()).(*support.ViewErrorBag)
	if errors == nil {
		errors = support.NewViewErrorBag()
	}
	r.session.Flash("errors", errors.Put(bagName, parseErrors(provider)))
	return r
}

// parseErrors turns any of the shapes WithErrors accepts into a
// *support.MessageBag.
func parseErrors(provider any) *support.MessageBag {
	switch typed := provider.(type) {
	case nil:
		return support.NewMessageBag(nil)
	case support.MessageProvider:
		return typed.GetMessageBag()
	case *support.MessageBag:
		return typed
	case validation.Errors:
		return support.NewMessageBag(map[string][]string(typed))
	case map[string][]string:
		return support.NewMessageBag(typed)
	case map[string]string:
		messages := make(map[string][]string, len(typed))
		for field, message := range typed {
			messages[field] = []string{message}
		}
		return support.NewMessageBag(messages)
	case []string:
		return support.NewMessageBag(map[string][]string{support.DefaultErrorBag: typed})
	case string:
		return support.NewMessageBag(map[string][]string{support.DefaultErrorBag: {typed}})
	}
	return support.NewMessageBag(map[string][]string{support.DefaultErrorBag: {stringify(provider)}})
}

// WithFragment puts a "#anchor" on the target, replacing one that is already
// there.
func (r *RedirectResponse) WithFragment(fragment string) *RedirectResponse {
	return r.WithoutFragment().
		SetTargetUrl(r.GetTargetUrl() + "#" + strAfter(fragment, "#"))
}

// WithoutFragment drops any "#anchor" from the target.
func (r *RedirectResponse) WithoutFragment() *RedirectResponse {
	return r.SetTargetUrl(strBefore(r.GetTargetUrl(), "#"))
}

// EnforceSameOrigin replaces the target with the fallback unless it is on
// the same origin as the request.
//
// It is the open-redirect defence on the Response side, and it answers the
// same question [LocalPath] answers on the Request side -- by comparing
// hosts rather than by refusing a host at all. A framework redirect built
// from a path should still go through LocalPath; this is for the target
// that genuinely carries a host.
//
// The variadic arguments are validateScheme and validatePort, both
// defaulting to true.
func (r *RedirectResponse) EnforceSameOrigin(fallback string, args ...bool) *RedirectResponse {
	validateScheme, validatePort := true, true
	if len(args) > 0 {
		validateScheme = args[0]
	}
	if len(args) > 1 {
		validatePort = args[1]
	}

	target, err := url.Parse(r.targetURL)
	if err != nil {
		return r.SetTargetUrl(fallback)
	}
	// A target with no host is already a path on this origin, which is the one
	// shape that cannot leave it.
	if target.Host == "" && target.Scheme == "" {
		return r
	}
	if r.request == nil {
		return r.SetTargetUrl(fallback)
	}
	current, err := url.Parse(r.request.SchemeAndHttpHost())
	if err != nil {
		return r.SetTargetUrl(fallback)
	}

	if !strings.EqualFold(target.Hostname(), current.Hostname()) ||
		(validateScheme && !strings.EqualFold(target.Scheme, current.Scheme)) ||
		(validatePort && target.Port() != current.Port()) {
		return r.SetTargetUrl(fallback)
	}
	return r
}

// GetOriginalContent returns nil: a redirect has no body to have had an
// original.
func (r *RedirectResponse) GetOriginalContent() any { return nil }

// GetRequest is the request this redirect answers.
func (r *RedirectResponse) GetRequest() *Request { return r.request }

// SetRequest sets the request WithInput and OnlyInput read from.
func (r *RedirectResponse) SetRequest(request *Request) *RedirectResponse {
	r.request = request
	return r
}

// GetSession is the session store this redirect flashes into.
func (r *RedirectResponse) GetSession() *session.Store { return r.session }

// SetSession sets the session store With, WithInput and WithErrors flash
// into.
func (r *RedirectResponse) SetSession(store *session.Store) *RedirectResponse {
	r.session = store
	return r
}

// The next block redeclares the Response methods that return the receiver.
// Go has no covariant return type, so the promoted method from the embedded
// Response would hand back a *Response and end the chain; these keep the
// type so that
//
//	redirect("/home").With("status", "saved").WithCookie(c).WithFragment("top")
//
// compiles.

// Header is [Response.Header], typed for the chain.
func (r *RedirectResponse) Header(key string, values ...any) *RedirectResponse {
	r.Response.Header(key, values...)
	return r
}

// WithHeaders is [Response.WithHeaders], typed for the chain.
func (r *RedirectResponse) WithHeaders(headers stdhttp.Header) *RedirectResponse {
	r.Response.WithHeaders(headers)
	return r
}

// WithoutHeader is [Response.WithoutHeader], typed for the chain.
func (r *RedirectResponse) WithoutHeader(keys ...string) *RedirectResponse {
	r.Response.WithoutHeader(keys...)
	return r
}

// Cookie is [Response.Cookie], typed for the chain.
func (r *RedirectResponse) Cookie(cookie *stdhttp.Cookie) *RedirectResponse {
	return r.WithCookie(cookie)
}

// WithCookie is [Response.WithCookie], typed for the chain.
func (r *RedirectResponse) WithCookie(cookie *stdhttp.Cookie) *RedirectResponse {
	r.Response.WithCookie(cookie)
	return r
}

// WithoutCookie is [Response.WithoutCookie], typed for the chain.
func (r *RedirectResponse) WithoutCookie(name string, args ...string) *RedirectResponse {
	r.Response.WithoutCookie(name, args...)
	return r
}

// WithException is [Response.WithException], typed for the chain.
func (r *RedirectResponse) WithException(err error) *RedirectResponse {
	r.Response.WithException(err)
	return r
}
