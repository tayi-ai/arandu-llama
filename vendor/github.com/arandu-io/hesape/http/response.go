package http

import (
	"encoding/json"
	"fmt"
	stdhttp "net/http"
	"reflect"
	"strings"

	"github.com/arandu-io/hesape/http/exceptions"
	"github.com/arandu-io/hesape/support"
)

// Arrayable is a value that knows how to present itself as a map.
// SetContent turns one into JSON, which is the whole reason it is named
// here.
type Arrayable interface {
	// ToArray is the map representation.
	ToArray() map[string]any
}

// Renderable is a value that draws itself, a view being the one that
// matters. SetContent calls it rather than casting to string, so a failure
// inside a template is a failure and not an empty page.
//
// Render returns (string, error): a rendering failure is reported through
// the error rather than by panicking.
type Renderable interface {
	// Render draws the value and returns the result.
	Render() (string, error)
}

// JsonSerializable is a value that names what of itself is encoded. Go's
// encoding/json reaches the same end through json.Marshaler, which is
// checked too.
type JsonSerializable interface {
	// JsonSerialize is the value that gets encoded in this one's place.
	JsonSerialize() any
}

// Response is a status, a set of headers, a set of cookies and a body,
// built before anything is written to the wire.
//
// It is a value and not a stdhttp.ResponseWriter: a controller returns it,
// middleware may still add a header to it, and [Response.Send] is the one
// place it meets the standard library.
type Response struct {
	// status is the status code.
	status int
	// headers holds the response headers.
	headers stdhttp.Header
	// cookies are the cookies this response carries. They are a slice and
	// not a map because two cookies may share a name on different paths,
	// which is how WithoutCookie expires one.
	cookies []*stdhttp.Cookie
	// content is the encoded body.
	content string
	// original is what SetContent was handed, before it became JSON or
	// rendered HTML. It is what a test asserts on.
	original any
	// exception is the error that produced this answer, when one did.
	exception error
	// callback is the JSONP callback. It lives here and not on JsonResponse
	// because GetCallback reads it off the embedded Response, and only
	// JsonResponse ever sets it.
	callback string
	// protocolVersion is the response's HTTP protocol version string.
	// NewResponse sets "1.0".
	protocolVersion string
}

// NewResponse builds a Response.
//
// Returns (*Response, error): the error is set when the content cannot be
// encoded. The variadic arguments are status 200 and no headers.
func NewResponse(content any, args ...any) (*Response, error) {
	r := &Response{
		status:          stdhttp.StatusOK,
		headers:         stdhttp.Header{},
		protocolVersion: "1.0",
	}
	if len(args) > 0 {
		if status, ok := args[0].(int); ok {
			r.status = status
		}
	}
	if len(args) > 1 {
		switch headers := args[1].(type) {
		case stdhttp.Header:
			for key, values := range headers {
				r.headers[stdhttp.CanonicalHeaderKey(key)] = append([]string(nil), values...)
			}
		case map[string]string:
			for key, value := range headers {
				r.headers.Set(key, value)
			}
		}
	}
	if _, err := r.SetContent(content); err != nil {
		return nil, err
	}
	return r, nil
}

// GetContent is the encoded body, empty when there is none.
func (r *Response) GetContent() string { return r.content }

// SetContent sets the body, encoding as needed.
//
// It keeps what it was handed on original: a value that is "JSONable" --
// Arrayable, Jsonable, JsonSerializable, json.Marshaler, a map or a slice --
// becomes JSON and sets the Content-Type; a Renderable is rendered;
// anything else is cast to a string.
//
// Returns (*Response, error): the error is set when JSON encoding fails.
func (r *Response) SetContent(content any) (*Response, error) {
	r.original = content

	switch typed := content.(type) {
	case nil:
		r.content = ""
		return r, nil
	case string:
		r.content = typed
		return r, nil
	case []byte:
		r.content = string(typed)
		return r, nil
	}

	if shouldBeJson(content) {
		r.Header("Content-Type", "application/json")
		encoded, err := morphToJson(content)
		if err != nil {
			return nil, fmt.Errorf("http: the response content could not be encoded as JSON: %w", err)
		}
		r.content = encoded
		return r, nil
	}

	if renderable, ok := content.(Renderable); ok {
		rendered, err := renderable.Render()
		if err != nil {
			return nil, fmt.Errorf("http: the response content could not be rendered: %w", err)
		}
		r.content = rendered
		return r, nil
	}

	r.content = stringify(content)
	return r, nil
}

// shouldBeJson reports whether the content is a value that becomes JSON
// rather than being cast to a string.
func shouldBeJson(content any) bool {
	switch content.(type) {
	case Arrayable, support.Jsonable, JsonSerializable, json.Marshaler:
		return true
	}
	// A map, a slice and an array are what turns into JSON here. A struct is
	// deliberately NOT included: a Renderable is usually a struct, and it is
	// checked separately, after this one.
	kind := reflect.ValueOf(content).Kind()
	return kind == reflect.Map || kind == reflect.Slice || kind == reflect.Array
}

// morphToJson encodes a JSONable value (see shouldBeJson) to its JSON
// string.
func morphToJson(content any) (string, error) {
	switch typed := content.(type) {
	case support.Jsonable:
		return typed.ToJson()
	case Arrayable:
		encoded, err := json.Marshal(typed.ToArray())
		return string(encoded), err
	case JsonSerializable:
		encoded, err := json.Marshal(typed.JsonSerialize())
		return string(encoded), err
	}
	encoded, err := json.Marshal(content)
	return string(encoded), err
}

// Status is the status code.
func (r *Response) Status() int { return r.status }

// GetStatusCode is an alias for Status.
func (r *Response) GetStatusCode() int { return r.status }

// SetStatusCode sets the status code.
func (r *Response) SetStatusCode(code int) *Response {
	r.status = code
	return r
}

// StatusText is the reason phrase.
func (r *Response) StatusText() string { return stdhttp.StatusText(r.status) }

// Content is an alias for GetContent.
func (r *Response) Content() string { return r.GetContent() }

// GetOriginalContent is what SetContent was handed, before it became JSON
// or HTML. A Response wrapping a Response unwraps.
func (r *Response) GetOriginalContent() any {
	if nested, ok := r.original.(*Response); ok {
		return nested.GetOriginalContent()
	}
	return r.original
}

// Header sets a header, replacing what is there unless replace is false.
func (r *Response) Header(key string, values ...any) *Response {
	if r.headers == nil {
		r.headers = stdhttp.Header{}
	}
	replace := true
	list := make([]string, 0, len(values))
	for i, value := range values {
		// A trailing bool is the replace flag and never a header value, so
		// values ends with one only when the caller means to set it.
		if flag, ok := value.(bool); ok && i == len(values)-1 && i > 0 {
			replace = flag
			continue
		}
		switch typed := value.(type) {
		case []string:
			list = append(list, typed...)
		default:
			list = append(list, stringify(value))
		}
	}
	if replace {
		r.headers.Del(key)
	}
	for _, value := range list {
		r.headers.Add(key, value)
	}
	return r
}

// WithHeaders adds several headers at once.
func (r *Response) WithHeaders(headers stdhttp.Header) *Response {
	if r.headers == nil {
		r.headers = stdhttp.Header{}
	}
	for key, values := range headers {
		r.headers.Del(key)
		for _, value := range values {
			r.headers.Add(key, value)
		}
	}
	return r
}

// WithoutHeader removes headers.
func (r *Response) WithoutHeader(keys ...string) *Response {
	for _, key := range keys {
		r.headers.Del(key)
	}
	return r
}

// Headers is the response's headers. It is a getter because a Go field
// would let a caller swap the whole map.
func (r *Response) Headers() stdhttp.Header {
	if r.headers == nil {
		r.headers = stdhttp.Header{}
	}
	return r.headers
}

// Cookie is an alias for WithCookie.
func (r *Response) Cookie(cookie *stdhttp.Cookie) *Response { return r.WithCookie(cookie) }

// WithCookie adds a cookie.
func (r *Response) WithCookie(cookie *stdhttp.Cookie) *Response {
	if cookie != nil {
		r.cookies = append(r.cookies, cookie)
	}
	return r
}

// WithoutCookie expires a cookie when the response is sent. MaxAge -1 is
// what tells a browser to delete it immediately.
//
// The variadic arguments are an optional path and domain.
func (r *Response) WithoutCookie(name string, args ...string) *Response {
	expired := &stdhttp.Cookie{Name: name, Value: "", MaxAge: -1}
	if len(args) > 0 {
		expired.Path = args[0]
	}
	if len(args) > 1 {
		expired.Domain = args[1]
	}
	return r.WithCookie(expired)
}

// Cookies returns the cookies this response carries. [Response.Send] writes
// them.
func (r *Response) Cookies() []*stdhttp.Cookie { return r.cookies }

// GetCallback is the JSONP callback, empty when there is none. Only a
// JsonResponse ever sets one.
func (r *Response) GetCallback() string { return r.callback }

// WithException records the error that produced this answer.
func (r *Response) WithException(err error) *Response {
	r.exception = err
	return r
}

// Exception returns the error WithException recorded.
func (r *Response) Exception() error { return r.exception }

// ThrowResponse wraps this response in an HttpResponseException so the
// layer above sends it instead of rendering.
//
// The error is returned rather than thrown; the caller returns it in turn.
func (r *Response) ThrowResponse() error {
	return exceptions.NewHttpResponseException(r)
}

// SetProtocolVersion sets the HTTP protocol version string. NewResponse
// calls it with "1.0".
func (r *Response) SetProtocolVersion(version string) *Response {
	r.protocolVersion = version
	return r
}

// GetProtocolVersion is the HTTP protocol version string.
func (r *Response) GetProtocolVersion() string { return r.protocolVersion }

// Send writes the status, the headers, the cookies and the body to the
// wire.
//
// It is the one place this type meets stdhttp.ResponseWriter, which is the
// point of building a Response at all: everything before this is a value that
// middleware may still change.
func (r *Response) Send(w stdhttp.ResponseWriter) error {
	header := w.Header()
	for key, values := range r.headers {
		header.Del(key)
		for _, value := range values {
			header.Add(key, value)
		}
	}
	for _, cookie := range r.cookies {
		stdhttp.SetCookie(w, cookie)
	}
	status := r.status
	if status == 0 {
		status = stdhttp.StatusOK
	}
	w.WriteHeader(status)
	if r.content == "" {
		return nil
	}
	_, err := w.Write([]byte(r.content))
	return err
}

// ServeHTTP makes a Response a stdhttp.Handler, so a route may answer with one
// directly. It is Send with the arguments a handler is given.
func (r *Response) ServeHTTP(w stdhttp.ResponseWriter, _ *stdhttp.Request) { _ = r.Send(w) }

// StreamedEvent is one named event on a server-sent event stream.
type StreamedEvent struct {
	// Event is the name of the event.
	Event string
	// Data is the payload.
	Data any
}

// NewStreamedEvent builds a StreamedEvent.
func NewStreamedEvent(event string, data any) *StreamedEvent {
	return &StreamedEvent{Event: event, Data: data}
}

// String formats the event the way the server-sent event wire format wants it:
// an "event:" line, one "data:" line per line of payload, and a blank line.
//
// It is here and not in a writer of its own because StreamedEvent is two
// fields and nothing else, and the framework that reads it needs the bytes.
func (e *StreamedEvent) String() string {
	var out strings.Builder
	if e.Event != "" {
		out.WriteString("event: ")
		out.WriteString(e.Event)
		out.WriteString("\n")
	}
	payload := stringify(e.Data)
	if shouldBeJson(e.Data) {
		if encoded, err := morphToJson(e.Data); err == nil {
			payload = encoded
		}
	}
	for _, line := range strings.Split(payload, "\n") {
		out.WriteString("data: ")
		out.WriteString(line)
		out.WriteString("\n")
	}
	out.WriteString("\n")
	return out.String()
}
