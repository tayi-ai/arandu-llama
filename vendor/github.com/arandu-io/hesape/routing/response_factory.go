package routing

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"iter"
	"net/http"
	"os"
	"strings"
	"unicode"

	hhttp "github.com/arandu-io/hesape/http"
	"github.com/arandu-io/hesape/routing/exceptions"
)

// ResponseFactory is the one place a handler asks for an answer of a given
// shape -- a body, a view, JSON, a stream, a file, a redirect -- instead of
// assembling one by hand.
//
// It holds two collaborators: a [ViewRenderer], which is this package's seam
// onto the view layer, and a [Redirector]. Every method named Redirect* is a
// delegation to that redirector and nothing else -- there is one redirect
// mechanism in this package, not two, which is why those methods answer with
// the [Redirect] the Redirector builds rather than with an
// hhttp.RedirectResponse.
type ResponseFactory struct {
	// view is this package's seam onto the view layer.
	view ViewRenderer
	// redirector builds every Redirect* response.
	redirector *Redirector
}

// NewResponseFactory builds a ResponseFactory from its two collaborators.
//
// Either collaborator may be nil, and the method that needs the missing one
// says so rather than panicking: a project that never returns a view has no
// renderer to give.
func NewResponseFactory(view ViewRenderer, redirector *Redirector) *ResponseFactory {
	return &ResponseFactory{view: view, redirector: redirector}
}

// Make builds a response around a body.
//
// A status of 0 means 200 with no headers, as it does everywhere else in this
// package. It returns (*hhttp.Response, error) so a body that cannot be
// encoded is reported rather than panicking.
func (f *ResponseFactory) Make(content any, status int, headers http.Header) (*hhttp.Response, error) {
	return hhttp.NewResponse(content, responseFactoryStatus(status, http.StatusOK), responseFactoryHeaders(headers))
}

// NoContent builds an empty body under a status that says there is nothing to
// read. A status of 0 means 204.
func (f *ResponseFactory) NoContent(status int, headers http.Header) (*hhttp.Response, error) {
	return f.Make("", responseFactoryStatus(status, http.StatusNoContent), headers)
}

// View renders a named view and answers with what it drew.
//
// It goes through [ViewRenderer], which is the seam this package already uses
// to reach the view layer without importing it.
//
// This takes a single view name; a caller that wants a fallback among several
// names chooses one before calling.
//
// It returns (*hhttp.Response, error), so a missing or failing template is
// reported rather than panicking.
func (f *ResponseFactory) View(view string, data any, status int, headers http.Header) (*hhttp.Response, error) {
	if f.view == nil {
		return nil, errors.New("routing: expected a view renderer on the response factory, got none")
	}
	status = responseFactoryStatus(status, http.StatusOK)
	headers = responseFactoryHeaders(headers)

	var body bytes.Buffer
	if err := f.view.Render(&body, view, data, status, headers); err != nil {
		return nil, fmt.Errorf("routing: the view %q could not be rendered: %w", view, err)
	}
	return f.Make(body.String(), status, headers)
}

// JSON builds a response whose body is data encoded as JSON.
//
// options is a bitmask of hhttp's JSON* flags. encoding/json has no
// counterpart for a hex-escaping default, so 0 -- no flags -- is the default
// here.
//
// It returns (*hhttp.JsonResponse, error), so data that cannot be encoded is
// reported rather than panicking.
func (f *ResponseFactory) JSON(data any, status int, headers http.Header, options int) (*hhttp.JsonResponse, error) {
	return hhttp.NewJsonResponse(data, responseFactoryStatus(status, http.StatusOK), responseFactoryHeaders(headers), options)
}

// JSONP is [JSON] with the payload wrapped in a call to the named callback,
// which turns the Content-Type into text/javascript.
func (f *ResponseFactory) JSONP(callback string, data any, status int, headers http.Header, options int) (*hhttp.JsonResponse, error) {
	response, err := f.JSON(data, status, headers, options)
	if err != nil {
		return nil, err
	}
	return response.SetCallback(callback), nil
}

// Stream builds a response whose body is written as it is produced, rather
// than assembled and then sent.
//
// The callback is handed an io.Writer that flushes after every write, so the
// bytes reach the client as soon as they are written. It also receives a
// context, because a stream is I/O that outlives the call that started it and
// the client may hang up mid-body.
func (f *ResponseFactory) Stream(callback StreamCallback, status int, headers http.Header) *StreamedResponse {
	return &StreamedResponse{
		Callback: callback,
		Status:   responseFactoryStatus(status, http.StatusOK),
		Headers:  responseFactoryHeaders(headers),
	}
}

// StreamJSON encodes the value straight onto the wire instead of building the
// whole document in memory first.
//
// It answers with a [StreamedResponse] whose callback is the encoder, rather
// than a dedicated streaming-JSON response type.
//
// encodingOptions is the same flag word [JSON] takes.
func (f *ResponseFactory) StreamJSON(data any, status int, headers http.Header, encodingOptions int) *StreamedResponse {
	response := f.Stream(func(_ context.Context, w io.Writer) error {
		return responseFactoryEncodeJSON(w, data, encodingOptions)
	}, status, headers)

	if response.Headers.Get("Content-Type") == "" {
		response.Headers.Set("Content-Type", "application/json")
	}
	return response
}

// EventStream builds a server-sent event stream, with the three headers that
// stop a proxy or a browser from buffering it.
//
// callback is an iter.Seq: a producer that is pulled, not a slice built in
// advance. A message that is an *hhttp.StreamedEvent names its own event;
// anything else is sent under "update", and a value that is neither a string
// nor a number is encoded as JSON.
//
// endStreamWith is optional and defaults to "</stream>": omit it for that
// default, and pass nil to close the stream without a final event. Passing an
// *hhttp.StreamedEvent names the closing event.
//
// The iteration stops when the request context is done.
func (f *ResponseFactory) EventStream(callback iter.Seq[any], headers http.Header, endStreamWith ...any) *StreamedResponse {
	end := any(responseFactoryDefaultEndStream)
	if len(endStreamWith) > 0 {
		end = endStreamWith[0]
	}

	merged := responseFactoryHeaders(headers)
	merged.Set("Content-Type", "text/event-stream")
	merged.Set("Cache-Control", "no-cache")
	merged.Set("X-Accel-Buffering", "no")

	return f.Stream(func(ctx context.Context, w io.Writer) error {
		var failure error
		if callback != nil {
			callback(func(message any) bool {
				if ctx.Err() != nil {
					return false
				}
				if _, err := io.WriteString(w, responseFactoryEvent(message)); err != nil {
					failure = err
					return false
				}
				return true
			})
		}
		if failure != nil {
			return failure
		}
		// The closing event is written outside the loop, so it happens even
		// when the loop stopped early.
		if responseFactoryFilled(end) {
			if _, err := io.WriteString(w, responseFactoryEvent(end)); err != nil {
				return err
			}
		}
		return nil
	}, http.StatusOK, merged)
}

// StreamDownload is a [Stream] the browser saves instead of showing, because
// of the Content-Disposition header.
//
// By the time the callback runs, the status and the headers are already sent
// and there is no error page left to send, so a failure is wrapped in
// exceptions.StreamedResponseError rather than returned bare.
//
// An empty name leaves the header off. An empty disposition defaults to
// "attachment".
func (f *ResponseFactory) StreamDownload(callback StreamCallback, name string, headers http.Header, disposition string) *StreamedResponse {
	response := f.Stream(callback, http.StatusOK, headers)
	response.wrapErrors = true

	if name != "" {
		response.Headers.Set("Content-Disposition", responseFactoryMakeDisposition(
			responseFactoryDisposition(disposition),
			name,
			responseFactoryFallbackName(name),
		))
	}
	return response
}

// Download sends a file from disk under a Content-Disposition that makes the
// browser save it.
//
// It takes the path, and the file is opened when the response is sent rather
// than read into memory when it is built -- [BinaryFileResponse] serves it
// through http.ServeContent, so a range request and a conditional request are
// answered the way the standard library answers them.
//
// An empty name means the file's own base name. An empty disposition defaults
// to "attachment".
func (f *ResponseFactory) Download(path, name string, headers http.Header, disposition string) *BinaryFileResponse {
	response := &BinaryFileResponse{Path: path, Headers: responseFactoryHeaders(headers)}
	if name == "" {
		name = responseFactoryBaseName(path)
	}
	return response.SetContentDisposition(responseFactoryDisposition(disposition), name, responseFactoryFallbackName(name))
}

// File sends the raw contents of a file, with no Content-Disposition, so a
// PDF or an image is shown rather than saved.
//
// It takes the path, for the reason [Download] does.
func (f *ResponseFactory) File(path string, headers http.Header) *BinaryFileResponse {
	return &BinaryFileResponse{Path: path, Headers: responseFactoryHeaders(headers)}
}

// RedirectTo delegates to Redirector.To.
func (f *ResponseFactory) RedirectTo(path string, status int, headers http.Header, secure *bool) (*Redirect, error) {
	if f.redirector == nil {
		return nil, responseFactoryNoRedirector()
	}
	return f.redirector.To(path, responseFactoryStatus(status, http.StatusFound), headers, secure), nil
}

// RedirectToRoute delegates to Redirector.Route.
func (f *ResponseFactory) RedirectToRoute(route string, parameters map[string]any, status int, headers http.Header) (*Redirect, error) {
	if f.redirector == nil {
		return nil, responseFactoryNoRedirector()
	}
	return f.redirector.Route(route, parameters, responseFactoryStatus(status, http.StatusFound), headers)
}

// RedirectToAction delegates to Redirector.Action.
func (f *ResponseFactory) RedirectToAction(action string, parameters map[string]any, status int, headers http.Header) (*Redirect, error) {
	if f.redirector == nil {
		return nil, responseFactoryNoRedirector()
	}
	return f.redirector.Action(action, parameters, responseFactoryStatus(status, http.StatusFound), headers)
}

// RedirectGuest delegates to Redirector.Guest: the redirect that remembers
// where the visitor was going, so a sign-in can send them back there.
func (f *ResponseFactory) RedirectGuest(path string, status int, headers http.Header, secure *bool) (*Redirect, error) {
	if f.redirector == nil {
		return nil, responseFactoryNoRedirector()
	}
	return f.redirector.Guest(path, responseFactoryStatus(status, http.StatusFound), headers, secure), nil
}

// RedirectToIntended delegates to Redirector.Intended: the other half of
// [RedirectGuest].
//
// An empty default falls back to "/".
func (f *ResponseFactory) RedirectToIntended(def string, status int, headers http.Header, secure *bool) (*Redirect, error) {
	if f.redirector == nil {
		return nil, responseFactoryNoRedirector()
	}
	if def == "" {
		def = "/"
	}
	return f.redirector.Intended(def, responseFactoryStatus(status, http.StatusFound), headers, secure), nil
}

// StreamCallback is the callback [ResponseFactory.Stream], StreamDownload and
// EventStream take. It writes to w, which flushes after every write.
//
// The context is the request's: it is done when the client hangs up.
type StreamCallback func(ctx context.Context, w io.Writer) error

// StreamedResponse is a status, a set of headers and a body that is produced
// while it is being sent. [ResponseFactory.Stream], StreamJSON, EventStream
// and StreamDownload all build one.
//
// The fields are public because the whole point of returning a response instead
// of writing one is that a caller may still add a header to it.
type StreamedResponse struct {
	// Callback writes the body.
	Callback StreamCallback
	// Status is the status code, 200 when zero.
	Status int
	// Headers are sent before the first byte of the body.
	Headers http.Header
	// wrapErrors reports whether a failure from the callback is wrapped in an
	// exceptions.StreamedResponseError. Only StreamDownload sets it.
	wrapErrors bool
}

// SendContent writes the headers, the status and then the body, flushing as it
// goes.
//
// It returns what the callback returned, which ServeHTTP cannot: by the time
// the callback fails the status line is already sent, so there is nowhere left
// to report it to the client, and a caller that wants to log it needs this.
func (r *StreamedResponse) SendContent(w http.ResponseWriter, req *http.Request) error {
	if w == nil {
		return errors.New("routing: expected a response writer to stream into, got none")
	}
	responseFactoryApplyHeaders(w, r.Headers)

	status := responseFactoryStatus(r.Status, http.StatusOK)
	w.WriteHeader(status)

	if r.Callback == nil {
		return nil
	}

	ctx := context.Background()
	if req != nil {
		ctx = req.Context()
	}

	out := &responseFactoryFlushWriter{w: w}
	if flusher, ok := w.(http.Flusher); ok {
		out.flusher = flusher
	}
	// The headers reach the client before the first message, which is what an
	// event stream needs to start being read at all.
	out.Flush()

	if err := r.Callback(ctx, out); err != nil {
		if r.wrapErrors {
			return &exceptions.StreamedResponseError{Err: err}
		}
		return err
	}
	return nil
}

// ServeHTTP makes a StreamedResponse an http.Handler, so a route may answer
// with one directly. It is SendContent with the arguments a handler is given,
// mirroring how hhttp.Response does it.
func (r *StreamedResponse) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	_ = r.SendContent(w, req)
}

// BinaryFileResponse is a path on disk plus the headers to send it under.
// [ResponseFactory.Download] and File both build one.
//
// The bytes are read when the response is sent, never before, so a large file
// costs a descriptor and not memory.
type BinaryFileResponse struct {
	// Path is the file to send.
	Path string
	// Headers are sent alongside the ones http.ServeContent derives from the
	// file itself.
	Headers http.Header
}

// SetContentDisposition says whether the browser saves the file or shows it,
// and under what name.
//
// The fallback is the ASCII name a client that does not understand RFC 5987
// falls back to, which is what [responseFactoryFallbackName] produces.
func (r *BinaryFileResponse) SetContentDisposition(disposition, name, fallback string) *BinaryFileResponse {
	if r.Headers == nil {
		r.Headers = http.Header{}
	}
	r.Headers.Set("Content-Disposition", responseFactoryMakeDisposition(disposition, name, fallback))
	return r
}

// SendContent opens the file and hands it to http.ServeContent, which picks
// the status -- 200, 206 for a range request, 304 for a conditional one --
// and streams the bytes.
func (r *BinaryFileResponse) SendContent(w http.ResponseWriter, req *http.Request) error {
	if w == nil {
		return errors.New("routing: expected a response writer to send a file into, got none")
	}
	if req == nil {
		return errors.New("routing: expected a request to serve a file against, got none")
	}

	file, err := os.Open(r.Path)
	if err != nil {
		return fmt.Errorf("routing: the file %q could not be opened to be sent: %w", r.Path, err)
	}
	defer func() { _ = file.Close() }()

	info, err := file.Stat()
	if err != nil {
		return fmt.Errorf("routing: the file %q could not be inspected to be sent: %w", r.Path, err)
	}
	if info.IsDir() {
		return fmt.Errorf("routing: expected a file to send, got the directory %q", r.Path)
	}

	responseFactoryApplyHeaders(w, r.Headers)
	http.ServeContent(w, req, info.Name(), info.ModTime(), file)
	return nil
}

// ServeHTTP makes a BinaryFileResponse an http.Handler. A file that cannot be
// opened answers 404.
func (r *BinaryFileResponse) ServeHTTP(w http.ResponseWriter, req *http.Request) {
	if err := r.SendContent(w, req); err != nil {
		http.Error(w, http.StatusText(http.StatusNotFound), http.StatusNotFound)
	}
}

// responseFactoryDefaultEndStream is the default value of EventStream's
// endStreamWith argument.
const responseFactoryDefaultEndStream = "</stream>"

// responseFactoryDefaultEvent is the event name eventStream sends a message
// under when the message did not name one.
const responseFactoryDefaultEvent = "update"

// responseFactoryNoRedirector is the error every Redirect* method answers with
// when the factory was built without one.
func responseFactoryNoRedirector() error {
	return errors.New("routing: expected a redirector on the response factory, got none")
}

// responseFactoryStatus applies a default: zero means def.
func responseFactoryStatus(status, def int) int {
	if status == 0 {
		return def
	}
	return status
}

// responseFactoryDisposition applies the default disposition, "attachment".
func responseFactoryDisposition(disposition string) string {
	if disposition == "" {
		return "attachment"
	}
	return disposition
}

// responseFactoryHeaders copies the caller's headers into a bag this package
// owns, canonicalised and never nil, so that setting Content-Type on a stream
// does not reach back into the map the caller still holds.
func responseFactoryHeaders(headers http.Header) http.Header {
	out := http.Header{}
	for key, values := range headers {
		out[http.CanonicalHeaderKey(key)] = append([]string(nil), values...)
	}
	return out
}

// responseFactoryApplyHeaders writes a header bag onto a response writer,
// replacing rather than appending.
func responseFactoryApplyHeaders(w http.ResponseWriter, headers http.Header) {
	target := w.Header()
	for key, values := range headers {
		target.Del(key)
		for _, value := range values {
			target.Add(key, value)
		}
	}
}

// responseFactoryFlushWriter is the io.Writer a StreamCallback is handed:
// every write reaches the client immediately.
type responseFactoryFlushWriter struct {
	w       io.Writer
	flusher http.Flusher
}

func (fw *responseFactoryFlushWriter) Write(p []byte) (int, error) {
	n, err := fw.w.Write(p)
	fw.Flush()
	return n, err
}

// Flush makes the writer an http.Flusher too, so a callback that batches its
// own writes can still say when a message is complete.
func (fw *responseFactoryFlushWriter) Flush() {
	if fw.flusher != nil {
		fw.flusher.Flush()
	}
}

// responseFactoryEvent frames one server-sent event.
//
// hhttp.StreamedEvent already knows the wire format and already encodes a
// payload that is neither a string nor a number as JSON, so this only supplies
// the default event name.
func responseFactoryEvent(message any) string {
	if event, ok := message.(*hhttp.StreamedEvent); ok && event != nil {
		name := event.Event
		if name == "" {
			name = responseFactoryDefaultEvent
		}
		// Copied rather than mutated: the caller still owns the event it
		// yielded.
		return (&hhttp.StreamedEvent{Event: name, Data: event.Data}).String()
	}
	return (&hhttp.StreamedEvent{Event: responseFactoryDefaultEvent, Data: message}).String()
}

// responseFactoryFilled reports whether value should be treated as present:
// nil and the empty string are not filled.
func responseFactoryFilled(value any) bool {
	switch typed := value.(type) {
	case nil:
		return false
	case string:
		return strings.TrimSpace(typed) != ""
	case *hhttp.StreamedEvent:
		return typed != nil
	}
	return true
}

// responseFactoryEncodeJSON writes the value as JSON, honouring the flags of
// the options word that encoding/json can express -- the same two hhttp's
// JsonResponse honours, for the same reason.
func responseFactoryEncodeJSON(w io.Writer, data any, options int) error {
	encoder := json.NewEncoder(w)
	if options&(hhttp.JSONUnescapedSlashes|hhttp.JSONUnescapedUnicode) != 0 {
		encoder.SetEscapeHTML(false)
	}
	if options&hhttp.JSONPrettyPrint != 0 {
		encoder.SetIndent("", "    ")
	}
	if err := encoder.Encode(data); err != nil {
		return fmt.Errorf("routing: the streamed payload could not be encoded as JSON: %w", err)
	}
	return nil
}

// responseFactoryBaseName is the file's own name, used by Download when given
// no name.
func responseFactoryBaseName(path string) string {
	// Deliberately not filepath.Base: the name goes into an HTTP header, where
	// the separator is "/" whatever the server's operating system thinks.
	trimmed := strings.TrimRight(path, "/")
	if index := strings.LastIndexAny(trimmed, `/\`); index >= 0 {
		return trimmed[index+1:]
	}
	return trimmed
}

// responseFactoryFallbackName is the ASCII name a client that does not read
// RFC 5987 falls back to.
//
// A character outside ASCII is dropped rather than transliterated, since this
// module carries no transliteration table; the real name still travels, in
// the filename* parameter. "/" and "\" are dropped too, along with the
// control characters.
func responseFactoryFallbackName(name string) string {
	var out strings.Builder
	for _, r := range name {
		if r > unicode.MaxASCII || r == '%' || r == '/' || r == '\\' {
			continue
		}
		if r < 0x20 || r == 0x7f {
			continue
		}
		out.WriteRune(r)
	}
	if out.Len() == 0 {
		return "file"
	}
	return out.String()
}

// responseFactoryMakeDisposition builds the Content-Disposition header value
// that StreamDownload and Download both send.
func responseFactoryMakeDisposition(disposition, name, fallback string) string {
	quoted := strings.NewReplacer(`\`, `\\`, `"`, `\"`).Replace(fallback)
	out := disposition + `; filename="` + quoted + `"`
	if name != fallback {
		out += "; filename*=utf-8''" + responseFactoryEncodeRFC5987(name)
	}
	return out
}

// responseFactoryEncodeRFC5987 percent-encodes everything outside the attr-char
// set of RFC 5987, which is what the filename* parameter takes.
func responseFactoryEncodeRFC5987(value string) string {
	const hex = "0123456789ABCDEF"
	const unreserved = "!#$&+-.^_`|~"

	var out strings.Builder
	for _, b := range []byte(value) {
		switch {
		case b >= 'a' && b <= 'z', b >= 'A' && b <= 'Z', b >= '0' && b <= '9',
			strings.IndexByte(unreserved, b) >= 0:
			out.WriteByte(b)
		default:
			out.WriteByte('%')
			out.WriteByte(hex[b>>4])
			out.WriteByte(hex[b&0x0f])
		}
	}
	return out.String()
}
