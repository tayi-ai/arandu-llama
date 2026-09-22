package http

import (
	"bytes"
	"encoding/json"
	"fmt"
	stdhttp "net/http"

	"github.com/arandu-io/hesape/support"
)

// The encoding flags are bit values JsonResponse takes as its options
// argument, and HasEncodingOption tests a caller's payload against. The
// numbers are fixed: a caller reads back a flag it set through
// HasEncodingOption, and renumbering the constants would answer a different
// question for anyone who already depends on the current values.
//
// encoding/json has no flag word, so only the flags that change what Go can
// emit are honoured: JSONPrettyPrint indents, JSONUnescapedSlashes and
// JSONUnescapedUnicode turn off Go's HTML escaping, and
// JSONPartialOutputOnError is read by SetData when deciding whether an
// encoding failure is fatal. The rest are accepted, stored and reported by
// HasEncodingOption, and otherwise change nothing.
const (
	// JSONHexTag is the "hex tag" encoding flag: escape "<" and ">".
	JSONHexTag = 1
	// JSONHexAmp is the "hex amp" encoding flag: escape "&".
	JSONHexAmp = 2
	// JSONHexApos is the "hex apos" encoding flag: escape "'".
	JSONHexApos = 4
	// JSONHexQuot is the "hex quot" encoding flag: escape the double quote.
	JSONHexQuot = 8
	// JSONForceObject forces an empty value or a map to encode as a JSON
	// object rather than an array.
	JSONForceObject = 16
	// JSONNumericCheck encodes a numeric string as a number.
	JSONNumericCheck = 32
	// JSONUnescapedSlashes and JSONUnescapedUnicode both mean "stop
	// rewriting characters that were already valid", which in encoding/json
	// is the one switch SetEscapeHTML: setting either flag turns it off.
	JSONUnescapedSlashes = 64
	// JSONPrettyPrint indents the encoded output.
	JSONPrettyPrint = 128
	// JSONUnescapedUnicode is documented on JSONUnescapedSlashes.
	JSONUnescapedUnicode = 256
	// JSONPartialOutputOnError falls back to an empty object instead of
	// failing when the value cannot be encoded. SetData reads it.
	JSONPartialOutputOnError = 512
	// JSONPreserveZeroFraction keeps a trailing ".0" on a float that has no
	// fractional part.
	JSONPreserveZeroFraction = 1024
	// JSONThrowOnError is accepted for compatibility; an encoding failure is
	// always reported through the returned error here.
	JSONThrowOnError = 4194304
)

// JsonResponse is a Response whose body is the JSON encoding of a value,
// with the encoding flags kept so that SetEncodingOptions can re-encode what
// is already there.
//
// It embeds Response, so Status, Header, WithCookie, ThrowResponse and the
// rest are the same methods on the same fields.
type JsonResponse struct {
	Response

	// data is the encoded payload, kept separate from the embedded
	// Response's content because the JSONP callback wraps it.
	data string
	// encodingOptions is the encoding flags this response was built or last
	// re-encoded with.
	encodingOptions int
}

// NewJsonResponse builds a JsonResponse.
//
// The variadic arguments are, in order: status (200), headers (none),
// options (0) and json (false -- whether data is already an encoded JSON
// string).
//
// Returns (*JsonResponse, error): the error is set when data cannot be
// encoded.
func NewJsonResponse(data any, args ...any) (*JsonResponse, error) {
	status := stdhttp.StatusOK
	var headers stdhttp.Header
	options := 0
	alreadyJSON := false

	if len(args) > 0 {
		if value, ok := args[0].(int); ok {
			status = value
		}
	}
	if len(args) > 1 {
		switch value := args[1].(type) {
		case stdhttp.Header:
			headers = value
		case map[string]string:
			headers = stdhttp.Header{}
			for key, item := range value {
				headers.Set(key, item)
			}
		}
	}
	if len(args) > 2 {
		if value, ok := args[2].(int); ok {
			options = value
		}
	}
	if len(args) > 3 {
		if value, ok := args[3].(bool); ok {
			alreadyJSON = value
		}
	}

	r := &JsonResponse{
		Response: Response{
			status:          status,
			headers:         stdhttp.Header{},
			protocolVersion: "1.0",
		},
		encodingOptions: options,
	}
	r.Response.headers.Set("Content-Type", "application/json")
	r.WithHeaders(headers)

	if alreadyJSON {
		r.setJSONString(stringify(data))
		return r, nil
	}
	if _, err := r.SetData(data); err != nil {
		return nil, err
	}
	return r, nil
}

// FromJsonString builds a response around a string that is already encoded
// JSON.
//
// The variadic arguments are status (200) and headers (none).
func FromJsonString(data string, args ...any) (*JsonResponse, error) {
	rest := make([]any, 0, 4)
	if len(args) > 0 {
		rest = append(rest, args[0])
	} else {
		rest = append(rest, stdhttp.StatusOK)
	}
	if len(args) > 1 {
		rest = append(rest, args[1])
	} else {
		rest = append(rest, stdhttp.Header(nil))
	}
	rest = append(rest, 0, true)
	return NewJsonResponse(data, rest...)
}

// GetData is the payload decoded back out of the encoded body.
//
// The variadic args are accepted and ignored: they exist for a caller
// passing a decoding mode and a depth limit, neither of which encoding/json
// needs -- it always decodes into map[string]any and has no depth limit. An
// empty body decodes as (nil, nil).
func (r *JsonResponse) GetData(args ...any) (any, error) {
	var decoded any
	if r.data == "" {
		return nil, nil
	}
	if err := json.Unmarshal([]byte(r.data), &decoded); err != nil {
		return nil, fmt.Errorf("http: the response payload is not valid JSON: %w", err)
	}
	return decoded, nil
}

// SetData encodes the value and makes it the body. It keeps the value on
// original, which is what GetOriginalContent returns.
//
// Returns an error when encoding fails, unless JSONPartialOutputOnError is
// set, in which case it falls back to an empty object instead.
func (r *JsonResponse) SetData(data any) (*JsonResponse, error) {
	r.original = data

	encoded, err := r.encode(data)
	if err != nil {
		if r.HasEncodingOption(JSONPartialOutputOnError) {
			r.setJSONString("{}")
			return r, nil
		}
		return nil, fmt.Errorf("http: the response payload could not be encoded as JSON: %w", err)
	}
	r.setJSONString(encoded)
	return r, nil
}

// encode applies the encoding flags encoding/json can express.
func (r *JsonResponse) encode(data any) (string, error) {
	switch typed := data.(type) {
	case support.Jsonable:
		return typed.ToJson()
	case Arrayable:
		data = typed.ToArray()
	case JsonSerializable:
		data = typed.JsonSerialize()
	}

	var buffer bytes.Buffer
	encoder := json.NewEncoder(&buffer)
	// JSONUnescapedSlashes and JSONUnescapedUnicode both mean "stop
	// rewriting characters that were already valid", which in encoding/json
	// is the one switch SetEscapeHTML.
	if r.encodingOptions&(JSONUnescapedSlashes|JSONUnescapedUnicode) != 0 {
		encoder.SetEscapeHTML(false)
	}
	if r.encodingOptions&JSONPrettyPrint != 0 {
		encoder.SetIndent("", "    ")
	}
	if err := encoder.Encode(data); err != nil {
		return "", err
	}
	// Encoder.Encode appends a trailing newline; it is trimmed so the body
	// has none.
	return string(bytes.TrimRight(buffer.Bytes(), "\n")), nil
}

// setJSONString writes the payload and rebuilds the body, wrapping it in the
// JSONP callback when there is one.
func (r *JsonResponse) setJSONString(data string) {
	r.data = data
	if r.callback != "" {
		r.Response.headers.Set("Content-Type", "text/javascript")
		r.content = "/**/" + r.callback + "(" + data + ");"
		return
	}
	r.Response.headers.Set("Content-Type", "application/json")
	r.content = data
}

// SetEncodingOptions changes the flags and re-encodes what is already there.
func (r *JsonResponse) SetEncodingOptions(options int) (*JsonResponse, error) {
	r.encodingOptions = options
	return r.SetData(r.original)
}

// GetEncodingOptions is the encoding flags currently set.
func (r *JsonResponse) GetEncodingOptions() int { return r.encodingOptions }

// HasEncodingOption reports whether a flag is set in the encoding options.
func (r *JsonResponse) HasEncodingOption(option int) bool {
	return r.encodingOptions&option != 0
}

// WithCallback sets the JSONP callback.
func (r *JsonResponse) WithCallback(callback string) *JsonResponse {
	return r.SetCallback(callback)
}

// SetCallback sets the JSONP callback and rebuilds the body. WithCallback
// calls it.
func (r *JsonResponse) SetCallback(callback string) *JsonResponse {
	r.callback = callback
	r.setJSONString(r.data)
	return r
}

// The next block redeclares the Response methods that return the receiver.
// Go has no covariant return type, so the promoted method from the embedded
// Response would hand back a *Response and end the chain.

// Header is [Response.Header], typed for the chain.
func (r *JsonResponse) Header(key string, values ...any) *JsonResponse {
	r.Response.Header(key, values...)
	return r
}

// WithHeaders is [Response.WithHeaders], typed for the chain.
func (r *JsonResponse) WithHeaders(headers stdhttp.Header) *JsonResponse {
	r.Response.WithHeaders(headers)
	return r
}

// WithoutHeader is [Response.WithoutHeader], typed for the chain.
func (r *JsonResponse) WithoutHeader(keys ...string) *JsonResponse {
	r.Response.WithoutHeader(keys...)
	return r
}

// Cookie is [Response.Cookie], typed for the chain.
func (r *JsonResponse) Cookie(cookie *stdhttp.Cookie) *JsonResponse {
	return r.WithCookie(cookie)
}

// WithCookie is [Response.WithCookie], typed for the chain.
func (r *JsonResponse) WithCookie(cookie *stdhttp.Cookie) *JsonResponse {
	r.Response.WithCookie(cookie)
	return r
}

// WithoutCookie is [Response.WithoutCookie], typed for the chain.
func (r *JsonResponse) WithoutCookie(name string, args ...string) *JsonResponse {
	r.Response.WithoutCookie(name, args...)
	return r
}

// WithException is [Response.WithException], typed for the chain.
func (r *JsonResponse) WithException(err error) *JsonResponse {
	r.Response.WithException(err)
	return r
}

// SetStatusCode is [Response.SetStatusCode], typed for the chain.
func (r *JsonResponse) SetStatusCode(code int) *JsonResponse {
	r.Response.SetStatusCode(code)
	return r
}
