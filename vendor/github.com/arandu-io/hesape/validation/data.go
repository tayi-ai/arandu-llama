package validation

import (
	"context"
	"net"
	"net/url"
	"slices"
	"strconv"
	"strings"

	"github.com/arandu-io/hesape/auth"
)

// Data is a submitted request, one entry per input name, and it is what a
// Validator is built over.
//
// A value is one of: nil -- the null a browser cannot send but a JSON body can
// --, a string, a []any (what a multi-select or a repeated input sends), a Data
// (a nested object), or a File. A number or a bool put in by hand is read as its
// printed form by every rule that asks.
//
// This is the shape `array`, `list`, `distinct`, `contains`, `nullable` and the
// file rules need, and the reason the package does not stop at url.Values:
// url.Values cannot hold a null, a nested value or an upload, and six of the
// rules are about exactly those.
type Data map[string]any

// DataFrom builds the Data of a submitted HTML form.
//
// A name sent once is a string and a name sent more than once is a list: reading
// `tags[]` as a string would make `array` unable to pass on input that really is
// one. A name present with no value at all is the empty list, which `required`
// refuses for having no members.
func DataFrom(values url.Values) Data {
	d := make(Data, len(values))
	for name, sent := range values {
		switch len(sent) {
		case 1:
			d[name] = sent[0]
		default:
			list := make([]any, len(sent))
			for i, v := range sent {
				list[i] = v
			}
			d[name] = list
		}
	}
	return d
}

// Values renders the Data back as a submitted form, for a caller that hands the
// whole thing on. A value that is not text -- a nested Data, an upload -- has no
// spelling in url.Values and is left out.
func (d Data) Values() url.Values {
	out := make(url.Values, len(d))
	for name, value := range d {
		switch v := value.(type) {
		case nil:
			out[name] = []string{""}
		case string:
			out[name] = []string{v}
		case []any:
			list := make([]string, 0, len(v))
			for _, item := range v {
				if s, ok := item.(string); ok {
					list = append(list, s)
					continue
				}
				if _, nested := item.(Data); nested {
					continue
				}
				list = append(list, stringOf(item))
			}
			out[name] = list
		case Data:
			continue
		case File:
			continue
		default:
			out[name] = []string{stringOf(v)}
		}
	}
	return out
}

// Has reports whether the key exists, dot notation included.
func (d Data) Has(key string) bool {
	_, ok := lookup(d, key)
	return ok
}

// HasAny reports whether at least one of the keys exists. It is what
// required_with, missing_with and present_with ask.
func (d Data) HasAny(keys []string) bool {
	for _, key := range keys {
		if d.Has(key) {
			return true
		}
	}
	return false
}

// HasAll reports whether every one of the keys exists. It is what
// missing_with_all and present_with_all ask.
func (d Data) HasAll(keys []string) bool {
	for _, key := range keys {
		if !d.Has(key) {
			return false
		}
	}
	return len(keys) > 0
}

// Get returns the value at the key, and nil when there is none.
func (d Data) Get(key string) any {
	v, _ := lookup(d, key)
	return v
}

// Forget drops the key. Only a top-level key is dropped, which is every key an
// exclude rule names.
func (d Data) Forget(key string) { delete(d, key) }

// Clone is a shallow copy, so that a Validator cannot write through to the
// request's own map.
func (d Data) Clone() Data {
	out := make(Data, len(d))
	for k, v := range d {
		out[k] = v
	}
	return out
}

// lookup walks a dotted path. The literal key is tried first, so an input
// genuinely named "a.b" is found before the path a -> b.
func lookup(d Data, key string) (any, bool) {
	if v, ok := d[key]; ok {
		return v, true
	}
	if !strings.Contains(key, ".") {
		return nil, false
	}
	var current any = d
	for _, segment := range strings.Split(key, ".") {
		switch node := current.(type) {
		case Data:
			v, ok := node[segment]
			if !ok {
				return nil, false
			}
			current = v
		case map[string]any:
			v, ok := node[segment]
			if !ok {
				return nil, false
			}
			current = v
		case []any:
			i, err := strconv.Atoi(segment)
			if err != nil || i < 0 || i >= len(node) {
				return nil, false
			}
			current = node[i]
		default:
			return nil, false
		}
	}
	return current, true
}

// ---------------------------------------------------------------------------
// The value predicates. Every rule body reads a value through one of these, so
// that what a rule accepts is decided in one place.
// ---------------------------------------------------------------------------

// asString reads a value as text, and reports whether it was.
func asString(v any) (string, bool) {
	s, ok := v.(string)
	return s, ok
}

// stringOf renders any value a request carries as text. A float is rendered
// without a trailing zero, and a value of no other kind renders as empty.
func stringOf(v any) string {
	switch n := v.(type) {
	case nil:
		return ""
	case string:
		return n
	case bool:
		if n {
			return "1"
		}
		return ""
	case int:
		return strconv.Itoa(n)
	case int64:
		return strconv.FormatInt(n, 10)
	case float64:
		return strconv.FormatFloat(n, 'f', -1, 64)
	case float32:
		return strconv.FormatFloat(float64(n), 'f', -1, 32)
	default:
		return ""
	}
}

// asList reads a value as a list of members. A Data is an array too, and countOf
// reads it, but the rules that walk members want a list and get one only from
// []any.
func asList(v any) ([]any, bool) {
	list, ok := v.([]any)
	return list, ok
}

// isArray reports whether the value is an array of either shape, a list or a
// keyed one.
func isArray(v any) bool {
	switch v.(type) {
	case []any, Data, map[string]any:
		return true
	}
	return false
}

// countOf returns how many members a countable value has, and reports whether it
// was one.
func countOf(v any) (int, bool) {
	switch n := v.(type) {
	case []any:
		return len(n), true
	case Data:
		return len(n), true
	case map[string]any:
		return len(n), true
	}
	return 0, false
}

// numberOf reads a value as a number, and reports whether it was one.
func numberOf(v any) (float64, bool) {
	switch n := v.(type) {
	case int:
		return float64(n), true
	case int64:
		return float64(n), true
	case float64:
		return n, true
	case float32:
		return float64(n), true
	case string:
		return number(strings.TrimSpace(n))
	}
	return 0, false
}

// asFile reads a value as a File, and reports whether it was one.
func asFile(v any) (File, bool) {
	f, ok := v.(File)
	return f, ok
}

// sameValue compares two values by type and by content. `same` and `different`
// compare with it, and two lists compare member by member.
func sameValue(a, b any) bool {
	if la, ok := asList(a); ok {
		lb, ok := asList(b)
		if !ok || len(la) != len(lb) {
			return false
		}
		for i := range la {
			if !sameValue(la[i], lb[i]) {
				return false
			}
		}
		return true
	}
	if _, ok := asList(b); ok {
		return false
	}
	if a == nil || b == nil {
		return a == nil && b == nil
	}
	return a == b
}

// looseContains reports whether the value appears among the parameters, reading
// nil as "null" and a bool as its name or its digit. It is how every dependent
// rule compares the other field.
func looseContains(list []string, v any) bool {
	if v == nil {
		return slices.Contains(list, "null") || slices.Contains(list, "NULL")
	}
	if b, ok := v.(bool); ok {
		want := "false"
		if b {
			want = "true"
		}
		return slices.Contains(list, want) || (b && slices.Contains(list, "1")) ||
			(!b && slices.Contains(list, "0"))
	}
	return slices.Contains(list, stringOf(v))
}

// ---------------------------------------------------------------------------
// What the rules that leave the process need.
//
// The Grant the database rules carry is auth.Grant, the one the whole framework
// carries. There is no second Grant declared here: a second type of that name
// would be a second answer to the question of who is allowed to read what.
// ---------------------------------------------------------------------------

// PresenceVerifier is what `exists` and `unique` count rows through. Both methods
// take the context the query runs under and the Grant that authorizes it.
type PresenceVerifier interface {
	// GetCount returns how many rows of the collection hold the value in the
	// column, excluding the row whose idColumn is excludeID when one is given,
	// and matching every extra condition.
	GetCount(ctx context.Context, g auth.Grant, collection, column string, value any, excludeID any, idColumn string, extra map[string]string) (int, error)

	// GetMultiCount returns how many rows hold any of the values.
	GetMultiCount(ctx context.Context, g auth.Grant, collection, column string, values []any, extra map[string]string) (int, error)
}

// CurrentPasswordChecker is what `current_password` asks. Nothing here resolves a
// guard or a hasher: the question arrives already answerable.
type CurrentPasswordChecker interface {
	// CheckCurrentPassword reports whether the plain password is the one the
	// subject the Grant was issued to already has. It answers false for a
	// subject that is not authenticated.
	CheckCurrentPassword(ctx context.Context, g auth.Grant, guard, password string) bool
}

// Resolver is what `active_url` asks whether a host has an address record.
// *net.Resolver implements it, and net.DefaultResolver is the default.
type Resolver interface {
	// LookupIPAddr returns the A and AAAA records of the host.
	LookupIPAddr(ctx context.Context, host string) ([]net.IPAddr, error)
}

// File is the shape `file`, `mimes`, `mimetypes`, `extensions`, `dimensions` and
// the size rules ask about an upload.
//
// It is an interface rather than a struct because the upload itself belongs to
// the HTTP layer. A type there satisfies this by having the methods, with no
// import either way.
type File interface {
	// GetPath returns the non-empty location or upload name that identifies the
	// file. An empty path is a file that was never written, which `mimes` and
	// `mimetypes` refuse.
	GetPath() string
	// GetRealPath returns where the bytes are.
	GetRealPath() string
	// GetSize returns the size in bytes. The size rules divide it by 1024,
	// because `max:100` on a file means kilobytes.
	GetSize() int64
	// GetMimeType returns the media type, guessed from the content.
	GetMimeType() string
	// GetExtension returns the extension of the file's own name.
	GetExtension() string
	// GuessExtension returns the extension implied by the content, which is what
	// `mimes` compares and why a .png renamed to .jpg does not pass.
	GuessExtension() string
}

// UploadedFile is a File that also knows what the client called it and whether
// the upload finished.
type UploadedFile interface {
	File
	// GetClientOriginalExtension returns the extension of the name the browser
	// sent, which is what `extensions` compares -- the name, not the content.
	GetClientOriginalExtension() string
	// IsValid reports whether the upload finished.
	IsValid() bool
}

// Dimensioner is a File that knows its own pixel size. One that implements it is
// measured through it; one that does not has its bytes decoded instead.
type Dimensioner interface {
	// Dimensions returns the pixel size of an image, and false when the bytes
	// are not one.
	Dimensions() (width, height int, ok bool)
}
