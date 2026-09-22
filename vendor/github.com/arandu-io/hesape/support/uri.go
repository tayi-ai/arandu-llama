package support

import (
	"errors"
	nethttp "net/http"
	"net/url"
	"sort"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/arandu-io/hesape/collections/arr"
)

// Uri is a parsed URI whose every writer hands back a new instance and leaves
// the old one alone. It wraps a *url.URL, so this package carries no
// dependency of its own.
type Uri struct {
	uri *url.URL
}

// UrlGenerator builds absolute URLs for named routes and actions. The routing
// package fills it in; only what [Uri] calls is declared here, so this package
// carries no dependency on routing.
type UrlGenerator interface {
	// To returns an absolute URL for the given path.
	To(path string) string
	// Route returns the URL of a named route, with its parameters filled in.
	Route(name string, parameters map[string]any, absolute bool) (string, error)
	// SignedRoute returns the URL of a named route carrying a signature, and
	// an expiry when one is given.
	SignedRoute(name string, parameters map[string]any, expiration *time.Time, absolute bool) (string, error)
	// Action returns the URL of a controller action.
	Action(action string, parameters map[string]any, absolute bool) (string, error)
}

var (
	urlGeneratorMu       sync.RWMutex
	urlGeneratorResolver func() UrlGenerator
)

// ErrNoUrlGenerator is returned by [To], [Route], [SignedRoute] and [Action]
// when no resolver was set, or when the resolver returned nothing.
var ErrNoUrlGenerator = errors.New("support: no URL generator resolver has been set")

// SetUrlGeneratorResolver sets the function that hands back the
// [UrlGenerator]. It is process-wide.
func SetUrlGeneratorResolver(resolver func() UrlGenerator) {
	urlGeneratorMu.Lock()
	defer urlGeneratorMu.Unlock()
	urlGeneratorResolver = resolver
}

func generator() (UrlGenerator, error) {
	urlGeneratorMu.RLock()
	resolver := urlGeneratorResolver
	urlGeneratorMu.RUnlock()
	if resolver == nil {
		return nil, ErrNoUrlGenerator
	}
	g := resolver()
	if g == nil {
		return nil, ErrNoUrlGenerator
	}
	return g, nil
}

// NewUri parses a URI, returning the error when it cannot be read.
func NewUri(uri string) (*Uri, error) {
	parsed, err := url.Parse(uri)
	if err != nil {
		return nil, err
	}
	return &Uri{uri: parsed}, nil
}

// Of parses a URI, the same as [NewUri].
func Of(uri string) (*Uri, error) { return NewUri(uri) }

// To returns an absolute URI for the path, built by the [UrlGenerator].
func To(path string) (*Uri, error) {
	g, err := generator()
	if err != nil {
		return nil, err
	}
	return NewUri(g.To(path))
}

// Route returns the URI of a named route, with its parameters filled in.
func Route(name string, parameters map[string]any, absolute bool) (*Uri, error) {
	g, err := generator()
	if err != nil {
		return nil, err
	}
	generated, err := g.Route(name, parameters, absolute)
	if err != nil {
		return nil, err
	}
	return NewUri(generated)
}

// SignedRoute returns the URI of a named route carrying a signature, and an
// expiry when one is given.
func SignedRoute(name string, parameters map[string]any, expiration *time.Time, absolute bool) (*Uri, error) {
	g, err := generator()
	if err != nil {
		return nil, err
	}
	generated, err := g.SignedRoute(name, parameters, expiration, absolute)
	if err != nil {
		return nil, err
	}
	return NewUri(generated)
}

// TemporarySignedRoute is [SignedRoute] with the expiry required and moved to
// the front.
func TemporarySignedRoute(name string, expiration time.Time, parameters map[string]any, absolute bool) (*Uri, error) {
	return SignedRoute(name, parameters, &expiration, absolute)
}

// Action returns the URI of a controller action.
func Action(action string, parameters map[string]any, absolute bool) (*Uri, error) {
	g, err := generator()
	if err != nil {
		return nil, err
	}
	generated, err := g.Action(action, parameters, absolute)
	if err != nil {
		return nil, err
	}
	return NewUri(generated)
}

// Scheme returns the URI's scheme.
func (u *Uri) Scheme() string { return u.uri.Scheme }

// User returns the user name, or the whole user info when withPassword is
// true. A URI carrying neither is the empty string.
func (u *Uri) User(withPassword bool) string {
	if u.uri.User == nil {
		return ""
	}
	if withPassword {
		return u.uri.User.String()
	}
	return u.uri.User.Username()
}

// Password returns the password, or the empty string when the URI carries
// none.
func (u *Uri) Password() string {
	if u.uri.User == nil {
		return ""
	}
	password, _ := u.uri.User.Password()
	return password
}

// Host returns the host, without the port.
func (u *Uri) Host() string { return u.uri.Hostname() }

// Port returns the port, or zero when the URI carries none.
func (u *Uri) Port() int {
	port, err := strconv.Atoi(u.uri.Port())
	if err != nil {
		return 0
	}
	return port
}

// Path returns the path with its slashes trimmed off both ends. An empty or
// missing path is a single "/".
func (u *Uri) Path() string {
	path := strings.Trim(u.uri.Path, "/")
	if path == "" {
		return "/"
	}
	return path
}

// PathSegments returns the path split on its slashes. An empty path is an
// empty list.
func (u *Uri) PathSegments() []string {
	path := u.Path()
	if path == "/" {
		return []string{}
	}
	return strings.Split(path, "/")
}

// Query returns the query string as a [UriQueryString].
func (u *Uri) Query() *UriQueryString { return NewUriQueryString(u) }

// Fragment returns the fragment, without its leading hash.
func (u *Uri) Fragment() string { return u.uri.Fragment }

// with copies the URI, applies the change to the copy and returns it, so the
// receiver is never written to. The user info is copied too, because the copy
// would otherwise share the pointer.
func (u *Uri) with(mutate func(*url.URL)) *Uri {
	copied := *u.uri
	if u.uri.User != nil {
		user := *u.uri.User
		copied.User = &user
	}
	mutate(&copied)
	return &Uri{uri: &copied}
}

// WithScheme returns a copy carrying the given scheme.
func (u *Uri) WithScheme(scheme string) *Uri {
	return u.with(func(n *url.URL) { n.Scheme = scheme })
}

// WithUser returns a copy carrying the given user, and the password when one
// is given. An empty user drops the user info entirely.
func (u *Uri) WithUser(user string, password ...string) *Uri {
	return u.with(func(n *url.URL) {
		switch {
		case user == "":
			n.User = nil
		case len(password) > 0:
			n.User = url.UserPassword(user, password[0])
		default:
			n.User = url.User(user)
		}
	})
}

// WithHost returns a copy carrying the given host, keeping whatever port the
// URI already had.
func (u *Uri) WithHost(host string) *Uri {
	return u.with(func(n *url.URL) {
		if port := u.uri.Port(); port != "" {
			n.Host = host + ":" + port
			return
		}
		n.Host = host
	})
}

// WithPort returns a copy carrying the given port. A zero port removes it.
func (u *Uri) WithPort(port int) *Uri {
	return u.with(func(n *url.URL) {
		if port == 0 {
			n.Host = u.uri.Hostname()
			return
		}
		n.Host = u.uri.Hostname() + ":" + strconv.Itoa(port)
	})
}

// WithPath returns a copy carrying the given path, which is given a leading
// slash when it does not already have one.
func (u *Uri) WithPath(path string) *Uri {
	return u.with(func(n *url.URL) {
		if !strings.HasPrefix(path, "/") {
			path = "/" + path
		}
		n.Path = path
	})
}

// WithQuery returns a copy with the pairs written into the query string by
// dotted key. The variadic argument is whether to merge into the query already
// there and defaults to true; false replaces it instead.
func (u *Uri) WithQuery(query map[string]any, merge ...bool) *Uri {
	newQuery := map[string]any{}
	if firstOr(merge, true) {
		newQuery = u.Query().ToArray()
	}
	for _, key := range sortedMapKeys(query) {
		arr.Set(newQuery, key, query[key])
	}
	return u.with(func(n *url.URL) { n.RawQuery = arr.Query(newQuery) })
}

// WithQueryIfMissing returns a copy carrying only the keys the query string
// does not already hold.
func (u *Uri) WithQueryIfMissing(query map[string]any) *Uri {
	current := u.Query()
	pending := map[string]any{}
	for key, v := range query {
		if current.Missing(key) {
			pending[key] = v
		}
	}
	return u.WithQuery(pending)
}

// PushOntoQuery appends to the list a query parameter holds. A list already
// holding the value is left alone; a single value becomes a list of the two,
// with no such check.
func (u *Uri) PushOntoQuery(key string, v any) *Uri {
	current, _ := arr.Get(u.Query().ToArray(), key)
	values := arr.Wrap(v)

	switch existing := current.(type) {
	case nil:
		return u.WithQuery(map[string]any{key: values})
	case []any:
		merged := append([]any{}, existing...)
		for _, candidate := range values {
			found := false
			for _, held := range merged {
				if held == candidate {
					found = true
					break
				}
			}
			if !found {
				merged = append(merged, candidate)
			}
		}
		return u.WithQuery(map[string]any{key: merged})
	default:
		return u.WithQuery(map[string]any{key: append([]any{existing}, values...)})
	}
}

// WithoutQuery returns a copy whose query string has dropped the given keys.
func (u *Uri) WithoutQuery(keys ...string) *Uri {
	return u.ReplaceQuery(arr.Except(u.Query().ToArray(), keys...))
}

// ReplaceQuery returns a copy whose query string is thrown away and written
// again from the given pairs.
func (u *Uri) ReplaceQuery(query map[string]any) *Uri {
	return u.WithQuery(query, false)
}

// WithFragment returns a copy carrying the given fragment.
func (u *Uri) WithFragment(fragment string) *Uri {
	return u.with(func(n *url.URL) { n.Fragment = fragment })
}

// ToHtml returns the URI as markup, ready to write into a template.
func (u *Uri) ToHtml() string { return u.Value() }

// Redirect returns a handler that redirects to this URI with the given status.
//
// It hands back a net/http.Handler rather than a response type of this
// framework's, because such a type lives in the http package and that package
// imports this one.
//
// headers is applied before the redirect is written, because WriteHeader
// freezes the header map.
func (u *Uri) Redirect(status int, headers map[string]string) nethttp.Handler {
	value := u.Value()
	return nethttp.HandlerFunc(func(w nethttp.ResponseWriter, r *nethttp.Request) {
		for name, v := range headers {
			w.Header().Set(name, v)
		}
		nethttp.Redirect(w, r, value, status)
	})
}

// ToResponse returns a handler that redirects to this URI with status 302.
//
// The request is taken and ignored: the parameter is there so the method fits
// the shape a caller turning a value into a response expects.
func (u *Uri) ToResponse(*nethttp.Request) nethttp.Handler {
	return u.Redirect(nethttp.StatusFound, nil)
}

// Decode returns the URI with its query string percent-decoded, which is what
// a person reads in a browser bar.
func (u *Uri) Decode() string {
	query := u.Query()
	if len(query.ToArray()) == 0 {
		return u.Value()
	}
	full := u.Value()
	index := strings.Index(full, "?")
	if index < 0 {
		return full
	}
	return full[:index+1] + query.Decode()
}

// Value returns the URI written out as a string.
func (u *Uri) Value() string { return u.uri.String() }

// String returns the URI written out, so Uri satisfies fmt.Stringer.
func (u *Uri) String() string { return u.Value() }

// IsEmpty reports whether the URI written out is blank.
func (u *Uri) IsEmpty() bool { return strings.TrimSpace(u.Value()) == "" }

// GetUri returns the *url.URL underneath. Writing to it writes through to this
// Uri, which every other method avoids.
func (u *Uri) GetUri() *url.URL { return u.uri }

// UriQueryString is the query string of a [Uri], read as nested data. The
// typed readers of the embedded data source read that data.
type UriQueryString struct {
	dataSource
	uri *Uri
}

// NewUriQueryString builds the query string reader for a [Uri].
func NewUriQueryString(uri *Uri) *UriQueryString {
	q := &UriQueryString{uri: uri}
	q.dataSource = dataSource{
		allFn:  q.All,
		dataFn: func(key string, def any) any { return q.Get(key, def) },
	}
	return q
}

// All returns the whole query string when given no key, and the subset under
// the given dotted keys otherwise.
func (q *UriQueryString) All(keys ...string) map[string]any {
	query := q.ToArray()
	if len(keys) == 0 {
		return query
	}
	return subsetByKeys(query, keys)
}

// Get returns one parameter by dotted key, falling back to the optional
// default. An empty key returns the whole query string, and the default is not
// consulted.
func (q *UriQueryString) Get(key string, def ...any) any {
	query := q.ToArray()
	if key == "" {
		return query
	}
	if held, ok := arr.Get(query, key); ok {
		return held
	}
	return firstOr(def, nil)
}

// Decode returns the whole query string percent-decoded, or as it stands when
// it cannot be decoded.
func (q *UriQueryString) Decode() string {
	decoded, err := url.PathUnescape(q.Value())
	if err != nil {
		return q.Value()
	}
	return decoded
}

// Value returns the raw query string, or the empty string when there is no
// URI behind it.
func (q *UriQueryString) Value() string {
	if q.uri == nil || q.uri.uri == nil {
		return ""
	}
	return q.uri.uri.RawQuery
}

// ToArray returns the query string as nested data.
//
// A key is decoded before its brackets are read, so a%5Bb%5D=c and a[b]=c give
// the same nesting.
func (q *UriQueryString) ToArray() map[string]any {
	return parseQueryString(q.Value())
}

// parseQueryString reads key[sub]=value and key[]=value back into nested maps
// and lists. A pair carrying no key is skipped.
func parseQueryString(raw string) map[string]any {
	results := map[string]any{}
	if raw == "" {
		return results
	}
	for _, pair := range strings.Split(raw, "&") {
		if pair == "" {
			continue
		}
		rawKey, rawValue, _ := strings.Cut(pair, "=")
		key := rawURLDecode(rawKey)
		if key == "" {
			continue
		}
		setQueryValue(results, querySegments(key), rawURLDecode(rawValue))
	}
	return results
}

// querySegments splits a[b][] into "a", "b", "".
func querySegments(key string) []string {
	open := strings.Index(key, "[")
	if open < 0 {
		return []string{key}
	}
	segments := []string{key[:open]}
	rest := key[open:]
	for strings.HasPrefix(rest, "[") {
		end := strings.Index(rest, "]")
		if end < 0 {
			segments = append(segments, rest[1:])
			return segments
		}
		segments = append(segments, rest[1:end])
		rest = rest[end+1:]
	}
	return segments
}

func setQueryValue(target map[string]any, segments []string, v string) {
	segment := segments[0]
	if len(segments) == 1 {
		if segment == "" {
			return
		}
		target[segment] = v
		return
	}
	if segments[1] == "" && len(segments) == 2 {
		list, _ := target[segment].([]any)
		target[segment] = append(list, v)
		return
	}
	child, ok := target[segment].(map[string]any)
	if !ok {
		child = map[string]any{}
		target[segment] = child
	}
	setQueryValue(child, segments[1:], v)
}

// rawURLDecode turns percent-triplets back into bytes, leaving a plus as a
// plus. A string that cannot be decoded is returned as it stands.
func rawURLDecode(s string) string {
	decoded, err := url.PathUnescape(s)
	if err != nil {
		return s
	}
	return decoded
}

func sortedMapKeys(m map[string]any) []string {
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	return keys
}
