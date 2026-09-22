package pagination

import (
	"encoding/json"
	"net/url"
	"strconv"
	"strings"
)

// Default names and window width. They are exported because a caller that
// overrides one of them usually wants to compare against the default first.
const (
	// DefaultPageName is the query parameter LengthAwarePaginator and Paginator
	// write the page number into.
	DefaultPageName = "page"

	// DefaultCursorName is the query parameter CursorPaginator writes the
	// encoded cursor into.
	DefaultCursorName = "cursor"

	// DefaultOnEachSide is how many numbered links sit either side of the
	// current page before the window collapses into a "..." separator.
	DefaultOnEachSide = 3
)

// Separator is the label of a Link that stands for the pages a window left out.
// Such a Link has an empty URL and a Page of zero, so it is never clickable.
const Separator = "..."

// PreviousLabel and NextLabel are the labels LinkCollection gives the two steps
// either side of the numbered pages.
//
// There is no translator to ask here, so these two words are what is written; a
// rendered pager translates them in the view, where the reader's locale is
// known.
const (
	PreviousLabel = "Previous"
	NextLabel     = "Next"
)

// Options is everything a paginator needs to write the URL of another page.
//
// It is one value the three constructors take, rather than four things read out
// of the request when they are needed.
//
// The zero value is usable by the numbered paginators: it paginates the path
// "/" with no extra query, the parameter names "page" and "cursor", and three
// links either side. [CursorPaginate] needs Signer as well, and says so.
//
// Every constructor takes an Options by value and normalises its own copy, so
// filling one in after passing it changes nothing.
type Options struct {
	// Path is the URL the page links are built on, without query or fragment.
	// It may be a full URL or an absolute path. Empty means "/".
	Path string

	// Query is carried onto every generated URL, which is what keeps a filter
	// or a sort selected while the reader walks the pages. The page and cursor
	// parameters are dropped from it: the paginator writes those itself.
	Query url.Values

	// Fragment is appended after a "#", so a page link can land on the table
	// rather than the top of the document.
	Fragment string

	// PageName is the query parameter holding the page number. Empty means
	// DefaultPageName.
	PageName string

	// CursorName is the query parameter holding the encoded cursor. Empty means
	// DefaultCursorName.
	CursorName string

	// OnEachSide is how many numbered links surround the current page. Zero
	// means DefaultOnEachSide; a negative value means none, which is the only
	// way to ask for a window with nothing but the caps.
	OnEachSide int

	// Signer signs the cursor a [CursorPaginator] writes into its links, and is
	// what read it back on the way in. It is required by [CursorPaginate] and
	// ignored by the numbered paginators, whose position is a page number: a
	// rewritten page number reaches nothing a reader could not reach by typing
	// one, and a rewritten cursor names a row.
	Signer *CursorSigner
}

// normalize returns a copy with the defaults applied and the query cloned, so
// that a paginator can never be changed through the map its caller kept.
func (o Options) normalize() Options {
	if o.Path == "" {
		o.Path = "/"
	}
	if o.Path != "/" {
		o.Path = strings.TrimRight(o.Path, "/")
	}
	if o.PageName == "" {
		o.PageName = DefaultPageName
	}
	if o.CursorName == "" {
		o.CursorName = DefaultCursorName
	}
	switch {
	case o.OnEachSide == 0:
		o.OnEachSide = DefaultOnEachSide
	case o.OnEachSide < 0:
		o.OnEachSide = 0
	}

	query := make(url.Values, len(o.Query))
	for k, values := range o.Query {
		if k == o.PageName || k == o.CursorName {
			continue
		}
		query[k] = append([]string(nil), values...)
	}
	o.Query = query
	return o
}

// url builds the address of one page, with name set to value. An empty value
// writes no parameter at all, which is how the first page keeps a clean URL.
//
// The parameters are sorted by url.Values.Encode, so the same page always
// produces the same string -- a URL that reshuffles itself between renders
// defeats the browser cache and every test that compares one.
func (o Options) url(name, value string) string {
	query := make(url.Values, len(o.Query)+1)
	for k, values := range o.Query {
		query[k] = values
	}
	if value != "" {
		query.Set(name, value)
	}

	out := o.Path
	if encoded := query.Encode(); encoded != "" {
		separator := "?"
		if strings.Contains(out, "?") {
			separator = "&"
		}
		out += separator + encoded
	}
	if o.Fragment != "" {
		out += "#" + o.Fragment
	}
	return out
}

// OptionsFrom reads the paginator options out of the URL of the request being
// served.
//
// The path keeps whatever the URL had: for a *http.Request served by the
// framework that is the path alone, and for a URL parsed from an absolute
// address it is scheme, host and path. Query and fragment are stripped from the
// path and the query is carried over, so a page link keeps every filter the
// reader chose.
//
// A nil URL yields the zero Options, normalised.
func OptionsFrom(u *url.URL) Options {
	if u == nil {
		return Options{}.normalize()
	}
	base := *u
	base.RawQuery = ""
	base.ForceQuery = false
	base.Fragment = ""
	base.RawFragment = ""
	return Options{Path: base.String(), Query: u.Query()}.normalize()
}

// ResolveCurrentPage reads the page number out of the URL of the request being
// served. The URL is passed in rather than reached for.
//
// Anything that is not a whole number of at least one -- absent, empty, "0",
// "-3", "two", "1e3" -- is page one. A bad page number is a reader following a
// stale link, not an incident, and every one of those spellings means the same
// thing to them.
//
// An empty pageName means DefaultPageName.
func ResolveCurrentPage(u *url.URL, pageName string) int {
	if pageName == "" {
		pageName = DefaultPageName
	}
	if u == nil {
		return 1
	}
	page, err := strconv.Atoi(u.Query().Get(pageName))
	if err != nil || page < 1 {
		return 1
	}
	return page
}

// ResolveCurrentPath reads the address of the request being served, without its
// query string or fragment, which is the base every page link is built on.
//
// The URL is passed in rather than reached for. A nil URL yields the default,
// which is an argument because Go has no default ones.
func ResolveCurrentPath(u *url.URL, def string) string {
	if u == nil {
		return def
	}
	base := *u
	base.RawQuery = ""
	base.ForceQuery = false
	base.Fragment = ""
	base.RawFragment = ""
	return base.String()
}

// ResolveQueryString reads the query string of the request being served, which
// is what WithQueryString carries onto every page link.
//
// A nil URL yields the default.
func ResolveQueryString(u *url.URL, def url.Values) url.Values {
	if u == nil {
		return def
	}
	return u.Query()
}

// Link is one entry of a rendered pager: a numbered page, the previous or next
// step, or the separator that stands for the pages a window left out.
//
// A separator, and a previous or next step with nowhere to go, has an empty URL
// and a Page of zero. The component that renders it tests URL, and prints the
// label without a link.
type Link struct {
	// URL is where the link points. Empty on a separator, and on a previous or
	// next step that has nowhere to go.
	URL string

	// Label is what the reader sees: the page number, Separator, or the word
	// for the previous or next step.
	Label string

	// Page is the page number, and zero where there is none.
	Page int

	// Active marks the page being read, which a pager renders as the current
	// one rather than as somewhere to go.
	Active bool
}

// MarshalJSON writes the link as the payload carries it: the keys url, label,
// page and active, with url and page null rather than zero where there is no
// page to link to. A client that tests for null would follow a link to the
// empty string otherwise.
func (l Link) MarshalJSON() ([]byte, error) {
	return json.Marshal(struct {
		URL    any    `json:"url"`
		Label  string `json:"label"`
		Page   any    `json:"page"`
		Active bool   `json:"active"`
	}{
		URL:    nullable(l.URL),
		Label:  l.Label,
		Page:   nullable(l.Page),
		Active: l.Active,
	})
}
