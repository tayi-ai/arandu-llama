package pagination

import (
	"encoding/json"
	"iter"
	"net/url"
	"strconv"
)

// Paginator is a page of a result set whose size is not known.
//
// There is no total, so there is no last page and no numbered window: it can
// offer previous and next, and that is the
// whole of it. What it buys is the COUNT query it does not run, which on a
// large table is the reason to reach for it.
//
// Build one with SimplePaginate.
type Paginator[T any] struct {
	items       []T
	perPage     int
	currentPage int
	hasMore     bool
	onEachSide  int
	options     Options
}

// SimplePaginate returns the page holding items.
//
// The repository reads perPage+1 rows and hands all of them here: the extra row
// is how "is there a next page" is answered without counting, and SimplePaginate
// drops it before anybody sees it. Reading exactly perPage rows is not a bug, it
// just means the paginator reports no
// next page -- which is right whenever the result set ends there, and wrong by
// one page when it does not.
//
// perPage below one is read as one, and currentPage below one as one.
func SimplePaginate[T any](items []T, perPage, currentPage int, opts Options) *Paginator[T] {
	if perPage < 1 {
		perPage = 1
	}
	if currentPage < 1 {
		currentPage = 1
	}

	hasMore := len(items) > perPage
	if hasMore {
		items = items[:perPage]
	}

	normalized := opts.normalize()
	return &Paginator[T]{
		items:       items,
		perPage:     perPage,
		currentPage: currentPage,
		hasMore:     hasMore,
		onEachSide:  normalized.OnEachSide,
		options:     normalized,
	}
}

// Items returns the rows of this page, in the order they were read, with the
// probe row already removed.
func (p *Paginator[T]) Items() []T { return p.items }

// Count returns how many rows this page holds.
func (p *Paginator[T]) Count() int { return len(p.items) }

// IsEmpty reports whether this page holds no rows.
func (p *Paginator[T]) IsEmpty() bool { return len(p.items) == 0 }

// IsNotEmpty reports whether this page holds any rows.
func (p *Paginator[T]) IsNotEmpty() bool { return len(p.items) > 0 }

// GetIterator ranges over the rows of this page with their offsets.
func (p *Paginator[T]) GetIterator() iter.Seq2[int, T] { return seq2(p.items) }

// GetCollection returns the rows this page holds.
func (p *Paginator[T]) GetCollection() []T { return p.items }

// SetCollection replaces the rows this page holds, leaving the page number and
// the next-page answer alone.
func (p *Paginator[T]) SetCollection(items []T) *Paginator[T] {
	p.items = items
	return p
}

// GetOptions returns the options this page builds its URLs from, with the
// defaults already applied.
func (p *Paginator[T]) GetOptions() Options { return p.options }

// PerPage returns the page size the paginator was built with.
func (p *Paginator[T]) PerPage() int { return p.perPage }

// CurrentPage returns the page being read, counting from one.
func (p *Paginator[T]) CurrentPage() int { return p.currentPage }

// FirstItem returns the one-based index, in the whole result set, of the first
// row on this page, and zero when the page is empty.
func (p *Paginator[T]) FirstItem() int {
	if len(p.items) == 0 {
		return 0
	}
	return (p.currentPage-1)*p.perPage + 1
}

// LastItem returns the one-based index, in the whole result set, of the last
// row on this page, and zero when the page is empty.
func (p *Paginator[T]) LastItem() int {
	if len(p.items) == 0 {
		return 0
	}
	return p.FirstItem() + len(p.items) - 1
}

// HasMorePages reports whether a page follows this one, which is what the
// probe row answered.
func (p *Paginator[T]) HasMorePages() bool { return p.hasMore }

// HasMorePagesWhen overrides that answer, for a caller that knows something
// the probe row does not.
func (p *Paginator[T]) HasMorePagesWhen(hasMore bool) *Paginator[T] {
	p.hasMore = hasMore
	return p
}

// HasPages reports whether there is anywhere to go from here, in either
// direction.
func (p *Paginator[T]) HasPages() bool { return p.currentPage != 1 || p.hasMore }

// OnFirstPage reports whether this is page one.
func (p *Paginator[T]) OnFirstPage() bool { return p.currentPage <= 1 }

// OnLastPage reports whether no page follows this one.
func (p *Paginator[T]) OnLastPage() bool { return !p.hasMore }

// URL returns the address of the given page; a page below one is read as one.
func (p *Paginator[T]) URL(page int) string {
	if page < 1 {
		page = 1
	}
	return p.options.url(p.options.PageName, strconv.Itoa(page))
}

// GetURLRange returns the address of every page from start to end inclusive,
// keyed by page number.
//
// A simple paginator has no last page to bound the range with, so the caller
// supplies both ends.
func (p *Paginator[T]) GetURLRange(start, end int) map[int]string {
	pages := pageRange(start, end)
	if len(pages) == 0 {
		return nil
	}
	out := make(map[int]string, len(pages))
	for _, page := range pages {
		out[page] = p.URL(page)
	}
	return out
}

// PreviousPageURL returns the address of the page before this one, and the
// empty string on page one.
func (p *Paginator[T]) PreviousPageURL() string {
	if p.currentPage <= 1 {
		return ""
	}
	return p.URL(p.currentPage - 1)
}

// NextPageURL returns the address of the page after this one, and the empty
// string when no row was left over to prove there is one.
func (p *Paginator[T]) NextPageURL() string {
	if !p.hasMore {
		return ""
	}
	return p.URL(p.currentPage + 1)
}

// Path returns the base path the page links are built on.
func (p *Paginator[T]) Path() string { return p.options.Path }

// SetPath sets the base path the page links are built on.
func (p *Paginator[T]) SetPath(path string) *Paginator[T] {
	p.options.Path = path
	return p
}

// WithPath sets the base address every page link is built on.
func (p *Paginator[T]) WithPath(path string) *Paginator[T] { return p.SetPath(path) }

// GetPageName returns the query parameter the page number is written into.
func (p *Paginator[T]) GetPageName() string { return p.options.PageName }

// SetPageName sets the query parameter the page number is written into.
func (p *Paginator[T]) SetPageName(name string) *Paginator[T] {
	p.options.PageName = name
	return p
}

// OnEachSide sets how many numbered links sit either side of the current page.
//
// A simple paginator renders no numbered window, so nothing here reads it. It is
// kept so that a caller swapping a simple page for a length-aware one does not
// have to delete the call.
func (p *Paginator[T]) OnEachSide(count int) *Paginator[T] {
	if count < 0 {
		count = 0
	}
	p.onEachSide = count
	p.options.OnEachSide = count
	return p
}

// Fragment sets the fragment appended after a "#". See
// LengthAwarePaginator.Fragment for why this is the setter alone.
func (p *Paginator[T]) Fragment(fragment string) *Paginator[T] {
	p.options.Fragment = fragment
	return p
}

// Appends carries extra query string values onto every generated URL. See
// LengthAwarePaginator.Appends for the forms key takes.
func (p *Paginator[T]) Appends(key any, value ...string) *Paginator[T] {
	appendQuery(&p.options, p.options.PageName, key, value)
	return p
}

// WithQueryString carries every parameter of the current request onto the
// generated URLs; there is no static resolver here, so they are passed in.
func (p *Paginator[T]) WithQueryString(query url.Values) *Paginator[T] {
	mergeQuery(&p.options, p.options.PageName, query)
	return p
}

// ToArray is the payload a simple page serialises to -- no total, no last page,
// and a current_page_url the length-aware payload does not carry.
func (p *Paginator[T]) ToArray() map[string]any {
	return map[string]any{
		"current_page":     p.currentPage,
		"current_page_url": p.URL(p.currentPage),
		"data":             p.items,
		"first_page_url":   p.URL(1),
		"from":             nullable(p.FirstItem()),
		"next_page_url":    nullable(p.NextPageURL()),
		"path":             p.Path(),
		"per_page":         p.perPage,
		"prev_page_url":    nullable(p.PreviousPageURL()),
		"to":               nullable(p.LastItem()),
	}
}

// MarshalJSON makes the paginator itself encodable, so a handler can hand one
// to a JSON encoder.
func (p *Paginator[T]) MarshalJSON() ([]byte, error) { return json.Marshal(p.ToArray()) }

// ToJSON returns the payload as bytes; bad UTF-8 is the error.
func (p *Paginator[T]) ToJSON() ([]byte, error) { return json.Marshal(p.ToArray()) }

// ToPrettyJSON is ToJSON indented four spaces.
func (p *Paginator[T]) ToPrettyJSON() ([]byte, error) { return prettyJSON(p.ToArray()) }

// ThroughSimple is the same page with every item passed through f. See Through
// for why it is a function and why there are three of them.
func ThroughSimple[A, B any](p *Paginator[A], f func(A) B) *Paginator[B] {
	items := make([]B, len(p.items))
	for i, item := range p.items {
		items[i] = f(item)
	}
	return &Paginator[B]{
		items:       items,
		perPage:     p.perPage,
		currentPage: p.currentPage,
		hasMore:     p.hasMore,
		onEachSide:  p.onEachSide,
		options:     p.options,
	}
}
