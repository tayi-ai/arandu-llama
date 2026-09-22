package pagination

// URLWindow is the arithmetic that decides which page numbers a numbered pager
// shows when there are too many to show them all.
//
// It reports three pieces -- the first pages, the pages around the current
// one, and the last pages -- and the renderer puts a separator wherever two
// pieces do not meet. Below a certain number of pages there is no window at
// all: every page fits, so every page is listed.
//
// The type parameter is on the paginator, not on the window: there is no
// paginator interface to take instead (see the package doc for why).
type URLWindow[T any] struct {
	paginator *LengthAwarePaginator[T]
}

// URLWindowRanges is three page-to-URL ranges, any of which may be absent.
//
// A range that does not apply is a nil map. Each range is keyed by page number
// -- Go maps have no order, so a renderer walking one sorts the keys, which is
// what the page numbers are for.
type URLWindowRanges struct {
	// First is the opening run of pages: page one and two when there is a
	// slider, the whole range when there are few enough pages, and a wider run
	// when the current page is close to the beginning.
	First map[int]string

	// Slider is the run of pages around the current one, and is absent unless
	// there is room for a full window on both sides.
	Slider map[int]string

	// Last is the closing run of pages, and is absent when the first run
	// already reaches the end.
	Last map[int]string
}

// NewURLWindow builds the window for a length-aware paginator.
func NewURLWindow[T any](paginator *LengthAwarePaginator[T]) *URLWindow[T] {
	return &URLWindow[T]{paginator: paginator}
}

// Make is NewURLWindow followed by Get, in one call.
func Make[T any](paginator *LengthAwarePaginator[T]) URLWindowRanges {
	return NewURLWindow(paginator).Get()
}

// Get is the window of URLs to be shown.
func (w *URLWindow[T]) Get() URLWindowRanges {
	first, slider, last := w.pages()
	return URLWindowRanges{
		First:  w.paginator.urlsFor(first),
		Slider: w.paginator.urlsFor(slider),
		Last:   w.paginator.urlsFor(last),
	}
}

// pages is Get in page numbers rather than URLs, and in reading order.
//
// It exists because the elements a pager renders are a sequence and a Go map is
// not, and because Links and LinkCollection need the same three pieces Get
// returns. Both go through here, so the branching lives once.
func (w *URLWindow[T]) pages() (first, slider, last []int) {
	onEachSide := w.paginator.onEachSide

	if w.paginator.LastPage() < onEachSide*2+8 {
		// The small slider: not enough pages to leave any out.
		return pageRange(1, w.paginator.LastPage()), nil, nil
	}

	width := onEachSide + 4

	if !w.HasPages() {
		return nil, nil, nil
	}

	switch {
	// Close to the beginning: render the opening run, then the last two, since
	// there is no room for a full slider on the left.
	case w.paginator.CurrentPage() <= width:
		return pageRange(1, width+onEachSide), nil, w.finish()

	// Close to the end: the first two, then a wider closing run.
	case w.paginator.CurrentPage() > w.paginator.LastPage()-width:
		return w.start(), nil, pageRange(w.paginator.LastPage()-(width+onEachSide-1), w.paginator.LastPage())

	// Room on both sides: caps at each end and a sliding window in the middle.
	default:
		return w.start(), w.adjacent(onEachSide), w.finish()
	}
}

// GetAdjacentURLRange is the run of pages onEachSide either side of the
// current one.
func (w *URLWindow[T]) GetAdjacentURLRange(onEachSide int) map[int]string {
	return w.paginator.urlsFor(w.adjacent(onEachSide))
}

// GetStart is the first two pages, the cap at the beginning of a slider.
func (w *URLWindow[T]) GetStart() map[int]string {
	return w.paginator.urlsFor(w.start())
}

// GetFinish is the last two pages, the cap at the end of a slider.
func (w *URLWindow[T]) GetFinish() map[int]string {
	return w.paginator.urlsFor(w.finish())
}

// HasPages reports whether the paginator being presented spans more than one
// page.
//
// It is not the paginator's own HasPages: this one asks only about the number
// of pages, where the paginator's is also true for a reader sitting past the
// end of a single-page result set.
func (w *URLWindow[T]) HasPages() bool { return w.paginator.LastPage() > 1 }

func (w *URLWindow[T]) start() []int { return pageRange(1, 2) }

func (w *URLWindow[T]) finish() []int {
	return pageRange(w.paginator.LastPage()-1, w.paginator.LastPage())
}

func (w *URLWindow[T]) adjacent(onEachSide int) []int {
	return pageRange(w.paginator.CurrentPage()-onEachSide, w.paginator.CurrentPage()+onEachSide)
}

// pageRange returns the page numbers from start to end inclusive, and nothing
// when the range is empty. A start below one is read as one, which is what
// keeps a window near the beginning from asking for page zero.
func pageRange(start, end int) []int {
	if start < 1 {
		start = 1
	}
	if end < start {
		return nil
	}
	out := make([]int, 0, end-start+1)
	for page := start; page <= end; page++ {
		out = append(out, page)
	}
	return out
}
