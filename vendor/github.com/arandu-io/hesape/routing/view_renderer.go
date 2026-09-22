package routing

import (
	"io"
	"net/http"
)

// ViewRenderer renders a named view with data. It is the minimal contract
// Router.View needs, and it is here rather than importing hesape/view because
// that import would make routing depend on the view layer -- and the view
// layer depends on routing for URL generation, which is a cycle. The concrete
// is hesape/view's Factory, wired by the kernel through SetViewRenderer.
//
// The contract is what a redirect-to-view route touches: make a view by name,
// hand it data, and let it write the response. How the view renders is the
// view layer's concern.
type ViewRenderer interface {
	// Render writes the rendered view to w. The status is the one a View
	// route answers with, 200 by default, and headers are the caller's to
	// add (the CSRF header, for instance).
	Render(w io.Writer, name string, data any, status int, headers http.Header) error
}
