package pagination

import "sync"

// The nine pager views, by the name a paginator asks for one by. They are names,
// not templates: nothing in this package renders (see the package doc), and the
// view layer is what turns a name into markup.
//
// The default is Tailwind.
const (
	TailwindView             = "pagination::tailwind"
	SimpleTailwindView       = "pagination::simple-tailwind"
	BootstrapFiveView        = "pagination::bootstrap-5"
	SimpleBootstrapFiveView  = "pagination::simple-bootstrap-5"
	BootstrapFourView        = "pagination::bootstrap-4"
	SimpleBootstrapFourView  = "pagination::simple-bootstrap-4"
	BootstrapThreeView       = "pagination::bootstrap-3"
	SimpleBootstrapThreeView = "pagination::simple-bootstrap-3"
	SemanticUIView           = "pagination::semantic-ui"
)

var (
	viewMu            sync.RWMutex
	defaultView       = TailwindView
	defaultSimpleView = SimpleTailwindView
)

// DefaultView reads and writes the view a numbered pager is rendered with.
// Called with a view it sets the default and returns it; called with none it
// reads.
//
// The argument is what tells the two directions apart, because both would
// otherwise want the same name.
//
// Set it once where the application is wired: it is global, and a request that
// changes it changes it for every other request in flight.
func DefaultView(view ...string) string {
	viewMu.Lock()
	defer viewMu.Unlock()
	if len(view) > 0 && view[0] != "" {
		defaultView = view[0]
	}
	return defaultView
}

// DefaultSimpleView reads and writes the view for the pager that has only
// previous and next, the same way [DefaultView] does.
func DefaultSimpleView(view ...string) string {
	viewMu.Lock()
	defer viewMu.Unlock()
	if len(view) > 0 && view[0] != "" {
		defaultSimpleView = view[0]
	}
	return defaultSimpleView
}

// UseTailwind is the default.
func UseTailwind() {
	DefaultView(TailwindView)
	DefaultSimpleView(SimpleTailwindView)
}

// UseBootstrap is [UseBootstrapFour] under the name it had before there were
// four and five.
func UseBootstrap() { UseBootstrapFour() }

// UseBootstrapThree selects the Bootstrap 3 pager views.
func UseBootstrapThree() {
	DefaultView(BootstrapThreeView)
	DefaultSimpleView(SimpleBootstrapThreeView)
}

// UseBootstrapFour selects the Bootstrap 4 pager views.
func UseBootstrapFour() {
	DefaultView(BootstrapFourView)
	DefaultSimpleView(SimpleBootstrapFourView)
}

// UseBootstrapFive selects the Bootstrap 5 pager views.
func UseBootstrapFive() {
	DefaultView(BootstrapFiveView)
	DefaultSimpleView(SimpleBootstrapFiveView)
}
