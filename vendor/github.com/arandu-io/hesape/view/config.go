package view

// Config holds what genuinely varies between deployments for this package,
// and it is two things.
//
// It stays short on purpose: every view compiles into a Go function at build
// time, and the binary carries the functions -- there is no directory to
// search and no cache to place, so there is nothing to configure for either.
//
// Declared here, in the package that reads it.
type Config struct {
	// Reload injects the development reload script into any full HTML document
	// that leaves the application.
	//
	// It follows the application's debug setting rather than a variable of its
	// own, and this field exists so that a test can turn it off without
	// pretending to be in production. Serving the script outside development
	// would add a request per page and a connection per document to something
	// nobody asked for.
	Reload bool

	// Fragments makes the renderer answer only the named part of a view when
	// the request asks for a fragment.
	//
	// True by default: it is what makes an HTMX swap send back a piece of a page
	// rather than the whole thing, and turning it off means every interaction
	// re-renders the chrome around the part that changed.
	Fragments bool
}

// DefaultConfig is the configuration a project gets without setting anything.
//
// It is a function and not a package-level variable on purpose. A variable
// would be reachable from anywhere and assignable at init by any package in the
// build -- which is how a default becomes something nobody can find the source
// of.
func DefaultConfig() Config {
	return Config{Reload: false, Fragments: true}
}
