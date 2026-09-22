package view

import (
	"crypto/rand"
	"fmt"
	"io"
	"math/big"
	"path/filepath"
	"strings"
	"sync"

	"github.com/arandu-io/hesape/view/engines"
)

// Factory produces View instances from the compiled view registry and
// manages shared data, composers, creators, sections, stacks, loops,
// components, fragments and translations. The compiler that turns .kyse.go
// into the Go functions this Factory runs is a separate concern; this is the
// API that wraps it.
//
// A Factory is created once per application and shared across requests.
// Its methods are goroutine-safe.
type Factory struct {
	mu sync.RWMutex

	// Shared data available to every view.
	shared map[string]any

	// View composers and creators.
	composers []viewEvent
	creators  []viewEvent
	once      viewEventOnce
	events    Dispatcher

	// Engine resolver maps extensions to engines.
	resolver *engines.Resolver

	// File extensions -> engine name mapping.
	extensions map[string]string

	// Finder locates view files on disk.
	finder *FileViewFinder

	// Sections (layouts).
	sections      map[string]string
	sectionStack  []string
	lastSection   string
	sectionBuffer string

	// Loops.
	loopStack []*Loop

	// Stacks (push/prepend).
	pushes     map[string][]string
	prepends   map[string][]string
	pushStack  []pushFrame
	pushBuffer string

	// Components.
	componentStack       []string
	componentData        map[string]any
	currentComponent     string
	currentComponentData map[string]any
	slots                map[string]*ComponentSlot
	slotStack            []string
	componentBuffer      string
	slotBuffer           string

	// Fragments.
	fragments      map[string]string
	fragmentStack  []string
	fragmentBuffer string

	// Translations.
	transReplacements map[string]string
	transKey          string

	// Rendering count.
	renderCount  int
	renderedOnce map[string]bool

	// Parent placeholder salt, and the marker memoised per section.
	parentSalt         string
	parentPlaceholders map[string]string
}

// NewFactory returns a Factory backed by the compiled view registry.
func NewFactory() *Factory {
	f := &Factory{
		shared:       map[string]any{},
		resolver:     engines.NewResolver(),
		extensions:   map[string]string{},
		sections:     map[string]string{},
		pushes:       map[string][]string{},
		prepends:     map[string][]string{},
		slots:        map[string]*ComponentSlot{},
		fragments:    map[string]string{},
		renderedOnce: map[string]bool{},
		parentSalt:   randomString(40),
	}
	return f
}

// Make returns a View ready to render.
//
//	factory.Make("posts.index", map[string]any{"posts": posts})
//
// In Arandu the view function is already compiled in (registered via
// view.Register in init()), so Make looks it up by name. If the view is not
// registered, it returns a View whose Render method will produce an error
// naming the missing view — a blank page with 200 is the failure this exists
// to prevent.
func (f *Factory) Make(name string, data map[string]any) *View {
	normalized := f.normalizeName(name)
	if data == nil {
		data = map[string]any{}
	}

	v := &View{
		factory: f,
		engine:  NewRenderer(),
		name:    normalized,
		data:    data,
	}

	f.mu.RLock()
	shared := make(map[string]any, len(f.shared))
	for k, val := range f.shared {
		shared[k] = val
	}
	f.mu.RUnlock()
	v.shared = shared

	return v
}

// First renders the first view from the list that exists.
func (f *Factory) First(views []string, data map[string]any) *View {
	for _, name := range views {
		if f.Exists(name) {
			return f.Make(name, data)
		}
	}
	// Return the first name even if it doesn't exist, so Render produces a
	// clear error.
	if len(views) > 0 {
		return f.Make(views[0], data)
	}
	return f.Make("", data)
}

// RenderEach renders a view for each item in data, using the iterator key
// to pass each item as the view's data. When data is empty, the empty view
// is rendered instead.
//
//	factory.RenderEach("posts.item", posts, "post", "posts.empty")
func (f *Factory) RenderEach(viewName string, data any, iterator string, emptyView string) (string, error) {
	var results strings.Builder

	switch items := data.(type) {
	case []any:
		if len(items) == 0 && emptyView != "" {
			v := f.Make(emptyView, nil)
			s, err := v.Render()
			if err != nil {
				return "", err
			}
			return s, nil
		}
		for _, item := range items {
			v := f.Make(viewName, map[string]any{iterator: item})
			s, err := v.Render()
			if err != nil {
				return "", err
			}
			results.WriteString(s)
		}
	case []map[string]any:
		if len(items) == 0 && emptyView != "" {
			v := f.Make(emptyView, nil)
			s, err := v.Render()
			if err != nil {
				return "", err
			}
			return s, nil
		}
		for _, item := range items {
			v := f.Make(viewName, item)
			s, err := v.Render()
			if err != nil {
				return "", err
			}
			results.WriteString(s)
		}
	default:
		return "", fmt.Errorf("view: RenderEach expects []any or []map[string]any, got %T", data)
	}

	return results.String(), nil
}

// RenderWhen renders the view if the condition is true.
func (f *Factory) RenderWhen(condition bool, name string, data map[string]any) (string, error) {
	if !condition {
		return "", nil
	}
	v := f.Make(name, data)
	return v.Render()
}

// RenderUnless renders the view if the condition is false.
func (f *Factory) RenderUnless(condition bool, name string, data map[string]any) (string, error) {
	return f.RenderWhen(!condition, name, data)
}

// Exists reports whether a view is registered.
func (f *Factory) Exists(name string) bool {
	normalized := f.normalizeName(name)
	mu.RLock()
	_, known := views[normalized]
	mu.RUnlock()
	return known
}

// Share registers a value available to every view.
//
//	factory.Share("appName", "My App")
func (f *Factory) Share(key string, value any) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.shared[key] = value
}

// Shared returns a copy of the shared data.
func (f *Factory) Shared() map[string]any {
	return f.GetShared()
}

// GetShared returns a copy of the shared data.
func (f *Factory) GetShared() map[string]any {
	f.mu.RLock()
	defer f.mu.RUnlock()
	out := make(map[string]any, len(f.shared))
	for k, v := range f.shared {
		out[k] = v
	}
	return out
}

// GetFinder returns the FileViewFinder.
func (f *Factory) GetFinder() *FileViewFinder {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.finder
}

// SetFinder replaces the finder.
func (f *Factory) SetFinder(finder *FileViewFinder) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.finder = finder
}

// GetEngineResolver returns the engine resolver.
func (f *Factory) GetEngineResolver() *engines.Resolver {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.resolver
}

// AddNamespace delegates to the finder.
func (f *Factory) AddNamespace(ns string, hints []string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.finder != nil {
		f.finder.AddNamespace(ns, hints)
	}
}

// PrependNamespace delegates to the finder.
func (f *Factory) PrependNamespace(ns string, hints []string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.finder != nil {
		f.finder.PrependNamespace(ns, hints)
	}
}

// ReplaceNamespace delegates to the finder.
func (f *Factory) ReplaceNamespace(ns string, hints []string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.finder != nil {
		f.finder.ReplaceNamespace(ns, hints)
	}
}

// AddExtension maps a file extension to an engine.
func (f *Factory) AddExtension(ext string, engine string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.extensions[ext] = engine
}

// AddLocation delegates to the finder.
func (f *Factory) AddLocation(location string) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.finder == nil {
		return fmt.Errorf("view: no finder set")
	}
	return f.finder.AddLocation(location)
}

// GetEngineFromPath resolves the engine for the given path by matching its
// extension.
func (f *Factory) GetEngineFromPath(path string) (engines.Engine, error) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.getEngineFromPathLocked(path)
}

func (f *Factory) getEngineFromPathLocked(path string) (engines.Engine, error) {
	ext := f.getExtensionLocked(path)
	if eng, err := f.resolver.Resolve(ext); err == nil {
		return eng, nil
	}
	return nil, fmt.Errorf("view: no engine for extension %q of %s", ext, path)
}

func (f *Factory) getExtensionLocked(path string) string {
	ext := filepath.Ext(path)
	for registered := range f.extensions {
		if ext == registered {
			return ext
		}
	}
	if _, ok := f.extensions[ext]; ok {
		return ext
	}
	return ext
}

// IncrementRender bumps the rendering counter.
func (f *Factory) IncrementRender() {
	f.mu.Lock()
	f.renderCount++
	f.mu.Unlock()
}

// DecrementRender drops the rendering counter.
func (f *Factory) DecrementRender() {
	f.mu.Lock()
	if f.renderCount > 0 {
		f.renderCount--
	}
	f.mu.Unlock()
}

// DoneRendering reports whether all rendering is complete.
func (f *Factory) DoneRendering() bool {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.renderCount == 0
}

// HasRenderedOnce reports whether a @once block has already rendered.
func (f *Factory) HasRenderedOnce(id string) bool {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.renderedOnce[id]
}

// MarkAsRenderedOnce marks a @once block as rendered.
func (f *Factory) MarkAsRenderedOnce(id string) {
	f.mu.Lock()
	f.renderedOnce[id] = true
	f.mu.Unlock()
}

// Flush clears caches and shared state. Use between requests in tests.
func (f *Factory) Flush() {
	f.FlushState()
	f.FlushFinderCache()
}

// FlushState resets per-request state: sections, stacks, components,
// fragments and the render count.
//
// It used to take the lock and then call the four Flush methods, each of which
// takes the same lock. A sync.RWMutex is not reentrant, so the first call after
// a render deadlocked the request that made it. Nothing called it, which is why
// it survived.
func (f *Factory) FlushState() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.flushSectionsLocked()
	f.flushStacksLocked()
	f.flushComponentsLocked()
	f.flushFragmentsLocked()
	f.renderCount = 0
	f.renderedOnce = map[string]bool{}
}

// FlushStateIfDoneRendering calls FlushState only when DoneRendering reports
// true.
func (f *Factory) FlushStateIfDoneRendering() {
	if f.DoneRendering() {
		f.FlushState()
	}
}

// FlushFinderCache clears the finder's cache, if a finder is set.
func (f *Factory) FlushFinderCache() {
	f.mu.RLock()
	finder := f.finder
	f.mu.RUnlock()
	if finder != nil {
		finder.Flush()
	}
}

// GetDispatcher returns the registered event dispatcher, or nil.
func (f *Factory) GetDispatcher() Dispatcher {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.events
}

// SetDispatcher registers the event dispatcher composers and creators are
// announced through.
func (f *Factory) SetDispatcher(events Dispatcher) {
	f.mu.Lock()
	f.events = events
	f.mu.Unlock()
}

// Dispatcher is the event-dispatch contract the view layer uses.
//
// The contract lives in the package that consumes it, which is why it is
// declared here and not imported: a Factory that has one announces
// "composing: name" and "creating: name" as it draws, and a Factory that has
// none draws the same page without them.
type Dispatcher interface {
	// Dispatch fires the named event with the view being drawn.
	Dispatch(event string, payload ...any)
}

// RenderToString is a convenience method that makes and renders a view in
// one call.
func (f *Factory) RenderToString(name string, data map[string]any) (string, error) {
	v := f.Make(name, data)
	return v.Render()
}

// normalizeName turns a path-style name into dot notation.
func (f *Factory) normalizeName(name string) string {
	return (ViewName{}).Normalize(name)
}

// File returns a View for the given absolute path.
//
//	view := factory.File("/abs/path/to/resources/views/emails/welcome.kyse.go", data)
func (f *Factory) File(path string, data map[string]any) *View {
	// For compiled views in Arandu, the path maps to a registered name.
	name := pathToName(path)
	v := f.Make(name, data)
	v.path = path
	return v
}

// pathToName converts an absolute path to a registered view name.
func pathToName(path string) string {
	// Strip directories down to the view root, then convert to dot notation.
	// This is a best-effort mapping; the view must have been registered.
	parts := strings.Split(path, "views"+string(filepath.Separator))
	if len(parts) > 1 {
		path = parts[len(parts)-1]
	}
	// Remove extension.
	path = strings.TrimSuffix(path, ".kyse.go")
	path = strings.TrimSuffix(path, ".go")
	return strings.ReplaceAll(path, string(filepath.Separator), ".")
}

// RenderFile renders a View to an io.Writer.
func RenderFile(w io.Writer, v *View) error {
	s, err := v.Render()
	if err != nil {
		return err
	}
	_, err = io.WriteString(w, s)
	return err
}

// randomString generates a random alphanumeric string of length n.
func randomString(n int) string {
	const letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
	b := make([]byte, n)
	for i := range b {
		n, _ := randInt(0, len(letters))
		b[i] = letters[n]
	}
	return string(b)
}

func randInt(min, max int) (int, error) {
	n, err := rand.Int(rand.Reader, big.NewInt(int64(max-min)))
	if err != nil {
		return 0, err
	}
	return int(n.Int64()) + min, nil
}
