package view

import (
	"bytes"
	"fmt"
	"html"
	"io"
	"sort"
	"strings"

	"github.com/arandu-io/hesape/validation"
)

// View wraps a render operation: a compiled Func, a name, a path, and the
// data merged from shared globals and per-render additions. A View is
// produced by Factory.Make and drawn by Render.
type View struct {
	factory *Factory
	engine  *Renderer
	name    string
	path    string
	data    map[string]any
	shared  map[string]any
}

// With adds data to the view. It accepts a key and a value, or a map.
//
//	view.With("title", "Hello")
//	view.With(map[string]any{"title": "Hello", "count": 42})
func (v *View) With(key string, value any) *View {
	if v.data == nil {
		v.data = map[string]any{}
	}
	v.data[key] = value
	return v
}

// WithName sets the view name.
func (v *View) WithName(name string) *View {
	v.name = name
	return v
}

// WithErrors shares the validation error bag with the view.
//
// The bag is reachable as the "errors" key in the data.
func (v *View) WithErrors(errs validation.Errors) *View {
	if v.data == nil {
		v.data = map[string]any{}
	}
	v.data["errors"] = errs
	return v
}

// Nest embeds a sub-view as data.
//
//	nest("sidebar", factory.Make("partials.sidebar", data))
func (v *View) Nest(key string, child *View) *View {
	if v.data == nil {
		v.data = map[string]any{}
	}
	v.data[key] = child
	return v
}

// GetData returns the per-view data merged with the shared globals.
func (v *View) GetData() map[string]any {
	merged := make(map[string]any, len(v.shared)+len(v.data))
	for k, val := range v.shared {
		merged[k] = val
	}
	for k, val := range v.data {
		merged[k] = val
	}
	return merged
}

// GetName returns the view's logical name (e.g. "posts/index").
func (v *View) GetName() string { return v.name }

// GetPath returns the absolute path of the source file.
func (v *View) GetPath() string { return v.path }

// SetPath changes the path.
func (v *View) SetPath(path string) { v.path = path }

// Render increments the render counter, calls composers, gathers data, runs
// the compiled function and then flushes the per-request state if nothing
// else is still drawing. A failed render flushes unconditionally, so a
// half-filled set of sections does not leak into the next page.
func (v *View) Render() (string, error) { return v.render(nil) }

// render is Render, taking an optional callback that runs after the contents
// exist and before the state is flushed -- which is the window Fragment and
// RenderSections read their answer in.
func (v *View) render(after func()) (string, error) {
	contents, err := v.renderContents()
	if err != nil {
		if v.factory != nil {
			v.factory.FlushState()
		}
		return "", err
	}

	if after != nil {
		after()
	}
	if v.factory != nil {
		v.factory.FlushStateIfDoneRendering()
	}
	return contents, nil
}

// renderContents gathers data, runs composers and calls the compiled
// function, returning the rendered HTML.
func (v *View) renderContents() (string, error) {
	if v.factory != nil {
		v.factory.IncrementRender()
		defer v.factory.DecrementRender()

		if err := v.factory.CallComposer(v); err != nil {
			return "", err
		}
	}

	// Pick up shared data that was added after construction.
	var shared map[string]any
	if v.factory != nil {
		shared = v.factory.GetShared()
	}
	data := v.GetData()
	if shared == nil {
		shared = map[string]any{}
	}
	// Rebuild merged data with the latest shared values.
	for k, val := range shared {
		if _, overridden := v.data[k]; !overridden {
			data[k] = val
		}
	}

	// The buffer comes from the same pool the response renderer draws into, and
	// it holds *bytes.Buffer. It used to be asserted to *strings.Builder here,
	// which compiles -- the pool hands back an any -- and panics on the first
	// render. Found by reading, not by a test, because nothing called it.
	buf := bufferPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer func() {
		buf.Reset()
		bufferPool.Put(buf)
	}()

	if v.engine == nil {
		v.engine = NewRenderer()
	}

	mu.RLock()
	fn, known := views[v.name]
	mu.RUnlock()

	if !known {
		return "", fmt.Errorf("view: %q is not registered. Registered: %s",
			v.name, strings.Join(Registered(), ", "))
	}

	if err := fn(buf, data); err != nil {
		return "", fmt.Errorf("view: rendering %q: %w", v.name, err)
	}

	return buf.String(), nil
}

// String renders the view and returns the HTML, or a panic description.
func (v *View) String() string {
	s, err := v.Render()
	if err != nil {
		return html.EscapeString(err.Error())
	}
	return s
}

// ToHTML is an alias for String.
func (v *View) ToHTML() string { return v.String() }

// Gather returns the merged data without rendering.
//
// It is the shorter spelling the rest of this package already used when
// GatherData was missing. Both return the same map.
func (v *View) Gather() map[string]any { return v.GetData() }

// GatherData returns the merged data without rendering, resolving any
// nested *View into its rendered string first.
//
// The shared globals come first and the view's own data overrides them. A
// nested *View in the data is drawn to a string before the parent runs, which
// is what [View.Nest] relies on.
func (v *View) GatherData() map[string]any {
	data := map[string]any{}
	if v.factory != nil {
		for k, val := range v.factory.GetShared() {
			data[k] = val
		}
	} else {
		for k, val := range v.shared {
			data[k] = val
		}
	}
	for k, val := range v.data {
		data[k] = val
	}

	for k, val := range data {
		if child, ok := val.(*View); ok {
			rendered, err := child.Render()
			if err != nil {
				continue
			}
			data[k] = rendered
		}
	}
	return data
}

// Name is an alias for GetName.
func (v *View) Name() string { return v.GetName() }

// GetFactory returns the factory the view was made from, or nil.
func (v *View) GetFactory() *Factory { return v.factory }

// GetEngine returns the renderer engine the view uses.
func (v *View) GetEngine() *Renderer { return v.engine }

// Fragment draws the whole view and hands back only the named region, which
// is what an HTMX swap needs: the page is one template and the response is
// one piece of it.
func (v *View) Fragment(fragment string) (string, error) {
	if v.factory == nil {
		return "", fmt.Errorf("view: %q has no factory, so it has no fragments", v.name)
	}
	var part string
	if _, err := v.render(func() { part = v.factory.GetFragment(fragment) }); err != nil {
		return "", err
	}
	return part, nil
}

// Fragments draws the whole view. With no names it returns every fragment,
// in the order the view declared them; with names it returns those, in the
// order asked for.
func (v *View) Fragments(fragments ...string) (string, error) {
	if fragments == nil {
		return v.allFragments()
	}

	var out strings.Builder
	for _, name := range fragments {
		part, err := v.Fragment(name)
		if err != nil {
			return "", err
		}
		out.WriteString(part)
	}
	return out.String(), nil
}

// FragmentIf renders fragment if condition is true, or the whole view
// otherwise.
func (v *View) FragmentIf(condition bool, fragment string) (string, error) {
	if condition {
		return v.Fragment(fragment)
	}
	return v.Render()
}

// FragmentsIf renders fragments if condition is true, or the whole view
// otherwise.
func (v *View) FragmentsIf(condition bool, fragments ...string) (string, error) {
	if condition {
		return v.Fragments(fragments...)
	}
	return v.Render()
}

// allFragments renders the view and returns every captured fragment,
// concatenated in sorted name order.
func (v *View) allFragments() (string, error) {
	if v.factory == nil {
		return "", nil
	}

	var captured map[string]string
	if _, err := v.render(func() { captured = v.factory.GetFragments() }); err != nil {
		return "", err
	}

	names := make([]string, 0, len(captured))
	for name := range captured {
		names = append(names, name)
	}
	sort.Strings(names)

	var out strings.Builder
	for _, name := range names {
		out.WriteString(captured[name])
	}
	return out.String(), nil
}

// RenderSections renders the view and returns its sections instead of the
// contents.
//
// This is split from Render because Go cannot make one method return either
// a string or a map depending on whether a callback was given: Render keeps
// the string, and this returns the sections.
func (v *View) RenderSections() (map[string]string, error) {
	if v.factory == nil {
		return map[string]string{}, nil
	}
	var sections map[string]string
	if _, err := v.render(func() { sections = v.factory.GetSections() }); err != nil {
		return nil, err
	}
	return sections, nil
}

// OffsetExists reports whether key is present in the view's data.
func (v *View) OffsetExists(key string) bool {
	_, ok := v.GetData()[key]
	return ok
}

// OffsetGet returns the value of key in the view's data.
func (v *View) OffsetGet(key string) any { return v.GetData()[key] }

// OffsetSet is an alias for With, without chaining.
func (v *View) OffsetSet(key string, value any) { v.With(key, value) }

// OffsetUnset removes key from the view's data.
func (v *View) OffsetUnset(key string) { delete(v.data, key) }

// RenderView renders a View to an io.Writer directly.
func RenderView(w io.Writer, v *View) error {
	s, err := v.Render()
	if err != nil {
		return err
	}
	_, err = io.WriteString(w, s)
	return err
}
