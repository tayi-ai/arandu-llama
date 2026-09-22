package view

import (
	"fmt"
	"strings"
	"sync"
)

// View composers and creators are callbacks that run before a view renders.
// A composer runs every time; a creator runs only the first time.

// ViewCallback is the function signature for composers and creators.
type ViewCallback func(v *View) error

type viewEvent struct {
	views    []string
	callback ViewCallback
	prefix   string // "composing: " or "creating: "
}

// Composer registers a callback that runs every time any of the named views
// renders.
//
//	factory.Composer("posts.*", func(v *View) error { ... })
//	factory.Composer("*", func(v *View) error { ... }) // all views
func (f *Factory) Composer(views string, callback ViewCallback) {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.composers = append(f.composers, viewEvent{
		views:    []string{views},
		callback: callback,
		prefix:   "composing: ",
	})
}

// Composers registers multiple composer specs at once, from a map.
func (f *Factory) Composers(composers map[string]ViewCallback) {
	for v, cb := range composers {
		f.Composer(v, cb)
	}
}

// Creator registers a callback that runs once, when any of the named views
// is rendered for the first time in a request.
func (f *Factory) Creator(views string, callback ViewCallback) {
	f.mu.Lock()
	defer f.mu.Unlock()

	f.creators = append(f.creators, viewEvent{
		views:    []string{views},
		callback: callback,
		prefix:   "creating: ",
	})
}

// CallComposer dispatches a composing event for the view, then runs every
// composer registered for it.
func (f *Factory) CallComposer(v *View) error {
	f.mu.RLock()
	composers := make([]viewEvent, len(f.composers))
	copy(composers, f.composers)
	events := f.events
	f.mu.RUnlock()

	name := v.GetName()
	if events != nil {
		events.Dispatch("composing: "+name, v)
	}
	for _, evt := range composers {
		if matchViewPattern(name, evt.views) {
			if err := evt.callback(v); err != nil {
				return fmt.Errorf("view: composer for %q: %w", name, err)
			}
		}
	}
	return nil
}

// CallCreator dispatches a creating event for the view, then runs every
// creator registered for it.
func (f *Factory) CallCreator(v *View) error {
	f.mu.RLock()
	creators := make([]viewEvent, len(f.creators))
	copy(creators, f.creators)
	events := f.events
	f.mu.RUnlock()

	name := v.GetName()
	if events != nil {
		events.Dispatch("creating: "+name, v)
	}
	for _, evt := range creators {
		if matchViewPattern(name, evt.views) {
			if err := evt.callback(v); err != nil {
				return fmt.Errorf("view: creator for %q: %w", name, err)
			}
		}
	}
	return nil
}

// matchViewPattern checks whether a view name matches a composer/creator
// pattern. Supports wildcards (e.g., "posts.*" matches "posts.index" and
// "posts.show"), and "*" matches all views.
func matchViewPattern(name string, patterns []string) bool {
	for _, pattern := range patterns {
		if pattern == "*" {
			return true
		}
		if strings.HasSuffix(pattern, ".*") {
			prefix := strings.TrimSuffix(pattern, ".*")
			if strings.HasPrefix(name, prefix) {
				return true
			}
		} else if pattern == name {
			return true
		}
	}
	return false
}

// viewEventOnce tracks which creators have already run.
type viewEventOnce struct {
	mu   sync.Mutex
	done map[string]bool
}

func (o *viewEventOnce) do(name string, fn func() error) error {
	o.mu.Lock()
	if o.done == nil {
		o.done = map[string]bool{}
	}
	if o.done[name] {
		o.mu.Unlock()
		return nil
	}
	o.done[name] = true
	o.mu.Unlock()
	return fn()
}
