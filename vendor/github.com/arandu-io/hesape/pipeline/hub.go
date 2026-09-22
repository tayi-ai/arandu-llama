package pipeline

import (
	"fmt"
	"sync"
)

// Hub is a set of pipelines under names, so a caller sends an object through
// one by name instead of holding the list of pipes it takes.
//
// The callback is given a fresh [Pipeline] and the object, and it is the one
// that decides the pipes.
//
// The zero value is an empty Hub and it is usable.
type Hub[T any] struct {
	mu        sync.RWMutex
	pipelines map[string]func(p *Pipeline[T], passable T) (T, error)
}

// NewHub returns an empty hub, with no pipeline defined.
func NewHub[T any]() *Hub[T] {
	return &Hub[T]{}
}

// Pipeline defines a named pipeline.
//
// Defining a name twice keeps the second callback.
func (h *Hub[T]) Pipeline(name string, callback func(p *Pipeline[T], passable T) (T, error)) {
	h.mu.Lock()
	defer h.mu.Unlock()

	if h.pipelines == nil {
		h.pipelines = make(map[string]func(*Pipeline[T], T) (T, error), 1)
	}
	h.pipelines[name] = callback
}

// Defaults defines the pipeline named "default", which is the one [Hub.Pipe]
// reaches for when it is given no name.
func (h *Hub[T]) Defaults(callback func(p *Pipeline[T], passable T) (T, error)) {
	h.Pipeline("default", callback)
}

// Pipe sends an object through one of the available pipelines, using the name
// given or "default" when there is none.
//
// At most one name may be given. Each call builds a fresh [Pipeline], so
// nothing a pipeline was sent leaks into the next one.
//
// A name that was never defined is an error rather than a silent no-op, and a
// name defined with a nil callback is not defined.
func (h *Hub[T]) Pipe(object T, pipeline ...string) (T, error) {
	name := "default"
	if len(pipeline) > 0 && pipeline[0] != "" {
		name = pipeline[0]
	}

	h.mu.RLock()
	callback, ok := h.pipelines[name]
	h.mu.RUnlock()

	if !ok || callback == nil {
		var zero T
		return zero, fmt.Errorf("pipeline: pipeline [%s] is not defined", name)
	}
	return callback(New[T](), object)
}
