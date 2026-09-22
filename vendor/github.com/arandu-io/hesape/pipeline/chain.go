package pipeline

import "context"

// Middleware wraps a handler and returns a handler of the same type.
//
// It is a type and not a method.
//
// H is whatever the caller already handles with: http.Handler for a request, a
// job handler for queued work. Nothing is invented for a particular H, which is
// what keeps middleware written elsewhere in the Go ecosystem compatible.
type Middleware[H any] func(H) H

// Chain composes middlewares around h. The first in the list is the outermost,
// so the order of the list is the order of execution. With no middleware it
// returns h.
//
// [Pipeline] is the other shape, and the package comment says which is which.
func Chain[H any](h H, mws ...Middleware[H]) H {
	for i := len(mws) - 1; i >= 0; i-- {
		h = mws[i](h)
	}
	return h
}

// Identity returns h unchanged.
//
// It is the Middleware[H] to hand back when there is nothing to wrap. A feature
// that is switched off still has to produce a value where one is being
// collected into a slice, and the alternative -- a nil middleware that Chain
// would have to test for on every element -- moves the check to the wrong side.
func Identity[H any](h H) H { return h }

// Handler ends a pipeline whose passable is a value rather than a request.
//
// It is a type and not a method.
//
// The context is first because it always is, and the error is returned rather
// than handled here: what a failure means belongs to the caller that built the
// pipeline, not to the composition.
type Handler[T any] func(ctx context.Context, v T) error
