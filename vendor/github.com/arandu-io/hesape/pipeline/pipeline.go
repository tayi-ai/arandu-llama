package pipeline

import (
	"errors"
	"fmt"
)

// Destination is the closure a pipeline ends with, and the closure every pipe
// is handed as the rest of the pipeline. Both take the passable, so both are
// this one type.
type Destination[T any] func(passable T) (T, error)

// Pipe is one stage of a [Pipeline]: it is given the passable and the rest of
// the pipeline, and answers with the result.
//
// A pipe that returns without calling next stops the pipeline, which is what
// an authorization check or a rate limiter does.
//
// A pipe that is nil fails the call when the pipeline reaches it.
type Pipe[T any] func(passable T, next Destination[T]) (T, error)

// Pipeline sends an object through a list of pipes, each free to inspect it,
// change it, hand it on or refuse it, and a destination closes the onion.
//
// The passable and the result are the same type parameter: a pipe returns what
// the next one takes, so the onion is closed over one type. A stack whose
// answer is not the value it was given is [Chain] over [Middleware], which is
// the form the HTTP stack uses; the package comment says which shape belongs
// where.
//
// The zero value is a pipeline with no pipes over the zero passable, and it is
// usable. A Pipeline is not safe for concurrent use: it is a builder, built
// and run in one place.
type Pipeline[T any] struct {
	passable T
	pipes    []Pipe[T]
	finally  func(passable T)
}

// New returns a pipeline with no pipes over the zero passable, ready for
// [Pipeline.Send] and [Pipeline.Through].
func New[T any]() *Pipeline[T] {
	return &Pipeline[T]{}
}

// Send sets the object being sent through the pipeline.
func (p *Pipeline[T]) Send(passable T) *Pipeline[T] {
	p.passable = passable
	return p
}

// Through sets the pipes, replacing whatever was there.
//
// Calling Through twice keeps only the second list, and calling it with no
// pipes empties the list. [Pipeline.Pipe] is the one that appends. The list is
// copied, so a caller reusing its own slice cannot change a pipeline it already
// built.
func (p *Pipeline[T]) Through(pipes ...Pipe[T]) *Pipeline[T] {
	p.pipes = append([]Pipe[T](nil), pipes...)
	return p
}

// Pipe pushes additional pipes onto the pipeline. It is the one that appends
// where [Pipeline.Through] replaces.
func (p *Pipeline[T]) Pipe(pipes ...Pipe[T]) *Pipeline[T] {
	p.pipes = append(p.pipes, pipes...)
	return p
}

// Finally sets a callback to run after the pipeline ends, whatever the
// outcome.
//
// It is deferred, so it runs on the way out of a failure and of a panic as
// well. Calling Finally twice keeps the second callback.
func (p *Pipeline[T]) Finally(callback func(passable T)) *Pipeline[T] {
	p.finally = callback
	return p
}

// Then runs the pipeline with a final destination.
//
// The first pipe given to [Pipeline.Through] is the outermost. The error
// returned is the first one a pipe or the destination reports.
//
// A pipe that is nil fails the call, and only if the pipeline reaches it: a
// pipe that stops the pipeline before it shields it. A destination that is nil
// fails before anything runs, including the callback set with
// [Pipeline.Finally].
func (p *Pipeline[T]) Then(destination Destination[T]) (T, error) {
	if destination == nil {
		var zero T
		return zero, errors.New("pipeline: Then needs a destination, and nil is not one")
	}

	if p.finally != nil {
		defer p.finally(p.passable)
	}

	next := destination
	for i := len(p.pipes) - 1; i >= 0; i-- {
		pipe, stack, at := p.pipes[i], next, i
		next = func(passable T) (T, error) {
			if pipe == nil {
				var zero T
				return zero, fmt.Errorf("pipeline: pipe %d is nil, and a pipeline cannot be sent through nothing", at)
			}
			return pipe(passable, stack)
		}
	}
	return next(p.passable)
}

// ThenReturn runs the pipeline with a destination that returns the passable
// unchanged, so the result is whatever the last pipe handed on.
func (p *Pipeline[T]) ThenReturn() (T, error) {
	return p.Then(func(passable T) (T, error) { return passable, nil })
}

// When applies callback to the pipeline when condition is true, and otherwise
// when it is false. A nil callback for the branch taken leaves the pipeline
// unchanged, so a chain never breaks on one.
//
//	pipeline.New[Order]().
//		Send(order).
//		Through(validate).
//		When(cfg.Audit, func(p *pipeline.Pipeline[Order]) *pipeline.Pipeline[Order] {
//			return p.Pipe(audit)
//		}, nil).
//		ThenReturn()
func (p *Pipeline[T]) When(condition bool, callback, otherwise func(pipeline *Pipeline[T]) *Pipeline[T]) *Pipeline[T] {
	if condition {
		if callback == nil {
			return p
		}
		return callback(p)
	}
	if otherwise == nil {
		return p
	}
	return otherwise(p)
}

// Unless is [Pipeline.When] with the condition negated.
func (p *Pipeline[T]) Unless(condition bool, callback, otherwise func(pipeline *Pipeline[T]) *Pipeline[T]) *Pipeline[T] {
	return p.When(!condition, callback, otherwise)
}
