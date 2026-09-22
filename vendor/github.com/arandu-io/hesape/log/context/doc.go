// Package context is the log context that crosses a whole request: a Repository
// carried on the context.Context, and the handler that copies it onto every log
// line.
//
// # What this is
//
// Repository is the context that crosses a whole request and lands in every log
// line it produces. Add a value once, at the edge, and every line after it
// carries the value without any call site passing it down:
//
//	ctx = context.Into(ctx, context.New(dispatcher))
//	context.For(ctx).Add("order_id", order.ID)
//
// It has a visible half and a hidden half. The visible half reaches the log
// line, through ContextLogProcessor. The hidden half never does -- it exists so
// that something can travel with a queued job without being printed.
//
// # Where the repository lives
//
// The carrier is the context.Context: Into puts a repository in, For reads it
// back. There is no process-wide repository, because a Go server runs every
// request in the same process at the same time, and one shared repository would
// be one request reading another request's context.
//
// Every method is safe on a nil receiver, so a caller that got a repository out
// of a context that carries none does not have to check: reading gives the zero
// value and writing goes nowhere.
//
// # Errors
//
// Eight methods can fail, and they return an error rather than panicking: Push,
// Pop, PushHidden, PopHidden, StackContains, HiddenStackContains, Dehydrate and
// Hydrate. ErrUnableToPush, ErrUnableToPop and ErrNotAStack are the three
// sentinels the stack failures wrap.
//
// # Queued jobs
//
// Nothing here hooks itself into a queue. A queue integration calls the two
// halves itself: on the way out, Dehydrate and attach the result to the job
// payload; on the way in, Hydrate from it.
package context
