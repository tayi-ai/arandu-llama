// Package pipeline sends a value through a list of stages, each free to
// inspect it, change it, hand it on or refuse it.
//
// [Pipeline] is the fluent one: Send, Through, Pipe, Then, ThenReturn and
// Finally. [Hub] holds pipelines under names. [Chain] over [Middleware] is the
// same onion for a stack that wraps a handler instead of receiving the value.
//
// # Two shapes, one onion
//
//   - [Pipeline] carries a value in and a value out. It is the shape the bus,
//     the queue and every filter chain use, and every stage of one pipeline
//     works on the same type.
//   - [Chain] over [Middleware] is the shape where the value is curried into
//     the handler. Middleware[http.Handler] is the standard
//     func(http.Handler) http.Handler, so a stage wraps the handler rather than
//     being handed a request, and the stack can answer with something that is
//     not what it was given -- which no Pipeline[T] can say.
//
// hesape/http names the second one -- hhttp.Middleware is an alias of
// pipeline.Middleware[http.Handler] -- and hesape/routing composes it with
// [Chain]. Neither of them has a composer of its own, and neither should get
// one: there is one Chain and one Pipeline, and the choice between them is the
// shape of the stage, not a preference.
//
// # A stage is a func, not a name
//
// A [Pipe] is a func, so whatever it needs -- a logger, a repository, a clock
// -- is captured by the function that builds it:
//
//	func auditPipe(log *slog.Logger) pipeline.Pipe[Order]
//
// Nothing here resolves a stage from a string, so a pipeline is read by
// following values, and a stage that was renamed fails the build rather than
// the request.
//
// A stage that has to run before and after -- opening a transaction and
// committing it, timing the rest of the chain -- is written as one pipe that
// calls next between the two halves. That is what the onion is for, and it is
// the only form that says which resource it used.
package pipeline
