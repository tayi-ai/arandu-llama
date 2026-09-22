// Package log is the logger, the log channels and the development-time
// Collector, and it is the reason this framework exists.
//
// slog covers structured logging and OpenTelemetry covers production tracing.
// What is missing between the two is the development layer: the moment a request
// broke and you want the stack, the queries with their timing, what you dumped
// and what the framework thinks is wrong, on one screen.
//
// # Logging
//
// New builds the root logger -- readable text in development, JSON everywhere
// else, so it reaches the aggregator without fragile parsing. Middleware puts it
// at the top of the HTTP pipeline, For reads it back inside a handler or a
// service, and With attaches fields that everything downstream inherits. There
// is no exported global logger, on purpose: a log line without request_id and
// tenant is noise, and the only way to guarantee both is to force the context
// through.
//
// ParseLevel reads the eight level names and New renders all eight back under
// those names. Capture is the same logger writing into memory, which is what a
// test asserts against.
//
// # Logger and LogManager
//
// Logger is the eight levels under their own names -- Emergency, Alert,
// Critical, Error, Warning, Notice, Info, Debug -- plus Log and Write for an
// arbitrary level, WithContext and WithoutContext for the context every future
// line carries, and Listen for the MessageLogged event each written line fires.
//
// LogManager is the channels: Channel and Driver resolve one by name out of
// Config and cache it, Stack fans one line out to several, Build makes a channel
// the configuration does not name, Extend registers a driver the manager does
// not know, and ShareContext gives every channel the same fields. It is a logger
// itself, writing to the default channel.
//
// Both are handed what they need rather than reaching for it: the configuration
// as a Config, the event dispatcher as the Dispatcher interface.
//
// A failure returns an error, and the emergency logger alongside it, so that a
// broken logging configuration does not take the request down with it: a caller
// who wants to keep logging ignores the error, and a caller who wants to know
// reads it.
//
// # A channel is a handler
//
// A channel is a slog.Handler, a stack is a handler that writes into several,
// and the context processor is a handler that wraps another -- which is what
// log/context.ContextLogProcessor is. A formatter is the ChannelConfig.Format
// field, which selects between the two handlers slog ships.
//
// The eight drivers a channel can name are single, daily, monthly, stack,
// stderr, errorlog, null and custom.
//
// # Context
//
// The subpackage log/context is the context that crosses a whole request and
// lands in every line of it: Add, AddHidden, Push, Pop, Pull, Only, Except,
// Increment, Scope, Dehydrate. The repository lives in the context.Context, and
// that package's documentation says why.
//
// # The Collector
//
// The Collector is the development layer, and it is core rather than a plugin,
// so the error page knows the queries, the dumps and the routes without any
// extra install. It accumulates everything that happened inside one request;
// Recorder keeps the last of them in a ring; Console serves them at
// /_arandu/debug and says the probable cause out loud -- the N+1, the slow
// statement, the request that was three quarters database. Transport records
// outbound calls onto the same timeline, Dump records a value without writing to
// stdout or corrupting the response, and EditorLink turns a recorded frame into
// a link that opens the file in the IDE at the line.
//
// It costs nothing when it is off. Outside development, and without an
// authorized tracing header, FromContext returns nil and every Record method is
// a no-op on a nil receiver -- zero cost, not "low cost", and there are
// allocation tests that hold it to that.
//
// [DumpDie] is the one exception, and deliberately: the recording half does
// nothing without a Collector, and the die half runs everywhere. A forgotten
// call ends the request wherever it is made, because the alternative is a 200
// with the dump written into the middle of the page.
//
// The error page itself is not here: it renders a failure rather than recording
// one, and it lives in the exception package.
package log
