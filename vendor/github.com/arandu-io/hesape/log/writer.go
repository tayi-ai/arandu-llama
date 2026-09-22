package log

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"maps"
	"slices"
	"sync"

	"github.com/arandu-io/hesape/log/events"
)

// Dispatcher is the slice of an event dispatcher that this package needs:
// something to fire an event into and something to register a listener with.
//
// It is declared here, on the side that consumes it, rather than imported from
// the events package, so that one concrete dispatcher can serve every package
// that fires an event -- a Go method cannot be overloaded, so Dispatch has to
// take any rather than one concrete event type.
type Dispatcher interface {
	// Dispatch fires the event.
	Dispatch(event any)

	// Listen registers a listener that receives every event. Selecting one type
	// out of them is a type assertion, which is what Logger.Listen wraps.
	Listen(listener func(event any))
}

// ErrNoDispatcher is what Listen answers on a Logger built without a dispatcher.
//
// The dispatcher is optional, so this is not a broken Logger: it writes every
// line it is given and fires no event. What it cannot do is register a listener,
// because there is nothing to register with -- and answering an error there is
// better than accepting a callback that would never be called.
var ErrNoDispatcher = errors.New("log: events dispatcher has not been set")

// Logger is the eight levels, an accumulated context that every future line
// carries, and a MessageLogged event per line.
//
// It wraps a *slog.Logger: slog already is the chainable handler, and it is
// stdlib rather than a dependency.
//
// A Logger is safe for concurrent use.
type Logger struct {
	mu         sync.RWMutex
	logger     *slog.Logger
	dispatcher Dispatcher
	logContext map[string]any
}

// NewLogger returns a Logger writing to logger and firing its events on
// dispatcher.
//
// dispatcher may be nil: a Logger without one writes lines and fires no event,
// and Listen then reports ErrNoDispatcher. A nil logger falls back to
// slog.Default rather than panicking on the first line.
func NewLogger(logger *slog.Logger, dispatcher Dispatcher) *Logger {
	if logger == nil {
		logger = slog.Default()
	}
	return &Logger{logger: logger, dispatcher: dispatcher, logContext: map[string]any{}}
}

// Emergency logs that the system is unusable.
//
// slog hands ctx to the handler, which is how the request a line belongs to
// reaches the output. fields is the structured context of the line, and the
// variadic takes several maps, merged left to right with the last winning. Both
// notes hold for the other seven levels and for Log and Write.
func (l *Logger) Emergency(ctx context.Context, message any, fields ...map[string]any) {
	l.writeLog(ctx, LevelEmergency, message, fields)
}

// Alert logs that action must be taken immediately.
//
// Example: entire website down, database unavailable. This should trigger the
// alerts and wake you up.
func (l *Logger) Alert(ctx context.Context, message any, fields ...map[string]any) {
	l.writeLog(ctx, LevelAlert, message, fields)
}

// Critical logs a critical condition.
//
// Example: application component unavailable, unexpected exception.
func (l *Logger) Critical(ctx context.Context, message any, fields ...map[string]any) {
	l.writeLog(ctx, LevelCritical, message, fields)
}

// Error logs a runtime error that does not require immediate action but should
// typically be logged and monitored.
func (l *Logger) Error(ctx context.Context, message any, fields ...map[string]any) {
	l.writeLog(ctx, LevelError, message, fields)
}

// Warning logs an exceptional occurrence that is not an error.
//
// Example: use of deprecated APIs, poor use of an API, undesirable things that
// are not necessarily wrong.
func (l *Logger) Warning(ctx context.Context, message any, fields ...map[string]any) {
	l.writeLog(ctx, LevelWarning, message, fields)
}

// Notice logs a normal but significant event.
func (l *Logger) Notice(ctx context.Context, message any, fields ...map[string]any) {
	l.writeLog(ctx, LevelNotice, message, fields)
}

// Info logs an interesting event.
//
// Example: user logs in, SQL logs.
func (l *Logger) Info(ctx context.Context, message any, fields ...map[string]any) {
	l.writeLog(ctx, LevelInfo, message, fields)
}

// Debug logs detailed debug information.
func (l *Logger) Debug(ctx context.Context, message any, fields ...map[string]any) {
	l.writeLog(ctx, LevelDebug, message, fields)
}

// Log logs a message at an arbitrary level.
//
// level is a slog.Level, which is the value the handler compares against;
// ParseLevel turns a configured name into one.
func (l *Logger) Log(ctx context.Context, level slog.Level, message any, fields ...map[string]any) {
	l.writeLog(ctx, level, message, fields)
}

// Write is an alias of Log: both reach writeLog with the same three arguments.
func (l *Logger) Write(ctx context.Context, level slog.Level, message any, fields ...map[string]any) {
	l.writeLog(ctx, level, message, fields)
}

// writeLog is the body every level goes through.
//
// The order is the behaviour: a level the handler does not handle returns before
// writing and before firing the event, so a listener never sees a line that was
// never written.
func (l *Logger) writeLog(ctx context.Context, level slog.Level, message any, fields []map[string]any) {
	if l == nil {
		return
	}
	if ctx == nil {
		ctx = context.Background()
	}

	l.mu.RLock()
	logger, dispatcher := l.logger, l.dispatcher
	merged := make(map[string]any, len(l.logContext))
	maps.Copy(merged, l.logContext)
	l.mu.RUnlock()

	if !logger.Enabled(ctx, level) {
		return
	}

	for _, f := range fields {
		maps.Copy(merged, f)
	}

	text := formatMessage(message)
	logger.LogAttrs(ctx, level, text, contextAttrs(merged)...)
	l.fireLogEvent(dispatcher, level, text, merged)
}

// WithContext adds context to all future logs.
//
// It merges rather than replaces, and returns the receiver so that calls chain.
// No argument merges nothing and is not an error.
func (l *Logger) WithContext(fields ...map[string]any) *Logger {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.logContext == nil {
		l.logContext = map[string]any{}
	}
	for _, f := range fields {
		maps.Copy(l.logContext, f)
	}
	return l
}

// WithoutContext drops the given keys from the accumulated context, or drops all
// of it.
//
// Calling it with no key clears everything; calling it with keys drops exactly
// those and leaves the rest. A key that is not there is not an error.
func (l *Logger) WithoutContext(keys ...string) *Logger {
	if l == nil {
		return nil
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if len(keys) == 0 {
		l.logContext = map[string]any{}
		return l
	}
	for _, key := range keys {
		delete(l.logContext, key)
	}
	return l
}

// Listen registers a callback for when a log event is fired.
//
// A Logger built without a dispatcher reports ErrNoDispatcher and registers
// nothing. The callback receives only MessageLogged.
func (l *Logger) Listen(callback func(events.MessageLogged)) error {
	if l == nil {
		return ErrNoDispatcher
	}
	l.mu.RLock()
	dispatcher := l.dispatcher
	l.mu.RUnlock()

	if dispatcher == nil {
		return ErrNoDispatcher
	}
	dispatcher.Listen(func(event any) {
		if logged, ok := event.(events.MessageLogged); ok {
			callback(logged)
		}
	})
	return nil
}

// fireLogEvent dispatches the MessageLogged for one written line.
//
// Nothing guards against firing twice, because nothing can: the wrapped logger
// is a *slog.Logger and never a LogManager. LogManager holds Loggers, a Logger
// never holds a LogManager, and one line can only be fired once.
func (l *Logger) fireLogEvent(dispatcher Dispatcher, level slog.Level, message string, fields map[string]any) {
	if dispatcher == nil {
		return
	}
	dispatcher.Dispatch(events.MessageLogged{Level: level, Message: message, Context: fields})
}

// GetLogger returns the underlying *slog.Logger.
func (l *Logger) GetLogger() *slog.Logger {
	if l == nil {
		return nil
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.logger
}

// GetEventDispatcher returns the dispatcher, or nil when none was set.
func (l *Logger) GetEventDispatcher() Dispatcher {
	if l == nil {
		return nil
	}
	l.mu.RLock()
	defer l.mu.RUnlock()
	return l.dispatcher
}

// SetEventDispatcher sets the dispatcher the Logger fires its events on.
func (l *Logger) SetEventDispatcher(dispatcher Dispatcher) {
	if l == nil {
		return
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	l.dispatcher = dispatcher
}

// formatMessage renders a message of any type as the line's text.
//
// A string is itself; a []byte, an error and a fmt.Stringer each have one
// obvious rendering; and anything structured becomes JSON, which is the
// printable rendering of its shape.
func formatMessage(message any) string {
	switch m := message.(type) {
	case nil:
		return ""
	case string:
		return m
	case []byte:
		return string(m)
	case error:
		return m.Error()
	case fmt.Stringer:
		return m.String()
	}
	if encoded, err := json.Marshal(message); err == nil {
		return string(encoded)
	}
	return fmt.Sprint(message)
}

// contextAttrs turns a line's context map into slog attributes.
//
// The keys are sorted because a Go map has no order: without this, two identical
// lines would print their fields in a different order each run, and a test that
// reads the output would be flaky.
func contextAttrs(fields map[string]any) []slog.Attr {
	if len(fields) == 0 {
		return nil
	}
	keys := make([]string, 0, len(fields))
	for key := range fields {
		keys = append(keys, key)
	}
	slices.Sort(keys)

	out := make([]slog.Attr, 0, len(keys))
	for _, key := range keys {
		out = append(out, slog.Any(key, fields[key]))
	}
	return out
}
