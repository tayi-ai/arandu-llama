package log

import (
	"context"
	"log/slog"
	"net/http"
	"os"
	"strings"
)

type (
	ctxLoggerKey struct{}
	ctxFieldsKey struct{}
)

// The eight severity levels, expressed as slog levels.
//
// slog names four of them -- Debug, Info, Warn, Error -- and leaves the numeric
// space between them open exactly so a package can fill it. A level that parses
// but logs as "ERROR+4" is a level nobody trusts, so New renders all eight by
// name.
//
// Warning is another spelling of slog's Warn; it is the same level, not a second
// one.
const (
	LevelDebug     = slog.LevelDebug
	LevelInfo      = slog.LevelInfo
	LevelNotice    = slog.Level(2)
	LevelWarning   = slog.LevelWarn
	LevelError     = slog.LevelError
	LevelCritical  = slog.Level(12)
	LevelAlert     = slog.Level(16)
	LevelEmergency = slog.Level(20)
)

// levelNames is the closed set, in both directions. ParseLevel reads it and the
// handler renders through it, so a level can never print under a name that
// ParseLevel would not accept back.
var levelNames = []struct {
	name  string
	level slog.Level
}{
	{"debug", LevelDebug},
	{"info", LevelInfo},
	{"notice", LevelNotice},
	{"warning", LevelWarning},
	{"error", LevelError},
	{"critical", LevelCritical},
	{"alert", LevelAlert},
	{"emergency", LevelEmergency},
}

// ParseLevel turns a configured level name into a level. The half that reads the
// configuration and falls back to debug is LogManager's levelLocked.
//
// It accepts the eight names exactly as spelled, and nothing else -- no case
// folding, no trimming, and no "warn" beside "warning". Two names for one level
// is two spellings of the same configuration, and the handler renders the level
// back as "warning", so a second spelling would configure a level that never
// prints under the name it was written with.
//
// An unknown name is an error rather than a silent fallback: a typo in LOG_LEVEL
// that quietly restores the default is how a production process ends up logging
// more than it was told to.
//
// The level returned beside the error is debug, which is what
// LogManager.levelLocked falls back to for a configuration with no level in it
// at all: two different defaults for the same case is one of them being wrong
// wherever the two are read next to each other.
func ParseLevel(s string) (slog.Level, error) {
	for _, l := range levelNames {
		if l.name == s {
			return l.level, nil
		}
	}
	return LevelDebug, &UnknownLevelError{Name: s}
}

// UnknownLevelError is what ParseLevel returns for a name outside the eight.
type UnknownLevelError struct{ Name string }

// Error names the rejected value and lists the eight names that would have been
// accepted, because the only reason anybody reads this message is to fix the
// value in the configuration.
func (e *UnknownLevelError) Error() string {
	names := make([]string, 0, len(levelNames))
	for _, l := range levelNames {
		names = append(names, l.name)
	}
	return "log: unknown level " + e.Name + ", want one of " + strings.Join(names, ", ")
}

// levelName renders a level under its own name, falling back to slog's text for
// a level nobody here defined.
func levelName(l slog.Level) string {
	for _, known := range levelNames {
		if known.level == l {
			return strings.ToUpper(known.name)
		}
	}
	return l.String()
}

// New returns the root logger, which is what a process has before any
// configuration has been read: readable text in development, JSON everywhere
// else, so it reaches the aggregator without fragile parsing.
func New(env string, level slog.Level) *slog.Logger {
	opts := &slog.HandlerOptions{
		Level:     level,
		AddSource: true,
		ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
			if len(groups) == 0 && a.Key == slog.LevelKey {
				if l, ok := a.Value.Any().(slog.Level); ok {
					a.Value = slog.StringValue(levelName(l))
				}
			}
			return a
		},
	}
	if env == "dev" {
		return slog.New(slog.NewTextHandler(os.Stdout, opts))
	}
	return slog.New(slog.NewJSONHandler(os.Stdout, opts))
}

// Into stores the request-scoped logger in the context.
func Into(ctx context.Context, l *slog.Logger) context.Context {
	return context.WithValue(ctx, ctxLoggerKey{}, l)
}

// Middleware installs the application logger at the very top of the pipeline.
//
// Without it, For(ctx) inside a request falls back to slog.Default(), which
// ignores the configured handler: production would emit its request lines in the
// default text format instead of the JSON the aggregator expects, and the level
// filter from the configuration would not apply either. The Kernel installs this
// as the outermost middleware, so even a panic in Recover logs correctly.
func Middleware(l *slog.Logger) func(http.Handler) http.Handler {
	return func(next http.Handler) http.Handler {
		return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			next.ServeHTTP(w, r.WithContext(Into(r.Context(), l)))
		})
	}
}

// For is the read side of Into: it returns the request logger, and it never
// returns nil.
//
// This is the only way to log inside a handler or a service. There is no
// exported global logger, on purpose: a log line without request_id and tenant
// is noise, and the only way to guarantee both is to force the context through.
func For(ctx context.Context) *slog.Logger {
	if l, ok := ctx.Value(ctxLoggerKey{}).(*slog.Logger); ok && l != nil {
		return l
	}
	return slog.Default()
}

// With attaches fields to the logger the context carries, and returns the
// context. [Logger.WithContext] is the other half of the pair, and it mutates
// one Logger instead.
//
// It replaces the shape that was written everywhere:
//
//	log := log.For(ctx).With("component", "worker", "queue", name)
//
// which decorates a local variable and leaves everything called from there --
// the repository, the mailer, the panic handler -- logging without the fields.
// With puts them in the context, so anything downstream that asks For(ctx) gets
// them too.
//
// The arguments are slog's: alternating key and value, or a slog.Attr, which is
// what Field returns.
func With(ctx context.Context, args ...any) context.Context {
	if len(args) == 0 {
		return ctx
	}
	carried := append(Fields(ctx), attrs(args)...)
	ctx = Into(ctx, For(ctx).With(args...))
	return context.WithValue(ctx, ctxFieldsKey{}, carried)
}

// Field is one structured field.
//
// It is slog.Any under a name that says what it is at the call site, and it is
// what lets a caller pass a typed value where a bare key/value pair would leave
// the reader counting arguments.
func Field(key string, value any) slog.Attr {
	return slog.Any(key, value)
}

// Fields returns the fields With attached to this context, in the order they
// were attached.
//
// A *slog.Logger will not say what it carries, so anything that has to forward
// the request's fields somewhere that is not slog -- the error page, an outbound
// header, a test -- has no way to read them back. This is that way.
func Fields(ctx context.Context) []slog.Attr {
	carried, _ := ctx.Value(ctxFieldsKey{}).([]slog.Attr)
	if len(carried) == 0 {
		return nil
	}
	// A copy, because append on the value stored in a parent context would write
	// into a slice that a sibling context is also using.
	return append([]slog.Attr(nil), carried...)
}

// badKey is what slog itself names a value that arrived without a key.
const badKey = "!BADKEY"

// attrs reads slog's loose argument list the way slog does, so Fields reports
// exactly what the logger recorded.
func attrs(args []any) []slog.Attr {
	out := make([]slog.Attr, 0, len(args))
	for i := 0; i < len(args); {
		switch arg := args[i].(type) {
		case slog.Attr:
			out = append(out, arg)
			i++
		case string:
			if i+1 < len(args) {
				out = append(out, slog.Any(arg, args[i+1]))
				i += 2
				continue
			}
			out = append(out, slog.String(badKey, arg))
			i++
		default:
			out = append(out, slog.Any(badKey, arg))
			i++
		}
	}
	return out
}
