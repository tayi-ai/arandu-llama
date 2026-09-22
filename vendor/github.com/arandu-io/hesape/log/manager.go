package log

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"maps"
	"os"
	"path/filepath"
	"slices"
	"strings"
	"sync"
	"time"

	logcontext "github.com/arandu-io/hesape/log/context"
)

// Config is the logging configuration: the default channel and the channels
// themselves.
//
// A manager is handed it whole rather than resolving it from anywhere.
type Config struct {
	// Default names the channel Driver and Channel resolve when asked for none.
	Default string

	// Env is the application environment, and it decides two things: the name
	// stamped on a channel that did not name itself, and the format of a channel
	// that did not choose one -- readable text under "dev", JSON everywhere else.
	Env string

	// Channels are the configured channels, by name.
	Channels map[string]ChannelConfig
}

// ChannelConfig is the configuration of one channel.
type ChannelConfig struct {
	// Driver selects the implementation: "single", "daily", "monthly", "stack",
	// "stderr", "errorlog", "null", "custom", or a name registered with Extend.
	Driver string

	// Name is the channel name stamped on every line. Empty falls back to
	// Config.Env.
	Name string

	// Level is the lowest level the channel writes, one of the eight names
	// ParseLevel accepts. Empty is "debug".
	Level string

	// Path is the file the single, daily and monthly drivers write to.
	Path string

	// Days is how many daily files to keep. Zero means 7. MaxFiles wins over it
	// when both are set.
	Days int

	// MaxFiles is how many rotated files to keep. It is read before Days on the
	// daily driver, and it is the only count the monthly driver reads. Zero is
	// the default the driver names: 7 files for daily, 3 for monthly.
	MaxFiles int

	// Channels are the channels a stack fans out to.
	Channels []string

	// IgnoreExceptions swallows what a stack's handlers report. Every handler is
	// written to either way; without it the failures come back joined.
	IgnoreExceptions bool

	// Format is "text" or "json", the two handlers slog ships. Empty follows
	// Config.Env.
	Format string

	// Writer is where the stderr and errorlog drivers write. Empty means the
	// process error output.
	Writer io.Writer

	// Via is the factory the custom driver calls to build the logger.
	Via func(config ChannelConfig) (*slog.Logger, error)
}

// LogManager resolves channels by name, caches them, shares context across them,
// and is itself a logger that writes to the default channel.
//
// A LogManager is safe for concurrent use.
type LogManager struct {
	mu             sync.Mutex
	config         Config
	dispatcher     Dispatcher
	channels       map[string]*Logger
	sharedContext  map[string]any
	customCreators map[string]func(config ChannelConfig) (*slog.Logger, error)
}

// NewLogManager returns a manager over config, firing its events on dispatcher.
//
// Both may be zero: a manager with no channels resolves nothing and falls back
// to the emergency logger, which is also where a missing channel lands.
func NewLogManager(config Config, dispatcher Dispatcher) *LogManager {
	return &LogManager{
		config:         config,
		dispatcher:     dispatcher,
		channels:       map[string]*Logger{},
		sharedContext:  map[string]any{},
		customCreators: map[string]func(config ChannelConfig) (*slog.Logger, error){},
	}
}

// Build returns an on-demand channel from a configuration that is not in Config.
//
// It drops the previously built one first, so two Build calls never hand back
// the same logger.
func (m *LogManager) Build(config ChannelConfig) (*Logger, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.channels, onDemandChannel)
	return m.getLocked(onDemandChannel, &config)
}

// onDemandChannel is the cache slot Build clears and then resolves into.
const onDemandChannel = "ondemand"

// Stack returns a new aggregate logger over the named channels.
//
// channel names the stack, and empty falls back to Config.Env. The result is not
// cached.
func (m *LogManager) Stack(channels []string, channel string) (*Logger, error) {
	m.mu.Lock()
	defer m.mu.Unlock()

	logger, err := m.createStackDriverLocked(ChannelConfig{Channels: channels, Name: channel})
	if err != nil {
		return m.createEmergencyLoggerLocked(err), err
	}
	return NewLogger(logger, m.dispatcher).WithContext(m.sharedContext), nil
}

// Channel returns the channel by name, or the default one when the name is
// empty.
//
// A failure returns the emergency logger and the error both, so a caller that
// wants to keep logging can ignore the error and a caller that wants to know can
// read it.
func (m *LogManager) Channel(channel string) (*Logger, error) {
	return m.Driver(channel)
}

// Driver returns the channel by name, and is what Channel calls.
func (m *LogManager) Driver(driver string) (*Logger, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.getLocked(m.parseDriverLocked(driver), nil)
}

// getLocked is the cache, then the resolve, then the shared context, and the
// emergency logger when any of it fails.
//
// The caller holds m.mu. It is recursive -- a stack resolves its members through
// it -- which is why the lock is taken by the exported methods and not here.
func (m *LogManager) getLocked(name string, config *ChannelConfig) (*Logger, error) {
	if cached, ok := m.channels[name]; ok {
		return cached, nil
	}

	logger, err := m.resolveLocked(name, config)
	if err != nil {
		return m.createEmergencyLoggerLocked(err), err
	}

	channel := NewLogger(logger, m.dispatcher).WithContext(m.sharedContext)
	m.channels[name] = channel
	return channel, nil
}

// resolveLocked finds the configuration for the name, then builds the driver
// that configuration asks for.
func (m *LogManager) resolveLocked(name string, config *ChannelConfig) (*slog.Logger, error) {
	if config == nil {
		found, ok := m.config.Channels[name]
		if !ok {
			return nil, fmt.Errorf("log: log [%s] is not defined", name)
		}
		config = &found
	}

	if creator, ok := m.customCreators[config.Driver]; ok {
		return creator(*config)
	}

	switch config.Driver {
	case "single":
		return m.createSingleDriverLocked(*config)
	case "daily":
		return m.createDailyDriverLocked(*config)
	case "monthly":
		return m.createMonthlyDriverLocked(*config)
	case "stack":
		return m.createStackDriverLocked(*config)
	case "stderr", "errorlog":
		return m.createErrorlogDriverLocked(*config)
	case "null":
		return m.createNullDriverLocked(*config)
	case "custom":
		return m.createCustomDriverLocked(*config)
	default:
		return nil, fmt.Errorf("log: driver [%s] is not supported", config.Driver)
	}
}

// createSingleDriverLocked builds the single driver: one file, appended to.
func (m *LogManager) createSingleDriverLocked(config ChannelConfig) (*slog.Logger, error) {
	if config.Path == "" {
		return nil, errors.New("log: the single driver needs a path")
	}
	file, err := openLogFile(config.Path)
	if err != nil {
		return nil, err
	}
	return m.buildLoggerLocked(config, file)
}

// createDailyDriverLocked builds the daily driver: one file per day, keeping
// only the newest of them.
//
// The count is MaxFiles, then Days, then 7.
func (m *LogManager) createDailyDriverLocked(config ChannelConfig) (*slog.Logger, error) {
	keep := config.MaxFiles
	if keep <= 0 {
		keep = config.Days
	}
	if keep <= 0 {
		keep = defaultDailyFiles
	}
	return m.createRotatingDriverLocked(config, filePerDay, keep, "daily")
}

// createMonthlyDriverLocked builds the monthly driver: one file per month,
// keeping the last MaxFiles of them, three by default.
func (m *LogManager) createMonthlyDriverLocked(config ChannelConfig) (*slog.Logger, error) {
	keep := config.MaxFiles
	if keep <= 0 {
		keep = defaultMonthlyFiles
	}
	return m.createRotatingDriverLocked(config, filePerMonth, keep, "monthly")
}

// createRotatingDriverLocked is the body the daily and monthly drivers share.
func (m *LogManager) createRotatingDriverLocked(config ChannelConfig, format string, keep int, driver string) (*slog.Logger, error) {
	if config.Path == "" {
		return nil, fmt.Errorf("log: the %s driver needs a path", driver)
	}
	return m.buildLoggerLocked(config, &rotatingWriter{path: config.Path, format: format, keep: keep})
}

// The two rotation periods, written as Go layouts. They are what goes in the
// file name between the base and the extension.
const (
	filePerDay   = time.DateOnly
	filePerMonth = "2006-01"
)

// The counts the two rotating drivers fall back to when the configuration names
// none.
const (
	defaultDailyFiles   = 7
	defaultMonthlyFiles = 3
)

// createStackDriverLocked builds the stack driver: one logger over the handlers
// of every named channel.
//
// Every handler keeps the name of the channel that configured it, because a
// stack here is those handlers themselves and stamping the stack's own name over
// them would put two `channel` keys on one line.
func (m *LogManager) createStackDriverLocked(config ChannelConfig) (*slog.Logger, error) {
	if len(config.Channels) == 0 {
		return nil, errors.New("log: a stack needs at least one channel")
	}

	handlers := make([]slog.Handler, 0, len(config.Channels))
	for _, name := range config.Channels {
		channel, err := m.getLocked(strings.TrimSpace(name), nil)
		if err != nil {
			return nil, err
		}
		handlers = append(handlers, channel.GetLogger().Handler())
	}
	return slog.New(&multiHandler{handlers: handlers, ignoreExceptions: config.IgnoreExceptions}), nil
}

// createErrorlogDriverLocked builds the errorlog driver, and the stderr driver
// with it: both write to the process error output.
func (m *LogManager) createErrorlogDriverLocked(config ChannelConfig) (*slog.Logger, error) {
	destination := config.Writer
	if destination == nil {
		destination = os.Stderr
	}
	return m.buildLoggerLocked(config, destination)
}

// createNullDriverLocked builds the null driver: it discards everything.
func (m *LogManager) createNullDriverLocked(config ChannelConfig) (*slog.Logger, error) {
	return m.buildLoggerLocked(config, io.Discard)
}

// createCustomDriverLocked builds the custom driver: the factory in
// ChannelConfig.Via builds the logger.
func (m *LogManager) createCustomDriverLocked(config ChannelConfig) (*slog.Logger, error) {
	if config.Via == nil {
		return nil, errors.New("log: the custom driver needs a via factory")
	}
	return config.Via(config)
}

// buildLoggerLocked is the part every driver shares: the level, the format, the
// channel name, and the context processor every resolved channel carries.
func (m *LogManager) buildLoggerLocked(config ChannelConfig, destination io.Writer) (*slog.Logger, error) {
	level, err := m.levelLocked(config)
	if err != nil {
		return nil, err
	}
	handler := logcontext.NewContextLogProcessor(newHandler(destination, m.formatLocked(config), level))
	return slog.New(handler).With(slog.String("channel", m.parseChannelLocked(config))), nil
}

// levelLocked resolves the channel level: the configured one, or debug, and an
// error when the configured name is not a level.
func (m *LogManager) levelLocked(config ChannelConfig) (slog.Level, error) {
	if config.Level == "" {
		return LevelDebug, nil
	}
	level, err := ParseLevel(config.Level)
	if err != nil {
		return LevelDebug, fmt.Errorf("log: invalid log level: %w", err)
	}
	return level, nil
}

// parseChannelLocked resolves the name stamped on every line: the configured
// name, then the environment, then "production".
func (m *LogManager) parseChannelLocked(config ChannelConfig) string {
	if config.Name != "" {
		return config.Name
	}
	if m.config.Env != "" {
		return m.config.Env
	}
	return "production"
}

// formatLocked resolves the channel format: what it asked for, then what the
// environment implies.
func (m *LogManager) formatLocked(config ChannelConfig) string {
	if config.Format != "" {
		return strings.ToLower(config.Format)
	}
	if m.config.Env == "dev" {
		return "text"
	}
	return "json"
}

// parseDriverLocked trims the name, and falls back to the default when none was
// given.
func (m *LogManager) parseDriverLocked(driver string) string {
	driver = strings.TrimSpace(driver)
	if driver == "" {
		driver = strings.TrimSpace(m.config.Default)
	}
	return driver
}

// createEmergencyLoggerLocked builds the logger that exists so a broken logging
// configuration does not take the request down with it.
//
// It writes to the path of an "emergency" channel when one is configured, and to
// the process error output otherwise, because there is no application storage
// path to guess at here. It logs the failure on the way out.
func (m *LogManager) createEmergencyLoggerLocked(cause error) *Logger {
	var destination io.Writer = os.Stderr
	if config, ok := m.config.Channels["emergency"]; ok && config.Path != "" {
		if file, err := openLogFile(config.Path); err == nil {
			destination = file
		}
	}

	logger := NewLogger(slog.New(newHandler(destination, m.formatLocked(ChannelConfig{}), LevelDebug)), m.dispatcher)
	logger.Emergency(context.Background(), "Unable to create configured logger. Using emergency logger.", map[string]any{
		"exception": cause.Error(),
	})
	return logger
}

// ShareContext adds context that every channel gets, the ones already resolved
// included.
func (m *LogManager) ShareContext(fields map[string]any) *LogManager {
	m.mu.Lock()
	channels := slices.Collect(maps.Values(m.channels))
	maps.Copy(m.sharedContext, fields)
	m.mu.Unlock()

	for _, channel := range channels {
		channel.WithContext(fields)
	}
	return m
}

// SharedContext returns the context shared across channels and stacks.
//
// It is a copy, so writing to it changes nothing the manager holds.
func (m *LogManager) SharedContext() map[string]any {
	m.mu.Lock()
	defer m.mu.Unlock()
	return maps.Clone(m.sharedContext)
}

// WithoutContext drops the given keys from every resolved channel, or clears
// them all when given none.
//
// It leaves the shared context alone: the two are separate, and
// FlushSharedContext is the one that clears the shared half.
func (m *LogManager) WithoutContext(keys ...string) *LogManager {
	m.mu.Lock()
	channels := slices.Collect(maps.Values(m.channels))
	m.mu.Unlock()

	for _, channel := range channels {
		channel.WithoutContext(keys...)
	}
	return m
}

// FlushSharedContext clears the context shared across channels.
func (m *LogManager) FlushSharedContext() *LogManager {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.sharedContext = map[string]any{}
	return m
}

// GetDefaultDriver returns the name of the default channel.
func (m *LogManager) GetDefaultDriver() string {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.config.Default
}

// SetDefaultDriver sets the name of the default channel.
func (m *LogManager) SetDefaultDriver(name string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.config.Default = name
}

// Extend registers a factory for a driver name the manager does not know.
//
// The factory receives only the channel configuration; nothing inside the
// manager is reachable from it.
func (m *LogManager) Extend(driver string, callback func(config ChannelConfig) (*slog.Logger, error)) *LogManager {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.customCreators[driver] = callback
	return m
}

// ForgetChannel drops the resolved channel so the next call builds it again.
func (m *LogManager) ForgetChannel(driver string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.channels, m.parseDriverLocked(driver))
}

// GetChannels returns every channel resolved so far, by name. It is a copy of
// the map, and the loggers in it are the live ones.
func (m *LogManager) GetChannels() map[string]*Logger {
	m.mu.Lock()
	defer m.mu.Unlock()
	return maps.Clone(m.channels)
}

// Emergency logs at the emergency level. The eight levels and Log all write to
// the default channel.
func (m *LogManager) Emergency(ctx context.Context, message any, fields ...map[string]any) {
	m.Log(ctx, LevelEmergency, message, fields...)
}

// Alert logs at the alert level.
func (m *LogManager) Alert(ctx context.Context, message any, fields ...map[string]any) {
	m.Log(ctx, LevelAlert, message, fields...)
}

// Critical logs at the critical level.
func (m *LogManager) Critical(ctx context.Context, message any, fields ...map[string]any) {
	m.Log(ctx, LevelCritical, message, fields...)
}

// Error logs at the error level.
func (m *LogManager) Error(ctx context.Context, message any, fields ...map[string]any) {
	m.Log(ctx, LevelError, message, fields...)
}

// Warning logs at the warning level.
func (m *LogManager) Warning(ctx context.Context, message any, fields ...map[string]any) {
	m.Log(ctx, LevelWarning, message, fields...)
}

// Notice logs at the notice level.
func (m *LogManager) Notice(ctx context.Context, message any, fields ...map[string]any) {
	m.Log(ctx, LevelNotice, message, fields...)
}

// Info logs at the info level.
func (m *LogManager) Info(ctx context.Context, message any, fields ...map[string]any) {
	m.Log(ctx, LevelInfo, message, fields...)
}

// Debug logs at the debug level.
func (m *LogManager) Debug(ctx context.Context, message any, fields ...map[string]any) {
	m.Log(ctx, LevelDebug, message, fields...)
}

// Log logs at an arbitrary level, on the default channel.
//
// A channel that will not resolve does not silence the line: Driver hands back
// the emergency logger, and the line goes there.
func (m *LogManager) Log(ctx context.Context, level slog.Level, message any, fields ...map[string]any) {
	channel, _ := m.Driver("")
	channel.Log(ctx, level, message, fields...)
}

// newHandler builds the handler for a format: the two slog ships, rendering the
// level under one of the eight names ParseLevel accepts back.
func newHandler(destination io.Writer, format string, level slog.Level) slog.Handler {
	opts := &slog.HandlerOptions{
		Level: level,
		ReplaceAttr: func(groups []string, a slog.Attr) slog.Attr {
			if len(groups) == 0 && a.Key == slog.LevelKey {
				if l, ok := a.Value.Any().(slog.Level); ok {
					a.Value = slog.StringValue(levelName(l))
				}
			}
			return a
		},
	}
	if format == "text" {
		return slog.NewTextHandler(destination, opts)
	}
	return slog.NewJSONHandler(destination, opts)
}

// multiHandler is the stack: one record into every handler.
//
// ignoreExceptions swallows what the handlers report: a handler that fails does
// not stop the ones after it and does not surface. Without it, the failures come
// back joined.
type multiHandler struct {
	handlers         []slog.Handler
	ignoreExceptions bool
}

// Enabled reports whether any handler in the stack wants the level.
func (h *multiHandler) Enabled(ctx context.Context, level slog.Level) bool {
	for _, sub := range h.handlers {
		if sub.Enabled(ctx, level) {
			return true
		}
	}
	return false
}

// Handle writes the record into every handler that wants it.
func (h *multiHandler) Handle(ctx context.Context, record slog.Record) error {
	var failures []error
	for _, sub := range h.handlers {
		if !sub.Enabled(ctx, record.Level) {
			continue
		}
		if err := sub.Handle(ctx, record.Clone()); err != nil {
			failures = append(failures, err)
		}
	}
	if h.ignoreExceptions {
		return nil
	}
	return errors.Join(failures...)
}

// WithAttrs pushes the attributes down into every handler.
func (h *multiHandler) WithAttrs(attrs []slog.Attr) slog.Handler {
	out := make([]slog.Handler, len(h.handlers))
	for i, sub := range h.handlers {
		out[i] = sub.WithAttrs(slices.Clone(attrs))
	}
	return &multiHandler{handlers: out, ignoreExceptions: h.ignoreExceptions}
}

// WithGroup pushes the group down into every handler.
func (h *multiHandler) WithGroup(name string) slog.Handler {
	if name == "" {
		return h
	}
	out := make([]slog.Handler, len(h.handlers))
	for i, sub := range h.handlers {
		out[i] = sub.WithGroup(name)
	}
	return &multiHandler{handlers: out, ignoreExceptions: h.ignoreExceptions}
}

// openLogFile opens a log file for appending, creating the directory it lives in
// when that directory is missing.
func openLogFile(path string) (*os.File, error) {
	if dir := filepath.Dir(path); dir != "" && dir != "." {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, err
		}
	}
	return os.OpenFile(path, os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
}

// rotatingWriter writes one file per period, named after the configured path
// with the date before the extension, and keeps only the last `keep` of them.
//
// format is the period -- filePerDay or filePerMonth -- and it is the whole
// difference between the daily and the monthly driver.
type rotatingWriter struct {
	mu      sync.Mutex
	path    string
	format  string
	keep    int
	current string
	file    *os.File
}

// Write appends to the current period's file, opening it -- and pruning the old
// ones -- the first time that period is written to.
func (w *rotatingWriter) Write(p []byte) (int, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	format := w.format
	if format == "" {
		format = filePerDay
	}

	period := time.Now().Format(format)
	if w.file == nil || period != w.current {
		if w.file != nil {
			_ = w.file.Close()
			w.file = nil
		}
		file, err := openLogFile(rotatingPath(w.path, period))
		if err != nil {
			return 0, err
		}
		w.file, w.current = file, period
		w.prune()
	}
	return w.file.Write(p)
}

// rotatingPath builds the rotated file name: base-YYYY-MM-DD.ext for a daily
// channel and base-YYYY-MM.ext for a monthly one.
func rotatingPath(path, period string) string {
	ext := filepath.Ext(path)
	return strings.TrimSuffix(path, ext) + "-" + period + ext
}

// prune deletes everything past the newest `keep` files, on every rotation.
func (w *rotatingWriter) prune() {
	if w.keep <= 0 {
		return
	}
	ext := filepath.Ext(w.path)
	matches, err := filepath.Glob(strings.TrimSuffix(w.path, ext) + "-*" + ext)
	if err != nil || len(matches) <= w.keep {
		return
	}
	// The names sort the way the dates do, so the oldest are at the front.
	slices.Sort(matches)
	for _, old := range matches[:len(matches)-w.keep] {
		_ = os.Remove(old)
	}
}
