package log

import (
	"context"
	"log/slog"
	"sync"
	"time"
)

// Record is one captured log line.
//
// It is not related to Recorder, which is the ring of finished HTTP requests
// behind the console. This is a single line, and it exists so a test can assert
// on what was logged.
type Record struct {
	Time    time.Time
	Level   slog.Level
	Message string
	// Attrs is the line's fields, flattened. A group becomes a dotted prefix on
	// the keys inside it, which is what a test wants to write: "user.id", not a
	// walk over nested values.
	Attrs map[string]any
}

// Records holds what a captured logger wrote. Every method is safe to call
// while the logger is still being written to from another goroutine.
type Records struct {
	mu      sync.Mutex
	records []Record
}

func (r *Records) add(rec Record) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.records = append(r.records, rec)
}

// All returns a copy of the captured lines, oldest first.
func (r *Records) All() []Record {
	r.mu.Lock()
	defer r.mu.Unlock()
	return append([]Record(nil), r.records...)
}

// Len is how many lines were captured.
func (r *Records) Len() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return len(r.records)
}

// Capture returns a logger that writes into memory, and the memory.
//
// It is what a test installs with Into when the assertion is about a log line:
// that the throttle said why it refused, that the relay reported the batch it
// gave up on. Reading stdout to find that out means parsing text, and asserting
// on formatted text pins the test to the handler rather than to the behaviour.
//
// It captures every level, including debug: a test that has to lower the level
// first is a test that fails for the wrong reason.
func Capture() (*slog.Logger, *Records) {
	records := &Records{}
	return slog.New(&captureHandler{records: records}), records
}

// captured is one attribute with the group prefix it was written under, kept
// unflattened until Handle so a group added after the attribute still applies.
type captured struct {
	prefix string
	attr   slog.Attr
}

type captureHandler struct {
	records *Records
	attrs   []captured
	prefix  string
}

func (h *captureHandler) Enabled(context.Context, slog.Level) bool { return true }

func (h *captureHandler) Handle(_ context.Context, r slog.Record) error {
	rec := Record{Time: r.Time, Level: r.Level, Message: r.Message, Attrs: map[string]any{}}
	for _, c := range h.attrs {
		flatten(rec.Attrs, c.prefix, c.attr)
	}
	r.Attrs(func(a slog.Attr) bool {
		flatten(rec.Attrs, h.prefix, a)
		return true
	})
	h.records.add(rec)
	return nil
}

func (h *captureHandler) WithAttrs(as []slog.Attr) slog.Handler {
	if len(as) == 0 {
		return h
	}
	// A fresh slice rather than append onto h.attrs: two loggers derived from
	// the same parent would otherwise write into one backing array, and the
	// second would overwrite the first's fields.
	next := make([]captured, 0, len(h.attrs)+len(as))
	next = append(next, h.attrs...)
	for _, a := range as {
		next = append(next, captured{prefix: h.prefix, attr: a})
	}
	return &captureHandler{records: h.records, attrs: next, prefix: h.prefix}
}

func (h *captureHandler) WithGroup(name string) slog.Handler {
	if name == "" {
		return h
	}
	return &captureHandler{records: h.records, attrs: h.attrs, prefix: h.prefix + name + "."}
}

// flatten writes one attribute into the map, opening groups into dotted keys.
func flatten(dst map[string]any, prefix string, a slog.Attr) {
	a.Value = a.Value.Resolve()
	if a.Value.Kind() == slog.KindGroup {
		inner := prefix
		if a.Key != "" {
			inner = prefix + a.Key + "."
		}
		for _, g := range a.Value.Group() {
			flatten(dst, inner, g)
		}
		return
	}
	if a.Key == "" {
		return
	}
	dst[prefix+a.Key] = a.Value.Any()
}
