package context

import (
	"context"
	"log/slog"
)

// ContextLogProcessor is the handler that copies the context repository's
// visible entries onto every record it writes, which is what makes a value added
// once, at the edge, show up in every log line of the request without anybody
// passing it down.
//
// It is a slog.Handler wrapping another one rather than a func, because slog has
// no processor hook -- a wrapping handler is where slog puts "run before the
// next one sees the record".
//
// Hidden entries are not copied. That is the whole point of hidden: it travels
// with the request and into a queued job, and it does not land in the log.
type ContextLogProcessor struct {
	next slog.Handler
	// claimed is the set of names already spoken on this handler by WithAttrs,
	// in the namespace the last WithGroup opened. It stands in for walking a
	// record that never sees those attributes. See
	// [ContextLogProcessor.Handle].
	claimed map[string]struct{}
}

// NewContextLogProcessor wraps a handler so that the context repository's
// visible entries reach every record it writes.
func NewContextLogProcessor(next slog.Handler) *ContextLogProcessor {
	return &ContextLogProcessor{next: next}
}

// Enabled reports whether the wrapped handler wants this level. It is asked
// before the attributes are gathered, so a line below the level costs nothing.
func (p *ContextLogProcessor) Enabled(ctx context.Context, level slog.Level) bool {
	return p.next.Enabled(ctx, level)
}

// Handle copies the visible context onto the record and passes it on.
//
// The context repository is read from the ctx, which is where a request-scoped
// value lives.
//
// slog has one flat set of attributes, so a repeated name is a real collision.
// An entry whose name a caller already used is not overwritten: what the caller
// said about this one line beats what the request said about all of them.
//
// # The two ways a caller speaks a name
//
// There are two, and only one of them used to be looked at. A name passed to the
// log call is on the record; a name passed to .With is on the handler, and never
// reaches the record at all. So a logger derived as .With("order_id", "B"),
// logging under a repository holding order_id C, produced
// {"order_id":"C","order_id":"B"} -- a duplicate key, which JSON allows to be
// written and no two parsers agree on. The same collision on the record resolved
// the other way and dropped the repository's value, so the two spellings of the
// same rule disagreed with each other. claimed is what closes it: WithAttrs
// records the names it was handed, and both spellings now leave the caller's
// value standing.
//
// WithGroup opens a namespace, and everything after it -- the derived
// attributes, the record's own, and the ones added here -- is nested inside it
// together. So the claimed set starts empty again there, and a name claimed
// outside the group does not stop the repository reaching into it.
func (p *ContextLogProcessor) Handle(ctx context.Context, record slog.Record) error {
	repository := For(ctx)
	if repository == nil {
		return p.next.Handle(ctx, record)
	}
	visible := repository.All()
	if len(visible) == 0 {
		return p.next.Handle(ctx, record)
	}

	spoken := make(map[string]bool, record.NumAttrs()+len(p.claimed))
	for key := range p.claimed {
		spoken[key] = true
	}
	record.Attrs(func(attr slog.Attr) bool {
		spoken[attr.Key] = true
		return true
	})

	for key, value := range visible {
		if !spoken[key] {
			record.AddAttrs(slog.Any(key, value))
		}
	}
	return p.next.Handle(ctx, record)
}

// WithAttrs answers slog.Handler.WithAttrs, keeping the wrapper in place so
// that a logger derived with .With still carries the context.
//
// It remembers the names it was handed, because they are the ones
// [ContextLogProcessor.Handle] can no longer find by walking the record.
func (p *ContextLogProcessor) WithAttrs(attrs []slog.Attr) slog.Handler {
	claimed := make(map[string]struct{}, len(p.claimed)+len(attrs))
	for key := range p.claimed {
		claimed[key] = struct{}{}
	}
	for _, attr := range attrs {
		claimed[attr.Key] = struct{}{}
	}
	return &ContextLogProcessor{next: p.next.WithAttrs(attrs), claimed: claimed}
}

// WithGroup answers slog.Handler.WithGroup, keeping the wrapper in place for
// the same reason.
//
// The claimed set does not travel through it: a group is a namespace, and a name
// claimed outside it is not the name it holds.
func (p *ContextLogProcessor) WithGroup(name string) slog.Handler {
	return &ContextLogProcessor{next: p.next.WithGroup(name)}
}
