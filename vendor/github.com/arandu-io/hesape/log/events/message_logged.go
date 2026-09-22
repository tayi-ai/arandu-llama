package events

import "log/slog"

// MessageLogged is the event one written log line fires.
//
// A Logger dispatches it once per written line, after the handler accepted the
// line, and never for a line the level filter dropped. It is what a profiler, a
// request timeline or a test listens to in order to see every message one
// request produced.
type MessageLogged struct {
	// Level is the level the line was written at.
	//
	// It is the slog level rather than its name, because that is the value the
	// handler was asked about, and a listener that compares against
	// log.LevelError should not have to parse a word to do it.
	Level slog.Level

	// Message is the log message, already formatted rather than the original
	// argument.
	Message string

	// Context is the fields of the line, with the logger's own accumulated
	// context already merged in.
	//
	// It is the map the logger wrote, not a copy shared with the next line, so
	// a listener may read it freely.
	Context map[string]any
}
