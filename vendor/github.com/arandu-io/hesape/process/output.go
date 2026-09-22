package process

import (
	"strconv"
	"strings"
	"sync"
)

// Stream names which of a process's two output streams a chunk came from.
type Stream string

const (
	// Out is standard output.
	Out Stream = "out"
	// Err is standard error.
	Err Stream = "err"
)

// OutputHandler is called with output while the command is still running.
//
// Calls are serialised, so the two streams never overlap and the function
// needs no lock of its own -- a handler that writes to a terminal from both at
// once would otherwise interleave a line into nonsense.
type OutputHandler func(stream Stream, buffer string)

// PoolOutputHandler is OutputHandler with the key of the process it came from.
//
// A process added without a key is keyed by its position, counted only among
// the unkeyed ones.
type PoolOutputHandler func(stream Stream, buffer string, key string)

// commandLine renders a command the way a person would type it.
//
// It is a rendering and not a quoting: an argument that needs quotes gets
// Go's, which no shell has to agree with, and nothing in this package hands a
// string to a shell.
func commandLine(command []string) string {
	parts := make([]string, 0, len(command))
	for _, word := range command {
		if word == "" || strings.ContainsAny(word, " \t\n\"'\\$`") {
			word = strconv.Quote(word)
		}
		parts = append(parts, word)
	}
	return strings.Join(parts, " ")
}

// buffer holds one stream of a running process and remembers how much of it has
// been read.
//
// The cursor is what os/exec has no notion of: LatestOutput is everything
// written since the last time anybody asked, which is how a progress display
// reads a build without printing it twice.
type buffer struct {
	mu      sync.Mutex
	written strings.Builder
	read    int
}

func (b *buffer) write(chunk []byte) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.written.Write(chunk)
}

// all is everything the stream has produced so far.
func (b *buffer) all() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.written.String()
}

// incremental is everything produced since the last call to incremental.
func (b *buffer) incremental() string {
	b.mu.Lock()
	defer b.mu.Unlock()
	text := b.written.String()
	if b.read >= len(text) {
		return ""
	}
	fresh := text[b.read:]
	b.read = len(text)
	return fresh
}
