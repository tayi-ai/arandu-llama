package exception

import (
	"os"
	"runtime"
	"strings"
)

// StackFrame is one frame of the stack, already enriched with the source
// snippet around the failing line.
type StackFrame struct {
	Func    string
	File    string
	Line    int
	IsApp   bool     // false for the runtime, the stdlib and the collection itself
	Snippet []string // surrounding lines, rendered inline
	SnipTop int      // line number of the first snippet line
}

// collectionPrefix is this collection's own import path, collapsed by default:
// whoever is debugging wants to see THEIR code, not ours.
const collectionPrefix = "github.com/arandu-io/hesape"

// Capture reads the stack, folding the source snippet and whether the frame is
// the application's own into each entry.
//
// It collects the stack from skip onwards, marking which frames belong to the
// application. Same decision as Ignition: application frames are expanded by
// default and everything else is collapsed.
//
// appModule is the caller's module path. When empty, every frame that is not
// runtime, stdlib or part of this collection counts as application code.
func Capture(skip int, appModule string) []StackFrame {
	pcs := make([]uintptr, 64)
	n := runtime.Callers(skip, pcs)
	frames := runtime.CallersFrames(pcs[:n])

	var out []StackFrame
	for {
		f, more := frames.Next()

		sf := StackFrame{Func: f.Function, File: f.File, Line: f.Line}
		sf.IsApp = isApp(f.Function, appModule)
		if sf.IsApp {
			// Five lines each side of the failing one. It was six each side, thirteen
			// in all.
			sf.Snippet, sf.SnipTop = snippet(f.File, f.Line, 5)
		}
		out = append(out, sf)

		if !more {
			break
		}
	}
	return out
}

// isApp reports whether the frame is the application's own code.
//
// The test therefore never matched, every frame outside this collection was
// called application code, and the debug page expanded the whole standard
// library and read its source off disk.
//
// What says "this is the application" in Go is the module path, which is why
// Config.AppModule exists. A frame belongs to the application when its function
// name is inside that module, or when it is in package main -- the application's
// own entry package is called main once it is compiled, whatever its module is
// named.
//
// With no module path configured there is nothing left to tell the application
// from what it imported, so the fallback is the documented one: everything that
// is not this collection and not the standard library counts as the
// application's. It is generous on purpose -- a page that collapses the frame
// somebody is looking for is worse than one that expands a frame they are not.
func isApp(fn, appModule string) bool {
	switch {
	case strings.HasPrefix(fn, "main."):
		return true
	case appModule != "" && (strings.HasPrefix(fn, appModule+".") || strings.HasPrefix(fn, appModule+"/")):
		return true
	case strings.HasPrefix(fn, collectionPrefix):
		return false
	case isStdlib(fn):
		return false
	}
	return appModule == ""
}

// isStdlib reports whether the function belongs to the standard library.
//
// A Go function name begins with the import path of its package:
// "net/http.(*conn).serve", "runtime.gopanic", "github.com/acme/app/billing.Close".
// The first element of that path is a domain when the package was downloaded and
// a plain word when it came with the toolchain, which is the same rule the go
// command itself uses to decide what is a module path.
func isStdlib(fn string) bool {
	if first, _, hasSlash := strings.Cut(fn, "/"); hasSlash {
		return !strings.Contains(first, ".")
	}
	pkg, _, _ := strings.Cut(fn, ".")
	return pkg != "main"
}

// snippet reads the source around the line. When the binary runs far from its
// sources -- a production container -- it returns nothing and the page degrades
// instead of breaking.
func snippet(file string, line, radius int) ([]string, int) {
	b, err := os.ReadFile(file)
	if err != nil {
		return nil, 0
	}
	lines := strings.Split(string(b), "\n")
	from := max(0, line-radius-1)
	to := min(len(lines), line+radius)
	return lines[from:to], from + 1
}
