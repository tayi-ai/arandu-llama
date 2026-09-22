package translation

import (
	"embed"
	"io/fs"
	"maps"
)

// The English catalogue that ships with the framework: the sentences of
// validation, auth, passwords and pagination, which a framework produces before
// an application has written a line of its own.
//
// They are embedded rather than published into the project: a copy in every
// project is a copy that drifts from the rule that produces it, and a rule that
// gains an argument leaves the published sentence saying the old thing. An
// application overrides one of them by defining the same key in its own
// catalogue, which is one file rather than all four.
//
//go:embed lang
var langFS embed.FS

// bundled is those lines, read once at startup. Every [Translator] consults it
// after the application catalogue and after the fallback locale.
var bundled = mustBundled()

func mustBundled() Loader {
	root, err := fs.Sub(langFS, "lang")
	if err != nil {
		panic("translation: the embedded catalogue has no lang directory: " + err.Error())
	}
	l, err := NewFileLoader(root, ".")
	if err != nil {
		panic("translation: the embedded catalogue does not parse: " + err.Error())
	}
	return l
}

// Bundled returns the lines the framework's own catalogue carries for one group
// of one locale, with nested objects flattened to dotted items, and nil for a
// group it does not carry.
//
// It is for a package that produces one of these sentences without holding a
// [Translator] -- the validator answering with no catalogue configured, before
// an application has wired one. That package reads the sentences from here
// rather than keeping its own copy of them: two tables of the same English is
// two answers to what a message says, and which one a project reads is decided
// by which of them happened to be loaded.
//
// The catalogue that ships is English, so "en" is the locale that answers.
//
// The map is a copy. The catalogue is read once and shared by every
// [Translator], and a caller able to write into it would be editing the
// sentences of all of them.
func Bundled(locale, group string) Lines {
	lines := bundled.Load(locale, group, AppNamespace)
	if len(lines) == 0 {
		return nil
	}
	return maps.Clone(lines)
}
