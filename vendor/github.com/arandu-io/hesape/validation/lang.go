package validation

import (
	"maps"
	"slices"
	"strings"

	"github.com/arandu-io/hesape/translation"
)

// This file is the smallest Translator a Validator can fall back to, and it
// reads the English validation lines out of the catalogue that ships with the
// framework.
//
// It used to carry a copy of those lines. Two tables of the same English is two
// answers to what a message says, nothing held them equal, and a rule whose
// sentence was edited in one of them would read one way for a project that had
// wired a catalogue and another way for a project that had not.

// englishLines is the English sentence of every rule, in the shape this file
// looks lines up in: a group -- "min", "between", "size" -- is a map of its four
// size shapes, and every other rule is a sentence.
//
// The three groups that ship empty are absent, as they are from the catalogue:
// custom, attributes and values are what an application fills.
var englishLines = groupedLines(translation.Bundled("en", "validation"))

// groupedLines rebuilds the nesting of a catalogue group out of the flat lines a
// loader answers with, where the four size shapes of a rule arrive as
// "min.string" rather than under "min".
//
// A dotted item whose head is also a sentence on its own is dropped rather than
// silently replacing it: no rule in the catalogue is both, and one that became
// both would be a catalogue to fix, not a shape to guess at.
func groupedLines(lines translation.Lines) map[string]any {
	out := make(map[string]any, len(lines))
	for _, item := range slices.Sorted(maps.Keys(lines)) {
		head, tail, nested := strings.Cut(item, ".")
		if !nested {
			if _, taken := out[head]; !taken {
				out[head] = lines[item]
			}
			continue
		}
		group, isGroup := out[head].(map[string]string)
		if !isGroup {
			if _, taken := out[head]; taken {
				continue
			}
			group = map[string]string{}
			out[head] = group
		}
		group[tail] = lines[item]
	}
	return out
}

// englishTranslator is the Translator a Validator falls back to inside
// GetDisplayableValue and the custom-message lookups: it answers out of
// englishLines and returns the key for anything it does not hold, which is the
// contract every check in formats.go is written against.
//
// A Validator built with no translator does not read the LINES -- getMessage
// asks v.trans directly and stops at the compiled rule set's own sentence. This
// is what answers the group lookups either way.
type englishTranslator struct{}

// Get reads a line out of englishLines. The key is dot notation under
// "validation.", so "validation.min.string" reads the "string" line of the "min"
// group.
func (englishTranslator) Get(key string, replace map[string]any, locale string) any {
	rest, under := strings.CutPrefix(key, "validation.")
	if !under {
		return interpolate(key, replace)
	}
	head, tail, nested := strings.Cut(rest, ".")
	entry, held := englishLines[head]
	if !held {
		return key
	}
	if !nested {
		return interpolate(entry, replace)
	}
	group, isGroup := entry.(map[string]string)
	if !isGroup {
		return key
	}
	sentence, held := group[tail]
	if !held {
		return key
	}
	return interpolate(sentence, replace)
}

// Choice reads a line for a count. englishLines holds no line with a plural
// segment, so this is Get with :count filled in.
func (t englishTranslator) Choice(key string, number int, replace map[string]any, locale string) string {
	if replace == nil {
		replace = map[string]any{}
	}
	if _, given := replace["count"]; !given {
		replace["count"] = number
	}
	return line(t.Get(key, replace, locale), key)
}

// interpolate fills the :placeholders of a line before it is returned.
func interpolate(entry any, replace map[string]any) any {
	sentence, isSentence := entry.(string)
	if !isSentence || len(replace) == 0 {
		return entry
	}
	for _, name := range slices.Sorted(maps.Keys(replace)) {
		sentence = strings.ReplaceAll(sentence, ":"+name, stringOf(replace[name]))
	}
	return sentence
}
