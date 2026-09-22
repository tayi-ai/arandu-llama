package view

import (
	"regexp"
	"strings"
)

// ComponentSlot is what a component receives between its opening and closing
// tag: the content, plus the attributes written on the slot itself.
type ComponentSlot struct {
	// Attributes is the slot's own attribute bag.
	Attributes *ComponentAttributeBag

	contents string
}

// NewComponentSlot returns a slot with the given contents and attributes.
func NewComponentSlot(contents string, attributes map[string]any) *ComponentSlot {
	slot := &ComponentSlot{contents: contents}
	slot.WithAttributes(attributes)
	return slot
}

// WithAttributes replaces the slot's attribute bag, and returns s for
// chaining.
func (s *ComponentSlot) WithAttributes(attributes map[string]any) *ComponentSlot {
	s.Attributes = NewComponentAttributeBag(attributes)
	return s
}

// ToHTML returns the slot's contents.
func (s *ComponentSlot) ToHTML() string { return s.contents }

// IsEmpty reports whether the slot's contents are the empty string.
func (s *ComponentSlot) IsEmpty() bool { return s.contents == "" }

// IsNotEmpty reports the opposite of IsEmpty.
func (s *ComponentSlot) IsNotEmpty() bool { return !s.IsEmpty() }

// commentPattern matches an HTML comment, including one that spans lines.
var commentPattern = regexp.MustCompile(`(?s)<!--.*?-->`)

// HasActualContent reports whether the slot has content beyond HTML comments
// and whitespace, for deciding whether to draw the wrapper around it.
//
// filter is variadic so it can be omitted entirely; the default strips
// comments and trims.
func (s *ComponentSlot) HasActualContent(filter ...func(string) string) bool {
	strip := func(input string) string {
		return strings.TrimSpace(commentPattern.ReplaceAllString(input, ""))
	}
	if len(filter) > 0 && filter[0] != nil {
		strip = filter[0]
	}
	return strip(s.contents) != ""
}

// String is an alias for ToHTML.
func (s *ComponentSlot) String() string { return s.ToHTML() }
