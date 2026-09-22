package view

import (
	"crypto/sha256"
	"fmt"
	"io"
	"strings"
	"sync"
)

// These are methods on Factory. A view extending a layout is rendered in two
// passes: the child is evaluated to fill sections, then the layout renders
// with the sections inserted.

// StartSection opens a section for capture. If content is given, it extends
// an existing section directly.
func (f *Factory) StartSection(name string, content string) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if content != "" {
		f.extendSectionLocked(name, content)
		return
	}

	f.lastSection = name
	f.sectionStack = append(f.sectionStack, name)
}

// Inject section content directly (alias for StartSection with content).
func (f *Factory) Inject(name string, content string) {
	f.StartSection(name, content)
}

// StopSection closes the section being captured and stores its content.
// When overwrite is true, the previous content is replaced rather than
// appended.
func (f *Factory) StopSection(overwrite bool) string {
	f.mu.Lock()
	defer f.mu.Unlock()

	if len(f.sectionStack) == 0 {
		return ""
	}

	last := len(f.sectionStack) - 1
	name := f.sectionStack[last]
	f.sectionStack = f.sectionStack[:last]

	if overwrite {
		f.sections[name] = f.sectionBuffer
		f.sectionBuffer = ""
		return ""
	}

	existing := f.sections[name]
	if existing != "" {
		// @parent placeholder replacement: the child can say
		// @parent to insert the layout's existing content.
		placeholder := f.parentPlaceholderLocked(name)
		f.sections[name] = strings.Replace(existing, placeholder, f.sectionBuffer, 1)
	} else {
		f.sections[name] = f.sectionBuffer
	}
	f.sectionBuffer = ""
	return ""
}

// AppendSection appends to a section's content.
func (f *Factory) AppendSection() {
	f.mu.Lock()
	defer f.mu.Unlock()

	if len(f.sectionStack) == 0 {
		return
	}

	last := len(f.sectionStack) - 1
	name := f.sectionStack[last]
	f.sectionStack = f.sectionStack[:last]

	f.sections[name] += f.sectionBuffer
	f.sectionBuffer = ""
}

// ExtendSection replaces @parent in the existing section content with the
// new content.
func (f *Factory) ExtendSection(name string, content string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.extendSectionLocked(name, content)
}

func (f *Factory) extendSectionLocked(name string, content string) {
	existing := f.sections[name]
	if existing == "" {
		f.sections[name] = content
		return
	}
	placeholder := f.parentPlaceholderLocked(name)
	f.sections[name] = strings.Replace(existing, placeholder, content, 1)
}

// YieldContent returns a section's content or a default.
//
//	YieldContent("sidebar", "<!-- no sidebar -->")
func (f *Factory) YieldContent(name string, fallback ...string) string {
	// A write lock, not a read one: the placeholder is memoised on first ask.
	f.mu.Lock()
	defer f.mu.Unlock()

	content, ok := f.sections[name]
	if !ok || content == "" {
		if len(fallback) > 0 {
			return fallback[0]
		}
		return ""
	}

	// Strip parent placeholders for this section before returning.
	placeholder := f.parentPlaceholderLocked(name)
	content = strings.ReplaceAll(content, placeholder, "")

	return content
}

// YieldSection returns the content of the most recently started section.
func (f *Factory) YieldSection() string {
	f.mu.RLock()
	last := f.lastSection
	f.mu.RUnlock()
	return f.YieldContent(last)
}

// HasSection reports whether a section was defined and is non-empty.
func (f *Factory) HasSection(name string) bool {
	f.mu.RLock()
	defer f.mu.RUnlock()
	content, ok := f.sections[name]
	return ok && strings.TrimSpace(content) != ""
}

// SectionMissing is the inverse of HasSection.
func (f *Factory) SectionMissing(name string) bool { return !f.HasSection(name) }

// GetSection returns a section's raw content.
func (f *Factory) GetSection(name string) (string, bool) {
	f.mu.RLock()
	defer f.mu.RUnlock()
	content, ok := f.sections[name]
	return content, ok
}

// GetSections returns all section contents.
func (f *Factory) GetSections() map[string]string {
	f.mu.RLock()
	defer f.mu.RUnlock()
	out := make(map[string]string, len(f.sections))
	for k, v := range f.sections {
		out[k] = v
	}
	return out
}

// FlushSections resets every piece of per-render section state.
func (f *Factory) FlushSections() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.flushSectionsLocked()
}

func (f *Factory) flushSectionsLocked() {
	f.sections = map[string]string{}
	f.sectionStack = nil
	f.lastSection = ""
	f.sectionBuffer = ""
}

// SetSectionBuffer sets the captured output buffer for a section. Called by
// the compiler-generated code after a section block.
func (f *Factory) SetSectionBuffer(content string) {
	f.mu.Lock()
	f.sectionBuffer = content
	f.mu.Unlock()
}

// ParentPlaceholder returns the marker @parent compiles to.
//
// It has to be the same string every time it is asked for within one
// process, because the section that writes it and the section that replaces
// it are two different calls. It used to be generated fresh from
// crypto/rand on every call, which meant @parent never matched and the
// layout's own content was silently dropped.
//
// The salt is per Factory rather than per class, so two factories in one test
// binary do not share markers.
func (f *Factory) ParentPlaceholder(section string) string {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.parentPlaceholderLocked(section)
}

func (f *Factory) parentPlaceholderLocked(section string) string {
	if f.parentPlaceholders == nil {
		f.parentPlaceholders = map[string]string{}
	}
	if marker, known := f.parentPlaceholders[section]; known {
		return marker
	}
	sum := sha256.Sum256([]byte(f.parentSalt + section))
	marker := fmt.Sprintf("##parent-placeholder-%x##", sum[:8])
	f.parentPlaceholders[section] = marker
	return marker
}

// renderSection writes a section directly to an io.Writer.
//
// This is the runtime counterpart of YieldContent, for use in compiled
// layout functions that write directly to the response.
func (f *Factory) renderSection(w io.Writer, name string) error {
	content := f.YieldContent(name)
	if content == "" {
		return nil
	}
	_, err := io.WriteString(w, content)
	return err
}

// captureOutput is the running section output buffer, fed by the
// compiler-generated code between @section and @stop.
type sectionCapture struct {
	builder strings.Builder
	mu      sync.Mutex
}

func (c *sectionCapture) Write(p []byte) (int, error) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.builder.Write(p)
}

func (c *sectionCapture) String() string {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.builder.String()
}

func (c *sectionCapture) Reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.builder.Reset()
}
