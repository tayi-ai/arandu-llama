package view

import "strings"

// These are methods on Factory. Stacks let a child view push content into a
// named pile that the layout drains at render time. @push adds to the end;
// @prepend adds to the beginning.

// StartPush begins capturing output for a named stack. If content is given,
// it extends the stack directly.
func (f *Factory) StartPush(name string, content string) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if content != "" {
		f.extendPushLocked(name, content)
		return
	}

	f.pushStack = append(f.pushStack, pushFrame{name: name, prepend: false})
}

// StartPrepend begins capturing output for prepending to a named stack.
func (f *Factory) StartPrepend(name string, content string) {
	f.mu.Lock()
	defer f.mu.Unlock()

	if content != "" {
		f.extendPrependLocked(name, content)
		return
	}

	f.pushStack = append(f.pushStack, pushFrame{name: name, prepend: true})
}

type pushFrame struct {
	name    string
	prepend bool
}

// StopPush closes the current push capture and stores its content.
func (f *Factory) StopPush() {
	f.mu.Lock()
	defer f.mu.Unlock()

	if len(f.pushStack) == 0 {
		return
	}

	last := len(f.pushStack) - 1
	frame := f.pushStack[last]
	f.pushStack = f.pushStack[:last]

	if frame.prepend {
		f.extendPrependLocked(frame.name, f.pushBuffer)
	} else {
		f.extendPushLocked(frame.name, f.pushBuffer)
	}
	f.pushBuffer = ""
}

// StopPrepend is an alias for StopPush.
func (f *Factory) StopPrepend() { f.StopPush() }

func (f *Factory) extendPushLocked(name string, content string) {
	f.pushes[name] = append(f.pushes[name], content)
}

func (f *Factory) extendPrependLocked(name string, content string) {
	f.prepends[name] = append(f.prepends[name], content)
}

// YieldPushContent drains a stack: prepends first (reversed, so the last
// prepend appears first), then pushes.
func (f *Factory) YieldPushContent(name string, fallback ...string) string {
	f.mu.RLock()
	defer f.mu.RUnlock()

	var b strings.Builder

	if pre, ok := f.prepends[name]; ok {
		for i := len(pre) - 1; i >= 0; i-- {
			b.WriteString(pre[i])
		}
	}
	if push, ok := f.pushes[name]; ok {
		for _, s := range push {
			b.WriteString(s)
		}
	}

	result := b.String()
	if result == "" && len(fallback) > 0 {
		return fallback[0]
	}
	return result
}

// FlushStacks resets every piece of per-render stack state.
func (f *Factory) FlushStacks() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.flushStacksLocked()
}

func (f *Factory) flushStacksLocked() {
	f.pushes = map[string][]string{}
	f.prepends = map[string][]string{}
	f.pushStack = nil
	f.pushBuffer = ""
}

// SetPushBuffer sets the captured output for the current push/prepend block.
func (f *Factory) SetPushBuffer(content string) {
	f.mu.Lock()
	f.pushBuffer = content
	f.mu.Unlock()
}
