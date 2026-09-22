package view

// Fragments let HTMX partial renders deliver only the changed portion of a
// page, without re-rendering the layout. @fragment name / @endfragment
// capture a named region; the controller requests only those that changed.

// StartFragment begins capturing a named fragment.
func (f *Factory) StartFragment(name string) {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.fragmentStack = append(f.fragmentStack, name)
}

// StopFragment closes the current fragment capture and stores its content.
func (f *Factory) StopFragment() string {
	f.mu.Lock()
	defer f.mu.Unlock()

	if len(f.fragmentStack) == 0 {
		return ""
	}

	last := len(f.fragmentStack) - 1
	name := f.fragmentStack[last]
	f.fragmentStack = f.fragmentStack[:last]

	f.fragments[name] = f.fragmentBuffer
	f.fragmentBuffer = ""
	return ""
}

// GetFragment returns a captured fragment's content.
func (f *Factory) GetFragment(name string) string {
	f.mu.RLock()
	defer f.mu.RUnlock()
	return f.fragments[name]
}

// GetFragments returns all captured fragments.
func (f *Factory) GetFragments() map[string]string {
	f.mu.RLock()
	defer f.mu.RUnlock()
	out := make(map[string]string, len(f.fragments))
	for k, v := range f.fragments {
		out[k] = v
	}
	return out
}

// FlushFragments resets every piece of per-render fragment state.
func (f *Factory) FlushFragments() {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.flushFragmentsLocked()
}

func (f *Factory) flushFragmentsLocked() {
	f.fragments = map[string]string{}
	f.fragmentStack = nil
	f.fragmentBuffer = ""
}

// SetFragmentBuffer sets the captured output for the current fragment block.
func (f *Factory) SetFragmentBuffer(content string) {
	f.mu.Lock()
	f.fragmentBuffer = content
	f.mu.Unlock()
}
