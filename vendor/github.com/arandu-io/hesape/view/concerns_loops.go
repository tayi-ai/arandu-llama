package view

// These are methods on Factory. A loop stack tracks the state of nested
// loops so @foreach can expose $loop->index, $loop->first, $loop->last, etc.

// Loop holds the state of one level of a @foreach loop.
type Loop struct {
	Index     int
	Iteration int
	Remaining int
	Count     int
	First     bool
	Last      bool
	Odd       bool
	Even      bool
	Depth     int
	Parent    *Loop
}

// AddLoop pushes a new Loop onto the stack for a new iteration level.
func (f *Factory) AddLoop(count int) *Loop {
	f.mu.Lock()
	defer f.mu.Unlock()

	depth := 1
	if len(f.loopStack) > 0 {
		depth = len(f.loopStack) + 1
	}

	var parent *Loop
	if len(f.loopStack) > 0 {
		parent = f.loopStack[len(f.loopStack)-1]
	}

	loop := &Loop{
		Index:     0,
		Iteration: 1,
		Remaining: count,
		Count:     count,
		First:     true,
		Last:      count <= 1,
		Odd:       false,
		Even:      true,
		Depth:     depth,
		Parent:    parent,
	}

	f.loopStack = append(f.loopStack, loop)
	return loop
}

// IncrementLoopIndices advances the current loop's counters. Called at the
// top of every @foreach iteration.
func (f *Factory) IncrementLoopIndices() {
	f.mu.Lock()
	defer f.mu.Unlock()

	if len(f.loopStack) == 0 {
		return
	}

	loop := f.loopStack[len(f.loopStack)-1]
	loop.Iteration++
	loop.Index = loop.Iteration - 1
	loop.First = loop.Iteration == 1
	loop.Odd = !loop.Odd
	loop.Even = !loop.Even
	if loop.Remaining > 0 {
		loop.Remaining--
	}
	loop.Last = loop.Iteration >= loop.Count
}

// PopLoop removes the innermost loop from the stack. Called after @endforeach.
func (f *Factory) PopLoop() {
	f.mu.Lock()
	defer f.mu.Unlock()

	if len(f.loopStack) == 0 {
		return
	}
	f.loopStack = f.loopStack[:len(f.loopStack)-1]
}

// GetLastLoop returns the innermost Loop, or nil.
func (f *Factory) GetLastLoop() *Loop {
	f.mu.RLock()
	defer f.mu.RUnlock()

	if len(f.loopStack) == 0 {
		return nil
	}
	return f.loopStack[len(f.loopStack)-1]
}

// GetLoopStack returns the entire loop stack.
func (f *Factory) GetLoopStack() []*Loop {
	f.mu.RLock()
	defer f.mu.RUnlock()

	out := make([]*Loop, len(f.loopStack))
	copy(out, f.loopStack)
	return out
}
