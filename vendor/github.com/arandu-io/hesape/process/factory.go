package process

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"

	"github.com/arandu-io/hesape/str"
)

// FakeHandler is one command pattern and the answer Fake gives for it.
//
// Go has no untyped map that carries a pattern and a heterogeneous value with
// the order intact -- and order is load-bearing here, because fakeFor takes
// the first pattern that matches -- so it is a slice of this.
type FakeHandler struct {
	// Command is the pattern. Empty and "*" both match anything.
	Command string

	// Result is what the command answers with: a string, a []string, a
	// *FakeProcessResult, a *FakeProcessDescription, a *FakeProcessSequence, or
	// a func(*PendingProcess) any returning one of those.
	Result any
}

// callback normalises Result into the closure PendingProcess.fakeFor hands
// back.
func (h FakeHandler) callback() func(*PendingProcess) any {
	if fn, ok := h.Result.(func(*PendingProcess) any); ok {
		return fn
	}
	return func(*PendingProcess) any { return h.Result }
}

// ranProcess is one entry of the recording: the process as configured and the
// result it produced.
type ranProcess struct {
	process *PendingProcess
	result  ProcessResult
}

// Factory makes processes.
//
// It is the entry point of the component: Factory.Run for a command that runs,
// Factory.Fake for a test that must not shell out, and Factory.Pool for a set
// that runs at once.
//
// It is safe for concurrent use. Here a pool is goroutines, and the recording
// they all write to is behind a mutex.
type Factory struct {
	mu sync.Mutex

	fakeHandlers []FakeHandler
	recording    bool
	recorded     []ranProcess

	preventStray bool
}

// NewFactory constructs a Factory.
func NewFactory() *Factory { return &Factory{} }

// Fake registers fake answers and puts the factory in recording mode.
//
// It puts the factory in recording mode, so every process made from it answers
// from the handlers instead of running. Calling it with no handlers fakes
// everything with an empty successful result.
//
// Handlers are matched in order and the first match wins, so a "*" registered
// first answers everything after it.
func (f *Factory) Fake(handlers ...FakeHandler) *Factory {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.recording = true
	if len(handlers) == 0 {
		f.fakeHandlers = append(f.fakeHandlers, FakeHandler{Command: "*", Result: ""})
		return f
	}
	f.fakeHandlers = append(f.fakeHandlers, handlers...)
	return f
}

// IsRecording reports whether Fake has been called.
func (f *Factory) IsRecording() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.recording
}

// Record adds a process and the result it produced to the recording.
func (f *Factory) Record(process *PendingProcess, result ProcessResult) *Factory {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.recorded = append(f.recorded, ranProcess{process: process, result: result})
	return f
}

// RecordIfRecording records the process and its result when recording is on.
func (f *Factory) RecordIfRecording(process *PendingProcess, result ProcessResult) *Factory {
	if f.IsRecording() {
		f.Record(process, result)
	}
	return f
}

// PreventStrayProcesses makes a command no handler matched an error.
//
// With it on, a command that no handler matched is an error instead of running
// for real. It is the guard that turns "somebody forgot to fake git" from a
// test that quietly shells out on CI into a test that fails naming the command.
func (f *Factory) PreventStrayProcesses(prevent ...bool) *Factory {
	on := true
	if len(prevent) > 0 {
		on = prevent[0]
	}
	f.mu.Lock()
	defer f.mu.Unlock()
	f.preventStray = on
	return f
}

// PreventingStrayProcesses reports whether unmatched commands are refused.
func (f *Factory) PreventingStrayProcesses() bool {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.preventStray
}

// AllowStrayProcesses is the inverse of PreventStrayProcesses.
func (f *Factory) AllowStrayProcesses() *Factory { return f.PreventStrayProcesses(false) }

// NewPendingProcess builds a process bound to this factory, carrying the fake
// handlers registered so far.
func (f *Factory) NewPendingProcess() *PendingProcess {
	p := NewPendingProcess(f)
	f.mu.Lock()
	handlers := append([]FakeHandler(nil), f.fakeHandlers...)
	f.mu.Unlock()
	return p.WithFakeHandlers(handlers)
}

// Run builds a pending process and runs the command to completion.
func (f *Factory) Run(ctx context.Context, command []string, output OutputHandler) (ProcessResult, error) {
	return f.NewPendingProcess().Run(ctx, command, output)
}

// Start builds a pending process and starts the command without waiting.
func (f *Factory) Start(ctx context.Context, command []string, output OutputHandler) (InvokedProcess, error) {
	return f.NewPendingProcess().Start(ctx, command, output)
}

// Pool describes a set of processes that run at the same time.
//
// It builds the pool and runs nothing. [Pool.Start] starts the processes,
// [Pool.Run] and [Pool.Wait] start them and wait, and Concurrently is the one
// line that does both.
func (f *Factory) Pool(callback func(*Pool)) *Pool { return NewPool(f, callback) }

// Pipe runs processes one after another, each one reading what the one before
// it wrote.
//
// This takes the callback, which is the form the array form is written in: a
// caller with a list of commands ranges over it inside the callback.
func (f *Factory) Pipe(ctx context.Context, callback func(*Pipe), output PoolOutputHandler) (ProcessResult, error) {
	return NewPipe(f, callback).Run(ctx, output)
}

// Concurrently runs a pool of processes and waits for them to finish.
func (f *Factory) Concurrently(ctx context.Context, callback func(*Pool), output PoolOutputHandler) (*ProcessPoolResults, error) {
	started, err := f.Pool(callback).Start(ctx, output)
	if err != nil {
		return nil, err
	}
	return started.Wait()
}

// Result builds a fake result to register with Fake.
func (f *Factory) Result(output any, errorOutput any, exitCode int) *FakeProcessResult {
	return NewFakeProcessResult("", exitCode, output, errorOutput)
}

// Describe builds a fake description, for a process that is started rather than
// run and whose output arrives over time.
func (f *Factory) Describe() *FakeProcessDescription {
	return NewFakeProcessDescription()
}

// Sequence builds a series of results, one per call, for a command that is run
// more than once and answers differently each time.
func (f *Factory) Sequence(processes ...any) *FakeProcessSequence {
	return NewFakeProcessSequence(processes...)
}

// Recorded is the command line of every process that ran, in order.
//
// The command lines are what an assertion failure has to print, so this
// returns them rather than the pending processes, and the assertions below use
// it for their message.
//
// The line is PendingProcess.String, which is what a fake pattern is matched
// against -- for the reason matchesCommand is one function: rendering it any
// other way here would be a command a fake answered and an assertion then said
// never ran.
func (f *Factory) Recorded() []string {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]string, 0, len(f.recorded))
	for _, ran := range f.recorded {
		out = append(out, ran.process.String())
	}
	return out
}

// AssertRan fails the test when no recorded command matched the pattern.
func (f *Factory) AssertRan(t *testing.T, pattern string) {
	t.Helper()
	if n := f.countRan(pattern); n == 0 {
		t.Errorf("AssertRan(%q): no process matched it.\n%s", pattern, f.ranReport())
	}
}

// AssertRanTimes fails the test unless exactly times recorded commands matched
// the pattern.
func (f *Factory) AssertRanTimes(t *testing.T, pattern string, times int) {
	t.Helper()
	if n := f.countRan(pattern); n != times {
		t.Errorf("AssertRanTimes(%q, %d): %d matched.\n%s", pattern, times, n, f.ranReport())
	}
}

// AssertNotRan fails the test when a recorded command matched the pattern.
func (f *Factory) AssertNotRan(t *testing.T, pattern string) {
	t.Helper()
	if n := f.countRan(pattern); n != 0 {
		t.Errorf("AssertNotRan(%q): %d matched it.\n%s", pattern, n, f.ranReport())
	}
}

// AssertDidntRun is AssertNotRan under its other name.
func (f *Factory) AssertDidntRun(t *testing.T, pattern string) {
	t.Helper()
	f.AssertNotRan(t, pattern)
}

// AssertNothingRan fails the test when any process ran at all.
func (f *Factory) AssertNothingRan(t *testing.T) {
	t.Helper()
	if ran := f.Recorded(); len(ran) > 0 {
		t.Errorf("AssertNothingRan: %d processes ran.\n%s", len(ran), f.ranReport())
	}
}

// countRan reports how many recorded commands match the pattern, with the same
// matcher fakeFor uses so that a pattern asserted on and a pattern faked on
// behave the same.
func (f *Factory) countRan(pattern string) int {
	n := 0
	for _, command := range f.Recorded() {
		if matchesCommand(pattern, command) {
			n++
		}
	}
	return n
}

// matchesCommand reports whether a fake or assertion pattern answers a command
// line, and is the single implementation of that question.
//
// Two implementations would be a fake that answers a command an assertion then
// says never ran.
//
// An empty pattern and "*" both match anything, which is what an entry with no
// Command means.
func matchesCommand(pattern, command string) bool {
	if pattern == "" || pattern == "*" {
		return true
	}
	return str.Is([]string{pattern}, command, false)
}

// ranReport lists the commands that ran, for an assertion failure.
func (f *Factory) ranReport() string {
	ran := f.Recorded()
	if len(ran) == 0 {
		return "no process ran"
	}
	var b strings.Builder
	b.WriteString(fmt.Sprintf("%d ran:", len(ran)))
	for _, command := range ran {
		b.WriteString("\n  " + command)
	}
	return b.String()
}
