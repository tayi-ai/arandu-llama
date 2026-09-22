package process

import (
	"bytes"
	"context"
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sort"
	"strings"
	"syscall"
	"time"
)

// defaultTimeout is how long a command may run when nobody said.
const defaultTimeout = 60 * time.Second

// PendingProcess is a command being described, before it runs.
//
// It is a fluent builder: every setter returns the receiver, so calls chain.
//
//	result, err := factory.NewPendingProcess().
//		Path("/srv/app").
//		Timeout(2 * time.Minute).
//		Run(ctx, []string{"go", "test", "./..."}, nil)
//
// Go cannot have a field and a method of the same name, so the state is
// unexported and only the setters are public; String is how a fake handler or
// an assertion reads back the command line.
type PendingProcess struct {
	factory      *Factory
	command      []string
	path         string
	timeout      time.Duration
	idleTimeout  time.Duration
	environment  map[string]string
	input        any
	quietly      bool
	tty          bool
	options      *syscall.SysProcAttr
	fakeHandlers []FakeHandler
}

// NewPendingProcess builds a pending process bound to a factory.
func NewPendingProcess(factory *Factory) *PendingProcess {
	return &PendingProcess{factory: factory, timeout: defaultTimeout}
}

// String is the command line, rendered the way a person would type it.
func (p *PendingProcess) String() string { return commandLine(p.command) }

// Command sets the program and its arguments.
//
// This takes only the array form, and there is no string form to add: see the
// package comment.
func (p *PendingProcess) Command(command ...string) *PendingProcess {
	p.command = command
	return p
}

// Path sets the working directory.
func (p *PendingProcess) Path(path string) *PendingProcess {
	p.path = path
	return p
}

// Timeout sets how long the whole run may take.
//
// Zero is no limit, which is what Forever sets.
func (p *PendingProcess) Timeout(timeout time.Duration) *PendingProcess {
	p.timeout = timeout
	return p
}

// IdleTimeout sets how long the program may go without writing anything.
//
// It is the bound that catches what a total timeout does not: a fetch whose
// peer stopped answering, or a compiler waiting on a lock, both of which sit
// quiet for as long as they are given. A program that reports progress is never
// affected by it.
func (p *PendingProcess) IdleTimeout(timeout time.Duration) *PendingProcess {
	p.idleTimeout = timeout
	return p
}

// Forever removes the time limit.
func (p *PendingProcess) Forever() *PendingProcess {
	p.timeout = 0
	return p
}

// Env sets environment variables for the program.
//
// They are added to the ones this process already has, not substituted for
// them. A name already set is overridden by the value here.
func (p *PendingProcess) Env(environment map[string]string) *PendingProcess {
	p.environment = environment
	return p
}

// Input sets what is written to the program's standard input.
//
// Standard input is closed once it has been written. Without an Input the
// program reads an immediately closed input, never the terminal -- a command
// that waits for a person is a command that hangs a deploy.
func (p *PendingProcess) Input(input any) *PendingProcess {
	p.input = input
	return p
}

// Quietly discards the program's output instead of keeping it.
//
// Output and ErrorOutput then answer empty rather than throwing.
func (p *PendingProcess) Quietly() *PendingProcess {
	p.quietly = true
	return p
}

// Tty hands the program this process's own terminal.
//
// Output is not captured in this mode -- it goes straight to the terminal --
// and the run fails if there is no terminal to hand over, which SupportsTty
// answers in advance.
func (p *PendingProcess) Tty(tty ...bool) *PendingProcess {
	p.tty = len(tty) == 0 || tty[0]
	return p
}

// SupportsTty reports whether this machine has a terminal to hand over, by
// asking whether /dev/tty can be opened.
func (p *PendingProcess) SupportsTty() bool {
	terminal, err := os.OpenFile("/dev/tty", os.O_RDWR, 0)
	if err != nil {
		return false
	}
	_ = terminal.Close()
	return true
}

// Options sets the attributes the operating system is handed when the process
// is created.
//
// What is in it differs by platform, which is true on both sides.
func (p *PendingProcess) Options(options *syscall.SysProcAttr) *PendingProcess {
	p.options = options
	return p
}

// WithFakeHandlers gives the pending process the fakes it should answer with.
func (p *PendingProcess) WithFakeHandlers(fakeHandlers []FakeHandler) *PendingProcess {
	p.fakeHandlers = fakeHandlers
	return p
}

// Run runs the command and waits for it.
//
// The command may be given here or with Command; nil here keeps the one
// already set.
//
// A command that exits non-zero is not an error: the result comes back with a
// nil error and Failed reports it. The error is for the program that could not
// start, ran out of time, or was stray.
//
// Cancelling it kills the program.
func (p *PendingProcess) Run(ctx context.Context, command []string, output OutputHandler) (ProcessResult, error) {
	if len(command) > 0 {
		p.command = command
	}
	line := p.String()

	if fake := p.fakeFor(line); fake != nil {
		result, err := p.resolveSynchronousFake(line, fake)
		if err != nil {
			return nil, err
		}
		emitToHandler(output, result)
		p.factory.RecordIfRecording(p, result)
		return result, nil
	}
	if p.factory.IsRecording() && p.factory.PreventingStrayProcesses() {
		return nil, &StrayProcessError{Command: line}
	}

	invoked, err := p.start(ctx, output)
	if err != nil {
		return nil, err
	}
	return invoked.Wait(nil)
}

// Start starts the command and returns while it is still running.
//
// The caller must call Wait on what comes back, or the program is never reaped.
func (p *PendingProcess) Start(ctx context.Context, command []string, output OutputHandler) (InvokedProcess, error) {
	if len(command) > 0 {
		p.command = command
	}
	line := p.String()

	if fake := p.fakeFor(line); fake != nil {
		invoked, err := p.resolveAsynchronousFake(line, output, fake)
		if err != nil {
			return nil, err
		}
		p.factory.RecordIfRecording(p, invoked.PredictProcessResult())
		return invoked, nil
	}
	if p.factory.IsRecording() && p.factory.PreventingStrayProcesses() {
		return nil, &StrayProcessError{Command: line}
	}

	return p.start(ctx, output)
}

// start is the real thing: no fake matched, so a program is launched.
func (p *PendingProcess) start(ctx context.Context, output OutputHandler) (*invokedProcess, error) {
	if ctx == nil {
		return nil, errors.New("process: a nil context")
	}
	if len(p.command) == 0 || strings.TrimSpace(p.command[0]) == "" {
		return nil, errors.New("process: a command with no name")
	}
	environment, err := p.environ()
	if err != nil {
		return nil, err
	}
	stdin, err := p.stdin()
	if err != nil {
		return nil, err
	}
	if p.tty && !p.SupportsTty() {
		return nil, errors.New("process: TTY mode requires a terminal, and there is none")
	}

	// One cancellable context serves both limits: the deadline fires on its own
	// and the idle watchdog cancels by hand. Which one it was is decided in
	// classify, from the flag and from the cause -- an exit status cannot tell
	// them apart, because a killed process looks the same either way.
	run, cancel := ctx, context.CancelFunc(nil)
	switch {
	case p.timeout > 0:
		run, cancel = context.WithTimeout(ctx, p.timeout)
	case p.idleTimeout > 0:
		run, cancel = context.WithCancel(ctx)
	}

	shared := &sink{handler: output}
	if p.idleTimeout > 0 {
		shared.bump = make(chan struct{}, 1)
	}

	cmd := exec.CommandContext(run, p.command[0], p.command[1:]...)
	cmd.Dir = p.path
	cmd.Env = environment
	cmd.SysProcAttr = p.options
	cmd.WaitDelay = waitDelay

	invoked := &invokedProcess{
		command:     commandLine(p.command),
		cmd:         cmd,
		stdout:      &buffer{},
		stderr:      &buffer{},
		sink:        shared,
		timeout:     p.timeout,
		idleTimeout: p.idleTimeout,
		quietly:     p.quietly,
		parent:      ctx,
		cancel:      cancel,
		run:         run,
		watching:    make(chan struct{}),
		settled:     make(chan struct{}),
	}

	if p.tty {
		// TTY mode hands the child this process's own descriptors, so there is
		// nothing left to capture and nothing to hand a handler.
		cmd.Stdin, cmd.Stdout, cmd.Stderr = os.Stdin, os.Stdout, os.Stderr
	} else {
		cmd.Stdin = stdin
		cmd.Stdout = &collector{stream: Out, buf: invoked.stdout, sink: shared, quietly: p.quietly}
		cmd.Stderr = &collector{stream: Err, buf: invoked.stderr, sink: shared, quietly: p.quietly}
	}

	invoked.started = time.Now()
	if err := cmd.Start(); err != nil {
		if cancel != nil {
			cancel()
		}
		if errors.Is(err, exec.ErrNotFound) {
			return nil, fmt.Errorf("process: %q was not found in PATH: %w", p.command[0], err)
		}
		return nil, fmt.Errorf("process: starting [%s]: %w", invoked.command, err)
	}

	if p.idleTimeout > 0 {
		go invoked.watchIdle(p.idleTimeout, shared.bump)
	}
	// Reaped by this goroutine and not by Wait.
	go func() {
		waitErr := cmd.Wait()
		close(invoked.watching)
		if cancel != nil {
			cancel()
		}
		invoked.waitErr = waitErr
		close(invoked.settled)
	}()
	return invoked, nil
}

// environ is the environment the program is started with, or nil to inherit
// this one's unchanged.
//
// The additions go after os.Environ, which is what makes them win: os/exec
// keeps the last value for a repeated name. Sorted, so two runs of the same
// command hand the program the same slice and a test can say what it expects.
func (p *PendingProcess) environ() ([]string, error) {
	if len(p.environment) == 0 {
		return nil, nil
	}
	names := make([]string, 0, len(p.environment))
	for name := range p.environment {
		if name == "" || strings.ContainsAny(name, "=\x00") {
			return nil, fmt.Errorf("process: %q is not an environment variable name", name)
		}
		names = append(names, name)
	}
	sort.Strings(names)

	environment := os.Environ()
	for _, name := range names {
		environment = append(environment, name+"="+p.environment[name])
	}
	return environment, nil
}

// stdin is what the program reads, and never the terminal: exec leaves a nil
// Stdin connected to the null device, and a command that could read from the
// console is a command that can wait for somebody who is not there.
func (p *PendingProcess) stdin() (io.Reader, error) {
	switch input := p.input.(type) {
	case nil:
		return strings.NewReader(""), nil
	case string:
		return strings.NewReader(input), nil
	case []byte:
		return bytes.NewReader(input), nil
	case io.Reader:
		return input, nil
	default:
		return nil, fmt.Errorf("process: input of type %T is not a string, a []byte or an io.Reader", p.input)
	}
}

// emitToHandler hands a faked result's two streams to the output handler.
//
// A real run reaches the handler while the program is writing; a fake has all of
// it already, and gives each stream in one chunk. Without this a caller that
// streams through the handler instead of reading the result -- which is what a
// caller with unbounded output is told to do -- would see nothing under Fake,
// and the fake would answer a question that caller never asks.
func emitToHandler(output OutputHandler, result ProcessResult) {
	if output == nil {
		return
	}
	if out := result.Output(); out != "" {
		output(Out, out)
	}
	if errorOutput := result.ErrorOutput(); errorOutput != "" {
		output(Err, errorOutput)
	}
}

// fakeFor is the handler for a command line, or nil when none matched.
func (p *PendingProcess) fakeFor(command string) func(*PendingProcess) any {
	for _, handler := range p.fakeHandlers {
		if matchesCommand(handler.Command, command) {
			return handler.callback()
		}
	}
	return nil
}

// resolveSynchronousFake turns what a handler returned into a result.
func (p *PendingProcess) resolveSynchronousFake(command string, fake func(*PendingProcess) any) (ProcessResult, error) {
	switch result := fake(p).(type) {
	case string:
		return NewFakeProcessResult(command, 0, result, ""), nil
	case []string:
		return NewFakeProcessResult(command, 0, result, ""), nil
	case *FakeProcessResult:
		return result.WithCommand(command), nil
	case *FakeProcessDescription:
		return result.ToProcessResult(command), nil
	case *FakeProcessSequence:
		next, err := result.next()
		if err != nil {
			return nil, err
		}
		return p.resolveSynchronousFake(command, func(*PendingProcess) any { return next })
	case ProcessResult:
		return result, nil
	default:
		return nil, fmt.Errorf("%w: synchronous", ErrUnsupportedFakeResult)
	}
}

// resolveAsynchronousFake turns what a handler returned into a started fake.
//
// A plain result becomes a description that prints it all at once and is
// finished on the first question.
func (p *PendingProcess) resolveAsynchronousFake(command string, output OutputHandler, fake func(*PendingProcess) any) (*FakeInvokedProcess, error) {
	switch result := fake(p).(type) {
	case string:
		return p.fakeInvokedFrom(command, output, NewFakeProcessResult(command, 0, result, "")), nil
	case []string:
		return p.fakeInvokedFrom(command, output, NewFakeProcessResult(command, 0, result, "")), nil
	case *FakeProcessDescription:
		return NewFakeInvokedProcess(command, result).WithOutputHandler(output), nil
	case *FakeProcessSequence:
		next, err := result.next()
		if err != nil {
			return nil, err
		}
		return p.resolveAsynchronousFake(command, output, func(*PendingProcess) any { return next })
	case ProcessResult:
		return p.fakeInvokedFrom(command, output, result), nil
	default:
		return nil, fmt.Errorf("%w: asynchronous", ErrUnsupportedFakeResult)
	}
}

func (p *PendingProcess) fakeInvokedFrom(command string, output OutputHandler, result ProcessResult) *FakeInvokedProcess {
	description := NewFakeProcessDescription().
		ReplaceOutput(result.Output()).
		ReplaceErrorOutput(result.ErrorOutput()).
		RunsFor(0).
		ExitCode(result.ExitCode())

	return NewFakeInvokedProcess(command, description).WithOutputHandler(output)
}
