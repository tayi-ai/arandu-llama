package process

import (
	"context"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"sync"
	"sync/atomic"
	"syscall"
	"time"
)

// InvokedProcess is a command that was started and has not finished being dealt
// with.
//
// It is an interface because Start hands back a real process or a faked one and
// the caller is not supposed to be able to tell.
//
//	invoked, err := factory.Start(ctx, []string{"go", "build", "./..."}, nil)
//	for invoked.Running() {
//		// something else
//	}
//	result, err := invoked.Wait(nil)
type InvokedProcess interface {
	// ID is the process id the operating system gave the program.
	ID() int
	// Signal sends a signal to the program.
	Signal(signal os.Signal) error
	// Stop asks the program to end, and kills it when it does not.
	Stop(timeout time.Duration, signal os.Signal) error
	// Running reports whether the program is still going.
	Running() bool
	// Output is everything the program has written on standard output so far.
	Output() string
	// ErrorOutput is everything written on standard error so far.
	ErrorOutput() string
	// LatestOutput is everything written on standard output since the last
	// time this was asked.
	LatestOutput() string
	// LatestErrorOutput is the same for standard error.
	LatestErrorOutput() string
	// Wait waits for the program to finish. The handler takes over from the
	// one Start was given; pass nil to keep it.
	Wait(output OutputHandler) (ProcessResult, error)
}

// stopTimeout is how long Stop waits between asking and killing when the caller
// named no timeout.
const stopTimeout = 10 * time.Second

// waitDelay bounds how long the wait keeps reading after the program itself has
// exited.
//
// A child that spawned children of its own hands them the same pipes, and
// those stay open after it dies -- so a wait with no bound is a command that
// finished and a caller that never returns. Five seconds is long enough that a
// normal exit never notices and short enough that a stuck one is a hiccup
// rather than a hang.
const waitDelay = 5 * time.Second

// invokedProcess is the InvokedProcess of a program that really started.
//
// Unexported for the reason processResult is: everything a caller can ask is on
// the contract, and the concrete type only holds the machinery that answers.
type invokedProcess struct {
	command     string
	cmd         *exec.Cmd
	stdout      *buffer
	stderr      *buffer
	sink        *sink
	timeout     time.Duration
	idleTimeout time.Duration
	quietly     bool

	// parent is the context the caller passed; run is the one the process is
	// tied to. Telling them apart is what says whether it was the caller who
	// gave up or one of this package's own limits.
	parent context.Context
	cancel context.CancelFunc
	run    context.Context

	idled    atomic.Bool
	watching chan struct{}
	started  time.Time

	settled chan struct{}
	waitErr error

	once   sync.Once
	result ProcessResult
	err    error
}

var _ InvokedProcess = (*invokedProcess)(nil)

// ID is the process id.
func (i *invokedProcess) ID() int {
	if i.cmd.Process == nil {
		return 0
	}
	return i.cmd.Process.Pid
}

// Signal sends a signal to the program.
//
// It carries the platform's rules rather than hiding them: on Windows only
// Kill is delivered. Signalling a process that has already exited is an error.
func (i *invokedProcess) Signal(signal os.Signal) error {
	if i.cmd.Process == nil {
		return errors.New("process: signalling a command that was never started")
	}
	if !i.Running() {
		return fmt.Errorf("process: signalling [%s], which has already exited", i.command)
	}
	return i.cmd.Process.Signal(signal)
}

// Stop asks the program to end and kills it if it has not ended in time.
//
// The signal is sent, and after the timeout the program is killed outright. A
// zero timeout means ten seconds and a nil signal means SIGTERM.
func (i *invokedProcess) Stop(timeout time.Duration, signal os.Signal) error {
	if !i.Running() {
		return nil
	}
	if timeout <= 0 {
		timeout = stopTimeout
	}
	if signal == nil {
		signal = syscall.SIGTERM
	}
	if i.cmd.Process != nil {
		_ = i.cmd.Process.Signal(signal)
	}

	timer := time.NewTimer(timeout)
	defer timer.Stop()
	select {
	case <-i.settled:
		return nil
	case <-timer.C:
		if i.cmd.Process == nil {
			return nil
		}
		return i.cmd.Process.Kill()
	}
}

// Running reports whether the program is still going.
//
// It answers about the moment it was asked -- a program can exit between the
// answer and the next line -- and it becomes false on its own, without anybody
// calling Wait, because that is what `for invoked.Running()` needs to end.
func (i *invokedProcess) Running() bool {
	select {
	case <-i.settled:
		return false
	default:
		return true
	}
}

// Output is everything written on standard output so far.
func (i *invokedProcess) Output() string { return i.stdout.all() }

// ErrorOutput is everything written on standard error so far.
func (i *invokedProcess) ErrorOutput() string { return i.stderr.all() }

// LatestOutput is everything written on standard output since the last call.
func (i *invokedProcess) LatestOutput() string { return i.stdout.incremental() }

// LatestErrorOutput is the same for standard error.
func (i *invokedProcess) LatestErrorOutput() string { return i.stderr.incremental() }

// Wait waits for the program to finish and reports what happened.
//
// A handler passed here takes over from the one Start was given, for whatever
// output has not arrived yet. It can be called more than once and from more
// than one goroutine: the answer is worked out once and every call gets it.
//
// A command that exited non-zero comes back with a nil error, because a failed
// command is a result rather than a failure of the call; ProcessResult.Throw is
// what turns it into one. The error is for the program that timed out, or whose
// context the caller cancelled.
func (i *invokedProcess) Wait(output OutputHandler) (ProcessResult, error) {
	if output != nil {
		i.sink.mu.Lock()
		i.sink.handler = output
		i.sink.mu.Unlock()
	}
	<-i.settled

	i.once.Do(func() {
		i.result = &processResult{
			command:     i.command,
			exitCode:    exitCode(i.cmd),
			output:      i.stdout.all(),
			errorOutput: i.stderr.all(),
		}
		i.err = i.classify(i.waitErr)
	})
	return i.result, i.err
}

// classify turns what os/exec reported into what actually happened.
//
// The order is the whole point. A killed process comes back as an exit status
// whatever killed it, so every reason this package had to kill one has to be
// examined before the exit status is believed -- otherwise a command that hit
// its deadline is reported as a program that failed, and the person goes
// looking for a bug in it.
func (i *invokedProcess) classify(err error) error {
	if err == nil {
		return nil
	}
	// The caller gave up first: their cancellation, not our limit, and the error
	// they get back is the one they can test with errors.Is.
	if cause := i.parent.Err(); cause != nil {
		return fmt.Errorf("process: [%s]: %w", i.command, cause)
	}
	if i.idled.Load() {
		return &ProcessTimedOutException{
			Result:  i.result,
			Command: i.command,
			Timeout: i.idleTimeout,
			Idle:    true,
		}
	}
	if errors.Is(context.Cause(i.run), context.DeadlineExceeded) {
		return &ProcessTimedOutException{
			Result:  i.result,
			Command: i.command,
			Timeout: i.timeout,
		}
	}
	// A non-zero exit is not an error here: Failed reports it and Throw is
	// what raises it.
	var exit *exec.ExitError
	if errors.As(err, &exit) {
		return nil
	}
	return fmt.Errorf("process: [%s]: %w", i.command, err)
}

// watchIdle kills the program when it has gone quiet for too long.
//
// The timer is reset on every chunk either stream produces, so a program that
// prints anything at all is never touched.
func (i *invokedProcess) watchIdle(silence time.Duration, bump chan struct{}) {
	timer := time.NewTimer(silence)
	defer timer.Stop()
	for {
		select {
		case <-bump:
			timer.Stop()
			timer.Reset(silence)
		case <-timer.C:
			// Flagged before the kill, because the kill is what the wait sees
			// and by then the reason has to be readable.
			i.idled.Store(true)
			i.cancel()
			return
		case <-i.watching:
			return
		}
	}
}

// exitCode is what the program returned, and -1 for every way of leaving that
// is not a returned status.
func exitCode(cmd *exec.Cmd) int {
	if cmd.ProcessState == nil {
		return -1
	}
	return cmd.ProcessState.ExitCode()
}

// sink is what the two collectors share: the caller's handler and the signal
// that keeps the idle watchdog awake.
//
// One mutex for both streams is what lets an OutputHandler be written without a
// lock of its own -- stdout and stderr are copied by two goroutines, and a
// handler that writes to a terminal from both at once interleaves a line into
// nonsense.
type sink struct {
	mu      sync.Mutex
	handler OutputHandler
	bump    chan struct{}
}

// collector is one stream on its way to a buffer and a handler.
type collector struct {
	stream  Stream
	buf     *buffer
	sink    *sink
	quietly bool
}

func (c *collector) Write(p []byte) (int, error) {
	c.sink.mu.Lock()
	defer c.sink.mu.Unlock()

	if !c.quietly {
		c.buf.write(p)
	}
	if c.sink.handler != nil {
		c.sink.handler(c.stream, string(p))
	}
	if c.sink.bump != nil {
		// Non-blocking: the watchdog only needs to know that something
		// arrived, and a full buffer already says so. Blocking here would let
		// a quiet watchdog stall the program's own output.
		select {
		case c.sink.bump <- struct{}{}:
		default:
		}
	}
	// Always the full length: a short write stops exec's copy and turns "this
	// program printed a lot" into "this program failed".
	return len(p), nil
}
