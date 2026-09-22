package queue

import (
	"context"
	"errors"
	"fmt"
	"os"
	"runtime"
	"strconv"
	"time"

	"github.com/arandu-io/hesape/process"
)

// ListenerOptions configures a [Listener].
//
// It is [WorkerOptions] with one field added: the child workers run with the
// embedded options, and the listener turns them back into flags.
type ListenerOptions struct {
	// WorkerOptions is what each child worker runs with. The listener turns
	// them back into the flags it starts the child with.
	WorkerOptions

	// Environment is the environment the child workers run in, and empty means
	// they inherit this process's.
	Environment string
}

// Listener runs a worker in a child process and restarts it when it exits.
//
// It is what `aru queue:listen` runs, and it is not the way to run a queue in
// production -- [Worker] is. What it is for is development: a rebuilt binary is
// picked up without anybody remembering to restart anything.
//
// The public command is `aru queue:work`. Aru delegates it to the application's
// internal `<app-binary> work` subcommand, which is the protocol the listener
// uses to start each child. It is not a second public Aru command.
//
// The child is started with Command, which is the path of the application
// binary and the arguments before the ones this adds. An application with a
// different internal worker protocol says so there.
//
// The child is a program and a list of arguments, run through
// [github.com/arandu-io/hesape/process]. No shell is involved, so a queue name
// or a connection name is an argument whatever characters are in it.
type Listener struct {
	// commandPath is the working directory the child is started in.
	commandPath string
	// command is the binary and the leading arguments.
	command []string
	// outputHandler receives each line the child writes.
	outputHandler func(line string)
	// stop ends the loop. It is a function so [Listener.Stop] can be called
	// from a signal handler and from a test.
	stop context.CancelFunc
	// factory builds the child worker's process.
	factory *process.Factory
}

// NewListener returns a listener that starts command in dir.
//
// The binary is the thing being run, so the caller says what it is:
//
//	queue.NewListener(".", os.Args[0], "work")
func NewListener(dir string, command ...string) *Listener {
	return &Listener{commandPath: dir, command: command, factory: process.NewFactory()}
}

// UseProcessFactory points the listener at another process factory, and a nil
// one leaves the listener on the factory it has.
//
// It is the seam a test uses: a factory with fakes registered answers the child
// worker's command line without a worker ever being started, and
// [github.com/arandu-io/hesape/process.Factory.PreventStrayProcesses] turns a
// command nobody faked into a failure that names it.
func (l *Listener) UseProcessFactory(factory *process.Factory) *Listener {
	if factory != nil {
		l.factory = factory
	}
	return l
}

// SetOutputHandler sets what to do with each line the child writes.
//
// Without one the child's output goes nowhere, which is what a test wants; `aru
// queue:listen` passes one that writes to the terminal.
func (l *Listener) SetOutputHandler(handler func(line string)) {
	l.outputHandler = handler
}

// ErrNoListenerCommand is returned when a listener was built with no command to
// run.
var ErrNoListenerCommand = errors.New("queue: this listener has no command to run. Pass one to NewListener")

// Listen starts a worker for connection and queue, and restarts it whenever it
// exits.
//
// It returns when the context is cancelled or [Listener.Stop] is called.
//
// A child that exits with [ExitMemoryLimit] is started again, because that is
// what it stopped for. A child that cannot be started at all is an error: a
// loop that retries a binary that does not exist is a loop that fills a disk
// with the same line.
//
// The two are told apart by the process package rather than by reading an exit
// status: a worker that ran and exited non-zero is a result, and only a worker
// that never ran comes back as an error.
func (l *Listener) Listen(ctx context.Context, connectionName, queue string, options ListenerOptions) error {
	if len(l.command) == 0 {
		return ErrNoListenerCommand
	}

	ctx, cancel := context.WithCancel(ctx)
	defer cancel()
	l.stop = cancel

	for {
		if ctx.Err() != nil {
			return nil
		}

		if err := l.RunProcess(ctx, l.MakeProcess(connectionName, queue, options), options.Memory); err != nil {
			if ctx.Err() != nil {
				return nil
			}
			// Not the worker failing: the worker not starting. Retrying
			// cannot fix a path that is wrong.
			return fmt.Errorf("queue: starting the worker: %w", err)
		}

		if options.Rest > 0 {
			timer := time.NewTimer(options.Rest)
			select {
			case <-ctx.Done():
				timer.Stop()
				return nil
			case <-timer.C:
			}
		}
	}
}

// MakeProcess describes the child worker, without starting it.
//
// The flags are the worker's options turned back into the arguments the
// internal `<app-binary> work` subcommand parses. This is the private side of
// the public `aru queue:work` delegation and is the round trip that makes a
// listener and a worker configurable in one place. They are arguments and not
// a line: a queue named with a space, a semicolon or a quote is one argument all
// the same.
//
// The child is given no deadline. It runs one job and exits, and a job is as
// long as the work in it.
func (l *Listener) MakeProcess(connectionName, queue string, options ListenerOptions) *process.PendingProcess {
	command := append([]string(nil), l.command...)
	command = append(command,
		connectionName,
		"--once",
		"--name="+options.Name,
		"--queue="+queue,
		"--backoff="+seconds(options.Backoff),
		"--memory="+strconv.Itoa(options.Memory),
		"--sleep="+seconds(func(int) time.Duration { return options.Sleep }),
		"--tries="+strconv.Itoa(options.MaxTries),
	)
	if options.Force {
		command = append(command, "--force")
	}
	if options.Environment != "" {
		command = append(command, "--env="+options.Environment)
	}

	return l.factory.NewPendingProcess().
		Command(command...).
		Path(l.commandPath).
		Quietly().
		Forever()
}

// seconds renders a backoff schedule as the whole seconds a flag carries.
//
// The first attempt's wait is the one a child worker needs: it runs one job and
// exits, so it never sees a second.
func seconds(backoff func(int) time.Duration) string {
	if backoff == nil {
		return "0"
	}
	return strconv.Itoa(int(backoff(1) / time.Second))
}

// RunProcess runs one child worker to completion.
//
// A worker that exited non-zero is not an error: it is what the listener exists
// to start again. The error is for a worker that never ran, or one this
// process's context ended.
//
// Both of the child's streams go to the output handler as they arrive, which is
// what the child writing to standard error and standard output at once needs;
// nothing is kept, because the listener never reads it back.
//
// The memory check afterwards is about this process, the listener: it has
// grown, and something outside it should start it again with a clean heap.
func (l *Listener) RunProcess(ctx context.Context, worker *process.PendingProcess, memory int) error {
	var handler process.OutputHandler
	if l.outputHandler != nil {
		handler = func(_ process.Stream, chunk string) { l.outputHandler(chunk) }
	}

	if _, err := worker.Run(ctx, nil, handler); err != nil {
		return err
	}

	if l.MemoryExceeded(memory) {
		l.Stop()
	}
	return nil
}

// MemoryExceeded reports whether this process is holding more than limit
// megabytes.
//
// It reads MemStats.Sys, for the reason [Worker.MemoryExceeded] gives.
func (l *Listener) MemoryExceeded(limitMB int) bool {
	if limitMB <= 0 {
		return false
	}
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return m.Sys/1024/1024 >= uint64(limitMB)
}

// Stop ends the loop after the child that is running.
//
// It cancels rather than ending the process: a listener that killed the process
// would take the child with it, and the child is holding a job.
func (l *Listener) Stop() {
	if l.stop != nil {
		l.stop()
	}
}

// Stderr is where a listener with no output handler sends the child's output.
// It is here so a caller can say `l.SetOutputHandler(queue.Stderr)`.
func Stderr(line string) { fmt.Fprint(os.Stderr, line) }
