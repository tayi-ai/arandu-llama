package process

import (
	"context"
	"errors"
	"fmt"
	"time"
)

// ProcessFailedException is a command that ran and exited non-zero.
//
// It lives here and not in the exceptions directory because a Go error type
// belongs in the package that returns it -- see exceptions/doc.go.
//
// Nothing produces one on its own. It is what ProcessResult.Throw returns.
type ProcessFailedException struct {
	// Result is the result that failed.
	Result ProcessResult
	// Code is the exit status.
	Code int
}

// NewProcessFailedException builds the exception for a result.
func NewProcessFailedException(result ProcessResult) *ProcessFailedException {
	code := result.ExitCode()
	if code == 0 {
		// A failure with no status is still a failure, and zero would say the
		// opposite.
		code = 1
	}
	return &ProcessFailedException{Result: result, Code: code}
}

// Error is the command, the exit code, and each of the two output streams under
// a ruled heading when it has anything in it.
func (e *ProcessFailedException) Error() string {
	message := fmt.Sprintf("The command \"%s\" failed.\n\nExit Code: %d", e.Result.Command(), e.Result.ExitCode())
	if e.Result.Output() != "" {
		message += fmt.Sprintf("\n\nOutput:\n================\n%s", e.Result.Output())
	}
	if e.Result.ErrorOutput() != "" {
		message += fmt.Sprintf("\n\nError Output:\n================\n%s", e.Result.ErrorOutput())
	}
	return message
}

// ProcessTimedOutException is a command that was killed for taking too long, or
// -- when Idle is set -- for having gone too long without printing anything.
//
// It unwraps to context.DeadlineExceeded, so a caller that only wants to know
// whether time ran out never has to name this type.
type ProcessTimedOutException struct {
	// Result is what the program managed to say before it was killed.
	Result ProcessResult
	// Command is the command line that was killed.
	Command string
	// Timeout is the limit that was reached.
	Timeout time.Duration
	// Idle tells the two limits apart. They mean different things: a general
	// timeout is work that did not fit, an idle timeout is work that stopped
	// happening, and raising the limit does not help the second one.
	Idle bool
}

func (e *ProcessTimedOutException) Error() string {
	if e.Idle {
		return fmt.Sprintf("The process \"%s\" exceeded the idle timeout of %v seconds.", e.Command, e.Timeout.Seconds())
	}
	return fmt.Sprintf("The process \"%s\" exceeded the timeout of %v seconds.", e.Command, e.Timeout.Seconds())
}

// Unwrap reports the timeout as the standard one, so errors.Is against
// context.DeadlineExceeded answers yes.
func (e *ProcessTimedOutException) Unwrap() error { return context.DeadlineExceeded }

// StrayProcessError is a command that was about to really run inside a test
// that had asked for that not to happen.
//
// It is a named type rather than a bare error so that errors.As can find it and
// a test can tell it from a program that really failed.
type StrayProcessError struct {
	// Command is the command line that had no matching fake.
	Command string
}

func (e *StrayProcessError) Error() string {
	return fmt.Sprintf("Attempted process [%s] without a matching fake.", e.Command)
}

// ErrEmptyProcessSequence is a FakeProcessSequence that was asked for one more
// result than it was given.
//
// Call FakeProcessSequence.DontFailWhenEmpty or WhenEmpty to make the sequence
// answer instead of running out.
var ErrEmptyProcessSequence = errors.New("A process was invoked, but the process result sequence is empty.")

// ErrUnsupportedFakeResult is a fake handler that returned something no fake can
// be built from.
//
// What a handler may return is listed on Factory.Fake.
var ErrUnsupportedFakeResult = errors.New("unsupported process fake result provided")
