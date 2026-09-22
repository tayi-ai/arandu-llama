package process

import "strings"

// ProcessResult is what a command left behind.
//
// It is an interface because Run hands back a real result or a faked one, and
// the caller is not supposed to be able to tell.
//
// A command that exited non-zero is not an error. Run returns the result with a
// nil error and Failed reports the exit; Throw is what turns it into one. The
// error Run returns is for the process that never ran, timed out, or was stray.
type ProcessResult interface {
	// Command is the command line that was run.
	Command() string
	// Successful reports an exit code of zero.
	Successful() bool
	// Failed is the other half of Successful.
	Failed() bool
	// ExitCode is what the program returned, and -1 when it never got far
	// enough to return anything.
	ExitCode() int
	// Output is everything the program wrote on standard output.
	Output() string
	// SeeInOutput reports whether Output contains the given string.
	SeeInOutput(output string) bool
	// ErrorOutput is everything the program wrote on standard error.
	ErrorOutput() string
	// SeeInErrorOutput reports whether ErrorOutput contains the given string.
	SeeInErrorOutput(output string) bool
	// Throw returns a ProcessFailedException when the command failed.
	Throw(callback func(ProcessResult, *ProcessFailedException)) (ProcessResult, error)
	// ThrowIf is Throw when the condition holds.
	ThrowIf(condition bool, callback func(ProcessResult, *ProcessFailedException)) (ProcessResult, error)
}

// processResult is the ProcessResult of a program that really ran.
//
// It keeps the four answers rather than the command it came from, because by
// the time a result exists there is nothing left to ask an os/exec.Cmd that is
// not already read out of it. It is unexported for that reason: there is no
// state on it a caller could want that the interface does not already expose.
type processResult struct {
	command     string
	exitCode    int
	output      string
	errorOutput string
}

var _ ProcessResult = (*processResult)(nil)

// Command is the command line that was run.
func (r *processResult) Command() string { return r.command }

// Successful reports an exit code of zero.
func (r *processResult) Successful() bool { return r.exitCode == 0 }

// Failed reports that the command did not succeed.
func (r *processResult) Failed() bool { return !r.Successful() }

// ExitCode is the status the program returned.
func (r *processResult) ExitCode() int { return r.exitCode }

// Output is standard output.
func (r *processResult) Output() string { return r.output }

// SeeInOutput reports whether standard output contains the given string.
func (r *processResult) SeeInOutput(output string) bool {
	return strings.Contains(r.Output(), output)
}

// ErrorOutput is standard error.
func (r *processResult) ErrorOutput() string { return r.errorOutput }

// SeeInErrorOutput reports whether standard error contains the given string.
func (r *processResult) SeeInErrorOutput(output string) bool {
	return strings.Contains(r.ErrorOutput(), output)
}

// Throw reports a failed command as an error.
func (r *processResult) Throw(callback func(ProcessResult, *ProcessFailedException)) (ProcessResult, error) {
	return throw(r, callback)
}

// ThrowIf is Throw when the condition holds, and nothing when it does not.
func (r *processResult) ThrowIf(condition bool, callback func(ProcessResult, *ProcessFailedException)) (ProcessResult, error) {
	return throwIf(r, condition, callback)
}

// throw is the body both ProcessResult implementations share.
func throw(r ProcessResult, callback func(ProcessResult, *ProcessFailedException)) (ProcessResult, error) {
	if r.Successful() {
		return r, nil
	}
	err := NewProcessFailedException(r)
	if callback != nil {
		callback(r, err)
	}
	return r, err
}

func throwIf(r ProcessResult, condition bool, callback func(ProcessResult, *ProcessFailedException)) (ProcessResult, error) {
	if condition {
		return throw(r, callback)
	}
	return r, nil
}
