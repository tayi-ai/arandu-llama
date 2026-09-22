package process

import "strings"

// FakeProcessResult is a ProcessResult that no program produced.
//
// A fake handler may return one, and a test may build one directly.
//
// It is a separate type from the real result, and not a convenience over it.
// See normalizeOutput.
type FakeProcessResult struct {
	command     string
	exitCode    int
	output      string
	errorOutput string
}

var _ ProcessResult = (*FakeProcessResult)(nil)

// NewFakeProcessResult builds a fake result.
//
// Anything else is treated as empty.
func NewFakeProcessResult(command string, exitCode int, output, errorOutput any) *FakeProcessResult {
	return &FakeProcessResult{
		command:     command,
		exitCode:    exitCode,
		output:      normalizeOutput(output),
		errorOutput: normalizeOutput(errorOutput),
	}
}

// normalizeOutput turns what a test wrote into what a program would have
// printed.
//
// A string gets exactly one trailing newline, however many it had. A list of
// lines gets one newline between each and, unlike the string, none at the end.
func normalizeOutput(output any) string {
	switch value := output.(type) {
	case nil:
		return ""
	case string:
		if value == "" || value == "0" {
			return ""
		}
		return strings.TrimRight(value, "\n") + "\n"
	case []string:
		if len(value) == 0 {
			return ""
		}
		var lines strings.Builder
		for _, line := range value {
			lines.WriteString(strings.TrimRight(line, "\n"))
			lines.WriteString("\n")
		}
		return strings.TrimRight(lines.String(), "\n")
	default:
		return ""
	}
}

// Command is the command line this result claims to come from.
func (r *FakeProcessResult) Command() string { return r.command }

// WithCommand is a copy of this result attached to the given command line.
func (r *FakeProcessResult) WithCommand(command string) *FakeProcessResult {
	return &FakeProcessResult{
		command:     command,
		exitCode:    r.exitCode,
		output:      r.output,
		errorOutput: r.errorOutput,
	}
}

// Successful reports an exit code of zero.
func (r *FakeProcessResult) Successful() bool { return r.exitCode == 0 }

// Failed is the other half of Successful.
func (r *FakeProcessResult) Failed() bool { return !r.Successful() }

// ExitCode is the status this result claims.
func (r *FakeProcessResult) ExitCode() int { return r.exitCode }

// Output is the standard output this result claims.
func (r *FakeProcessResult) Output() string { return r.output }

// SeeInOutput reports whether Output contains the given string.
func (r *FakeProcessResult) SeeInOutput(output string) bool {
	return strings.Contains(r.Output(), output)
}

// ErrorOutput is the standard error this result claims.
func (r *FakeProcessResult) ErrorOutput() string { return r.errorOutput }

// SeeInErrorOutput reports whether ErrorOutput contains the given string.
func (r *FakeProcessResult) SeeInErrorOutput(output string) bool {
	return strings.Contains(r.ErrorOutput(), output)
}

// Throw reports a failed result as an error.
func (r *FakeProcessResult) Throw(callback func(ProcessResult, *ProcessFailedException)) (ProcessResult, error) {
	return throw(r, callback)
}

// ThrowIf is Throw when the condition holds.
func (r *FakeProcessResult) ThrowIf(condition bool, callback func(ProcessResult, *ProcessFailedException)) (ProcessResult, error) {
	return throwIf(r, condition, callback)
}
