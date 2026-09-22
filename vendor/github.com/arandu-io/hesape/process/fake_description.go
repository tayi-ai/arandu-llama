package process

import "strings"

// fakeOutputLine is one entry of a description's output: the
// ['type' => 'out'|'err', 'buffer' => string].
type fakeOutputLine struct {
	stream Stream
	buffer string
}

// FakeProcessDescription is a fake process written out as a lifecycle rather
// than as a result: what it prints, in what order, on which stream, how long it
// stays running, and what it exits with.
//
// It is what a test reaches for when the code under test starts a process and
// watches it -- a result alone cannot be watched, because it has already
// happened.
//
// Go cannot have both, so the methods win and the state is unexported.
type FakeProcessDescription struct {
	processID     int
	output        []fakeOutputLine
	exitCode      int
	runIterations int
}

// NewFakeProcessDescription builds an empty description, which takes nothing but does
// not start blank: the process id is 1000, the way it is there.
func NewFakeProcessDescription() *FakeProcessDescription {
	return &FakeProcessDescription{processID: 1000}
}

// ID sets the process id the fake will report.
func (d *FakeProcessDescription) ID(processID int) *FakeProcessDescription {
	d.processID = processID
	return d
}

// Output describes a line of standard output.
//
// Each line is stored with exactly one trailing newline, however many it was
// written with -- a described line is a line, and a test that wrote one without
// a newline should not get output that runs into the next.
func (d *FakeProcessDescription) Output(output any) *FakeProcessDescription {
	return d.describe(Out, output)
}

// ErrorOutput describes a line of error output.
//
// Order is kept across both streams: a description that writes a line out, then
// a line err, then a line out replays them in that order, which is what makes
// an interleaved output handler testable.
func (d *FakeProcessDescription) ErrorOutput(output any) *FakeProcessDescription {
	return d.describe(Err, output)
}

func (d *FakeProcessDescription) describe(stream Stream, output any) *FakeProcessDescription {
	switch value := output.(type) {
	case string:
		d.output = append(d.output, fakeOutputLine{stream: stream, buffer: strings.TrimRight(value, "\n") + "\n"})
	case []string:
		for _, line := range value {
			d.describe(stream, line)
		}
	}
	return d
}

// ReplaceOutput drops everything described on standard output and puts the
// given string there instead.
//
// The replacement goes on the end of the description, not where the dropped
// lines were.
func (d *FakeProcessDescription) ReplaceOutput(output string) *FakeProcessDescription {
	return d.replace(Out, output)
}

// ReplaceErrorOutput drops everything described on standard error and puts the
// given string there instead.
func (d *FakeProcessDescription) ReplaceErrorOutput(output string) *FakeProcessDescription {
	return d.replace(Err, output)
}

func (d *FakeProcessDescription) replace(stream Stream, output string) *FakeProcessDescription {
	kept := d.output[:0:0]
	for _, line := range d.output {
		if line.stream != stream {
			kept = append(kept, line)
		}
	}
	if len(output) > 0 {
		kept = append(kept, fakeOutputLine{stream: stream, buffer: strings.TrimRight(output, "\n") + "\n"})
	}
	d.output = kept
	return d
}

// ExitCode sets the status the fake will exit with.
func (d *FakeProcessDescription) ExitCode(exitCode int) *FakeProcessDescription {
	d.exitCode = exitCode
	return d
}

// Iterations sets how many times Running answers true before the fake is
// finished.
func (d *FakeProcessDescription) Iterations(iterations int) *FakeProcessDescription {
	return d.RunsFor(iterations)
}

// RunsFor sets how many times Running answers true before the fake is
// finished.
//
// It is a count of questions, not a length of time: a fake never really runs,
// so the only thing a test can control is how long the loop that watches it
// goes round.
func (d *FakeProcessDescription) RunsFor(iterations int) *FakeProcessDescription {
	d.runIterations = iterations
	return d
}

// ToProcessResult is the whole description collapsed into the result it ends
// with.
func (d *FakeProcessDescription) ToProcessResult(command string) ProcessResult {
	return &FakeProcessResult{
		command:     command,
		exitCode:    d.exitCode,
		output:      d.resolveOutput(Out),
		errorOutput: d.resolveOutput(Err),
	}
}

// resolveOutput joins one stream of the description.
func (d *FakeProcessDescription) resolveOutput(stream Stream) string {
	var joined strings.Builder
	found := false
	for _, line := range d.output {
		if line.stream == stream {
			joined.WriteString(line.buffer)
			found = true
		}
	}
	if !found {
		return ""
	}
	return strings.TrimRight(joined.String(), "\n") + "\n"
}
