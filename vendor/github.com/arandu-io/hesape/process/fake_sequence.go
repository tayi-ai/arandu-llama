package process

import "sync"

// FakeProcessSequence is a queue of fake results handed out one per run.
//
// Factory.Sequence is where one comes from. It is the piece that lets a test
// say what happens the second time a command runs:
//
//	factory.Fake(process.FakeHandler{Command: "git *", Handler: factory.Sequence(
//		process.NewFakeProcessResult("", 1, "", "not a repository"),
//		"ok",
//	)})
//
// Run it out and the next command is an error, because a sequence that quietly
// repeats its last answer is a test that passes for a reason nobody wrote down.
// WhenEmpty and DontFailWhenEmpty are how a test asks for the other behaviour.
//
// It is safe to hand one sequence to concurrently running processes, because a
// pool is goroutines and the queue is taken from under a lock.
type FakeProcessSequence struct {
	mu            sync.Mutex
	processes     []any
	failWhenEmpty bool
	emptyProcess  any
}

// NewFakeProcessSequence builds a sequence of the given results.
//
// Each entry is a string, a []string, a ProcessResult or a
// *FakeProcessDescription.
func NewFakeProcessSequence(processes ...any) *FakeProcessSequence {
	sequence := &FakeProcessSequence{failWhenEmpty: true}
	for _, process := range processes {
		sequence.processes = append(sequence.processes, sequence.toProcessResult(process))
	}
	return sequence
}

// Push adds one more result to the end of the sequence.
func (s *FakeProcessSequence) Push(process any) *FakeProcessSequence {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.processes = append(s.processes, s.toProcessResult(process))
	return s
}

// WhenEmpty makes a run-out sequence answer with the given result instead of
// failing.
func (s *FakeProcessSequence) WhenEmpty(process any) *FakeProcessSequence {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.failWhenEmpty = false
	s.emptyProcess = s.toProcessResult(process)
	return s
}

// DontFailWhenEmpty makes a run-out sequence answer with an empty successful
// result.
func (s *FakeProcessSequence) DontFailWhenEmpty() *FakeProcessSequence {
	return s.WhenEmpty(NewFakeProcessResult("", 0, "", ""))
}

// IsEmpty reports that the sequence has handed out everything it was given.
func (s *FakeProcessSequence) IsEmpty() bool {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.processes) == 0
}

// toProcessResult turns a written-down result into one of the two things a
// sequence stores.
func (s *FakeProcessSequence) toProcessResult(process any) any {
	switch value := process.(type) {
	case string:
		return NewFakeProcessResult("", 0, value, "")
	case []string:
		return NewFakeProcessResult("", 0, value, "")
	default:
		return value
	}
}

// next takes the front of the sequence.
func (s *FakeProcessSequence) next() (any, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if len(s.processes) == 0 {
		if s.failWhenEmpty {
			return nil, ErrEmptyProcessSequence
		}
		if s.emptyProcess == nil {
			return NewFakeProcessResult("", 0, "", ""), nil
		}
		return s.emptyProcess, nil
	}

	process := s.processes[0]
	s.processes = s.processes[1:]
	return process, nil
}
