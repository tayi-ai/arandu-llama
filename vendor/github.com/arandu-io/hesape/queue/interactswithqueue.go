package queue

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/arandu-io/hesape/queue/jobs"
)

// TestingT is the two methods the assertions here need from *testing.T.
//
// It is declared rather than imported so that nothing in this package depends
// on the testing package, which would put its flags on every binary that links
// the queue.
type TestingT interface {
	// Helper marks the caller as a test helper, so a failure points at the line
	// in the test rather than at the line inside this package.
	Helper()
	// Errorf records the failure and marks the test as failed.
	Errorf(format string, args ...any)
}

// InteractsWithQueue is what a handler embeds to reach the job it is running.
//
// A handler is already given the job -- Handler.Handle takes it -- so this is
// for the other shape: a handler that is a struct with state, whose methods
// settle the job on themselves rather than threading it through.
//
//	type SendInvoice struct {
//		queue.InteractsWithQueue
//		Invoices *invoice.Repository
//	}
//
//	func (h *SendInvoice) Handle(ctx context.Context, g auth.Grant, j *jobs.Job) error {
//		h.SetJob(j)
//		if !h.Invoices.Ready(ctx, g) {
//			return h.Release(ctx, time.Minute)
//		}
//		...
//	}
//
// A zero value with no job is safe: every method answers as though the job had
// already been settled.
type InteractsWithQueue struct {
	// Job is the job being run, or nil outside a worker.
	//
	// It is exported so a handler that wants the payload reads it here rather
	// than being handed a fourth accessor for it.
	Job *jobs.Job

	// fake is set by WithFakeQueueInteractions, and is what the assertions
	// read. It is separate from Job because Job is what a handler uses and this
	// is what a test does.
	fake *jobs.FakeJob
}

// SetJob gives the handler the job it is running, and returns the receiver so
// the call chains.
func (i *InteractsWithQueue) SetJob(j *jobs.Job) *InteractsWithQueue {
	i.Job = j
	return i
}

// Attempts is how many times this job has been delivered, counting now.
//
// A handler with no job answers 1: code that branches on "is this the first
// try" must behave outside a worker as it does on the first try.
func (i *InteractsWithQueue) Attempts() int {
	if i.Job == nil {
		return 1
	}
	return i.Job.Attempts
}

// Delete removes the job from its queue.
func (i *InteractsWithQueue) Delete(ctx context.Context) error {
	if i.Job == nil {
		return nil
	}
	return i.Job.Delete(ctx)
}

// Release puts the job back on its queue, eligible again after delay.
func (i *InteractsWithQueue) Release(ctx context.Context, delay time.Duration) error {
	if i.Job == nil {
		return nil
	}
	return i.Job.Release(ctx, delay)
}

// Fail parks the job.
//
// A nil cause parks it with [ErrManuallyFailed]: the record has to say
// something, and "the handler asked for it" is the truth.
func (i *InteractsWithQueue) Fail(ctx context.Context, cause error) error {
	if i.Job == nil {
		return nil
	}
	if cause == nil {
		cause = ErrManuallyFailed
	}
	return i.Job.Fail(ctx, cause)
}

// WithFakeQueueInteractions makes release, delete and fail record what was
// asked for instead of doing it, and returns the receiver so the call chains.
//
// It is what a test calls before running the handler directly, so the handler
// can be asked what it decided without a store, a worker or a transaction.
func (i *InteractsWithQueue) WithFakeQueueInteractions() *InteractsWithQueue {
	i.fake = jobs.NewFakeJob("fake", "fake")
	i.Job = i.fake.Job
	return i
}

// ensureFaked reports whether the assertions can be answered, and fails the
// test when they cannot.
//
// It fails the test rather than panicking: an assertion that panics takes the
// whole run with it, and the thing that went wrong is one missing line in one
// test.
func (i *InteractsWithQueue) ensureFaked(t TestingT) bool {
	t.Helper()
	if i.fake == nil {
		t.Errorf("queue interactions have not been faked. Call WithFakeQueueInteractions before running the handler")
		return false
	}
	return true
}

// AssertDeleted fails the test unless the job was deleted.
func (i *InteractsWithQueue) AssertDeleted(t TestingT) *InteractsWithQueue {
	t.Helper()
	if i.ensureFaked(t) && !i.fake.IsDeleted() {
		t.Errorf("the job was expected to be deleted, and was not")
	}
	return i
}

// AssertNotDeleted fails the test if the job was deleted.
func (i *InteractsWithQueue) AssertNotDeleted(t TestingT) *InteractsWithQueue {
	t.Helper()
	if i.ensureFaked(t) && i.fake.IsDeleted() {
		t.Errorf("the job was deleted, and was not expected to be")
	}
	return i
}

// AssertFailed fails the test unless the job was parked.
func (i *InteractsWithQueue) AssertFailed(t TestingT) *InteractsWithQueue {
	t.Helper()
	if i.ensureFaked(t) && !i.fake.HasFailed() {
		t.Errorf("the job was expected to be failed, and was not")
	}
	return i
}

// AssertFailedWith fails the test unless the job was parked with a cause that
// matches want under errors.Is.
//
// errors.Is rather than equality, because a handler that wrapped the cause with
// context still failed with it.
func (i *InteractsWithQueue) AssertFailedWith(t TestingT, want error) *InteractsWithQueue {
	t.Helper()
	if !i.ensureFaked(t) {
		return i
	}
	i.AssertFailed(t)
	if !errors.Is(i.fake.FailedWith, want) {
		t.Errorf("the job was expected to fail with %v, and failed with %v", want, i.fake.FailedWith)
	}
	return i
}

// AssertNotFailed fails the test if the job was parked.
func (i *InteractsWithQueue) AssertNotFailed(t TestingT) *InteractsWithQueue {
	t.Helper()
	if i.ensureFaked(t) && i.fake.HasFailed() {
		t.Errorf("the job was failed, and was not expected to be")
	}
	return i
}

// AssertReleased fails the test unless the job was released.
//
// A negative delay asserts only that the job was released: a test that cares
// when it comes back says so, and one that does not should not have to.
func (i *InteractsWithQueue) AssertReleased(t TestingT, delay time.Duration) *InteractsWithQueue {
	t.Helper()
	if !i.ensureFaked(t) {
		return i
	}
	if !i.fake.IsReleased() {
		t.Errorf("the job was expected to be released, and was not")
		return i
	}
	if delay >= 0 && i.fake.ReleaseDelay != delay {
		t.Errorf("the job was expected to be released after %s, and was released after %s",
			delay, i.fake.ReleaseDelay)
	}
	return i
}

// AssertNotReleased fails the test if the job was released.
func (i *InteractsWithQueue) AssertNotReleased(t TestingT) *InteractsWithQueue {
	t.Helper()
	if i.ensureFaked(t) && i.fake.IsReleased() {
		t.Errorf("the job was released, and was not expected to be")
	}
	return i
}

// String names the job this handler is running, for a test failure to print.
func (i *InteractsWithQueue) String() string {
	if i.Job == nil {
		return "queue: a handler with no job"
	}
	return fmt.Sprintf("queue: %s on %s", i.Job.Name, i.Job.Queue)
}
