package jobs

import (
	"context"
	"time"
)

// FakeJob is a job that came off nothing, and records what was done to it.
//
// It is what a test wires when it wants to call a handler directly and then ask
// whether the handler released,
// deleted or failed its job -- without a store, a worker or a transaction.
//
//	f := jobs.NewFakeJob("invoice.send", tenant)
//	err := sendInvoice(ctx, auth.SystemGrant("invoice.send", tenant), f.Job)
//	if !f.IsReleased() || f.ReleaseDelay != time.Minute { ... }
//
// It is both the job and the [Driver] behind it, which is the whole trick: the
// job settles itself against the fake, and the fake is what the assertions
// read.
type FakeJob struct {
	// Job is the job a handler is given.
	*Job

	// ReleaseDelay is how long the job asked to wait before its next delivery.
	// It is zero on a job that was not released.
	ReleaseDelay time.Duration

	// FailedWith is the cause the job was parked with. It is nil on a job that
	// was not parked.
	FailedWith error
}

var _ Driver = (*FakeJob)(nil)

// NewFakeJob returns a job for name, belonging to tenant, that came off a fake
// queue.
//
// Attempts is 1: the job is being handled, and the handling counts.
func NewFakeJob(name, tenant string) *FakeJob {
	f := &FakeJob{}
	f.Job = Popped(f, "fake", Job{
		Name:     name,
		Queue:    DefaultQueue,
		TenantID: tenant,
		Attempts: 1,
	})
	return f
}

// ReleaseJob records the release. It answers release() on the fake.
func (f *FakeJob) ReleaseJob(_ context.Context, _ *Job, delay time.Duration) error {
	f.ReleaseDelay = delay
	return nil
}

// DeleteJob records the deletion. It answers delete() on the fake.
func (f *FakeJob) DeleteJob(context.Context, *Job) error { return nil }

// FailJob records the cause. It answers fail() on the fake.
func (f *FakeJob) FailJob(_ context.Context, _ *Job, cause error) error {
	f.FailedWith = cause
	return nil
}
