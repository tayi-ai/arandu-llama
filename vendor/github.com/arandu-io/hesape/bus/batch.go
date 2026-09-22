package bus

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database"
)

// BatchOptions is the callbacks and the settings a batch carries: six moments
// that name a job, and three settings that say where the work goes and what a
// failure does to the rest of it.
//
// Extra is what [PendingBatch.WithOption] writes, for an application that hangs
// its own bookkeeping off a batch -- the id of the upload the import came from,
// the user who pressed the button.
type BatchOptions struct {
	// Connection and Queue are where the jobs and the callbacks go when a step
	// does not name its own.
	Connection string `json:"connection,omitempty"`
	Queue      string `json:"queue,omitempty"`
	// AllowFailures decides what a failure does to the rest of the batch: with
	// it, the remaining jobs keep going and the batch finishes when the last one
	// reports; without it, the first failure cancels the batch and the jobs
	// still queued are told to stop by Batchable.Batching.
	//
	// Catch fires on the first failure either way. AllowFailures is about the
	// other jobs, not about whether anyone is told.
	AllowFailures bool `json:"allow_failures,omitempty"`
	// Before runs once the batch has been stored and before its jobs are
	// pushed. Progress runs after every job that reports. Then runs when every
	// job succeeded. Catch runs on the first failure. Failure runs on every
	// failure of a batch that allows them. Finally runs when every job has
	// reported, whichever way each of them went.
	//
	// Each is a job rather than a function -- see the package doc for why.
	Before   Step `json:"before"`
	Progress Step `json:"progress"`
	Then     Step `json:"then"`
	Catch    Step `json:"catch"`
	Failure  Step `json:"failure"`
	Finally  Step `json:"finally"`
	// Extra is whatever WithOption put there.
	Extra map[string]string `json:"extra,omitempty"`
}

// Batch is a group of jobs dispatched together and counted as one.
//
// The counters are the whole point. TotalJobs never moves except when Add puts
// more in. PendingJobs falls by one
// for every job that *succeeded*. FailedJobs rises for every job that failed,
// and does not touch PendingJobs -- which is why "every job has reported" is
// PendingJobs minus FailedJobs reaching zero (UpdatedBatchJobCounts) while
// "every job succeeded" is PendingJobs reaching zero.
//
// A Batch value is a snapshot. It is read from the repository, it is already
// stale when it arrives, and nothing decides anything by mutating it: only
// DecrementPendingJobs and IncrementFailedJobs move a counter, and they move it
// where every worker can see it.
type Batch struct {
	// ID identifies the batch and travels inside the payload of every job in
	// it.
	ID string
	// Name is for people: it is what the batch is called on a dashboard and in
	// a log line. It is not unique and nothing looks a batch up by it.
	Name string
	// TenantID is who the batch belongs to. It comes from the Grant that
	// dispatched it, and every statement about the batch filters on it.
	TenantID string
	// TotalJobs is how many jobs the batch holds. PendingJobs is how many have
	// not succeeded yet. FailedJobs is how many reported an error.
	TotalJobs   int
	PendingJobs int
	FailedJobs  int
	// FailedJobIDs is which jobs failed. The list is kept so that a job which
	// failed, was retried and then succeeded stops counting as a failure;
	// DecrementPendingJobs removes an id from it and IncrementFailedJobs adds
	// one, both without duplicates.
	FailedJobIDs []string
	// Options is the callbacks and the settings.
	Options BatchOptions
	// CreatedAt is when it was dispatched. CancelledAt and FinishedAt are zero
	// until they happen.
	CreatedAt   time.Time
	CancelledAt time.Time
	FinishedAt  time.Time
}

// ProcessedJobs is how many jobs have succeeded.
//
// It is TotalJobs minus PendingJobs, and not "how many reported": a batch of
// three with one failure and two successes has processed two, and the third is
// still counted as owed.
func (b Batch) ProcessedJobs() int { return b.TotalJobs - b.PendingJobs }

// Progress is ProcessedJobs as a percentage of TotalJobs, 0 when the batch is
// empty.
//
// Rounded to the nearest whole percent.
func (b Batch) Progress() int {
	if b.TotalJobs <= 0 {
		return 0
	}
	return int(math.Round(float64(b.ProcessedJobs()) / float64(b.TotalJobs) * 100))
}

// Finished reports whether the batch has stopped, successfully or not.
func (b Batch) Finished() bool { return !b.FinishedAt.IsZero() }

// Cancelled reports whether the batch was cancelled.
//
// Cancelling also finishes a batch, because Cancel stamps both columns: the
// jobs already in the queue still get delivered and still report, but nothing
// they could report would change the outcome.
func (b Batch) Cancelled() bool { return !b.CancelledAt.IsZero() }

// Canceled is Cancelled.
//
// Both spellings are declared. The collection writes Cancelled with two Ls
// everywhere else; this one is the American spelling, kept so that either
// spelling compiles.
func (b Batch) Canceled() bool { return b.Cancelled() }

// HasFailures reports whether any job in the batch failed.
func (b Batch) HasFailures() bool { return b.FailedJobs > 0 }

// AllowsFailures reports whether a failing job leaves the rest of the batch
// running.
func (b Batch) AllowsFailures() bool { return b.Options.AllowFailures }

// HasBeforeCallbacks, HasProgressCallbacks, HasThenCallbacks, HasCatchCallbacks,
// HasFailureCallbacks and HasFinallyCallbacks report whether the batch names a
// job for that moment. A Step with no Name is the absence of one.
func (b Batch) HasBeforeCallbacks() bool { return b.Options.Before.declared() }

// HasProgressCallbacks reports whether a job runs after every report.
func (b Batch) HasProgressCallbacks() bool { return b.Options.Progress.declared() }

// HasThenCallbacks reports whether a job runs when every job succeeded.
func (b Batch) HasThenCallbacks() bool { return b.Options.Then.declared() }

// HasCatchCallbacks reports whether a job runs on the first failure.
func (b Batch) HasCatchCallbacks() bool { return b.Options.Catch.declared() }

// HasFailureCallbacks reports whether a job runs on every allowed failure.
func (b Batch) HasFailureCallbacks() bool { return b.Options.Failure.declared() }

// HasFinallyCallbacks reports whether a job runs when the batch stops.
func (b Batch) HasFinallyCallbacks() bool { return b.Options.Finally.declared() }

// Fresh re-reads the batch.
//
// A Batch in hand is a snapshot: between the read and the decision, fifty
// workers may have reported.
func (b Batch) Fresh(ctx context.Context, g auth.Grant, r BatchRepository) (Batch, error) {
	if r == nil {
		return Batch{}, fmt.Errorf("bus: reading batch %s again needs a repository", b.ID)
	}
	return r.Find(ctx, g, b.ID)
}

// Add puts more jobs in a batch that is already running, and returns it fresh.
//
// The counter is raised before the jobs are pushed, because a job that arrives
// against a counter that has not been raised can take the batch to zero and
// fire Then while the rest of the work is still being queued.
func (b Batch) Add(ctx context.Context, g auth.Grant, r BatchRepository, q Queue, jobs ...Step) (Batch, error) {
	if len(jobs) == 0 {
		return b, nil
	}
	if r == nil || q == nil {
		return Batch{}, fmt.Errorf("bus: adding jobs to batch %s needs a repository and a queue", b.ID)
	}
	for _, j := range jobs {
		if !j.declared() {
			return Batch{}, ErrNoName
		}
	}

	if err := r.IncrementTotalJobs(ctx, g, b.ID, len(jobs)); err != nil {
		return Batch{}, err
	}
	for _, j := range jobs {
		jobID, err := database.NewID()
		if err != nil {
			return Batch{}, err
		}
		payload, err := wrap(envelope{Bus: formatVersion, Batch: b.ID, Job: jobID, Body: j.Payload})
		if err != nil {
			return Batch{}, err
		}
		if err := q.Push(ctx, g, b.queueFor(j), j.Name, payload); err != nil {
			return Batch{}, fmt.Errorf("bus: pushing %s of batch %s: %w", j.Name, b.ID, err)
		}
	}
	return b.Fresh(ctx, g, r)
}

// Cancel stops the batch: the jobs still queued are delivered, ask Batching,
// and skip their work.
//
// Cancelling a batch that is already cancelled or already finished changes
// nothing and is not an error: the caller wanted it stopped and it is.
func (b Batch) Cancel(ctx context.Context, g auth.Grant, r BatchRepository) error {
	if r == nil {
		return fmt.Errorf("bus: cancelling batch %s needs a repository", b.ID)
	}
	return r.Cancel(ctx, g, b.ID)
}

// Delete removes the batch row.
func (b Batch) Delete(ctx context.Context, g auth.Grant, r BatchRepository) error {
	if r == nil {
		return fmt.Errorf("bus: deleting batch %s needs a repository", b.ID)
	}
	return r.Delete(ctx, g, b.ID)
}

// DecrementPendingJobs records that one job of the batch succeeded and reports
// the counters as they stand after it.
func (b Batch) DecrementPendingJobs(ctx context.Context, g auth.Grant, r BatchRepository, jobID string) (UpdatedBatchJobCounts, error) {
	if r == nil {
		return UpdatedBatchJobCounts{}, fmt.Errorf("bus: reporting a job of batch %s needs a repository", b.ID)
	}
	return r.DecrementPendingJobs(ctx, g, b.ID, jobID)
}

// IncrementFailedJobs records that one job of the batch failed and reports the
// counters as they stand after it.
func (b Batch) IncrementFailedJobs(ctx context.Context, g auth.Grant, r BatchRepository, jobID string) (UpdatedBatchJobCounts, error) {
	if r == nil {
		return UpdatedBatchJobCounts{}, fmt.Errorf("bus: reporting a job of batch %s needs a repository", b.ID)
	}
	return r.IncrementFailedJobs(ctx, g, b.ID, jobID)
}

// RecordSuccessfulJob registers that one job finished, and pushes whatever that
// triggers.
//
// The order is decrement, run Progress, mark the batch finished and run Then
// when nothing is pending, run Finally when every job has reported exactly
// once.
//
// Each fires once because the repository decides, not this method: only one
// caller can be the one that took PendingJobs to zero, and the repository is
// the only thing that saw the counter before and after.
func (b Batch) RecordSuccessfulJob(ctx context.Context, g auth.Grant, r BatchRepository, q Queue, jobID string) error {
	counts, err := b.DecrementPendingJobs(ctx, g, r, jobID)
	if err != nil {
		return err
	}
	after := counts.Batch

	var errs []error
	if after.HasProgressCallbacks() {
		errs = append(errs, push(ctx, g, q, after, after.Options.Progress))
	}
	if counts.PendingJobs == 0 {
		if err := r.MarkAsFinished(ctx, g, b.ID); err != nil {
			errs = append(errs, err)
		}
		if after.HasThenCallbacks() {
			errs = append(errs, push(ctx, g, q, after, after.Options.Then))
		}
	}
	if counts.AllJobsHaveRanExactlyOnce() && after.HasFinallyCallbacks() {
		errs = append(errs, push(ctx, g, q, after, after.Options.Finally))
	}
	return errors.Join(errs...)
}

// RecordFailedJob registers that one job failed, and pushes whatever that
// triggers.
//
// The first failure of a batch that does not allow them cancels it, which is
// what stops the jobs still queued; Catch fires on the first failure either
// way, and Failure fires on every failure of a batch that allows them.
func (b Batch) RecordFailedJob(ctx context.Context, g auth.Grant, r BatchRepository, q Queue, jobID string, cause error) error {
	counts, err := b.IncrementFailedJobs(ctx, g, r, jobID)
	if err != nil {
		return err
	}
	after := counts.Batch

	var errs []error
	if counts.FailedJobs == 1 && !after.AllowsFailures() {
		if err := r.Cancel(ctx, g, b.ID); err != nil {
			errs = append(errs, err)
		}
	}
	if after.AllowsFailures() {
		if after.HasProgressCallbacks() {
			errs = append(errs, push(ctx, g, q, after, after.Options.Progress))
		}
		if after.HasFailureCallbacks() {
			errs = append(errs, push(ctx, g, q, after, after.Options.Failure))
		}
	}
	if counts.FailedJobs == 1 && after.HasCatchCallbacks() {
		errs = append(errs, push(ctx, g, q, after, after.Options.Catch))
	}
	if counts.AllJobsHaveRanExactlyOnce() && after.HasFinallyCallbacks() {
		errs = append(errs, push(ctx, g, q, after, after.Options.Finally))
	}
	return errors.Join(errs...)
}

// ToArray is the batch as a map, for a dashboard or an API response.
//
// The keys are snake_case, which is the case the rest of the collection puts on
// the wire.
func (b Batch) ToArray() map[string]any {
	out := map[string]any{
		"id":             b.ID,
		"name":           b.Name,
		"total_jobs":     b.TotalJobs,
		"pending_jobs":   b.PendingJobs,
		"processed_jobs": b.ProcessedJobs(),
		"progress":       b.Progress(),
		"failed_jobs":    b.FailedJobs,
		"failed_job_ids": append([]string(nil), b.FailedJobIDs...),
		"options":        b.Options,
		"created_at":     b.CreatedAt,
	}
	if b.Cancelled() {
		out["cancelled_at"] = b.CancelledAt
	} else {
		out["cancelled_at"] = nil
	}
	if b.Finished() {
		out["finished_at"] = b.FinishedAt
	} else {
		out["finished_at"] = nil
	}
	return out
}

// queueFor is where a step goes: its own queue, or the batch's.
func (b Batch) queueFor(s Step) string {
	if s.Queue != "" {
		return s.Queue
	}
	return b.Options.Queue
}

// UpdatedBatchJobCounts is what one report did to a batch's counters.
//
// The batch is carried alongside the two counters: the repository hands back
// the state it just wrote, because it is the only thing that saw the counters
// move and a second read would see fifty other workers' reports as well.
type UpdatedBatchJobCounts struct {
	// PendingJobs is how many jobs have not succeeded yet.
	PendingJobs int
	// FailedJobs is how many failed.
	FailedJobs int
	// Batch is the batch as it stands after this report.
	Batch Batch
}

// AllJobsHaveRanExactlyOnce reports whether every job in the batch has come
// back, whichever way it went.
//
// It is PendingJobs minus FailedJobs reaching zero: a success decrements
// PendingJobs and a failure does not, so the jobs still owed are exactly the
// ones that neither succeeded nor failed.
func (c UpdatedBatchJobCounts) AllJobsHaveRanExactlyOnce() bool {
	return c.PendingJobs-c.FailedJobs == 0
}

// withFailure adds a job id to the failed list, without duplicates, and returns
// the list sorted so that two workers writing at once cannot produce two
// different encodings of the same set.
func withFailure(ids []string, jobID string) []string {
	if jobID == "" || slices.Contains(ids, jobID) {
		return ids
	}
	out := append(append([]string(nil), ids...), jobID)
	slices.Sort(out)
	return out
}

// withoutFailure drops a job id from the failed list. It is what makes a job
// that failed, was retried and then succeeded stop counting as a failure.
func withoutFailure(ids []string, jobID string) []string {
	if jobID == "" {
		return ids
	}
	out := slices.DeleteFunc(append([]string(nil), ids...), func(v string) bool { return v == jobID })
	return out
}
