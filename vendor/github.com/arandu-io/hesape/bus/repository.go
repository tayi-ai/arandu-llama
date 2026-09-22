package bus

import (
	"context"
	"errors"
	"fmt"
	"time"

	"github.com/arandu-io/hesape/auth"
	busevents "github.com/arandu-io/hesape/bus/events"
	"github.com/arandu-io/hesape/database"
	"github.com/arandu-io/hesape/events"
)

// BatchRepository is where batch counters live.
//
// There are two implementations in this package: DatabaseBatchRepository, which
// is the one an application runs, and Memory, which is the one a test runs.
//
// Every method takes a Grant and every statement filters by the tenant it
// carries, reads included -- a batch is a row about a customer's work, and
// there is no "but it is only a counter" exemption.
type BatchRepository interface {
	// Get returns batches newest first, at most limit of them, starting after
	// the id in before. An empty before starts at the newest.
	Get(ctx context.Context, g auth.Grant, limit int, before string) ([]Batch, error)

	// Find returns a batch, or an error wrapping database.ErrNotFound.
	Find(ctx context.Context, g auth.Grant, batchID string) (Batch, error)

	// Store persists a described batch and returns it with its id and timestamp
	// filled in. It is called once, by PendingBatch.Dispatch, before the first
	// job is pushed -- a job that arrives before its batch exists has nothing
	// to report to.
	Store(ctx context.Context, g auth.Grant, p *PendingBatch) (Batch, error)

	// IncrementTotalJobs raises TotalJobs and PendingJobs by amount and clears
	// FinishedAt, which is what lets Batch.Add put work into a batch that had
	// already reached zero.
	IncrementTotalJobs(ctx context.Context, g auth.Grant, batchID string, amount int) error

	// DecrementPendingJobs records one job that succeeded. jobID is dropped
	// from the failed list, so a job that failed and was then retried into
	// success stops counting as a failure.
	DecrementPendingJobs(ctx context.Context, g auth.Grant, batchID, jobID string) (UpdatedBatchJobCounts, error)

	// IncrementFailedJobs records one job that failed. jobID joins the failed
	// list, once however many times it is reported.
	IncrementFailedJobs(ctx context.Context, g auth.Grant, batchID, jobID string) (UpdatedBatchJobCounts, error)

	// MarkAsFinished stamps FinishedAt. Stamping a batch that is already
	// finished changes nothing.
	MarkAsFinished(ctx context.Context, g auth.Grant, batchID string) error

	// Cancel marks the batch cancelled and finished, so the jobs still queued
	// skip their work. Cancelling a batch that is already cancelled or already
	// finished changes nothing and is not an error.
	Cancel(ctx context.Context, g auth.Grant, batchID string) error

	// Delete removes the batch row.
	Delete(ctx context.Context, g auth.Grant, batchID string) error

	// Transaction runs fn inside whatever a transaction means to this store.
	Transaction(ctx context.Context, fn func(ctx context.Context) error) error

	// RollBack abandons the transaction in flight, for the failing batch
	// callback that runs inside the worker's transaction. It does nothing
	// outside one.
	RollBack(ctx context.Context) error
}

// PrunableBatchRepository is a BatchRepository that can be swept.
//
// A table nobody prunes is a table that only grows, and the batches worth
// keeping are not the same as the batches worth counting -- which is why there
// are three sweeps rather than one.
type PrunableBatchRepository interface {
	BatchRepository

	// Prune deletes finished batches created before the cut, and returns how
	// many went.
	Prune(ctx context.Context, g auth.Grant, before time.Time) (int, error)

	// PruneUnfinished deletes batches created before the cut that never
	// finished: the ones whose workers died holding them.
	PruneUnfinished(ctx context.Context, g auth.Grant, before time.Time) (int, error)

	// PruneCancelled deletes cancelled batches created before the cut.
	PruneCancelled(ctx context.Context, g auth.Grant, before time.Time) (int, error)
}

// BatchFactory builds a Batch from the columns a repository read.
//
// It is the one place that knows the order of the fields, which is worth having
// exactly once with two repositories reading the same rows.
type BatchFactory struct{}

// Make assembles a Batch.
func (BatchFactory) Make(id, name, tenantID string, totalJobs, pendingJobs, failedJobs int,
	failedJobIDs []string, options BatchOptions,
	createdAt, cancelledAt, finishedAt time.Time,
) Batch {
	return Batch{
		ID:           id,
		Name:         name,
		TenantID:     tenantID,
		TotalJobs:    totalJobs,
		PendingJobs:  pendingJobs,
		FailedJobs:   failedJobs,
		FailedJobIDs: failedJobIDs,
		Options:      options,
		CreatedAt:    createdAt,
		CancelledAt:  cancelledAt,
		FinishedAt:   finishedAt,
	}
}

// EventRecorder is where a repository reports what happened to a batch.
//
// It names the one method it needs rather than taking *events.Recorder, so a
// test can watch batch.dispatched, batch.cancelled and batch.finished without
// an outbox and a database behind them.
type EventRecorder interface {
	Record(e events.Event)
}

// ErrNoTenant is returned when a Grant carries no usable tenant.
//
// It is an error rather than a default. A batch with no tenant would be counted
// against every customer at once, and the callbacks would run under a Grant
// nobody issued.
var ErrNoTenant = errors.New("bus: the Grant carries no tenant, and a batch without one cannot be scoped")

// tenantOf reads the tenant a statement must filter by.
func tenantOf(g auth.Grant) (string, error) {
	tenant := auth.Tenant(g)
	if tenant == "" {
		return "", ErrNoTenant
	}
	if !auth.ValidTenant(tenant) {
		return "", fmt.Errorf("%w: %q cannot be one", ErrNoTenant, tenant)
	}
	return tenant, nil
}

// notFound is the one shape of "no such batch" in this package.
//
// It wraps database.ErrNotFound rather than introducing a sentinel of its own,
// so errors.Is keeps answering the question every caller actually asks -- and so
// an HTTP handler turns it into a 404 without learning a second name for the
// same thing.
func notFound(id string) error {
	return fmt.Errorf("%w: batch %s", database.ErrNotFound, id)
}

// record publishes one of the three batch events, when there is anywhere to
// publish it.
func record(rec EventRecorder, name string, b Batch) {
	if rec == nil {
		return
	}
	p := busevents.Payload{
		ID:          b.ID,
		Name:        b.Name,
		Tenant:      b.TenantID,
		TotalJobs:   b.TotalJobs,
		PendingJobs: b.PendingJobs,
		FailedJobs:  b.FailedJobs,
	}
	switch name {
	case busevents.BatchDispatched:
		rec.Record(busevents.NewBatchDispatched(p))
	case busevents.BatchCancelled:
		rec.Record(busevents.NewBatchCancelled(p))
	case busevents.BatchFinished:
		rec.Record(busevents.NewBatchFinished(p))
	}
}
