package bus

import (
	"context"
	"slices"
	"sync"
	"time"

	"github.com/arandu-io/hesape/auth"
	busevents "github.com/arandu-io/hesape/bus/events"
	"github.com/arandu-io/hesape/database"
)

// Memory is a BatchRepository in process memory.
//
// It is the driver and the test double in one type, for the reason the queue
// gives for its own: a fake that is not the thing gets the interesting cases
// wrong, and the interesting cases here are all about ordering -- two failures
// at the same instant, a report arriving after the batch already stopped.
// Memory serializes exactly like DatabaseBatchRepository does, so a test that
// passes against it is a test about the rule and not about the mutex.
//
// It is not a second way to run batches in production. Counters in one
// process's heap are lost with the process and invisible to the replica next to
// it, so a batch dispatched here is a batch only this binary can finish.
//
// It is a real implementation and not a stub: the counters move, so a test
// against it exercises the same rules a database does.
type Memory struct {
	mu      sync.Mutex
	batches map[string]Batch // keyed by tenant and id
	factory BatchFactory
	events  EventRecorder
	now     func() time.Time
}

// NewMemory returns an empty repository.
func NewMemory() *Memory {
	return &Memory{batches: map[string]Batch{}, now: func() time.Time { return time.Now().UTC() }}
}

var _ PrunableBatchRepository = (*Memory)(nil)

// key namespaces a batch by tenant, so two tenants holding the same id -- which
// nothing forbids, because an id is generated per tenant's work -- do not see
// each other's counters.
func key(tenant, id string) string { return tenant + "\x00" + id }

// Get returns batches newest first.
func (m *Memory) Get(_ context.Context, g auth.Grant, limit int, before string) ([]Batch, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 50
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	cursor, paging := m.batches[key(tenant, before)]
	if before != "" && !paging {
		// A page after a batch that has been pruned is an empty page, for the
		// reason DatabaseBatchRepository.Get gives.
		return nil, nil
	}

	var out []Batch
	for _, b := range m.batches {
		if b.TenantID != tenant {
			continue
		}
		if paging && !newer(cursor, b) {
			continue
		}
		out = append(out, b)
	}
	slices.SortFunc(out, func(a, b Batch) int {
		if newer(a, b) {
			return 1
		}
		if newer(b, a) {
			return -1
		}
		return 0
	})
	if len(out) > limit {
		out = out[:limit]
	}
	return out, nil
}

// newer reports whether a sorts after b in the newest-first order Get uses:
// created_at, with the id breaking the tie.
func newer(a, b Batch) bool {
	if a.CreatedAt.Equal(b.CreatedAt) {
		return a.ID > b.ID
	}
	return a.CreatedAt.After(b.CreatedAt)
}

// Find returns a batch.
func (m *Memory) Find(_ context.Context, g auth.Grant, batchID string) (Batch, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return Batch{}, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	b, ok := m.batches[key(tenant, batchID)]
	if !ok {
		return Batch{}, notFound(batchID)
	}
	return b, nil
}

// Store writes a described batch and returns it.
func (m *Memory) Store(_ context.Context, g auth.Grant, p *PendingBatch) (Batch, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return Batch{}, err
	}
	id, err := database.NewID()
	if err != nil {
		return Batch{}, err
	}

	b := m.factory.Make(id, p.name, tenant, len(p.jobs), len(p.jobs), 0, nil, p.options,
		m.now(), time.Time{}, time.Time{})

	m.mu.Lock()
	m.batches[key(tenant, b.ID)] = b
	m.mu.Unlock()

	if p.events != nil {
		record(p.events, busevents.BatchDispatched, b)
	} else {
		record(m.events, busevents.BatchDispatched, b)
	}
	return b, nil
}

// IncrementTotalJobs makes room in a batch for jobs added after it started.
func (m *Memory) IncrementTotalJobs(_ context.Context, g auth.Grant, batchID string, amount int) error {
	tenant, err := tenantOf(g)
	if err != nil {
		return err
	}
	if amount <= 0 {
		return nil
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	b, ok := m.batches[key(tenant, batchID)]
	if !ok {
		return notFound(batchID)
	}
	b.TotalJobs += amount
	b.PendingJobs += amount
	b.FinishedAt = time.Time{}
	m.batches[key(tenant, batchID)] = b
	return nil
}

// DecrementPendingJobs records one job that succeeded.
func (m *Memory) DecrementPendingJobs(_ context.Context, g auth.Grant, batchID, jobID string) (UpdatedBatchJobCounts, error) {
	return m.update(g, batchID, func(b Batch) Batch {
		// Unconditionally: a duplicate delivery of the last job takes the
		// counter to -1 rather than leaving it at 0, and that is exactly what
		// stops Then from firing a second time.
		b.PendingJobs--
		b.FailedJobIDs = withoutFailure(b.FailedJobIDs, jobID)
		b.FailedJobs = len(b.FailedJobIDs)
		return b
	})
}

// IncrementFailedJobs records one job that failed.
func (m *Memory) IncrementFailedJobs(_ context.Context, g auth.Grant, batchID, jobID string) (UpdatedBatchJobCounts, error) {
	return m.update(g, batchID, func(b Batch) Batch {
		if jobID == "" {
			b.FailedJobs++
			return b
		}
		b.FailedJobIDs = withFailure(b.FailedJobIDs, jobID)
		b.FailedJobs = len(b.FailedJobIDs)
		return b
	})
}

func (m *Memory) update(g auth.Grant, batchID string, apply func(Batch) Batch) (UpdatedBatchJobCounts, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return UpdatedBatchJobCounts{}, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	b, ok := m.batches[key(tenant, batchID)]
	if !ok {
		return UpdatedBatchJobCounts{}, notFound(batchID)
	}
	b = apply(b)
	m.batches[key(tenant, batchID)] = b
	return UpdatedBatchJobCounts{PendingJobs: b.PendingJobs, FailedJobs: b.FailedJobs, Batch: b}, nil
}

// MarkAsFinished stamps the batch finished.
func (m *Memory) MarkAsFinished(_ context.Context, g auth.Grant, batchID string) error {
	tenant, err := tenantOf(g)
	if err != nil {
		return err
	}

	m.mu.Lock()
	b, ok := m.batches[key(tenant, batchID)]
	if !ok {
		m.mu.Unlock()
		return notFound(batchID)
	}
	if b.Finished() {
		m.mu.Unlock()
		return nil
	}
	b.FinishedAt = m.now()
	m.batches[key(tenant, batchID)] = b
	m.mu.Unlock()

	record(m.events, busevents.BatchFinished, b)
	return nil
}

// Cancel marks the batch cancelled and finished.
func (m *Memory) Cancel(_ context.Context, g auth.Grant, batchID string) error {
	tenant, err := tenantOf(g)
	if err != nil {
		return err
	}

	m.mu.Lock()
	b, ok := m.batches[key(tenant, batchID)]
	if !ok {
		m.mu.Unlock()
		return notFound(batchID)
	}
	if b.Cancelled() {
		m.mu.Unlock()
		return nil
	}
	now := m.now()
	b.CancelledAt = now
	b.FinishedAt = now
	m.batches[key(tenant, batchID)] = b
	m.mu.Unlock()

	record(m.events, busevents.BatchCancelled, b)
	return nil
}

// Delete removes the batch.
func (m *Memory) Delete(_ context.Context, g auth.Grant, batchID string) error {
	tenant, err := tenantOf(g)
	if err != nil {
		return err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.batches, key(tenant, batchID))
	return nil
}

// Transaction runs fn. There is nothing to commit: a map is not a database, and
// pretending otherwise would make a test pass that a real store would fail.
func (m *Memory) Transaction(ctx context.Context, fn func(ctx context.Context) error) error {
	return fn(ctx)
}

// RollBack does nothing, for the reason DatabaseBatchRepository.RollBack gives.
func (m *Memory) RollBack(context.Context) error { return nil }

// Prune deletes finished batches created before the cut.
func (m *Memory) Prune(_ context.Context, g auth.Grant, before time.Time) (int, error) {
	return m.prune(g, before, func(b Batch) bool { return b.Finished() })
}

// PruneUnfinished deletes batches created before the cut that never finished.
func (m *Memory) PruneUnfinished(_ context.Context, g auth.Grant, before time.Time) (int, error) {
	return m.prune(g, before, func(b Batch) bool { return !b.Finished() })
}

// PruneCancelled deletes cancelled batches created before the cut.
func (m *Memory) PruneCancelled(_ context.Context, g auth.Grant, before time.Time) (int, error) {
	return m.prune(g, before, Batch.Cancelled)
}

func (m *Memory) prune(g auth.Grant, before time.Time, takes func(Batch) bool) (int, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return 0, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	n := 0
	for k, b := range m.batches {
		if b.TenantID != tenant || !takes(b) || !b.CreatedAt.Before(before) {
			continue
		}
		delete(m.batches, k)
		n++
	}
	return n, nil
}
