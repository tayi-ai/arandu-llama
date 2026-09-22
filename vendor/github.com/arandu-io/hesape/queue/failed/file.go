package failed

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database"
)

// FileFailedJobProvider keeps the failed jobs in a JSON file.
//
// It exists for the deployment that has a queue and no database: a single
// process on one machine, draining a RESP queue, that still wants a dead letter
// list to survive a restart.
//
// It holds the newest Limit failures and drops the rest, which is the right
// behaviour for a file: a dead letter list that grows without bound on a disk
// nobody is watching is an outage waiting to be filed as a disk-full alert.
//
// It is not for more than one process. The file is rewritten whole under a
// mutex this process holds, and two of them would lose each other's writes --
// which is exactly the case [DatabaseFailedJobProvider] is for.
type FileFailedJobProvider struct {
	mu    sync.Mutex
	path  string
	limit int
	// lastRemoved is how many records the last rewrite dropped. It is a field
	// rather than a second return value because rewrite has two callers that
	// each want a different half of the answer, and it is only read under the
	// same mutex that wrote it.
	lastRemoved int
}

// NewFileFailedJobProvider returns the provider over path.
//
// A limit of zero or less means a hundred.
func NewFileFailedJobProvider(path string, limit int) *FileFailedJobProvider {
	if limit <= 0 {
		limit = 100
	}
	return &FileFailedJobProvider{path: path, limit: limit}
}

var (
	_ FailedJobProvider          = (*FileFailedJobProvider)(nil)
	_ CountableFailedJobProvider = (*FileFailedJobProvider)(nil)
	_ PrunableFailedJobProvider  = (*FileFailedJobProvider)(nil)
)

// Log records a job that gave up, once. It answers log().
//
// The file has no key to refuse a duplicate with, so the check is here: a
// record already under this id is this failure, already listed.
func (p *FileFailedJobProvider) Log(ctx context.Context, g auth.Grant, job FailedJob) (string, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return "", err
	}

	id := job.UUID
	if id == "" {
		if id, err = database.NewID(); err != nil {
			return "", err
		}
	}
	job.ID, job.UUID, job.TenantID = id, id, tenant
	if job.FailedAt.IsZero() {
		job.FailedAt = time.Now()
	}
	job.FailedAt = job.FailedAt.UTC()

	p.mu.Lock()
	defer p.mu.Unlock()
	stored, err := p.read()
	if err != nil {
		return "", err
	}
	for _, existing := range stored {
		if existing.ID == id && existing.TenantID == tenant {
			return id, nil
		}
	}
	stored = append([]FailedJob{job}, stored...)
	if len(stored) > p.limit {
		stored = stored[:p.limit]
	}
	if err := p.write(stored); err != nil {
		return "", err
	}
	return id, nil
}

// IDs is the identifiers of this tenant's failed jobs, newest first. It answers
// ids().
func (p *FileFailedJobProvider) IDs(ctx context.Context, g auth.Grant, queue string) ([]string, error) {
	all, err := p.All(ctx, g)
	if err != nil {
		return nil, err
	}
	var out []string
	for _, job := range all {
		if queue == "" || job.Queue == queue {
			out = append(out, job.ID)
		}
	}
	return out, nil
}

// All is this tenant's failed jobs, newest first. It answers all().
func (p *FileFailedJobProvider) All(_ context.Context, g auth.Grant) ([]FailedJob, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return nil, err
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	stored, err := p.read()
	if err != nil {
		return nil, err
	}

	var out []FailedJob
	for _, job := range stored {
		if job.TenantID == tenant {
			out = append(out, job)
		}
	}
	sort.SliceStable(out, func(i, j int) bool { return out[i].FailedAt.After(out[j].FailedAt) })
	return out, nil
}

// Find is one of this tenant's failed jobs. It answers find().
func (p *FileFailedJobProvider) Find(ctx context.Context, g auth.Grant, id string) (FailedJob, error) {
	all, err := p.All(ctx, g)
	if err != nil {
		return FailedJob{}, err
	}
	for _, job := range all {
		if job.ID == id {
			return job, nil
		}
	}
	return FailedJob{}, fmt.Errorf("%w: %s", ErrNotFound, id)
}

// Forget removes one of this tenant's failed jobs. It answers forget().
func (p *FileFailedJobProvider) Forget(_ context.Context, g auth.Grant, id string) (bool, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return false, err
	}
	return p.rewrite(func(job FailedJob) bool {
		return job.TenantID == tenant && job.ID == id
	})
}

// Flush removes this tenant's failed jobs older than age, or all of them when
// age is zero. It answers flush().
func (p *FileFailedJobProvider) Flush(ctx context.Context, g auth.Grant, age time.Duration) error {
	before := time.Now().UTC()
	if age > 0 {
		before = before.Add(-age)
	}
	// A zero age is "everything", and an instant in the future says that
	// without a second branch: nothing failed after now.
	_, err := p.Prune(ctx, g, before.Add(time.Nanosecond))
	return err
}

// Prune removes this tenant's failed jobs that failed before an instant, and
// returns how many went. It answers prune().
func (p *FileFailedJobProvider) Prune(_ context.Context, g auth.Grant, before time.Time) (int, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return 0, err
	}
	if _, err := p.rewrite(func(job FailedJob) bool {
		return job.TenantID == tenant && job.FailedAt.Before(before.UTC())
	}); err != nil {
		return 0, err
	}

	p.mu.Lock()
	defer p.mu.Unlock()
	return p.lastRemoved, nil
}

// Count is how many of this tenant's jobs have failed. It answers count().
func (p *FileFailedJobProvider) Count(ctx context.Context, g auth.Grant, connectionName, queue string) (int, error) {
	all, err := p.All(ctx, g)
	if err != nil {
		return 0, err
	}
	count := 0
	for _, job := range all {
		if connectionName != "" && job.Connection != connectionName {
			continue
		}
		if queue != "" && job.Queue != queue {
			continue
		}
		count++
	}
	return count, nil
}

// rewrite drops every record the predicate claims, and reports whether any
// went.
func (p *FileFailedJobProvider) rewrite(drop func(FailedJob) bool) (bool, error) {
	p.mu.Lock()
	defer p.mu.Unlock()
	stored, err := p.read()
	if err != nil {
		return false, err
	}

	kept := make([]FailedJob, 0, len(stored))
	for _, job := range stored {
		if !drop(job) {
			kept = append(kept, job)
		}
	}
	p.lastRemoved = len(stored) - len(kept)
	if p.lastRemoved == 0 {
		return false, nil
	}
	return true, p.write(kept)
}

// read loads the file. A file that is not there is an empty list, not an error:
// nothing has failed yet.
func (p *FileFailedJobProvider) read() ([]FailedJob, error) {
	content, err := os.ReadFile(p.path)
	if os.IsNotExist(err) {
		return nil, nil
	}
	if err != nil {
		return nil, fmt.Errorf("queue/failed: reading %s: %w", p.path, err)
	}
	if len(content) == 0 {
		return nil, nil
	}

	var stored []FailedJob
	if err := json.Unmarshal(content, &stored); err != nil {
		return nil, fmt.Errorf("queue/failed: %s is not a failed job list: %w", p.path, err)
	}
	return stored, nil
}

// write replaces the file.
//
// Through a temporary file and a rename, so a process that dies mid-write
// leaves the previous list rather than half of the new one -- the whole reason
// to keep failures on disk is that they survive a crash.
func (p *FileFailedJobProvider) write(stored []FailedJob) error {
	content, err := json.Marshal(stored)
	if err != nil {
		return fmt.Errorf("queue/failed: encoding the failed job list: %w", err)
	}
	if err := os.MkdirAll(filepath.Dir(p.path), 0o755); err != nil {
		return fmt.Errorf("queue/failed: making room for %s: %w", p.path, err)
	}

	tmp := p.path + ".tmp"
	if err := os.WriteFile(tmp, content, 0o600); err != nil {
		return fmt.Errorf("queue/failed: writing %s: %w", p.path, err)
	}
	if err := os.Rename(tmp, p.path); err != nil {
		return fmt.Errorf("queue/failed: replacing %s: %w", p.path, err)
	}
	return nil
}
