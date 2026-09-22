package bus

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/arandu-io/hesape/auth"
	busevents "github.com/arandu-io/hesape/bus/events"
	"github.com/arandu-io/hesape/database"
	"github.com/arandu-io/hesape/database/migrations"
	"github.com/arandu-io/hesape/database/schema"
)

// BatchesTable is where batches are stored.
const BatchesTable = "job_batches"

// DatabaseBatchRepository is the BatchRepository over the application's own
// database.
//
// It is the one an application runs. A batch counter is shared state between
// every worker on every replica, so the only place it can live is the place
// they all already talk to -- and putting it in the application's database
// means a batch dispatched inside database.Transaction is committed by the same
// transaction as the rows it is about.
type DatabaseBatchRepository struct {
	db      *database.DB
	factory BatchFactory
	events  EventRecorder
	now     func() time.Time
}

// DatabaseOption configures the repository at construction.
type DatabaseOption func(*DatabaseBatchRepository)

// WithEvents records batch.dispatched, batch.cancelled and batch.finished into
// r.
//
// The recorder sits on the repository rather than on the dispatcher because the
// repository is the one collaborator both sides share: the request that
// dispatches a batch and the worker that finishes it are different processes,
// and only one of them would ever see a recorder wired anywhere else.
func WithEvents(r EventRecorder) DatabaseOption {
	return func(d *DatabaseBatchRepository) { d.events = r }
}

// NewDatabaseBatchRepository returns the repository.
func NewDatabaseBatchRepository(db *database.DB, opts ...DatabaseOption) *DatabaseBatchRepository {
	d := &DatabaseBatchRepository{db: db, now: func() time.Time { return time.Now().UTC() }}
	for _, o := range opts {
		o(d)
	}
	return d
}

var _ PrunableBatchRepository = (*DatabaseBatchRepository)(nil)

// CreateJobBatchesTable creates the table DatabaseBatchRepository reads and
// writes.
//
// It lives here rather than in the application, because a table the framework
// reads and writes is a table the framework has to be able to create -- the
// events outbox sets the same precedent.
type CreateJobBatchesTable struct{ migrations.BaseMigration }

// GetName returns the migration's name.
func (CreateJobBatchesTable) GetName() string {
	return "2026_08_10_000001_create_job_batches_table"
}

// Up creates the batches table and the index every read of it uses.
//
// INTEGER and TIMESTAMP are spelled the same way by all three engines, and
// anything that takes part in a key is database.KeyText -- see there for why
// TEXT is not portable in one.
//
// options is one TEXT column of JSON rather than a column per setting: nothing
// queries a batch by the name of its Catch job and never will, they are read
// together and written once, and a column per field would be a migration the
// first time a callback grows an option.
func (CreateJobBatchesTable) Up(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().Create(ctx, BatchesTable, func(table *schema.Blueprint) {
		// String rather than Text for the keyed columns. That used to be spelled
		// database.KeyText here, which is the same rule under another name: the
		// Blueprint knows which type each engine wants for a column an index
		// reaches.
		table.String("id").Primary()
		table.String("tenant_id")
		table.Text("name")
		table.BigInteger("total_jobs")
		table.BigInteger("pending_jobs")
		table.BigInteger("failed_jobs")
		table.Text("failed_job_ids")
		table.Text("options")
		table.Timestamp("created_at")
		table.Timestamp("cancelled_at").Nullable()
		table.Timestamp("finished_at").Nullable()

		table.Index([]string{"tenant_id", "created_at"}, "job_batches_tenant_created_idx")
	})
}

// Down drops the batches table, and the index with it.
func (CreateJobBatchesTable) Down(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().DropIfExists(ctx, BatchesTable)
}

// Migrations is the schema this repository needs.
func Migrations() []migrations.Migration {
	return []migrations.Migration{CreateJobBatchesTable{}}
}

// columns is the read shape, in the order scanBatch expects.
const columns = `id, tenant_id, name, total_jobs, pending_jobs, failed_jobs,
	failed_job_ids, options, created_at, cancelled_at, finished_at`

// Get returns batches newest first.
//
// database.NewID is a version 4 uuid and is not ordered, so paging cannot be
// `id < before`: the order is created_at with the id breaking the tie, and the
// index the migration declares is on exactly that pair.
//
// before is still a batch id. Its timestamp is read first, which is one extra
// statement per page and the price of an unordered id.
func (d *DatabaseBatchRepository) Get(ctx context.Context, g auth.Grant, limit int, before string) ([]Batch, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return nil, err
	}
	if limit <= 0 {
		limit = 50
	}

	query := `SELECT ` + columns + ` FROM ` + BatchesTable + ` WHERE tenant_id = ?`
	args := []any{tenant}
	if before != "" {
		cursor, err := d.find(ctx, tenant, before)
		if errors.Is(err, database.ErrNotFound) {
			// A page after a batch that has been pruned is an empty page, not
			// an error: the caller is holding a cursor the sweep took.
			return nil, nil
		}
		if err != nil {
			return nil, err
		}
		query += ` AND (created_at < ? OR (created_at = ? AND id < ?))`
		args = append(args, cursor.CreatedAt.UTC(), cursor.CreatedAt.UTC(), before)
	}
	query += ` ORDER BY created_at DESC, id DESC LIMIT ?`
	args = append(args, limit)

	rows, err := d.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("bus: listing batches: %w", err)
	}
	defer rows.Close()

	var out []Batch
	for rows.Next() {
		b, err := scanBatch(d.factory, rows)
		if err != nil {
			return nil, fmt.Errorf("bus: listing batches: %w", err)
		}
		out = append(out, b)
	}
	if err := rows.Err(); err != nil {
		return nil, fmt.Errorf("bus: listing batches: %w", err)
	}
	return out, nil
}

// Find returns a batch.
func (d *DatabaseBatchRepository) Find(ctx context.Context, g auth.Grant, batchID string) (Batch, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return Batch{}, err
	}
	return d.find(ctx, tenant, batchID)
}

func (d *DatabaseBatchRepository) find(ctx context.Context, tenant, id string) (Batch, error) {
	row := d.db.QueryRowContext(ctx,
		`SELECT `+columns+` FROM `+BatchesTable+` WHERE id = ? AND tenant_id = ?`, id, tenant)

	b, err := scanBatch(d.factory, row)
	if errors.Is(err, sql.ErrNoRows) {
		return Batch{}, notFound(id)
	}
	if err != nil {
		return Batch{}, fmt.Errorf("bus: reading batch %s: %w", id, err)
	}
	return b, nil
}

// Store writes a described batch and returns it.
func (d *DatabaseBatchRepository) Store(ctx context.Context, g auth.Grant, p *PendingBatch) (Batch, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return Batch{}, err
	}
	id, err := database.NewID()
	if err != nil {
		return Batch{}, err
	}

	b := d.factory.Make(id, p.name, tenant, len(p.jobs), len(p.jobs), 0, nil, p.options, d.now(), time.Time{}, time.Time{})

	raw, err := json.Marshal(b.Options)
	if err != nil {
		return Batch{}, fmt.Errorf("bus: serializing the options of batch %s: %w", b.ID, err)
	}
	_, err = d.db.ExecContext(ctx, `INSERT INTO `+BatchesTable+` (`+columns+`)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		b.ID, tenant, b.Name, b.TotalJobs, b.PendingJobs, b.FailedJobs,
		"[]", string(raw), b.CreatedAt.UTC(), nil, nil)
	if err != nil {
		return Batch{}, fmt.Errorf("bus: storing batch %s: %w", b.ID, err)
	}

	if p.events != nil {
		record(p.events, busevents.BatchDispatched, b)
	} else {
		record(d.events, busevents.BatchDispatched, b)
	}
	return b, nil
}

// IncrementTotalJobs makes room in a batch for jobs added after it started.
func (d *DatabaseBatchRepository) IncrementTotalJobs(ctx context.Context, g auth.Grant, batchID string, amount int) error {
	tenant, err := tenantOf(g)
	if err != nil {
		return err
	}
	if amount <= 0 {
		return nil
	}

	res, err := d.db.ExecContext(ctx, `UPDATE `+BatchesTable+`
		SET total_jobs = total_jobs + ?, pending_jobs = pending_jobs + ?, finished_at = NULL
		WHERE id = ? AND tenant_id = ?`, amount, amount, batchID, tenant)
	if err != nil {
		return fmt.Errorf("bus: adding %d jobs to batch %s: %w", amount, batchID, err)
	}
	if n, err := res.RowsAffected(); err == nil && n == 0 {
		if _, err := d.find(ctx, tenant, batchID); err != nil {
			return err
		}
	}
	return nil
}

// recordAttempts is how many times a counter update may lose its
// compare-and-set without anything having moved before it gives up.
//
// It does not bound the loop as a whole. A lost race means another worker wrote
// its own report in between, which is the system making progress, and counting
// that as a failure caps a batch at recordAttempts concurrent workers: fifty
// reporting at once exhausted a budget of ten every time. What the budget
// catches is the spin -- the UPDATE matching nothing while the row keeps
// reading back the same counters, which no amount of retrying resolves.
const recordAttempts = 10

// DecrementPendingJobs records one job that succeeded.
func (d *DatabaseBatchRepository) DecrementPendingJobs(ctx context.Context, g auth.Grant, batchID, jobID string) (UpdatedBatchJobCounts, error) {
	return d.update(ctx, g, batchID, func(b Batch) Batch {
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
func (d *DatabaseBatchRepository) IncrementFailedJobs(ctx context.Context, g auth.Grant, batchID, jobID string) (UpdatedBatchJobCounts, error) {
	return d.update(ctx, g, batchID, func(b Batch) Batch {
		if jobID == "" {
			b.FailedJobs++
			return b
		}
		b.FailedJobIDs = withFailure(b.FailedJobIDs, jobID)
		b.FailedJobs = len(b.FailedJobIDs)
		return b
	})
}

// update moves a batch's counters: read, decide, then write conditioned on the
// counters not having moved.
//
// The alternative -- UPDATE ... SET pending_jobs = pending_jobs - 1 followed by
// a read -- cannot say which caller was the one that reached zero, and firing
// the Then callback of a ten thousand job import twice is worse than a retry
// loop.
//
// It deliberately does not open a transaction. Inside one, every retry would
// re-read the same snapshot and the loop would spin to its limit; the
// conditional UPDATE is the whole guarantee and it is a single statement.
func (d *DatabaseBatchRepository) update(ctx context.Context, g auth.Grant, batchID string, apply func(Batch) Batch) (UpdatedBatchJobCounts, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return UpdatedBatchJobCounts{}, err
	}

	// stalled counts the consecutive attempts that lost the race without the
	// row having moved. Any attempt that reads different counters from the last
	// one resets it: somebody else recorded their job, and this one is queued
	// behind real work rather than spinning.
	stalled := 0
	lastPending, lastFailed := -1, -1
	for {
		if err := ctx.Err(); err != nil {
			return UpdatedBatchJobCounts{}, err
		}

		before, err := d.find(ctx, tenant, batchID)
		if err != nil {
			return UpdatedBatchJobCounts{}, err
		}
		if before.PendingJobs != lastPending || before.FailedJobs != lastFailed {
			lastPending, lastFailed, stalled = before.PendingJobs, before.FailedJobs, 0
		}
		after := apply(before)
		counts := UpdatedBatchJobCounts{PendingJobs: after.PendingJobs, FailedJobs: after.FailedJobs, Batch: after}

		// A report that changes nothing writes nothing. It happens on a
		// duplicate delivery of a job that already reported, and it has to be
		// short-circuited rather than sent: MySQL counts rows changed, not rows
		// matched, so an UPDATE that sets the values already there affects zero
		// rows and the loop below would read that as contention and spin to its
		// limit.
		if after.PendingJobs == before.PendingJobs && after.FailedJobs == before.FailedJobs {
			return counts, nil
		}

		ids, err := json.Marshal(after.FailedJobIDs)
		if err != nil {
			return UpdatedBatchJobCounts{}, fmt.Errorf("bus: recording a job of batch %s: %w", batchID, err)
		}
		res, err := d.db.ExecContext(ctx, `UPDATE `+BatchesTable+`
			SET pending_jobs = ?, failed_jobs = ?, failed_job_ids = ?
			WHERE id = ? AND tenant_id = ? AND pending_jobs = ? AND failed_jobs = ?`,
			after.PendingJobs, after.FailedJobs, string(ids),
			batchID, tenant, before.PendingJobs, before.FailedJobs)
		if err != nil {
			return UpdatedBatchJobCounts{}, fmt.Errorf("bus: recording a job of batch %s: %w", batchID, err)
		}
		n, err := res.RowsAffected()
		if err != nil {
			return UpdatedBatchJobCounts{}, fmt.Errorf("bus: recording a job of batch %s: %w", batchID, err)
		}
		if n > 0 {
			return counts, nil
		}

		stalled++
		if stalled >= recordAttempts {
			return UpdatedBatchJobCounts{}, fmt.Errorf("bus: batch %s did not move under %d attempts to record a job against it", batchID, recordAttempts)
		}
	}
}

// MarkAsFinished stamps the batch finished.
func (d *DatabaseBatchRepository) MarkAsFinished(ctx context.Context, g auth.Grant, batchID string) error {
	tenant, err := tenantOf(g)
	if err != nil {
		return err
	}

	res, err := d.db.ExecContext(ctx, `UPDATE `+BatchesTable+`
		SET finished_at = ?
		WHERE id = ? AND tenant_id = ? AND finished_at IS NULL`, d.now(), batchID, tenant)
	if err != nil {
		return fmt.Errorf("bus: finishing batch %s: %w", batchID, err)
	}
	if n, err := res.RowsAffected(); err == nil && n == 0 {
		// Nothing updated means the batch is already finished, or does not
		// exist. Only the second is an error, and the read settles which.
		if _, err := d.find(ctx, tenant, batchID); err != nil {
			return err
		}
		return nil
	}
	if b, err := d.find(ctx, tenant, batchID); err == nil {
		record(d.events, busevents.BatchFinished, b)
	}
	return nil
}

// Cancel marks the batch cancelled and finished.
func (d *DatabaseBatchRepository) Cancel(ctx context.Context, g auth.Grant, batchID string) error {
	tenant, err := tenantOf(g)
	if err != nil {
		return err
	}

	now := d.now()
	res, err := d.db.ExecContext(ctx, `UPDATE `+BatchesTable+`
		SET cancelled_at = ?, finished_at = ?
		WHERE id = ? AND tenant_id = ? AND cancelled_at IS NULL`,
		now, now, batchID, tenant)
	if err != nil {
		return fmt.Errorf("bus: cancelling batch %s: %w", batchID, err)
	}

	// Nothing updated means one of two things, and only one of them is an
	// error: the batch does not exist. Already cancelled means the caller got
	// what they asked for. The read settles which, and it only runs on the path
	// that changed nothing.
	if n, err := res.RowsAffected(); err == nil && n == 0 {
		if _, err := d.find(ctx, tenant, batchID); err != nil {
			return err
		}
		return nil
	}
	if b, err := d.find(ctx, tenant, batchID); err == nil {
		record(d.events, busevents.BatchCancelled, b)
	}
	return nil
}

// Delete removes the batch row.
func (d *DatabaseBatchRepository) Delete(ctx context.Context, g auth.Grant, batchID string) error {
	tenant, err := tenantOf(g)
	if err != nil {
		return err
	}
	if _, err := d.db.ExecContext(ctx,
		`DELETE FROM `+BatchesTable+` WHERE id = ? AND tenant_id = ?`, batchID, tenant); err != nil {
		return fmt.Errorf("bus: deleting batch %s: %w", batchID, err)
	}
	return nil
}

// Transaction runs fn inside a database transaction.
func (d *DatabaseBatchRepository) Transaction(ctx context.Context, fn func(ctx context.Context) error) error {
	return database.Transaction(ctx, d.db, fn)
}

// RollBack does nothing.
//
// A transaction here is the scope of Transaction's closure: returning an error
// from it is what rolls back, and there is nothing to reach in from outside.
func (d *DatabaseBatchRepository) RollBack(context.Context) error { return nil }

// Prune deletes finished batches created before the cut.
func (d *DatabaseBatchRepository) Prune(ctx context.Context, g auth.Grant, before time.Time) (int, error) {
	return d.prune(ctx, g, `finished_at IS NOT NULL AND created_at < ?`, before)
}

// PruneUnfinished deletes batches created before the cut that never finished.
func (d *DatabaseBatchRepository) PruneUnfinished(ctx context.Context, g auth.Grant, before time.Time) (int, error) {
	return d.prune(ctx, g, `finished_at IS NULL AND created_at < ?`, before)
}

// PruneCancelled deletes cancelled batches created before the cut.
func (d *DatabaseBatchRepository) PruneCancelled(ctx context.Context, g auth.Grant, before time.Time) (int, error) {
	return d.prune(ctx, g, `cancelled_at IS NOT NULL AND created_at < ?`, before)
}

func (d *DatabaseBatchRepository) prune(ctx context.Context, g auth.Grant, where string, before time.Time) (int, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return 0, err
	}

	res, err := d.db.ExecContext(ctx,
		`DELETE FROM `+BatchesTable+` WHERE tenant_id = ? AND `+where, tenant, before.UTC())
	if err != nil {
		return 0, fmt.Errorf("bus: pruning batches: %w", err)
	}
	n, err := res.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("bus: pruning batches: %w", err)
	}
	return int(n), nil
}

// GetConnection is the database handle this repository reads and writes.
func (d *DatabaseBatchRepository) GetConnection() *database.DB { return d.db }

// SetConnection points the repository at another handle.
//
// It is for the test that has to swap the connection under a repository it
// already built.
func (d *DatabaseBatchRepository) SetConnection(db *database.DB) { d.db = db }

// scanner is what both *sql.Row and *sql.Rows are.
type scanner interface{ Scan(dest ...any) error }

// scanBatch reads one row in the order of columns.
func scanBatch(f BatchFactory, row scanner) (Batch, error) {
	var (
		id, tenant, name       string
		total, pending, failed int
		rawIDs, rawOptions     string
		created                time.Time
		cancelled, finished    sql.NullTime
		failedIDs              []string
		options                BatchOptions
	)
	err := row.Scan(&id, &tenant, &name, &total, &pending, &failed,
		&rawIDs, &rawOptions, &created, &cancelled, &finished)
	if err != nil {
		return Batch{}, err
	}
	if err := json.Unmarshal([]byte(rawIDs), &failedIDs); err != nil {
		return Batch{}, fmt.Errorf("reading the failed jobs of batch %s: %w", id, err)
	}
	if err := json.Unmarshal([]byte(rawOptions), &options); err != nil {
		return Batch{}, fmt.Errorf("reading the options of batch %s: %w", id, err)
	}

	var cancelledAt, finishedAt time.Time
	if cancelled.Valid {
		cancelledAt = cancelled.Time.UTC()
	}
	if finished.Valid {
		finishedAt = finished.Time.UTC()
	}
	return f.Make(id, name, tenant, total, pending, failed, failedIDs, options,
		created.UTC(), cancelledAt, finishedAt), nil
}
