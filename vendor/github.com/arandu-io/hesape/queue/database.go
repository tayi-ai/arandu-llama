package queue

import (
	"context"
	"database/sql"
	"encoding/json"
	"errors"
	"fmt"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database"
	"github.com/arandu-io/hesape/database/migrations"
	"github.com/arandu-io/hesape/database/schema"
	"github.com/arandu-io/hesape/queue/jobs"
)

// DatabaseQueue is the queue backed by the application's own database.
//
// It is the default driver because it needs nothing installed: the jobs table
// sits in the database the application already has.
//
// What it offers that no other driver can is the outbox guarantee. A job pushed
// inside database.Transaction is committed by the same transaction as the row
// it is about, so it exists if and only if the write did -- the mechanism the
// events package uses for events, applied to work. The name is the driver and
// not the guarantee: naming it after the guarantee would hide it from everyone
// looking for the queue that lives in the application's database.
type DatabaseQueue struct {
	connection
	db *database.DB
}

// NewDatabaseQueue returns the queue over db.
func NewDatabaseQueue(db *database.DB) *DatabaseQueue {
	q := &DatabaseQueue{db: db}
	q.SetConnectionName(databaseConnection)
	return q
}

// GetDatabase is the connection this queue writes to.
//
// The queue and the application share it, which is what makes a job pushed
// inside database.Transaction part of that transaction.
func (q *DatabaseQueue) GetDatabase() *database.DB { return q.db }

// GetQueue resolves a queue name, turning empty into the default.
//
// It is exported because every method here starts with it and a driver in
// another module has the same first line.
func (q *DatabaseQueue) GetQueue(name string) string {
	if name == "" {
		return jobs.DefaultQueue
	}
	return name
}

var (
	_ Queue       = (*DatabaseQueue)(nil)
	_ jobs.Driver = (*DatabaseQueue)(nil)
)

// databaseConnection is what a popped job reports as its connection.
const databaseConnection = "database"

// CreateJobsTable creates the table DatabaseQueue pushes to and pops from.
type CreateJobsTable struct{ migrations.BaseMigration }

// GetName returns the migration's name.
func (CreateJobsTable) GetName() string { return "2026_07_31_000010_create_jobs_table" }

// Up creates the jobs table and the two indexes its queries are.
//
// Portable types only: TEXT, INTEGER and TIMESTAMP mean the same thing on
// SQLite, Postgres and MySQL.
func (CreateJobsTable) Up(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().Create(ctx, "jobs", func(table *schema.Blueprint) {
		table.String("id").Primary()
		// queue is indexed, so String rather than Text -- and the Blueprint is
		// what turns that from a rule to remember into the name of the column's
		// kind: the grammar picks the type each engine wants.
		table.String("queue")
		table.Text("name")
		table.String("tenant_id")
		table.Text("payload")
		table.Text("authorized_by")
		table.Text("action")
		table.Timestamp("run_at")
		table.Timestamp("reserved_until").Nullable()
		table.BigInteger("attempts").Default(0)
		table.Timestamp("failed_at").Nullable()
		table.Text("last_error").Nullable()

		// The pop query filters on queue, failed_at and run_at and orders by
		// run_at. This index is that query.
		table.Index([]string{"queue", "failed_at", "run_at"}, "idx_jobs_ready")
		// The dead letter queue is read by the diagnosis, newest failure first.
		table.Index([]string{"failed_at"}, "idx_jobs_parked")
	})
}

// Down drops the jobs table, and both indexes with it.
func (CreateJobsTable) Down(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().DropIfExists(ctx, "jobs")
}

// AddJobAttributesToJobsTable adds the per-job settings, which used to live
// nowhere: a job pushed with five tries got the worker's five whatever it asked
// for, because the number never reached the table.
//
// They travel as one JSON column rather than one column each -- see
// attributes.Attributes for why they are one struct -- and the two names beside
// it are the two things about a job that are neither its arguments nor a
// setting.
type AddJobAttributesToJobsTable struct{ migrations.BaseMigration }

// GetName returns the migration's name.
func (AddJobAttributesToJobsTable) GetName() string {
	return "2026_08_11_000010_add_job_attributes_to_jobs_table"
}

// Up adds the three columns.
//
// Nullable with a default, so the previous release's binary keeps inserting
// without them during a rollout.
func (AddJobAttributesToJobsTable) Up(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().Table(ctx, "jobs", func(table *schema.Blueprint) {
		// Nullable or defaulted, which is the rollout rule: a NOT NULL column
		// with no default added to a table that has rows fails on every row
		// already there.
		table.Text("display_name").Nullable()
		table.Text("attributes").Nullable()
		table.BigInteger("exceptions").Default(0)
	})
}

// Down drops the three columns.
func (AddJobAttributesToJobsTable) Down(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().Table(ctx, "jobs", func(table *schema.Blueprint) {
		table.DropColumn("display_name", "attributes", "exceptions")
	})
}

// AddCreatedAtToJobsTable adds the column Pop takes jobs in the order of, which
// used to be run_at.
//
// The queue is first in, first out over whatever is eligible. Ordering by
// run_at is not: a job pushed with a ten second delay waits out the delay and
// then goes behind everything queued while it waited, again on every pass, and
// on a queue that is never empty it is a job that never runs.
//
// The id cannot do the job here because it is a random uuid and sorts by
// nothing (see database.NewID), so the column that carries the order is the one
// that says when the row was written.
type AddCreatedAtToJobsTable struct{ migrations.BaseMigration }

// GetName returns the migration's name.
func (AddCreatedAtToJobsTable) GetName() string {
	return "2026_08_13_000010_add_created_at_to_jobs_table"
}

// Up adds the column and backfills it.
//
// Nullable, and with no default, because SQLite refuses a non-constant default
// on ADD COLUMN: the rows already in the table are backfilled here, and the
// ones the previous release's binary writes during the rollout arrive NULL,
// which is what the COALESCE in Pop is for.
//
// No index of its own. The filter is still served by idx_jobs_ready, and what
// is left is a top-N sort under a LIMIT.
//
// That was once written here as a portability argument -- an index on (queue,
// failed_at, created_at) would need a DROP INDEX in Down, which MySQL spells
// "DROP INDEX x ON jobs" and the other two spell "DROP INDEX x" -- and the
// argument no longer holds: DropIndex is a Blueprint command and each grammar
// spells it for its own engine. The migration keeps the shape it ran with all
// the same, because a published migration is not edited: the database that
// applied it would not apply it again, and the two would be one name over two
// schemas. An index here is a new migration, whenever the sort is measured to
// cost something.
//
// The backfill is a plain statement rather than a Blueprint command, and that
// is the line: UPDATE is data, and the Blueprint describes shape.
func (AddCreatedAtToJobsTable) Up(ctx context.Context, conn migrations.Connection) error {
	if err := conn.Schema().Table(ctx, "jobs", func(table *schema.Blueprint) {
		table.Timestamp("created_at").Nullable()
	}); err != nil {
		return err
	}
	return run(ctx, conn, `UPDATE jobs SET created_at = run_at WHERE created_at IS NULL`)
}

// Down drops the column.
func (AddCreatedAtToJobsTable) Down(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().Table(ctx, "jobs", func(table *schema.Blueprint) {
		table.DropColumn("created_at")
	})
}

// run sends statements in order and stops at the first one that fails.
//
// One call per statement, because the connection sends what it is given and
// SQLite's driver refuses more than one statement in an Exec.
func run(ctx context.Context, conn migrations.Connection, statements ...string) error {
	for _, statement := range statements {
		if _, err := conn.Statement(ctx, statement, nil); err != nil {
			return err
		}
	}
	return nil
}

// Migrations returns the jobs table.
//
// The schema is this driver's and only this one's. [Module] collects it, so an
// application wired to another driver declares no schema for a table it will
// never read.
func (q *DatabaseQueue) Migrations() []migrations.Migration {
	return []migrations.Migration{
		CreateJobsTable{},
		AddJobAttributesToJobsTable{},
		AddCreatedAtToJobsTable{},
	}
}

// Push adds a job.
//
// Inside database.Transaction it joins it, which is the property this driver
// exists for: the job is committed by the same transaction as the row it
// describes, so it cannot refer to a write that rolled back.
func (q *DatabaseQueue) Push(ctx context.Context, g auth.Grant, j jobs.Job) error {
	// The job has to match the Grant pushing it. See jobs.Authorized: without
	// this, the queue is the one way past the authorization the collection
	// exists to enforce.
	if err := jobs.Authorized(g, j); err != nil {
		return err
	}
	j = jobs.Prepare(g, j)

	settings, err := json.Marshal(j.Attributes)
	if err != nil {
		return fmt.Errorf("%w: %s: %w", ErrInvalidPayload, j.Name, err)
	}

	// created_at is when the job entered the queue, and it is what Pop orders
	// by. It is not RunAt -- a delayed job entered the queue when it was
	// pushed, and is older than everything queued while it waited.
	_, err = q.db.ExecContext(ctx, `
		INSERT INTO jobs (
			id, queue, name, display_name, tenant_id, payload, authorized_by, action,
			run_at, attempts, exceptions, attributes, created_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		j.UUID, j.Queue, j.Name, j.DisplayName, j.TenantID, string(j.Payload),
		j.AuthorizedBy, j.Action, j.RunAt, j.Attempts, j.Exceptions, string(settings),
		time.Now().UTC())
	if err != nil {
		return fmt.Errorf("queue: pushing %s: %w", j.Name, err)
	}
	return nil
}

// PushRaw adds a job whose arguments are already encoded.
//
// The envelope is the record's columns, so what is raw is the arguments: bytes
// the caller already has, that would only be unmarshalled and marshalled again
// on the way through jobs.New.
//
// It goes through jobs.Authorized like every other push. Raw is about the
// encoding, never about the authorization.
func (q *DatabaseQueue) PushRaw(ctx context.Context, g auth.Grant, name string, payload []byte, queue string) error {
	j, err := jobs.New(g, queue, name, nil)
	if err != nil {
		return err
	}
	j.Payload = payload
	return q.Push(ctx, g, j)
}

// PushOn adds a job to a named queue.
func (q *DatabaseQueue) PushOn(ctx context.Context, g auth.Grant, queue string, j jobs.Job) error {
	j.Queue = queue
	return q.Push(ctx, g, j)
}

// Later adds a job that becomes eligible after delay.
func (q *DatabaseQueue) Later(ctx context.Context, g auth.Grant, delay time.Duration, j jobs.Job) error {
	j.RunAt = time.Now().UTC().Add(delay)
	return q.Push(ctx, g, j)
}

// Bulk adds many jobs.
//
// One statement each rather than a multi-row insert, because inside
// database.Transaction the difference is a round trip and outside it a
// multi-row insert would be the only place in the package where a partial
// failure leaves half a batch behind with no way to say which half.
func (q *DatabaseQueue) Bulk(ctx context.Context, g auth.Grant, js []jobs.Job) error {
	for _, j := range js {
		if err := q.Push(ctx, g, j); err != nil {
			return err
		}
	}
	return nil
}

// Pop takes jobs off the queue and hides them for the lease.
//
// Two statements rather than one, and the reason is portability: the tight form
// is UPDATE ... RETURNING with a FOR UPDATE SKIP LOCKED subquery, which
// Postgres has and SQLite does not. Selecting the candidates and then claiming
// each one by its id -- with reserved_until in the WHERE -- is correct on every
// engine, because the claim itself is the compare-and-set.
//
// The cost is that two workers can pick the same candidate and one of them
// loses the claim. It gets nothing back, which is exactly right.
//
// The candidates come back in the order they were queued. It used to be ORDER
// BY run_at, which put a job pushed with a delay behind everything queued while
// it waited -- forever, on a queue that is never empty. See the created_at
// migration.
//
// COALESCE, because a row written by the previous release's binary during a
// rollout has no created_at and its run_at is the closest thing to one.
func (q *DatabaseQueue) Pop(ctx context.Context, queue string, n int, lease time.Duration) ([]*jobs.Job, error) {
	if queue == "" {
		queue = jobs.DefaultQueue
	}
	if n <= 0 {
		n = 1
	}
	if lease <= 0 {
		lease = 5 * time.Minute
	}

	now := time.Now().UTC()
	candidates, err := q.query(ctx, `
		WHERE queue = ? AND failed_at IS NULL AND run_at <= ?
		  AND (reserved_until IS NULL OR reserved_until < ?)
		ORDER BY COALESCE(created_at, run_at), id
		LIMIT ?`, queue, now, now, n)
	if err != nil {
		return nil, err
	}

	until := now.Add(lease)
	out := make([]*jobs.Job, 0, len(candidates))
	for _, j := range candidates {
		res, err := q.db.ExecContext(ctx, `
			UPDATE jobs SET reserved_until = ?, attempts = attempts + 1
			WHERE id = ? AND (reserved_until IS NULL OR reserved_until < ?)`,
			until, j.UUID, now)
		if err != nil {
			return nil, fmt.Errorf("queue: reserving %s: %w", j.UUID, err)
		}
		claimed, err := res.RowsAffected()
		if err != nil || claimed == 0 {
			// Another worker got it between the select and the update. Not an
			// error: that is the compare-and-set doing its job.
			continue
		}
		// Attempts was incremented by the claim, so the worker sees the number
		// this delivery is.
		j.Attempts++
		out = append(out, jobs.Popped(q, databaseConnection, j))
	}
	return out, nil
}

// DeleteJob removes a finished job.
//
// Deleted rather than marked done. A jobs table that keeps every job ever run
// is a table that needs its own cleanup job, and the history that matters --
// what ran, how long it took, what it queried -- is on the console.
func (q *DatabaseQueue) DeleteJob(ctx context.Context, j *jobs.Job) error {
	if _, err := q.db.ExecContext(ctx, `DELETE FROM jobs WHERE id = ?`, j.UUID); err != nil {
		return fmt.Errorf("queue: deleting %s: %w", j.UUID, err)
	}
	return nil
}

// ReleaseJob puts the job back on its queue, eligible again after delay.
//
// It writes the exception count, and that is the whole reason a released job
// can ever be parked by MaxExceptions: the count has to cross deliveries. While
// the column went unwritten every delivery read zero, so a job with
// MaxExceptions 2 and MaxTries 5 was delivered all five times.
//
// created_at moves to now, because a released job goes to the back of the
// queue: leaving the original there would put a job that has failed twice in
// front of work nobody has looked at yet.
func (q *DatabaseQueue) ReleaseJob(ctx context.Context, j *jobs.Job, delay time.Duration) error {
	if delay < 0 {
		delay = 0
	}
	now := time.Now().UTC()
	_, err := q.db.ExecContext(ctx, `
		UPDATE jobs SET run_at = ?, last_error = ?, exceptions = ?, created_at = ?,
		                reserved_until = NULL
		WHERE id = ?`,
		now.Add(delay), j.LastError, j.Exceptions, now, j.UUID)
	if err != nil {
		return fmt.Errorf("queue: releasing %s: %w", j.UUID, err)
	}
	return nil
}

// FailJob parks the job.
func (q *DatabaseQueue) FailJob(ctx context.Context, j *jobs.Job, cause error) error {
	message := j.LastError
	if cause != nil {
		message = cause.Error()
	}
	_, err := q.db.ExecContext(ctx,
		`UPDATE jobs SET failed_at = ?, last_error = ?, reserved_until = NULL WHERE id = ?`,
		time.Now().UTC(), message, j.UUID)
	if err != nil {
		return fmt.Errorf("queue: parking %s: %w", j.UUID, err)
	}
	return nil
}

// Failed lists the jobs that gave up, most recent failure first.
func (q *DatabaseQueue) Failed(ctx context.Context, limit int) ([]jobs.Job, error) {
	if limit <= 0 {
		limit = 100
	}
	return q.query(ctx, `WHERE failed_at IS NOT NULL ORDER BY failed_at DESC LIMIT ?`, limit)
}

// Retry puts a failed job back in line with its attempts reset.
//
// The row that failed is the row that comes back, which is what makes this the
// way out of a dead letter list rather than a second push: the id, the action,
// the tenant and the payload are already there and none of them is rebuilt, so
// nothing about the job can be lost in the round trip.
//
// The exception count is reset with the attempts: a retry that kept the count
// would be parked again by the failure that parked it the first time, without
// the handler ever having run. created_at moves to now for the reason it moves
// on a release -- the job goes to the back of the queue.
//
// failed_at is in the WHERE, so this un-parks and nothing else. Without it a
// retry aimed at a parked id that had already been retried would reset the
// attempts of a job that is running, and a job whose count is reset mid-flight
// is one the worker will not park when it next fails.
//
// Nothing updated is [ErrNotParked] rather than success. The caller is holding a
// dead letter record and has somewhere else to put the job; told the retry
// worked, it forgets the record and the work is gone.
func (q *DatabaseQueue) Retry(ctx context.Context, uuid string) error {
	now := time.Now().UTC()
	res, err := q.db.ExecContext(ctx, `
		UPDATE jobs SET failed_at = NULL, attempts = 0, exceptions = 0, last_error = NULL,
		                reserved_until = NULL, run_at = ?, created_at = ?
		WHERE id = ? AND failed_at IS NOT NULL`, now, now, uuid)
	if err != nil {
		return fmt.Errorf("queue: retrying %s: %w", uuid, err)
	}
	updated, err := res.RowsAffected()
	if err != nil {
		return fmt.Errorf("queue: retrying %s: %w", uuid, err)
	}
	if updated == 0 {
		return fmt.Errorf("%w: %s", ErrNotParked, uuid)
	}
	return nil
}

// Size is how many jobs the queue holds, waiting or in flight.
func (q *DatabaseQueue) Size(ctx context.Context, queue string) (int, error) {
	if queue == "" {
		queue = jobs.DefaultQueue
	}
	var count int
	err := q.db.QueryRowContext(ctx,
		`SELECT count(*) FROM jobs WHERE queue = ? AND failed_at IS NULL`, queue).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("queue: sizing %s: %w", queue, err)
	}
	return count, nil
}

// PendingSize is how many jobs are waiting.
func (q *DatabaseQueue) PendingSize(ctx context.Context, queue string) (int, error) {
	if queue == "" {
		queue = jobs.DefaultQueue
	}
	// A reserved job is running, not waiting. Pop has always known that -- its
	// WHERE says so -- and these two did not, so a worker doing exactly what it
	// should looked like a backlog: a two-minute handler with a one-minute
	// threshold made /_arandu/health answer 503 while the job ran, and a load
	// balancer took the instance out of rotation for it.
	//
	// An expired lease is waiting again, which is why the comparison is against
	// now rather than a plain IS NULL.
	var count int
	err := q.db.QueryRowContext(ctx, `
		SELECT count(*) FROM jobs
		WHERE queue = ? AND failed_at IS NULL
		  AND (reserved_until IS NULL OR reserved_until < ?)`,
		queue, time.Now().UTC()).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("queue: counting %s: %w", queue, err)
	}
	return count, nil
}

// CreationTimeOfOldestPendingJob is when the oldest waiting job became
// eligible, or the zero time when nothing is waiting.
func (q *DatabaseQueue) CreationTimeOfOldestPendingJob(ctx context.Context, queue string) (time.Time, error) {
	if queue == "" {
		queue = jobs.DefaultQueue
	}

	// ORDER BY ... LIMIT 1 rather than min(run_at): an aggregate loses the
	// declared type of the column, and SQLite then hands back a string that
	// will not scan into a time.Time.
	// The same definition of "waiting" as PendingSize and Pop: a reserved job
	// is running.
	var oldest time.Time
	err := q.db.QueryRowContext(ctx, `
		SELECT run_at FROM jobs
		WHERE queue = ? AND failed_at IS NULL
		  AND (reserved_until IS NULL OR reserved_until < ?)
		ORDER BY run_at
		LIMIT 1`, queue, time.Now().UTC()).Scan(&oldest)
	if errors.Is(err, sql.ErrNoRows) {
		return time.Time{}, nil
	}
	if err != nil {
		return time.Time{}, fmt.Errorf("queue: measuring the age of %s: %w", queue, err)
	}
	return oldest, nil
}

// Clear removes every job waiting or in flight on a queue, and returns how many
// went.
//
// Parked jobs are not cleared: a job that gave up is no longer on a queue, it
// is in the dead letter list, and [DatabaseQueue.Failed] and
// [DatabaseQueue.Retry] are how it is dealt with. The RESP driver draws the
// line in the same place.
func (q *DatabaseQueue) Clear(ctx context.Context, queue string) (int, error) {
	if queue == "" {
		queue = jobs.DefaultQueue
	}
	res, err := q.db.ExecContext(ctx, `DELETE FROM jobs WHERE queue = ? AND failed_at IS NULL`, queue)
	if err != nil {
		return 0, fmt.Errorf("queue: clearing %s: %w", queue, err)
	}
	removed, err := res.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("queue: clearing %s: %w", queue, err)
	}
	return int(removed), nil
}

// query runs the standard projection with a caller-supplied tail.
func (q *DatabaseQueue) query(ctx context.Context, tail string, args ...any) ([]jobs.Job, error) {
	// The newline is load-bearing: without it a tail starting with WHERE
	// concatenates into "FROM jobsWHERE", which SQLite reads as a table alias
	// and then fails on the next word -- an error that names a keyword three
	// tokens away from the actual problem.
	rows, err := q.db.QueryContext(ctx, `
		SELECT id, queue, name, display_name, tenant_id, payload, authorized_by, action,
		       run_at, attempts, exceptions, last_error, attributes
		FROM jobs
		`+tail, args...)
	if err != nil {
		return nil, fmt.Errorf("queue: reading the queue: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var out []jobs.Job
	for rows.Next() {
		var j jobs.Job
		var payload string
		var displayName, lastError, settings sql.NullString
		if err := rows.Scan(&j.UUID, &j.Queue, &j.Name, &displayName, &j.TenantID, &payload,
			&j.AuthorizedBy, &j.Action, &j.RunAt, &j.Attempts, &j.Exceptions,
			&lastError, &settings); err != nil {
			return nil, fmt.Errorf("queue: reading the queue: %w", err)
		}
		j.Payload = []byte(payload)
		j.DisplayName = displayName.String
		j.LastError = lastError.String
		if settings.String != "" {
			// A row written before the settings column existed, or by a binary
			// that predates it, decodes to the zero value -- which is what "the
			// worker decides" already means. It is not an error, and treating
			// it as one would fail every job queued during a rollout.
			_ = json.Unmarshal([]byte(settings.String), &j.Attributes)
		}
		out = append(out, j)
	}
	return out, rows.Err()
}

// Release puts a job back on its queue, eligible again after delay.
//
// It is the driver-side half of jobs.Job.Release: a handler and a middleware
// call the job, and the job calls this.
func (q *DatabaseQueue) Release(ctx context.Context, queue string, j *jobs.Job, delay time.Duration) error {
	j.Queue = q.GetQueue(queue)
	return q.ReleaseJob(ctx, j, delay)
}

// DeleteAndRelease removes a reserved job and queues it again.
//
// A release here is one UPDATE -- the row never left -- so this is that
// update, under the name a driver that claims by deletion would need.
func (q *DatabaseQueue) DeleteAndRelease(ctx context.Context, queue string, j *jobs.Job, delay time.Duration) error {
	return q.Release(ctx, queue, j, delay)
}

// DeleteReserved removes a reserved job by its id.
//
// It is what a command reaches for when a job is stuck: the row is gone and no
// worker will pick it up when the lease expires.
func (q *DatabaseQueue) DeleteReserved(ctx context.Context, queue, id string) error {
	if _, err := q.db.ExecContext(ctx, `DELETE FROM jobs WHERE id = ? AND queue = ?`,
		id, q.GetQueue(queue)); err != nil {
		return fmt.Errorf("queue: deleting %s: %w", id, err)
	}
	return nil
}

// DelayedSize is how many jobs are waiting for a time that has not come.
//
// It is the number [DatabaseQueue.PendingSize] leaves out: a job scheduled for
// tomorrow is not a backlog, and counting it as one is how a health check pages
// somebody at three in the morning about a report due at nine.
func (q *DatabaseQueue) DelayedSize(ctx context.Context, queue string) (int, error) {
	var count int
	err := q.db.QueryRowContext(ctx, `
		SELECT count(*) FROM jobs
		WHERE queue = ? AND failed_at IS NULL AND run_at > ?`,
		q.GetQueue(queue), time.Now().UTC()).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("queue: counting the delayed jobs on %s: %w", queue, err)
	}
	return count, nil
}

// ReservedSize is how many jobs a worker is holding right now.
//
// A number that stays high while PendingSize stays high is a worker that took
// work and stopped: the leases have not expired yet, so nothing is visibly
// wrong, and this is what shows it.
func (q *DatabaseQueue) ReservedSize(ctx context.Context, queue string) (int, error) {
	var count int
	err := q.db.QueryRowContext(ctx, `
		SELECT count(*) FROM jobs
		WHERE queue = ? AND failed_at IS NULL AND reserved_until >= ?`,
		q.GetQueue(queue), time.Now().UTC()).Scan(&count)
	if err != nil {
		return 0, fmt.Errorf("queue: counting the reserved jobs on %s: %w", queue, err)
	}
	return count, nil
}
