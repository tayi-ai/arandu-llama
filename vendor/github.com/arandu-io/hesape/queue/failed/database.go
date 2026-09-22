package failed

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database"
	"github.com/arandu-io/hesape/database/migrations"
	"github.com/arandu-io/hesape/database/schema"
)

// DatabaseFailedJobProvider keeps the failed jobs in a table.
//
// It is what an application wires when it wants the dead letter list to
// survive the queue --
// a job that failed on a Redis queue that was then flushed is still a job
// somebody has to answer for.
//
// The table is its own, and that is the point of the type: a queue whose store
// is not a database has nowhere else to keep them.
type DatabaseFailedJobProvider struct {
	db    *database.DB
	table string
}

// DefaultTable is where failures are logged when no table name is given.
const DefaultTable = "failed_jobs"

// NewDatabaseFailedJobProvider returns the provider over db.
//
// An empty table means DefaultTable.
func NewDatabaseFailedJobProvider(db *database.DB, table string) *DatabaseFailedJobProvider {
	if table == "" {
		table = DefaultTable
	}
	return &DatabaseFailedJobProvider{db: db, table: table}
}

var (
	_ FailedJobProvider          = (*DatabaseFailedJobProvider)(nil)
	_ CountableFailedJobProvider = (*DatabaseFailedJobProvider)(nil)
	_ PrunableFailedJobProvider  = (*DatabaseFailedJobProvider)(nil)
)

// DatabaseUUIDFailedJobProvider is [DatabaseFailedJobProvider] under the name
// for the provider keyed by the job's uuid.
//
// It is an alias because there is nothing left to distinguish: the id is the
// uuid (see database.NewID), so a provider that looked up by uuid would run the
// same query.
type DatabaseUUIDFailedJobProvider = DatabaseFailedJobProvider

// GetTable is the name of the table this provider reads.
//
// It returns the name rather than a query over it: a query handed out is a
// caller writing its own SQL against a table it does not own, and the tenant
// filter is not optional.
func (p *DatabaseFailedJobProvider) GetTable() string { return p.table }

// CreateFailedJobsTable creates the table a DatabaseFailedJobProvider logs to.
//
// The table name is a field because the provider's is: an application that
// keeps its failures somewhere other than failed_jobs migrates the name it
// wired.
type CreateFailedJobsTable struct {
	migrations.BaseMigration

	// Table is the table to create. Empty means DefaultTable.
	Table string
}

// GetName returns the migration's name.
func (CreateFailedJobsTable) GetName() string {
	return "2026_08_11_000020_create_failed_jobs_table"
}

// Up creates the failed jobs table and the index every read of it uses.
//
// Portable types only: TEXT, INTEGER and TIMESTAMP mean the same thing on
// SQLite, Postgres and MySQL.
func (m CreateFailedJobsTable) Up(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().Create(ctx, m.table(), func(table *schema.Blueprint) {
		table.String("id").Primary()
		table.String("uuid")
		table.String("tenant_id")
		table.String("connection")
		table.String("queue")
		table.Text("name")
		table.Text("payload")
		table.Text("exception")
		table.Timestamp("failed_at")

		// Every read filters by tenant and orders by failure time, and the
		// monitor narrows by queue. This index is those queries.
		table.Index([]string{"tenant_id", "queue", "failed_at"}, "idx_failed_jobs_tenant")
	})
}

// Down drops the failed jobs table, and the index with it.
func (m CreateFailedJobsTable) Down(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().DropIfExists(ctx, m.table())
}

// table is m.Table with the default filled in.
func (m CreateFailedJobsTable) table() string {
	if m.Table == "" {
		return DefaultTable
	}
	return m.Table
}

// AddActionToFailedJobsTable adds the column that carries the permission a job
// was pushed under, which the record used to drop.
//
// A retry rebuilds the job from this row, and the Grant it runs under is built
// from an action. Without the column the only action a retry could name was the
// one the dead letter list is read with, so the work came back as an
// administrator rather than as itself.
//
// It is its own migration rather than an edit to [CreateFailedJobsTable]: a
// published migration is not changed, because a database that already applied
// it would not apply it again and the two would be one name over two schemas.
type AddActionToFailedJobsTable struct {
	migrations.BaseMigration

	// Table is the table to alter. Empty means DefaultTable.
	Table string
}

// GetName returns the migration's name.
func (AddActionToFailedJobsTable) GetName() string {
	return "2026_09_03_000010_add_action_to_failed_jobs_table"
}

// Up adds the column.
//
// Nullable, so the release running while this is applied keeps inserting
// without it: a NOT NULL column with no default added to a table that has rows
// fails on every row already there, and the previous binary's insert names nine
// columns rather than ten.
func (m AddActionToFailedJobsTable) Up(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().Table(ctx, m.table(), func(table *schema.Blueprint) {
		table.Text("action").Nullable()
	})
}

// Down drops the column.
func (m AddActionToFailedJobsTable) Down(ctx context.Context, conn migrations.Connection) error {
	return conn.Schema().Table(ctx, m.table(), func(table *schema.Blueprint) {
		table.DropColumn("action")
	})
}

// table is m.Table with the default filled in.
func (m AddActionToFailedJobsTable) table() string {
	if m.Table == "" {
		return DefaultTable
	}
	return m.Table
}

// Migrations returns the failed jobs table.
//
// The schema is on the provider rather than on the queue module because it
// belongs to whoever wired this provider: an application that keeps its
// failures in the jobs table declares nothing here.
func (p *DatabaseFailedJobProvider) Migrations() []migrations.Migration {
	return []migrations.Migration{
		CreateFailedJobsTable{Table: p.table},
		AddActionToFailedJobsTable{Table: p.table},
	}
}

// Log records a job that gave up, once.
//
// The id is the job's own uuid and it is the primary key, so a second record of
// one failure is refused by the table rather than written. That refusal is the
// answer the caller wanted -- the failure is listed -- so it is read back as
// such, and only an insert that failed for any other reason is returned as an
// error.
func (p *DatabaseFailedJobProvider) Log(ctx context.Context, g auth.Grant, job FailedJob) (string, error) {
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
	failedAt := job.FailedAt
	if failedAt.IsZero() {
		failedAt = time.Now()
	}

	_, err = p.db.ExecContext(ctx, `
		INSERT INTO `+p.table+` (
			id, uuid, tenant_id, connection, queue, name, action, payload, exception, failed_at
		) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		id, id, tenant, job.Connection, job.Queue, job.Name, job.Action,
		string(job.Payload), job.Exception, failedAt.UTC())
	if err != nil {
		// Asked afterwards rather than before: a check that ran first would be
		// a read every failure pays for, and it would still race with a second
		// worker recording the same id.
		if recorded, lookupErr := p.recorded(ctx, tenant, id); lookupErr == nil && recorded {
			return id, nil
		}
		return "", fmt.Errorf("queue/failed: recording %s: %w", job.Name, err)
	}
	return id, nil
}

// recorded reports whether this tenant already has a failure under id.
//
// The tenant is in the WHERE, so a row somebody else's id collided with is not
// this tenant's failure and the insert that hit it is still an error.
func (p *DatabaseFailedJobProvider) recorded(ctx context.Context, tenant, id string) (bool, error) {
	var count int
	err := p.db.QueryRowContext(ctx,
		`SELECT count(*) FROM `+p.table+` WHERE tenant_id = ? AND id = ?`, tenant, id).Scan(&count)
	if err != nil {
		return false, err
	}
	return count > 0, nil
}

// IDs is the identifiers of this tenant's failed jobs, newest first.
func (p *DatabaseFailedJobProvider) IDs(ctx context.Context, g auth.Grant, queue string) ([]string, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return nil, err
	}

	query := `SELECT id FROM ` + p.table + ` WHERE tenant_id = ?`
	args := []any{tenant}
	if queue != "" {
		query += ` AND queue = ?`
		args = append(args, queue)
	}
	query += ` ORDER BY failed_at DESC`

	rows, err := p.db.QueryContext(ctx, query, args...)
	if err != nil {
		return nil, fmt.Errorf("queue/failed: listing the failed jobs: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var out []string
	for rows.Next() {
		var id string
		if err := rows.Scan(&id); err != nil {
			return nil, fmt.Errorf("queue/failed: listing the failed jobs: %w", err)
		}
		out = append(out, id)
	}
	return out, rows.Err()
}

// All is this tenant's failed jobs, newest first.
func (p *DatabaseFailedJobProvider) All(ctx context.Context, g auth.Grant) ([]FailedJob, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return nil, err
	}
	return p.query(ctx, `WHERE tenant_id = ? ORDER BY failed_at DESC`, tenant)
}

// Find is one of this tenant's failed jobs.
func (p *DatabaseFailedJobProvider) Find(ctx context.Context, g auth.Grant, id string) (FailedJob, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return FailedJob{}, err
	}
	// The tenant is in the WHERE and not checked afterwards: a query that finds
	// the row and then refuses it has already read another customer's payload
	// into this process.
	found, err := p.query(ctx, `WHERE tenant_id = ? AND id = ?`, tenant, id)
	if err != nil {
		return FailedJob{}, err
	}
	if len(found) == 0 {
		return FailedJob{}, fmt.Errorf("%w: %s", ErrNotFound, id)
	}
	return found[0], nil
}

// Forget removes one of this tenant's failed jobs.
func (p *DatabaseFailedJobProvider) Forget(ctx context.Context, g auth.Grant, id string) (bool, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return false, err
	}
	res, err := p.db.ExecContext(ctx,
		`DELETE FROM `+p.table+` WHERE tenant_id = ? AND id = ?`, tenant, id)
	if err != nil {
		return false, fmt.Errorf("queue/failed: forgetting %s: %w", id, err)
	}
	removed, err := res.RowsAffected()
	if err != nil {
		return false, fmt.Errorf("queue/failed: forgetting %s: %w", id, err)
	}
	return removed > 0, nil
}

// Flush removes this tenant's failed jobs older than age, or all of them when
// age is zero.
func (p *DatabaseFailedJobProvider) Flush(ctx context.Context, g auth.Grant, age time.Duration) error {
	tenant, err := tenantOf(g)
	if err != nil {
		return err
	}

	query := `DELETE FROM ` + p.table + ` WHERE tenant_id = ?`
	args := []any{tenant}
	if age > 0 {
		query += ` AND failed_at <= ?`
		args = append(args, time.Now().UTC().Add(-age))
	}
	if _, err := p.db.ExecContext(ctx, query, args...); err != nil {
		return fmt.Errorf("queue/failed: flushing the failed jobs: %w", err)
	}
	return nil
}

// Prune removes this tenant's failed jobs that failed before an instant, and
// returns how many went.
func (p *DatabaseFailedJobProvider) Prune(ctx context.Context, g auth.Grant, before time.Time) (int, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return 0, err
	}
	res, err := p.db.ExecContext(ctx,
		`DELETE FROM `+p.table+` WHERE tenant_id = ? AND failed_at < ?`, tenant, before.UTC())
	if err != nil {
		return 0, fmt.Errorf("queue/failed: pruning the failed jobs: %w", err)
	}
	removed, err := res.RowsAffected()
	if err != nil {
		return 0, fmt.Errorf("queue/failed: pruning the failed jobs: %w", err)
	}
	return int(removed), nil
}

// Count is how many of this tenant's jobs have failed.
func (p *DatabaseFailedJobProvider) Count(ctx context.Context, g auth.Grant, connectionName, queue string) (int, error) {
	tenant, err := tenantOf(g)
	if err != nil {
		return 0, err
	}

	query := `SELECT count(*) FROM ` + p.table + ` WHERE tenant_id = ?`
	args := []any{tenant}
	if connectionName != "" {
		query += ` AND connection = ?`
		args = append(args, connectionName)
	}
	if queue != "" {
		query += ` AND queue = ?`
		args = append(args, queue)
	}

	var count int
	if err := p.db.QueryRowContext(ctx, query, args...).Scan(&count); err != nil {
		return 0, fmt.Errorf("queue/failed: counting the failed jobs: %w", err)
	}
	return count, nil
}

// query runs the standard projection with a caller-supplied tail.
func (p *DatabaseFailedJobProvider) query(ctx context.Context, tail string, args ...any) ([]FailedJob, error) {
	// The newline is load-bearing: without it a tail starting with WHERE
	// concatenates into "failed_jobsWHERE", which SQLite reads as a table alias
	// and then fails on the next word.
	rows, err := p.db.QueryContext(ctx, `
		SELECT id, uuid, tenant_id, connection, queue, name, action, payload, exception, failed_at
		FROM `+p.table+`
		`+tail, args...)
	if err != nil {
		return nil, fmt.Errorf("queue/failed: reading the failed jobs: %w", err)
	}
	defer func() { _ = rows.Close() }()

	var out []FailedJob
	for rows.Next() {
		var job FailedJob
		var payload string
		// The action column is nullable, because it was added to a table that
		// already had rows: a record written before it existed scans as NULL,
		// and a plain string destination refuses that outright.
		var action sql.NullString
		if err := rows.Scan(&job.ID, &job.UUID, &job.TenantID, &job.Connection, &job.Queue,
			&job.Name, &action, &payload, &job.Exception, &job.FailedAt); err != nil {
			if errors.Is(err, sql.ErrNoRows) {
				return out, nil
			}
			return nil, fmt.Errorf("queue/failed: reading the failed jobs: %w", err)
		}
		job.Action = action.String
		job.Payload = []byte(payload)
		out = append(out, job)
	}
	return out, rows.Err()
}
