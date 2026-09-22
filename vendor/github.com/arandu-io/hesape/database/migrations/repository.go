package migrations

import (
	"context"
	"fmt"
	"sort"
)

// Resolver answers a connection by name, narrowed to what the repository and the
// Migrator ask of it.
//
// It is declared here rather than imported for the reason every interface in
// this component is: the database package resolves migrations, so it imports
// this one, and naming its concrete resolver here would close the cycle.
type Resolver interface {
	// Connection returns the named connection, or an error for an unknown
	// name.
	Connection(name string) (Connection, error)

	// GetDefaultConnection returns the default connection name.
	GetDefaultConnection() string

	// SetDefaultConnection replaces the default connection name.
	SetDefaultConnection(name string)
}

// MigrationRecord is one row of the repository table: an id, a migration
// name, and the batch it ran in.
type MigrationRecord struct {
	// ID is the row's ordinal.
	ID int

	// Migration is the migration's name, the same string GetName answers.
	Migration string

	// Batch is the run it belonged to. One `aru migrate` is one batch, which
	// is what makes a rollback undo a deploy rather than a single file.
	Batch int
}

// MigrationRepositoryInterface is the record of which migrations have run.
//
// Every method returns an error, because every one of them talks to the
// database: a caller that cannot tell "no migrations have run" from "the
// connection is gone" writes a migrate command that reports success on a dead
// database.
type MigrationRepositoryInterface interface {
	// GetRan answers getRan: the names of every applied migration, ordered by
	// batch and then by name.
	GetRan(ctx context.Context) ([]string, error)

	// GetMigrations answers getMigrations: the last N applied, most recent
	// first.
	GetMigrations(ctx context.Context, steps int) ([]MigrationRecord, error)

	// GetMigrationsByBatch answers getMigrationsByBatch.
	GetMigrationsByBatch(ctx context.Context, batch int) ([]MigrationRecord, error)

	// GetLast answers getLast: the whole of the most recent batch, in reverse
	// order, which is the order a rollback undoes them in.
	GetLast(ctx context.Context) ([]MigrationRecord, error)

	// GetMigrationBatches answers getMigrationBatches: name to batch, for the
	// status table.
	GetMigrationBatches(ctx context.Context) (map[string]int, error)

	// Log answers log: record that a migration ran.
	Log(ctx context.Context, file string, batch int) error

	// Delete answers delete: forget that one did.
	Delete(ctx context.Context, migration MigrationRecord) error

	// GetNextBatchNumber answers getNextBatchNumber.
	GetNextBatchNumber(ctx context.Context) (int, error)

	// CreateRepository answers createRepository: create the table.
	CreateRepository(ctx context.Context) error

	// RepositoryExists answers repositoryExists.
	RepositoryExists(ctx context.Context) bool

	// DeleteRepository answers deleteRepository: drop the table.
	DeleteRepository(ctx context.Context) error

	// SetSource answers setSource: the connection to read and write the table
	// on.
	SetSource(name string)
}

// DatabaseMigrationRepository is the table that remembers which migrations have
// run.
//
// Its queries are written out as SQL rather than built, because that is what
// this framework does everywhere else and because the four statements involved
// are the same on all three engines.
type DatabaseMigrationRepository struct {
	// resolver is how the repository reaches a connection by name.
	resolver Resolver

	// table is the name of the table migrations are recorded in.
	table string

	// connection is the name SetSource put there.
	connection string
}

// DefaultTable is the table name to pass to NewDatabaseMigrationRepository
// unless a project has a reason not to.
const DefaultTable = "migrations"

// NewDatabaseMigrationRepository creates a DatabaseMigrationRepository.
func NewDatabaseMigrationRepository(resolver Resolver, table string) *DatabaseMigrationRepository {
	return &DatabaseMigrationRepository{resolver: resolver, table: table}
}

// GetRan returns the names of every applied migration, ordered by batch and
// then by name.
func (r *DatabaseMigrationRepository) GetRan(ctx context.Context) ([]string, error) {
	records, err := r.query(ctx, `SELECT id, migration, batch FROM `+r.table+` ORDER BY batch ASC, migration ASC`)
	if err != nil {
		return nil, err
	}
	out := make([]string, 0, len(records))
	for _, record := range records {
		out = append(out, record.Migration)
	}
	return out, nil
}

// GetMigrations returns the last steps applied, newest first.
//
// The `batch >= 1` filter is not redundant: a batch of zero is what `migrate
// --pretend` and hand-written rows leave behind, and neither should be
// rolled back.
func (r *DatabaseMigrationRepository) GetMigrations(ctx context.Context, steps int) ([]MigrationRecord, error) {
	records, err := r.query(ctx,
		`SELECT id, migration, batch FROM `+r.table+
			` WHERE batch >= 1 ORDER BY batch DESC, migration DESC LIMIT ?`, steps)
	if err != nil {
		return nil, err
	}
	return records, nil
}

// GetMigrationsByBatch returns every migration recorded under batch, newest
// first.
func (r *DatabaseMigrationRepository) GetMigrationsByBatch(ctx context.Context, batch int) ([]MigrationRecord, error) {
	return r.query(ctx,
		`SELECT id, migration, batch FROM `+r.table+` WHERE batch = ? ORDER BY migration DESC`, batch)
}

// GetLast returns the most recent batch, in the order a rollback wants it.
func (r *DatabaseMigrationRepository) GetLast(ctx context.Context) ([]MigrationRecord, error) {
	last, err := r.GetLastBatchNumber(ctx)
	if err != nil {
		return nil, err
	}
	return r.query(ctx,
		`SELECT id, migration, batch FROM `+r.table+` WHERE batch = ? ORDER BY migration DESC`, last)
}

// GetMigrationBatches returns the name-to-batch map `migrate:status` prints.
func (r *DatabaseMigrationRepository) GetMigrationBatches(ctx context.Context) (map[string]int, error) {
	records, err := r.query(ctx, `SELECT id, migration, batch FROM `+r.table+` ORDER BY batch ASC, migration ASC`)
	if err != nil {
		return nil, err
	}
	out := make(map[string]int, len(records))
	for _, record := range records {
		out[record.Migration] = record.Batch
	}
	return out, nil
}

// Log records that a migration ran, in a batch.
//
// The id is computed rather than left to the engine: the three engines spell
// auto-increment three different ways -- SERIAL, AUTO_INCREMENT, AUTOINCREMENT
// -- so a portable CREATE TABLE cannot have one. Nothing reads the id for
// anything but order, and migrations run one at a time in a pipeline step, so
// max + 1 is the whole of it.
func (r *DatabaseMigrationRepository) Log(ctx context.Context, file string, batch int) error {
	conn, err := r.GetConnection()
	if err != nil {
		return err
	}

	next, err := r.nextID(ctx)
	if err != nil {
		return err
	}

	_, err = conn.Statement(ctx,
		`INSERT INTO `+r.table+` (id, migration, batch) VALUES (?, ?, ?)`,
		[]any{next, file, batch})
	if err != nil {
		return fmt.Errorf("recording migration %s: %w", file, err)
	}
	return nil
}

// Delete forgets that migration ran.
func (r *DatabaseMigrationRepository) Delete(ctx context.Context, migration MigrationRecord) error {
	conn, err := r.GetConnection()
	if err != nil {
		return err
	}
	_, err = conn.Statement(ctx, `DELETE FROM `+r.table+` WHERE migration = ?`, []any{migration.Migration})
	if err != nil {
		return fmt.Errorf("clearing the record of %s: %w", migration.Migration, err)
	}
	return nil
}

// GetNextBatchNumber returns one past the highest batch number recorded.
func (r *DatabaseMigrationRepository) GetNextBatchNumber(ctx context.Context) (int, error) {
	last, err := r.GetLastBatchNumber(ctx)
	if err != nil {
		return 0, err
	}
	return last + 1, nil
}

// GetLastBatchNumber returns the highest batch number recorded, or zero for
// an empty table.
//
// It reads every row and takes the max in Go rather than asking the engine
// for MAX(batch), which avoids a NULL on an empty table and a driver that
// scans NULL into an int. The table never holds more than a few hundred
// rows, so reading them is not a cost worth avoiding.
func (r *DatabaseMigrationRepository) GetLastBatchNumber(ctx context.Context) (int, error) {
	records, err := r.query(ctx, `SELECT id, migration, batch FROM `+r.table)
	if err != nil {
		return 0, err
	}
	last := 0
	for _, record := range records {
		if record.Batch > last {
			last = record.Batch
		}
	}
	return last, nil
}

// CreateRepository creates the migrations table.
//
// The column types are the portable spellings: INTEGER and VARCHAR(255).
// VARCHAR rather than TEXT because migration is what every query here filters
// on and MySQL refuses a TEXT column in a key without a prefix length -- the
// mistake that once stopped `aru migrate` on its own first statement.
//
// # Why this one is not written with the Blueprint
//
// Every other table in the collection is, and this is the exception rather than
// the one that was missed. Two reasons, and both are about it being first:
//
// IF NOT EXISTS is the idempotency the migrator relies on -- CreateRepository
// is called on a path that may already have run -- and the Blueprint has no
// command for it. Asking HasTable first would answer the same question with an
// extra round trip and a race between the two statements.
//
// And this is the table that records which migrations ran, so it exists before
// there is anywhere to record that it was created. A table with no migration
// behind it is not a migration, and running it through the builder that writes
// migrations would say otherwise.
//
// The statement is portable as written: all three engines accept this exact
// text. Statement is the declared escape for what the Blueprint does not
// describe, and this is what it is for.
func (r *DatabaseMigrationRepository) CreateRepository(ctx context.Context) error {
	conn, err := r.GetConnection()
	if err != nil {
		return err
	}
	_, err = conn.Statement(ctx, `CREATE TABLE IF NOT EXISTS `+r.table+` (
	id        INTEGER NOT NULL,
	migration VARCHAR(255) NOT NULL,
	batch     INTEGER NOT NULL,
	PRIMARY KEY (migration)
)`, nil)
	if err != nil {
		return fmt.Errorf("creating %s: %w", r.table, err)
	}
	return nil
}

// RepositoryExists reports whether the migrations table exists.
//
// It selects from the table rather than asking the engine's catalogue,
// which would need three different queries for three engines: a table that
// is not there cannot be selected from, and that works everywhere.
func (r *DatabaseMigrationRepository) RepositoryExists(ctx context.Context) bool {
	conn, err := r.GetConnection()
	if err != nil {
		return false
	}
	_, err = conn.Select(ctx, `SELECT migration FROM `+r.table+` WHERE 1 = 0`, nil)
	return err == nil
}

// DeleteRepository drops the migrations table.
func (r *DatabaseMigrationRepository) DeleteRepository(ctx context.Context) error {
	conn, err := r.GetConnection()
	if err != nil {
		return err
	}
	_, err = conn.Statement(ctx, `DROP TABLE `+r.table, nil)
	return err
}

// GetConnectionResolver returns the resolver the repository reaches
// connections through.
func (r *DatabaseMigrationRepository) GetConnectionResolver() Resolver { return r.resolver }

// GetConnection returns the connection the repository reads and writes the
// table on.
func (r *DatabaseMigrationRepository) GetConnection() (Connection, error) {
	if r.resolver == nil {
		return nil, fmt.Errorf("the migration repository has no connection resolver")
	}
	return r.resolver.Connection(r.connection)
}

// SetSource sets the connection the repository reads and writes the table
// on.
func (r *DatabaseMigrationRepository) SetSource(name string) { r.connection = name }

// GetTable is the table name the repository was built with, exported
// because a caller in another package cannot reach the field directly.
func (r *DatabaseMigrationRepository) GetTable() string { return r.table }

// nextID is the ordinal Log writes. See there for why it is computed.
func (r *DatabaseMigrationRepository) nextID(ctx context.Context) (int, error) {
	records, err := r.query(ctx, `SELECT id, migration, batch FROM `+r.table)
	if err != nil {
		return 0, err
	}
	next := 0
	for _, record := range records {
		if record.ID > next {
			next = record.ID
		}
	}
	return next + 1, nil
}

// query runs one of the reads above and shapes the rows.
func (r *DatabaseMigrationRepository) query(ctx context.Context, sql string, bindings ...any) ([]MigrationRecord, error) {
	conn, err := r.GetConnection()
	if err != nil {
		return nil, err
	}

	rows, err := conn.Select(ctx, sql, bindings)
	if err != nil {
		return nil, fmt.Errorf("reading %s: %w", r.table, err)
	}

	out := make([]MigrationRecord, 0, len(rows))
	for _, row := range rows {
		out = append(out, MigrationRecord{
			ID:        asInt(row["id"]),
			Migration: asString(row["migration"]),
			Batch:     asInt(row["batch"]),
		})
	}
	return out, nil
}

// asInt reads an integer column whichever way the driver handed it over.
//
// database/sql gives an INTEGER back as int64 from one driver, int from
// another, and []byte from MySQL when the column went through a text protocol.
// A type switch here is what keeps that from being three bugs.
func asInt(v any) int {
	switch n := v.(type) {
	case int:
		return n
	case int32:
		return int(n)
	case int64:
		return int(n)
	case float64:
		return int(n)
	case []byte:
		return parseInt(string(n))
	case string:
		return parseInt(n)
	default:
		return 0
	}
}

func parseInt(s string) int {
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return n
		}
		n = n*10 + int(c-'0')
	}
	return n
}

func asString(v any) string {
	switch s := v.(type) {
	case string:
		return s
	case []byte:
		return string(s)
	case nil:
		return ""
	default:
		return fmt.Sprint(v)
	}
}

// sortRecordsByName orders records the way the rollback paths want them.
func sortRecordsByName(records []MigrationRecord, descending bool) {
	sort.SliceStable(records, func(i, j int) bool {
		if descending {
			return records[i].Migration > records[j].Migration
		}
		return records[i].Migration < records[j].Migration
	})
}
