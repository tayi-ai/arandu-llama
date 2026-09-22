package migrations

import (
	"context"

	"github.com/arandu-io/hesape/database/schema"
)

// Connection is what a migration runs against.
//
// It is narrowed to the two calls a schema change makes. The interface is
// declared here rather than imported because the database package resolves
// migrations, and naming the concrete type would close the cycle.
//
// There is no auth.Grant on it, and that is a decision rather than an
// oversight. The path to application rows is the one that needs one: every
// List, Find, Get, Paginate, report and export goes through a Policy and filters
// by
// auth.Tenant(g). A migration is not on that path -- it is DDL, run by `aru
// migrate` as a pipeline step, in a process with no request, no subject and
// therefore nothing a Grant could be built from. Giving one to a migration
// would mean inventing a fake subject, which is worse than not having one.
type Connection interface {
	// GetName returns the connection's name.
	GetName() string

	// Schema returns the schema builder this migration writes DDL with.
	//
	// It is the standard path, and Statement is the escape: a CREATE INDEX
	// CONCURRENTLY, an extension, a backfill that reads before it writes.
	// Neither is a second kind of migration -- Up has one signature, and it is
	// the one above.
	//
	// It takes no auth.Grant, and neither does anything it returns. DDL names a
	// table, not a row: there is no tenant to scope by and no subject to
	// attribute to, so the only Grant this path could hold is one somebody
	// invented. The path to application rows is the one that needs a Grant, and
	// it still has one on every method.
	Schema() *schema.Builder

	// Statement runs a statement that returns neither rows nor a count,
	// which is every DDL statement there is.
	Statement(ctx context.Context, query string, bindings []any) (bool, error)

	// Select runs a query, for the migration that has to read before it
	// writes -- a backfill, a check that a column is empty before dropping
	// it.
	Select(ctx context.Context, query string, bindings []any) ([]map[string]any, error)
}

// Migration is one schema change.
//
// The whole surface is this interface, and BaseMigration answers the half that
// is the same for every migration -- embed it and only Up, Down and GetName are
// left to write.
//
// GetName exists because a migration is code and code has no path to read a name
// off: it says its own name, and that string is what lands in the repository
// table.
type Migration interface {
	// GetName is the migration's identity: "2026_08_11_000000_create_users_table".
	//
	// The date prefix is not decoration. It is what makes two machines apply
	// the same migrations in the same order, and it is the whole of the
	// ordering rule -- the registry sorts by this string and nothing else.
	GetName() string

	// Up applies the change.
	Up(ctx context.Context, conn Connection) error

	// GetConnection returns the connection this migration runs on, empty for
	// the default.
	GetConnection() string

	// ShouldRun reports whether the migration should run at all. A
	// migration that returns false is skipped and NOT recorded, so it is
	// reconsidered on the next run.
	ShouldRun() bool

	// WithinTransaction reports whether the migration should run inside a
	// transaction.
	//
	// It is a method rather than a field because Go's zero value for a bool
	// is false and the conventional default is true: a struct field would
	// turn every migration that did not think about it into one that runs
	// unprotected.
	WithinTransaction() bool
}

// ReversibleMigration is a Migration that can be rolled back.
//
// A type assertion is what tests for it: a migration without a matching
// Down is simply not reversed. That happens at compile time for the
// migration and at run time for the Migrator -- which is strictly better
// than a dynamic method check, because a Down with the wrong signature is a
// build failure rather than a rollback that silently does nothing.
type ReversibleMigration interface {
	Migration

	// Down reverses what Up applied.
	Down(ctx context.Context, conn Connection) error
}

// IrreversibleMigration is a Migration that declares nothing can undo it.
//
// A rollback that reaches one leaves it applied, says so with the reason, and
// carries on with the rest of the batch. Its record is kept, because the record
// is what says the change is applied and it still is -- deleting it would send
// the next migrate through Up a second time.
//
// It is for the change that genuinely has no inverse: a backfill whose rows are
// no longer distinguishable from the rows it did not touch. It is not a way out
// of writing a Down. A migration that created a table and declares this turns
// every rollback of its batch into a no-op, and the next deploy of the previous
// binary meets a schema nobody undid.
//
// Declaring it is what separates "cannot be undone" from "nobody wrote the
// Down yet", which are the same thing to a compiler and opposite things at
// three in the morning. A migration that declares neither is refused by the
// rollback rather than reported as undone, and refused before any of its batch
// is undone rather than on being reached.
//
// Declaring both this and Down is refused too: the two say opposite things
// about the same migration, and nothing outside the migration can tell which
// one its author meant.
type IrreversibleMigration interface {
	Migration

	// Irreversible returns why the change cannot be undone. The rollback
	// prints it beside the migration's name, so it is what somebody reads
	// while deciding what to do instead.
	Irreversible() string
}

// BaseMigration is the half of a Migration that is the same for every one of
// them: the connection name, the guard, and whether it runs in a transaction.
//
// A migration embeds it and writes GetName and Up:
//
//	type CreateUsersTable struct{ migrations.BaseMigration }
//
//	func (CreateUsersTable) GetName() string {
//	    return "2026_08_11_000000_create_users_table"
//	}
//
//	func (CreateUsersTable) Up(ctx context.Context, conn migrations.Connection) error {
//	    _, err := conn.Statement(ctx, `CREATE TABLE users (...)`, nil)
//	    return err
//	}
type BaseMigration struct {
	// Connection is the connection to run on, empty for the default. It is
	// exported because a migration sets it directly.
	Connection string

	// OutsideTransaction inverts WithinTransaction.
	//
	// The conventional default for running inside a transaction is true, and
	// a Go bool defaults to false, so carrying the name over directly would
	// have made "I did not think about this" mean "do not wrap this in a
	// transaction" -- the opposite of the intended default. The flag is
	// inverted so the zero value keeps the safe default, and the name says
	// which way it points. Set it for the statement an engine refuses inside
	// a transaction: CREATE INDEX CONCURRENTLY is the one that bites first.
	OutsideTransaction bool
}

// GetConnection returns m.Connection.
func (m BaseMigration) GetConnection() string { return m.Connection }

// ShouldRun returns true unless a migration overrides it to say otherwise.
func (m BaseMigration) ShouldRun() bool { return true }

// WithinTransaction returns the opposite of m.OutsideTransaction.
func (m BaseMigration) WithinTransaction() bool { return !m.OutsideTransaction }

// MigrationResult is how one migration turned out: Success, Failure or
// Skipped. Its String is the word the Migrator prints beside the migration's
// name.
//
// Skipped is the case worth knowing about, and it is not a failure: a
// migration whose ShouldRun returns false is passed over deliberately -- the
// guard exists for a migration that only applies to one engine, or one
// deployment -- and the run carries on.
type MigrationResult int

// The three cases of MigrationResult.
const (
	// Success is a migration that ran without error.
	Success MigrationResult = 1
	// Failure is a migration that returned an error.
	Failure MigrationResult = 2
	// Skipped is what a migration whose ShouldRun returns false gets instead
	// of running.
	Skipped MigrationResult = 3
)

// String is the label the console prints.
func (r MigrationResult) String() string {
	switch r {
	case Success:
		return "Success"
	case Failure:
		return "Failure"
	case Skipped:
		return "Skipped"
	default:
		return "Unknown"
	}
}
