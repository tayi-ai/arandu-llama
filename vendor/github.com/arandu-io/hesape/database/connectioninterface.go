package database

import (
	"context"

	"github.com/arandu-io/hesape/database/query"
)

// ConnectionInterface is everything that runs statements on a connection.
//
// Every method takes a context.Context first, because a statement that cannot be
// cancelled holds a server connection for as long as the server likes, and
// returns an error rather than failing silently.
//
// A caller that holds one of these is below the authorization layer, not
// outside it -- see the Connection doc for where the Grant lives and why it is
// not on these methods.
type ConnectionInterface interface {
	// Table returns a query builder against one table, with an optional
	// alias.
	Table(table any, as ...string) *query.Builder

	// Query returns a fresh query builder bound to this connection. Table is
	// unusable without it, and there is no other route to a builder.
	Query() *query.Builder

	// Raw wraps value as a fragment of SQL that the grammar leaves
	// untouched.
	Raw(value any) query.Expression

	// SelectOne runs a query and returns the first row, or false when there
	// was none.
	SelectOne(ctx context.Context, query string, bindings []any, useReadPDO bool) (query.Record, bool, error)

	// Scalar runs a query and returns the first column of the first row.
	Scalar(ctx context.Context, query string, bindings []any, useReadPDO bool) (any, error)

	// Select runs a query and returns every matching row.
	Select(ctx context.Context, query string, bindings []any, useReadPDO bool) ([]query.Record, error)

	// Cursor runs a query and returns a range-over-func iterator that
	// yields the rows one at a time, without holding the whole result set
	// in memory.
	Cursor(ctx context.Context, query string, bindings []any, useReadPDO bool) func(yield func(query.Record, error) bool)

	// Insert runs an insert statement, reporting whether it succeeded.
	Insert(ctx context.Context, query string, bindings []any) (bool, error)

	// Update runs an update statement and returns the number of rows it
	// changed.
	Update(ctx context.Context, query string, bindings []any) (int64, error)

	// Delete runs a delete statement and returns the number of rows it
	// removed.
	Delete(ctx context.Context, query string, bindings []any) (int64, error)

	// Statement runs a statement that returns neither rows nor an
	// affected-row count, reporting whether it succeeded.
	Statement(ctx context.Context, query string, bindings []any) (bool, error)

	// AffectingStatement runs a statement and returns the number of rows it
	// affected.
	AffectingStatement(ctx context.Context, query string, bindings []any) (int64, error)

	// Unprepared runs a statement as it is, with no bindings to prepare.
	Unprepared(ctx context.Context, query string) (bool, error)

	// PrepareBindings converts each binding into the value a driver
	// accepts.
	PrepareBindings(bindings []any) []any

	// Transaction runs callback inside a transaction, retrying up to
	// attempts times on a deadlock. A Go method cannot be generic, so the
	// callback carries its result out through the closure.
	Transaction(callback func() error, attempts int) error

	// BeginTransaction opens a new transaction, or a nested savepoint if
	// one is already open.
	BeginTransaction() error

	// Commit commits the current transaction, or releases a savepoint.
	Commit() error

	// RollBack rolls the transaction back to toLevel, or back one level
	// when toLevel is nil.
	RollBack(toLevel *int) error

	// TransactionLevel reports how many transactions are currently nested.
	TransactionLevel() int

	// Pretend runs callback without executing its statements, and returns
	// the log of what would have run.
	Pretend(ctx context.Context, callback func(*Connection) error) ([]QueryLogEntry, error)

	// GetDatabaseName returns the name of the database this connection is
	// open on.
	GetDatabaseName() string
}

// Connection satisfies ConnectionInterface, asserted here at compile time,
// which is the only place a claim like that is worth making.
var _ ConnectionInterface = (*Connection)(nil)
