package database

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/arandu-io/hesape/log"
)

// txKey carries the open transaction on the context.
//
// On the context rather than in every repository signature, for one reason: the
// Repository contract already takes a Grant, and adding a second required
// argument to five methods would make every generated repository exist in two
// shapes -- one for transactions and one for outside them.
//
// It is also how the outbox works at all: events.Outbox.Store writes through the
// same DB handle the repository just wrote through, and lands in the same
// transaction without anyone threading a handle through the call.
//
// The key carries the handle the transaction belongs to. It used to be an empty
// struct, so one context held one transaction no matter how many databases the
// application had: a write issued through the analytics handle inside a
// transaction on the primary joined the primary's transaction and executed
// against the wrong database, silently and with no error. Found by audit.
type txKey struct{ db *DB }

// Tx is an instrumented transaction.
//
// Statements run through it are recorded on the Collector exactly like the ones
// outside, which matters more than it sounds: a query that only misbehaves
// inside a transaction is the one nobody can see on the debug page.
type Tx struct {
	inner   *sql.Tx
	dialect Dialect

	// The work waiting on this transaction committing. It is held here rather
	// than on the context because only the outermost transaction builds a Tx --
	// a nested one joins this same value -- so a callback registered at any
	// depth is a callback this one owns and this one runs.
	mu          sync.Mutex
	afterCommit []func(context.Context)
}

// addAfterCommit queues work for when this transaction commits.
func (t *Tx) addAfterCommit(fn func(context.Context)) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.afterCommit = append(t.afterCommit, fn)
}

// takeAfterCommit hands over the queue and empties it.
//
// Emptied, so a callback cannot run twice however this transaction ends, and
// taken under the lock because the queue is appended to by whatever ran inside
// the transaction.
func (t *Tx) takeAfterCommit() []func(context.Context) {
	t.mu.Lock()
	defer t.mu.Unlock()
	queued := t.afterCommit
	t.afterCommit = nil
	return queued
}

// Transaction runs fn inside a database transaction.
//
// Every statement issued through the same *DB while fn runs joins it, because
// the transaction travels on the context. Returning an error rolls back;
// returning nil commits. A panic rolls back and keeps panicking -- swallowing it
// would leave the caller believing the write happened.
//
// A Transaction inside a Transaction joins the outer one rather than opening a
// second. There are no savepoints: partial rollback is a second failure mode for
// the same operation, and the shape this framework wants is one write, one
// outcome.
func Transaction(ctx context.Context, db *DB, fn func(context.Context) error) error {
	return TransactionAt(ctx, db, sql.LevelDefault, fn)
}

// TransactionAt runs fn inside a transaction opened at the named isolation
// level.
//
// It is Transaction with the level stated instead of inherited, and everything
// said there holds here: the transaction travels on the context, returning an
// error rolls back, a panic keeps panicking, and a transaction inside a
// transaction joins the outer one.
//
// # Why the level is named when the transaction opens
//
// The alternative is a SET statement as the first thing inside it, and that
// alternative is not portable. PostgreSQL takes SET TRANSACTION ISOLATION LEVEL
// inside an open transaction; MySQL refuses it there -- the characteristics of a
// transaction cannot be changed once it is in progress -- and would need SET
// SESSION instead, which outlives the transaction and rides the connection back
// into the pool set for everybody. Naming it here hands the level to BeginTx,
// where the driver of each engine spells it the way that engine takes.
//
// # A joined transaction keeps the level it was opened at
//
// Where the context already carries a transaction on this handle, fn joins it
// and the level asked for here is not applied. Changing the level of a
// transaction somebody else opened would be deciding about statements this call
// cannot see, and on MySQL it is not expressible at all. A caller that needs a
// guarantee from the level opens the transaction itself.
//
// sql.LevelDefault is what Transaction passes, and it means the engine's own
// default -- which is not the same on every engine, and is a setting an operator
// can change for a whole cluster. Code whose correctness rests on what a
// predicate is evaluated against while another transaction changes the same row
// names the level rather than inheriting it.
func TransactionAt(ctx context.Context, db *DB, level sql.IsolationLevel, fn func(context.Context) error) error {
	if db == nil {
		return errors.New("database: Transaction needs a database handle")
	}
	// Reentrant per handle: a transaction on this database joins; one on a
	// different database opens its own, because they are different databases.
	if _, ok := txFrom(ctx, db); ok {
		return fn(ctx)
	}

	var opts *sql.TxOptions
	if level != sql.LevelDefault {
		opts = &sql.TxOptions{Isolation: level}
	}
	inner, err := db.inner.BeginTx(ctx, opts)
	if err != nil {
		return fmt.Errorf("beginning the transaction: %w", err)
	}
	tx := &Tx{inner: inner, dialect: db.dialect}

	committed := false
	defer func() {
		if committed {
			return
		}
		// A rollback that fails after the caller already failed has nothing left
		// to report to: the original error is the one worth keeping.
		_ = inner.Rollback()
	}()

	if err := fn(context.WithValue(ctx, txKey{db}, tx)); err != nil {
		return err
	}

	if err := inner.Commit(); err != nil {
		return fmt.Errorf("committing: %w", err)
	}
	committed = true

	// Only here, and only on this line's side of it. Everything the transaction
	// wrote is durable now, so what these were waiting for has happened.
	runAfterCommit(ctx, db, tx.takeAfterCommit())
	return nil
}

// AfterCommit runs fn once the outermost transaction on db has committed.
//
// It is how work that must not happen twice, and must not happen at all if the
// write did not, is attached to a write: a notification, a cache invalidation,
// a job handed to a queue. Registered at any depth, it belongs to the outermost
// transaction, because that is the one whose commit makes anything durable.
//
// Outside a transaction it runs immediately, which is the same promise kept
// under the only circumstances there are: the write it is about has already
// happened.
//
// A rollback discards it. So does a panic, because the deferred rollback is
// what runs. There is no callback for that case here -- a caller that needs to
// know a transaction failed reads the error it returned.
//
// fn is handed a context on which InTransaction reports false, so a callback
// cannot mistake itself for part of the write and cannot open a statement that
// joins a transaction that has ended.
//
// # This is not durable delivery
//
// A process that dies between the commit and the callback loses it, and no
// amount of ordering inside one process fixes that. What after-commit removes
// is the announcement of a write that was rolled back; what it does not add is
// a guarantee that the announcement arrives. Work that has to arrive is written
// into the same transaction as the row and read out of the database afterwards
// -- an outbox -- and deferring a callback is not that, however carefully it is
// deferred.
func AfterCommit(ctx context.Context, db *DB, fn func(context.Context)) error {
	if db == nil {
		return errors.New("database: AfterCommit needs a database handle")
	}
	if fn == nil {
		return errors.New("database: AfterCommit needs something to run")
	}

	tx, ok := txFrom(ctx, db)
	if !ok {
		fn(withoutTransaction(ctx, db))
		return nil
	}
	tx.addAfterCommit(fn)
	return nil
}

// runAfterCommit runs what a committed transaction left waiting.
//
// Every one of them runs, in the order they were registered, and one that
// panics does not stop the rest: the commit already happened, so a callback
// cannot undo it, and letting the first failure swallow the others would hide
// work that had nothing to do with it.
func runAfterCommit(ctx context.Context, db *DB, queued []func(context.Context)) {
	if len(queued) == 0 {
		return
	}
	after := withoutTransaction(ctx, db)
	for _, fn := range queued {
		func() {
			defer func() {
				if r := recover(); r != nil {
					log.FromContext(ctx).RecordQuery(
						"after-commit callback panicked", []any{r}, 0, -1, nil)
				}
			}()
			fn(after)
		}()
	}
}

// withoutTransaction is ctx with the transaction on db reported as ended.
func withoutTransaction(ctx context.Context, db *DB) context.Context {
	return context.WithValue(ctx, txKey{db}, (*Tx)(nil))
}

// InTransaction reports whether the context is inside a transaction on db.
//
// The outbox uses it to refuse to store an event outside a transaction, which is
// the whole guarantee: an event written next to a row that rolled back is worse
// than no event at all.
//
// It takes the handle because "in a transaction" is only meaningful about one
// database. An outbox on the analytics handle is not protected by a transaction
// open on the primary.
func InTransaction(ctx context.Context, db *DB) bool {
	_, ok := txFrom(ctx, db)
	return ok
}

// txFrom returns the open transaction on db, if any.
//
// A nil under the key is not a transaction: it is how the context handed to an
// after-commit callback says that the transaction it is about has ended. The
// alternative would be a context with the key removed, and a context cannot
// have a value removed.
func txFrom(ctx context.Context, db *DB) (*Tx, bool) {
	tx, ok := ctx.Value(txKey{db}).(*Tx)
	return tx, ok && tx != nil
}

func (t *Tx) queryContext(ctx context.Context, query string, args ...any) (*sql.Rows, error) {
	query = t.dialect.Rebind(query)
	start := time.Now()
	rows, err := t.inner.QueryContext(ctx, query, args...)
	log.FromContext(ctx).RecordQuery(query, args, time.Since(start), -1, err)
	return rows, err
}

func (t *Tx) execContext(ctx context.Context, query string, args ...any) (sql.Result, error) {
	query = t.dialect.Rebind(query)
	start := time.Now()
	res, err := t.inner.ExecContext(ctx, query, args...)
	rows := -1
	if err == nil && res != nil {
		if n, e := res.RowsAffected(); e == nil {
			rows = int(n)
		}
	}
	log.FromContext(ctx).RecordQuery(query, args, time.Since(start), rows, err)
	return res, err
}

func (t *Tx) queryRowContext(ctx context.Context, query string, args ...any) *sql.Row {
	query = t.dialect.Rebind(query)
	start := time.Now()
	row := t.inner.QueryRowContext(ctx, query, args...)
	log.FromContext(ctx).RecordQuery(query, args, time.Since(start), 1, nil)
	return row
}
