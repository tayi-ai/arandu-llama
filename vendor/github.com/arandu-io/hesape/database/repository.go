package database

import (
	"context"
	"crypto/rand"
	"database/sql"
	"errors"
	"fmt"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/database/query"
	"github.com/arandu-io/hesape/log"
)

// ErrNotFound is what Find returns when the row is not there.
//
// It is a sentinel rather than sql.ErrNoRows for two reasons. A repository that
// leaks sql.ErrNoRows makes every caller import database/sql to compare against
// it, and the exception classifier turns this one into a 404 -- so a module that
// returns it gets the right status without writing a status anywhere.
//
// A repository wraps it (fmt.Errorf("%w: invoice %s", database.ErrNotFound, id))
// so errors.Is keeps working and the message still says which row.
var ErrNotFound = errors.New("database: no such row")

// Repository is the contract every module repository implements.
//
// Look at the signature: auth.Grant is mandatory and comes before the id, on a
// read exactly as on a write.
//
// # Three things hold this up, and they are not the same thing
//
// The compiler guarantees you hold a Grant. auth.Grant has only unexported
// fields, so it cannot be written as a struct literal: a call to any method here
// that omits it does not compile, and neither does one that tries to build a
// valid one by hand.
//
// `aru doctor` guarantees it is the right Grant. auth.Authorize is the path
// where a Policy answered, and it is not the only exported way to obtain one --
// auth.SystemGrant issues a Grant for work that has no subject, and the queue's
// GrantFor reissues one for a job. So a handler can hold a Grant no Policy was
// ever asked about. What reports that is a lint, not the type system.
//
// Convention guarantees you came through here at all. DB.QueryContext,
// DB.ExecContext and DB.QueryRowContext are exported, take no Grant, and reach
// the same rows. They take none because they could not use one: they are handed
// a finished string, and a parameter that looks like enforcement while filtering
// nothing is worse than no parameter. So a module reads rows through a
// Repository, and a module that reaches them through the connection is a module
// that gets sent back in review.
//
// This comment used to say that a Grant cannot be constructed outside the auth
// package, and therefore that no path from a handler to the database skips a
// Policy. Neither half was true, and stating it wrong here costs more than
// elsewhere: this is the interface every module implements, so it is the doc a
// reader checks the claim against.
type Repository[T any, ID comparable] interface {
	Find(ctx context.Context, g auth.Grant, id ID) (T, error)
	List(ctx context.Context, g auth.Grant, q Query) (Page[T], error)
	Create(ctx context.Context, g auth.Grant, entity T) (T, error)
	Update(ctx context.Context, g auth.Grant, entity T) (T, error)
	Delete(ctx context.Context, g auth.Grant, id ID) error
}

// Query is pagination and ordering with an allowlist. The sort field is NEVER
// interpolated directly: the repository validates it against a permitted set,
// or ordering becomes injection through another door.
//
// There is no Filter here, and that is a decision rather than an omission.
//
// The field existed, exported, with no producer and no consumer in any of the
// ten repositories: List(ctx, g, Query{Filter: ...}) returned the whole list,
// with no error and no warning. A field that silently does nothing is worse than
// one that does not exist -- the reader assumes it filtered, and in an
// application where rows belong to tenants that assumption is how a leak starts.
//
// Filling it in would mean a generic predicate language over columns, which is a
// query builder, which is a second way to reach data. A module that needs to
// read by something other than the id declares
// the method it needs on its own repository, with the SQL written out and the
// values in placeholders.
type Query struct {
	Limit  int
	Cursor string
	Sort   string
}

// Page is what List returns: the rows, and the cursor that reaches the next
// page.
//
// It replaced a bare []T, which could not say whether there was more. The
// caller had to infer it from len(items) == q.Limit, which is wrong exactly
// once per result set -- on the page whose last row is the last row.
//
// Next is opaque and goes straight back into Query.Cursor. It is the empty
// string on the last page, which is the whole of the "is there more" question:
//
//	for q := database.Query{Limit: 100}; ; {
//	    page, err := repo.List(ctx, g, q)
//	    ...
//	    if page.Next == "" {
//	        break
//	    }
//	    q.Cursor = page.Next
//	}
//
// It is not a paginator and does not become one. The links, the page numbers and
// the per-page window a template draws live in hesape/pagination, which builds
// them from rows somebody already fetched. This type is the repository's half:
// what came back, and where to resume.
type Page[T any] struct {
	Items []T
	Next  string
}

// DB wraps *sql.DB to instrument the Collector and to rebind placeholders for
// the connection's dialect. Repositories use this type rather than *sql.DB, which
// is what makes every query show up on the debug page with the file and line that
// issued it.
//
// It holds no driver import: the driver is chosen by the application, so the
// core keeps its two dependencies.
type DB struct {
	inner   *sql.DB
	dialect Dialect

	// grammar and processor are what make this handle a model connection: the
	// grammar its statements compile through and the processor their results
	// are read back through. See dbmodel.go.
	//
	// They are resolved once, here, rather than per statement, because a
	// grammar is a value with a self reference and building one per query
	// would allocate on every read.
	grammar   query.Grammar
	processor query.Processor
}

// Wrap returns an instrumented handle over an open *sql.DB.
//
// The dialect is what queries written with "?" are rebound to. An empty dialect
// means SQLite, which is the development default.
func Wrap(db *sql.DB, dialect Dialect) *DB {
	if dialect == "" {
		dialect = DialectSQLite
	}

	wrapped := &DB{inner: db, dialect: dialect}
	if DefaultQueryGrammar != nil {
		wrapped.grammar = DefaultQueryGrammar(dialect)
	}
	if DefaultPostProcessor != nil {
		wrapped.processor = DefaultPostProcessor(dialect)
	}
	return wrapped
}

// Dialect reports the flavour this handle speaks. Repositories use it only when
// a statement genuinely cannot be written portably -- which should be rare, and
// is a smell worth explaining in a comment when it happens.
func (d *DB) Dialect() Dialect { return d.dialect }

// Unwrap returns the underlying handle, for the rare case that needs a driver
// specific feature. Prefer the wrapper: what goes through Unwrap does not show
// up on the debug page.
func (d *DB) Unwrap() *sql.DB { return d.inner }

// PingContext verifies the connection. It feeds module health checks.
func (d *DB) PingContext(ctx context.Context) error { return d.inner.PingContext(ctx) }

// QueryContext runs a query and records it.
//
// Inside database.Transaction it runs on the open transaction. That is what lets
// a repository written once work in both places, and what puts the outbox write
// in the same transaction as the row it describes.
func (d *DB) QueryContext(ctx context.Context, query string, args ...any) (*sql.Rows, error) {
	if tx, ok := txFrom(ctx, d); ok {
		return tx.queryContext(ctx, query, args...)
	}
	query = d.dialect.Rebind(query)
	start := time.Now()
	rows, err := d.inner.QueryContext(ctx, query, args...)
	log.FromContext(ctx).RecordQuery(query, args, time.Since(start), -1, err)
	return rows, err
}

// ExecContext runs a statement and records it, with the affected row count.
//
// Inside database.Transaction it runs on the open transaction.
func (d *DB) ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error) {
	if tx, ok := txFrom(ctx, d); ok {
		return tx.execContext(ctx, query, args...)
	}
	query = d.dialect.Rebind(query)
	start := time.Now()
	res, err := d.inner.ExecContext(ctx, query, args...)
	rows := -1
	if err == nil && res != nil {
		if n, e := res.RowsAffected(); e == nil {
			rows = int(n)
		}
	}
	log.FromContext(ctx).RecordQuery(query, args, time.Since(start), rows, err)
	return res, err
}

// QueryRowContext runs a single-row query and records it.
//
// The duration measured here covers issuing the query only: database/sql defers
// the actual work to Row.Scan, so a slow row shows up on the timeline as scan
// time rather than query time.
func (d *DB) QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row {
	if tx, ok := txFrom(ctx, d); ok {
		return tx.queryRowContext(ctx, query, args...)
	}
	query = d.dialect.Rebind(query)
	start := time.Now()
	row := d.inner.QueryRowContext(ctx, query, args...)
	log.FromContext(ctx).RecordQuery(query, args, time.Since(start), 1, nil)
	return row
}

// BeginTx starts a raw transaction on the underlying handle.
//
// Prefer database.Transaction: statements run through this one are invisible to
// the Collector, and the outbox refuses to store events on it, because nothing
// connects it to the context that repositories read.
func (d *DB) BeginTx(ctx context.Context, opts *sql.TxOptions) (*sql.Tx, error) {
	return d.inner.BeginTx(ctx, opts)
}

// NewID returns a version 4 UUID as text.
//
// Ids are generated by the application, not by the database: gen_random_uuid,
// UUID() and randomblob are three different spellings of the same idea, and
// depending on any of them would tie the schema to one engine.
func NewID() (string, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", fmt.Errorf("database: reading random bytes: %w", err)
	}
	b[6] = (b[6] & 0x0f) | 0x40 // version 4
	b[8] = (b[8] & 0x3f) | 0x80 // variant 10
	return fmt.Sprintf("%x-%x-%x-%x-%x", b[0:4], b[4:6], b[6:8], b[8:10], b[10:16]), nil
}
