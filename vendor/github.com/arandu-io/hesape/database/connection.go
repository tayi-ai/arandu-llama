package database

import (
	"context"
	"database/sql"
	"database/sql/driver"
	"fmt"
	"reflect"
	"slices"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/arandu-io/hesape/database/concerns"
	dbevents "github.com/arandu-io/hesape/database/events"
	"github.com/arandu-io/hesape/database/query"
)

// Dispatcher is what a connection needs of an event dispatcher, narrowed to the
// one method it calls.
//
// It is declared here rather than imported from hesape/events so that the SQL
// package keeps its own dependency list, which is the same reason every other
// interface in this component is declared where it is consumed.
type Dispatcher interface {
	// Dispatch fires an event.
	Dispatch(event any)
}

// QueryLogEntry is one row of the query log a Connection keeps.
type QueryLogEntry struct {
	// Query is the SQL as it was issued.
	Query string

	// Bindings are the values that went with it.
	Bindings []any

	// Time is how long it took, in milliseconds, the unit every listener
	// compares against.
	Time float64

	// ReadWriteType is "read", "write", or empty.
	ReadWriteType string
}

// Connection is one open connection, and everything that runs statements
// through it.
//
// # Where the Grant is, and why it is not on these methods
//
// The door application code goes through is Repository, one file over, whose
// every method takes an auth.Grant and filters by auth.Tenant(g) -- on the way
// out as much as on the way in.
//
// A Connection is the plumbing underneath that door, and it takes no Grant on
// purpose. It could not use one: a Grant is enforced by filtering a query by
// tenant, and this type is handed a finished string. A parameter that looks
// like enforcement and enforces nothing is worse than no parameter, because the
// next reader stops checking -- the same reasoning that removed Query.Filter.
// The other half of the argument is that `aru migrate` runs here, in a process
// with no request and no subject, where a Grant cannot be constructed at all.
//
// So what keeps rows behind a Policy at this level is convention, not the type
// system: Select, Statement, Insert and the rest are exported, take no Grant,
// and reach the same rows a Repository does. Reach them through a Repository.
// Reaching them through a Connection is how a module gets rejected in review.
//
// # A connection is a pool
//
// The write handle and the read handle are both *sql.DB, which is a pool rather
// than a socket -- so "reconnect" means replacing the pool, and the read/write
// split is two pools rather than two connections.
type Connection struct {
	// ManagesTransactions embeds the concerns.ManagesTransactions
	// implementation: Transaction, BeginTransaction, Commit, RollBack,
	// TransactionLevel, AfterCommit and AfterRollBack are all reached through
	// it.
	concerns.ManagesTransactions

	mu sync.RWMutex

	pdo     *sql.DB
	readPDO *sql.DB

	// txConn is the single pooled connection an open transaction is pinned to.
	//
	// A *sql.DB is a pool, and a BEGIN issued on the pool would be followed by
	// statements on whichever other connection happened to be free. Pinning is
	// what makes "the transaction" mean anything here.
	txConn *sql.Conn

	readPDOConfig map[string]any
	database      string
	readWriteType string
	tablePrefix   string
	config        map[string]any

	reconnector func(*Connection) error

	queryGrammar  query.Grammar
	schemaGrammar any
	postProcessor query.Processor

	events Dispatcher

	recordsModified       bool
	readOnWriteConnection bool

	queryLog           []QueryLogEntry
	loggingQueries     bool
	totalQueryDuration float64

	queryDurationHandlers []*queryDurationHandler
	queryListeners        []func(*dbevents.QueryExecuted)

	pretending bool

	beforeExecutingCallbacks []func(query string, bindings []any, connection *Connection)

	latestPDOTypeRetrieved string
}

// queryDurationHandler is one handler registered through
// WhenQueryingForLongerThan, with the threshold it fires at and whether it
// already has run.
type queryDurationHandler struct {
	hasRun    bool
	threshold float64
	handler   func(*Connection, *dbevents.QueryExecuted)
}

// NewConnection creates a Connection over an already-open pool.
//
// A nil pool is valid: it stands for a connection that has not been opened
// yet, and Reconnect is what fills it in.
func NewConnection(pdo *sql.DB, database, tablePrefix string, config map[string]any) *Connection {
	if config == nil {
		config = map[string]any{}
	}
	c := &Connection{
		pdo:         pdo,
		database:    database,
		tablePrefix: tablePrefix,
		config:      config,
	}
	c.UseDefaultQueryGrammar()
	c.UseDefaultPostProcessor()
	c.UseTransactions(c)
	return c
}

// DefaultQueryGrammar is where UseDefaultQueryGrammar gets its grammar.
//
// It holds the shipped grammar for the dialect, and a connector that speaks an
// engine this framework does not ship replaces it from its own init, next to
// the driver it registers. Assigning nil leaves the grammar unset, which a
// connection built for a test that never compiles SQL can do.
var DefaultQueryGrammar func(dialect Dialect) query.Grammar

// UseDefaultQueryGrammar sets the connection's query grammar from
// DefaultQueryGrammar, if one was registered.
func (c *Connection) UseDefaultQueryGrammar() {
	if DefaultQueryGrammar == nil {
		return
	}
	c.SetQueryGrammar(DefaultQueryGrammar(c.dialect()))
}

// UseDefaultSchemaGrammar does nothing: a connection with no driver-specific
// override has no default schema grammar to set.
func (c *Connection) UseDefaultSchemaGrammar() {}

// DefaultPostProcessor is where UseDefaultPostProcessor gets its processor.
//
// It is keyed by dialect like DefaultQueryGrammar, because the processors
// differ by dialect: reading back the identifier of an inserted row is a
// returning clause on one engine and an out-of-band value on another.
var DefaultPostProcessor func(dialect Dialect) query.Processor

// UseDefaultPostProcessor sets the connection's post-processor from
// DefaultPostProcessor, if one was registered.
func (c *Connection) UseDefaultPostProcessor() {
	if DefaultPostProcessor == nil {
		return
	}
	c.SetPostProcessor(DefaultPostProcessor(c.dialect()))
}

// dialect reads the driver name off the configuration and returns the Dialect
// it maps to, which is what DefaultQueryGrammar is keyed by.
func (c *Connection) dialect() Dialect {
	if d, err := ParseDialect(c.GetDriverName()); err == nil {
		return d
	}
	return DialectSQLite
}

// Table returns a query builder against one table, with an optional alias.
//
// It takes no context, and it used to. The context belongs to the statement,
// not to the builder: every terminal method takes one, and a builder is a value
// a caller may hold across more than one of them.
func (c *Connection) Table(table any, as ...string) *query.Builder {
	return c.Query().From(table, as...)
}

// Query returns a fresh query builder on this connection.
func (c *Connection) Query() *query.Builder {
	return query.NewBuilder(&boundConnection{connection: c}, c.GetQueryGrammar(), c.GetPostProcessor())
}

// boundConnection is what makes a Connection usable as a query.Connection.
//
// It used to carry the context too, bound at the moment the builder was
// created. That was one of two doors for a context -- the other being the one
// in every terminal method's signature, which reached only as far as a
// ctx.Err() pre-flight -- and the door that worked was the one nobody could
// see. query.Connection takes the context now, so a statement is cancelled by
// the request that asked for it rather than by whoever built the builder.
type boundConnection struct {
	connection *Connection

	// lastInsertID is what the most recent Insert through this binding was
	// told the engine assigned.
	//
	// Holding it here is safe where holding it on the Connection would not be:
	// a boundConnection is built by Query, once per builder, and never shared
	// between goroutines, while a Connection is the pool and serves every
	// request at once. On the pool, "the last identifier" is whoever inserted
	// most recently, which is the classic way to hand one request another
	// request's row.
	lastInsertID int64
}

func (b *boundConnection) Select(ctx context.Context, q string, bindings []any, useReadPDO bool) ([]query.Record, error) {
	return b.connection.Select(ctx, q, bindings, useReadPDO)
}

func (b *boundConnection) Insert(ctx context.Context, q string, bindings []any) (bool, error) {
	// The identifier is read from the statement that caused it and kept for
	// GetLastInsertID, so that InsertGetID costs one round trip rather than
	// two. An error reporting it is not an error inserting: the row is in, and
	// only a caller that asked for the identifier has a problem -- which is
	// what GetLastInsertID reports, to that caller.
	id, err := b.connection.InsertReturningID(ctx, q, bindings)
	b.lastInsertID = id
	return err == nil, err
}

// GetLastInsertID satisfies processors.LastInsertIDConnection: it returns the
// identifier the most recent insert through this binding was told the engine
// assigned.
//
// The sequence argument names the generator to ask, for an engine with more
// than one. It is unused here because the engines that travel this path --
// MySQL, MariaDB and SQLite -- have one per table and report it from the
// statement. Postgres, which does have named sequences, never arrives: its
// processor compiles RETURNING and reads the value out of the result set.
func (b *boundConnection) GetLastInsertID(sequence string) (int64, error) {
	if b.lastInsertID == 0 {
		return 0, fmt.Errorf(
			"database: no identifier was reported for the last insert on this query. " +
				"The driver may not report one -- lib/pq does not -- in which case the insert needs a RETURNING clause and the processor for that engine")
	}
	return b.lastInsertID, nil
}

func (b *boundConnection) Update(ctx context.Context, q string, bindings []any) (int64, error) {
	return b.connection.Update(ctx, q, bindings)
}

func (b *boundConnection) Delete(ctx context.Context, q string, bindings []any) (int64, error) {
	return b.connection.Delete(ctx, q, bindings)
}

func (b *boundConnection) Statement(ctx context.Context, q string, bindings []any) (bool, error) {
	return b.connection.Statement(ctx, q, bindings)
}

// SelectOne runs a query and returns the first row, or false when there was
// none.
//
// A nil record is indistinguishable from a row of no columns, so the second
// value says which it was.
func (c *Connection) SelectOne(ctx context.Context, q string, bindings []any, useReadPDO bool) (query.Record, bool, error) {
	records, err := c.Select(ctx, q, bindings, useReadPDO)
	if err != nil {
		return nil, false, err
	}
	if len(records) == 0 {
		return nil, false, nil
	}
	return records[0], true, nil
}

// Scalar runs a query and returns the first column of the first row.
//
// More than one column selected is an error: a query that returns two columns
// and is read as one is a query somebody edited without reading its caller.
func (c *Connection) Scalar(ctx context.Context, q string, bindings []any, useReadPDO bool) (any, error) {
	record, found, err := c.SelectOne(ctx, q, bindings, useReadPDO)
	if err != nil || !found {
		return nil, err
	}
	if len(record) > 1 {
		return nil, ErrMultipleColumnsSelected
	}
	for _, value := range record {
		return value, nil
	}
	return nil, nil
}

// SelectFromWriteConnection runs a read against the write pool, for a read
// that must see writes this request already made.
func (c *Connection) SelectFromWriteConnection(ctx context.Context, q string, bindings []any) ([]query.Record, error) {
	return c.Select(ctx, q, bindings, false)
}

// Select runs a query and returns every matching row.
func (c *Connection) Select(ctx context.Context, q string, bindings []any, useReadPDO bool) ([]query.Record, error) {
	var records []query.Record

	err := c.run(ctx, q, bindings, func(runQuery string, runBindings []any) error {
		if c.Pretending() {
			return nil
		}

		pool, err := c.runner(useReadPDO)
		if err != nil {
			return err
		}

		rows, err := pool.QueryContext(ctx, runQuery, c.PrepareBindings(runBindings)...)
		if err != nil {
			return err
		}
		defer func() { _ = rows.Close() }()

		records, err = scanRecords(rows)
		return err
	})

	return records, err
}

// SelectResultSets runs a statement and returns every result set it
// produced, not only the first.
//
// database/sql exposes the extra sets through Rows.NextResultSet. A driver
// that does not support it returns exactly one set.
func (c *Connection) SelectResultSets(ctx context.Context, q string, bindings []any, useReadPDO bool) ([][]query.Record, error) {
	var sets [][]query.Record

	err := c.run(ctx, q, bindings, func(runQuery string, runBindings []any) error {
		if c.Pretending() {
			return nil
		}

		pool, err := c.runner(useReadPDO)
		if err != nil {
			return err
		}

		rows, err := pool.QueryContext(ctx, runQuery, c.PrepareBindings(runBindings)...)
		if err != nil {
			return err
		}
		defer func() { _ = rows.Close() }()

		for {
			set, err := scanRecords(rows)
			if err != nil {
				return err
			}
			sets = append(sets, set)

			if !rows.NextResultSet() {
				return rows.Err()
			}
		}
	})

	return sets, err
}

// Cursor runs a query and returns the rows one at a time, without holding the
// whole result set in memory.
//
// It returns a range-over-func iterator with the same shape as concerns.Lazy:
// an error is yielded once and ends the iteration, so a caller that forgets
// to check it still stops.
//
// It does not go through run, because run wraps a callback that has to finish
// before it returns and this one hands rows back as they arrive. The
// renumbering is therefore repeated here rather than inherited, and it is the
// only place in this file where that is true.
func (c *Connection) Cursor(ctx context.Context, q string, bindings []any, useReadPDO bool) func(yield func(query.Record, error) bool) {
	q = c.rebind(q, bindings)

	return func(yield func(query.Record, error) bool) {
		if c.Pretending() {
			return
		}

		pool, err := c.runner(useReadPDO)
		if err != nil {
			yield(nil, err)
			return
		}

		start := time.Now()
		rows, err := pool.QueryContext(ctx, q, c.PrepareBindings(bindings)...)
		c.LogQuery(q, bindings, elapsed(start))
		if err != nil {
			yield(nil, c.wrapQueryError(err, q, bindings))
			return
		}
		defer func() { _ = rows.Close() }()

		columns, err := rows.Columns()
		if err != nil {
			yield(nil, err)
			return
		}

		for rows.Next() {
			record, err := scanRecord(rows, columns)
			if err != nil {
				yield(nil, err)
				return
			}
			if !yield(record, nil) {
				return
			}
		}
		if err := rows.Err(); err != nil {
			yield(nil, err)
		}
	}
}

// Insert runs an insert statement, reporting whether it succeeded.
func (c *Connection) Insert(ctx context.Context, q string, bindings []any) (bool, error) {
	return c.Statement(ctx, q, bindings)
}

// Update runs an update statement and returns the number of rows it changed.
func (c *Connection) Update(ctx context.Context, q string, bindings []any) (int64, error) {
	return c.AffectingStatement(ctx, q, bindings)
}

// Delete runs a delete statement and returns the number of rows it removed.
func (c *Connection) Delete(ctx context.Context, q string, bindings []any) (int64, error) {
	return c.AffectingStatement(ctx, q, bindings)
}

// Statement runs a statement that returns neither rows nor an affected-row
// count, reporting whether it succeeded.
func (c *Connection) Statement(ctx context.Context, q string, bindings []any) (bool, error) {
	ok := false

	err := c.run(ctx, q, bindings, func(runQuery string, runBindings []any) error {
		if c.Pretending() {
			ok = true
			return nil
		}

		pool, err := c.runner(false)
		if err != nil {
			return err
		}

		c.RecordsHaveBeenModified(true)

		_, err = pool.ExecContext(ctx, runQuery, c.PrepareBindings(runBindings)...)
		ok = err == nil
		return err
	})

	return ok, err
}

// InsertReturningID runs an insert and returns the identifier the engine
// assigned to the row.
//
// The statement that caused the identifier already reports it, through
// sql.Result.LastInsertId, with no round trip needed to ask for it separately.
// Asking a second time would be a second way and, on a pooled connection, a
// way that can report somebody else's row.
//
// Postgres does not travel this path at all: its processor compiles the insert
// with a RETURNING clause and reads the identifier out of the result set, which
// is why PostgresProcessor.ProcessInsertGetID calls Select rather than this.
//
// A driver that cannot report one -- lib/pq is the usual case -- returns an
// error from LastInsertId, and it is passed through rather than flattened to a
// zero, because a zero identifier reads like a row.
func (c *Connection) InsertReturningID(ctx context.Context, q string, bindings []any) (int64, error) {
	var id int64

	err := c.run(ctx, q, bindings, func(runQuery string, runBindings []any) error {
		if c.Pretending() {
			return nil
		}

		pool, err := c.runner(false)
		if err != nil {
			return err
		}

		c.RecordsHaveBeenModified(true)

		result, err := pool.ExecContext(ctx, runQuery, c.PrepareBindings(runBindings)...)
		if err != nil {
			return err
		}

		id, err = result.LastInsertId()
		if err != nil {
			return fmt.Errorf("database: the insert ran and the driver cannot report the identifier it assigned: %w", err)
		}
		return nil
	})

	return id, err
}

// AffectingStatement runs a statement and returns the number of rows it
// affected.
func (c *Connection) AffectingStatement(ctx context.Context, q string, bindings []any) (int64, error) {
	var count int64

	err := c.run(ctx, q, bindings, func(runQuery string, runBindings []any) error {
		if c.Pretending() {
			return nil
		}

		pool, err := c.runner(false)
		if err != nil {
			return err
		}

		result, err := pool.ExecContext(ctx, runQuery, c.PrepareBindings(runBindings)...)
		if err != nil {
			return err
		}

		count, err = result.RowsAffected()
		if err != nil {
			// A driver that cannot count is not a failed statement, so the
			// count is simply zero rather than an error.
			count = 0
		}
		c.RecordsHaveBeenModified(count > 0)
		return nil
	})

	return count, err
}

// Unprepared runs the statement as it is, with no prepare and no bindings,
// reporting whether it succeeded.
//
// It is what a schema dump is loaded with, and nothing else should use it: a
// value that reaches a database through this has not been through a
// placeholder.
func (c *Connection) Unprepared(ctx context.Context, q string) (bool, error) {
	changed := false

	err := c.run(ctx, q, nil, func(runQuery string, _ []any) error {
		if c.Pretending() {
			changed = true
			return nil
		}

		pool, err := c.runner(false)
		if err != nil {
			return err
		}

		_, err = pool.ExecContext(ctx, runQuery)
		changed = err == nil
		c.RecordsHaveBeenModified(changed)
		return err
	})

	return changed, err
}

// ThreadCount returns how many connections the server has open, or zero when
// the grammar cannot ask.
func (c *Connection) ThreadCount(ctx context.Context) (int64, error) {
	grammar, ok := c.GetQueryGrammar().(interface{ CompileThreadCount() string })
	if !ok {
		return 0, nil
	}
	q := grammar.CompileThreadCount()
	if q == "" {
		return 0, nil
	}

	value, err := c.Scalar(ctx, q, nil, true)
	if err != nil {
		return 0, err
	}
	return toInt64(value), nil
}

// Pretend runs callback with nothing reaching the server, and returns the
// statements it would have run.
func (c *Connection) Pretend(ctx context.Context, callback func(*Connection) error) ([]QueryLogEntry, error) {
	var log []QueryLogEntry

	err := c.withFreshQueryLog(func() error {
		c.mu.Lock()
		c.pretending = true
		c.mu.Unlock()

		defer func() {
			c.mu.Lock()
			c.pretending = false
			c.mu.Unlock()
		}()

		err := callback(c)

		log = c.GetQueryLog()
		return err
	})

	return log, err
}

// WithoutPretending runs callback for real, even inside a Pretend.
func (c *Connection) WithoutPretending(callback func() error) error {
	c.mu.RLock()
	pretending := c.pretending
	c.mu.RUnlock()

	if !pretending {
		return callback()
	}

	c.mu.Lock()
	c.pretending = false
	c.mu.Unlock()

	defer func() {
		c.mu.Lock()
		c.pretending = true
		c.mu.Unlock()
	}()

	return callback()
}

// withFreshQueryLog runs callback with query logging turned on and the log
// cleared first, restoring the previous logging state afterward.
func (c *Connection) withFreshQueryLog(callback func() error) error {
	c.mu.Lock()
	logging := c.loggingQueries
	c.loggingQueries = true
	c.queryLog = nil
	c.mu.Unlock()

	err := callback()

	c.mu.Lock()
	c.loggingQueries = logging
	c.mu.Unlock()

	return err
}

// BindValues is PrepareBindings under another name: database/sql infers each
// value's wire type from its Go type, so there is no separate type-binding
// step to do -- an int is an int, a byte slice is a blob, and everything else
// is a string as far as the driver is concerned.
func (c *Connection) BindValues(bindings []any) []any { return c.PrepareBindings(bindings) }

// PrepareBindings converts each binding into the value a driver accepts.
//
// A time becomes the string the grammar's date format spells, and a bool
// becomes 0 or 1 -- both because the engines disagree about the wire form and
// the grammar is the one that knows which. A value that spells its own column
// is asked for that spelling first, and what it answers is converted the same
// way anything else is.
func (c *Connection) PrepareBindings(bindings []any) []any {
	return prepareBindings(c.GetQueryGrammar(), bindings)
}

// prepareBindings is the conversion itself, shared with the instrumented
// handle: a statement compiled by a grammar carries the same values whichever
// of the two runs it, so it converts them the same way or the two disagree
// about what a Tuesday is.
//
// A nil grammar falls back to the format every shipped grammar spells a
// timestamp in, which is what a connection built for a test that never
// compiles SQL has.
func prepareBindings(grammar query.Grammar, bindings []any) []any {
	if len(bindings) == 0 {
		return bindings
	}

	layout := "2006-01-02 15:04:05"
	if grammar != nil {
		layout = grammar.GetDateFormat()
	}

	out := make([]any, len(bindings))
	for i, value := range bindings {
		switch v := resolveValuer(value).(type) {
		case time.Time:
			out[i] = v.Format(layout)
		default:
			out[i] = v
		}
	}
	return out
}

// resolveValuer asks a value that spells its own column for that spelling, and
// returns anything else unchanged.
//
// It runs before the conversion above rather than after it, and that ordering
// is the point. database/sql resolves a Valuer too, but it does so downstream of
// this function -- so a type answering a time.Time reached the driver as a
// time.Time while a plain time.Time reached it as the string the grammar
// spells, and the two disagreed about what a Tuesday is. The same held for a
// bool, which one path sent as 0 or 1 and the other as true.
//
// A Value that reports an error is left alone. database/sql asks again and
// reports it against the statement it belongs to, which is where a person can
// see which binding it was; swallowing it here would send NULL instead and
// write a row nobody asked for.
func resolveValuer(value any) any {
	valuer, ok := value.(driver.Valuer)
	if !ok {
		return value
	}

	// A nil pointer whose type declares Value on the value underneath would
	// panic inside the method rather than answer. It is NULL, which is what
	// database/sql answers for the same case.
	if v := reflect.ValueOf(valuer); v.Kind() == reflect.Pointer && v.IsNil() &&
		v.Type().Elem().Implements(valuerType) {
		return nil
	}

	resolved, err := valuer.Value()
	if err != nil {
		return value
	}
	return resolved
}

// valuerType is driver.Valuer as a reflect.Type, taken once rather than per
// binding.
var valuerType = reflect.TypeFor[driver.Valuer]()

// rebind numbers the placeholders of a finished statement for the dialect this
// connection speaks.
//
// Every grammar compiles a placeholder as "?", Postgres included, because a
// grammar builds fragments and cannot know where a fragment will sit in a
// statement it has not finished -- a subquery compiled on its own would start
// its numbering over at $1. So the numbering is done once, here, on the
// finished statement, at the last point every read and every write passes
// through.
//
// A statement carrying no values is left alone. It has no placeholder to
// number, so a "?" in one is an operator -- Postgres spells jsonb containment
// that way -- or an ordinary character in a literal, and renumbering either
// would rewrite SQL somebody wrote by hand. That is also what keeps
// Unprepared, which is how a schema dump is loaded, running the statement as
// it is.
func (c *Connection) rebind(q string, bindings []any) string {
	if len(bindings) == 0 {
		return q
	}
	return c.dialect().Rebind(q)
}

// run wraps callback with the before-hooks, the reconnect, the timing, the
// retry on a lost connection, and the log.
//
// The statement is renumbered before any of that, so the hooks, the query log
// and a QueryException all carry the statement the server was actually sent
// rather than the portable form it was written in.
func (c *Connection) run(ctx context.Context, q string, bindings []any, callback func(string, []any) error) error {
	q = c.rebind(q, bindings)

	c.mu.RLock()
	before := slices.Clone(c.beforeExecutingCallbacks)
	c.mu.RUnlock()

	for _, hook := range before {
		hook(q, bindings, c)
	}

	if err := c.ReconnectIfMissingConnection(); err != nil {
		return err
	}

	start := time.Now()

	err := c.runQueryCallback(q, bindings, callback)
	if err != nil {
		err = c.handleQueryException(err, q, bindings, callback)
	}

	c.LogQuery(q, bindings, elapsed(start))

	return err
}

// runQueryCallback runs callback and turns a driver error into a
// QueryException carrying the statement and its values.
func (c *Connection) runQueryCallback(q string, bindings []any, callback func(string, []any) error) error {
	if err := callback(q, bindings); err != nil {
		return c.wrapQueryError(err, q, bindings)
	}
	return nil
}

// wrapQueryError builds the QueryException, or the unique-constraint subtype
// when the driver said that is what it was.
func (c *Connection) wrapQueryError(err error, q string, bindings []any) error {
	if c.IsUniqueConstraintError(err) {
		return NewUniqueConstraintViolationException(
			c.GetNameWithReadWriteType(), q, c.PrepareBindings(bindings), err,
			c.GetConnectionDetails(), c.latestReadWriteTypeUsed())
	}
	return NewQueryException(
		c.GetNameWithReadWriteType(), q, c.PrepareBindings(bindings), err,
		c.GetConnectionDetails(), c.latestReadWriteTypeUsed())
}

// IsUniqueConstraintError reports whether err came from a unique constraint
// violation, using the UniqueConstraintDetector a connector registered; it is
// false when none was.
//
// It is exported so a driver connection living in another package can call it
// -- a connector sets UniqueConstraintDetector rather than overriding a
// method, which is the same decision as DefaultQueryGrammar.
func (c *Connection) IsUniqueConstraintError(err error) bool {
	if UniqueConstraintDetector == nil {
		return false
	}
	return UniqueConstraintDetector(c.GetDriverName(), err)
}

// UniqueConstraintDetector says whether a driver error was a unique constraint
// violation. A connector sets it next to the driver it registers, because the
// answer is the driver's SQLSTATE and nothing this package can read.
var UniqueConstraintDetector func(driver string, err error) bool

// LogQuery records a query's execution: it dispatches the QueryExecuted
// event, runs the query listeners and duration handlers, and appends to the
// query log when logging is enabled.
func (c *Connection) LogQuery(q string, bindings []any, timeMS float64) {
	c.mu.Lock()
	c.totalQueryDuration += timeMS
	logging := c.loggingQueries
	pretending := c.pretending
	handlers := slices.Clone(c.queryDurationHandlers)
	listeners := slices.Clone(c.queryListeners)
	total := c.totalQueryDuration
	c.mu.Unlock()

	readWriteType := c.latestReadWriteTypeUsed()

	event := dbevents.NewQueryExecuted(q, bindings, timeMS, c, readWriteType)
	c.event(event)

	for _, listener := range listeners {
		listener(event)
	}

	for _, handler := range handlers {
		if !handler.hasRun && total > handler.threshold {
			handler.handler(c, event)
			handler.hasRun = true
		}
	}

	logged := q
	if pretending {
		logged = substituteBindings(q, c.PrepareBindings(bindings))
	}

	if logging {
		c.mu.Lock()
		c.queryLog = append(c.queryLog, QueryLogEntry{
			Query: logged, Bindings: bindings, Time: timeMS, ReadWriteType: readWriteType,
		})
		c.mu.Unlock()
	}
}

// elapsed returns the time since start, in milliseconds, rounded to two
// decimals.
func elapsed(start time.Time) float64 {
	ms := float64(time.Since(start).Microseconds()) / 1000
	return float64(int64(ms*100+0.5)) / 100
}

// WhenQueryingForLongerThan runs the handler once the connection has spent more
// than the threshold on queries.
//
// The threshold is a time.Duration, which is the one type Go has for a
// duration.
func (c *Connection) WhenQueryingForLongerThan(threshold time.Duration, handler func(*Connection, *dbevents.QueryExecuted)) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.queryDurationHandlers = append(c.queryDurationHandlers, &queryDurationHandler{
		threshold: float64(threshold.Microseconds()) / 1000,
		handler:   handler,
	})
}

// AllowQueryDurationHandlersToRunAgain resets every WhenQueryingForLongerThan
// handler so it can fire again.
func (c *Connection) AllowQueryDurationHandlersToRunAgain() {
	c.mu.Lock()
	defer c.mu.Unlock()
	for _, handler := range c.queryDurationHandlers {
		handler.hasRun = false
	}
}

// TotalQueryDuration returns the total time spent on queries so far, in
// milliseconds.
func (c *Connection) TotalQueryDuration() float64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.totalQueryDuration
}

// ResetTotalQueryDuration sets the total query duration back to zero.
func (c *Connection) ResetTotalQueryDuration() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.totalQueryDuration = 0
}

// handleQueryException decides what to do with a query error. Inside a
// transaction it goes straight out, because retrying half a transaction is
// not retrying anything; otherwise it is retried once if it was caused by a
// lost connection.
func (c *Connection) handleQueryException(err error, q string, bindings []any, callback func(string, []any) error) error {
	if c.TransactionLevel() >= 1 {
		return err
	}
	return c.tryAgainIfCausedByLostConnection(err, q, bindings, callback)
}

// tryAgainIfCausedByLostConnection reconnects and retries callback once when
// err was caused by a lost connection, otherwise it returns err unchanged.
func (c *Connection) tryAgainIfCausedByLostConnection(err error, q string, bindings []any, callback func(string, []any) error) error {
	var previous error = err
	if queryErr, ok := err.(*QueryException); ok {
		previous = queryErr.Previous
	}

	if c.CausedByLostConnection(previous) {
		if reconnectErr := c.Reconnect(); reconnectErr != nil {
			return err
		}
		return c.runQueryCallback(q, bindings, callback)
	}
	return err
}

// Reconnect replaces the pool by calling the registered reconnector, or fails
// when none was set.
func (c *Connection) Reconnect() error {
	c.mu.RLock()
	reconnector := c.reconnector
	c.mu.RUnlock()

	if reconnector != nil {
		return reconnector(c)
	}
	return NewLostConnectionException("Lost connection and no reconnector available.")
}

// ReconnectIfMissingConnection reconnects when the connection has no pool
// yet, and does nothing otherwise.
func (c *Connection) ReconnectIfMissingConnection() error {
	c.mu.RLock()
	missing := c.pdo == nil
	c.mu.RUnlock()

	if missing {
		return c.Reconnect()
	}
	return nil
}

// Disconnect clears both the write and the read pool.
func (c *Connection) Disconnect() {
	c.SetPDO(nil)
	c.SetReadPDO(nil)
}

// BeforeExecuting registers a callback that runs before every statement.
func (c *Connection) BeforeExecuting(callback func(query string, bindings []any, connection *Connection)) *Connection {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.beforeExecutingCallbacks = append(c.beforeExecutingCallbacks, callback)
	return c
}

// Listen registers callback to run for every query.
//
// The connection keeps its own list of listeners and calls each one alongside
// dispatching the QueryExecuted event, which is what a listener registered on
// the event dispatcher directly would receive anyway.
func (c *Connection) Listen(callback func(*dbevents.QueryExecuted)) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.queryListeners = append(c.queryListeners, callback)
}

// FireConnectionEvent dispatches the transaction event named by event: one
// of "beganTransaction", "committed", "committing" or "rollingBack".
//
// It is exported because ManagesTransactions calls it from the concerns
// package.
func (c *Connection) FireConnectionEvent(event string) {
	switch event {
	case "beganTransaction":
		c.event(dbevents.NewTransactionBeginning(c))
	case "committed":
		c.event(dbevents.NewTransactionCommitted(c))
	case "committing":
		c.event(dbevents.NewTransactionCommitting(c))
	case "rollingBack":
		c.event(dbevents.NewTransactionRolledBack(c))
	}
}

// event dispatches e on the connection's event dispatcher, if one is set.
func (c *Connection) event(e any) {
	c.mu.RLock()
	dispatcher := c.events
	c.mu.RUnlock()

	if dispatcher != nil {
		dispatcher.Dispatch(e)
	}
}

// Raw wraps value as a fragment of SQL that the grammar leaves untouched.
func (c *Connection) Raw(value any) query.Expression { return query.Raw(value) }

// Escape renders value as a SQL literal.
//
// It is the one method in this file that should almost never be called: a
// value that goes through here did not go through a placeholder. The grammar
// uses it to write a literal into a schema dump, which is the case it exists
// for.
func (c *Connection) Escape(value any, binary bool) (string, error) {
	switch v := value.(type) {
	case nil:
		return "null", nil
	case int, int8, int16, int32, int64, float32, float64:
		return fmt.Sprint(v), nil
	case bool:
		if v {
			return "1", nil
		}
		return "0", nil
	case []byte:
		if !binary {
			return "", fmt.Errorf("Strings with null bytes cannot be escaped. Use the binary escape option.")
		}
		return "X'" + fmt.Sprintf("%x", v) + "'", nil
	case string:
		if binary {
			return "X'" + fmt.Sprintf("%x", v) + "'", nil
		}
		if strings.ContainsRune(v, 0) {
			return "", fmt.Errorf("Strings with null bytes cannot be escaped. Use the binary escape option.")
		}
		return "'" + strings.ReplaceAll(v, "'", "''") + "'", nil
	default:
		return "", fmt.Errorf("The database connection does not support escaping values of this type.")
	}
}

// HasModifiedRecords reports whether a write has gone through this
// connection.
func (c *Connection) HasModifiedRecords() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.recordsModified
}

// RecordsHaveBeenModified sets the modified flag, but only from false to
// true: once a write has happened, nothing clears it but
// ForgetRecordModificationState.
func (c *Connection) RecordsHaveBeenModified(value bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	if !c.recordsModified {
		c.recordsModified = value
	}
}

// SetRecordModificationState sets the modified flag directly, unlike
// RecordsHaveBeenModified, which only ever sets it to true.
func (c *Connection) SetRecordModificationState(value bool) *Connection {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.recordsModified = value
	return c
}

// ForgetRecordModificationState clears the modified flag.
func (c *Connection) ForgetRecordModificationState() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.recordsModified = false
}

// UseWriteConnectionWhenReading sets whether a read should go to the write
// pool instead of the read pool.
func (c *Connection) UseWriteConnectionWhenReading(value bool) *Connection {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.readOnWriteConnection = value
	return c
}

// getPDOForSelect returns the read pool when useReadPDO is true, and the
// write pool otherwise.
func (c *Connection) getPDOForSelect(useReadPDO bool) (*sql.DB, error) {
	if useReadPDO {
		return c.GetReadPDO()
	}
	return c.GetPDO()
}

// sqlRunner is what a statement runs on: the part of *sql.DB and *sql.Conn a
// query needs, so that one call site serves both.
type sqlRunner interface {
	// ExecContext runs a statement that returns no rows.
	ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error)

	// QueryContext runs a query and returns the rows.
	QueryContext(ctx context.Context, query string, args ...any) (*sql.Rows, error)
}

// runner returns what a statement should run on.
//
// The connection an open transaction is pinned to wins over both pools, and
// that is the whole of the rule. A BEGIN issued on one pooled connection and an
// INSERT sent to another is not a transaction: the insert commits on its own,
// and the rollback that was supposed to cover it undoes nothing. Where the pool
// holds a single connection -- SQLite's does -- the same mistake is a deadlock
// instead, because the insert waits for the connection the BEGIN is holding and
// neither one ever gives it up.
func (c *Connection) runner(useReadPDO bool) (sqlRunner, error) {
	c.mu.RLock()
	pinned := c.txConn
	c.mu.RUnlock()

	if pinned != nil {
		return pinned, nil
	}

	pool, err := c.getPDOForSelect(useReadPDO)
	if err != nil {
		return nil, err
	}
	return pool, nil
}

// GetPDO returns the write pool, or an error when the connection has none.
func (c *Connection) GetPDO() (*sql.DB, error) {
	c.mu.Lock()
	c.latestPDOTypeRetrieved = "write"
	pool := c.pdo
	c.mu.Unlock()

	if pool == nil {
		return nil, NewLostConnectionException("Lost connection and no reconnector available.")
	}
	return pool, nil
}

// GetRawPDO returns the write pool as it is, nil included, with no attempt to
// reconnect.
func (c *Connection) GetRawPDO() *sql.DB {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.pdo
}

// GetReadPDO returns the read pool, or the write pool when it is more
// appropriate.
//
// Three conditions send a read to the write pool instead: an open
// transaction, an explicit UseWriteConnectionWhenReading, and a sticky
// connection that has already written. The last is what keeps a
// read-after-write in one request from landing on a replica that has not
// caught up.
func (c *Connection) GetReadPDO() (*sql.DB, error) {
	if c.TransactionLevel() > 0 {
		return c.GetPDO()
	}

	c.mu.RLock()
	onWrite := c.readOnWriteConnection
	modified := c.recordsModified
	readPool := c.readPDO
	c.mu.RUnlock()

	sticky, _ := c.GetConfig("sticky").(bool)
	if onWrite || (modified && sticky) {
		return c.GetPDO()
	}

	if readPool == nil {
		return c.GetPDO()
	}

	c.mu.Lock()
	c.latestPDOTypeRetrieved = "read"
	c.mu.Unlock()

	return readPool, nil
}

// GetRawReadPDO returns the read pool as it is, nil included.
func (c *Connection) GetRawReadPDO() *sql.DB {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.readPDO
}

// SetPDO replaces the write pool. It resets the transaction level, because
// whatever transaction was open belonged to the pool being replaced.
func (c *Connection) SetPDO(pdo *sql.DB) *Connection {
	c.ResetTransactionLevel()

	c.mu.Lock()
	defer c.mu.Unlock()
	c.pdo = pdo
	return c
}

// SetReadPDO replaces the read pool.
func (c *Connection) SetReadPDO(pdo *sql.DB) *Connection {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.readPDO = pdo
	return c
}

// SetReadPDOConfig replaces the configuration reported for the read pool.
func (c *Connection) SetReadPDOConfig(config map[string]any) *Connection {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.readPDOConfig = config
	return c
}

// SetReconnector sets the callback Reconnect calls to replace the pool.
func (c *Connection) SetReconnector(reconnector func(*Connection) error) *Connection {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.reconnector = reconnector
	return c
}

// GetName returns the connection's configured name.
func (c *Connection) GetName() string {
	name, _ := c.GetConfig("name").(string)
	return name
}

// GetNameWithReadWriteType returns the connection's name, suffixed with
// "::read" or "::write" when a read/write type is set.
func (c *Connection) GetNameWithReadWriteType() string {
	c.mu.RLock()
	readWriteType := c.readWriteType
	c.mu.RUnlock()

	name := c.GetName()
	if readWriteType != "" {
		name += "::" + readWriteType
	}
	return name
}

// GetConfig returns one configuration value by key. An empty key returns
// nothing rather than the whole configuration, because a caller that wants
// all of it can ask for the keys it needs.
func (c *Connection) GetConfig(option string) any {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.config[option]
}

// GetConnectionDetails returns the driver, name, host, port, database and
// socket this connection reports, reading from the read configuration when a
// read pool was last used. It is exported because QueryException carries it.
func (c *Connection) GetConnectionDetails() map[string]any {
	c.mu.RLock()
	config := c.config
	if c.latestReadWriteTypeUsedLocked() == "read" && c.readPDOConfig != nil {
		config = c.readPDOConfig
	}
	c.mu.RUnlock()

	return map[string]any{
		"driver":      config["driver"],
		"name":        c.GetNameWithReadWriteType(),
		"host":        config["host"],
		"port":        config["port"],
		"database":    config["database"],
		"unix_socket": config["unix_socket"],
	}
}

// GetDriverName returns the configured driver name.
func (c *Connection) GetDriverName() string {
	driver, _ := c.GetConfig("driver").(string)
	return driver
}

// GetDriverTitle returns the human-facing name of the driver, which by
// default is just the driver name.
func (c *Connection) GetDriverTitle() string { return c.GetDriverName() }

// GetQueryGrammar returns the grammar this connection compiles queries with.
func (c *Connection) GetQueryGrammar() query.Grammar {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.queryGrammar
}

// SetQueryGrammar replaces the grammar this connection compiles queries with.
func (c *Connection) SetQueryGrammar(grammar query.Grammar) *Connection {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.queryGrammar = grammar
	return c
}

// GetSchemaGrammar returns the schema grammar this connection holds.
//
// It is any because the schema grammars live in database/schema/grammars,
// which nothing here should import: a connection needs to hold one and never
// to call it, and the schema builder that does call it knows the type.
func (c *Connection) GetSchemaGrammar() any {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.schemaGrammar
}

// SetSchemaGrammar replaces the schema grammar this connection holds.
func (c *Connection) SetSchemaGrammar(grammar any) *Connection {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.schemaGrammar = grammar
	return c
}

// GetPostProcessor returns the processor this connection post-processes
// query results with.
func (c *Connection) GetPostProcessor() query.Processor {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.postProcessor
}

// SetPostProcessor replaces the processor this connection post-processes
// query results with.
func (c *Connection) SetPostProcessor(processor query.Processor) *Connection {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.postProcessor = processor
	return c
}

// GetEventDispatcher returns the dispatcher this connection fires events on.
func (c *Connection) GetEventDispatcher() Dispatcher {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.events
}

// SetEventDispatcher replaces the dispatcher this connection fires events on.
func (c *Connection) SetEventDispatcher(dispatcher Dispatcher) *Connection {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = dispatcher
	return c
}

// UnsetEventDispatcher clears the event dispatcher, so events stop firing.
func (c *Connection) UnsetEventDispatcher() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.events = nil
}

// Pretending reports whether the connection is inside a Pretend call.
func (c *Connection) Pretending() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.pretending
}

// GetQueryLog returns a copy of the queries logged so far.
func (c *Connection) GetQueryLog() []QueryLogEntry {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return slices.Clone(c.queryLog)
}

// GetRawQueryLog returns the query log with each statement's bindings
// written directly into it.
func (c *Connection) GetRawQueryLog() []QueryLogEntry {
	entries := c.GetQueryLog()
	out := make([]QueryLogEntry, 0, len(entries))
	for _, entry := range entries {
		out = append(out, QueryLogEntry{
			Query:         substituteBindings(entry.Query, c.PrepareBindings(entry.Bindings)),
			Time:          entry.Time,
			ReadWriteType: entry.ReadWriteType,
		})
	}
	return out
}

// FlushQueryLog clears the query log.
func (c *Connection) FlushQueryLog() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.queryLog = nil
}

// EnableQueryLog turns query logging on.
func (c *Connection) EnableQueryLog() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.loggingQueries = true
}

// DisableQueryLog turns query logging off.
func (c *Connection) DisableQueryLog() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.loggingQueries = false
}

// Logging reports whether query logging is on.
func (c *Connection) Logging() bool {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.loggingQueries
}

// GetDatabaseName returns the name of the database this connection is open
// on.
func (c *Connection) GetDatabaseName() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.database
}

// SetDatabaseName replaces the name of the database this connection is open
// on.
func (c *Connection) SetDatabaseName(database string) *Connection {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.database = database
	return c
}

// SetReadWriteType sets the read/write suffix GetNameWithReadWriteType
// reports.
func (c *Connection) SetReadWriteType(readWriteType string) *Connection {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.readWriteType = readWriteType
	return c
}

// latestReadWriteTypeUsed returns the configured read/write type, or which
// pool was last retrieved when none was configured.
func (c *Connection) latestReadWriteTypeUsed() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.latestReadWriteTypeUsedLocked()
}

func (c *Connection) latestReadWriteTypeUsedLocked() string {
	if c.readWriteType != "" {
		return c.readWriteType
	}
	return c.latestPDOTypeRetrieved
}

// GetTablePrefix returns the prefix prepended to every table name.
func (c *Connection) GetTablePrefix() string {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.tablePrefix
}

// SetTablePrefix replaces the prefix prepended to every table name.
func (c *Connection) SetTablePrefix(prefix string) *Connection {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.tablePrefix = prefix
	return c
}

// WithoutTablePrefix runs callback with the table prefix cleared, restoring
// it afterward.
func (c *Connection) WithoutTablePrefix(callback func(*Connection) error) error {
	prefix := c.GetTablePrefix()

	c.SetTablePrefix("")
	defer c.SetTablePrefix(prefix)

	return callback(c)
}

// GetServerVersion asks the server for its version string.
//
// database/sql has no driver-level way to read it, so it is read with a
// query: version() is spelled the same by PostgreSQL and MySQL, and SQLite's
// sqlite_version() is substituted for it.
func (c *Connection) GetServerVersion(ctx context.Context) (string, error) {
	q := "SELECT version()"
	if c.GetDriverName() == string(DialectSQLite) {
		q = "SELECT sqlite_version()"
	}
	value, err := c.Scalar(ctx, q, nil, true)
	if err != nil {
		return "", err
	}
	return asText(value), nil
}

// resolvers holds the per-driver constructors registered with ResolverFor.
var (
	resolversMu sync.RWMutex
	resolvers   = map[string]func(pdo *sql.DB, database, prefix string, config map[string]any) *Connection{}
)

// ResolverFor registers how a driver builds its connection.
//
// It is what lets a connector supply a driver-specific Connection without this
// package importing it, and what lets a project add a driver this framework
// does not ship.
func ResolverFor(driver string, callback func(pdo *sql.DB, database, prefix string, config map[string]any) *Connection) {
	resolversMu.Lock()
	defer resolversMu.Unlock()
	resolvers[driver] = callback
}

// GetResolver returns the constructor ResolverFor registered for driver, or
// nil when none was.
func GetResolver(driver string) func(pdo *sql.DB, database, prefix string, config map[string]any) *Connection {
	resolversMu.RLock()
	defer resolversMu.RUnlock()
	return resolvers[driver]
}

// The rest of concerns.TransactionDriver, which Connection satisfies so
// ManagesTransactions can drive it.

// ExecuteBeginTransactionStatement issues a BEGIN on a dedicated connection
// taken from the pool, and holds that connection for the life of the
// transaction.
//
// database/sql has no BEGIN outside sql.Tx, so this spells out what BeginTx
// does internally: ManagesTransactions' bookkeeping is what decides when the
// matching COMMIT runs.
func (c *Connection) ExecuteBeginTransactionStatement() error {
	pool, err := c.GetPDO()
	if err != nil {
		return err
	}

	conn, err := pool.Conn(context.Background())
	if err != nil {
		return err
	}
	if _, err := conn.ExecContext(context.Background(), "BEGIN"); err != nil {
		_ = conn.Close()
		return err
	}

	c.mu.Lock()
	c.txConn = conn
	c.mu.Unlock()
	return nil
}

// CommitTransactionStatement issues a COMMIT on the pinned transaction
// connection, and releases it back to the pool.
func (c *Connection) CommitTransactionStatement() error {
	c.mu.Lock()
	conn := c.txConn
	c.txConn = nil
	c.mu.Unlock()

	if conn == nil {
		return nil
	}
	_, err := conn.ExecContext(context.Background(), "COMMIT")
	_ = conn.Close()
	return err
}

// RollBackTransactionStatement issues a ROLLBACK on the pinned transaction
// connection and releases it, reporting false when there was no transaction
// connection to roll back.
func (c *Connection) RollBackTransactionStatement() (bool, error) {
	c.mu.Lock()
	conn := c.txConn
	c.txConn = nil
	c.mu.Unlock()

	if conn == nil {
		return false, nil
	}
	_, err := conn.ExecContext(context.Background(), "ROLLBACK")
	_ = conn.Close()
	return true, err
}

// ExecuteSavepointStatement runs a savepoint statement on the open transaction.
func (c *Connection) ExecuteSavepointStatement(statement string) error {
	c.mu.RLock()
	conn := c.txConn
	c.mu.RUnlock()

	if conn == nil {
		pool, err := c.GetPDO()
		if err != nil {
			return err
		}
		_, err = pool.ExecContext(context.Background(), statement)
		return err
	}
	_, err := conn.ExecContext(context.Background(), statement)
	return err
}

// SupportsSavepoints reports whether the query grammar supports savepoints,
// falling back to the base grammar when none is set.
func (c *Connection) SupportsSavepoints() bool {
	if grammar := c.GetQueryGrammar(); grammar != nil {
		return grammar.SupportsSavepoints()
	}
	return baseGrammar.SupportsSavepoints()
}

// CompileSavepoint returns the statement that creates savepoint name, using
// the query grammar, or the base grammar when none is set.
func (c *Connection) CompileSavepoint(name string) string {
	if grammar := c.GetQueryGrammar(); grammar != nil {
		return grammar.CompileSavepoint(name)
	}
	return baseGrammar.CompileSavepoint(name)
}

// CompileSavepointRollBack returns the statement that rolls back to
// savepoint name, using the query grammar, or the base grammar when none is
// set.
func (c *Connection) CompileSavepointRollBack(name string) string {
	if grammar := c.GetQueryGrammar(); grammar != nil {
		return grammar.CompileSavepointRollBack(name)
	}
	return baseGrammar.CompileSavepointRollBack(name)
}

// CausedByLostConnection reports whether err means the connection is gone,
// using the package-level detector.
func (c *Connection) CausedByLostConnection(err error) bool { return CausedByLostConnection(err) }

// CausedByConcurrencyError reports whether err means a deadlock or a
// serialization failure, using the package-level detector.
func (c *Connection) CausedByConcurrencyError(err error) bool { return CausedByConcurrencyError(err) }

// SubstituteBindingsIntoRawSQL writes bindings directly into sql, so a
// Connection satisfies events.RawSQLConnection.
func (c *Connection) SubstituteBindingsIntoRawSQL(sql string, bindings []any) string {
	return substituteBindings(sql, bindings)
}

// baseGrammar is the fallback for the three savepoint questions, for a
// connection with no query grammar of its own set.
var baseGrammar = &query.BaseGrammar{}

// scanRecords reads a whole result set into records.
func scanRecords(rows *sql.Rows) ([]query.Record, error) {
	columns, err := rows.Columns()
	if err != nil {
		return nil, err
	}

	var out []query.Record
	for rows.Next() {
		record, err := scanRecord(rows, columns)
		if err != nil {
			return nil, err
		}
		out = append(out, record)
	}
	return out, rows.Err()
}

// scanRecord reads one row into a record.
//
// Every column is scanned into an any: the driver decides the Go type, and
// the caller reads it. A []byte is turned into a string, because MySQL's text
// protocol returns every column that way and a caller comparing against a
// string would otherwise never match.
func scanRecord(rows *sql.Rows, columns []string) (query.Record, error) {
	values := make([]any, len(columns))
	pointers := make([]any, len(columns))
	for i := range values {
		pointers[i] = &values[i]
	}

	if err := rows.Scan(pointers...); err != nil {
		return nil, err
	}

	record := make(query.Record, len(columns))
	for i, column := range columns {
		if b, ok := values[i].([]byte); ok {
			record[column] = string(b)
			continue
		}
		record[column] = values[i]
	}
	return record, nil
}

func asText(v any) string {
	switch s := v.(type) {
	case nil:
		return ""
	case string:
		return s
	case []byte:
		return string(s)
	default:
		return fmt.Sprint(v)
	}
}

func toInt64(v any) int64 {
	switch n := v.(type) {
	case int:
		return int64(n)
	case int32:
		return int64(n)
	case int64:
		return n
	case float64:
		return int64(n)
	case string:
		return int64(parseUint(n))
	case []byte:
		return int64(parseUint(string(n)))
	default:
		return 0
	}
}

func parseUint(s string) int {
	n := 0
	for _, c := range s {
		if c < '0' || c > '9' {
			return n
		}
		n = n*10 + int(c-'0')
	}
	return n
}

// sortedNames is used by the manager to list connections in a stable order.
func sortedNames(names []string) []string {
	sort.Strings(names)
	return names
}

// SchemaBuilderFactory is where the schema package registers how to build a
// schema builder for a connection.
//
// Constructing one directly here would make this package import
// database/schema, while that package imports this one for the connection it
// builds against. The registration goes the other way and closes the cycle.
var SchemaBuilderFactory func(*Connection) any

// GetSchemaBuilder returns the schema builder for this connection, built
// through SchemaBuilderFactory.
//
// It returns any rather than a schema builder type for the reason above. Nil
// means the binary never imported the schema package, which is the truthful
// response to "give me the schema builder I did not link".
func (c *Connection) GetSchemaBuilder() any {
	if c.GetSchemaGrammar() == nil {
		c.UseDefaultSchemaGrammar()
	}
	if SchemaBuilderFactory == nil {
		return nil
	}
	return SchemaBuilderFactory(c)
}
