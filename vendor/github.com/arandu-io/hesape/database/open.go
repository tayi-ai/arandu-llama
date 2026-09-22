package database

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"sync"
	"time"
)

// registry maps a dialect to the database/sql driver name a compartment
// registered for it.
var (
	registryMu sync.RWMutex
	registry   = map[Dialect]string{}
)

// sqlOpen is a variable so a test can substitute it. Opening a real connection
// to prove that the registry resolves the right driver name would need three
// servers running, and the thing under test is the resolution, not the network.
var sqlOpen = sql.Open

// Register records that a connector for this dialect is linked into the binary.
//
// Connectors call it from init(). It is not meant for application code: a
// project that registers its own driver name is a project that will get a
// different pool policy than every other, for no gain.
//
// Registering the same dialect twice panics rather than picking one. Two drivers
// for one dialect is an import nobody meant to add, and finding out at boot is
// better than finding out from a query that behaves differently.
func Register(c Connector) {
	registryMu.Lock()
	defer registryMu.Unlock()

	d, driverName := c.Dialect(), c.DriverName()

	if existing, taken := registry[d]; taken && existing != driverName {
		panic(fmt.Sprintf("database: %s is already registered to the %q driver, and %q wants it too -- remove one of the imports",
			d, existing, driverName))
	}
	registry[d] = driverName
}

// Registered reports the dialects this binary can speak, sorted.
//
// `aru doctor` reads it, and so does the error below.
func Registered() []Dialect {
	registryMu.RLock()
	defer registryMu.RUnlock()
	return sortedLocked()
}

// sortedLocked returns the registered dialects. The caller holds the lock,
// which is what keeps this callable from inside another locked section.
func sortedLocked() []Dialect {
	out := make([]Dialect, 0, len(registry))
	for d := range registry {
		out = append(out, d)
	}
	sort.Slice(out, func(i, j int) bool { return out[i] < out[j] })
	return out
}

// The pool this package keeps when the configuration asks for nothing.
//
// They are the answer to "what should a web application hold", not a tuning
// guide: 25 in flight is more than a request-per-core process ever needs at
// once, 5 idle covers the gap between two bursts without holding a server's
// connection table, and an hour is short enough that a rotated credential or a
// proxy that moved is noticed by the pool rather than by a request.
const (
	defaultMaxOpenConns    = 25
	defaultMaxIdleConns    = 5
	defaultConnMaxLifetime = time.Hour
)

// pool answers the three settings [Open] applies, reading a zero on any of them
// as the default of this package.
//
// Zero is never database/sql's meaning for zero. SetMaxOpenConns(0) there is an
// unbounded pool, and every Config that reaches this comes out of [ParseURL]
// with these three at zero -- so taking that meaning would remove the bound from
// every pool that exists rather than from the ones that asked. The bound is what
// [Open] is for.
func (d Config) pool() (maxOpen, maxIdle int, lifetime time.Duration) {
	maxOpen, maxIdle, lifetime = defaultMaxOpenConns, defaultMaxIdleConns, defaultConnMaxLifetime
	if d.MaxOpenConns > 0 {
		maxOpen = d.MaxOpenConns
	}
	if d.MaxIdleConns > 0 {
		maxIdle = d.MaxIdleConns
	}
	if d.ConnMaxLifetime > 0 {
		lifetime = d.ConnMaxLifetime
	}
	return maxOpen, maxIdle, lifetime
}

// Open connects, tunes the pool, and returns the instrumented handle plus the
// function that closes it.
//
// The pool policy lives here rather than in each project's main, because it is
// not a preference: the defaults of database/sql are an unbounded pool, which
// turns one traffic spike into "too many connections" on the server instead of a
// queue in the process. What a project may choose is the size, through the three
// pool fields of [Config]; what it may not choose is having no size at all.
func Open(cfg Config) (*DB, func(), error) {
	driverName, err := driverFor(cfg.Connection)
	if err != nil {
		return nil, nil, err
	}

	// SQLite creates the database file but never the directory above it, and the
	// error it gives for a missing directory names neither.
	if path := cfg.SQLitePath(); path != "" {
		if err := os.MkdirAll(filepath.Dir(path), 0o755); err != nil {
			return nil, nil, fmt.Errorf("creating the database directory: %w", err)
		}
	}

	sqldb, err := sqlOpen(driverName, cfg.DSN())
	if err != nil {
		return nil, nil, fmt.Errorf("opening %s: %w", cfg.Redacted(), err)
	}

	maxOpen, maxIdle, lifetime := cfg.pool()
	if cfg.Connection == DialectSQLite {
		// One writer, whatever the configuration asked for. SQLite serializes
		// writes anyway, so a larger pool does not open a second writer -- it
		// only converts the wait into "database is locked", which reads like
		// corruption and is really a pool setting. A number that cannot do what
		// it says is worse than no setting, so this one overrides rather than
		// being refused at the top: the configuration is portable across
		// engines, and the same value is right on the server it was written for.
		maxOpen = 1
	}
	sqldb.SetMaxOpenConns(maxOpen)
	sqldb.SetMaxIdleConns(maxIdle)
	sqldb.SetConnMaxLifetime(lifetime)

	// And it connects, here, before anything else runs.
	//
	// sql.Open resolves the driver and builds a pool; it does not speak to the
	// server. So a wrong password, a database that was never created, a role
	// that cannot log in, a port nothing listens on, a host that does not
	// resolve -- every one of them used to be deferred to the first query, and
	// the first query is inside a request. The application booted, said it was
	// listening, and then answered the first visitor with a driver sentence on
	// an error page.
	//
	// A process that cannot reach its database is not up, and saying so at boot
	// is what turns "the site is broken" into a line in the log of the deploy
	// that broke it.
	ctx, cancel := context.WithTimeout(context.Background(), connectTimeout)
	defer cancel()

	if err := sqldb.PingContext(ctx); err != nil {
		_ = sqldb.Close()
		return nil, nil, fmt.Errorf("connecting to %s: %w", cfg.Redacted(), err)
	}

	return Wrap(sqldb, cfg.Connection), func() { _ = sqldb.Close() }, nil
}

// connectTimeout bounds the check above.
//
// Long enough for a container that is still starting, short enough that a
// deploy pointed at an unreachable host fails while somebody is still watching
// it. Without a bound, a host that drops packets instead of refusing them hangs
// the boot until the operating system gives up, which is minutes.
const connectTimeout = 5 * time.Second

// driverFor resolves the dialect, or explains exactly what is missing.
//
// This error is the shape the whole product aims at: it names the probable
// cause and the command that fixes it. "unknown driver pgx" -- what
// database/sql says on its own -- sends people looking for a typo in .env.
func driverFor(d Dialect) (string, error) {
	// The read lock is taken once, and everything under it stays lock-free.
	//
	// It used to call Registered() from in here, which takes the read lock
	// again. sync.RWMutex is not reentrant: with a writer waiting between the
	// two acquisitions, Go blocks the second reader to keep the writer from
	// starving, and the process deadlocks permanently. Found by audit.
	registryMu.RLock()
	name, found := registry[d]
	dialects := sortedLocked()
	registryMu.RUnlock()

	if found {
		return name, nil
	}

	linked := "none"
	if len(dialects) > 0 {
		linked = ""
		for i, a := range dialects {
			if i > 0 {
				linked += ", "
			}
			linked += string(a)
		}
	}
	return "", fmt.Errorf(
		"DATABASE_URL asks for %s and no connector for it is linked into this binary (linked: %s).\n"+
			"Add it:\n\n    go get github.com/arandu-io/hesape/database/connectors/%s\n\n"+
			"and blank-import it in main.go, next to the other connectors:\n\n    _ \"github.com/arandu-io/hesape/database/connectors/%s\"",
		d, linked, connectorFor(d), connectorFor(d))
}

// connectorFor is the module that carries the connector for a dialect.
func connectorFor(d Dialect) string {
	switch d {
	case DialectPostgres:
		return "pgx"
	case DialectMySQL:
		return "mysql"
	default:
		return "sqlite"
	}
}

// DriverName returns the database/sql driver a compartment registered for a
// dialect, or the error that names the missing import.
//
// It exists for the conformance suite, which has to open a connection with the
// same driver Open would use rather than a name it hardcoded -- a suite testing
// a driver nobody links is a suite testing nothing.
func DriverName(d Dialect) (string, error) { return driverFor(d) }
