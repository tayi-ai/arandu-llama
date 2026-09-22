// Package database is the data access contract: one connection, one repository
// shape, and no ORM.
//
// Queries are plain parameterized SQL, written by hand in the templates
// `aru make:module` emits, which keeps the query plan predictable and the value
// always in a placeholder. What this package adds on top is:
//
//  1. an auth.Grant required by every operation (the mandatory path);
//  2. tenant scoping taken from the Grant, never from a parameter;
//  3. automatic instrumentation into the Collector;
//  4. one Open, one pool policy, and each driver in its own module.
//
// # The tenant does not live here
//
// auth.Tenant(g) is the single source of a tenant for SQL. It sits in the
// package that owns the Grant rather than in this one, so that the cache, the
// filesystem and the scheduler can scope a key by customer without importing
// the database. The tenant comes from the Grant, never from a path, a body, a
// query or a header.
//
// # Why the drivers are separate modules
//
// In Go there is no optional dependency. A single module carrying pgx, MySQL and
// SQLite would put all three in the go.sum of every project -- in the build, in
// the binary, and in the vulnerability surface. That is not hypothetical: the
// skeleton carried pgx and modernc/sqlite together, and govulncheck found a pgx
// advisory in a project that could have been SQLite-only.
//
// So each connector is its own module with its own go.mod:
//
//	go get github.com/arandu-io/hesape/database/connectors/sqlite   // needs nothing installed
//	go get github.com/arandu-io/hesape/database/connectors/pgx      // Postgres
//	go get github.com/arandu-io/hesape/database/connectors/mysql    // MySQL
//
// and the project blank-imports the ones it uses:
//
//	import (
//	    "github.com/arandu-io/hesape/database"
//	    _ "github.com/arandu-io/hesape/database/connectors/pgx"
//	    _ "github.com/arandu-io/hesape/database/connectors/sqlite"
//	)
//
//	db, closeDB, err := database.Open(cfg)
//
// Switching engines is a line in .env. The import list is what decides which
// engines a build can speak at all.
//
// ConnectionFactory is the piece that reads that registry. It lives here rather
// than under the connectors, and that is what makes the inversion work: a factory
// that imported the three connector modules to choose between them would put
// all three back in every go.sum. It never names a driver package -- it looks
// the dialect up in the registry the connector's init filled in. A project that
// speaks an engine this framework does not ship registers its own the same way;
// the ConnectionFactory doc walks through the four lines.
//
// The conformance subpackage is what makes "one Repository, three engines" a
// measurement rather than a claim: one suite, run by all three connectors
// against a real server.
//
// # Where the Grant is
//
// Repository is the door: every method on it takes an auth.Grant and filters by
// auth.Tenant(g), on a read exactly as on a write. Connection, DB and the
// migrator are the plumbing under that door and take none -- not as an
// exemption, but because a Grant they could not use to filter anything would be
// a parameter that looks like enforcement and is not. `aru migrate` also runs
// down here, in a process with no request and no subject, where a Grant cannot
// be constructed at all.
//
// So a module reaches rows through a Repository. A module that reaches them
// through a Connection is a module that gets sent back in review.
//
// # There is one way to connect
//
// A driver registers itself with database/sql and everything it takes arrives in
// the DSN. [Connector] is deliberately the smaller thing -- it says which driver
// it linked and never opens a connection -- and Open resolves the rest from
// DATABASE_URL, so there is one way in. There is no fetch mode to set
// beforehand either: database/sql scans into whatever the caller passed to Scan,
// so the mode is the destination.
//
// An automatic identifier is read from the statement that caused it, through
// sql.Result.LastInsertId, by [Connection.InsertReturningID]. It is one round
// trip rather than a SELECT afterwards, and it cannot answer about somebody
// else's row -- on a pool, "the last identifier" is whoever inserted most
// recently, which is the classic way one request is handed another request's id.
// The processor asks for it through processors.LastInsertIDConnection, and what
// implements that is the binding Query builds per builder, not the pool.
//
// # Dumping a query does not end the process
//
// [github.com/arandu-io/hesape/database/query.Builder.Dump] and
// [github.com/arandu-io/hesape/database/query.Builder.DumpRawSQL] write the
// query and hand the builder back. There is no variant that exits afterwards: a
// library that kills the process is a library nobody can wrap, and a dump that
// exits is the same function with a way it cannot be tested.
//
// # Why there is no schema-state getter
//
// The dump-and-load helpers exist --
// [github.com/arandu-io/hesape/database/schema.NewMySqlSchemaState] and its two
// siblings -- but they live in the schema package, which declares its own narrow
// Connection interface and is imported BY this package's callers rather than by
// this package. A getter here would have to import schema, and schema would go
// on needing a connection: Go refuses the cycle, so the constructor is the call
// site instead. It takes the connection and the process factory.
//
// A seeder is handed what it needs rather than reaching for it; see [Seeder].
package database
