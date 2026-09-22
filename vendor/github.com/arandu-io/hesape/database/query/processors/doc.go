// Package processors holds the hook a driver takes to adjust results on the way
// out of the connection: MySQLProcessor, MariaDBProcessor, PostgresProcessor and
// SQLiteProcessor over the shared Processor.
//
// A processor exists so that three engines answering the same question three
// ways arrive at the caller as one answer: Postgres returns an inserted
// identifier as a row, MySQL reports it out of band, and both come back as an
// int64. Everything else it does is reshaping what a schema introspection query
// reported.
//
// # Where authorization is, and where it is not
//
// Not here. ProcessInsertGetID does run a statement, through the connection the
// builder is holding, and the Grant that allowed it was checked one layer up: a
// *query.Builder is reachable only from a repository that holds an auth.Grant
// and has already filtered by auth.Tenant(g), on reads exactly as on writes. A
// processor that took a Grant would be a second place to enforce authorization,
// and a second place for it to be forgotten.
//
// # Signatures
//
// An initialism is upper case: ProcessInsertGetID, MySQLProcessor,
// MariaDBProcessor.
//
//   - ProcessInsertGetID returns (int64, error); query.Processor declares the
//     signature.
//   - The connection is asked for the identifier through LastInsertIDConnection,
//     because query.connection is narrowed to running statements and holds no
//     driver handle to reach through.
//   - ProcessColumns takes the CREATE TABLE statement as a variadic argument,
//     because Go has no default argument and SQLite's is the one that reads it.
//   - A result row is a query.Record -- a map. The values are coerced on the way
//     out, since a driver hands back a count as []byte or int64 depending on
//     which one it is.
//
// There is no SQL Server processor: it is a driver this ecosystem does not
// carry.
package processors
