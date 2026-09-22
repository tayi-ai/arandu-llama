// Package events holds the values the database dispatches: connection
// established, query executed, statement prepared, the four transaction events,
// the migration events, the schema dump and load events, and model pruning.
//
// Every event is a plain value with a constructor that reads the connection name
// off the connection rather than taking it, which is why ConnectionName is never
// a parameter.
//
// # Why the connection is an interface here
//
// These events hold a small interface declared in this package, because
// Connection lives in the database package and that package dispatches these
// events: naming the concrete type would close an import cycle. It is the same
// move query.Connection makes one package over, and it costs the events nothing
// -- they call one method on it.
package events
