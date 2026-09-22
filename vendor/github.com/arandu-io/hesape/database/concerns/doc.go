// Package concerns holds the shared halves of a connection and a query builder:
// ManagesTransactions, BuildsQueries, BuildsWhereDateClauses, CompilesJsonPaths,
// ExplainsQueries and ParsesSearchPath.
//
// Each takes one of two shapes, chosen by what it does:
//
//   - one that carries state becomes a struct the user embeds, with the calls it
//     makes back into its user written out as an interface -- ManagesTransactions,
//     which Connection embeds;
//   - one that carries none becomes functions taking what the receiver was.
//
// An initialism is upper case: ToSQL, ChunkByID, WrapJSONPath.
//
// # Every read here carries a Grant
//
// Chunk, Each, First, FirstOrFail, Sole, Lazy and Explain all execute, so they
// all take an auth.Grant and hand it to the query on every page they fetch.
// Authorization is not a rule about writes: a chunked export that skipped the
// Policy is a leak between tenants with a pleasant name, and a chunk loop is
// exactly where somebody would be tempted to authorize once and then not.
//
// # Two of these are declared here for a reason worth reading
//
// DeadlockError and ErrRecordNotFound are declared here rather than in the
// database package, which imports this one -- Connection embeds
// ManagesTransactions -- so declaring them there would close an import cycle.
// The database package re-exports both as aliases, so database.DeadlockException
// and concerns.DeadlockError are one type, and errors.Is works across the two
// names because there is only one.
//
// Building a page is not here: hesape/pagination builds a page from rows
// somebody already fetched, and that is the one way.
package concerns
