// Package grammars holds one grammar per engine, each one compiling a
// *query.Builder into the SQL that engine speaks: MySQLGrammar, MariaDBGrammar,
// PostgresGrammar and SQLiteGrammar, over the shared Grammar in grammar.go. The
// JSON path handling they all use is in jsonpath.go.
//
// # Where authorization is, and where it is not
//
// Not here. A grammar turns a builder into a string; it decides how a question
// is spelled, never who may ask it. Authorization lives one layer up, in the
// repository that holds an auth.Grant and filters by auth.Tenant(g) -- on reads
// exactly as on writes, because a read path without a policy is a tenant
// reading another tenant's rows. Nothing in this package takes a Grant, and
// nothing in this package should be reachable except through something that
// does.
//
// One consequence shows up in the code: a where clause the grammar cannot spell
// is never dropped. It compiles to a false clause carrying the reason, so the
// query returns nothing instead of returning rows nobody filtered for. See
// unsupportedClause.
//
// # Placeholders are "?" in all four
//
// Postgres numbers its placeholders and this package still emits "?" for it.
// The numbering happens once, at the connection, in database.Dialect.Rebind,
// which rewrites each "?" to $1, $2 and so on over the finished statement while
// leaving the ones inside string literals and comments alone. Doing it here
// would mean a grammar had to know a fragment's position in a statement it has
// not finished building -- a subquery compiled on its own would start over at
// $1 -- and it would give the project two placeholder conventions where one
// does. See Grammar.Parameter.
//
// # Dialect differences go through a self reference
//
// Go resolves an embedded method at compile time, so a base method calling
// another base method would always run the base version and every dialect
// difference would vanish without a single failure. Grammar therefore holds a
// self reference, typed as the unexported dialect interface, and the pipeline
// calls through it. A driver grammar embeds *Grammar and points self at itself.
//
// The visibility mapping follows from that: a method a driver grammar overrides
// is exported, because it is the extension point. A helper nothing overrides
// stays unexported.
//
// # The base grammar is incomplete, and the compiler enforces it
//
// Grammar deliberately does not implement CompileInsertOrIgnore or
// CompileUpsert, because no engine spells them the standard way. Without them
// *Grammar does not satisfy query.Grammar and cannot be handed to a builder at
// all, so the gap is a compile error rather than a failure when the statement
// runs.
//
// # Signatures
//
// An initialism is upper case: CompileInsertGetID, CompileJSONLength, ToSQL.
// The rest of the shape is:
//
//   - (T, error) wherever the signature is free to say so: CompileJoinLateral,
//     CompileJSONContains, CompileJSONLength, WhereFullText,
//     SupportsStraightJoins, CompileInsertOrIgnoreUsing,
//     CompileInsertOrIgnoreReturning, SubstituteBindingsIntoRawSQL.
//   - A false clause where the failure happens inside the compile path, whose
//     signature query.Grammar fixes as returning a string.
//   - The empty string for an absent fragment, since concatenate drops it.
//   - A values map is walked in sorted key order, because a Go map has none and
//     the statement has to be the same on every run for the bindings to line up.
//     See Grammar.CompileInsert.
//   - CompileUpsert takes the update list as column names only, which is the
//     shape query.Grammar declares.
//
// There is no SQL Server grammar: the three connectors are pgx, mysql and
// sqlite.
package grammars
