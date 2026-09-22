package database

// KeyText is how a text column that takes part in a key is declared.
//
// TEXT is the portable spelling for text, and it is the wrong one for anything
// indexed: MySQL stores TEXT off-page and refuses it in a key without a prefix
// length, so `id TEXT PRIMARY KEY` fails with "BLOB/TEXT column used in key
// specification without a key length". That is the first statement `aru
// migrate` runs, which means MySQL never got past creating the tracking table
// -- and every project table repeated the same mistake. Found by audit.
//
// VARCHAR(255) is accepted by all three: PostgreSQL treats it as varchar,
// SQLite gives it TEXT affinity because the name contains CHAR, and MySQL
// indexes it. 255 rather than something tighter so there is one width to
// remember, and because two of them in a composite index still fit under
// InnoDB's key limit.
//
// The rule: TEXT for free-form content nobody indexes, KeyText for an id, a
// tenant, or anything a UNIQUE or an index names.
const KeyText = "VARCHAR(255)"
