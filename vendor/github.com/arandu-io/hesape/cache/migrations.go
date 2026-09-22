package cache

// Table is the table [DatabaseStore] keeps entries in.
const Table = "cache"

// LockTable is the table it keeps locks in.
//
// Two tables and not one because flushing the locks must not flush the cache,
// and a store whose locks live in the cache table refuses to do it at all --
// see [DatabaseStore.HasSeparateLockStore].
const LockTable = "cache_locks"

// The migration that creates the two is in cache/console, and not here, and
// that is a constraint rather than a preference: database/migrations tests its
// own isolation against a real cache lock, so a cache that imported
// database/migrations would close a cycle. The names live here because the
// store reads them; the migration lives beside the generator that writes it out.
