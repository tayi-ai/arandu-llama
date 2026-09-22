package cache

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"strconv"
	"time"
)

// defaultLockTimeout is how long a database lock is held when nobody says.
//
// A day is not a sensible time to hold a lock; it is a sensible time for one to
// survive the process that died holding it, which is the only job it has.
const defaultLockTimeout = 24 * time.Hour

// defaultLockLottery is how often acquiring a lock also prunes the expired ones.
//
// It answers the [2, 100] of DatabaseStore: two acquisitions in a hundred pay
// for the cleanup, so the table does not grow without bound and no single caller
// pays for it often.
var defaultLockLottery = [2]int{2, 100}

// Connection is the slice of a database connection DatabaseStore needs.
//
// It is the shape of *sql.DB and *sql.Tx, and it is declared here rather than
// imported because hesape/database is being written in parallel -- a store that
// could not compile until it landed would block on it. When it lands, its
// connection satisfies this.
//
// The placeholders are the ? of MySQL and SQLite; a connection to Postgres is
// expected to rebind them, which is the arrangement the outbox in hesape/events
// already runs on.
type Connection interface {
	ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error)
	QueryContext(ctx context.Context, query string, args ...any) (*sql.Rows, error)
	QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
}

// NamedConnection is the optional half of Connection: one that knows what it is
// called.
//
// It is a second interface rather than a method on Connection because *sql.DB
// has no name and would stop satisfying it.
type NamedConnection interface {
	Connection

	// GetName returns the connection's name in the configuration.
	GetName() string
}

// DatabaseStore is the cache in a table.
//
// It is the store for an application that already has a database and does not
// want a second piece of infrastructure.
//
// It is the slowest store here and it is the only one that is shared, durable
// and transactional at the same time. Reach for it when the alternative is no
// cache at all, or when the locks are what is wanted: a database lock is the
// only one in this package that survives the cache being emptied, because it
// lives in its own table.
//
// # The tables
//
// The cache table has key (primary), value and expiration. The lock table has
// key (primary), owner and expiration. Both are what `aru make:cache-table`
// writes.
//
// The expiration is unix milliseconds, and that is why the column is a BIGINT:
// Store.Put is given a duration, and a store whose resolution is a second
// cannot honour a ttl shorter than one -- it would either expire the entry on
// arrival or keep it for most of a second longer than it was told.
type DatabaseStore struct {
	connection     Connection
	lockConnection Connection
	table          string
	lockTable      string
	prefix         string
	lockLottery    [2]int
	lockTimeout    time.Duration
}

var (
	_ Store         = (*DatabaseStore)(nil)
	_ Locking       = (*DatabaseStore)(nil)
	_ CanFlushLocks = (*DatabaseStore)(nil)
	_ CurrentOwner  = (*DatabaseStore)(nil)
	_ Taggable      = (*DatabaseStore)(nil)
)

// NewDatabaseStore returns a store over a table.
//
// An empty lockTable means "cache_locks". A lock table separate from the cache
// table is what keeps FlushLocks possible -- see HasSeparateLockStore.
func NewDatabaseStore(connection Connection, table, prefix, lockTable string) *DatabaseStore {
	if lockTable == "" {
		lockTable = LockTable
	}
	return &DatabaseStore{
		connection:  connection,
		table:       table,
		lockTable:   lockTable,
		prefix:      prefix,
		lockLottery: defaultLockLottery,
		lockTimeout: defaultLockTimeout,
	}
}

// GetConnection returns the connection the entries are read and written on.
func (s *DatabaseStore) GetConnection() Connection { return s.connection }

// SetConnection sets that connection and returns the store.
func (s *DatabaseStore) SetConnection(c Connection) *DatabaseStore {
	s.connection = c
	return s
}

// GetLockConnection returns the connection the locks are managed on, and nil
// when there is none.
func (s *DatabaseStore) GetLockConnection() Connection { return s.lockConnection }

// SetLockConnection sets that connection and returns the store.
//
// It answers DatabaseStore::setLockConnection(). It is worth setting to a
// connection of its own: a lock taken on the same connection as the work it
// guards is a lock inside the transaction the work is in, and a rollback takes
// it with it.
func (s *DatabaseStore) SetLockConnection(c Connection) *DatabaseStore {
	s.lockConnection = c
	return s
}

// GetConnectionName is the name of the connection the locks are managed on.
//
// It answers DatabaseLock::getConnectionName(). A connection that does not know
// its own name -- a bare *sql.DB is one -- answers the empty string rather than
// making one up.
func (s *DatabaseStore) GetConnectionName() string {
	if named, ok := s.lockConn().(NamedConnection); ok {
		return named.GetName()
	}
	return ""
}

// GetPrefix is what goes in front of every key this store writes.
func (s *DatabaseStore) GetPrefix() string { return s.prefix }

// SetPrefix sets it. It answers DatabaseStore::setPrefix().
func (s *DatabaseStore) SetPrefix(prefix string) { s.prefix = prefix }

// HasSeparateLockStore reports whether the locks live in another table.
func (s *DatabaseStore) HasSeparateLockStore() bool { return s.lockTable != s.table }

// lockConn is the connection the locks are managed on: the one that was set for
// them, or the ordinary one.
func (s *DatabaseStore) lockConn() Connection {
	if s.lockConnection != nil {
		return s.lockConnection
	}
	return s.connection
}

// Get returns the stored bytes, or ErrNotFound.
//
// An expired row is deleted on the way out and reported as a miss, which is what
// DatabaseStore::many() does with the expired half of its result.
func (s *DatabaseStore) Get(ctx context.Context, key string) ([]byte, error) {
	var value []byte
	var expiration int64

	err := s.connection.QueryRowContext(ctx,
		"SELECT value, expiration FROM "+s.table+" WHERE key = ?", s.prefix+key,
	).Scan(&value, &expiration)
	switch {
	case errors.Is(err, sql.ErrNoRows):
		return nil, ErrNotFound
	case err != nil:
		return nil, fmt.Errorf("cache: reading %q from %s: %w", key, s.table, err)
	}

	if expiration <= nowMillis() {
		_ = s.ForgetIfExpired(ctx, key)
		return nil, ErrNotFound
	}
	return value, nil
}

// Many returns the stored bytes for several keys at once.
//
// It answers DatabaseStore::many(): every key asked for is in the result, and
// the misses carry nil. It is one statement, which is the reason to call it
// rather than Get in a loop.
func (s *DatabaseStore) Many(ctx context.Context, keys []string) (map[string][]byte, error) {
	out := make(map[string][]byte, len(keys))
	if len(keys) == 0 {
		return out, nil
	}
	for _, key := range keys {
		out[key] = nil
	}

	placeholders := ""
	args := make([]any, 0, len(keys))
	for i, key := range keys {
		if i > 0 {
			placeholders += ", "
		}
		placeholders += "?"
		args = append(args, s.prefix+key)
	}

	rows, err := s.connection.QueryContext(ctx,
		"SELECT key, value, expiration FROM "+s.table+" WHERE key IN ("+placeholders+")", args...)
	if err != nil {
		return nil, fmt.Errorf("cache: reading %d keys from %s: %w", len(keys), s.table, err)
	}
	defer func() { _ = rows.Close() }()

	now := nowMillis()
	var expired []string
	for rows.Next() {
		var stored string
		var value []byte
		var expiration int64
		if err := rows.Scan(&stored, &value, &expiration); err != nil {
			return nil, err
		}
		key := stored[len(s.prefix):]
		if expiration <= now {
			expired = append(expired, key)
			continue
		}
		if _, asked := out[key]; asked {
			out[key] = value
		}
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	for _, key := range expired {
		_ = s.ForgetIfExpired(ctx, key)
	}
	return out, nil
}

// Put stores value for ttl, replacing whatever was there.
//
// It is an update, and an insert when the update matched nothing. An upsert is
// a different statement on every dialect; this is one pair that every database
// this package supports understands, and the outcome -- last writer wins -- is
// the same.
func (s *DatabaseStore) Put(ctx context.Context, key string, value []byte, ttl time.Duration) error {
	if ttl <= 0 {
		return ErrNoTTL
	}
	expiration := expiresAt(ttl)

	result, err := s.connection.ExecContext(ctx,
		"UPDATE "+s.table+" SET value = ?, expiration = ? WHERE key = ?",
		value, expiration, s.prefix+key)
	if err != nil {
		return fmt.Errorf("cache: writing %q to %s: %w", key, s.table, err)
	}
	if n, err := result.RowsAffected(); err == nil && n > 0 {
		return nil
	}

	if _, err := s.connection.ExecContext(ctx,
		"INSERT INTO "+s.table+" (key, value, expiration) VALUES (?, ?, ?)",
		s.prefix+key, value, expiration); err != nil {
		// Somebody inserted the row between the update and the insert. The
		// update is the answer either way, and running it again is what makes
		// this pair equivalent to the upsert.
		if _, retry := s.connection.ExecContext(ctx,
			"UPDATE "+s.table+" SET value = ?, expiration = ? WHERE key = ?",
			value, expiration, s.prefix+key); retry != nil {
			return fmt.Errorf("cache: writing %q to %s: %w", key, s.table, err)
		}
	}
	return nil
}

// PutMany stores several values under one ttl.
//
// It answers DatabaseStore::putMany(). It is a loop over Put rather than one
// multi-row upsert, for the reason Put is a pair: the upsert is a different
// statement on every grammar.
func (s *DatabaseStore) PutMany(ctx context.Context, values map[string][]byte, ttl time.Duration) error {
	for key, value := range values {
		if err := s.Put(ctx, key, value, ttl); err != nil {
			return err
		}
	}
	return nil
}

// Add stores value only if the key is absent, and reports whether it did.
//
// It answers DatabaseStore::add(): read first, which clears an expired row on
// the way past, then insert -- and a primary key violation means somebody else
// got there in between, which is a false and not an error.
func (s *DatabaseStore) Add(ctx context.Context, key string, value []byte, ttl time.Duration) (bool, error) {
	if ttl <= 0 {
		return false, ErrNoTTL
	}
	switch _, err := s.Get(ctx, key); {
	case err == nil:
		return false, nil
	case !errors.Is(err, ErrNotFound):
		return false, err
	}

	if _, err := s.connection.ExecContext(ctx,
		"INSERT INTO "+s.table+" (key, value, expiration) VALUES (?, ?, ?)",
		s.prefix+key, value, expiresAt(ttl)); err != nil {
		return false, nil
	}
	return true, nil
}

// Forever stores a value with no expiry the caller has to think about.
//
// It answers DatabaseStore::forever(), which writes ten years. Here it is a
// century, which is what everything else in this package writes: see
// Repository.Forever.
func (s *DatabaseStore) Forever(ctx context.Context, key string, value []byte) error {
	return s.Put(ctx, key, value, foreverTTL)
}

// Increment adds delta to the counter under key and returns the new value.
//
// The counter keeps the deadline it was created with, which is what makes a
// fixed window fixed.
//
// It is not atomic across connections: an exact count would need a row lock,
// which this interface cannot ask for without a transaction, and a counter that
// has to be exact belongs in a store that counts atomically. What it does
// instead is refuse to widen the race: the
// update carries the value it read in its WHERE, so a lost update is a lost
// count and never a wrong one.
func (s *DatabaseStore) Increment(ctx context.Context, key string, delta int64, ttl time.Duration) (int64, error) {
	if ttl <= 0 {
		return 0, ErrNoTTL
	}

	switch current, err := s.Get(ctx, key); {
	case errors.Is(err, ErrNotFound):
		if added, err := s.Add(ctx, key, []byte(strconv.FormatInt(delta, 10)), ttl); err != nil {
			return 0, err
		} else if added {
			return delta, nil
		}
		// Somebody created it in between. Read it again and add to that.
		return s.increment(ctx, key, delta)
	case err != nil:
		return 0, err
	default:
		parsed, perr := strconv.ParseInt(string(current), 10, 64)
		if perr != nil {
			return 0, errNotACounter(key)
		}
		next := parsed + delta
		result, err := s.connection.ExecContext(ctx,
			"UPDATE "+s.table+" SET value = ? WHERE key = ? AND value = ?",
			strconv.FormatInt(next, 10), s.prefix+key, current)
		if err != nil {
			return 0, fmt.Errorf("cache: incrementing %q in %s: %w", key, s.table, err)
		}
		if n, err := result.RowsAffected(); err == nil && n == 0 {
			return 0, fmt.Errorf("cache: %q was changed by somebody else while it was being incremented", key)
		}
		return next, nil
	}
}

// increment is the second attempt of Increment, when the counter appeared
// between the read and the insert.
func (s *DatabaseStore) increment(ctx context.Context, key string, delta int64) (int64, error) {
	current, err := s.Get(ctx, key)
	if err != nil {
		return 0, err
	}
	parsed, perr := strconv.ParseInt(string(current), 10, 64)
	if perr != nil {
		return 0, errNotACounter(key)
	}
	next := parsed + delta
	if _, err := s.connection.ExecContext(ctx,
		"UPDATE "+s.table+" SET value = ? WHERE key = ? AND value = ?",
		strconv.FormatInt(next, 10), s.prefix+key, current); err != nil {
		return 0, err
	}
	return next, nil
}

// Decrement subtracts delta from the counter under key.
func (s *DatabaseStore) Decrement(ctx context.Context, key string, delta int64, ttl time.Duration) (int64, error) {
	return s.Increment(ctx, key, -delta, ttl)
}

// Touch gives a live entry a new expiry and reports whether there was one.
func (s *DatabaseStore) Touch(ctx context.Context, key string, ttl time.Duration) (bool, error) {
	if ttl <= 0 {
		return false, ErrNoTTL
	}
	result, err := s.connection.ExecContext(ctx,
		"UPDATE "+s.table+" SET expiration = ? WHERE key = ? AND expiration > ?",
		expiresAt(ttl), s.prefix+key, nowMillis())
	if err != nil {
		return false, err
	}
	n, err := result.RowsAffected()
	if err != nil {
		return false, err
	}
	return n > 0, nil
}

// Forget removes a key, present or not.
//
// It answers DatabaseStore::forget(), including the companion entry: a flexible
// value keeps its timestamp under a second key, and leaving that behind would
// have the next reader age a value that is no longer there.
func (s *DatabaseStore) Forget(ctx context.Context, key string) error {
	_, err := s.connection.ExecContext(ctx,
		"DELETE FROM "+s.table+" WHERE key IN (?, ?)",
		s.prefix+key, s.prefix+flexibleCreatedKey(key))
	if err != nil {
		return fmt.Errorf("cache: removing %q from %s: %w", key, s.table, err)
	}
	return nil
}

// ForgetIfExpired removes a key, and only if it has expired.
//
// It answers DatabaseStore::forgetIfExpired(). It is what a read does with a row
// it found expired, and the expiration in the WHERE is what stops it from
// removing an entry that somebody rewrote in between.
func (s *DatabaseStore) ForgetIfExpired(ctx context.Context, key string) error {
	_, err := s.connection.ExecContext(ctx,
		"DELETE FROM "+s.table+" WHERE key IN (?, ?) AND expiration <= ?",
		s.prefix+key, s.prefix+flexibleCreatedKey(key), nowMillis())
	return err
}

// Flush removes every entry whose key begins with prefix.
//
// This store holds every tenant's cache, so it deletes by prefix. An empty
// prefix empties the table.
func (s *DatabaseStore) Flush(ctx context.Context, prefix string) error {
	if prefix == "" {
		_, err := s.connection.ExecContext(ctx, "DELETE FROM "+s.table)
		return err
	}
	_, err := s.connection.ExecContext(ctx,
		"DELETE FROM "+s.table+" WHERE key LIKE ?", likePrefix(s.prefix+prefix))
	if err != nil {
		return fmt.Errorf("cache: flushing %q from %s: %w", prefix, s.table, err)
	}
	return nil
}

// FlushLocks releases every lock this store holds.
//
// It answers DatabaseStore::flushLocks(), including the refusal: a store keeping
// its locks in the cache table cannot empty the first without emptying the
// second, and says so instead of doing it.
func (s *DatabaseStore) FlushLocks(ctx context.Context) error {
	if !s.HasSeparateLockStore() {
		return fmt.Errorf("%w: flushing locks needs a lock table separate from the cache table, and this store has none", ErrUnsupported)
	}
	_, err := s.lockConn().ExecContext(ctx, "DELETE FROM "+s.lockTable)
	return err
}

// AcquireLock takes the lock if it is free.
//
// It answers DatabaseLock::acquire(): insert, and when the row is already there,
// update it if it is this owner's or has expired. Both are one statement, so two
// processes racing cannot both win.
func (s *DatabaseStore) AcquireLock(ctx context.Context, key, token string, ttl time.Duration) (bool, error) {
	if ttl <= 0 {
		ttl = s.lockTimeout
	}
	expiration := expiresAt(ttl)

	acquired := false
	if _, err := s.lockConn().ExecContext(ctx,
		"INSERT INTO "+s.lockTable+" (key, owner, expiration) VALUES (?, ?, ?)",
		s.prefix+key, token, expiration); err == nil {
		acquired = true
	} else {
		result, err := s.lockConn().ExecContext(ctx,
			"UPDATE "+s.lockTable+" SET owner = ?, expiration = ? WHERE key = ? AND (owner = ? OR expiration <= ?)",
			token, expiration, s.prefix+key, token, nowMillis())
		if err != nil {
			return false, fmt.Errorf("cache: taking the lock %q in %s: %w", key, s.lockTable, err)
		}
		n, err := result.RowsAffected()
		if err != nil {
			return false, err
		}
		acquired = n >= 1
	}

	if s.drawLottery() {
		_ = s.PruneExpiredLocks(ctx)
	}
	return acquired, nil
}

// ReleaseLock releases the lock only if token still holds it.
//
// It answers DatabaseLock::release(), where the ownership check is in the WHERE
// rather than in a read before it: a check made in the application is a check
// that was true a moment ago.
func (s *DatabaseStore) ReleaseLock(ctx context.Context, key, token string) error {
	_, err := s.lockConn().ExecContext(ctx,
		"DELETE FROM "+s.lockTable+" WHERE key = ? AND owner = ?", s.prefix+key, token)
	return err
}

// CurrentOwner returns the token holding the lock, or the empty string.
//
// It answers DatabaseLock::getCurrentOwner(). A row past its expiration is a
// lock nobody holds, and says so rather than naming whoever held it last.
func (s *DatabaseStore) CurrentOwner(ctx context.Context, key string) (string, error) {
	var owner string
	var expiration int64
	err := s.lockConn().QueryRowContext(ctx,
		"SELECT owner, expiration FROM "+s.lockTable+" WHERE key = ?", s.prefix+key,
	).Scan(&owner, &expiration)
	switch {
	case errors.Is(err, sql.ErrNoRows):
		return "", nil
	case err != nil:
		return "", err
	case expiration <= nowMillis():
		return "", nil
	}
	return owner, nil
}

// PruneExpiredLocks deletes the locks that are past their expiration.
//
// AcquireLock runs it two times in a hundred, so the table stays bounded
// without any one caller paying for it often. Call it directly from a scheduled
// task to make it certain.
func (s *DatabaseStore) PruneExpiredLocks(ctx context.Context) error {
	_, err := s.lockConn().ExecContext(ctx,
		"DELETE FROM "+s.lockTable+" WHERE expiration <= ?", nowMillis())
	return err
}

// Lock returns a handle on a named lock. It does not touch the store.
func (s *DatabaseStore) Lock(name string, ttl time.Duration, owner string) *Lock {
	return &Lock{store: s, name: name, ttl: ttl, owner: owner, held: owner != ""}
}

// RestoreLock returns a handle on a lock owner already holds.
func (s *DatabaseStore) RestoreLock(name, owner string) *Lock { return s.Lock(name, 0, owner) }

// drawLottery decides whether this acquisition pays for the pruning.
func (s *DatabaseStore) drawLottery() bool {
	if s.lockLottery[1] <= 0 {
		return false
	}
	// The clock is the die. It is not a good source of randomness and it does not
	// need to be: nothing depends on which acquisition pays for the pruning, only
	// on it being roughly two in a hundred of them.
	return int(time.Now().UnixNano()%int64(s.lockLottery[1])) < s.lockLottery[0]
}

// nowMillis is the moment every comparison in this store is made against.
//
// Milliseconds and not seconds: see DatabaseStore for what that costs and what
// it buys.
func nowMillis() int64 { return time.Now().UnixMilli() }

// expiresAt is when an entry written now with this ttl stops being live.
func expiresAt(ttl time.Duration) int64 { return time.Now().Add(ttl).UnixMilli() }

// likePrefix turns a key prefix into a LIKE pattern, escaping the two
// characters LIKE reads as wildcards.
//
// Without it a namespace containing an underscore -- which auth.ValidTenant
// allows, so a tenant may contain one -- would match one character of any other
// tenant's name, and a flush for one customer would take entries belonging to
// another.
func likePrefix(prefix string) string {
	out := make([]byte, 0, len(prefix)+8)
	for i := range len(prefix) {
		switch prefix[i] {
		case '%', '_', '\\':
			out = append(out, '\\')
		}
		out = append(out, prefix[i])
	}
	return string(out) + "%"
}
