package session

import (
	"context"
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"net/http"
	"net/url"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/arandu-io/hesape/filesystem"
)

// sessionIsExpired is the one place the expiry boundary is written, for the
// three handlers that draw one.
//
// age is how long since the session was last written, and lifetime is how
// long a session lives. It is strictly greater, so the session whose age is
// exactly the lifetime is kept.
//
// It is one comparison rather than three, because three copies of a
// boundary is three places for the next one to drift: a session exactly two
// hours old under a two-hour lifetime has to come back the same way no
// matter which handler is asked.
func sessionIsExpired(age, lifetime time.Duration) bool { return age > lifetime }

// ArraySessionHandler keeps sessions in the process memory.
//
// It is the right handler for a test and for a single process that is restarted
// by hand. Behind more than one replica it signs people out on every deploy and
// on every request routed to the other pod, because each process holds its own
// map -- use [CacheBasedSessionHandler] or [DatabaseSessionHandler] there.
//
// It is not [ArrayHandler], which is the same idea for [RecordStore]: that one
// stores a typed [Record] and this one stores the bytes a [Store] serialised.
type ArraySessionHandler struct {
	mu       sync.RWMutex
	storage  map[string]arraySession
	lifetime time.Duration
}

type arraySession struct {
	data    string
	written time.Time
}

// NewArraySessionHandler returns an empty in-memory handler whose sessions
// live for lifetime.
func NewArraySessionHandler(lifetime time.Duration) *ArraySessionHandler {
	return &ArraySessionHandler{storage: map[string]arraySession{}, lifetime: lifetime}
}

var _ SessionHandler = (*ArraySessionHandler)(nil)

// Open does nothing: there is nothing to open.
func (h *ArraySessionHandler) Open(context.Context, string, string) error { return nil }

// Close does nothing.
func (h *ArraySessionHandler) Close(context.Context) error { return nil }

// Read returns the payload, or "" when the id is unknown or the session has
// been idle for longer than the lifetime.
//
// The boundary keeps the session whose age is exactly the lifetime. This
// used to drop it, so a session written exactly two hours ago under a
// two-hour lifetime read back empty -- a person signed out one request
// early, on the clock tick nobody reproduces on purpose.
func (h *ArraySessionHandler) Read(_ context.Context, sessionID string) (string, error) {
	h.mu.RLock()
	defer h.mu.RUnlock()

	entry, ok := h.storage[sessionID]
	if !ok {
		return "", nil
	}
	if sessionIsExpired(time.Since(entry.written), h.lifetime) {
		return "", nil
	}
	return entry.data, nil
}

// Write stores the payload and stamps it.
func (h *ArraySessionHandler) Write(_ context.Context, sessionID, data string) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	h.storage[sessionID] = arraySession{data: data, written: time.Now()}
	return nil
}

// Destroy removes the session, if present.
func (h *ArraySessionHandler) Destroy(_ context.Context, sessionID string) error {
	h.mu.Lock()
	defer h.mu.Unlock()
	delete(h.storage, sessionID)
	return nil
}

// GC removes every session idle for longer than lifetime and reports how
// many went. The entry whose age is exactly the lifetime is kept, which is
// the boundary [ArraySessionHandler.Read] names.
func (h *ArraySessionHandler) GC(_ context.Context, lifetime time.Duration) (int, error) {
	h.mu.Lock()
	defer h.mu.Unlock()

	deleted := 0
	for id, entry := range h.storage {
		if sessionIsExpired(time.Since(entry.written), lifetime) {
			delete(h.storage, id)
			deleted++
		}
	}
	return deleted, nil
}

// NullSessionHandler stores nothing and reads nothing back.
//
// Every request gets an empty session, and nothing said in one survives it. It
// is what a stateless API or a console binary wires so that code reaching for
// the session does not have to branch on whether there is one.
type NullSessionHandler struct{}

// NewNullSessionHandler returns a handler that stores nothing and reads
// nothing back.
func NewNullSessionHandler() *NullSessionHandler { return &NullSessionHandler{} }

var _ SessionHandler = (*NullSessionHandler)(nil)

// Open does nothing.
func (h *NullSessionHandler) Open(context.Context, string, string) error { return nil }

// Close does nothing.
func (h *NullSessionHandler) Close(context.Context) error { return nil }

// Read always answers with an empty session.
func (h *NullSessionHandler) Read(context.Context, string) (string, error) { return "", nil }

// Write discards the payload.
func (h *NullSessionHandler) Write(context.Context, string, string) error { return nil }

// Destroy does nothing.
func (h *NullSessionHandler) Destroy(context.Context, string) error { return nil }

// GC has nothing to collect.
func (h *NullSessionHandler) GC(context.Context, time.Duration) (int, error) { return 0, nil }

// FileSessionHandler stores each session as a file in a directory.
//
// It is the handler that needs nothing installed, and it is correct for one
// machine. Behind a load balancer it is the classic wrong answer: two pods have
// two directories, and a person is signed in on whichever one their request
// happened to reach.
//
// The read is a shared-locked read and the write is a locked write, so a request
// never sees half of what another one is writing. See
// [filesystem.LockableFile].
type FileSessionHandler struct {
	files    *filesystem.Filesystem
	path     string
	lifetime time.Duration
}

// NewFileSessionHandler returns a handler over a directory, creating it
// immediately rather than waiting for the first write.
func NewFileSessionHandler(files *filesystem.Filesystem, path string, lifetime time.Duration) (*FileSessionHandler, error) {
	if path == "" {
		return nil, errors.New("session: the file handler needs a directory")
	}
	if files == nil {
		files = filesystem.NewFilesystem()
	}
	if err := files.EnsureDirectoryExists(path, 0o700, true); err != nil {
		return nil, err
	}
	return &FileSessionHandler{files: files, path: path, lifetime: lifetime}, nil
}

var _ SessionHandler = (*FileSessionHandler)(nil)

// Open does nothing: the directory was made by the constructor.
func (h *FileSessionHandler) Open(context.Context, string, string) error { return nil }

// Close does nothing.
func (h *FileSessionHandler) Close(context.Context) error { return nil }

// file is where a session lives.
//
// The id has already been through [Store.IsValidID], which allows letters and
// digits only, so it cannot name a file outside this directory. filepath.Base is
// the second lock on the same door, because this method is reachable from a
// handler somebody wired by hand.
func (h *FileSessionHandler) file(sessionID string) string {
	return filepath.Join(h.path, filepath.Base(sessionID))
}

// Read returns the payload, or "" when the file is missing or older than
// the lifetime. A file modified exactly one lifetime ago is kept, which is
// the boundary [ArraySessionHandler.Read] names.
func (h *FileSessionHandler) Read(_ context.Context, sessionID string) (string, error) {
	path := h.file(sessionID)
	if !h.files.IsFile(path) {
		return "", nil
	}
	modified, err := h.files.LastModified(path)
	if err != nil {
		return "", err
	}
	if sessionIsExpired(time.Since(modified), h.lifetime) {
		return "", nil
	}
	body, err := h.files.SharedGet(path)
	if err != nil {
		if errors.Is(err, filesystem.ErrNotFound) {
			return "", nil
		}
		return "", err
	}
	return string(body), nil
}

// Write stores the payload under an exclusive lock.
func (h *FileSessionHandler) Write(_ context.Context, sessionID, data string) error {
	return h.files.Put(h.file(sessionID), []byte(data), true)
}

// Destroy removes the file, if present.
func (h *FileSessionHandler) Destroy(_ context.Context, sessionID string) error {
	return h.files.Delete(h.file(sessionID))
}

// GC removes every session file older than lifetime and reports how many
// went.
//
// The sweep is a recursive walk over files, skipping anything whose name
// starts with a dot -- including a dot directory, which is not descended
// into either. This used to do the opposite of both halves -- it read one
// level and deleted everything it found -- so a session written into a
// subdirectory was never collected and grew forever, while the .gitignore
// that is how an empty storage directory survives a clone went on the first
// sweep. Both were silent.
func (h *FileSessionHandler) GC(_ context.Context, lifetime time.Duration) (int, error) {
	deleted := 0
	err := filepath.WalkDir(h.path, func(path string, entry fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if strings.HasPrefix(entry.Name(), ".") && path != h.path {
			if entry.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}
		if entry.IsDir() {
			return nil
		}
		info, err := entry.Info()
		if err != nil {
			// The file went between the walk and the stat, which is another
			// process collecting the same directory. It is gone either way.
			return nil
		}
		if time.Since(info.ModTime()) < lifetime {
			return nil
		}
		if err := h.files.Delete(path); err != nil {
			return err
		}
		deleted++
		return nil
	})
	if err != nil {
		return deleted, fmt.Errorf("session: collecting %s: %w", h.path, err)
	}
	return deleted, nil
}

// CookieJar is the part of a cookie jar [CookieSessionHandler] uses.
//
// It is declared here, minimally, rather than imported: this package sits below
// the one that writes cookies to a response, and an interface with two methods
// on it is what keeps that true. hesape/cookie's jar is what an application
// wires behind it, through the two lines that turn a name, a value and a
// lifetime into the *http.Cookie that jar queues.
type CookieJar interface {
	// Queue puts a cookie on the response that is about to be written.
	Queue(name, value string, lifetime time.Duration)
	// Forget queues the cookie that tells the browser to drop this one.
	Forget(name string)
}

// CookieSessionHandler stores the whole session in the cookie.
//
// It is the handler with no server-side storage at all, which is the only thing
// to be said for it and is genuinely worth something: nothing to provision,
// nothing to garbage-collect, nothing that goes down. What it costs is that
// every byte of the session travels on every request, that it cannot be
// invalidated from the server, and that a browser drops a cookie over about
// 4096 bytes without saying so.
//
// Wire it with an encrypting jar or an [EncryptedStore]. Unencrypted, the
// session is readable by whoever holds the machine.
type CookieSessionHandler struct {
	cookie        CookieJar
	request       *http.Request
	lifetime      time.Duration
	expireOnClose bool
}

// NewCookieSessionHandler returns a handler that stores the session
// entirely in a cookie.
//
// expireOnClose gives the cookie no expiry, so the browser drops it when it is
// closed.
func NewCookieSessionHandler(cookie CookieJar, lifetime time.Duration, expireOnClose bool) *CookieSessionHandler {
	return &CookieSessionHandler{cookie: cookie, lifetime: lifetime, expireOnClose: expireOnClose}
}

var (
	_ SessionHandler = (*CookieSessionHandler)(nil)
	_ RequestAware   = (*CookieSessionHandler)(nil)
)

// SetRequest hands the handler the request it reads the cookie off.
//
// [Store.SetRequestOnHandler] calls it, and [StartSession] calls that before
// [Store.Start]. Without it there is nothing to read and every session is empty.
func (h *CookieSessionHandler) SetRequest(r *http.Request) { h.request = r }

// Open does nothing.
func (h *CookieSessionHandler) Open(context.Context, string, string) error { return nil }

// Close does nothing.
func (h *CookieSessionHandler) Close(context.Context) error { return nil }

// cookiePayload is what travels: the session, and when it stops being valid.
//
// The expiry is inside the value as well as on the cookie, because the cookie's
// own expiry is enforced by the browser and the browser is the thing being
// distrusted here.
type cookiePayload struct {
	Data    string `json:"data"`
	Expires int64  `json:"expires"`
}

// Read returns the payload carried by the request's cookie, or "" when
// there is none, it is unreadable, or it has run out.
//
// The value is percent-decoded first, unless it already begins with the brace
// that opens the JSON -- which is the cookie a jar handed over after decoding it
// itself. The discriminator is exact: a percent-encoded payload begins "%7B" and
// can begin nothing else, so there is no input the two readings both claim. See
// [CookieSessionHandler.Write] for what is on the wire and why.
func (h *CookieSessionHandler) Read(_ context.Context, sessionID string) (string, error) {
	if h.request == nil {
		return "", nil
	}
	c, err := h.request.Cookie(sessionID)
	if err != nil || c.Value == "" {
		return "", nil
	}
	value := c.Value
	if !strings.HasPrefix(value, "{") {
		decoded, err := url.QueryUnescape(value)
		if err != nil {
			return "", nil
		}
		value = decoded
	}
	var payload cookiePayload
	if err := json.Unmarshal([]byte(value), &payload); err != nil {
		return "", nil
	}
	if time.Now().Unix() > payload.Expires {
		return "", nil
	}
	return payload.Data, nil
}

// Write queues the cookie carrying the session.
//
// # What is on the wire
//
// The session and its expiry are JSON, percent-encoded before they are
// queued.
//
// It used to be base64 of the same JSON, and the reason given was that a
// cookie value cannot hold a quote, a comma or a semicolon -- true, and
// net/http's SetCookie sanitises those bytes away rather than encoding
// them, which is why the encoding has to happen here and not in the jar.
// But [CookieSessionHandler.Read] percent-decodes what it is handed, so a
// cookie written as base64 read back as an empty session, silently, and
// started a fresh one: the exact failure that waits until production to
// appear.
func (h *CookieSessionHandler) Write(_ context.Context, sessionID, data string) error {
	encoded, err := json.Marshal(cookiePayload{
		Data:    data,
		Expires: time.Now().Add(h.lifetime).Unix(),
	})
	if err != nil {
		return fmt.Errorf("session: encoding the session cookie: %w", err)
	}
	lifetime := h.lifetime
	if h.expireOnClose {
		lifetime = 0
	}
	h.cookie.Queue(sessionID, url.QueryEscape(string(encoded)), lifetime)
	return nil
}

// Destroy queues the cookie that tells the browser to drop it.
func (h *CookieSessionHandler) Destroy(_ context.Context, sessionID string) error {
	h.cookie.Forget(sessionID)
	return nil
}

// GC has nothing to collect: the browser expires the cookie.
func (h *CookieSessionHandler) GC(context.Context, time.Duration) (int, error) { return 0, nil }

// Cache is the part of a cache [CacheBasedSessionHandler] uses.
//
// It is declared here, minimally, rather than imported, so that a project
// keeping sessions on disk does not drag the cache package in behind them.
// hesape/cache's repository satisfies it, and so does anything else with these
// three methods.
type Cache interface {
	// Get returns the value, or "" when the key is not there.
	Get(ctx context.Context, key string) (string, error)
	// Put stores it for ttl.
	Put(ctx context.Context, key, value string, ttl time.Duration) error
	// Forget removes it.
	Forget(ctx context.Context, key string) error
}

// CacheBasedSessionHandler stores sessions in a cache.
//
// It is the handler for more than one replica: every pod reads the same store,
// so a deploy does not sign anybody out and a request may land anywhere. What it
// gives up is durability -- a cache that is flushed is every session ending at
// once, which is survivable and is why this is the usual answer rather than the
// database.
type CacheBasedSessionHandler struct {
	cache    Cache
	lifetime time.Duration
}

// NewCacheBasedSessionHandler returns a handler that stores sessions in
// cache.
func NewCacheBasedSessionHandler(cache Cache, lifetime time.Duration) *CacheBasedSessionHandler {
	return &CacheBasedSessionHandler{cache: cache, lifetime: lifetime}
}

var _ SessionHandler = (*CacheBasedSessionHandler)(nil)

// GetCache returns the cache underneath.
func (h *CacheBasedSessionHandler) GetCache() Cache { return h.cache }

// Open does nothing.
func (h *CacheBasedSessionHandler) Open(context.Context, string, string) error { return nil }

// Close does nothing.
func (h *CacheBasedSessionHandler) Close(context.Context) error { return nil }

// Read returns the payload, or "" when the key is not there.
func (h *CacheBasedSessionHandler) Read(ctx context.Context, sessionID string) (string, error) {
	return h.cache.Get(ctx, sessionID)
}

// Write stores the payload for the session lifetime.
func (h *CacheBasedSessionHandler) Write(ctx context.Context, sessionID, data string) error {
	return h.cache.Put(ctx, sessionID, data, h.lifetime)
}

// Destroy removes it.
func (h *CacheBasedSessionHandler) Destroy(ctx context.Context, sessionID string) error {
	return h.cache.Forget(ctx, sessionID)
}

// GC has nothing to collect: the cache expires entries itself.
func (h *CacheBasedSessionHandler) GC(context.Context, time.Duration) (int, error) { return 0, nil }

// Connection is the part of a database [DatabaseSessionHandler] uses.
//
// It is declared here rather than imported from hesape/database, minimally and
// on purpose: this handler needs two calls, and *sql.DB and *sql.Tx both satisfy
// it unchanged.
type Connection interface {
	ExecContext(ctx context.Context, query string, args ...any) (sql.Result, error)
	QueryRowContext(ctx context.Context, query string, args ...any) *sql.Row
}

// DatabaseSessionHandler stores sessions in a table.
//
// It is the durable answer: a restart of anything loses nothing, and a session
// can be ended from the database by an operator who has to. What it costs is a
// write on every request, which is the reason [CacheBasedSessionHandler] is the
// usual choice and this one is for when the sessions matter.
//
// The table is the one [console.SessionTableCommand] emits a migration for.
// Three of its columns -- user_id, ip_address and user_agent -- are always
// left null: a handler that accepted a request to fill them would make
// [Store.HandlerNeedsRequest] answer true for it, which is the flag that
// says "this session lives in the request", and it does not. An
// application that wants those columns writes them beside the session,
// where it knows who is signed in.
type DatabaseSessionHandler struct {
	connection Connection
	table      string
	lifetime   time.Duration
	exists     bool
}

// NewDatabaseSessionHandler returns a handler that stores sessions in
// table, over connection. See [DatabaseSessionHandler] for what it does and
// does not write.
//
// table is quoted into the statements as an identifier, so it must be a name the
// application chose and never one that came from a request. It is checked for
// that: see [ErrBadTableName].
func NewDatabaseSessionHandler(connection Connection, table string, lifetime time.Duration) (*DatabaseSessionHandler, error) {
	if !validTableName(table) {
		return nil, fmt.Errorf("%w: %q", ErrBadTableName, table)
	}
	return &DatabaseSessionHandler{connection: connection, table: table, lifetime: lifetime}, nil
}

// ErrBadTableName is returned for a table name that is not a plain identifier.
//
// A table name cannot be a parameter in SQL -- it is part of the statement, not
// of the data -- so the only defence is refusing anything that is not letters,
// digits and underscores. This is the line that means a configuration value read
// from the environment cannot become an injection.
var ErrBadTableName = errors.New("session: the table name must be letters, digits and underscores")

func validTableName(table string) bool {
	if table == "" {
		return false
	}
	for _, c := range table {
		switch {
		case c >= 'a' && c <= 'z', c >= 'A' && c <= 'Z', c >= '0' && c <= '9', c == '_':
		default:
			return false
		}
	}
	return true
}

var (
	_ SessionHandler          = (*DatabaseSessionHandler)(nil)
	_ ExistenceAwareInterface = (*DatabaseSessionHandler)(nil)
)

// SetExists tells the handler whether the row is already there, which is
// what decides between an insert and an update.
func (h *DatabaseSessionHandler) SetExists(value bool) { h.exists = value }

// Open does nothing: the connection was opened by whoever passed it in.
func (h *DatabaseSessionHandler) Open(context.Context, string, string) error { return nil }

// Close does nothing, and deliberately does not close the connection: it is
// shared with the rest of the application.
func (h *DatabaseSessionHandler) Close(context.Context) error { return nil }

// Read returns the payload, or "" when the row is missing or too old. It
// records that the row exists even when the session has expired, because
// the row is still there and the write that follows has to be an update.
//
// A row whose payload is not valid base64 reads back empty rather than as
// garbage.
//
// The row whose last_activity is exactly one lifetime old is live, which
// is the boundary [ArraySessionHandler.Read] names. It used to be treated
// as expired instead.
func (h *DatabaseSessionHandler) Read(ctx context.Context, sessionID string) (string, error) {
	var payload string
	var lastActivity int64
	err := h.connection.
		QueryRowContext(ctx, `SELECT payload, last_activity FROM `+h.table+` WHERE id = $1`, sessionID).
		Scan(&payload, &lastActivity)
	if errors.Is(err, sql.ErrNoRows) {
		return "", nil
	}
	if err != nil {
		return "", fmt.Errorf("session: reading the session row: %w", err)
	}

	h.exists = true
	if sessionIsExpired(time.Since(time.Unix(lastActivity, 0)), h.lifetime) {
		return "", nil
	}
	decoded, err := base64.StdEncoding.DecodeString(payload)
	if err != nil {
		// A row this binary cannot decode is one an older release wrote. An
		// empty session signs the person out; an error would give them a 500 on
		// every page until somebody went into the table.
		return "", nil
	}
	return string(decoded), nil
}

// Write inserts or updates the row. The columns user_id, ip_address and
// user_agent are left null; see [DatabaseSessionHandler].
func (h *DatabaseSessionHandler) Write(ctx context.Context, sessionID, data string) error {
	payload := base64.StdEncoding.EncodeToString([]byte(data))
	now := time.Now().Unix()

	if !h.exists {
		if _, err := h.Read(ctx, sessionID); err != nil {
			return err
		}
	}
	if h.exists {
		_, err := h.connection.ExecContext(ctx,
			`UPDATE `+h.table+` SET payload = $1, last_activity = $2 WHERE id = $3`,
			payload, now, sessionID)
		if err != nil {
			return fmt.Errorf("session: updating the session row: %w", err)
		}
		return nil
	}

	_, err := h.connection.ExecContext(ctx,
		`INSERT INTO `+h.table+` (id, payload, last_activity) VALUES ($1, $2, $3)`,
		sessionID, payload, now)
	if err != nil {
		// Two requests of the same session inserting at once is a real race
		// and not an error the person can act on: one of them wins the
		// insert and the other updates.
		if _, updateErr := h.connection.ExecContext(ctx,
			`UPDATE `+h.table+` SET payload = $1, last_activity = $2 WHERE id = $3`,
			payload, now, sessionID); updateErr != nil {
			return fmt.Errorf("session: writing the session row: %w", err)
		}
	}
	h.exists = true
	return nil
}

// Destroy removes the row.
func (h *DatabaseSessionHandler) Destroy(ctx context.Context, sessionID string) error {
	if _, err := h.connection.ExecContext(ctx, `DELETE FROM `+h.table+` WHERE id = $1`, sessionID); err != nil {
		return fmt.Errorf("session: deleting the session row: %w", err)
	}
	return nil
}

// GC removes every row idle for longer than lifetime and reports how many
// went.
func (h *DatabaseSessionHandler) GC(ctx context.Context, lifetime time.Duration) (int, error) {
	cutoff := time.Now().Add(-lifetime).Unix()
	result, err := h.connection.ExecContext(ctx, `DELETE FROM `+h.table+` WHERE last_activity <= $1`, cutoff)
	if err != nil {
		return 0, fmt.Errorf("session: collecting expired sessions: %w", err)
	}
	deleted, err := result.RowsAffected()
	if err != nil {
		// A driver that does not count rows is not a failure to collect them.
		return 0, nil
	}
	return int(deleted), nil
}
