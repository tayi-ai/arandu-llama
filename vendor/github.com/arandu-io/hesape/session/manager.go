package session

import (
	"errors"
	"fmt"
	"net/http"
	"sync"
	"time"
)

// Config is the session configuration, as a struct.
//
// Typed rather than a map, for the reason every build configuration here is:
// a misspelled key in a map is an application that boots and then behaves like
// a different one -- the cookie secure flag silently false, the lifetime
// silently zero -- and a struct makes the misspelling a compile error.
type Config struct {
	// Driver names the handler: "file", "cookie", "database", "cache", "array"
	// or "null", or one registered with [SessionManager.Extend].
	Driver string

	// Cookie is the name the session id travels under.
	Cookie string

	// Lifetime is how long a session survives without being touched. Zero means
	// [DefaultLifetime].
	Lifetime time.Duration

	// ExpireOnClose gives the cookie no expiry, so the browser drops it when it
	// closes.
	ExpireOnClose bool

	// Encrypt seals the payload before it reaches the handler. See
	// [EncryptedStore]; it is required in practice for the cookie driver.
	Encrypt bool

	// Files is the directory the file driver writes into.
	Files string

	// Connection and Table configure the database driver.
	Connection string
	Table      string

	// Store names the cache the cache driver uses.
	Store string

	// The cookie's own attributes. Secure and HTTPOnly are not defaulted true
	// by this struct because the zero value of a bool is false and a field that
	// silently corrects itself would also silently ignore an explicit false --
	// [Config.Check] refuses an unsafe combination out loud instead, and
	// [SessionManager.Driver] runs it before there is a session to put in a
	// cookie.
	Path        string
	Domain      string
	Secure      bool
	HTTPOnly    bool
	SameSite    http.SameSite
	Partitioned bool

	// Block makes requests of the same session wait for each other, so two
	// tabs cannot write the session at the same time and lose one of the
	// writes. BlockStore names the lock store; the two second fields are the
	// ceilings.
	Block            bool
	BlockStore       string
	BlockLockSeconds time.Duration
	BlockWaitSeconds time.Duration

	// Lottery is the odds of a request collecting garbage: {2, 100} is two in
	// a hundred. A zero denominator turns collection off, which is right for
	// a handler whose store expires entries itself.
	Lottery [2]int
}

// The defaults for the fields a zero value cannot stand in for.
const (
	// DefaultLifetime is two hours.
	DefaultLifetime = 2 * time.Hour
	// DefaultBlockLockSeconds is how long a route holds the session lock.
	DefaultBlockLockSeconds = 10 * time.Second
	// DefaultBlockWaitSeconds is how long it waits to take one.
	DefaultBlockWaitSeconds = 10 * time.Second
)

// ErrNoDriver is returned for a driver nobody registered.
var ErrNoDriver = errors.New("session: no such session driver")

// HandlerCreator builds a handler out of the configuration.
//
// It is what [SessionManager.Extend] registers, and it is how the drivers that
// need a collaborator get built: the cookie driver needs a jar, the cache driver
// a cache and the database driver a connection, and there is no container here
// to fetch them from. The application that has them registers three
// closures at boot, in one place a person can read.
type HandlerCreator func(cfg Config) (SessionHandler, error)

// SessionManager builds sessions from configuration: which driver, which
// cookie, how long, and whether requests of one session wait for each
// other.
//
// # Driver returns a new Store every call
//
// A process here serves thousands of requests at once, and a [Store] is a
// bag of keys with no lock on it: handing the same one to two requests
// would interleave their writes and save each other's session. The HANDLER
// is shared -- that is the connection, the directory or the map, and it is
// built once.
type SessionManager struct {
	mu        sync.RWMutex
	config    Config
	encrypter Encrypter
	creators  map[string]HandlerCreator
	handlers  map[string]SessionHandler
}

// NewSessionManager returns a manager built from a typed [Config].
//
// encrypter may be nil when Config.Encrypt is false. When it is true and the
// encrypter is nil, [SessionManager.Driver] refuses rather than quietly storing
// the session in the clear.
func NewSessionManager(cfg Config, encrypter Encrypter) *SessionManager {
	if cfg.Lifetime <= 0 {
		cfg.Lifetime = DefaultLifetime
	}
	if cfg.Cookie == "" {
		cfg.Cookie = CookieName
	}
	if cfg.Path == "" {
		cfg.Path = "/"
	}
	return &SessionManager{
		config:    cfg,
		encrypter: encrypter,
		creators:  map[string]HandlerCreator{},
		handlers:  map[string]SessionHandler{},
	}
}

// Extend registers a handler creator under a driver name. It returns the
// handler, and the [Store] around it is built the same way for every
// driver.
//
// Registering a name that is already built in replaces it, which is what makes
// "file" swappable in a test without a second configuration path.
func (m *SessionManager) Extend(driver string, creator HandlerCreator) *SessionManager {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.creators[driver] = creator
	delete(m.handlers, driver)
	return m
}

// Driver returns a session for this request, on the named driver. The empty
// name is the configured default.
//
// The id it starts with is empty, so a fresh one is minted; [StartSession] sets
// the one the browser sent immediately afterwards, which is why a missing
// cookie is not a special case anywhere.
//
// It refuses an unsafe cookie configuration here rather than where the cookie is
// written, because this is the one call every request makes before there is a
// session id to carry: a refusal that arrives at the cookie has already let the
// request run.
func (m *SessionManager) Driver(name string) (*Store, error) {
	if err := m.GetSessionConfig().Check(); err != nil {
		return nil, err
	}
	if name == "" {
		name = m.GetDefaultDriver()
	}
	if name == "" {
		return nil, fmt.Errorf("%w: no driver was asked for and none is configured", ErrNoDriver)
	}
	handler, err := m.handler(name)
	if err != nil {
		return nil, err
	}
	return m.buildSession(handler, "")
}

// handler returns the shared handler for a driver, building it once.
func (m *SessionManager) handler(name string) (SessionHandler, error) {
	m.mu.RLock()
	handler, ok := m.handlers[name]
	m.mu.RUnlock()
	if ok {
		return handler, nil
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	if handler, ok := m.handlers[name]; ok {
		return handler, nil
	}
	handler, err := m.createHandler(name)
	if err != nil {
		return nil, err
	}
	m.handlers[name] = handler
	return handler, nil
}

// createHandler builds the handler for a driver. The caller holds the lock.
func (m *SessionManager) createHandler(name string) (SessionHandler, error) {
	if creator, ok := m.creators[name]; ok {
		return creator(m.config)
	}
	switch name {
	case "array":
		return NewArraySessionHandler(m.config.Lifetime), nil
	case "null":
		return NewNullSessionHandler(), nil
	case "file", "native":
		return NewFileSessionHandler(nil, m.config.Files, m.config.Lifetime)
	case "cookie", "database", "cache", "redis":
		// These three need a collaborator this package has no way to reach for
		// -- a cookie jar, a connection, a cache. Saying so names the fix; a
		// creator built here that reached for a global would be the container
		// this collection does not have.
		return nil, fmt.Errorf(
			"%w: the %q driver needs a collaborator. Register it with SessionManager.Extend(%q, ...)",
			ErrNoDriver, name, name)
	}
	return nil, fmt.Errorf("%w: %q", ErrNoDriver, name)
}

// buildSession wraps a handler in a [Store], or an [EncryptedStore] when the
// configuration asks for one.
func (m *SessionManager) buildSession(handler SessionHandler, id string) (*Store, error) {
	if !m.config.Encrypt {
		return NewStore(m.config.Cookie, handler, id), nil
	}
	if m.encrypter == nil {
		return nil, errors.New("session: the configuration asks for encrypted sessions and no encrypter was given")
	}
	return NewEncryptedStore(m.config.Cookie, handler, m.encrypter, id).Store, nil
}

// ShouldBlock reports whether requests of one session should wait for each
// other.
//
// Two tabs writing the same session at once is a lost write: both read, both
// change one key, and the second save overwrites the first. Blocking costs a
// lock per request and is worth it wherever the session carries anything a
// person would notice losing.
func (m *SessionManager) ShouldBlock() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.config.Block
}

// BlockDriver names the store the session lock is taken in, or "" for the
// default one.
func (m *SessionManager) BlockDriver() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.config.BlockStore
}

// DefaultRouteBlockLockSeconds is how long a route may hold the session
// lock before it is broken. Zero configures [DefaultBlockLockSeconds].
func (m *SessionManager) DefaultRouteBlockLockSeconds() time.Duration {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.config.BlockLockSeconds <= 0 {
		return DefaultBlockLockSeconds
	}
	return m.config.BlockLockSeconds
}

// DefaultRouteBlockWaitSeconds is how long a request waits to take the
// session lock before giving up. Zero configures [DefaultBlockWaitSeconds].
func (m *SessionManager) DefaultRouteBlockWaitSeconds() time.Duration {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.config.BlockWaitSeconds <= 0 {
		return DefaultBlockWaitSeconds
	}
	return m.config.BlockWaitSeconds
}

// GetSessionConfig returns the configuration.
func (m *SessionManager) GetSessionConfig() Config {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.config
}

// GetDefaultDriver returns the configured driver name.
func (m *SessionManager) GetDefaultDriver() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.config.Driver
}

// SetDefaultDriver changes the configured driver name.
func (m *SessionManager) SetDefaultDriver(name string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.config.Driver = name
}
