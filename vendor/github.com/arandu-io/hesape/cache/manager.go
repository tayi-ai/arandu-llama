package cache

import (
	"fmt"
	"io/fs"
	"sync"
)

// Config is every cache store an application has, and which of them is the
// default.
//
// It is a typed struct and not a map of strings because a build configuration
// that is checked by the compiler is a misconfiguration found at build time
// rather than at the first cache miss.
type Config struct {
	// Default is the store Store and Driver return when nobody names one. An
	// empty one means "null".
	Default string

	// Prefix goes in front of every key a store writes, under the tenant and
	// the namespace. It is the one for sharing a backend with another
	// application -- not for separating tenants, which is Repository's job and
	// is not optional.
	Prefix string

	// Stores is every store by name.
	Stores map[string]StoreConfig
}

// StoreConfig is one store's configuration.
//
// It holds the union of what the drivers need, and each driver reads its own
// fields. That is one struct rather than one per driver because the
// configuration of a cache is read by a person choosing between them, and a
// person choosing between them wants one page.
type StoreConfig struct {
	// Driver names the kind: "array", "file", "database", "null", "failover",
	// or anything Extend registered.
	Driver string

	// Name is what this store is called, and it is filled in by Resolve from
	// the key in Config.Stores. It is what every event reports.
	Name string

	// Prefix overrides Config.Prefix for this store.
	Prefix string

	// NoEvents turns the events off for this store. The memo and failover
	// drivers set it, and it is negative so that the zero value keeps the
	// events on.
	NoEvents bool

	// Path is the directory the file driver writes in.
	Path string

	// LockPath is the directory the file driver keeps locks in. Setting it is
	// what makes FlushLocks possible.
	LockPath string

	// Permission is the mode the file driver gives what it creates; zero leaves
	// whatever the umask produced.
	Permission fs.FileMode

	// Connection is the database driver's connection, already resolved: there
	// is nothing here to look a name up in, so the connection itself is what
	// the configuration carries.
	Connection Connection

	// LockConnection is the connection the database driver manages locks on.
	LockConnection Connection

	// Table is the database driver's table.
	Table string

	// LockTable is the database driver's lock table; empty means "cache_locks".
	LockTable string

	// Stores is the failover driver's ordered set of store names.
	Stores []string
}

// Creator builds a repository for a driver the manager does not know.
//
// It takes the manager so a creator can reach the other stores -- which is what
// the failover driver does -- and returns the finished repository, because a
// driver that needs a store this package has never heard of also decides how to
// wrap it.
type Creator func(m *CacheManager, config StoreConfig) (*Repository, error)

// CacheManager is every cache store an application has, by name.
//
// It resolves a store the first time it is asked for and keeps it, so
// Store("redis") called in forty places is one connection and not forty.
//
// It is not a container. It holds a Config that was built at wiring time, and
// the thing it hands back is an ordinary *Repository that a caller could
// equally have built itself. What it buys is the one place that knows what "the
// file store" means, so a second module cannot decide it means a different
// directory.
//
// The drivers it builds are array, file, database, null and failover. Redis is
// not one of them: the RESP store is hesape/redis, a separate module, so that
// its driver ships only in the binaries that use it. Register it with Extend,
// which is what Extend is for.
//
// Memo is not a driver either. It is a method, because a memoized store is per
// request and a configuration is not.
//
// A CacheManager is safe for concurrent use.
type CacheManager struct {
	mu       sync.Mutex
	config   Config
	stores   map[string]*Repository
	creators map[string]Creator
	events   Dispatcher
}

// NewCacheManager returns the manager over a configuration.
func NewCacheManager(config Config) *CacheManager {
	return &CacheManager{
		config:   config,
		stores:   map[string]*Repository{},
		creators: map[string]Creator{},
	}
}

// Store returns the named store, wrapped in a repository, building it the first
// time.
//
// An empty name means the default. The repository is kept, so two callers
// asking for the same store get the same one -- which matters for the memo
// store, whose whole value is that it is shared for the length of a request.
func (m *CacheManager) Store(name string) (*Repository, error) {
	if name == "" {
		name = m.GetDefaultDriver()
	}

	m.mu.Lock()
	if store, ok := m.stores[name]; ok {
		m.mu.Unlock()
		return store, nil
	}
	m.mu.Unlock()

	store, err := m.Resolve(name)
	if err != nil {
		return nil, err
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	// Somebody may have resolved the same name while this one was building.
	// Theirs wins, so everybody ends up with one store and not two.
	if existing, ok := m.stores[name]; ok {
		return existing, nil
	}
	m.stores[name] = store
	return store, nil
}

// Driver is Store under another name.
func (m *CacheManager) Driver(driver string) (*Repository, error) { return m.Store(driver) }

// Memo returns a repository that remembers, for as long as it is held, what the
// named store already answered.
//
// The repository is kept like any other, so every caller asking for the same
// memo store shares one map -- which is the point: four parts of one request
// asking for the same feature flag make one round trip.
//
// Hold it for a request or a job and let it go. One held for the life of the
// process never forgets anything: see MemoizedStore.
//
// Its events are off, because the store underneath already fired them and a
// memoized hit is not a second cache hit.
func (m *CacheManager) Memo(driver string) (*Repository, error) {
	if driver == "" {
		driver = m.GetDefaultDriver()
	}
	name := "__memoized:" + driver

	m.mu.Lock()
	if store, ok := m.stores[name]; ok {
		m.mu.Unlock()
		return store, nil
	}
	m.mu.Unlock()

	underneath, err := m.Store(driver)
	if err != nil {
		return nil, err
	}
	built := m.Repository(NewMemoizedStore(driver, underneath.GetStore()), StoreConfig{Name: driver, NoEvents: true})

	m.mu.Lock()
	defer m.mu.Unlock()
	if existing, ok := m.stores[name]; ok {
		return existing, nil
	}
	m.stores[name] = built
	return built, nil
}

// Resolve builds the named store, without keeping it.
//
// A name that is not in the configuration is an error naming it, because the
// alternative is a cache that silently does nothing.
func (m *CacheManager) Resolve(name string) (*Repository, error) {
	if name == "null" {
		return m.Build(StoreConfig{Driver: "null", Name: "null"})
	}

	m.mu.Lock()
	config, ok := m.config.Stores[name]
	m.mu.Unlock()
	if !ok {
		return nil, fmt.Errorf("cache: the store %q is not defined in the configuration", name)
	}
	if config.Name == "" {
		config.Name = name
	}
	return m.Build(config)
}

// Build makes a repository out of one store's configuration.
//
// It is exported because a store that exists for one job, built from a
// configuration nobody wants to name, is a real thing. A configuration carrying
// no name is called "ondemand".
func (m *CacheManager) Build(config StoreConfig) (*Repository, error) {
	if config.Name == "" {
		config.Name = "ondemand"
	}

	m.mu.Lock()
	creator, custom := m.creators[config.Driver]
	prefix := m.config.Prefix
	m.mu.Unlock()

	if custom {
		return creator(m, config)
	}
	if config.Prefix == "" {
		config.Prefix = prefix
	}

	switch config.Driver {
	case "array", "":
		return m.Repository(NewArrayStore(), config), nil
	case "null":
		return m.Repository(NewNullStore(), config), nil
	case "file":
		if config.Path == "" {
			return nil, fmt.Errorf("cache: the file store %q has no path, and a file cache with no directory has nowhere to be", config.Name)
		}
		store := NewFileStore(LocalFilesystem{}, config.Path, config.Permission)
		if config.LockPath != "" {
			store.SetLockDirectory(config.LockPath)
		}
		return m.Repository(store, config), nil
	case "database":
		if config.Connection == nil {
			return nil, fmt.Errorf("cache: the database store %q has no connection", config.Name)
		}
		if config.Table == "" {
			return nil, fmt.Errorf("cache: the database store %q has no table", config.Name)
		}
		store := NewDatabaseStore(config.Connection, config.Table, config.Prefix, config.LockTable)
		if config.LockConnection != nil {
			store.SetLockConnection(config.LockConnection)
		}
		return m.Repository(store, config), nil
	case "failover":
		stores := make([]Store, 0, len(config.Stores))
		for _, name := range config.Stores {
			built, err := m.Store(name)
			if err != nil {
				return nil, err
			}
			stores = append(stores, built.GetStore())
		}
		config.NoEvents = true
		return m.Repository(NewFailoverStore(m.events, config.Stores, stores...), config), nil
	default:
		return nil, fmt.Errorf("cache: the driver %q is not one this package builds. Register it with Extend, which is how the RESP store in hesape/redis arrives", config.Driver)
	}
}

// Repository wraps a store in a repository, with the events wired unless the
// configuration turned them off.
//
// It is exported because a driver registered with Extend has to be able to
// build the same kind of repository the built-in ones do, and doing it by hand
// means getting the event wiring right by hand.
func (m *CacheManager) Repository(store Store, config StoreConfig) *Repository {
	out := New(store).SetName(config.Name)

	m.mu.Lock()
	events := m.events
	m.mu.Unlock()

	if config.NoEvents || events == nil {
		return out
	}
	return out.SetEventDispatcher(events)
}

// SetEventDispatcher sets the dispatcher new repositories are built with.
//
// It does not touch the repositories already built: RefreshEventDispatcher does
// that, and keeping the two apart is what lets an application wire the bus after
// the cache without every store having been built with a nil one.
func (m *CacheManager) SetEventDispatcher(d Dispatcher) *CacheManager {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.events = d
	return m
}

// RefreshEventDispatcher re-sets the dispatcher on every store already built.
//
// It is what an application calls after the event bus exists: the stores
// resolved during boot were built without one, and this is what gives it to
// them.
func (m *CacheManager) RefreshEventDispatcher() *CacheManager {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.events == nil {
		return m
	}
	for name, store := range m.stores {
		m.stores[name] = store.SetEventDispatcher(m.events)
	}
	return m
}

// GetDefaultDriver is the store that is used when nobody names one.
//
// An application that configured no default gets "null", which caches nothing
// and breaks nothing.
func (m *CacheManager) GetDefaultDriver() string {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.config.Default == "" {
		return "null"
	}
	return m.config.Default
}

// SetDefaultDriver sets it.
//
// It does not forget the store that was the default: call ForgetDriver for
// that.
func (m *CacheManager) SetDefaultDriver(name string) *CacheManager {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.config.Default = name
	return m
}

// ForgetDriver drops the named stores, so the next Store rebuilds them.
//
// No names means the default one.
func (m *CacheManager) ForgetDriver(names ...string) *CacheManager {
	if len(names) == 0 {
		names = []string{m.GetDefaultDriver()}
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	for _, name := range names {
		delete(m.stores, name)
	}
	return m
}

// Purge drops one store, so the next Store rebuilds it. It is ForgetDriver for
// a single name. An empty name means the default one.
func (m *CacheManager) Purge(name string) {
	if name == "" {
		name = m.GetDefaultDriver()
	}
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.stores, name)
}

// Extend registers a driver this package does not know.
//
// It is how the RESP store in hesape/redis arrives -- that store is a separate
// module, so registering it here is what keeps its driver out of binaries that
// do not use it -- and how anything else does: the creator is called with the
// manager and the configuration, and returns the finished repository.
//
// It replaces a creator of the same name rather than refusing, so an
// application can override a built-in driver by registering one with its name.
func (m *CacheManager) Extend(driver string, creator Creator) *CacheManager {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.creators[driver] = creator
	return m
}
