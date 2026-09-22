package database

import (
	"crypto/md5"
	"fmt"
	"sort"
	"strings"
	"sync"
)

// Configuration is where DatabaseManager reads its connections: the two keys
// 'database.default' and 'database.connections'.
//
// It is an interface rather than hesape/config directly because the manager has
// to be constructible in a test with three lines and no file.
type Configuration interface {
	// Get returns the configuration value for key.
	Get(key string) any

	// Set replaces the configuration value for key.
	Set(key string, value any)
}

// MapConfiguration is a Configuration backed by a map, for a test or a worker
// wired by hand.
type MapConfiguration map[string]any

// Get answers Configuration.Get.
func (m MapConfiguration) Get(key string) any { return m[key] }

// Set answers Configuration.Set.
func (m MapConfiguration) Set(key string, value any) { m[key] = value }

// DatabaseManager is the resolver that makes connections on demand from
// configuration, keeps them, and hands the same one back next time.
//
// Nothing is forwarded to the default connection: Connection(name) answers the
// connection, and the call goes on that. It is one more word and one fewer
// indirection, and it is the only shape a compiler can check.
type DatabaseManager struct {
	mu sync.RWMutex

	// config is where the manager reads and writes the default connection
	// name.
	config Configuration

	// factory builds a *Connection from a driver configuration.
	factory *ConnectionFactory

	// connections is every connection the manager has made, keyed by name.
	connections map[string]*Connection

	// dynamicConnectionConfigurations is what Build leaves behind so a
	// reconnect can rebuild an on-demand connection.
	dynamicConnectionConfigurations map[string]map[string]any

	// extensions is the per-name and per-driver connection constructors
	// registered with Extend.
	extensions map[string]func(config map[string]any, name string) (*Connection, error)

	// reconnector is set on every connection the manager makes.
	reconnector func(*Connection) error

	// events is the dispatcher the manager puts on every connection it
	// makes.
	events Dispatcher

	// transactions is the manager put on every connection the manager makes.
	transactions *DatabaseTransactionsManager
}

// NewDatabaseManager creates a DatabaseManager over config, building
// connections with factory, or with a default ConnectionFactory when
// factory is nil.
func NewDatabaseManager(config Configuration, factory *ConnectionFactory) *DatabaseManager {
	if factory == nil {
		factory = NewConnectionFactory()
	}

	m := &DatabaseManager{
		config:                          config,
		factory:                         factory,
		connections:                     map[string]*Connection{},
		dynamicConnectionConfigurations: map[string]map[string]any{},
		extensions:                      map[string]func(map[string]any, string) (*Connection, error){},
	}

	m.reconnector = func(connection *Connection) error {
		fresh, err := m.Reconnect(connection.GetNameWithReadWriteType())
		if err != nil {
			return err
		}
		connection.SetPDO(fresh.GetRawPDO())
		return nil
	}

	return m
}

// Connection returns the named connection, creating it if this is the first
// request for it.
//
// The name may carry a read/write suffix, `::read` or `::write`, which is
// what makes a replica addressable by name.
func (m *DatabaseManager) Connection(name string) (ConnectionInterface, error) {
	return m.connection(name)
}

// connection is Connection with the concrete type, which everything inside this
// file wants and the interface method cannot answer.
func (m *DatabaseManager) connection(name string) (*Connection, error) {
	if name == "" {
		name = m.GetDefaultConnection()
	}

	m.mu.RLock()
	existing, found := m.connections[name]
	m.mu.RUnlock()
	if found {
		return existing, nil
	}

	database, typ := parseConnectionName(name)

	made, err := m.makeConnection(database)
	if err != nil {
		return nil, err
	}
	made = m.configure(made, typ)

	m.mu.Lock()
	// Another goroutine may have made the same connection while this one was
	// opening a socket. There is no lock across that gap, so the first one to
	// arrive wins here, and the second's pool is closed rather than leaked.
	if raced, taken := m.connections[name]; taken {
		m.mu.Unlock()
		if pool := made.GetRawPDO(); pool != nil {
			_ = pool.Close()
		}
		return raced, nil
	}
	m.connections[name] = made
	m.mu.Unlock()

	m.dispatchConnectionEstablishedEvent(made)

	return made, nil
}

// Build opens a connection from a configuration nobody wrote in a file.
func (m *DatabaseManager) Build(config map[string]any) (*Connection, error) {
	name, _ := config["name"].(string)
	if name == "" {
		name = CalculateDynamicConnectionName(config)
		config["name"] = name
	}

	m.mu.Lock()
	m.dynamicConnectionConfigurations[name] = config
	m.mu.Unlock()

	return m.ConnectUsing(name, config, true)
}

// CalculateDynamicConnectionName derives a stable name for an on-demand
// connection by hashing its configuration.
//
// It concatenates key and value for every entry and hashes the result; map
// order is not stable, so the keys are sorted first, or the same
// configuration would get two different names on different runs.
func CalculateDynamicConnectionName(config map[string]any) string {
	keys := make([]string, 0, len(config))
	for key := range config {
		keys = append(keys, key)
	}
	sort.Strings(keys)

	var b strings.Builder
	for _, key := range keys {
		b.WriteString(key)
		switch value := config[key].(type) {
		case string:
			b.WriteString(value)
		case int:
			fmt.Fprintf(&b, "%d", value)
		}
	}

	return fmt.Sprintf("dynamic_%x", md5.Sum([]byte(b.String())))
}

// ConnectUsing opens a connection under name from config, failing if one
// already exists under that name unless force purges it first.
func (m *DatabaseManager) ConnectUsing(name string, config map[string]any, force bool) (*Connection, error) {
	if force {
		m.Purge(name)
	}

	m.mu.RLock()
	_, taken := m.connections[name]
	m.mu.RUnlock()
	if taken {
		return nil, fmt.Errorf("Cannot establish connection [%s] because another connection with that name already exists.", name)
	}

	made, err := m.factory.Make(config, name)
	if err != nil {
		return nil, err
	}
	made = m.configure(made, "")

	m.mu.Lock()
	m.connections[name] = made
	m.mu.Unlock()

	m.dispatchConnectionEstablishedEvent(made)

	return made, nil
}

// parseConnectionName splits name into the base connection name and its
// read/write type, stripping a trailing `::read` or `::write`.
func parseConnectionName(name string) (string, string) {
	for _, suffix := range []string{"::read", "::write"} {
		if strings.HasSuffix(name, suffix) {
			return strings.TrimSuffix(name, suffix), strings.TrimPrefix(suffix, "::")
		}
	}
	return name, ""
}

// makeConnection builds a *Connection for name, using a constructor
// registered for that name or its driver, or the factory otherwise.
func (m *DatabaseManager) makeConnection(name string) (*Connection, error) {
	config, err := m.configuration(name)
	if err != nil {
		return nil, err
	}

	m.mu.RLock()
	byName := m.extensions[name]
	driver, _ := config["driver"].(string)
	byDriver := m.extensions[driver]
	m.mu.RUnlock()

	if byName != nil {
		return byName(config, name)
	}
	if byDriver != nil {
		return byDriver(config, name)
	}
	return m.factory.Make(config, name)
}

// configuration returns the driver configuration for name: a dynamic one
// Build registered, or one read from the configured connections.
func (m *DatabaseManager) configuration(name string) (map[string]any, error) {
	m.mu.RLock()
	dynamic := m.dynamicConnectionConfigurations[name]
	m.mu.RUnlock()

	if dynamic != nil {
		return dynamic, nil
	}

	connections, _ := m.config.Get("database.connections").(map[string]any)
	config, found := connections[name].(map[string]any)
	if !found {
		return nil, fmt.Errorf("Database connection [%s] not configured.", name)
	}
	return config, nil
}

// configure finishes a freshly made connection: the read/write pool split,
// the event dispatcher, the transactions manager and the reconnector.
func (m *DatabaseManager) configure(connection *Connection, typ string) *Connection {
	connection = m.setPDOForType(connection, typ).SetReadWriteType(typ)

	m.mu.RLock()
	events := m.events
	transactions := m.transactions
	reconnector := m.reconnector
	m.mu.RUnlock()

	if events != nil {
		connection.SetEventDispatcher(events)
	}
	if transactions != nil {
		connection.SetTransactionManager(transactions)
	}
	connection.SetReconnector(reconnector)

	return connection
}

// setPDOForType points the write pool at the read pool for a "read"
// connection, or the read pool at the write pool for a "write" connection.
func (m *DatabaseManager) setPDOForType(connection *Connection, typ string) *Connection {
	switch typ {
	case "read":
		if pool, err := connection.GetReadPDO(); err == nil {
			connection.SetPDO(pool)
		}
	case "write":
		if pool, err := connection.GetPDO(); err == nil {
			connection.SetReadPDO(pool)
		}
	}
	return connection
}

// dispatchConnectionEstablishedEvent answers the protected method of the same
// name.
func (m *DatabaseManager) dispatchConnectionEstablishedEvent(connection *Connection) {
	m.mu.RLock()
	events := m.events
	m.mu.RUnlock()

	if events == nil {
		return
	}
	events.Dispatch(newConnectionEstablished(connection))
}

// Purge disconnects and forgets the named connection.
func (m *DatabaseManager) Purge(name string) {
	if name == "" {
		name = m.GetDefaultConnection()
	}
	m.Disconnect(name)

	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.connections, name)
}

// Disconnect closes the named connection's pools, if it is open.
func (m *DatabaseManager) Disconnect(name string) {
	if name == "" {
		name = m.GetDefaultConnection()
	}

	m.mu.RLock()
	connection, found := m.connections[name]
	m.mu.RUnlock()

	if found {
		connection.Disconnect()
	}
}

// Reconnect closes and reopens the named connection, or opens it fresh if it
// was not already open.
func (m *DatabaseManager) Reconnect(name string) (*Connection, error) {
	if name == "" {
		name = m.GetDefaultConnection()
	}
	m.Disconnect(name)

	m.mu.RLock()
	_, found := m.connections[name]
	m.mu.RUnlock()

	if !found {
		return m.connection(name)
	}

	refreshed, err := m.refreshPDOConnections(name)
	if err != nil {
		return nil, err
	}
	m.dispatchConnectionEstablishedEvent(refreshed)
	return refreshed, nil
}

// UsingConnection runs callback with a different default connection, and
// puts the old one back afterward.
func (m *DatabaseManager) UsingConnection(name string, callback func() error) error {
	previous := m.GetDefaultConnection()

	m.SetDefaultConnection(name)
	defer m.SetDefaultConnection(previous)

	return callback()
}

// refreshPDOConnections rebuilds the named connection's pools and copies
// them onto the existing *Connection, so a reference a caller is holding
// keeps working after a reconnect.
func (m *DatabaseManager) refreshPDOConnections(name string) (*Connection, error) {
	database, typ := parseConnectionName(name)

	made, err := m.makeConnection(database)
	if err != nil {
		return nil, err
	}
	fresh := m.configure(made, typ)

	m.mu.RLock()
	existing := m.connections[name]
	m.mu.RUnlock()

	if existing == nil {
		return fresh, nil
	}
	return existing.SetPDO(fresh.GetRawPDO()).SetReadPDO(fresh.GetRawReadPDO()), nil
}

// GetDefaultConnection returns the configured default connection name.
func (m *DatabaseManager) GetDefaultConnection() string {
	name, _ := m.config.Get("database.default").(string)
	return name
}

// SetDefaultConnection replaces the configured default connection name.
func (m *DatabaseManager) SetDefaultConnection(name string) {
	m.config.Set("database.default", name)
}

// SupportedDrivers returns the dialects this package can open.
func (m *DatabaseManager) SupportedDrivers() []string { return SupportedDrivers() }

// AvailableDrivers returns the supported drivers that are also linked into
// the binary.
func (m *DatabaseManager) AvailableDrivers() []string { return AvailableDrivers() }

// Extend registers resolver as the connection constructor for name, either a
// connection name or a driver name.
func (m *DatabaseManager) Extend(name string, resolver func(config map[string]any, name string) (*Connection, error)) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.extensions[name] = resolver
}

// ForgetExtension removes the connection constructor registered for name.
func (m *DatabaseManager) ForgetExtension(name string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	delete(m.extensions, name)
}

// GetConnections returns a copy of every connection the manager has made,
// keyed by name.
func (m *DatabaseManager) GetConnections() map[string]*Connection {
	m.mu.RLock()
	defer m.mu.RUnlock()

	out := make(map[string]*Connection, len(m.connections))
	for name, connection := range m.connections {
		out[name] = connection
	}
	return out
}

// ConnectionNames returns the sorted names of the open connections.
//
// Go's maps have no stable order, so a console table built from
// GetConnections would print its rows in a different order on every run
// without the sort.
func (m *DatabaseManager) ConnectionNames() []string {
	m.mu.RLock()
	defer m.mu.RUnlock()

	names := make([]string, 0, len(m.connections))
	for name := range m.connections {
		names = append(names, name)
	}
	return sortedNames(names)
}

// SetReconnector replaces the reconnector set on every connection the
// manager makes.
func (m *DatabaseManager) SetReconnector(reconnector func(*Connection) error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.reconnector = reconnector
}

// SetApplication replaces the configuration the manager reads and writes the
// default connection name through.
func (m *DatabaseManager) SetApplication(config Configuration) *DatabaseManager {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.config = config
	return m
}

// SetEventDispatcher puts a dispatcher on every connection the manager makes.
//
// It is set here once rather than looked up per connection.
func (m *DatabaseManager) SetEventDispatcher(events Dispatcher) *DatabaseManager {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.events = events
	return m
}

// SetTransactionManager puts a transactions manager on every connection the
// manager makes.
func (m *DatabaseManager) SetTransactionManager(manager *DatabaseTransactionsManager) *DatabaseManager {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.transactions = manager
	return m
}

// DatabaseManager satisfies the resolver interface, which is what lets a
// Migrator take either it or a hand-built ConnectionResolver.
var _ ConnectionResolverInterface = (*DatabaseManager)(nil)
