package database

import (
	"fmt"
	"sync"
)

// ConnectionResolverInterface answers a connection by name.
//
// Connection returns an error for a name nobody registered, rather than a nil
// connection that fails four frames away.
type ConnectionResolverInterface interface {
	// Connection returns the named connection. An empty name means the
	// default connection.
	Connection(name string) (ConnectionInterface, error)

	// GetDefaultConnection returns the default connection name.
	GetDefaultConnection() string

	// SetDefaultConnection replaces the default connection name.
	SetDefaultConnection(name string)
}

// ConnectionResolver is a map of name to connection, and the name of the default
// one.
//
// It is the resolver for something that already has its connections -- a test,
// a worker built by hand, the capsule. DatabaseManager is the resolver that
// makes them on demand from configuration.
type ConnectionResolver struct {
	mu sync.RWMutex

	// connections is every connection the resolver knows, keyed by name.
	connections map[string]ConnectionInterface

	// defaultConnection is the name Connection resolves an empty name to.
	defaultConnection string
}

// NewConnectionResolver creates a ConnectionResolver over the given
// connections.
func NewConnectionResolver(connections map[string]ConnectionInterface) *ConnectionResolver {
	r := &ConnectionResolver{connections: map[string]ConnectionInterface{}}
	for name, connection := range connections {
		r.AddConnection(name, connection)
	}
	return r
}

// Connection returns the named connection, or the default when name is
// empty.
func (r *ConnectionResolver) Connection(name string) (ConnectionInterface, error) {
	r.mu.RLock()
	defer r.mu.RUnlock()

	if name == "" {
		name = r.defaultConnection
	}

	connection, found := r.connections[name]
	if !found {
		return nil, fmt.Errorf("Database connection [%s] not configured.", name)
	}
	return connection, nil
}

// AddConnection registers connection under name.
func (r *ConnectionResolver) AddConnection(name string, connection ConnectionInterface) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if r.connections == nil {
		r.connections = map[string]ConnectionInterface{}
	}
	r.connections[name] = connection
}

// HasConnection reports whether a connection is registered under name.
func (r *ConnectionResolver) HasConnection(name string) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, found := r.connections[name]
	return found
}

// GetDefaultConnection returns the default connection name.
func (r *ConnectionResolver) GetDefaultConnection() string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.defaultConnection
}

// SetDefaultConnection replaces the default connection name.
func (r *ConnectionResolver) SetDefaultConnection(name string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.defaultConnection = name
}

// ConnectionResolver satisfies ConnectionResolverInterface.
var _ ConnectionResolverInterface = (*ConnectionResolver)(nil)
