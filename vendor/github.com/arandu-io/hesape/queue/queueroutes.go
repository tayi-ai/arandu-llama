package queue

import (
	"sort"
	"sync"
)

// Route is where a job goes when nothing at the call site said.
//
// It is a struct rather than a pair, because two strings side by side are two
// strings somebody eventually puts in the wrong order.
type Route struct {
	// Connection is the queue connection, by the name it was registered under.
	// Empty means the default.
	Connection string
	// Queue is the queue on that connection. Empty means jobs.DefaultQueue.
	Queue string
}

// QueueRoutes says which connection and which queue a job name belongs on.
//
// It exists so "reports go on the slow queue" is written once, at boot, instead
// of at every dispatch -- and so that moving them is one line rather than a
// search.
//
//	m.Route("report.monthly", "reports", "")
//	m.Route("invoice.send", "", "redis")
//
// The key is the job name and there is no hierarchy to walk: a name is a name.
// What that costs is a route inherited by a family of jobs; what it buys is
// that the routing of a job is one lookup a person can predict.
type QueueRoutes struct {
	mu     sync.RWMutex
	routes map[string]Route
}

// NewQueueRoutes returns an empty table.
func NewQueueRoutes() *QueueRoutes { return &QueueRoutes{routes: map[string]Route{}} }

// Set registers the route for a job name.
//
// An empty queue or connection means "the default", and setting a name twice
// replaces the first -- the table is built at boot, in one place, and the last
// line wins the way the last assignment does.
func (r *QueueRoutes) Set(name, queue, connection string) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.routes[name] = Route{Connection: connection, Queue: queue}
}

// GetRoute is the route for a job name, and whether there is one.
func (r *QueueRoutes) GetRoute(name string) (Route, bool) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	route, routed := r.routes[name]
	return route, routed
}

// GetConnection is the connection a job name belongs on, or empty for the
// default. It answers getConnection().
func (r *QueueRoutes) GetConnection(name string) string {
	route, _ := r.GetRoute(name)
	return route.Connection
}

// GetQueue is the queue a job name belongs on, or empty for the default. It
// answers getQueue().
func (r *QueueRoutes) GetQueue(name string) string {
	route, _ := r.GetRoute(name)
	return route.Queue
}

// All is every registered route. It answers all().
//
// The copy is not a courtesy: the table is read by whatever dispatches jobs,
// concurrently, and handing out the map would be a data race with a nice name.
func (r *QueueRoutes) All() map[string]Route {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make(map[string]Route, len(r.routes))
	for name, route := range r.routes {
		out[name] = route
	}
	return out
}

// Names is every routed job name, sorted, for `aru queue:monitor` to print.
func (r *QueueRoutes) Names() []string {
	r.mu.RLock()
	defer r.mu.RUnlock()
	out := make([]string, 0, len(r.routes))
	for name := range r.routes {
		out = append(out, name)
	}
	sort.Strings(out)
	return out
}
