package queue

import "time"

// Config is the queue's settings: which connection the application dispatches
// to, and how each one behaves.
//
// It is declared here, in the package that reads it, and not in a configuration
// package of its own. Nothing looks a value up by key, so the component is
// handed its settings and the compiler checks the field.
type Config struct {
	// Default is the connection a Dispatch goes to when it names none, and
	// empty means "database".
	Default string

	// Connections is every connection the application knows, by name.
	Connections map[string]ConnectionConfig

	// Failed is where a job goes when it has run out of tries.
	Failed FailedConfig
}

// ConnectionConfig is one entry of [Config.Connections]: how one connection
// behaves.
type ConnectionConfig struct {
	// Driver is "sync", "database" or "redis".
	//
	// There is no "beanstalkd", "sqs" or "null", and none of the three is
	// pending: the first two are stores this ecosystem does not speak to, and
	// a driver that silently discards work is a thing to reach for in a test,
	// which "sync" and a recorder already answer without pretending a job ran.
	Driver string

	// Name is the connection's own name, so that a value carries it without the
	// map key having to travel alongside.
	Name string

	// Queue is the queue a job lands on when it names none.
	Queue string

	// RetryAfter is how long a reserved job may be held before another worker
	// may take it.
	//
	// It has to be longer than the longest job actually takes. Shorter, and a
	// slow job is picked up a second time while the first run is still going --
	// which is at-least-once delivery arriving as a duplicate that nobody asked
	// for.
	RetryAfter time.Duration

	// Table is the table a database connection reads, and empty means "jobs".
	Table string

	// Connection is the database or redis connection this queue rides on, empty
	// meaning the application's default.
	Connection string

	// AfterCommit delays the dispatch until the surrounding transaction commits.
	//
	// The failure it guards against is this: a job dispatched inside a
	// transaction can be picked up by a worker before the transaction commits,
	// and it then reads a row that does not exist yet.
	//
	// It is false by default, because the outbox already has the better answer:
	// it writes the job in the same transaction as the change that produced it,
	// so there is no window at all. AfterCommit narrows the window; the outbox
	// closes it.
	AfterCommit bool
}

// FailedConfig is where a job goes when it has run out of tries.
type FailedConfig struct {
	// Driver is "database" or empty for none.
	Driver string

	// Database is the connection the failed table lives on, empty meaning the
	// application's default.
	Database string

	// Table is the table itself, and empty means "failed_jobs".
	Table string
}
