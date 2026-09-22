package queue

import (
	"context"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/queue/jobs"
)

// connection is the one thing every driver shares: the name it was registered
// under.
//
// It is an unexported embedded struct rather than an exported base type,
// because Go promotes the methods either way and an exported one would be a
// second thing called Queue in a package that already has the contract by that
// name. Every driver in this package embeds it, which is what makes
// GetConnectionName answerable on all of them without six copies of two lines.
//
// Nothing else is on it. The application constructs its queues in
// bootstrap/app.go and hands them to [QueueManager], so there is no
// configuration for a driver to stash and nothing to resolve later.
type connection struct {
	name string
}

// GetConnectionName is the name this queue was registered under.
//
// It is empty on a queue built directly and never handed to a [QueueManager],
// which is what a test does.
func (c *connection) GetConnectionName() string { return c.name }

// SetConnectionName names the connection this queue answers to.
//
// It returns nothing rather than the queue, because an embedded struct cannot
// return the type that embeds it.
func (c *connection) SetConnectionName(name string) { c.name = name }

// LaterOn adds a job to a named queue, eligible after delay.
//
// It is a package function rather than a method on the contract because it
// cannot be anything but Later with the queue overwritten. PushOn is on the
// contract instead, because a driver can do that one in a single round trip.
//
//	queue.LaterOn(ctx, q, g, "reports", time.Hour, j)
func LaterOn(ctx context.Context, q Queue, g auth.Grant, name string, delay time.Duration, j jobs.Job) error {
	j.Queue = name
	return q.Later(ctx, g, delay, j)
}
