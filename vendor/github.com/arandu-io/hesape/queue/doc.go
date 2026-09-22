// Package queue is work that happens after the response: the [Queue] contract,
// the drivers that ship with the collection, and the [Worker] that drains them.
//
// The surface is Push, PushOn, Later, Bulk, Pop, Size and Clear, and the job
// itself is [github.com/arandu-io/hesape/queue/jobs].
//
// The contract lives in the collection and so do the drivers that need nothing
// installed, for the same reason as database.Repository: Push takes an
// auth.Grant, and the tenant comes from it. Moving that into an optional
// package would make the guarantee optional, and an optional guarantee is not
// one.
//
//	DatabaseQueue    the application's own database (the default)
//	SyncQueue        runs the job at Push, for tests and for a laptop
//	DeferredQueue    runs the job after the response, in this process
//	BackgroundQueue  runs the job in another process
//	FailoverQueue    writes to the first connection that accepts
//	NullQueue        accepts everything and keeps nothing
//	RedisQueue       github.com/arandu-io/hesape/queue/connectors/redis
//
// RedisQueue is a separate module because in Go there is no optional dependency
// and a collection that carried a Redis client would put it in every project's
// go.sum.
//
// # The outbox is the mechanism, not the name
//
// A job pushed inside database.Transaction is committed by the same transaction
// as the row it is about: it exists if and only if the write did. That is the
// outbox guarantee -- the one the events package uses for events -- applied to
// work, and it is the reason [DatabaseQueue] is the default driver rather than
// the fallback one. The relay that drains it is the [Worker].
//
// The name does not change because of it. Calling it Outbox would name the
// mechanism instead of the thing, and hide it from everyone who came looking
// for the driver that keeps jobs in the application's database.
//
// # At-least-once
//
// A handler that cannot run twice safely is a handler with a bug. The process
// can die between doing the work and deleting the job, and no queue anywhere
// solves that.
//
// # Where the rest of it lives
//
//	queue/attributes  the per-job settings: tries, backoff, timeout
//	queue/connectors  opening a connection lazily
//	queue/console     queue:work, queue:retry, queue:pause and the rest
//	queue/events      what the queue announces about itself
//	queue/failed      a dead letter list that outlives the queue
//	queue/jobs        the job itself
//	queue/middleware  what wraps the handling of one job
package queue
