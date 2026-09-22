package queue

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/arandu-io/hesape/auth"
	"github.com/arandu-io/hesape/queue/events"
	"github.com/arandu-io/hesape/queue/jobs"
)

// FailoverQueue writes to the first connection that accepts a job, and reads
// from the first one only.
//
// It is what stands between "the broker is down" and "the checkout returned a
// 500": the push tries each
// connection in order, and a job that could not go on Redis goes in the
// database instead.
//
// The asymmetry is deliberate. Only the push fails over; every read -- Pop,
// Size, the oldest pending job -- goes to the first
// connection. A worker that drained all of them would be a worker whose lease
// and whose ordering mean different things per job, and two workers, one per
// connection, is the arrangement that keeps both simple. The jobs that landed
// on the fallback are drained by a worker pointed at the fallback.
//
// It is not a way to spread load. Every job goes to the first connection that
// works, so as long as the first one is up the others are empty.
type FailoverQueue struct {
	connection
	manager     *QueueManager
	events      Dispatcher
	connections []string

	mu sync.Mutex
	// failing is which connections were down on the last attempt, so a broker
	// that is refusing everything produces one event and not one per job.
	failing map[string]bool
}

// NewFailoverQueue returns the queue over connections, in the order they should
// be tried.
//
// The first is the one that is read from, so it is the one a worker drains and
// the one the health check measures.
func NewFailoverQueue(m *QueueManager, names ...string) *FailoverQueue {
	q := &FailoverQueue{manager: m, connections: names, failing: map[string]bool{}}
	q.SetConnectionName("failover")
	return q
}

// SetEvents gives the queue somewhere to send QueueFailedOver, and returns it
// so the call chains.
func (q *FailoverQueue) SetEvents(d Dispatcher) *FailoverQueue {
	q.events = d
	return q
}

var (
	_ Queue       = (*FailoverQueue)(nil)
	_ jobs.Driver = (*FailoverQueue)(nil)
)

// ErrNoFailoverConnections is returned when a failover queue was built with
// none.
var ErrNoFailoverConnections = errors.New("queue: this failover queue has no connections. Name at least two in NewFailoverQueue")

// primary is the connection everything is read from.
func (q *FailoverQueue) primary() (Queue, error) {
	if len(q.connections) == 0 {
		return nil, ErrNoFailoverConnections
	}
	return q.manager.Connection(q.connections[0])
}

// attempt runs write on each connection in order, and returns when one accepts.
//
// The event goes out once per connection that newly went down, not once per
// job: q.failing is what remembers which ones were already down.
func (q *FailoverQueue) attempt(j *jobs.Job, write func(Queue) error) error {
	if len(q.connections) == 0 {
		return ErrNoFailoverConnections
	}

	var last error
	failed := map[string]bool{}
	for _, name := range q.connections {
		target, err := q.manager.Connection(name)
		if err == nil {
			if err = write(target); err == nil {
				q.recordFailures(failed)
				return nil
			}
		}
		last = err
		failed[name] = true
		q.reportFailure(name, j, err)
	}
	q.recordFailures(failed)

	if last == nil {
		last = errors.New("every failover connection refused the job")
	}
	return fmt.Errorf("queue: no failover connection accepted the job: %w", last)
}

// reportFailure announces a connection that has newly gone down.
func (q *FailoverQueue) reportFailure(name string, j *jobs.Job, cause error) {
	q.mu.Lock()
	alreadyKnown := q.failing[name]
	q.mu.Unlock()
	if alreadyKnown || q.events == nil || j == nil {
		return
	}
	q.events.Dispatch(events.QueueFailedOver{ConnectionName: name, Job: j, Exception: cause})
}

// recordFailures replaces what is known to be down.
func (q *FailoverQueue) recordFailures(failed map[string]bool) {
	q.mu.Lock()
	defer q.mu.Unlock()
	q.failing = failed
}

// Push writes the job to the first connection that accepts it.
func (q *FailoverQueue) Push(ctx context.Context, g auth.Grant, j jobs.Job) error {
	// Authorized once, here, rather than once per connection: the answer cannot
	// differ between them, and a refusal must not look like a connection being
	// down.
	if err := jobs.Authorized(g, j); err != nil {
		return err
	}
	return q.attempt(&j, func(target Queue) error { return target.Push(ctx, g, j) })
}

// PushOn writes the job to a named queue on the first connection that accepts
// it.
func (q *FailoverQueue) PushOn(ctx context.Context, g auth.Grant, queue string, j jobs.Job) error {
	j.Queue = queue
	return q.Push(ctx, g, j)
}

// PushRaw writes a job whose arguments are already encoded. It answers
// pushRaw().
func (q *FailoverQueue) PushRaw(ctx context.Context, g auth.Grant, name string, payload []byte, queue string) error {
	j, err := jobs.New(g, queue, name, nil)
	if err != nil {
		return err
	}
	j.Payload = payload
	return q.Push(ctx, g, j)
}

// Later writes a delayed job to the first connection that accepts it.
func (q *FailoverQueue) Later(ctx context.Context, g auth.Grant, delay time.Duration, j jobs.Job) error {
	if err := jobs.Authorized(g, j); err != nil {
		return err
	}
	return q.attempt(&j, func(target Queue) error { return target.Later(ctx, g, delay, j) })
}

// Bulk writes each job to the first connection that accepts it.
//
// Per job rather than per batch, because a connection that goes down halfway
// through would otherwise leave half the batch on one store and half on
// another, with nothing recording which half.
func (q *FailoverQueue) Bulk(ctx context.Context, g auth.Grant, js []jobs.Job) error {
	for _, j := range js {
		if err := q.Push(ctx, g, j); err != nil {
			return err
		}
	}
	return nil
}

// Pop takes jobs off the first connection. It answers pop().
func (q *FailoverQueue) Pop(ctx context.Context, queue string, n int, lease time.Duration) ([]*jobs.Job, error) {
	target, err := q.primary()
	if err != nil {
		return nil, err
	}
	return target.Pop(ctx, queue, n, lease)
}

// Size is the first connection's. It answers size().
func (q *FailoverQueue) Size(ctx context.Context, queue string) (int, error) {
	target, err := q.primary()
	if err != nil {
		return 0, err
	}
	return target.Size(ctx, queue)
}

// PendingSize is the first connection's. It answers pendingSize().
func (q *FailoverQueue) PendingSize(ctx context.Context, queue string) (int, error) {
	target, err := q.primary()
	if err != nil {
		return 0, err
	}
	return target.PendingSize(ctx, queue)
}

// DelayedSize is the first connection's, when it can answer. It answers
// delayedSize().
func (q *FailoverQueue) DelayedSize(ctx context.Context, queue string) (int, error) {
	target, err := q.primary()
	if err != nil {
		return 0, err
	}
	counter, can := target.(interface {
		DelayedSize(context.Context, string) (int, error)
	})
	if !can {
		return 0, nil
	}
	return counter.DelayedSize(ctx, queue)
}

// ReservedSize is the first connection's, when it can answer. It answers
// reservedSize().
func (q *FailoverQueue) ReservedSize(ctx context.Context, queue string) (int, error) {
	target, err := q.primary()
	if err != nil {
		return 0, err
	}
	counter, can := target.(interface {
		ReservedSize(context.Context, string) (int, error)
	})
	if !can {
		return 0, nil
	}
	return counter.ReservedSize(ctx, queue)
}

// CreationTimeOfOldestPendingJob is the first connection's. It answers
// creationTimeOfOldestPendingJob().
func (q *FailoverQueue) CreationTimeOfOldestPendingJob(ctx context.Context, queue string) (time.Time, error) {
	target, err := q.primary()
	if err != nil {
		return time.Time{}, err
	}
	return target.CreationTimeOfOldestPendingJob(ctx, queue)
}

// Clear empties every connection, and returns how many went in total.
//
// Every one rather than the first: clearing is the operation whose whole point
// is that nothing is left, and a clear that emptied one store while jobs sat on
// the fallback would be the most misleading command in the collection.
func (q *FailoverQueue) Clear(ctx context.Context, queue string) (int, error) {
	total := 0
	for _, name := range q.connections {
		target, err := q.manager.Connection(name)
		if err != nil {
			return total, err
		}
		removed, err := target.Clear(ctx, queue)
		total += removed
		if err != nil {
			return total, err
		}
	}
	return total, nil
}

// Failed lists the parked jobs on every connection, for the same reason Clear
// touches every one: a job that failed on the fallback is still a job somebody
// has to see.
func (q *FailoverQueue) Failed(ctx context.Context, limit int) ([]jobs.Job, error) {
	var out []jobs.Job
	for _, name := range q.connections {
		target, err := q.manager.Connection(name)
		if err != nil {
			return out, err
		}
		failed, err := target.Failed(ctx, limit)
		if err != nil {
			return out, err
		}
		out = append(out, failed...)
	}
	return out, nil
}

// Retry puts a failed job back on whichever connection has it.
func (q *FailoverQueue) Retry(ctx context.Context, uuid string) error {
	var last error
	for _, name := range q.connections {
		target, err := q.manager.Connection(name)
		if err != nil {
			last = err
			continue
		}
		if err := target.Retry(ctx, uuid); err == nil {
			return nil
		} else {
			last = err
		}
	}
	if last == nil {
		return ErrNoFailoverConnections
	}
	return last
}

// ReleaseJob settles the job against the queue it actually came off.
//
// A popped job carries its own driver -- jobs.Popped attached it -- so these
// three are only reached by a job built by hand, and there is nothing to settle.
func (q *FailoverQueue) ReleaseJob(context.Context, *jobs.Job, time.Duration) error { return nil }

// DeleteJob does nothing, for the reason [FailoverQueue.ReleaseJob] gives.
func (q *FailoverQueue) DeleteJob(context.Context, *jobs.Job) error { return nil }

// FailJob does nothing, for the reason [FailoverQueue.ReleaseJob] gives.
func (q *FailoverQueue) FailJob(context.Context, *jobs.Job, error) error { return nil }
