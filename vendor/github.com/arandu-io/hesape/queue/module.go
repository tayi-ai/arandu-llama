package queue

import (
	"context"
	"fmt"
	"time"

	"github.com/arandu-io/hesape/foundation"
	"github.com/arandu-io/hesape/queue/jobs"
	"github.com/arandu-io/hesape/routing"
)

// maxLag is how far behind the queue may fall before the health check fails.
//
// Five minutes on a worker that polls every second means nothing is draining --
// the process is down, every handler is failing, or nobody ever started
// `aru queue:work`. The last one is the common case, and it is the one that
// otherwise shows up as a customer asking why they never got the email.
const maxLag = 5 * time.Minute

// hintLag is when a backlog stops being normal and starts being worth
// mentioning on an error page, well before the health check trips.
const hintLag = time.Minute

// Module brings the jobs table, and reports on the queue.
//
// It registers no routes and runs no worker: the worker is `aru queue:work`, a
// separate process from the same image. What this module owns is the schema
// and the answer to "is anything draining this".
type Module struct {
	queue Queue
	// queues are the ones the health check and the diagnosis look at. Empty
	// means the default queue only.
	queues []string
}

// NewModule returns the module.
//
// Name the queues the application actually uses. A queue nobody watches is a
// queue that can stop draining without anyone noticing, which is the failure
// this module exists to make visible.
func NewModule(q Queue, queues ...string) *Module {
	if len(queues) == 0 {
		queues = []string{jobs.DefaultQueue}
	}
	return &Module{queue: q, queues: queues}
}

var (
	_ foundation.Module     = (*Module)(nil)
	_ foundation.Migratable = (*Module)(nil)
	_ foundation.Health     = (*Module)(nil)
	_ foundation.Diagnostic = (*Module)(nil)
)

// Name is the module identifier.
func (*Module) Name() string { return "queue" }

// Routes registers nothing.
func (*Module) Routes(*routing.Router) {}

// Migrations returns the schema the wired driver needs.
//
// The jobs table belongs to [DatabaseQueue] and is declared there. An
// application wired to a driver that stores jobs elsewhere gets nothing here,
// rather than an empty table it will never read -- which is why this asks the
// driver instead of answering for it.
func (m *Module) Migrations() []foundation.Migration {
	if driver, owns := m.queue.(foundation.Migratable); owns {
		return driver.Migrations()
	}
	return nil
}

// Health fails when a queue stops draining.
func (m *Module) Health(ctx context.Context) error {
	for _, q := range m.queues {
		lag, err := m.lag(ctx, q)
		if err != nil {
			return err
		}
		if lag > maxLag {
			return fmt.Errorf("the oldest job on %s has been waiting %s -- is a worker running?",
				q, lag.Truncate(time.Second))
		}
	}
	return nil
}

// Diagnose reports a backlog and a dead letter queue, on the error page.
//
// Next to the failure somebody is already looking at, which is the moment they
// are most likely to act on it.
func (m *Module) Diagnose(ctx context.Context) []string {
	var out []string

	for _, q := range m.queues {
		lag, err := m.lag(ctx, q)
		if err != nil {
			continue
		}
		if lag > hintLag {
			pending, _ := m.queue.PendingSize(ctx, q)
			out = append(out, fmt.Sprintf(
				"The %s queue has %d job(s) waiting, the oldest for %s. Is `aru queue:work` running?",
				q, pending, lag.Truncate(time.Second)))
		}
	}

	if failed, err := m.queue.Failed(ctx, 5); err == nil && len(failed) > 0 {
		out = append(out, fmt.Sprintf(
			"%d job(s) gave up after repeated failures, the most recent being %s: %s. They stay in the table until retried.",
			len(failed), failed[0].Name, failed[0].LastError))
	}
	return out
}

// lag is how long the oldest waiting job has been waiting.
//
// Zero when nothing is waiting and zero when the oldest job is scheduled for
// the future: a job that is not due yet is not late.
func (m *Module) lag(ctx context.Context, queue string) (time.Duration, error) {
	oldest, err := m.queue.CreationTimeOfOldestPendingJob(ctx, queue)
	if err != nil {
		return 0, err
	}
	if oldest.IsZero() {
		return 0, nil
	}
	if wait := time.Since(oldest); wait > 0 {
		return wait, nil
	}
	return 0, nil
}
