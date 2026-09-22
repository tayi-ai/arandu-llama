package events

import (
	"context"
	"fmt"
	"time"
)

// Module runs the relay when one is wired, and reports on it.
//
// It registers no routes: the relay is infrastructure, and nothing here is
// reachable from a request. Register it in bootstrap/app.go next to the modules
// that store events.
//
// The outbox table travels with this module rather than being copied into every
// project's migrations. The two migrations that create it are the one piece not
// declared here yet: a migration is a value the migrator consumes, so the type
// belongs to the package that migrates, and that package has not moved into
// this collection.
type Module struct {
	relay *Relay
	// stop cancels the relay loop at shutdown.
	stop context.CancelFunc
	done chan struct{}
}

// NewModule returns the module with no relay, so the table exists, events are
// stored, and nothing publishes them yet.
//
// A module wires nothing: the application does that in bootstrap/app.go.
//
// That is a useful state rather than a broken one. Storing is what cannot be
// recovered later; publishing can start on the day there is something to
// publish to.
func NewModule() *Module { return &Module{} }

// WithRelay returns the module running the relay in this process.
//
// In-process, like the scheduler and for the same reason: a second deployable
// for background work is a second thing to monitor, page on, and forget to
// restart. With more than one replica, give the relay a cache.Locks -- otherwise
// each one publishes every event.
func WithRelay(r *Relay) *Module { return &Module{relay: r} }

// Name is the module identifier.
func (*Module) Name() string { return "events" }

// Start begins the relay loop, and only the process that serves calls it.
//
// It is Start and not Boot, which every command calls: otherwise each worker
// started by `aru queue:work` would run a relay of its own, and so would the
// `aru routes` command. The lock makes that duplicate harmless rather than
// correct.
func (m *Module) Start(ctx context.Context) error {
	if m.relay == nil {
		return nil
	}

	// The loop outlives the boot context, which is cancelled once boot returns.
	// It is stopped by Close, which the kernel calls on shutdown.
	loop, cancel := context.WithCancel(context.WithoutCancel(ctx))
	m.stop = cancel
	m.done = make(chan struct{})

	go func() {
		defer close(m.done)
		_ = m.relay.Run(loop)
	}()
	return nil
}

// Close stops the relay and waits for the pass in flight.
//
// Waiting matters: a pass interrupted between publishing and marking published
// delivers the event again on the next start, and that is the duplicate this
// framework can avoid rather than the one it cannot.
func (m *Module) Close(ctx context.Context) error {
	if m.stop == nil {
		return nil
	}
	m.stop()

	select {
	case <-m.done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// maxLag is how far behind the relay may fall before the health check fails.
//
// A minute is generous for a loop that ticks every second. Past it, something
// is wrong -- the relay is not running, the publisher is refusing everything,
// or another replica holds the lock and died.
const maxLag = time.Minute

// hintLag is when a backlog stops being normal and starts being worth
// mentioning on an error page. It is well below the threshold that fails the
// health check: by the time the health check trips, somebody is already paged.
const hintLag = 30 * time.Second

// Diagnose says what is wrong with event delivery, in a sentence.
//
// It is a hint of the shape "invoice.paid has been waiting four minutes -- is
// the relay running?", and it shows up on the error page, next to the failure
// somebody is already looking at, which is the moment they are most likely to
// act on it.
func (m *Module) Diagnose(ctx context.Context) []string {
	if m.relay == nil {
		return nil
	}
	var out []string

	if lag, err := m.relay.Lag(ctx); err == nil && lag > hintLag {
		out = append(out, fmt.Sprintf(
			"The oldest unpublished event has been waiting %s. Is the relay running, and is the publisher accepting?",
			lag.Truncate(time.Second)))
	}

	if parked, err := m.relay.Parked(ctx, 5); err == nil && len(parked) > 0 {
		out = append(out, fmt.Sprintf(
			"%d event(s) gave up after repeated failures, the most recent being %s: %s. They stay in the outbox until retried.",
			len(parked), parked[0].Name, parked[0].LastError))
	}
	return out
}

// Health fails when the outbox is falling behind.
//
// A relay that stopped looks exactly like a relay with nothing to do, and the
// age of the oldest pending event is what tells them apart. Without this, the
// first sign is a customer asking why they never got the email.
func (m *Module) Health(ctx context.Context) error {
	if m.relay == nil {
		return nil
	}

	lag, err := m.relay.Lag(ctx)
	if err != nil {
		return err
	}
	if lag > maxLag {
		return fmt.Errorf("the oldest unpublished event has been waiting %s -- is the relay running?", lag.Truncate(time.Second))
	}
	return nil
}
