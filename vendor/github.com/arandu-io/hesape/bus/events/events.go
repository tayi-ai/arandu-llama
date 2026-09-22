// Package events names the three things that happen to a batch.
//
// They are names and not types, for the reason the auth events are: an event
// crosses processes and versions of the binary through the outbox, so what
// travels is a name and a JSON body, and a Go struct on the publishing side
// would be a struct the subscribing side has to have compiled in.
//
// A name is a contract. Change one and the listeners stop hearing, silently,
// which is why they are constants here rather than string literals at the four
// places that would otherwise spell them.
//
// The payload of each is the batch's identity and its counters, not the whole
// batch: an outbox row is JSON, and the callbacks a batch holds are jobs rather
// than values a listener could run.
package events

import (
	"time"

	"github.com/arandu-io/hesape/events"
)

// The three names. Dotted and lowercase, like every other event in the
// collection.
//
// Cancelled is spelled with two Ls, which is the spelling the collection uses
// everywhere.
const (
	// BatchDispatched is published when a batch's jobs have been queued.
	BatchDispatched = "batch.dispatched"
	// BatchCancelled is published when a batch is cancelled.
	BatchCancelled = "batch.cancelled"
	// BatchFinished is published when a batch stops, successfully or not.
	BatchFinished = "batch.finished"
)

// Aggregate is what these events happen to, for the outbox.
const Aggregate = "batch"

// Payload is what a consumer of the three events receives.
//
// It carries the batch's identity and its counters, never the arguments of the
// jobs in it: a batch of ten thousand invoice rows would put ten thousand
// customer records in a row that everything downstream reads.
type Payload struct {
	// ID is the batch.
	ID string `json:"id"`
	// Name is what it is called on a dashboard.
	Name string `json:"name"`
	// Tenant is the tenant off the Grant that dispatched it.
	Tenant string `json:"tenant"`
	// TotalJobs, PendingJobs and FailedJobs are the counters as they stood
	// when the event was recorded.
	TotalJobs   int `json:"total_jobs"`
	PendingJobs int `json:"pending_jobs"`
	FailedJobs  int `json:"failed_jobs"`
}

// NewBatchDispatched returns the event recorded once a batch has been stored.
func NewBatchDispatched(p Payload) events.Event { return newEvent(BatchDispatched, p) }

// NewBatchCancelled returns the event recorded when a batch is cancelled.
func NewBatchCancelled(p Payload) events.Event { return newEvent(BatchCancelled, p) }

// NewBatchFinished returns the event recorded when a batch stops.
func NewBatchFinished(p Payload) events.Event { return newEvent(BatchFinished, p) }

func newEvent(name string, p Payload) events.Event {
	return events.Event{
		Name:        name,
		Aggregate:   Aggregate,
		AggregateID: p.ID,
		Payload:     p,
		OccurredAt:  time.Now().UTC(),
	}
}
