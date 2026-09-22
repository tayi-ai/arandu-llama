package bus

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"

	"github.com/arandu-io/hesape/auth"
)

// Queue is the slice of the queue contract this package uses: put a job on a
// named queue, under a Grant, with its arguments already serialized.
//
// It is declared here rather than imported because bus sits above the queue and
// needs four of its words, not all of them. Declaring the narrow interface at
// the consumer is how Go keeps a layer from depending on everything the layer
// below happens to export -- and it is what lets a test drive a batch with
// twenty lines of recorder instead of a running worker.
//
// hesape/queue satisfies it. Nothing in this package knows how a job is stored,
// reserved or retried, and nothing in it should.
type Queue interface {
	Push(ctx context.Context, g auth.Grant, queue, name string, payload []byte) error
}

// Step is one job: where it goes, what it is called, and its arguments.
//
// It is the unit a batch counts, a chain advances through and a callback is
// made of, because those three are the same thing seen from three angles. A
// Step with an empty Name is the absence of a step, which is how "this batch
// has no Catch" is spelled.
type Step struct {
	// Queue is which queue it goes on. Empty means the batch's queue, and an
	// empty batch queue means whatever the driver calls its default.
	Queue string `json:"queue,omitempty"`
	// Name routes it to a handler: "invoice.import", "report.monthly".
	Name string `json:"name"`
	// Payload is the arguments as JSON, already encoded.
	Payload json.RawMessage `json:"payload,omitempty"`
}

// declared reports whether the step names a job at all.
func (s Step) declared() bool { return s.Name != "" }

// ErrNoName is returned when a job is added with no name to route it by.
var ErrNoName = errors.New("bus: a job with no name cannot be routed to a handler")

// ErrEmptyBatch is returned when a batch is dispatched with no jobs in it.
//
// An empty batch is refused rather than finished immediately, because the two
// are indistinguishable at the callback and only one of them was intended: a
// batch built from an empty result set would fire Then, and Then is usually the
// step that reports success.
var ErrEmptyBatch = errors.New("bus: a batch with no jobs is not a batch")

// encode serializes the arguments of a step.
func encode(name string, payload any) (json.RawMessage, error) {
	if payload == nil {
		return nil, nil
	}
	raw, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("bus: serializing the payload of %s: %w", name, err)
	}
	return raw, nil
}

// step builds a Step, refusing the one that could not be routed.
func step(queue, name string, payload any) (Step, error) {
	if name == "" {
		return Step{}, ErrNoName
	}
	raw, err := encode(name, payload)
	if err != nil {
		return Step{}, err
	}
	return Step{Queue: queue, Name: name, Payload: raw}, nil
}
