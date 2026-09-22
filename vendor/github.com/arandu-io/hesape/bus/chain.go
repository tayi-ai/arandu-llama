package bus

import (
	"context"
	"errors"
	"fmt"

	"github.com/arandu-io/hesape/auth"
)

// ErrEmptyChain is returned when a chain is dispatched with no links.
var ErrEmptyChain = errors.New("bus: a chain with no jobs is not a chain")

// Chain runs jobs one after another, each starting only if the one before it
// succeeded.
//
// It is the other half of what a batch is not. A batch is N jobs that do not
// care about each other and one question at the end; a chain is N jobs where
// link two assumes link one happened -- export the rows, then compress the
// file, then email the link. Compressing a file that was never written is the
// failure a chain exists to prevent.
//
// Nothing about a chain is stored. The links that have not run travel inside
// the payload of the link that is running, so the chain is exactly as durable
// as the queue and an abandoned one leaves no row behind. The cost is that a
// chain of a thousand links carries a thousand payloads in its first job, which
// is the honest signal that a thousand independent things want a batch.
type Chain struct {
	queue string
	jobs  []Step
	catch Step
	err   error
}

// NewChain starts describing a chain.
func NewChain() *Chain { return &Chain{} }

// Add appends a link. Order is the order of the calls.
func (c *Chain) Add(name string, payload any) *Chain {
	s, err := step("", name, payload)
	if err != nil {
		c.fail(err)
		return c
	}
	c.jobs = append(c.jobs, s)
	return c
}

// AddStep appends links that are already built.
//
// It is Add for the caller that has Steps in hand -- Dispatcher.Chain does --
// and it is the only place in this package where a link joins a chain without
// being serialized again.
func (c *Chain) AddStep(jobs ...Step) *Chain {
	for _, s := range jobs {
		if !s.declared() {
			c.fail(ErrNoName)
			return c
		}
		c.jobs = append(c.jobs, s)
	}
	return c
}

// Jobs is the links described so far.
func (c *Chain) Jobs() []Step { return append([]Step(nil), c.jobs...) }

// Catch names the job dispatched when a link fails and the chain stops.
//
// There is no Then: the last link is the Then. A chain that wants a step after
// the work adds it as the last link, which is one concept instead of two.
func (c *Chain) Catch(name string, payload any) *Chain {
	s, err := step("", name, payload)
	if err != nil {
		c.fail(err)
		return c
	}
	c.catch = s
	return c
}

// OnQueue is which queue the links go on.
func (c *Chain) OnQueue(queue string) *Chain {
	c.queue = queue
	return c
}

// Dispatch pushes the first link, carrying the rest.
//
// Only the first: the second is pushed by Handled when the first reports
// success, and so on. There is no moment at which two links of a chain are in
// the queue at once, which is the property that makes the order real rather
// than hoped for.
func (c *Chain) Dispatch(ctx context.Context, g auth.Grant, q Queue) error {
	if c.err != nil {
		return c.err
	}
	if len(c.jobs) == 0 {
		return ErrEmptyChain
	}
	if q == nil {
		return errors.New("bus: dispatching a chain needs a queue")
	}
	if _, err := tenantOf(g); err != nil {
		return err
	}

	// The queue is resolved here, into the steps that travel in the envelope,
	// and not when each link is pushed. Handled pushes link N+1 from the
	// envelope alone -- it never sees this Chain -- so a link whose queue was
	// left to be filled in later would be filled in by nobody and land on the
	// driver's default. The first link would go to "imports" and the rest to
	// "default", which is the kind of thing that only shows up under load.
	links := make([]Step, len(c.jobs))
	for i, s := range c.jobs {
		links[i] = c.queueFor(s)
	}
	first, rest := links[0], links[1:]

	payload, err := wrap(envelope{Bus: formatVersion, Chain: rest, Catch: c.queueFor(c.catch), Body: first.Payload})
	if err != nil {
		return err
	}
	if err := q.Push(ctx, g, first.Queue, first.Name, payload); err != nil {
		return fmt.Errorf("bus: pushing the first link %s: %w", first.Name, err)
	}
	return nil
}

// queueFor returns the step with its queue settled: its own, or the chain's.
func (c *Chain) queueFor(s Step) Step {
	if s.Queue == "" {
		s.Queue = c.queue
	}
	return s
}

// fail keeps the first error, for the reason PendingBatch.fail gives.
func (c *Chain) fail(err error) {
	if c.err == nil {
		c.err = err
	}
}
