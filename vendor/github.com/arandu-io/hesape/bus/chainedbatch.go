package bus

import (
	"context"
	"fmt"

	"github.com/arandu-io/hesape/auth"
)

// ChainedBatchJob is the job name a chained batch is pushed under.
//
// A chain link that is itself a batch has to be a job like any other, because
// the queue only knows how to deliver jobs. This is the name a worker routes to
// ChainedBatch.Handle.
const ChainedBatchJob = "bus.chained-batch"

// ChainedBatch is a batch that is one link of a chain.
//
// It exists because the two shapes compose in only one direction on their own:
// a chain of jobs is easy, and a job that is a whole batch is not, because the
// link after it must not start until the last job of the batch reports. What
// makes it work is that the rest of the chain is moved into the batch's Finally
// callback, so "the batch finished" and "the next link starts" are the same
// event.
//
// It travels as the payload of an ordinary job. Handle is what a worker calls
// when it arrives.
type ChainedBatch struct {
	// Name is what the batch is called.
	Name string `json:"name"`
	// Jobs is the batch's own jobs.
	Jobs []Step `json:"jobs"`
	// Options is the batch's settings and callbacks.
	Options BatchOptions `json:"options"`
	// Queueable carries the rest of the chain, which becomes the batch's
	// Finally callback when Handle runs.
	Queueable
}

// NewChainedBatch turns a described batch into a job that can be a link of a
// chain.
func NewChainedBatch(p *PendingBatch) *ChainedBatch {
	c := &ChainedBatch{Name: p.name, Jobs: p.Jobs(), Options: p.options}
	c.Queue = p.options.Queue
	c.Connection = p.options.Connection
	return c
}

// Step is the ChainedBatch as a job, ready to be added to a chain.
func (c *ChainedBatch) Step() (Step, error) {
	return step(c.Queue, ChainedBatchJob, c)
}

// ToPendingBatch turns the job back into a batch that can be dispatched.
//
// The chain's Catch becomes the batch's Catch: a chain stops at its first
// failure, and a batch that does not allow failures is cancelled by its first
// one, so the two mean the same thing here.
func (c *ChainedBatch) ToPendingBatch() *PendingBatch {
	p := (&PendingBatch{name: c.Name, options: c.Options}).AddStep(c.Jobs...)
	if c.Queue != "" {
		p = p.OnQueue(c.Queue)
	}
	if c.Connection != "" {
		p = p.OnConnection(c.Connection)
	}
	if c.ChainCatch.declared() && !p.options.Catch.declared() {
		p.options.Catch = c.ChainCatch
	}
	return p
}

// Handle dispatches the batch, with the remainder of the chain attached to the
// end of it.
//
// It converts to a pending batch, moves the rest of the chain into the Finally
// callback and dispatches. Finally is the next link itself rather than a
// function that dispatches it, because a queue cannot carry a closure.
func (c *ChainedBatch) Handle(ctx context.Context, g auth.Grant, r BatchRepository, q Queue) (Batch, error) {
	p, err := c.attachRemainderOfChainToEndOfBatch(c.ToPendingBatch())
	if err != nil {
		return Batch{}, err
	}
	return p.Dispatch(ctx, g, r, q)
}

// attachRemainderOfChainToEndOfBatch moves the links that follow this batch
// into its Finally callback.
//
// Finally and not Then, because the chain has to be told either way: a batch
// that failed still has to reach the link that reports it, and that link is the
// chain's Catch.
func (c *ChainedBatch) attachRemainderOfChainToEndOfBatch(p *PendingBatch) (*PendingBatch, error) {
	if len(c.Chained) == 0 {
		return p, nil
	}

	next, rest := c.Chained[0], c.Chained[1:]
	payload, err := wrap(envelope{Bus: formatVersion, Chain: rest, Catch: c.ChainCatch, Body: next.Payload})
	if err != nil {
		return nil, err
	}
	p.options.Finally = Step{Queue: c.queueForChain(next), Name: next.Name, Payload: payload}
	return p, nil
}

// PrepareNestedBatches turns any batch found among a list of chain links into a
// ChainedBatch job.
//
// It is what lets a batch stand in the middle of a chain at all: a batch is not
// a job, and this is where it becomes one.
func PrepareNestedBatches(links []any) ([]Step, error) {
	out := make([]Step, 0, len(links))
	for _, link := range links {
		switch v := link.(type) {
		case nil:
			continue
		case Step:
			if !v.declared() {
				return nil, ErrNoName
			}
			out = append(out, v)
		case *PendingBatch:
			s, err := NewChainedBatch(v).Step()
			if err != nil {
				return nil, err
			}
			out = append(out, s)
		case *ChainedBatch:
			s, err := v.Step()
			if err != nil {
				return nil, err
			}
			out = append(out, s)
		default:
			return nil, fmt.Errorf("bus: %T cannot be a link of a chain: a link is a Step or a batch", link)
		}
	}
	return out, nil
}

// DecodeChainedBatch reads a ChainedBatch back out of a job payload, together
// with the chain it belongs to.
//
// It is what a worker routed to ChainedBatchJob calls before Handle.
func DecodeChainedBatch(payload []byte) (*ChainedBatch, error) {
	var c ChainedBatch
	m, err := Batched(payload, &c)
	if err != nil {
		return nil, err
	}
	if c.Name == "" && len(c.Jobs) == 0 {
		return nil, fmt.Errorf("bus: this payload is not a chained batch")
	}
	// The chain the batch belongs to travels on the envelope, not inside the
	// batch: it is the envelope that Handled rewrites as each link goes by.
	c.Chained = m.Chained
	c.ChainCatch = m.ChainCatch
	if c.ChainQueue == "" {
		c.ChainQueue = m.ChainQueue
	}
	return &c, nil
}
