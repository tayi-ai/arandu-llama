package llama

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/arandu-io/framework/security"
)

// Taking a reading with the engine this package already is.
//
// The wrapper that used to sit here declared an Engine interface and held none,
// so that an application listing readings would link no llama.cpp. That
// separation is gone: this package is both halves now, and an application that
// registers this module links the engine whether it scores anything or not.
// The manifest says so.
//
// What is kept from that design is the shape of a reading. The sum and the count
// travel apart and never as a mean, because a caller combining readings has to
// weight each by its length; and the representation the weights are in is read
// from the model rather than named by the caller, because a caller that can name
// the bit-width can name it wrongly, and a reading labelled with the wrong
// representation is worse than an unlabelled one -- it enters a comparison and
// moves the answer.

// ErrNoContext is returned when a reading is asked for without one.
//
// An error and not a zero reading: a loss of zero reads downstream as a perfect
// prediction rather than as an absent measurement, and a caller that cannot tell
// those apart will eventually act on the wrong one.
var ErrNoContext = errors.New("llama: a reading needs a context; there is no model to score against")

// Take scores a token sequence and records what it read.
//
// It is a function rather than a method on the service because it reaches no
// Model of its own: the scoring happens in the context and the storing happens
// in Create, which authorises for itself. Every exported method of the service is
// audited for reaching the Model after security.Authorize decided it may, and a
// method here would have to be excused from that audit -- an audit with an
// exception is an audit nobody trusts.
//
// The reading is stored before it is returned, and that ordering is the point. A
// twenty-step run once reported the same loss at every step because the
// measurement was reading a model with no adapter applied; telling that apart
// from a run that had converged needed the readings side by side, and a run that
// keeps its numbers in a variable has nothing to put side by side.
func Take(ctx context.Context, s *LlamaService, actor security.Subject, c *Context, name, subject string, tokens []int32, skip int) (*Llama, error) {
	if c == nil {
		return nil, ErrNoContext
	}
	quantisation, err := c.model.Describe()
	if err != nil {
		return nil, err
	}
	started := time.Now()
	score, err := c.Score(tokens, skip)
	if err != nil {
		return nil, err
	}
	if score.Tokens == 0 {
		return nil, errors.New("llama: no position was scored; a mean over none reads as a perfect prediction")
	}
	policy, err := c.appliedPolicy()
	if err != nil {
		return nil, err
	}
	return s.Create(ctx, actor, CreateRequest{
		Name:         name,
		Subject:      subject,
		Policy:       policy,
		Quantisation: quantisation,
		Loss:         score.SumNLL,
		Tokens:       score.Tokens,
		Milliseconds: time.Since(started).Milliseconds(),
	})
}

// appliedPolicy names the adapters currently applied to this context.
//
// A reading that cannot say which policy produced it cannot be compared with any
// other, and comparison is the only thing a reading is for. An empty string is
// refused rather than stored: "no adapter" is a real answer and has to be said
// out loud, because the base model and an unrecorded adapter look identical in a
// table afterwards.
func (c *Context) appliedPolicy() (string, error) {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.appliedPolicyLocked()
}

// appliedPolicyLocked is appliedPolicy for a caller that already holds c.mu, in
// either mode. The mutex is not reentrant, and CaptureFinal labels the policy
// under the write lock it computes the rows under -- so that the label names
// the adapters the rows were actually produced with, and not whatever a
// concurrent SetAdapters put on the context a moment later.
func (c *Context) appliedPolicyLocked() (string, error) {
	if c.closed {
		return "", errors.New("llama: context is closed")
	}
	if len(c.applied) == 0 {
		return "base", nil
	}
	names := make([]string, 0, len(c.applied))
	for _, adapter := range c.applied {
		digest := adapter.Digest()
		// An adapter whose file could not be hashed is labelled "unknown", and
		// two different adapters labelled "unknown" compare equal -- which is
		// precisely the confusion the digest exists to prevent. A reading that
		// cannot name its policy is refused rather than stored under a label that
		// collides, because the collision is invisible in the table afterwards.
		if digest == "" || digest == unknownDigest {
			return "", fmt.Errorf("llama: the adapter at %s has no digest; a reading labelled %q cannot be told from any other", adapter.Path(), unknownDigest)
		}
		names = append(names, digest)
	}
	return strings.Join(names, ","), nil
}
