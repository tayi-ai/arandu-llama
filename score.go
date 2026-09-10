package llama

/*
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"

import "fmt"

// Score is the loss a model assigns to a token sequence it did not produce.
//
// The summed negative log likelihood and the number of scored positions are
// reported separately rather than as a mean, because a caller averaging several
// examples has to weight each one by its length. Averaging the means instead
// over-weights short completions, and a batch loss built that way drifts away
// from the figure a backpropagating trainer reports for the same data - which
// defeats the point of measuring it here, since the two numbers exist to be
// compared against each other.
//
// Divide with Mean for the per-token figure of a single scored sequence; sum
// SumNLL and Tokens across examples and divide once for a batch.
type Score struct {
	SumNLL float64 // Summed negative log likelihood over the scored positions
	Tokens int     // Number of positions scored, after the skipped prefix
}

// Mean returns the per-token negative log likelihood, or zero when nothing was
// scored.
//
// Zero rather than NaN when Tokens is zero: a NaN propagates silently through an
// averaged batch loss and through the difference of two perturbed forward passes
// in a zeroth-order step, so it arrives as a poisoned parameter update with
// nothing left to identify where it came from. A caller that has to distinguish
// an empty score from a loss of zero reads Tokens.
func (s Score) Mean() float64 {
	if s.Tokens == 0 {
		return 0
	}
	return s.SumNLL / float64(s.Tokens)
}

// Score computes the negative log likelihood the model assigns to a token
// sequence by teacher forcing.
//
// The log probability read at position i is the one the model gave to the token
// that actually followed at i+1. Position zero is never scored: nothing preceded
// it to predict it.
//
// The sequence is decoded in windows, with the KV cache carried across them, so
// it remains one pass over the tokens rather than several. The windows are not an
// optimisation: a flagged position costs n_vocab floats of output, which is
// 993280 bytes for a 248320-token vocabulary, and llama.cpp asserts rather than
// erroring when a decode flags more positions than the context allows.
//
// The skip argument drops that many leading tokens, so the loss covers a
// completion rather than the prompt that led to it - a loss that included the
// prompt would reward a policy for predicting text it was handed. A sequence of
// n tokens scored with skip k therefore yields n - max(k, 1) positions, which is
// the value returned in Score.Tokens.
//
// Scoring clears the KV cache before decoding. The cache was filled under
// whatever adapter was applied when it was written, and reading it back after
// the adapter moved would score a mixture of two models. The consequence for the
// caller is that scoring must not be interleaved with prefix-cached generation
// on the same context: the next Generate call reprocesses its whole prompt, and
// a cached prefix measured before a Score call is gone after it. Use a separate
// context for scoring when generation performance matters.
//
// Fewer than two tokens, or a skip that leaves nothing to score, are refused
// before crossing into C.
//
// Thread safety: Score takes the context's write lock, so it serialises against
// generation on the same context. Use separate contexts for concurrent work.
//
// See also: ScoreText to score a completion given a prompt, Score.Mean for the
// per-token figure.
//
// Examples:
//
//	// Loss over an entire sequence
//	tokens, _ := ctx.Tokenize("The capital of France is Paris")
//	score, err := ctx.Score(tokens, 0)
//	fmt.Printf("%f over %d positions\n", score.Mean(), score.Tokens)
//
//	// Loss over the completion alone, with the prompt skipped
//	prompt, _ := ctx.Tokenize("2 + 2 =")
//	full, _ := ctx.Tokenize("2 + 2 = 4")
//	score, err := ctx.Score(full, len(prompt))
func (c *Context) Score(tokens []int32, skip int) (Score, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return Score{}, fmt.Errorf("context is closed")
	}

	// Check if model is closed
	if c.model == nil {
		return Score{}, fmt.Errorf("model is closed")
	}
	c.model.mu.RLock()
	modelClosed := c.model.closed
	c.model.mu.RUnlock()
	if modelClosed {
		return Score{}, fmt.Errorf("model is closed")
	}

	// One token conditions, the next is predicted; a single token has neither
	if len(tokens) < 2 {
		return Score{}, fmt.Errorf("at least two tokens required to score, got %d", len(tokens))
	}

	// A skip that consumes the sequence would return a mean over nothing, and
	// zero positions scored reads the same as a perfect prediction downstream
	if skip < 0 || skip >= len(tokens) {
		return Score{}, fmt.Errorf("invalid skip: %d for %d tokens", skip, len(tokens))
	}

	// Convert tokens to C array
	cTokens := make([]C.int, len(tokens))
	for i, token := range tokens {
		cTokens[i] = C.int(token)
	}

	var cSum C.double
	var cCount C.int

	result := C.llama_wrapper_score(
		c.contextPtr,
		&cTokens[0],
		C.int(len(tokens)),
		C.int(skip),
		&cSum,
		&cCount,
	)

	if result != 0 {
		return Score{}, fmt.Errorf("scoring failed: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	return Score{
		SumNLL: float64(cSum),
		Tokens: int(cCount),
	}, nil
}

// ScoreText computes the negative log likelihood of a completion given a prompt.
//
// The prompt and the completion are tokenised jointly, and the token count of
// the prompt alone becomes the number of leading positions skipped, so the loss
// covers the completion only. Tokenising the two separately and concatenating
// the results would be wrong: a tokeniser merges across the join, so the
// concatenated sequence is not the one the model would see for that text, and
// the number would not match what a trainer measured on the same example.
//
// The boundary is still measured on the prompt in isolation. Where a merge at
// the join makes the prompt tokens shorter or longer than their prefix in the
// joint sequence, the first scored position moves with it by a token; where the
// merge leaves no room at all, the call is refused rather than scoring a span
// the caller did not ask for.
//
// An empty completion is refused for the same reason, surfacing as the invalid
// skip error from Score.
//
// Thread safety: as Score - the context is locked for the duration of the
// forward pass, and scoring clears the KV cache.
//
// See also: Score for pre-tokenised input and the full description of the KV
// cache behaviour.
//
// Example:
//
//	score, err := ctx.ScoreText("The capital of France is", " Paris")
//	fmt.Printf("mean nll over the completion: %f\n", score.Mean())
func (c *Context) ScoreText(prompt, completion string) (Score, error) {
	// Tokenise before Score takes the write lock: the mutex is not reentrant, and
	// tokenising inside the scored region would deadlock the context permanently
	promptTokens, err := c.Tokenize(prompt)
	if err != nil {
		return Score{}, fmt.Errorf("failed to tokenise prompt: %w", err)
	}

	tokens, err := c.Tokenize(prompt + completion)
	if err != nil {
		return Score{}, fmt.Errorf("failed to tokenise prompt and completion: %w", err)
	}

	return c.Score(tokens, len(promptTokens))
}
