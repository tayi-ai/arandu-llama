---
name: llama-measure
description: Change what a reading is or how it is produced — the loss, the scoring window, the skipped prompt, the adapter scale, the policy label, the quantisation field, or anything that ends up compared against another number. Use when the request is to "change the loss", "make scoring faster", "average the batch", "add a metric", "apply the adapter differently", "store the mean", "skip the prompt", "compare against the reference", or when a change touches score.go, engine.go, adapter.go or llama_wrapper_score.
license: MIT
---

# Every failure here is a plausible number

Nothing in this file crashes. That is the point of it being a skill.

An adapter applied at the wrong magnitude, a mean of means, a loss read off a
KV cache filled under a different adapter — none of those fails anything. They
produce a figure, the figure enters a comparison, and the comparison moves a
decision. The question this package exists to answer is what a lower bit-width
costs, and that question is a difference between two numbers. A constant error
in both survives every test and changes the answer.

So the rule for this file is: **a measurement that cannot be trusted is
refused, never stored.**

## The four that are already right

Do not undo these. Each was a defect once.

### The sum and the count travel apart

`Score` returns `SumNLL` and `Tokens`, never a mean, and `Llama` stores both as
columns. A caller averaging several examples has to weight each by its length.
Averaging the means over-weights short completions, and a batch loss built that
way drifts away from the figure a backpropagating trainer reports for the same
data — which defeats the point, since the two exist to be compared.

If asked for "the average loss of the batch": sum the sums, sum the counts,
divide once.

### Zero, not NaN, when nothing was scored

`Mean()` returns zero when `Tokens` is zero. A NaN propagates silently through
an averaged batch loss and through the difference of two perturbed passes, and
arrives as a poisoned parameter update with nothing left to identify where it
came from. A caller that must tell an empty score from a loss of zero reads
`Tokens`.

`Take` refuses outright when nothing was scored, because at that layer a zero
reads downstream as a perfect prediction.

### The KV cache is cleared before scoring

The cache was filled under whatever adapter was applied when it was written.
Reading it back after the adapter moved scores a mixture of two models.

Prefix reuse is what makes generation fast and it is exactly wrong here. If
somebody asks to "make scoring faster by reusing the cache", the answer is a
separate context for scoring, not a reused cache.

### The policy label refuses to collide

`appliedPolicy` builds the label from adapter digests. An adapter whose file
could not be hashed is labelled `unknown`, and two of those compare equal in
the table afterwards — the one confusion the digest exists to prevent. So the
reading is refused rather than stored under a label that collides.

The quantisation is read from the model, never named by the caller. A caller
that can name the bit-width can name it wrongly, and a reading labelled with
the wrong representation is worse than an unlabelled one.

## The adapter scale is not the trainer's scale

llama.cpp applies `adapter_scale * alpha / rank` when the file declares a
non-zero alpha, and the bare `adapter_scale` when it does not. PEFT applied
`alpha / r`.

So an adapter that lost the key during conversion is applied at a different
magnitude than the trainer applied it, every weight is displaced by one
constant factor, and the loss belongs to a different model — not to a different
quantisation of the same one.

Read `adapter.lora.alpha` at load and refuse when it is absent. An absent key
is an error and not an empty string: a caller that cannot tell "the key says
nothing" from "there is no key" cannot make the check that matters.

**Clear the adapters after a probe.** An engine left perturbed is a model the
next measurement reads without anybody intending it, and the difference is
small enough to pass for noise.

## A capture names its tensor and its rows

`CaptureFinal` returns per-token rows of the tensor named `result_norm` at the
pinned llama.cpp commit: the output of the final RMS norm, learned scale
included, before `result_output` (the lm_head). Not a pooled embedding, not the
residual before the norm, not the logits. `Capture.Tensor`, `Capture.DType`
and `Capture.Version` say so, and the version carries the submodule commit
because a pin bump changes every number while the name stays the same.
`tests/Unit` ties the constant to the gitlink; bumping the pin means re-running
the probe and the bitwise specs, then editing the constant.

**Every row of every window is an output row.** A subset decode runs the last
layer's FFN over a different number of rows and differs by up to 1.9e-6 from
the full rows. That rule is part of the version; the positions choose only
what is copied out. Do not "flag only the requested rows to save memory": it
changes the numbers, and the memory is reserved per row either way.

`SnapshotDigest` folds the quantisation, the policy label **and the adapter
scales**; the reading label does not carry scales, so `+1` and `-1` on one
adapter collide there and must not collide here. The KV cache is cleared first,
as before scoring. A non-finite value in a row is refused, never returned.

## Touching the scoring window

The windows are not an optimisation and cannot be removed for speed.

A flagged position costs `n_vocab` floats. At 248320 tokens that is 993280
bytes, so a 2477-token example flagged whole asks for 2.46 GB of logits on a
card holding 15.32 GiB of weights. And `n_outputs_max` defaults to `n_batch`;
exceeding it is a `GGML_ASSERT` inside `output_reserve`, which **aborts the
process** rather than returning an error a caller could handle.

The window is bounded by `n_batch` and by a memory budget, whichever is
smaller. Raising either means checking both.

## Touching the arithmetic

- The log softmax **subtracts the maximum** before exponentiating. The naive
  form overflows at this vocabulary, and the overflow arrives as `inf`, then as
  `nan`.
- The denominator accumulates in `double`. Over 248320 terms a float loses the
  tail, and this number is compared against a reference measured to sixteen
  digits.
- The per-position results are summed **after the threads join, in a fixed
  order**. Floating point addition is not associative, and a loss that changes
  with thread scheduling is a loss two runs cannot be compared on. Do not sum
  inside the workers, and do not use an atomic accumulator.

## Before you finish

Changing anything here means re-measuring, not re-reading:

1. Score one known example twice and confirm the two figures are identical to
   the last digit.
2. Score it under an adapter at scale `+1`, at `-1`, and cleared. Three
   different numbers, and the cleared one equal to the base.
3. If a chain of steps is involved, run at least three and confirm the loss
   moves. Three identical readings in a row is the signature of a measurement
   that is not reading the adapter at all — it has happened here, for twenty
   steps.

Record the command, the environment and the figures. RULE 8: what was not
measured is declared not measured.
