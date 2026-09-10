# Adapters and scoring

The two capabilities this package exists for. An inference binding has no
reason to carry either: it scores what the model wrote, and never changes a
weight after load.

Zeroth-order optimisation needs the opposite of both. It scores a completion
somebody else wrote, and it moves the adapter between one forward pass and the
next whilst the base weights stay on the cards.

## An adapter over a quantised base

The base is loaded once, in whatever representation it was quantised to. The
adapter is a separate GGUF, loaded against that model:

```go
model, err := llama.LoadModel("qwen3.8-27b-q4_k_m.gguf", llama.WithGPULayers(-1))
defer model.Close()

adapter, err := model.LoadAdapter("rank4.gguf")
defer adapter.Close()
```

Loading reads the adapter against the model it will be applied to, so a
mismatch in architecture or tensor shape is refused here rather than surfacing
later as a plausible-looking loss.

Applying is per context, and the call replaces the whole set:

```go
err = ctx.SetAdapters([]*llama.Adapter{adapter}, []float32{1.0})   // positive half
err = ctx.SetAdapters([]*llama.Adapter{adapter}, []float32{-1.0})  // negative half
err = ctx.ClearAdapters()                                          // back to base
```

The scale multiplies the adapter's contribution, so a probe moves the policy by
re-applying the same handle with a different number rather than by reloading
tensors between passes. Measured at 0.4 ms.

**Clear after a probe.** An engine left perturbed is a model the next
measurement reads without anybody intending it, and the difference is small
enough to pass for noise.

### The scale is not what you think it is

llama.cpp applies `adapter_scale * alpha / rank` when the file declares a
non-zero alpha, and the bare `adapter_scale` when it does not. PEFT applied
`alpha / r`. So an adapter that lost the key during conversion is applied at a
different magnitude than the trainer applied it, every weight is displaced by
one constant factor, and the loss that comes out belongs to a different model —
not to a different quantisation of the same one.

That failure reports nothing. It produces a plausible number, and a plausible
number is what gets compared against a measured reference and believed. Check
it at load:

```go
alpha, err := adapter.Meta("adapter.lora.alpha")
if err != nil {
	return fmt.Errorf("the adapter declares no alpha, so its scale is not the trainer's: %w", err)
}
```

An absent key is an error rather than an empty string, because a caller that
cannot tell "the key says nothing" from "there is no key" cannot make the check
that matters.

### Ordering, and what is not thread-safe

`Adapter` is safe for concurrent use. The context it is applied to is not:
every decode reads the adapter's tensors, so removing or freeing one whilst
work is in flight corrupts whatever that work returns.

`Adapter` carries no finaliser, unlike `Model` and `Context`. A context holds
the raw handle rather than a Go reference this package can see, so a finaliser
would be free to run whilst a context still points at the tensors. An adapter
never closed leaks memory; an adapter freed whilst applied is read after the
free, and that is the worse of the two.

Close the adapter before the model. `llama_model_free` destroys every adapter
loaded against the model, so a model closed first has already freed the handle.

## Teacher-forced scoring

The negative log likelihood the model assigns to a sequence it did not produce.
Every position is decoded with its logits requested, and the log probability
read at position *i* is the one the model gave to the token that actually
followed at *i+1*.

```go
score, err := ctx.ScoreText("2 + 2 =", " 4")
fmt.Printf("%f over %d positions\n", score.Mean(), score.Tokens)
```

Or pre-tokenised, with the number of leading positions to drop:

```go
score, err := ctx.Score(tokens, len(promptTokens))
```

### Why the sum and the count travel apart

`Score` returns `SumNLL` and `Tokens`, never a mean. A caller averaging several
examples has to weight each by its length; averaging the means over-weights
short completions, and a batch loss built that way drifts away from the figure
a backpropagating trainer reports for the same data — which defeats the point,
since the two numbers exist to be compared against each other.

`Mean()` is there for a single sequence, and returns zero rather than NaN when
nothing was scored. A NaN propagates silently through an averaged batch loss and
through the difference of two perturbed passes, and arrives as a poisoned
parameter update with nothing left to identify where it came from.

### Skipping the prompt

`skip` drops that many leading tokens, so the loss covers the completion alone.
A loss that included the prompt would reward a policy for predicting text it
was handed.

`ScoreText` tokenises the prompt and the joint string separately and uses the
prompt's length as the skip. Tokenising the two halves and concatenating would
be wrong: a tokeniser merges across the join, so the concatenated sequence is
not the one the model would see, and the number would not match what a trainer
measured on the same example.

### The KV cache

Scoring **clears it** before decoding. The cache was filled under whatever
adapter was applied when it was written, and reading it back after the adapter
moved would score a mixture of two models.

The consequence for a caller: do not interleave scoring with prefix-cached
generation on the same context. The next `Generate` reprocesses its whole
prompt, and a cached prefix measured before a `Score` call is gone after it.
Use a separate context for scoring when generation performance matters.

### Windows, which are not an optimisation

A flagged position costs `n_vocab` floats of output. On a 248320-token
vocabulary that is 993280 bytes, so a 2477-token example flagged whole asks for
**2.46 GB of logits** on a card holding 15.32 GiB of weights.

The second reason is harder to recover from. `llama_context_params` carries
`n_outputs_max`, which defaults to `n_batch`, and exceeding it is a
`GGML_ASSERT` inside `output_reserve` — the process aborts rather than returning
an error a caller could handle.

So the sequence is decoded in windows bounded by both, with the KV cache
carried across them. It remains one pass over the tokens, split into pieces.

### The arithmetic, and why it is in double

The log softmax subtracts the maximum before exponentiating. The naive form
overflows on a vocabulary this size, and the overflow arrives as `inf`, then as
`nan`, and a NaN loss in a zeroth-order step poisons every parameter it touches
without failing anything.

The denominator accumulates in `double` where the reference uses `float`: over
248320 terms the float loses the tail, and this number is compared against a
reference measured to sixteen digits.

The per-position work is threaded — one position is a pass to find the maximum
and a pass of 248320 exponentials, so a 2477-token example is on the order of a
billion operations, which serial costs about what the forward pass costs and
doubles the price of every probe. The results are summed **after the threads
join, in a fixed order**: floating point addition is not associative, and a
loss that changes with thread scheduling is a loss two runs cannot be compared
on.

## Capturing the final representation

A loss says the two models disagree. A representation says where. The
objective that trains a quantised base plus an adapter to match a
high-precision reference compares, token by token, the hidden state after the
final normalisation and before the output projection — so the quantity is a
matrix, one row per requested position, and never a pooled embedding of the
sequence.

```go
capture, err := ctx.CaptureFinal(tokens, llama.PositionsOf(mask))
nEmbd, nPositions := capture.Shape()      // (2048, len(positions)) for SmolLM3-3B
row := capture.Row(k)                     // the representation at capture.Positions[k]
```

### Which tensor, which rows

At the pinned llama.cpp commit the tensor is the graph node named
`result_norm`: the output of the final RMS norm including its learned scale,
the immediate input of `result_output`, the lm_head. `Capture.Tensor` says so,
`Capture.DType` is `f32`, and `Capture.Version` carries the commit, because
the pin moving is the one event that changes every number while the name
stays the same. A `tests/Unit` check ties the constant to the gitlink.

The rows are read through llama.cpp's own copy of that tensor — the embeddings
flag is switched on for the decode and restored afterwards — and were measured
bitwise equal to the tensor a scheduler callback observes, 14 of 14 rows. A
callback would have had to be installed at context creation and would
synchronise every split of every decode on that context for the rest of its
life.

**Every row of every decode window is computed as an output row.** Embeddings
mode overrides a partial selection anyway, and the choice moves the numbers: a
decode that flags a subset of rows runs the last layer's FFN over a different
number of rows and differs from the full-window rows by up to 1.9e-6. Which
rows are computed is therefore part of the capture version; the positions only
choose which rows are copied out.

### Identity, so two captures cannot be confused

`SnapshotDigest` names what produced the rows: the quantisation read from the
model, the policy label the readings carry, **and the scale of each adapter**,
folded with the version. The reading label alone does not carry scales, so a
`+1` and a `-1` probe on one adapter share a policy string; their captures do
not share a snapshot digest. `TokenDigest` names what was captured: the tokens
and the positions, length-prefixed. Cache references by both.

The model's identity is its description (`smollm3 3B Q4_K - Medium`), not a
file digest: two files quantised the same way carry the same snapshot digest.
A caller comparing against a checkpoint keys its cache by the checkpoint
digest it already records.

### Cost, cache and windows

A capture costs `len(positions) × Width × 4` bytes of representation:
16,777,216 bytes for SmolLM3-3B (`Width` 2048) at 2048 positions. Inside
llama.cpp, transiently per window, it costs `window × (n_vocab + Width) × 4`
bytes — 136,037,376 bytes at the 261-row window the 128 MiB logit cap gives
SmolLM3 — because logits are reserved per output row in embeddings mode too.
The windows are the scoring windows, bounded by `n_batch` and the logit cap for
the same two reasons.

The KV cache is **cleared** first, as before scoring: a cache filled under
another adapter would capture a mixture of two models. A non-finite value
anywhere in a row is refused rather than returned — a NaN target enters the
objective and poisons the update without failing anything. A context that
pools its embeddings is refused: the copied tensor would be a sequence
embedding.

Measured on CPU: two captures identical to the last digit, on the same context
and on a fresh one; a changed token leaves every row before it bitwise
unchanged and moves every row from it on; windows of 8 equal a single window.
Not measured: the effect of an adapter on the rows (no LoRA file existed on the
machine), and bit-stability of the copy on Metal or CUDA.

## Storing a reading

`Take` scores and records in one call:

```go
reading, err := llama.Take(request, service, actor, ctx,
	"step-14 positive", "example-3", tokens, skip)
```

It stores before it returns, and that ordering is the point. A twenty-step run
once reported the same loss at every step because the measurement was reading a
model with no adapter applied, and telling that apart from a run that had
converged needed the readings side by side.

The policy label comes from the adapters applied to the context — their
digests, joined. A reading whose adapter could not be hashed is **refused**
rather than stored: two adapters that both failed to hash would carry the same
label and compare equal in the table afterwards, which is the one confusion the
digest exists to prevent.

The quantisation is read from the model rather than named by the caller. A
caller that can name the bit-width can name it wrongly, and a reading labelled
with the wrong representation is worse than an unlabelled one — it enters a
comparison and moves the answer.
