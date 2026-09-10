# Changelog

Everything worth knowing about a release of Arandu Llama is recorded here.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and
the versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

A published module version is immutable: Go serves it from the proxy forever, so
a release is corrected by another release and never by moving a tag.

## [Unreleased]

- `Context.CaptureFinal` returns the per-token final representation of a
  token sequence at requested positions: the tensor named `result_norm` at the
  pinned llama.cpp commit (after the final norm, before the output projection),
  as `f32` rows of `n_embd_out`. Every decode-window row is computed as an
  output row, the KV cache is cleared first, and non-finite rows are refused.
  A `Capture` carries `TokenDigest` (tokens and positions) and `SnapshotDigest`
  (quantisation, policy and adapter scales, with `CaptureVersion`), so two
  captures of different policies never share a key; readings still label by
  digests alone and do not carry scales.

## [0.2.0] - 2026-09-10

The first release of this repository. It starts at `0.2.0` rather than `0.1.0`
because the proxy already holds `v0.1.0`, `v0.1.1` and `v0.1.2` of this module
path from a previous repository, with checksums that do not match this history.
Reusing those numbers would hand a consumer a checksum mismatch, which reads as
an attack rather than as a version change. **Those three versions are not this
code and must never be tagged again.**

### The package

- An Arandu module: one entity, one policy that denies everything until an
  action is opened, one service that owns the database handle, and the routes
  behind it.
- Five actions, all closed by default: `LlamaView`, `LlamaList`,
  `LlamaCreate`, `LlamaUpdate` and `LlamaDelete`. Opening one is the
  installer's decision, taken in their own policy registration.
- One migration, `20260823_0001_create_llamas`, which creates the `llamas`
  table. `aru migrate` is a pipeline step before the rollout and never runs at
  boot: with N replicas starting together, N migrations race.
- An inference engine in the same package: llama.cpp linked through cgo, with
  the submodule pinned at `90c26fcd` — build b10675.

### What it does that an inference binding does not

- **Adapters over a quantised base, without merging.** Loaded once, applied per
  context, scale changed between forward passes in 0.4 ms.
- **Teacher-forced scoring.** The loss of a completion the model did not write,
  of the same quantity a backpropagating trainer reports. Windowed and bounded
  by `n_batch`, because flagging every position of a 248320-token vocabulary
  asks for 2.46 GB of logits and exceeding `n_outputs_max` is a `GGML_ASSERT`
  that aborts the process.
- **A reading that keeps.** Sum, count, policy digest, quantisation and
  duration as a row, so two measurements can be put side by side.

### Refusals worth knowing about

- A reading whose adapter could not be hashed is **refused**, not stored. Two
  adapters labelled `unknown` compare equal, which is the one confusion the
  digest exists to prevent.
- `Score` returns a sum and a count and never a mean, so a caller combining
  examples weights each by its length.
- Scoring clears the KV cache before decoding, because the cache was filled
  under whatever adapter was applied when it was written.

### Documentation

- [docs/building.md](docs/building.md) — the archives, the linkage modes, the
  linker flag, upgrading llama.cpp.
- [docs/scoring.md](docs/scoring.md) — adapters and scoring, and every place a
  plausible number can come out wrong.
- `.agents/skills/` — six procedures, one per situation.

[Unreleased]: https://github.com/tayi-ai/arandu-llama/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/tayi-ai/arandu-llama/releases/tag/v0.2.0
