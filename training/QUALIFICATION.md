# Native training calculations

These packages implement differentiable calculations and explicit Ornith text
assembly. They do not start fleet jobs, launch Python, or qualify a production
training run. The application's native-backend refusal is independent.

The model computations are Go code. `training/torch` is an optional cgo bridge
to LibTorch tensor primitives and autograd. The C++ bridge does not contain a
model, a loss function, an optimizer, or a job supervisor. The application's Go
loss and AdamW remain the sources of the training recipe.

## Implemented

- `training`: independent Float64 gated-delta forward and complete VJP, including
  initial/final state derivatives, checkpoint recomputation, cancellation and
  admission of owned float-array memory before allocation.
- `training/torch`: explicit tensor ownership, dtype/device selection, native
  operations, VJP and errors across the cgo boundary. Builds without `libtorch`
  and cgo refuse execution with `ErrUnavailable`.
- `training/tensor`: bounded gated-delta chunks, detached forward and a
  recomputed VJP. Its caller must propagate the initial-state cotangent to the
  previous chunk. A chunk is limited by both token and element budgets. The
  arithmetic memory estimate excludes native allocator and autograd overhead;
  it is not a measured GPU peak. At the full model's state dimensions the
  element budget can require a chunk shorter than the absolute 32-token cap.
- `training/layers`: frozen base projections with LoRA, Qwen3.5 RMSNorm offset
  weights, Float32 gated RMSNorm, SwiGLU, and causal gated grouped-query
  attention. Query/gate projection heads are interleaved. Rotary inputs are
  explicit, including partial rotation. No checkpoint weight conversion is
  implicit.
- `training/sequence`: eight-token recurrent checkpoints and reverse state VJP;
  `training/ornith`: complete 32-layer text forward/VJP, reference-bound loading,
  rotary tables, candidate-logit gradients and atomic adapter replacement.
- `training/loading`: bounded Safetensors reads and explicit precision/device
  copies. `checkpoint.WriteFloat32` exports deterministic adapter tensors.
- `training/tokenizer`: fixed-schema Qwen byte BPE. All 378 frozen prompts match
  the official tokenizers 0.22.2 reference: 239249 IDs, zero differences. Unicode
  coverage outside this corpus is explicitly limited by the normalization tables.
- `training/collective`: caller-owned authenticated connections, ordered gradient
  gathering and parameter broadcast. Fleet retains all process/job ownership.
- Exact initial reconstruction: seed83, 24 historical GDN prelude draws and 32
  matching LoRA tensors; aggregate SHA256
  `75185d68fcfdd09bb2dcb6dffe0902a35300f52d9fda716b408189991773aa7c`.

## Reproducing native CPU checks

Use Go 1.27.0 and matching LibTorch 2.14.0 headers and libraries. This release of
LibTorch requires C++20. Obtain the native distribution from PyTorch's official
download index; no Python interpreter or Python package installation is needed.
`HeaderVersion()` describes the compile-time headers; deployments must verify
the actual loaded library hashes separately.

With `TAYI_LIBTORCH_ROOT` pointing to the extracted `libtorch` directory:

```sh
export GOWORK=off GOTOOLCHAIN=go1.27.0 CGO_ENABLED=1
export CGO_CXXFLAGS="-I$TAYI_LIBTORCH_ROOT/include -I$TAYI_LIBTORCH_ROOT/include/torch/csrc/api/include"
export CGO_LDFLAGS="-L$TAYI_LIBTORCH_ROOT/lib -Wl,-rpath,$TAYI_LIBTORCH_ROOT/lib"
go vet -tags libtorch ./training/... ./tests/Unit/Training/...
go test -race -tags libtorch ./tests/Unit/Training/...
```

The Linux CUDA bridge compiled against the installed LibTorch 2.14.0+cu126 with
GCC13/C++20, and its CPU tests passed in the registered Tayi container. This
checks the native ABI, not CUDA kernels, full model behavior or memory on SM75.

Tests compare derivatives to scalar analytical calculations and central finite
differences, cover gradients through frozen recurrent state, cross chunk
boundaries, verify causal attention with LoRA and partial rotation, and check
ownership, cancellation, invalid inputs and error paths. CPU fixtures are not
model training, fleet execution, GPU admission or benchmark evidence.

## Local Apple Metal qualification (2026-09-25)

The optional bridge now admits the unindexed MPS device. The frozen CUDA
reference is still validated before `PlanLocalMPSAssembly` explicitly places
its 459 tensors on one MPS device. The plan requires a caller-supplied aggregate
payload cap; this is not a measured process-memory limit. CPU/CUDA placement and
the frozen two-GPU path remain available.

On an M4 Max with 36 GiB unified memory, the official macOS arm64 LibTorch
2.14.0 archive (SHA-256
`2985d3e27e7c8509e17862a0aac140556f944e22a7a53bea7d753d3acc3236ce`)
passed the full tagged suite, including a 32-layer small-model causal loss/VJP
comparison with CPU. The four source Safetensors shards at Ornith revision
`489cb97981b8654bcfcf30ce1f94ed1b62e07b53` matched their pinned hashes.
The full 9B text assembly loaded and verified all 459 reference tensors in
24.25 seconds. One admitted Arandu completion (743 input tokens, 74 supervised)
produced loss 3.635525942 and 32 finite LoRA gradients, 294912 nonzero. A
separately identified local batch-one AdamW step changed 294912 parameters;
the adapter and moments were saved as Safetensors. A fresh process reloaded the
adapter and reproduced the post-update logits hash. Evidence lives under
`runtime/arasa-local-train-20260925/` in the parent project.

This establishes one local update and adapted text inference. It does not
establish a completed SFT curriculum, teacher fusion, Recovery/Protection,
quantized variants, quality gains, or a safe memory peak for long contexts.
The local batch-one numerical mode is distinct from the frozen distributed
20-rank optimizer protocol.

The Go `training/optim` package now owns a batch-one AdamW update with
validated, resumable FP32 parameters and moments. A second real Arandu example
resumed from the first checkpoint and saved a second checkpoint: loss
3.006498074 over 69 supervised tokens and 557053 changed parameters. A fresh
process verified the saved adapter and reproduced its logits digest exactly.
Three more 1024-token-bounded Arandu examples resumed the same optimizer to
step five. Atomic adapter and moments checkpoints exist for each step. A fresh
process reloaded step five and reproduced its logits digest. This is five of
103 examples used for qualification, not a complete SFT curriculum. A LibTorch
MPS allocator sample after reload reported 22,110,151,168 tensor bytes and
22,423,224,320 driver bytes against a 30,150,672,384-byte recommended working
set. This instantaneous sample is not a peak or a full process-memory budget.
The parent project's evidence is `runtime/arasa-local-train-20260925/`.

## Remaining before full model training and promotion

The local-first path needs integration of the qualified resume state into an
Arandu Controller Job, plus
admission of longer contexts without silent truncation, measured Metal memory
limits, the full admitted SFT curriculum, teacher caches, Recovery/Protection
and independent quality gates before Master or quantized variants are claimed.
The following Linux/SM75 work is the separate historical distributed path:

1. Qualify forward and backward on the frozen calibration inputs, including
   numeric casts and the admitted loss/optimizer trajectory.
2. Build the complete Linux SM75 worker, verify placement across two GPUs and measure
   load/forward/backward memory under the existing guards.
3. Qualify the 21-node gradient exchange through the existing fleet job lifecycle
   while preserving the Go optimizer's deterministic rank reduction order.

The final native CPU suite passed 123 top-level tests and 296 subtests across
15 packages with the race detector. Two external-fixture tests were skipped in
that suite; frozen model metadata and corpus tokenization were checked separately.
No optimizer step on GPUs is established by these CPU results.

No capability above is established merely by publishing these packages.

References: [LibTorch installation](https://docs.pytorch.org/cppdocs/installing.html),
[native autograd](https://docs.pytorch.org/tutorials/advanced/cpp_autograd), and
[Qwen3.5 implementation, Transformers 5.16.1](https://raw.githubusercontent.com/huggingface/transformers/v5.16.1/src/transformers/models/qwen3_5/modeling_qwen3_5.py).
