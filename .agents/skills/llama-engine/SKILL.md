---
name: llama-engine
description: Work on the cgo half of this package — the C wrapper, the pinned llama.cpp submodule, the mirrored headers, the static archives, the build tag or a link failure. Use when the request is to "add a C function", "expose something from llama.cpp", "upgrade llama.cpp", "bump the submodule", "fix the build", "it says library not found", "add a backend", "make it build on CUDA", or when a change touches wrapper.h, wrapper_adapter.cpp, the Makefile, linkage_*.go, llama_*.go or cgo_headers/.
license: MIT
---

# The two halves fail differently

Tell them apart before doing anything. It is the difference between a
five-minute fix and an afternoon.

| command | proves | needs |
| --- | --- | --- |
| `go vet ./...` | the Go half | nothing |
| `go vet -tags llama ./...` | the cgo **compiles** | headers only |
| `go build -tags llama ./...` | the archives are found and complete | archives |

`ld: library 'binding' not found` is **not** a compile error. The cgo already
compiled; the archives are missing or the linker was not told where they are.

## Where includes come from, and why there are two places

`cgo_headers/` mirrors the llama.cpp headers the wrapper includes,
byte-for-byte. `llama.cpp/` is the submodule.

The Go module proxy **does not package git submodules** — it strips gitlinks
from the zip. A consumer fetching this module through the proxy gets no
`llama.cpp/` at all. Without the mirror they could not compile the wrapper, and
the failure would arrive as a missing header rather than as a missing
submodule.

So: the submodule is needed to **build the archives**, the mirror is needed to
**consume the module**. Both, always, and in the same commit when either moves.

Never let a formatter touch `cgo_headers/`. Those files carry third-party
copyright and permission notices this repository is required to redistribute
intact, and the drift check against upstream is byte-for-byte.

## Adding a C function

Three files, in this order.

1. **`wrapper_adapter.cpp`** — the implementation. New capabilities go here and
   not into `wrapper.cpp`. Keeping the diff against upstream to one file plus a
   few declarations is what makes following llama.cpp cheap; a fork that edits
   the middle of `wrapper.cpp` conflicts on every release.
2. **`wrapper.h`** — the declaration, appended after the existing ones.
3. The Go side — a file of its own, with the cgo preamble and `runtime.KeepAlive`
   for anything the C side reads for the length of the call.

Rules the existing functions follow, and the reason each exists:

- **Write `g_last_error` on every failure path, and return a sentinel.** It is
  the one error slot; `wrapper.cpp` defines it and this file declares it
  `extern`. A failure with no message reaches Go as "something went wrong".
- **Catch `std::exception` and turn it into that message.** An exception
  crossing the cgo boundary is undefined behaviour.
- **Validate pointers and lengths before dereferencing.** The Go side checks
  too, and both checks stay: the C function is callable from anywhere.
- **Never retain a Go pointer past the call.** cgo forbids it and the collector
  will not know.

## Signature changes come from upstream, not from us

The submodule is pinned rather than tracked, and the pin lives in the gitlink.
`.gitmodules` carries the path and the URL and no commit.

This is not caution. `llama_set_adapter_lora` became `llama_set_adapters_lora`
between builds. A floating reference would have broken the wrapper on a Tuesday
with nothing in this repository having changed.

To move the pin:

```bash
git -C llama.cpp fetch --tags
git -C llama.cpp checkout <commit>
make vendor-headers
go build -tags llama ./...
git add llama.cpp cgo_headers
```

Both in the same commit. A mirror describing one build beside a submodule at
another compiles and then behaves as neither.

Read the upstream diff for the functions the wrapper calls before assuming a
build failure is yours.

## Backends

`BUILD_TYPE` selects one, and each sets `CMAKE_ARGS`, `BACKEND_STATIC_LIBS` and
the LDFLAGS in the matching `llama_*.go`. Adding one means all three.

`CUDA_ARCHITECTURES=75` for this fleet — compute 7.5, a Tesla T10. A higher
number produces an image that loads, starts, and dies at the first kernel
launch, on a node, at the end of a queue.

Do **not** add a single backend object to `EXTRA_TARGETS` without checking it
still exists. Metal used to compile to one `ggml-metal.m.o`; upstream made it a
backend library, and the copy failed after every object in llama.cpp had
already been compiled. The archive in `BACKEND_STATIC_LIBS` already carries
those objects.

## The backend tag

`BUILD_TYPE=cublas` compiles the CUDA archives. It does **not** tell the Go
linker to use them: `llama_cublas_static.go` sits behind `//go:build cublas &&
!shared_lib`, and that file is where `-lggml-cuda -lcudart -lcublas -lcuda
-lnccl` live.

So a CUDA build is two things, and forgetting the second one fails at the link
with `undefined reference to 'cudaMallocHost'` after the whole CUDA compile has
succeeded:

```bash
make BUILD_TYPE=cublas BUILD_LINKAGE=static CUDA_ARCHITECTURES=75 libbinding.a
go build -tags cublas ./...
```

Metal and CPU need no tag. That is why this passes on a laptop and fails on the
first node, forty minutes in.

## Linking from a consumer

The package declares `-L./`, which for a dependency resolves to the module
cache: read-only, shared by every build on the machine, verified against
`go.sum`. **Never make it writable to drop archives in.** A second `-L` costs
nothing:

```bash
CGO_LDFLAGS="-L/opt/tayi/lib" go build -tags llama ./...
```

## Before you finish

```bash
gofmt -l .            # excluding llama.cpp/ and cgo_headers/
go vet ./...
go vet -tags llama ./...
go test -race ./...
```

If you changed the wrapper, also build the archives for at least one backend.
"It compiles" is not "it links", and "it links" is not "it returns the right
number".
