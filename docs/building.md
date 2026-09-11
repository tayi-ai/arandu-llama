# Building

This package compiles in two halves that fail in different ways, and telling
them apart saves an afternoon.

The **Go half** compiles anywhere, with no C toolchain and no llama.cpp. It
resolves the wrapper's includes from `cgo_headers/`, which is why that mirror
exists. `go vet ./...` proves this half.

The **link** needs `libbinding.a` and the ggml archives beside it. Those are
produced from the pinned submodule and are not in the repository. A missing one
arrives as:

```text
ld: library 'binding' not found
```

That is not a compile error. The cgo already compiled.

## The archives

```bash
git submodule update --init llama.cpp
make BUILD_TYPE=<backend> BUILD_LINKAGE=static libbinding.a
```

It produces nine files in the package root: `libbinding.a` and the llama and
ggml archives the chosen backend needs.

| `BUILD_TYPE` | for |
| --- | --- |
| *(empty)* | CPU only |
| `metal` | Apple silicon |
| `cublas` | NVIDIA, with `CUDA_ARCHITECTURES` |
| `hipblas` | AMD, with `AMDGPU_TARGETS` |
| `openblas`, `blis` | CPU with a BLAS library |
| `vulkan`, `sycl`, `clblas` | the rest |

For the fleet this package was written for:

```bash
make BUILD_TYPE=cublas BUILD_LINKAGE=static CUDA_ARCHITECTURES=75 libbinding.a
```

**75 is compute 7.5**, which is what a Tesla T10 is. Building for a higher
architecture produces something that loads, starts, and dies at the first
kernel launch — on the node, at the end of a queue.

## Static or shared

`BUILD_LINKAGE=static` is the default and what the images use: the archives are
folded into the binary, and the runtime image needs no copy of libllama or
libggml on its library path.

`BUILD_LINKAGE=shared` produces `.so` files instead. It makes a rebuild
cheaper and a deployment harder, because every host then needs the libraries
and an `LD_LIBRARY_PATH` that finds them.

## The build tag that selects the backend

Building the archives is half of it. The Go side needs a tag to link them, and
the tag has the same name as the `BUILD_TYPE`:

| `BUILD_TYPE` | `go build -tags` |
| --- | --- |
| *(empty)*, `metal` | none |
| `cublas` | `cublas` |
| `hipblas` | `hipblas` |
| `clblas` | `clblas` |
| any, with `BUILD_LINKAGE=shared` | add `shared_lib` |

`llama_cublas_static.go` carries `-lggml-cuda -lcudart -lcublas -lcuda -lnccl`
behind `//go:build cublas && !shared_lib`. Without the tag that file is not
compiled, those flags never reach the linker, and the build fails with a page
of

```text
undefined reference to `cudaMallocHost'
undefined reference to `cublasGemmStridedBatchedEx'
```

**after** the CUDA compile has reached 100%, which is the expensive place to
find out. A consumer that also has its own tag passes both:

```bash
go build -tags "llama cublas" .
```

Metal and the plain CPU build need no tag, which is why this is easy to miss on
a laptop and fails on the first node.

## Linking from an application

The package declares `-L./`, which resolves to its own directory. For a
dependency that directory is the module cache: read-only by design, shared by
every build on the machine, verified against `go.sum`. Making it writable to
drop archives in corrupts something that is not yours to corrupt.

Name a second directory instead:

```bash
CGO_ENABLED=1 CGO_LDFLAGS="-L/opt/tayi/lib" go build -tags llama ./...
```

The linker collects every `-L` it is given and searches all of them for each
`-l`, so `-lbinding` resolves from wherever the archives actually are and the
cache is never touched.

The build tag is the application's choice, not this package's. Putting the
import behind one lets the same application build with no C toolchain at all
when it does not need the engine.

## In an image

Build the archives in one stage and the binary in another. The stage that
produces archives needs the submodule; the stage that produces the binary needs
only the archives and the module cache.

Pin the two to the same version. The archives come from a clone at a tag and
the cgo shim comes from whatever `go.mod` resolves, and those drifting apart is
not a link error — the symbol names survive a release, so the build succeeds and
the wrong code runs. Compare them before building:

```dockerfile
ARG BINDING_VERSION
RUN want="$(go list -m -f '{{.Version}}' github.com/tayi-ai/arandu-llama)" && \
    [ "$want" = "${BINDING_VERSION}" ] || \
    (echo "FATAL: go.mod wants $want, archives built from ${BINDING_VERSION}" && exit 1)
```

## Upgrading llama.cpp

The submodule is pinned rather than tracked, and the pin lives in the gitlink —
`.gitmodules` carries the path and the URL and no commit.

Pinning is not caution. The upstream C API moves: `llama_set_adapter_lora`
became `llama_set_adapters_lora` between builds, and a floating reference would
have broken `wrapper_adapter.cpp` on a Tuesday with nothing in this repository
having changed.

To move it:

```bash
git -C llama.cpp fetch --tags
git -C llama.cpp checkout <new-commit>
make vendor-headers          # re-mirror cgo_headers/ from the new tree
go build -tags llama ./...   # the wrapper is where a signature change lands
git add llama.cpp cgo_headers
```

Re-mirror the headers in the same commit as the gitlink. A `cgo_headers/` that
describes one build and a submodule at another compiles and then behaves as
neither.

## Verifying

| command | what it proves |
| --- | --- |
| `go vet ./...` | the Go half, no archives needed |
| `go vet -tags llama ./...` | the cgo compiles, still no link |
| `go build -tags llama ./...` | the archives are found and complete |
| `go test -race ./...` | the policy denials and the structural twin |
| `make BUILD_TYPE=… libbinding.a` | llama.cpp builds for that backend |

The first three fail in that order, and each one narrows where to look.
