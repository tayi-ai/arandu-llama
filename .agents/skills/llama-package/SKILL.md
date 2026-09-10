---
name: llama-package
description: Install and wire this package into an application. Use when the request is to "add arandu-llama", "install the llama module", "wire the readings", "run a model in process", "score an example", "it says library not found", "the build fails with cgo", or when working in a project that has github.com/tayi-ai/arandu-llama in its go.mod and needs it registered, built or linked.
license: MIT
---

# You are in the application, not in the package

This skill is for the repository running `go get`. It covers the four things
that go wrong in that order: the build, the registration, the migration and the
closed policy.

## 1. The build is the part that surprises people

```bash
go get github.com/tayi-ai/arandu-llama
```

That gets you code that **compiles and does not link**. The package is a cgo
binding over llama.cpp, and the static archives are not in the module.

```text
ld: library 'binding' not found
```

is the expected first failure and is not a mistake you made.

Build the archives once, from a clone of the package at the version your
`go.mod` resolves:

```bash
git clone --depth 1 --branch v0.2.0 https://github.com/tayi-ai/arandu-llama /binding
cd /binding && git submodule update --init llama.cpp
make BUILD_TYPE=cublas BUILD_LINKAGE=static CUDA_ARCHITECTURES=75 libbinding.a
```

Then point the linker at them:

```bash
CGO_ENABLED=1 CGO_LDFLAGS="-L/binding" go build -tags llama ./...
```

**Never copy archives into `$GOMODCACHE`.** It is read-only by design, shared
by every build on the machine, and verified against `go.sum`. A second `-L`
costs nothing.

**Pin the clone and `go.mod` to the same version.** They drifting apart is not
a link error: the symbol names survive a release, so the build succeeds and the
wrong code runs.

### Put it behind a build tag

Import the engine from a file with `//go:build llama`. Without the tag the
application builds with no C toolchain at all, which is what lets the same
repository produce a small static image for the half that does not score
anything.

## 2. Register it

An Arandu application registers a module explicitly. No provider, no container,
no discovery. All of it is in `bootstrap/app.go`, and nothing else in the
application changes.

The import, with the other module imports:

```go
import (
	llama "github.com/tayi-ai/arandu-llama"
)
```

The construction, in `Build`, after the session store exists and before
`k.Register`:

```go
	llamaModule, err := llama.New(llama.Config{
		Tenant: cfg.Auth.Tenant,
		Prefix: "/readings",
	}, db, sessions)
	if err != nil {
		return App{}, err
	}
```

And the registration, inside the `k.Register(...)` call already there:

```go
		llamaModule,
```

`Tenant` is required and has no default. A visitor with no session has to be
read as some customer, and it cannot be one the request names.

## 3. Run the migration

```bash
aru migrate
```

A **pipeline step, before the rollout**. Never at boot: with N replicas
starting together, N migrations race.

The package declares `migrations = true` in its manifest, which is how you were
supposed to find this out before deploying.

## 4. The policy denies everything

Out of the box every action is refused, and that is the shipped state rather
than a bug. Open what your application needs, in your own policy registration —
one action at a time.

A 404 from a route you believe exists is usually the tenant check, not routing.
The two are indistinguishable on purpose.

## Using the engine

```go
model, err := llama.LoadModel("model-q4_k_m.gguf", llama.WithGPULayers(-1))
defer model.Close()

ctx, err := model.NewContext(llama.WithContextSize(8192))
defer ctx.Close()

adapter, err := model.LoadAdapter("rank4.gguf")
defer adapter.Close()

err = ctx.SetAdapters([]*llama.Adapter{adapter}, []float32{1.0})
score, err := ctx.ScoreText(prompt, completion)
```

Four things to get right, each of which fails quietly rather than loudly:

- **Close the adapter before the model.** Freeing the model destroys every
  adapter loaded against it, so closing the model first leaves a double free.
- **Clear the adapters after a probe.** An engine left perturbed is a model the
  next measurement reads without anybody intending it.
- **Do not interleave scoring with prefix-cached generation on one context.**
  Scoring clears the KV cache. Use a separate context for scoring.
- **Combine scores by summing sums and counts**, then dividing once. Averaging
  the means over-weights short completions silently.

`Context` is not safe for concurrent use. Give each goroutine its own.

## Verifying the install

```bash
go vet ./...                 # the Go half, no archives needed
go vet -tags llama ./...     # the cgo compiles
go build -tags llama ./...   # the archives are found
```

They fail in that order, and each one narrows where to look.
