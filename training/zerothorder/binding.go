//go:build llama

package zerothorder

import (
	"context"
	"errors"
	"fmt"
	"path/filepath"
	"time"

	llama "github.com/tayi-ai/arandu-llama"
	"github.com/tayi-ai/arandu-llama/training/adapter"
)

// The engine that holds the weights, reached in process.
//
// This file is behind a build tag because it is the only part of the
// application that links llama.cpp. The control plane never scores anything, and
// a control plane that had to be built with CGO_ENABLED=1 would give up the
// static binary its image is built around. The node image builds with -tags
// llama; everything else does not, and `go build ./...` on a laptop stays honest
// about what it proved.
//
// What it implements is the Engine interface in ZeroOrder.go, unchanged. The
// HTTP engine and this one are interchangeable for scoring, which is what made
// it possible to measure the method over a server before the binding existed.

// LlamaBinding is a resident model with the policy snapshot applied to it.
//
// The first version kept the direction as a second loaded adapter and probed by
// rescaling it, because a scale change costs 0.4 ms against 2.90 s for the
// forward pass. T02 measured what that probes: llama.cpp sums adapters in
// product space, so the probe read the derivative along ZB*ZA while the fold
// moved the factors along B*ZA + ZB*A. Now every probe and every step applies
// a candidate built by the same constructor -- a file per probe, loaded through
// the same path as the policy -- and the cost of that is what zo:probe's
// `seconds - seconds_plus - seconds_minus` measures.
type LlamaBinding struct {
	model   *llama.Model
	context *llama.Context
	*adapter.Snapshot
}

// LlamaBindingConfig is what the node needs to open one.
type LlamaBindingConfig struct {
	// ModelPath is the quantised base, in GGUF.
	ModelPath string
	// PolicyPath is the adapter being trained, converted to a GGUF LoRA in F32.
	PolicyPath string
	// DirectionDir is the scratch the probe candidates are written to before
	// the engine loads them. The name is kept so the flag it came from stays.
	DirectionDir string
	// ContextTokens has to be at least as long as the longest example scored.
	ContextTokens int
	// GPULayers offloaded. Negative means all of them.
	GPULayers int
}

// llamaHost is the binding's side of the AdapterHost seam.
type llamaHost struct {
	model   *llama.Model
	context *llama.Context
}

func (h llamaHost) LoadAdapter(path string) (adapter.AdapterHandle, error) {
	return h.model.LoadAdapter(path)
}

func (h llamaHost) SetAdapters(handles []adapter.AdapterHandle, scales []float32) error {
	if len(handles) == 0 {
		return h.context.ClearAdapters()
	}
	adapters := make([]*llama.Adapter, len(handles))
	for i, handle := range handles {
		adapter, ok := handle.(*llama.Adapter)
		if !ok {
			return fmt.Errorf("cluster: handle %d is a %T, not an adapter this model loaded", i, handle)
		}
		adapters[i] = adapter
	}
	return h.context.SetAdapters(adapters, scales)
}

// OpenLlama loads the base and the policy, and leaves them on the cards with
// the policy applied.
//
// Everything is loaded once. The whole argument for zeroth-order optimisation on
// this backend rests on the base staying resident between probes.
func OpenLlama(cfg LlamaBindingConfig) (*LlamaBinding, error) {
	if cfg.ModelPath == "" || cfg.PolicyPath == "" {
		return nil, errors.New("cluster: the engine needs a base model and a policy adapter")
	}
	if cfg.DirectionDir == "" {
		return nil, errors.New("cluster: the engine needs a scratch directory for the probe candidates")
	}
	if cfg.ContextTokens <= 0 {
		return nil, errors.New("cluster: the engine needs a context length; a default here would silently truncate the examples it was given")
	}

	model, err := llama.LoadModel(cfg.ModelPath, llama.WithGPULayers(cfg.GPULayers))
	if err != nil {
		return nil, fmt.Errorf("cluster: loading %s: %w", filepath.Base(cfg.ModelPath), err)
	}
	binding := &LlamaBinding{model: model}

	// The batch has to be at least as long as the context, because the number of
	// positions a single decode may flag is bounded by it, and scoring flags
	// every position it reads.
	binding.context, err = model.NewContext(
		llama.WithContext(cfg.ContextTokens),
		llama.WithBatch(cfg.ContextTokens),
	)
	if err != nil {
		binding.Close()
		return nil, fmt.Errorf("cluster: creating a context of %d tokens: %w", cfg.ContextTokens, err)
	}

	binding.Snapshot, err = adapter.OpenSnapshot(llamaHost{model: model, context: binding.context}, cfg.PolicyPath, cfg.DirectionDir)
	if err != nil {
		binding.Close()
		return nil, err
	}
	// The scale a converted adapter is applied at depends on whether the file
	// carries an alpha, and an adapter that lost it during conversion is applied
	// at a different magnitude than the trainer applied it. That failure produces
	// a plausible number rather than an error, so it is checked at open rather
	// than discovered by a loss that will not reconcile.
	if err := binding.admitPolicyScale(cfg.PolicyPath); err != nil {
		binding.Close()
		return nil, err
	}
	return binding, nil
}

// admitPolicyScale refuses a policy whose alpha llama.cpp would read differently
// from the trainer that produced it.
//
// llama.cpp multiplies by adapter_scale * alpha / rank when the file declares a
// non-zero alpha and by adapter_scale alone when it does not; PEFT applied
// alpha / r. A converted adapter that lost the key is therefore applied at
// rank / alpha times the intended magnitude, changing the model being scored.
func (b *LlamaBinding) admitPolicyScale(path string) error {
	adapter, ok := b.Handle().(*llama.Adapter)
	if !ok {
		return fmt.Errorf("cluster: the policy handle is a %T, not an adapter this model loaded", b.Handle())
	}
	alpha, err := adapter.Meta("adapter.lora.alpha")
	if err != nil || alpha == "" {
		return fmt.Errorf("cluster: %s declares no adapter.lora.alpha; llama.cpp would apply it at the bare scale while the trainer applied alpha over rank, and every loss would be off by a constant factor", filepath.Base(path))
	}
	return nil
}

// Close gives the cards back: the snapshot detaches its handles from the
// context before any of them is freed, then the context, then the model.
func (b *LlamaBinding) Close() error {
	if b.Snapshot != nil {
		b.Snapshot.Close()
		b.Snapshot = nil
	}
	if b.context != nil {
		b.context.Close()
		b.context = nil
	}
	if b.model != nil {
		b.model.Close()
		b.model = nil
	}
	return nil
}

// Score reads what the example costs under whatever candidate is applied.
//
// A pre-tokenised example is scored as the sequence the trainer stored rather
// than through this engine's tokeniser; the reason is on Example.Tokens.
func (b *LlamaBinding) Score(_ context.Context, e Example) (Reading, error) {
	started := time.Now()
	var score llama.Score
	var err error
	if len(e.Tokens) > 0 {
		score, err = b.context.Score(e.Tokens, e.PromptTokens)
	} else {
		score, err = b.context.ScoreText(e.Prompt, e.Completion)
	}
	if err != nil {
		return Reading{}, err
	}
	if score.Tokens == 0 {
		return Reading{}, errors.New("cluster: the engine scored no position; a mean over nothing reads the same as a perfect prediction")
	}
	return Reading{Loss: score.Mean(), Tokens: score.Tokens, Elapsed: time.Since(started)}, nil
}
