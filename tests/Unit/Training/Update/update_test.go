//go:build libtorch && cgo

package update_test

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	testfixture "github.com/tayi-ai/arandu-llama/tests/Unit/Training/Fixture"
	"math"
	"slices"
	"sync/atomic"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

type fixture struct {
	model   *decoder.LoadedTextModel
	initial *decoder.InitialAdapter
	base    *torch.Tensor
}

func native(t *testing.T, values []float32, shape []int64, grad bool) *torch.Tensor {
	t.Helper()
	value, err := torch.FromFloat32(values, shape, torch.CPUDevice(), grad)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = value.Close() })
	return value
}

func newFixture(t *testing.T) *fixture {
	t.Helper()
	initial, err := decoder.InitializeAdapter(context.Background(), testfixture.Spec(t))
	if err != nil {
		t.Fatal(err)
	}
	base := native(t, []float32{0.25}, []int64{1}, false)
	m := &decoder.LoadedTextModel{Model: &decoder.TextModel{Layers: make([]decoder.Layer, 32), Embedding: base},
		Parameters: slices.Clone(initial.Parameters), Summary: decoder.AssemblySummary{AdapterTensors: 32, AdapterElements: 557056}}
	for i := range m.Model.Layers {
		m.Model.Layers[i].Device = torch.CPUDevice()
	}
	for i := 0; i < 8; i++ {
		p := m.Parameters[i*4 : i*4+4]
		m.Model.Layers[3+i*4].Adapter = &layers.AttentionLoRA{QueryA: p[0].Value, QueryB: p[1].Value, ValueA: p[2].Value, ValueB: p[3].Value, Alpha: 8}
	}
	t.Cleanup(func() { _ = m.Close(); _ = initial.Close() })
	return &fixture{model: m, initial: initial, base: base}
}

func flatten(t *testing.T, parameters []decoder.InitialParameter) []float32 {
	t.Helper()
	var result []float32
	for _, parameter := range parameters {
		values, err := parameter.Value.Float32Values()
		if err != nil {
			t.Fatal(err)
		}
		result = append(result, values...)
	}
	return result
}

func digest(t *testing.T, parameters []decoder.InitialParameter) string {
	t.Helper()
	h := sha256.New()
	var encoded [4]byte
	for _, parameter := range parameters {
		_, _ = h.Write([]byte(parameter.Name))
		values, err := parameter.Value.Float32Values()
		if err != nil {
			t.Fatal(err)
		}
		for _, value := range values {
			binary.LittleEndian.PutUint32(encoded[:], math.Float32bits(value))
			_, _ = h.Write(encoded[:])
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

func expectedDigest(t *testing.T, parameters []decoder.InitialParameter, values []float32) string {
	t.Helper()
	h := sha256.New()
	var encoded [4]byte
	start := 0
	for _, parameter := range parameters {
		info, err := parameter.Value.Info()
		if err != nil {
			t.Fatal(err)
		}
		_, _ = h.Write([]byte(parameter.Name))
		for _, value := range values[start : start+int(info.Elements)] {
			binary.LittleEndian.PutUint32(encoded[:], math.Float32bits(value))
			_, _ = h.Write(encoded[:])
		}
		start += int(info.Elements)
	}
	if start != len(values) {
		t.Fatal("fixture shape differs")
	}
	return hex.EncodeToString(h.Sum(nil))
}

func TestReplaceParametersPreservesIdentityOwnershipAndFrozenBase(t *testing.T) {
	f := newFixture(t)
	values := flatten(t, f.model.Parameters)
	if len(values) != 557056 || digest(t, f.model.Parameters) != testfixture.Spec(t).ExpectedSHA256 {
		t.Fatal("reference initializer differs")
	}
	old := slices.Clone(f.model.Parameters)
	got, err := f.model.ReplaceParameters(context.Background(), values)
	if err != nil || got != testfixture.Spec(t).ExpectedSHA256 {
		t.Fatalf("same-value installation digest=%s error=%v", got, err)
	}
	for _, parameter := range old {
		if _, err := parameter.Value.Info(); !errors.Is(err, torch.ErrClosed) {
			t.Fatalf("old handle not closed: %v", err)
		}
	}
	old = slices.Clone(f.model.Parameters)
	values[0] += 0.125
	values[16384] = math.Float32frombits(0x80000000) // Preserve negative zero bitwise.
	values[len(values)-1] = math.SmallestNonzeroFloat32
	before := slices.Clone(values)
	want := expectedDigest(t, old, values)
	got, err = f.model.ReplaceParameters(context.Background(), values)
	if err != nil || got != want || digest(t, f.model.Parameters) != want || got == testfixture.Spec(t).ExpectedSHA256 {
		t.Fatalf("updated installation digest=%s want=%s error=%v", got, want, err)
	}
	observed := flatten(t, f.model.Parameters)
	for i, value := range observed {
		if math.Float32bits(value) != math.Float32bits(values[i]) || math.Float32bits(before[i]) != math.Float32bits(values[i]) {
			t.Fatalf("value/input mismatch at %d", i)
		}
	}
	var elements int64
	for i, parameter := range f.model.Parameters {
		info, err := parameter.Value.Info()
		if err != nil || info.DType != torch.Float32 || info.Device != torch.CPUDevice() || !info.RequiresGrad {
			t.Fatalf("new metadata: %+v %v", info, err)
		}
		elements += info.Elements
		a := f.model.Model.Layers[3+(i/4)*4].Adapter
		refs := [4]*torch.Tensor{a.QueryA, a.QueryB, a.ValueA, a.ValueB}
		if refs[i%4] != parameter.Value || parameter.Value == old[i].Value || parameter.Name != old[i].Name {
			t.Fatal("parameter references/order not replaced")
		}
		if _, err := old[i].Value.Info(); !errors.Is(err, torch.ErrClosed) {
			t.Fatal("replaced owned handle still live")
		}
	}
	if elements != 557056 || f.model.Summary.AdapterElements != 557056 || f.model.Summary.AdapterTensors != 32 {
		t.Fatal("geometry summary changed")
	}
	base, err := f.base.Float32Values()
	if err != nil || len(base) != 1 || base[0] != 0.25 {
		t.Fatal("frozen base changed")
	}
	current := slices.Clone(f.model.Parameters)
	if err := f.model.Close(); err != nil {
		t.Fatal(err)
	}
	if err := f.model.Close(); err != nil {
		t.Fatal(err)
	}
	for _, parameter := range current {
		if _, err := parameter.Value.Info(); !errors.Is(err, torch.ErrClosed) {
			t.Fatal("owner did not close new handle")
		}
	}
	if _, err := f.base.Info(); err != nil {
		t.Fatal("borrowed base handle closed")
	}
}

func TestRejectedReplacementLeavesAllOldTensorsAndReferencesIntact(t *testing.T) {
	for _, mode := range []string{"short", "long", "NaN", "infinity", "name", "order", "missing adapter", "extra adapter", "alpha", "shape", "dtype", "requires grad", "device", "reference", "duplicate handle", "base alias", "copied base alias", "closed owner"} {
		t.Run(mode, func(t *testing.T) {
			f := newFixture(t)
			values := flatten(t, f.model.Parameters)
			switch mode {
			case "short":
				values = values[:len(values)-1]
			case "long":
				values = append(values, 0)
			case "NaN":
				values[len(values)-1] = float32(math.NaN())
			case "infinity":
				values[len(values)-1] = float32(math.Inf(1))
			case "name":
				f.model.Parameters[31].Name += "wrong"
			case "order":
				f.model.Parameters[0], f.model.Parameters[2] = f.model.Parameters[2], f.model.Parameters[0]
			case "missing adapter":
				f.model.Model.Layers[31].Adapter = nil
			case "extra adapter":
				f.model.Model.Layers[0].Adapter = f.model.Model.Layers[3].Adapter
			case "alpha":
				f.model.Model.Layers[31].Adapter.Alpha = 0
			case "shape":
				value := native(t, make([]float32, 16384), []int64{4096, 4}, true)
				f.model.Parameters[0].Value, f.model.Model.Layers[3].Adapter.QueryA = value, value
			case "dtype":
				value, err := f.model.Parameters[0].Value.To(torch.CPUDevice(), torch.Float16)
				if err != nil {
					t.Fatal(err)
				}
				t.Cleanup(func() { _ = value.Close() })
				f.model.Parameters[0].Value, f.model.Model.Layers[3].Adapter.QueryA = value, value
			case "requires grad":
				value := native(t, make([]float32, 16384), []int64{4, 4096}, false)
				f.model.Parameters[0].Value, f.model.Model.Layers[3].Adapter.QueryA = value, value
			case "device":
				f.model.Model.Layers[31].Device = torch.CUDADevice(0)
			case "reference":
				f.model.Model.Layers[3].Adapter.QueryA = f.model.Parameters[2].Value
			case "duplicate handle":
				f.model.Parameters[2].Value, f.model.Model.Layers[3].Adapter.ValueA = f.model.Parameters[0].Value, f.model.Parameters[0].Value
			case "base alias":
				f.model.Model.Head = f.model.Parameters[0].Value
			case "copied base alias":
				copy := *f.model.Parameters[0].Value
				f.model.Model.Head = &copy
			case "closed owner":
				_ = f.model.Close()
			}
			before := digest(t, f.initial.Parameters)
			refs := slices.Clone(f.model.Parameters)
			got, err := f.model.ReplaceParameters(context.Background(), values)
			if got != "" || !errors.Is(err, decoder.ErrParameters) {
				t.Fatalf("digest=%s error=%v", got, err)
			}
			if digest(t, f.initial.Parameters) != before {
				t.Fatal("old values changed")
			}
			if len(refs) != len(f.model.Parameters) {
				t.Fatal("registry length changed")
			}
			for i := range refs {
				if refs[i] != f.model.Parameters[i] {
					t.Fatal("registry entry changed on failure")
				}
			}
		})
	}
}

type checkingContext struct {
	context.Context
	cancel    context.CancelFunc
	calls     atomic.Int32
	threshold int32
}

func TestCurrentOwnerRejectsForeignOrClosedParameter(t *testing.T) {
	for _, mode := range []string{"foreign", "closed"} {
		t.Run(mode, func(t *testing.T) {
			f := newFixture(t)
			values := flatten(t, f.model.Parameters)
			if _, err := f.model.ReplaceParameters(context.Background(), values); err != nil {
				t.Fatal(err)
			}
			owned := slices.Clone(f.model.Parameters)
			unchanged := owned
			if mode == "foreign" {
				foreign := native(t, values[:16384], []int64{4, 4096}, true)
				f.model.Parameters[0].Value, f.model.Model.Layers[3].Adapter.QueryA = foreign, foreign
			} else {
				_ = f.model.Parameters[31].Value.Close()
				unchanged = owned[:31]
			}
			before := digest(t, unchanged)
			refs := slices.Clone(f.model.Parameters)
			got, err := f.model.ReplaceParameters(context.Background(), values)
			if got != "" || !errors.Is(err, decoder.ErrParameters) {
				t.Fatalf("ownership mismatch accepted: %s %v", got, err)
			}
			if digest(t, unchanged) != before {
				t.Fatal("ownership rejection changed live old tensors")
			}
			for i := range refs {
				if refs[i] != f.model.Parameters[i] {
					t.Fatal("ownership rejection changed model references")
				}
			}
		})
	}
}

func (c *checkingContext) Err() error {
	if calls := c.calls.Add(1); c.threshold > 0 && calls >= c.threshold {
		c.cancel()
	}
	return c.Context.Err()
}

func TestCancellationBeforeCommitRetainsExactOldGeneration(t *testing.T) {
	f := newFixture(t)
	values := flatten(t, f.model.Parameters)
	counting := &checkingContext{Context: context.Background()}
	if _, err := f.model.ReplaceParameters(counting, values); err != nil {
		t.Fatal(err)
	}
	// The last cancellation boundary is immediately before the Go commit, after
	// every native leaf and materialized hash has been prepared. Derive its count
	// from a complete call rather than binding the test to a hard-coded number.
	checks := counting.calls.Load()
	if checks < 32 {
		t.Fatal("replacement lacked bounded cancellation checks")
	}
	for _, threshold := range []int32{1, checks - 20, checks} {
		ctx, cancel := context.WithCancel(context.Background())
		checking := &checkingContext{Context: ctx, cancel: cancel, threshold: threshold}
		old := slices.Clone(f.model.Parameters)
		before := digest(t, old)
		got, err := f.model.ReplaceParameters(checking, values)
		cancel()
		if got != "" || !errors.Is(err, context.Canceled) {
			t.Fatalf("threshold=%d digest=%s error=%v", threshold, got, err)
		}
		if checking.calls.Load() != threshold || digest(t, old) != before {
			t.Fatal("failed preparation changed old native values")
		}
		for i := range old {
			if old[i] != f.model.Parameters[i] {
				t.Fatal("cancelled preparation changed ownership")
			}
		}
	}
	if _, err := f.model.ReplaceParameters(context.Background(), values); err != nil {
		t.Fatalf("cancellation poisoned later replacement: %v", err)
	}
}

// A tiny forward provides a real Snapshot through the public API. The adapter-
// only fixed registry is restored solely during replacement; no large base
// checkpoint is allocated. Restoring the original tiny layer before VJP isolates
// generation validation from the independent state-count and geometry checks.
func tinySnapshot(t *testing.T, f *fixture) (*decoder.Snapshot, []decoder.Layer, *torch.Tensor) {
	t.Helper()
	n := func(values []float32, shape ...int64) *torch.Tensor { return native(t, values, shape, false) }
	model := f.model.Model
	model.Embedding = n([]float32{0.2, -0.1, 0.3, 0.4, -0.2, 0.5}, 3, 2)
	model.FinalNorm = n([]float32{0, 0}, 2)
	model.Head = n([]float32{0.1, 0.2, 0.3, -0.1, -0.2, 0.4}, 3, 2)
	model.Epsilon = 1e-6
	full := &layers.AttentionWeights{Query: n([]float32{0.1, 0.2, 0.2, -0.1, 0.3, 0.1, 0.1, 0.2}, 4, 2),
		Key: n([]float32{0.2, 0.1, -0.1, 0.3}, 2, 2), Value: n([]float32{0.1, 0.3, 0.2, -0.1}, 2, 2),
		Output: n([]float32{0.2, 0.1, 0.3, -0.1}, 2, 2), QueryNorm: n([]float32{0, 0}, 2), KeyNorm: n([]float32{0, 0}, 2)}
	weights := layers.DecoderWeights{InputNorm: n([]float32{0, 0}, 2), PostAttentionNorm: n([]float32{0, 0}, 2), Full: full,
		Gate: n([]float32{0.1, 0.2, 0.2, -0.1}, 2, 2), Up: n([]float32{0.2, 0.1, -0.1, 0.2}, 2, 2), Down: n([]float32{0.1, 0.3, 0.2, -0.1}, 2, 2)}
	adapter := &layers.AttentionLoRA{QueryA: native(t, []float32{0.1, 0.2}, []int64{1, 2}, true), QueryB: native(t, []float32{0.1, 0.2, -0.1, 0.3}, []int64{4, 1}, true),
		ValueA: native(t, []float32{0.2, 0.1}, []int64{1, 2}, true), ValueB: native(t, []float32{0.1, -0.1}, []int64{2, 1}, true), Alpha: 2}
	tiny := []decoder.Layer{{Device: torch.CPUDevice(), Weights: weights, Adapter: adapter, Cosine: n([]float32{1}, 1, 1), Sine: n([]float32{0}, 1, 1),
		Config: layers.DecoderConfig{Epsilon: 1e-6, MaxInputElements: 2, Full: layers.AttentionConfig{Heads: 1, KVHeads: 1, HeadDimension: 2, RotaryDimension: 2, Epsilon: 1e-6, MaxScoreElements: 1}}}}
	registry := model.Layers
	model.Layers = tiny
	snapshot, err := model.Forward(context.Background(), []int64{1}, decoder.Limits{MaxTokens: 1, LogitRows: 1, MaxCheckpointBytes: 24})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = snapshot.Close() })
	model.Layers = registry
	return snapshot, tiny, n([]float32{1, 0, -1}, 1, 1, 3)
}

func TestSuccessfulReplacementInvalidatesOlderSnapshotOnly(t *testing.T) {
	f := newFixture(t)
	snapshot, tiny, seed := tinySnapshot(t, f)
	values := flatten(t, f.model.Parameters)
	bad := slices.Clone(values)
	bad[0] = float32(math.NaN())
	if _, err := f.model.ReplaceParameters(context.Background(), bad); !errors.Is(err, decoder.ErrParameters) {
		t.Fatal(err)
	}
	registry := f.model.Model.Layers
	f.model.Model.Layers = tiny
	gradients, err := f.model.Model.VJP(context.Background(), snapshot, seed)
	if err != nil {
		t.Fatalf("rejected replacement invalidated prior snapshot: %v", err)
	}
	_ = gradients.Close()
	f.model.Model.Layers = registry
	if _, err := f.model.ReplaceParameters(context.Background(), values); err != nil {
		t.Fatal(err)
	}
	registry = f.model.Model.Layers
	f.model.Model.Layers = tiny
	if gradients, err := f.model.Model.VJP(context.Background(), snapshot, seed); gradients != nil || !errors.Is(err, decoder.ErrStaleSnapshot) {
		_ = gradients.Close()
		t.Fatalf("old generation accepted: %v", err)
	}
	fresh, err := f.model.Model.Forward(context.Background(), []int64{1}, decoder.Limits{MaxTokens: 1, LogitRows: 1, MaxCheckpointBytes: 24})
	if err != nil {
		t.Fatal(err)
	}
	defer fresh.Close()
	gradients, err = f.model.Model.VJP(context.Background(), fresh, seed)
	if err != nil {
		t.Fatalf("fresh snapshot rejected: %v", err)
	}
	_ = gradients.Close()
	f.model.Model.Layers = registry
}

func TestReplacementUsesSingleLayerShapeAndExplicitAlpha(t *testing.T) {
	a := &layers.AttentionLoRA{QueryA: native(t, []float32{.1, .2}, []int64{1, 2}, true), QueryB: native(t, []float32{.3, .4, .5, .6}, []int64{4, 1}, true), ValueA: native(t, []float32{.7, .8}, []int64{1, 2}, true), ValueB: native(t, []float32{.9, 1}, []int64{2, 1}, true), Alpha: 3.5}
	m := &decoder.LoadedTextModel{Model: &decoder.TextModel{Layers: []decoder.Layer{{Adapter: a, Device: torch.CPUDevice()}}}}
	for i, value := range []*torch.Tensor{a.QueryA, a.QueryB, a.ValueA, a.ValueB} {
		m.Parameters = append(m.Parameters, decoder.InitialParameter{Name: "base_model.model.model.language_model.layers.0.self_attn." + []string{"q_proj.lora_A.default.weight", "q_proj.lora_B.default.weight", "v_proj.lora_A.default.weight", "v_proj.lora_B.default.weight"}[i], Value: value})
	}
	defer m.Close()
	vector := flatten(t, m.Parameters)
	vector[0] += .01
	got, err := m.ReplaceParameters(context.Background(), vector)
	if err != nil || got == "" {
		t.Fatalf("single layer replacement: %v", err)
	}
	if m.Model.Layers[0].Adapter.Alpha != 3.5 || len(m.Parameters) != 4 {
		t.Fatal("replacement changed admitted geometry or scale")
	}
}
