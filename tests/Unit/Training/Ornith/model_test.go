//go:build libtorch && cgo

package ornith_test

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/ornith"
	"github.com/tayi-ai/arandu-llama/training/sequence"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

type fixture struct {
	model  *ornith.TextModel
	base   []*torch.Tensor
	tokens []int64
	limits ornith.Limits
	seed   []float32
}

func values(size int, phase, scale float64) []float32 {
	out := make([]float32, size)
	for i := range out {
		out[i] = float32(scale * math.Sin(phase+float64(i)*0.73))
	}
	return out
}

func own(t *testing.T, value *torch.Tensor, err error) *torch.Tensor {
	t.Helper()
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = value.Close() })
	return value
}

func tensor(t *testing.T, data []float32, shape []int64, grad bool) *torch.Tensor {
	t.Helper()
	value, err := torch.FromFloat32(data, shape, torch.CPUDevice(), grad)
	return own(t, value, err)
}

func read(t *testing.T, value *torch.Tensor) []float32 {
	t.Helper()
	data, err := value.Float32Values()
	if err != nil {
		t.Fatal(err)
	}
	return data
}

func newFixture(t *testing.T, storage torch.DType) *fixture {
	t.Helper()
	f := &fixture{tokens: []int64{1, 4, 2}, limits: ornith.Limits{MaxTokens: 3, LogitRows: 2, MaxCheckpointBytes: 1632}, seed: values(12, 0.3, 0.7)}
	base := func(shape []int64, phase, scale float64, stored bool) *torch.Tensor {
		size := 1
		for _, dim := range shape {
			size *= int(dim)
		}
		value := tensor(t, values(size, phase, scale), shape, false)
		if stored && storage != torch.Float32 {
			converted, err := value.To(torch.CPUDevice(), storage)
			value = own(t, converted, err)
		}
		f.base = append(f.base, value)
		return value
	}
	full := &layers.AttentionWeights{
		Query: base([]int64{8, 4}, 0.2, 0.2, false), Key: base([]int64{2, 4}, 0.7, 0.2, false),
		Value: base([]int64{2, 4}, 0.4, 0.3, false), Output: base([]int64{4, 4}, 0.9, 0.3, false),
		QueryNorm: base([]int64{2}, 0.5, 0.02, false), KeyNorm: base([]int64{2}, 0.8, 0.02, false),
	}
	linear := &layers.LinearAttentionWeights{
		QKV: base([]int64{8, 4}, 0.3, 0.2, false), Z: base([]int64{4, 4}, 0.8, 0.3, false),
		Beta: base([]int64{2, 4}, 0.5, 0.2, false), Alpha: base([]int64{2, 4}, 0.6, 0.2, false),
		Convolution: base([]int64{8, 1, 4}, 0.4, 0.25, false), ALog: base([]int64{2}, 0.6, 0.2, false),
		DTBias: base([]int64{2}, 0.5, 0.2, false), Norm: base([]int64{2}, 0.7, 0.6, false),
		Output: base([]int64{4, 4}, 0.3, 0.15, false),
	}
	shared := layers.DecoderWeights{
		InputNorm: base([]int64{4}, 0.1, 0.02, true), PostAttentionNorm: base([]int64{4}, 0.2, 0.02, true),
		Gate: base([]int64{5, 4}, 0.2, 0.1, true), Up: base([]int64{5, 4}, 0.6, 0.1, true), Down: base([]int64{4, 5}, 0.4, 0.1, true),
	}
	cos, sin := make([]float32, 3), make([]float32, 3)
	for i := range cos {
		cos[i], sin[i] = float32(math.Cos(float64(i)*0.3)), float32(math.Sin(float64(i)*0.3))
	}
	cosine, sine := tensor(t, cos, []int64{3, 1}, false), tensor(t, sin, []int64{3, 1}, false)
	f.base = append(f.base, cosine, sine)
	f.model = &ornith.TextModel{
		Embedding: base([]int64{6, 4}, 0.2, 0.4, true), FinalNorm: base([]int64{4}, 0.3, 0.02, true), Head: base([]int64{6, 4}, 0.7, 0.3, true),
		Epsilon: 1e-6, Layers: make([]ornith.Layer, 32),
	}
	config := layers.DecoderConfig{
		Epsilon: 1e-6, MaxInputElements: 12,
		Full: layers.AttentionConfig{Heads: 2, KVHeads: 1, HeadDimension: 2, RotaryDimension: 2, Epsilon: 1e-6, MaxScoreElements: 18},
		Linear: layers.LinearAttentionConfig{KeyHeads: 1, ValueHeads: 2, KeyDimension: 2, ValueDimension: 2, Epsilon: 1e-6, MaxWorkingElements: 1 << 20,
			Sequence: sequence.SequenceLimits{ChunkTokens: 2, MaxTokens: 3, MaxOwnedElements: 1 << 20}},
	}
	for index := range f.model.Layers {
		layer := ornith.Layer{Weights: shared, Config: config, Device: torch.CPUDevice()}
		if index%4 == 3 {
			layer.Weights.Full = full
			layer.Cosine, layer.Sine = cosine, sine
			phase := float64(index) * 0.13
			layer.Adapter = &layers.AttentionLoRA{
				QueryA: tensor(t, values(16, phase+0.2, 0.1), []int64{4, 4}, true),
				QueryB: tensor(t, values(32, phase+0.4, 0.1), []int64{8, 4}, true),
				ValueA: tensor(t, values(16, phase+0.6, 0.1), []int64{4, 4}, true),
				ValueB: tensor(t, values(8, phase+0.8, 0.1), []int64{2, 4}, true), Alpha: 8,
			}
		} else {
			layer.Weights.Linear = linear
		}
		f.model.Layers[index] = layer
	}
	return f
}

func baseHash(t *testing.T, f *fixture) [32]byte {
	t.Helper()
	hash := sha256.New()
	for _, value := range f.base {
		info, err := value.Info()
		if err != nil {
			t.Fatal(err)
		}
		if info.RequiresGrad {
			t.Fatal("base parameter is trainable")
		}
		data, err := value.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		var size [8]byte
		binary.LittleEndian.PutUint64(size[:], uint64(len(data)))
		_, _ = hash.Write(size[:])
		_, _ = hash.Write(data)
	}
	var result [32]byte
	copy(result[:], hash.Sum(nil))
	return result
}

func forward(t *testing.T, f *fixture) *ornith.Snapshot {
	t.Helper()
	snapshot, err := f.model.Forward(context.Background(), f.tokens, f.limits)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = snapshot.Close() })
	return snapshot
}

func backward(t *testing.T, f *fixture, snapshot *ornith.Snapshot) ornith.Gradients {
	t.Helper()
	seed := tensor(t, f.seed, []int64{1, 2, 6}, false)
	gradients, err := f.model.VJP(context.Background(), snapshot, seed)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = gradients.Close() })
	return gradients
}

func assertDetachedFinite(t *testing.T, value *torch.Tensor, dtype torch.DType) {
	t.Helper()
	info, err := value.Info()
	if err != nil {
		t.Fatal(err)
	}
	if info.RequiresGrad || info.DType != dtype {
		t.Fatalf("result metadata: %+v", info)
	}
	finite, err := value.AllFinite()
	if err != nil || !finite {
		t.Fatalf("nonfinite result: %v", err)
	}
}

func gradientNames() []string {
	var names []string
	for layer := 3; layer < 32; layer += 4 {
		for _, suffix := range []string{"q_proj.lora_A.default.weight", "q_proj.lora_B.default.weight", "v_proj.lora_A.default.weight", "v_proj.lora_B.default.weight"} {
			names = append(names, fmt.Sprintf("base_model.model.model.language_model.layers.%d.self_attn.%s", layer, suffix))
		}
	}
	return names
}

func TestTiny32LayersProducesAll32LoRACotangentsAndPreservesFrozenBase(t *testing.T) {
	f := newFixture(t, torch.Float32)
	before := baseHash(t, f)
	snapshot := forward(t, f)
	assertDetachedFinite(t, snapshot.Logits, torch.Float32)
	info, _ := snapshot.Logits.Info()
	if fmt.Sprint(info.Shape) != "[1 2 6]" {
		t.Fatalf("logits shape %v", info.Shape)
	}
	gradients := backward(t, f, snapshot)
	expected := gradientNames()
	if len(gradients) != 32 {
		t.Fatalf("got %d gradients, expected32", len(gradients))
	}
	for index, gradient := range gradients {
		if gradient.Name != expected[index] {
			t.Fatalf("parameter%d got%s expected%s", index, gradient.Name, expected[index])
		}
		assertDetachedFinite(t, gradient.Value, torch.Float32)
		parameterInfo, err := (*parameter(f.model.Layers[index/4*4+3].Adapter, index%4)).Info()
		if err != nil {
			t.Fatal(err)
		}
		gradientInfo, err := gradient.Value.Info()
		if err != nil || fmt.Sprint(gradientInfo.Shape) != fmt.Sprint(parameterInfo.Shape) {
			t.Fatalf("parameter%d gradient geometry differs: %v", index, err)
		}
	}
	var firstNorm float64
	for _, gradient := range gradients[:4] {
		for _, value := range read(t, gradient.Value) {
			firstNorm += math.Abs(float64(value))
		}
	}
	if firstNorm <= 1e-8 {
		t.Fatalf("earliest adapter gradient vanished: %.9g", firstNorm)
	}
	if baseHash(t, f) != before {
		t.Fatal("frozen base bytes changed")
	}
	// All eight adapters are separate leaves, even with shared frozen weights.
	for i := 3; i < 32; i += 4 {
		for j := i + 4; j < 32; j += 4 {
			if f.model.Layers[i].Adapter.QueryA == f.model.Layers[j].Adapter.QueryA {
				t.Fatal("adapter leaves were shared")
			}
		}
	}
	// The snapshot remains reusable: VJP recomputes its own graphs.
	again := backward(t, f, snapshot)
	maxRepeatError := 0.0
	for i := range gradients {
		a, b := read(t, gradients[i].Value), read(t, again[i].Value)
		for j := range a {
			err := math.Abs(float64(a[j]) - float64(b[j]))
			maxRepeatError = math.Max(maxRepeatError, err)
			if err > 1e-7+1e-5*math.Abs(float64(a[j])) {
				t.Fatalf("repeated VJP changed parameter%d element%d: %.9g != %.9g", i, j, a[j], b[j])
			}
		}
	}
	t.Logf("32-layer CPU fixture:32 adapter tensors; first layer3 gradient L1 %.9g; repeated VJP max error %.9g; base SHA256 %x", firstNorm, maxRepeatError, before)
}

func parameter(adapter *layers.AttentionLoRA, index int) **torch.Tensor {
	switch index {
	case 0:
		return &adapter.QueryA
	case 1:
		return &adapter.QueryB
	case 2:
		return &adapter.ValueA
	default:
		return &adapter.ValueB
	}
}

func scalarLoss(t *testing.T, f *fixture) float64 {
	t.Helper()
	snapshot, err := f.model.Forward(context.Background(), f.tokens, f.limits)
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()
	var loss float64
	for i, value := range read(t, snapshot.Logits) {
		loss += float64(value) * float64(f.seed[i])
	}
	return loss
}

func TestFirstAndLastAdaptersMatchFreshForwardCentralDifferences(t *testing.T) {
	f := newFixture(t, torch.Float32)
	before := baseHash(t, f)
	// Finish analytic backward before any replacement; never reuse a snapshot
	// after changing the borrowed model's adapters.
	snapshot := forward(t, f)
	gradients := backward(t, f, snapshot)
	_ = snapshot.Close()
	maxAbsolute, maxRelative := 0.0, 0.0
	for _, layer := range []int{3, 31} {
		for kind := 0; kind < 4; kind++ {
			gradient := read(t, gradients[(layer-3)/4*4+kind].Value)
			// Choose the strongest coordinate for a meaningful FP32 signal, using
			// the analytic gradient only to select which coordinate to perturb.
			coordinate := 0
			for i := range gradient {
				if math.Abs(float64(gradient[i])) > math.Abs(float64(gradient[coordinate])) {
					coordinate = i
				}
			}
			pointer := parameter(f.model.Layers[layer].Adapter, kind)
			original := *pointer
			info, err := original.Info()
			if err != nil {
				t.Fatal(err)
			}
			baseline := read(t, original)
			for _, step := range []float32{0.001, 0.005} {
				losses := [2]float64{}
				coordinates := [2]float32{}
				for direction, sign := range []float32{1, -1} {
					perturbed := append([]float32(nil), baseline...)
					perturbed[coordinate] += sign * step
					coordinates[direction] = perturbed[coordinate]
					replacement, err := torch.FromFloat32(perturbed, info.Shape, torch.CPUDevice(), true)
					replacement = own(t, replacement, err)
					*pointer = replacement
					losses[direction] = scalarLoss(t, f)
					*pointer = original
					_ = replacement.Close()
				}
				finiteDifference := (losses[0] - losses[1]) / float64(coordinates[0]-coordinates[1])
				analytic := float64(gradient[coordinate])
				absolute := math.Abs(analytic - finiteDifference)
				relative := absolute / math.Max(1e-8, math.Abs(analytic))
				maxAbsolute = math.Max(maxAbsolute, absolute)
				maxRelative = math.Max(maxRelative, relative)
				// FP32 model outputs accumulate rounding through32 residual blocks.
				// The absolute allowance is small compared with the selected signal.
				tolerance := 8e-5 + 0.002*math.Abs(analytic)
				if step == 0.005 {
					tolerance = 2e-5 + 0.002*math.Abs(analytic)
				}
				if math.Abs(analytic) < 1e-5 {
					t.Fatalf("weak derivative layer%d kind%d: %.9g", layer, kind, analytic)
				}
				if absolute > tolerance {
					t.Fatalf("layer%d kind%d coordinate%d step%.3g analytic%.9g central%.9g error%.6g tolerance%.6g", layer, kind, coordinate, step, analytic, finiteDifference, absolute, tolerance)
				}
				t.Logf("layer%d kind%d coordinate%d step%.3g analytic%.9g central%.9g abs_error%.6g", layer, kind, coordinate, step, analytic, finiteDifference, absolute)
			}
		}
	}
	if baseHash(t, f) != before {
		t.Fatal("finite difference changed base weights")
	}
	t.Logf("16 central differences: maximum absolute error %.9g relative error %.6g", maxAbsolute, maxRelative)
}

func TestCheckpointAdmissionSnapshotOwnershipAndMissingAdapters(t *testing.T) {
	f := newFixture(t, torch.Float32)
	before := baseHash(t, f)
	for name, change := range map[string]func(*ornith.Limits){
		"one byte below exact boundary": func(l *ornith.Limits) { l.MaxCheckpointBytes = 1631 },
		"zero bytes":                    func(l *ornith.Limits) { l.MaxCheckpointBytes = 0 },
		"token limit":                   func(l *ornith.Limits) { l.MaxTokens = 2 },
		"too many output rows":          func(l *ornith.Limits) { l.LogitRows = 4 },
		"no output rows":                func(l *ornith.Limits) { l.LogitRows = 0 },
	} {
		t.Run(name, func(t *testing.T) {
			limits := f.limits
			change(&limits)
			if snapshot, err := f.model.Forward(context.Background(), f.tokens, limits); snapshot != nil || err == nil {
				if snapshot != nil {
					_ = snapshot.Close()
				}
				t.Fatal("invalid limits accepted")
			}
		})
	}
	for _, tokens := range [][]int64{nil, {-1, 1, 2}, {0, 1, 6}} {
		if snapshot, err := f.model.Forward(context.Background(), tokens, f.limits); snapshot != nil || err == nil {
			if snapshot != nil {
				_ = snapshot.Close()
			}
			t.Fatal("invalid tokens accepted")
		}
	}
	snapshot := forward(t, f)
	seed := tensor(t, f.seed, []int64{1, 2, 6}, false)
	foreign := *f.model
	if gradient, err := foreign.VJP(context.Background(), snapshot, seed); gradient != nil || err == nil {
		_ = gradient.Close()
		t.Fatal("foreign snapshot accepted")
	}
	_ = snapshot.Close()
	if gradient, err := f.model.VJP(context.Background(), snapshot, seed); gradient != nil || err == nil {
		_ = gradient.Close()
		t.Fatal("closed snapshot accepted")
	}
	for i := range f.model.Layers {
		f.model.Layers[i].Adapter = nil
	}
	baseOnly := forward(t, f)
	if gradient, err := f.model.VJP(context.Background(), baseOnly, seed); gradient != nil || err == nil || !strings.Contains(err.Error(), "no trainable adapters") {
		_ = gradient.Close()
		t.Fatalf("missing adapters: %v", err)
	}
	if baseHash(t, f) != before {
		t.Fatal("admission or rejected VJP changed base")
	}
}

type checkingContext struct {
	context.Context
	cancel    context.CancelFunc
	calls     atomic.Int32
	threshold int32
}

func (c *checkingContext) Err() error {
	if c.calls.Add(1) >= c.threshold {
		c.cancel()
	}
	return c.Context.Err()
}

func TestCancellationAndCotangentValidation(t *testing.T) {
	f := newFixture(t, torch.Float32)
	before := baseHash(t, f)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if snapshot, err := f.model.Forward(ctx, f.tokens, f.limits); snapshot != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("canceled forward: %v", err)
	}
	if snapshot, err := f.model.Forward(nil, f.tokens, f.limits); snapshot != nil || err == nil {
		t.Fatal("nil context accepted")
	}
	snapshot := forward(t, f)
	seed := tensor(t, f.seed, []int64{1, 2, 6}, false)
	if gradients, err := f.model.VJP(ctx, snapshot, seed); gradients != nil || !errors.Is(err, context.Canceled) {
		_ = gradients.Close()
		t.Fatalf("canceled backward: %v", err)
	}
	for _, backward := range []bool{false, true} {
		ctx, cancel := context.WithCancel(context.Background())
		checking := &checkingContext{Context: ctx, cancel: cancel, threshold: 500}
		if backward {
			gradient, err := f.model.VJP(checking, snapshot, seed)
			_ = gradient.Close()
			if gradient != nil || !errors.Is(err, context.Canceled) {
				t.Fatalf("mid-backward cancellation: %v", err)
			}
		} else {
			result, err := f.model.Forward(checking, f.tokens, f.limits)
			_ = result.Close()
			if result != nil || !errors.Is(err, context.Canceled) {
				t.Fatalf("mid-forward cancellation: %v", err)
			}
		}
		cancel()
	}
	badSeeds := []*torch.Tensor{nil, tensor(t, []float32{1}, []int64{1}, false)}
	nonfinite := append([]float32(nil), f.seed...)
	nonfinite[0] = float32(math.NaN())
	badSeeds = append(badSeeds, tensor(t, nonfinite, []int64{1, 2, 6}, false))
	wrongType, err := seed.To(torch.CPUDevice(), torch.Float16)
	badSeeds = append(badSeeds, own(t, wrongType, err))
	for _, bad := range badSeeds {
		gradient, err := f.model.VJP(context.Background(), snapshot, bad)
		_ = gradient.Close()
		if gradient != nil || err == nil {
			t.Fatal("invalid logit cotangent accepted")
		}
	}
	if baseHash(t, f) != before {
		t.Fatal("cancellation or invalid cotangent changed base")
	}
	// Failed backwards must not consume the live snapshot.
	if gradients := backward(t, f, snapshot); len(gradients) != 32 {
		t.Fatal("snapshot was damaged by failed backward")
	}
}

func TestStorageCastsAndFrozenWeights(t *testing.T) {
	f := newFixture(t, torch.Float16)
	before := baseHash(t, f)
	snapshot := forward(t, f)
	assertDetachedFinite(t, snapshot.Logits, torch.Float32)
	gradients := backward(t, f, snapshot)
	if len(gradients) != 32 {
		t.Fatalf("half storage returned%d gradients", len(gradients))
	}
	for _, gradient := range gradients {
		assertDetachedFinite(t, gradient.Value, torch.Float32)
	}
	if baseHash(t, f) != before {
		t.Fatal("storage casts changed frozen weights")
	}
	// Storage types must agree; attention weights deliberately remain FP32.
	first := f.model.Layers[0].Weights.InputNorm
	wrong, err := first.To(torch.CPUDevice(), torch.Float32)
	f.model.Layers[0].Weights.InputNorm = own(t, wrong, err)
	if value, err := f.model.Forward(context.Background(), f.tokens, f.limits); value != nil || err == nil {
		_ = value.Close()
		t.Fatal("mismatched storage weight accepted")
	}
	f.model.Layers[0].Weights.InputNorm = first
	for _, target := range []**torch.Tensor{&f.model.Embedding, &f.model.Head, &f.model.FinalNorm, &f.model.Layers[0].Weights.InputNorm} {
		original := *target
		detached, err := original.Detach()
		detached = own(t, detached, err)
		trainable, err := detached.SetRequiresGrad(true)
		*target = own(t, trainable, err)
		result, err := f.model.Forward(context.Background(), f.tokens, f.limits)
		_ = result.Close()
		*target = original
		if result != nil || err == nil {
			t.Fatal("trainable base weight accepted")
		}
	}
}
