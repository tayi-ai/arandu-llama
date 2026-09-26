//go:build libtorch && cgo

package step_test

import (
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"math"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

type tinyModel struct {
	model      *decoder.TextModel
	base       []*torch.Tensor
	parameters []**torch.Tensor
	prompt     []int64
	candidates [4]int64
	limits     decoder.Limits
}

func pattern(size int, phase, scale float64) []float32 {
	result := make([]float32, size)
	for i := range result {
		result[i] = float32(scale * math.Sin(phase+float64(i)*.73))
	}
	return result
}

func newTiny(t *testing.T) *tinyModel {
	t.Helper()
	f := &tinyModel{prompt: []int64{1, 4, 5}, candidates: [4]int64{5, 1, 4, 2}, limits: decoder.Limits{MaxTokens: 3, LogitRows: 2, MaxCheckpointBytes: 4096}}
	makeTensor := func(data []float32, shape []int64, grad bool) *torch.Tensor {
		value, err := torch.FromFloat32(data, shape, torch.CPUDevice(), grad)
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = value.Close() })
		if !grad {
			f.base = append(f.base, value)
		}
		return value
	}
	base := func(shape []int64, phase, scale float64) *torch.Tensor {
		size := int64(1)
		for _, dim := range shape {
			size *= dim
		}
		return makeTensor(pattern(int(size), phase, scale), shape, false)
	}
	f.model = &decoder.TextModel{Embedding: base([]int64{6, 4}, .2, .4), FinalNorm: base([]int64{4}, .3, .02), Head: base([]int64{6, 4}, .7, .3), Layers: make([]decoder.Layer, 2), Epsilon: 1e-6}
	for i := range f.model.Layers {
		phase := float64(i) * .8
		weights := layers.DecoderWeights{InputNorm: base([]int64{4}, .1, .02), PostAttentionNorm: base([]int64{4}, .2, .02), Gate: base([]int64{5, 4}, .2, .1), Up: base([]int64{5, 4}, .6, .1), Down: base([]int64{4, 5}, .4, .1), Full: &layers.AttentionWeights{
			Query: base([]int64{8, 4}, phase+.2, .2), Key: base([]int64{2, 4}, phase+.7, .2), Value: base([]int64{2, 4}, phase+.4, .3), Output: base([]int64{4, 4}, phase+.9, .3), QueryNorm: base([]int64{2}, .5, .02), KeyNorm: base([]int64{2}, .8, .02),
		}}
		adapter := &layers.AttentionLoRA{QueryA: makeTensor(pattern(8, phase+.2, .15), []int64{2, 4}, true), QueryB: makeTensor(pattern(16, phase+.4, .15), []int64{8, 2}, true), ValueA: makeTensor(pattern(8, phase+.6, .15), []int64{2, 4}, true), ValueB: makeTensor(pattern(4, phase+.8, .15), []int64{2, 2}, true), Alpha: 4}
		cosine, sine := make([]float32, 3), make([]float32, 3)
		for n := range cosine {
			cosine[n] = float32(math.Cos(float64(n) * .3))
			sine[n] = float32(math.Sin(float64(n) * .3))
		}
		f.model.Layers[i] = decoder.Layer{Weights: weights, Adapter: adapter, Device: torch.CPUDevice(), Cosine: makeTensor(cosine, []int64{3, 1}, false), Sine: makeTensor(sine, []int64{3, 1}, false), Config: layers.DecoderConfig{Epsilon: 1e-6, MaxInputElements: 12, Full: layers.AttentionConfig{Heads: 2, KVHeads: 1, HeadDimension: 2, RotaryDimension: 2, Epsilon: 1e-6, MaxScoreElements: 18}}}
		f.parameters = append(f.parameters, &adapter.QueryA, &adapter.QueryB, &adapter.ValueA, &adapter.ValueB)
	}
	return f
}

func flatLogits(t *testing.T, f *tinyModel) []float32 {
	t.Helper()
	snapshot, err := f.model.Forward(context.Background(), f.prompt, f.limits)
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()
	values, err := snapshot.Logits.Float32Values()
	if err != nil {
		t.Fatal(err)
	}
	return values
}

func frozenDigest(t *testing.T, f *tinyModel) [32]byte {
	t.Helper()
	hash := sha256.New()
	for _, value := range f.base {
		raw, err := value.Bytes()
		if err != nil {
			t.Fatal(err)
		}
		_, _ = hash.Write(raw)
	}
	var result [32]byte
	copy(result[:], hash.Sum(nil))
	return result
}

func TestCandidateReadUsesPenultimateRowAndExplicitCandidateOrder(t *testing.T) {
	f := newTiny(t)
	before := frozenDigest(t, f)
	raw := flatLogits(t, f)
	got, err := decoder.ReadCandidateLogits(context.Background(), f.model, f.prompt, f.candidates, f.limits)
	if err != nil {
		t.Fatal(err)
	}
	var lastDiff float64
	for i, id := range f.candidates {
		if got[i] != float64(raw[id]) {
			t.Fatalf("candidate%d got%g firstrow%g", i, got[i], raw[id])
		}
		lastDiff += math.Abs(got[i] - float64(raw[6+id]))
	}
	if lastDiff < 1e-3 {
		t.Fatal("fixture does not distinguish penultimate and last positions")
	}
	again, err := decoder.ReadCandidateLogits(context.Background(), f.model, f.prompt, f.candidates, f.limits)
	if err != nil || again != got {
		t.Fatalf("repeated score differs:%v", err)
	}
	for i := range f.model.Layers {
		f.model.Layers[i].Adapter = nil
	}
	if _, err := decoder.ReadCandidateLogits(context.Background(), f.model, f.prompt, f.candidates, f.limits); err != nil {
		t.Fatalf("base-only scoring needs no optimizer:%v", err)
	}
	if frozenDigest(t, f) != before {
		t.Fatal("frozen base changed")
	}
}

func TestCandidateGradientMatchesIndependentPenultimateFiniteDifferences(t *testing.T) {
	f := newTiny(t)
	before := frozenDigest(t, f)
	coefficients := [4]float64{.7, -.2, .4, -.9}
	calls := 0
	raw := flatLogits(t, f)
	var observed [4]float64
	for i, id := range f.candidates {
		observed[i] = float64(raw[id])
	}
	result, err := decoder.CandidateGradient(context.Background(), f.model, f.prompt, f.candidates, f.limits, func(logits [4]float64) ([4]float64, error) {
		calls++
		if logits != observed {
			t.Fatal("callback received different logits")
		}
		return coefficients, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if calls != 1 || result.Logits != observed || len(result.Gradients) != 8 {
		t.Fatalf("gradient result:%+v callbacks%d", result, calls)
	}
	objective := func() float64 {
		raw := flatLogits(t, f)
		var result float64
		for i, id := range f.candidates {
			result += coefficients[i] * float64(raw[id])
		}
		return result
	}
	// FP32 forwards lose resolution at very small perturbations. Combine central
	// differences at h and 2h to cancel the leading quadratic truncation error;
	// this oracle uses only forward values, never the candidate/VJP helpers.
	const delta = float32(.01)
	var largestError, largestGradient float64
	for i, gradient := range result.Gradients {
		original := *f.parameters[i]
		info, err := original.Info()
		if err != nil {
			t.Fatal(err)
		}
		values, err := original.Float32Values()
		if err != nil {
			t.Fatal(err)
		}
		suffix := []string{"q_proj.lora_A.default.weight", "q_proj.lora_B.default.weight", "v_proj.lora_A.default.weight", "v_proj.lora_B.default.weight"}[i%4]
		expectedName := fmt.Sprintf("base_model.model.model.language_model.layers.%d.self_attn.%s", i/4, suffix)
		if gradient.Name != expectedName || !reflect.DeepEqual(gradient.Shape, info.Shape) || len(gradient.ValuesF32) != len(values) {
			t.Fatalf("gradient identity:%+v", gradient)
		}
		for element := range values {
			central := func(step float32) (float64, float64) {
				plus, minus := append([]float32(nil), values...), append([]float32(nil), values...)
				plus[element] += step
				minus[element] -= step
				evaluate := func(data []float32) float64 {
					perturbed, err := torch.FromFloat32(data, info.Shape, torch.CPUDevice(), true)
					if err != nil {
						t.Fatal(err)
					}
					*f.parameters[i] = perturbed
					defer func() { *f.parameters[i] = original; _ = perturbed.Close() }()
					return objective()
				}
				span := float64(plus[element]) - float64(minus[element])
				return (evaluate(plus) - evaluate(minus)) / span, span
			}
			d1, span1 := central(delta)
			d2, span2 := central(2 * delta)
			ratioSquared := (span2 / span1) * (span2 / span1)
			oracle := (ratioSquared*d1 - d2) / (ratioSquared - 1)
			actual := float64(gradient.ValuesF32[element])
			difference := math.Abs(oracle - actual)
			largestError = math.Max(largestError, difference)
			largestGradient = math.Max(largestGradient, math.Abs(actual))
			if difference > 1e-5+math.Abs(oracle)*.001 {
				t.Fatalf("parameter%d element%d got%.8g finiteDifference%.8g diff%.4g", i, element, actual, oracle, difference)
			}
		}
	}
	if largestGradient < 1e-4 {
		t.Fatal("gradient fixture is degenerate")
	}
	t.Logf("all_parameter_finite_difference_max_abs_error=%.9g", largestError)
	if frozenDigest(t, f) != before {
		t.Fatal("frozen base changed")
	}
	// Returned slices own Go data; modifying them must not change model leaves.
	first, _ := (*f.parameters[0]).Float32Values()
	result.Gradients[0].ValuesF32[0] = 999
	result.Gradients[0].Shape[0] = 99
	after, _ := (*f.parameters[0]).Float32Values()
	if !reflect.DeepEqual(first, after) {
		t.Fatal("returned gradients alias model")
	}
}

func TestCandidateCallbackOwnsScalingAndCanReturnZero(t *testing.T) {
	f := newTiny(t)
	gradient := [4]float64{.5, -.5, .25, -.25}
	call := func(scale float64) decoder.CandidateGradientResult {
		t.Helper()
		result, err := decoder.CandidateGradient(context.Background(), f.model, f.prompt, f.candidates, f.limits, func([4]float64) ([4]float64, error) {
			g := gradient
			for i := range g {
				g[i] *= scale
			}
			return g, nil
		})
		if err != nil {
			t.Fatal(err)
		}
		return result
	}
	unscaled, scaled, zero := call(1), call(1.0/1024), call(0)
	for i := range unscaled.Gradients {
		for j, value := range unscaled.Gradients[i].ValuesF32 {
			if math.Abs(float64(scaled.Gradients[i].ValuesF32[j]-value/1024)) > 1e-10 {
				t.Fatal("callback scaling changed")
			}
			if zero.Gradients[i].ValuesF32[j] != 0 {
				t.Fatal("zero cotangent produced nonzero gradient")
			}
		}
	}
}

func TestCandidateFailureRefusesPartialResultsAndPreservesModel(t *testing.T) {
	f := newTiny(t)
	before := frozenDigest(t, f)
	marker := errors.New("caller derivative refused")
	for _, test := range []struct {
		name       string
		derivative decoder.LossGradient
		expected   error
	}{
		{"callback_error", func([4]float64) ([4]float64, error) { return [4]float64{}, marker }, marker},
		{"nan", func([4]float64) ([4]float64, error) { return [4]float64{math.NaN()}, nil }, decoder.ErrCandidateStep},
		{"infinity", func([4]float64) ([4]float64, error) { return [4]float64{math.Inf(-1)}, nil }, decoder.ErrCandidateStep},
		{"fp32_overflow", func([4]float64) ([4]float64, error) { return [4]float64{math.MaxFloat64}, nil }, decoder.ErrCandidateStep},
	} {
		t.Run(test.name, func(t *testing.T) {
			result, err := decoder.CandidateGradient(context.Background(), f.model, f.prompt, f.candidates, f.limits, test.derivative)
			if !errors.Is(err, test.expected) || len(result.Gradients) != 0 || result.Logits != [4]float64{} {
				t.Fatalf("invalid derivative:%+v %v", result, err)
			}
			if _, err := decoder.ReadCandidateLogits(context.Background(), f.model, f.prompt, f.candidates, f.limits); err != nil {
				t.Fatalf("failure invalidated model:%v", err)
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	result, err := decoder.CandidateGradient(ctx, f.model, f.prompt, f.candidates, f.limits, func([4]float64) ([4]float64, error) { cancel(); return [4]float64{1}, nil })
	if !errors.Is(err, context.Canceled) || len(result.Gradients) != 0 || result.Logits != [4]float64{} {
		t.Fatalf("callback cancellation:%+v %v", result, err)
	}
	if frozenDigest(t, f) != before {
		t.Fatal("failure changed frozen base")
	}
}

func TestCandidateValidationRejectsAmbiguousScoringWindow(t *testing.T) {
	f := newTiny(t)
	for _, test := range []struct {
		name   string
		prompt []int64
		ids    [4]int64
		rows   int64
	}{
		{"last_row_only", f.prompt, f.candidates, 1},
		{"all_rows", f.prompt, f.candidates, 3},
		{"missing_appended_A", []int64{1, 4, 2}, f.candidates, 2},
		{"too_short", []int64{5}, f.candidates, 2},
		{"duplicates", f.prompt, [4]int64{5, 1, 1, 2}, 2},
		{"negative", f.prompt, [4]int64{5, -1, 4, 2}, 2},
		{"outside_vocabulary", f.prompt, [4]int64{5, 1, 4, 6}, 2},
	} {
		t.Run(test.name, func(t *testing.T) {
			limits := f.limits
			limits.LogitRows = test.rows
			result, err := decoder.ReadCandidateLogits(context.Background(), f.model, test.prompt, test.ids, limits)
			if !errors.Is(err, decoder.ErrCandidateStep) || result != [4]float64{} {
				t.Fatalf("invalid window accepted:%v", err)
			}
		})
	}
	if _, err := decoder.CandidateGradient(context.Background(), f.model, f.prompt, f.candidates, f.limits, nil); !errors.Is(err, decoder.ErrCandidateStep) {
		t.Fatal("nil derivative accepted")
	}
}
