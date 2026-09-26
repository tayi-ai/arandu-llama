//go:build libtorch && cgo

package decoder_test

import (
	"context"
	"errors"
	testfixture "github.com/tayi-ai/arandu-llama/tests/Unit/Training/Fixture"
	"math"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

func featureFixture(t *testing.T, storage torch.DType) *fixture {
	t.Helper()
	f := newFixture(t, storage)
	// Two trainable blocks, separated by frozen recurrent blocks. Keeping both
	// proves that an intermediate feature does not reach a later adapter.
	f.model.Layers = f.model.Layers[:8]
	f.limits.LogitRows = 3
	return f
}

func featureTargets() []decoder.FeatureTarget {
	return []decoder.FeatureTarget{
		{Source: "teacher-a", Layer: 2, Position: 0, Weight: 0.15, Values: []float32{0.8, -0.7, 0.4, 0.2}},
		{Source: "teacher-a", Layer: 4, Position: 1, Weight: 0.6, Values: []float32{0.3, -0.9, 0.8, -0.4}},
		{Source: "teacher-a", Layer: 7, Position: 2, Weight: 0.35, Values: []float32{-0.5, 0.6, -0.2, 0.9}},
		{Source: "teacher-b", Layer: 7, Position: 2, Weight: 0.25, Values: []float32{0.7, -0.3, 0.4, -0.8}},
	}
}

// manualFeatureLoss runs decoders directly, outside Snapshot/VJPWithFeatures,
// and scores their outputs in Go. It does not use the feature implementation.
func manualFeatureLoss(t *testing.T, f *fixture, targets []decoder.FeatureTarget) float64 {
	t.Helper()
	embedding := read(t, f.model.Embedding)
	info, err := f.model.Embedding.Info()
	if err != nil {
		t.Fatal(err)
	}
	width := int(info.Shape[1])
	input := make([]float32, 0, len(f.tokens)*width)
	for _, id := range f.tokens {
		input = append(input, embedding[int(id)*width:(int(id)+1)*width]...)
	}
	current, err := torch.FromFloat32(input, []int64{1, int64(len(f.tokens)), int64(width)}, info.Device, false)
	if err != nil {
		t.Fatal(err)
	}
	defer func() { _ = current.Close() }()
	loss := 0.0
	for index, layer := range f.model.Layers {
		placed, err := current.To(layer.Device, info.DType)
		if err != nil {
			t.Fatal(err)
		}
		_ = current.Close()
		current = placed
		output, err := layers.DecoderForward(context.Background(), current, layer.Weights, layer.Adapter, layer.Cosine, layer.Sine, layer.Config)
		if err != nil {
			t.Fatal(err)
		}
		_ = current.Close()
		current = output
		for _, target := range targets {
			if target.Layer != index {
				continue
			}
			values := read(t, output)
			squared := 0.0
			for column, want := range target.Values {
				difference := float64(values[int(target.Position)*width+column]) - float64(want)
				squared += difference * difference
			}
			loss += target.Weight * squared / float64(width)
		}
	}
	return loss
}

func fusedFeatures(t *testing.T, f *fixture, features []decoder.FeatureTarget, scale float64) decoder.FusionCompletionResult {
	t.Helper()
	result, err := decoder.FusionCompletionGradientWithFeatures(context.Background(), f.model, f.tokens, 1, f.limits, scale, nil, features)
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func TestFeatureFusionMatchesIndependentLossAndCentralDifferences(t *testing.T) {
	f := featureFixture(t, torch.Float32)
	before := baseHash(t, f)
	targets := featureTargets()
	result := fusedFeatures(t, f, targets, 1)
	wantFeatures := manualFeatureLoss(t, f, targets)
	wantCE := manualCompletionLoss(t, f, 1, f.limits)
	if math.Abs(result.FeatureLoss-wantFeatures) > 1e-12 || math.Abs(result.Loss-wantCE-wantFeatures) > 1e-12 {
		t.Fatalf("loss total=%.12g features=%.12g manual CE=%.12g features=%.12g", result.Loss, result.FeatureLoss, wantCE, wantFeatures)
	}
	if result.FeatureLoss <= 0 || len(result.Gradients) != 8 {
		t.Fatalf("missing feature signal: loss=%g gradients=%d", result.FeatureLoss, len(result.Gradients))
	}
	// Each source contributes its own squared loss even at the same boundary.
	separate := manualFeatureLoss(t, f, targets[:3]) + manualFeatureLoss(t, f, targets[3:])
	if math.Abs(result.FeatureLoss-separate) > 1e-12 {
		t.Fatal("teachers at one position were merged before scoring")
	}
	maxError := 0.0
	for _, layer := range []int{3, 7} {
		for kind := 0; kind < 4; kind++ {
			gradient := result.Gradients[(layer-3)/4*4+kind].ValuesF32
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
			losses, coordinates := [2]float64{}, [2]float32{}
			for direction, sign := range []float32{1, -1} {
				perturbed := append([]float32(nil), baseline...)
				perturbed[coordinate] += sign * 0.005
				coordinates[direction] = perturbed[coordinate]
				replacement, err := torch.FromFloat32(perturbed, info.Shape, info.Device, true)
				if err != nil {
					t.Fatal(err)
				}
				*pointer = replacement
				losses[direction] = manualCompletionLoss(t, f, 1, f.limits) + manualFeatureLoss(t, f, targets)
				*pointer = original
				_ = replacement.Close()
			}
			central := (losses[0] - losses[1]) / float64(coordinates[0]-coordinates[1])
			analytic := float64(gradient[coordinate])
			error := math.Abs(central - analytic)
			maxError = math.Max(maxError, error)
			if math.Abs(analytic) < 1e-5 || error > 3e-5+0.003*math.Abs(analytic) {
				t.Fatalf("layer=%d kind=%d coordinate=%d analytic=%.9g central=%.9g", layer, kind, coordinate, analytic, central)
			}
			t.Logf("layer=%d kind=%d analytic=%.9g central=%.9g error=%.9g", layer, kind, analytic, central, error)
		}
	}
	if baseHash(t, f) != before {
		t.Fatal("feature differentiation changed frozen weights")
	}
	t.Logf("feature loss=%.12g total=%.12g eight central differences max error=%.9g", result.FeatureLoss, result.Loss, maxError)
}

func TestFeatureFusionZeroWeightsScaleAndLayerBoundary(t *testing.T) {
	f := featureFixture(t, torch.Float32)
	baseline, err := decoder.FusionCompletionGradient(context.Background(), f.model, f.tokens, 1, f.limits, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	zero := featureTargets()
	for i := range zero {
		zero[i].Weight = 0
	}
	if got := fusedFeatures(t, f, zero, 1); !reflect.DeepEqual(got, baseline) {
		t.Fatal("zero-weight features changed loss or gradients")
	}
	intermediate := []decoder.FeatureTarget{featureTargets()[1]}
	result := fusedFeatures(t, f, intermediate, 1)
	if result.Loss <= baseline.Loss {
		t.Fatal("feature objective did not change the loss")
	}
	changed := false
	for i := range result.Gradients {
		equal := reflect.DeepEqual(result.Gradients[i], baseline.Gradients[i])
		if i < 4 {
			changed = changed || !equal
		} else if !equal {
			t.Fatal("intermediate feature reached a later adapter")
		}
	}
	if !changed {
		t.Fatal("intermediate feature did not reach the earlier adapter through the frozen block")
	}
	// A feature before the first adapter contributes a constant w.r.t. LoRA.
	constant := fusedFeatures(t, f, featureTargets()[:1], 1)
	if constant.FeatureLoss <= 0 || !reflect.DeepEqual(constant.Gradients, baseline.Gradients) {
		t.Fatal("feature before the first adapter has an incorrect parameter derivative")
	}
	scaled := fusedFeatures(t, f, intermediate, 2)
	if scaled.Loss != result.Loss || scaled.FeatureLoss != result.FeatureLoss {
		t.Fatal("loss scale changed a reported loss")
	}
	for i := range result.Gradients {
		for j, value := range result.Gradients[i].ValuesF32 {
			if math.Abs(float64(scaled.Gradients[i].ValuesF32[j])-2*float64(value)) > 1e-6 {
				t.Fatal("loss scale did not scale both cotangents")
			}
		}
	}
}

func TestFeatureFusionCombinesIndependentTeacherLogitsAndFeatures(t *testing.T) {
	f := featureFixture(t, torch.Float32)
	teachers := []decoder.FusionTeacher{{Name: "logit-teacher", Weight: 0.4, Positions: []decoder.FusionTeacherPosition{
		{RetainedMass: 0.7, TopK: []decoder.FusionTokenProbability{{TokenID: 2, Probability: 0.3}, {TokenID: 4, Probability: 0.4}}},
		{RetainedMass: 0.8, TopK: []decoder.FusionTokenProbability{{TokenID: 1, Probability: 0.3}, {TokenID: 3, Probability: 0.5}}},
	}}}
	features := featureTargets()
	baseline, err := decoder.FusionCompletionGradient(context.Background(), f.model, f.tokens, 1, f.limits, 1, teachers)
	if err != nil {
		t.Fatal(err)
	}
	combined, err := decoder.FusionCompletionGradientWithFeatures(context.Background(), f.model, f.tokens, 1, f.limits, 1, teachers, features)
	if err != nil {
		t.Fatal(err)
	}
	hard := fusedFeatures(t, f, nil, 1)
	withFeatures := fusedFeatures(t, f, features, 1)
	if math.Abs(combined.Loss-baseline.Loss-withFeatures.FeatureLoss) > 1e-12 ||
		!reflect.DeepEqual(combined.TeacherLosses, baseline.TeacherLosses) || combined.HardLoss != baseline.HardLoss {
		t.Fatal("feature supervision changed the teacher or hard-label objective")
	}
	for i := range combined.Gradients {
		for j, value := range combined.Gradients[i].ValuesF32 {
			featureDelta := float64(withFeatures.Gradients[i].ValuesF32[j]) - float64(hard.Gradients[i].ValuesF32[j])
			want := float64(baseline.Gradients[i].ValuesF32[j]) + featureDelta
			if math.Abs(float64(value)-want) > 1e-6 {
				t.Fatal("teacher-logit and feature derivatives are not additive")
			}
		}
	}
}

func TestFeatureTargetsRejectMalformedDataAndPreserveSnapshot(t *testing.T) {
	f := featureFixture(t, torch.Float32)
	snapshot := forward(t, f)
	seed := tensor(t, values(18, 0.3, 0.7), []int64{1, 3, 6}, false)
	valid := featureTargets()[1]
	for name, mutate := range map[string]func(*decoder.FeatureTarget){
		"missing source":     func(x *decoder.FeatureTarget) { x.Source = " " },
		"negative layer":     func(x *decoder.FeatureTarget) { x.Layer = -1 },
		"past final layer":   func(x *decoder.FeatureTarget) { x.Layer = 8 },
		"negative position":  func(x *decoder.FeatureTarget) { x.Position = -1 },
		"past final token":   func(x *decoder.FeatureTarget) { x.Position = 3 },
		"wrong hidden width": func(x *decoder.FeatureTarget) { x.Values = []float32{1} },
		"negative weight":    func(x *decoder.FeatureTarget) { x.Weight = -1 },
		"NaN weight":         func(x *decoder.FeatureTarget) { x.Weight = math.NaN() },
		"infinite weight":    func(x *decoder.FeatureTarget) { x.Weight = math.Inf(1) },
		"nonfinite values":   func(x *decoder.FeatureTarget) { x.Values = []float32{1, 2, float32(math.Inf(1)), 4} },
		"overflow":           func(x *decoder.FeatureTarget) { x.Weight = math.MaxFloat64 },
	} {
		t.Run(name, func(t *testing.T) {
			bad := valid
			mutate(&bad)
			gradients, loss, err := f.model.VJPWithFeatures(context.Background(), snapshot, seed, []decoder.FeatureTarget{bad}, 1)
			_ = gradients.Close()
			if !errors.Is(err, decoder.ErrFeatureTarget) || gradients != nil || loss != 0 {
				t.Fatalf("invalid target returned gradients=%d loss=%g err=%v", len(gradients), loss, err)
			}
		})
	}
	if gradients, loss, err := f.model.VJPWithFeatures(context.Background(), snapshot, seed, []decoder.FeatureTarget{valid, valid}, 1); gradients != nil || loss != 0 || !errors.Is(err, decoder.ErrFeatureTarget) {
		_ = gradients.Close()
		t.Fatalf("duplicate target accepted: %v", err)
	}
	for _, scale := range []float64{0, -1, math.NaN(), math.Inf(1)} {
		if gradients, loss, err := f.model.VJPWithFeatures(context.Background(), snapshot, seed, nil, scale); gradients != nil || loss != 0 || !errors.Is(err, decoder.ErrFeatureTarget) {
			_ = gradients.Close()
			t.Fatalf("invalid scale accepted: %v", err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if gradients, _, err := f.model.VJPWithFeatures(ctx, snapshot, seed, []decoder.FeatureTarget{valid}, 1); gradients != nil || !errors.Is(err, context.Canceled) {
		_ = gradients.Close()
		t.Fatalf("canceled feature backward accepted: %v", err)
	}
	foreign := *f.model
	if gradients, _, err := foreign.VJPWithFeatures(context.Background(), snapshot, seed, nil, 1); gradients != nil || err == nil {
		_ = gradients.Close()
		t.Fatal("foreign snapshot accepted")
	}
	// None of the errors consumes the snapshot. Repeated valid calls are equal.
	var previous []float32
	for i := 0; i < 3; i++ {
		gradients, loss, err := f.model.VJPWithFeatures(context.Background(), snapshot, seed, []decoder.FeatureTarget{valid}, 1)
		if err != nil || loss <= 0 {
			t.Fatalf("valid backward after failure: %v", err)
		}
		got := read(t, gradients[0].Value)
		_ = gradients.Close()
		if i > 0 && !reflect.DeepEqual(got, previous) {
			t.Fatal("repeated feature VJP changed a result")
		}
		previous = got
	}
	_ = snapshot.Close()
	if gradients, _, err := f.model.VJPWithFeatures(context.Background(), snapshot, seed, nil, 1); gradients != nil || err == nil {
		_ = gradients.Close()
		t.Fatal("closed snapshot accepted")
	}
}

func TestFeatureFusionHalfStorageAndGradientSteps(t *testing.T) {
	for _, storage := range []torch.DType{torch.Float32, torch.Float16} {
		f := featureFixture(t, storage)
		features := featureTargets()[1:]
		before := baseHash(t, f)
		first := fusedFeatures(t, f, features, 1)
		if math.Abs(first.FeatureLoss-manualFeatureLoss(t, f, features)) > 1e-12 {
			t.Fatal("feature loss changed under boundary storage conversion")
		}
		pointer := parameter(f.model.Layers[3].Adapter, 2)
		original := *pointer
		info, _ := original.Info()
		baseline := read(t, original)
		losses := [3]float64{}
		for i, sign := range []float32{1, -1, 0} {
			changed := append([]float32(nil), baseline...)
			for j := range changed {
				changed[j] += sign * 0.1 * first.Gradients[2].ValuesF32[j]
			}
			replacement, err := torch.FromFloat32(changed, info.Shape, info.Device, true)
			if err != nil {
				t.Fatal(err)
			}
			*pointer = replacement
			losses[i] = fusedFeatures(t, f, features, 1).Loss
			*pointer = original
			_ = replacement.Close()
		}
		if !(losses[1] < first.Loss && losses[0] > first.Loss) || losses[2] != first.Loss {
			t.Fatalf("storage=%v gradient direction losses=%v baseline=%g", storage, losses, first.Loss)
		}
		// Three explicit gradient updates must change the measurement each time.
		prior := first.Loss
		for step := 0; step < 3; step++ {
			gradient := fusedFeatures(t, f, features, 1)
			changed := read(t, *pointer)
			for j := range changed {
				changed[j] -= 0.1 * gradient.Gradients[2].ValuesF32[j]
			}
			replacement, err := torch.FromFloat32(changed, info.Shape, info.Device, true)
			if err != nil {
				t.Fatal(err)
			}
			if *pointer != original {
				_ = (*pointer).Close()
			}
			*pointer = replacement
			current := fusedFeatures(t, f, features, 1).Loss
			if current >= prior {
				t.Fatalf("storage=%v step=%d loss=%g previous=%g", storage, step, current, prior)
			}
			t.Logf("storage=%v step=%d loss=%.12g", storage, step+1, current)
			prior = current
		}
		_ = (*pointer).Close()
		*pointer = original
		if baseHash(t, f) != before || fusedFeatures(t, f, features, 1).Loss != first.Loss {
			t.Fatal("gradient probes did not restore the fixture")
		}
	}
}

func TestFeatureVJPRejectsStaleParameterGeneration(t *testing.T) {
	f := newFixture(t, torch.Float32)
	snapshot := forward(t, f)
	seed := tensor(t, f.seed, []int64{1, 2, 6}, false)
	initial, err := decoder.InitializeAdapter(context.Background(), testfixture.Spec(t))
	if err != nil {
		t.Fatal(err)
	}
	defer initial.Close()
	loaded := &decoder.LoadedTextModel{Model: f.model, Parameters: append([]decoder.InitialParameter(nil), initial.Parameters...)}
	defer loaded.Close()
	tinyLayers := append([]decoder.Layer(nil), f.model.Layers...)
	var all []float32
	for i, p := range initial.Parameters {
		all = append(all, read(t, p.Value)...)
		if i%4 == 0 {
			parameters := initial.Parameters[i : i+4]
			f.model.Layers[3+i].Adapter = &layers.AttentionLoRA{QueryA: parameters[0].Value, QueryB: parameters[1].Value,
				ValueA: parameters[2].Value, ValueB: parameters[3].Value, Alpha: 8}
		}
	}
	if _, err := loaded.ReplaceParameters(context.Background(), all); err != nil {
		t.Fatal(err)
	}
	f.model.Layers = tinyLayers
	gradients, loss, err := f.model.VJPWithFeatures(context.Background(), snapshot, seed, featureTargets(), 1)
	_ = gradients.Close()
	if !errors.Is(err, decoder.ErrStaleSnapshot) || gradients != nil || loss != 0 {
		t.Fatalf("stale feature snapshot accepted: %v", err)
	}
}
