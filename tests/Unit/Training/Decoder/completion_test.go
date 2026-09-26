//go:build libtorch && cgo

package decoder_test

import (
	"context"
	"errors"
	"math"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

func manualCompletionLoss(t *testing.T, f *fixture, prompt int, limits decoder.Limits) float64 {
	t.Helper()
	snapshot, err := f.model.Forward(context.Background(), f.tokens, limits)
	if err != nil {
		t.Fatal(err)
	}
	defer snapshot.Close()
	info, err := snapshot.Logits.Info()
	if err != nil {
		t.Fatal(err)
	}
	values := read(t, snapshot.Logits)
	vocabulary := int(info.Shape[2])
	targets := len(f.tokens) - prompt
	total := 0.0
	for row := 0; row < targets; row++ {
		offset := row * vocabulary
		maximum := float64(values[offset])
		for column := 1; column < vocabulary; column++ {
			maximum = math.Max(maximum, float64(values[offset+column]))
		}
		sum := 0.0
		for column := 0; column < vocabulary; column++ {
			sum += math.Exp(float64(values[offset+column]) - maximum)
		}
		target := int(f.tokens[prompt+row])
		total += maximum + math.Log(sum) - float64(values[offset+target])
	}
	return total / float64(targets)
}

func TestCompletionGradientMatchesShiftedMeanCrossEntropy(t *testing.T) {
	f := newFixture(t, torch.Float32)
	before := baseHash(t, f)
	limits := f.limits
	limits.LogitRows = 3
	want := manualCompletionLoss(t, f, 1, limits)

	result, err := decoder.CompletionGradient(context.Background(), f.model, f.tokens, 1, limits, 1)
	if err != nil {
		t.Fatal(err)
	}
	if result.Tokens != 2 || len(result.Gradients) != 32 {
		t.Fatalf("completion result tokens=%d gradients=%d", result.Tokens, len(result.Gradients))
	}
	if math.Abs(result.Loss-want) > 1e-10 {
		t.Fatalf("completion loss %.12g != manual %.12g", result.Loss, want)
	}
	norm := 0.0
	for _, gradient := range result.Gradients {
		for _, value := range gradient.ValuesF32 {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				t.Fatal("nonfinite completion gradient")
			}
			norm += math.Abs(float64(value))
		}
	}
	if norm <= 1e-8 {
		t.Fatal("completion gradient vanished")
	}
	if baseHash(t, f) != before {
		t.Fatal("completion gradient changed frozen base")
	}
	scaled, err := decoder.CompletionGradient(context.Background(), f.model, f.tokens, 1, limits, 2)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(scaled.Loss-result.Loss) > 1e-12 {
		t.Fatal("loss scale changed reported loss")
	}
	for i := range result.Gradients {
		if result.Gradients[i].Name != scaled.Gradients[i].Name {
			t.Fatal("loss scale changed gradient ordering")
		}
		for j, value := range result.Gradients[i].ValuesF32 {
			want := 2 * float64(value)
			got := float64(scaled.Gradients[i].ValuesF32[j])
			if math.Abs(got-want) > 1e-6+1e-5*math.Abs(want) {
				t.Fatalf("scaled gradient %d/%d %.9g != %.9g", i, j, got, want)
			}
		}
	}
	gradient := result.Gradients[0].ValuesF32
	coordinate := 0
	for i := range gradient {
		if math.Abs(float64(gradient[i])) > math.Abs(float64(gradient[coordinate])) {
			coordinate = i
		}
	}
	pointer := parameter(f.model.Layers[3].Adapter, 0)
	original := *pointer
	info, err := original.Info()
	if err != nil {
		t.Fatal(err)
	}
	baseline := read(t, original)
	step := float32(0.005)
	losses := [2]float64{}
	positions := [2]float32{}
	for direction, sign := range []float32{1, -1} {
		perturbed := append([]float32(nil), baseline...)
		perturbed[coordinate] += sign * step
		positions[direction] = perturbed[coordinate]
		replacement, err := torch.FromFloat32(perturbed, info.Shape, torch.CPUDevice(), true)
		if err != nil {
			t.Fatal(err)
		}
		*pointer = replacement
		losses[direction] = manualCompletionLoss(t, f, 1, limits)
		*pointer = original
		if err := replacement.Close(); err != nil {
			t.Fatal(err)
		}
	}
	finiteDifference := (losses[0] - losses[1]) / float64(positions[0]-positions[1])
	analytic := float64(gradient[coordinate])
	tolerance := 3e-5 + 0.003*math.Abs(analytic)
	if math.Abs(analytic-finiteDifference) > tolerance {
		t.Fatalf("completion derivative coordinate=%d analytic=%.9g central=%.9g tolerance=%.9g",
			coordinate, analytic, finiteDifference, tolerance)
	}
	if baseHash(t, f) != before {
		t.Fatal("completion finite difference changed frozen base")
	}
}

func TestCompletionGradientRejectsInvalidMaskScaleAndCancellation(t *testing.T) {
	f := newFixture(t, torch.Float32)
	limits := f.limits
	limits.LogitRows = 3
	for name, call := range map[string]func() error{
		"empty prompt": func() error {
			_, err := decoder.CompletionGradient(context.Background(), f.model, f.tokens, 0, limits, 1)
			return err
		},
		"empty completion": func() error {
			_, err := decoder.CompletionGradient(context.Background(), f.model, f.tokens, len(f.tokens), limits, 1)
			return err
		},
		"wrong logit rows": func() error {
			wrong := limits
			wrong.LogitRows = 2
			_, err := decoder.CompletionGradient(context.Background(), f.model, f.tokens, 1, wrong, 1)
			return err
		},
		"zero loss scale": func() error {
			_, err := decoder.CompletionGradient(context.Background(), f.model, f.tokens, 1, limits, 0)
			return err
		},
	} {
		t.Run(name, func(t *testing.T) {
			if err := call(); err == nil {
				t.Fatal("invalid completion request accepted")
			}
		})
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := decoder.CompletionGradient(ctx, f.model, f.tokens, 1, limits, 1)
	if !errors.Is(err, context.Canceled) || result.Tokens != 0 || len(result.Gradients) != 0 {
		t.Fatalf("canceled completion result=%+v err=%v", result, err)
	}
}
