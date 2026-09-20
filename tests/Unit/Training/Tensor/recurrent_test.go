//go:build libtorch && cgo

package tensor_test

import (
	"context"
	"errors"
	"math"
	"slices"
	"testing"

	"github.com/tayi-ai/arandu-llama/training"
	"github.com/tayi-ai/arandu-llama/training/tensor"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

func TestForwardMatchesIndependentScalarReference(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 2, Tokens: 3, Heads: 2, KeyDim: 3, ValueDim: 2}
	input, _, _ := fixture(g)
	expected, err := training.GatedDeltaForward(context.Background(), g, input, training.DefaultRecurrentOptions())
	if err != nil {
		t.Fatal(err)
	}
	native := nativeInput(t, g, input, true)
	output, err := tensor.Forward(context.Background(), native, tensor.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = output.Close() })
	closeValues(t, "values", values(t, output.Values), expected.Values, 2e-12)
	closeValues(t, "state", values(t, output.FinalState), expected.FinalState, 2e-12)
	for _, result := range []*torch.Tensor{output.Values, output.FinalState} {
		info, err := result.Info()
		if err != nil || info.RequiresGrad {
			t.Fatalf("forward retained an autograd graph: info=%+v err=%v", info, err)
		}
	}
	for _, input := range tensors(native) {
		info, err := input.Info()
		if err != nil || !info.RequiresGrad {
			t.Fatalf("forward mutated the caller's requires-grad flag: info=%+v err=%v", info, err)
		}
	}
}

func TestVJPMatchesReferenceWithInitialAndFinalState(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 2, Tokens: 2, Heads: 2, KeyDim: 2, ValueDim: 3}
	input, dValues, dState := fixture(g)
	expected, err := training.GatedDeltaVJP(context.Background(), g, input, dValues, dState, training.DefaultRecurrentOptions())
	if err != nil {
		t.Fatal(err)
	}
	native := nativeInput(t, g, input, false)
	seeds := nativeSeeds(t, g, dValues, dState)
	actual, err := tensor.GradVJP(context.Background(), native, seeds[0], seeds[1], tensor.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = actual.Close() })
	for i, gradient := range gradientTensors(actual) {
		closeValues(t, fieldNames[i], values(t, gradient), vectors(expected.Gradients)[i], 3e-12)
		info, err := gradient.Info()
		if err != nil || info.RequiresGrad {
			t.Fatalf("gradient retained a higher-order graph: info=%+v err=%v", info, err)
		}
	}
	if !anyNonzero(values(t, actual.InitialState)) {
		t.Fatal("nonzero initial/final state fixture produced no initial-state gradient")
	}
	for i, current := range tensors(native) {
		closeValues(t, "unchanged "+fieldNames[i], values(t, current), vectors(input)[i], 0)
		info, err := current.Info()
		if err != nil || info.RequiresGrad {
			t.Fatalf("VJP mutated the caller's requires-grad flag: info=%+v err=%v", info, err)
		}
	}
}

func TestTwoTokenVJPMatchesFiniteDifferences(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 1, Tokens: 2, Heads: 1, KeyDim: 2, ValueDim: 2}
	input, dValues, dState := fixture(g)
	native := nativeInput(t, g, input, false)
	seeds := nativeSeeds(t, g, dValues, dState)
	actual, err := tensor.GradVJP(context.Background(), native, seeds[0], seeds[1], tensor.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = actual.Close() })
	const epsilon = 1e-6
	for field, gradient := range gradientTensors(actual) {
		got := values(t, gradient)
		for index := range got {
			plus, minus := copyInput(input), copyInput(input)
			vectors(plus)[field][index] += epsilon
			vectors(minus)[field][index] -= epsilon
			want := (objective(t, g, plus, dValues, dState) - objective(t, g, minus, dValues, dState)) / (2 * epsilon)
			if difference := math.Abs(got[index] - want); difference > 2e-8*(1+math.Abs(want)) {
				t.Fatalf("%s[%d]: VJP %.16g finite difference %.16g", fieldNames[field], index, got[index], want)
			}
		}
	}
}

func TestOneTokenPreservesExplicitTransformsAndSeeds(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 1, Tokens: 1, Heads: 1, KeyDim: 1, ValueDim: 1}
	input := training.RecurrentInput{
		Query: []float64{2}, Key: []float64{3}, Value: []float64{5},
		LogDecay: []float64{math.Log(.5)}, Beta: []float64{.2}, InitialState: []float64{4},
	}
	native := nativeInput(t, g, input, false)
	output, err := tensor.Forward(context.Background(), native, tensor.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = output.Close() })
	// P=2, e=-1, u=-.2, S=1.4 and o=2.8. q/k/beta must not be normalized
	// or activated again inside this operator.
	closeValues(t, "value", values(t, output.Values), []float64{2.8}, 2e-14)
	closeValues(t, "state", values(t, output.FinalState), []float64{1.4}, 2e-14)
	seeds := nativeSeeds(t, g, []float64{.7}, []float64{-.2})
	gradient, err := tensor.GradVJP(context.Background(), native, seeds[0], seeds[1], tensor.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = gradient.Close() })
	want := []float64{.98, -1.68, .72, -1.92, -3.6, -.48}
	for i, current := range gradientTensors(gradient) {
		closeValues(t, fieldNames[i], values(t, current), []float64{want[i]}, 2e-14)
	}
}

func TestChunkBoundaryCarriesTheCompleteStateAdjoint(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 2, Tokens: 2, Heads: 2, KeyDim: 2, ValueDim: 3}
	input, dValues, dState := fixture(g)
	native := nativeInput(t, g, input, true)
	seeds := nativeSeeds(t, g, dValues, dState)
	whole, err := tensor.GradVJP(context.Background(), native, seeds[0], seeds[1], tensor.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = whole.Close() })
	first := tokenInput(t, native, 0, native.InitialState)
	firstOutput, err := tensor.Forward(context.Background(), first, tensor.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = firstOutput.Close() })
	second := tokenInput(t, native, 1, firstOutput.FinalState)
	secondSeed := own(t)(seeds[0].Slice(1, 1, 2, 1))
	secondGradient, err := tensor.GradVJP(context.Background(), second, secondSeed, seeds[1], tensor.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = secondGradient.Close() })
	firstSeed := own(t)(seeds[0].Slice(1, 0, 1, 1))
	firstGradient, err := tensor.GradVJP(context.Background(), first, firstSeed, secondGradient.InitialState, tensor.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = firstGradient.Close() })
	firstFields, secondFields, wholeFields := gradientTensors(firstGradient), gradientTensors(secondGradient), gradientTensors(whole)
	for i := 0; i < 5; i++ {
		combined := own(t)(torch.Cat([]*torch.Tensor{firstFields[i], secondFields[i]}, 1))
		closeValues(t, "chunked "+fieldNames[i], values(t, combined), values(t, wholeFields[i]), 3e-12)
	}
	closeValues(t, "chunked initial state", values(t, firstGradient.InitialState), values(t, whole.InitialState), 3e-12)
}

func TestInvalidInputsLimitsAndCancellationReturnNoPartialResult(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 1, Tokens: 2, Heads: 1, KeyDim: 2, ValueDim: 2}
	input, dValues, dState := fixture(g)
	tests := []struct {
		name   string
		change func(*testing.T, *tensor.Input, *tensor.Limits)
		want   error
	}{
		{"nil query", func(_ *testing.T, in *tensor.Input, _ *tensor.Limits) { in.Query = nil }, tensor.ErrInvalidInput},
		{"closed key", func(_ *testing.T, in *tensor.Input, _ *tensor.Limits) { _ = in.Key.Close() }, tensor.ErrInvalidInput},
		{"wrong value shape", func(t *testing.T, in *tensor.Input, _ *tensor.Limits) {
			in.Value = own(t)(in.Value.Reshape([]int64{1, 1, 2, 2}))
		}, tensor.ErrInvalidInput},
		{"mixed dtype", func(t *testing.T, in *tensor.Input, _ *tensor.Limits) {
			in.Key = own(t)(in.Key.To(torch.CPUDevice(), torch.Float32))
		}, tensor.ErrInvalidInput},
		{"token limit", func(_ *testing.T, _ *tensor.Input, limits *tensor.Limits) { limits.Tokens = 1 }, tensor.ErrLimit},
		{"working limit", func(_ *testing.T, _ *tensor.Input, limits *tensor.Limits) { limits.WorkingElements = 1 }, tensor.ErrLimit},
		{"unset limits", func(_ *testing.T, _ *tensor.Input, limits *tensor.Limits) { *limits = tensor.Limits{} }, tensor.ErrLimit},
		{"absolute limit", func(_ *testing.T, _ *tensor.Input, limits *tensor.Limits) { limits.Tokens = tensor.MaxChunkTokens + 1 }, tensor.ErrLimit},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			native := nativeInput(t, g, input, false)
			limits := tensor.DefaultLimits()
			test.change(t, &native, &limits)
			output, err := tensor.Forward(context.Background(), native, limits)
			if output != nil || !errors.Is(err, test.want) {
				t.Fatalf("output=%v error=%v, want %v", output, err, test.want)
			}
		})
	}
	native := nativeInput(t, g, input, false)
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if output, err := tensor.Forward(ctx, native, tensor.DefaultLimits()); output != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled forward output=%v err=%v", output, err)
	}
	if output, err := tensor.Forward(nil, native, tensor.DefaultLimits()); output != nil || !errors.Is(err, tensor.ErrInvalidInput) {
		t.Fatalf("nil-context forward output=%v err=%v", output, err)
	}
	seeds := nativeSeeds(t, g, dValues, dState)
	for _, pair := range [][2]*torch.Tensor{{nil, seeds[1]}, {seeds[0], nil}, {seeds[1], seeds[0]}} {
		gradient, err := tensor.GradVJP(context.Background(), native, pair[0], pair[1], tensor.DefaultLimits())
		if gradient != nil || !errors.Is(err, tensor.ErrInvalidInput) {
			t.Fatalf("invalid cotangents gradient=%v err=%v", gradient, err)
		}
	}
	g.Tokens = 0
	empty, _, _ := fixture(g)
	if output, err := tensor.Forward(context.Background(), nativeInput(t, g, empty, false), tensor.DefaultLimits()); output != nil || !errors.Is(err, tensor.ErrInvalidInput) {
		t.Fatalf("empty chunk output=%v err=%v", output, err)
	}
}

func TestNonFiniteInputsResultsAndGradientsAreRejected(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 1, Tokens: 1, Heads: 1, KeyDim: 1, ValueDim: 1}
	input, dValues, dState := fixture(g)
	for _, value := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		bad := copyInput(input)
		bad.Value[0] = value
		output, err := tensor.Forward(context.Background(), nativeInput(t, g, bad, false), tensor.DefaultLimits())
		if output != nil || !errors.Is(err, tensor.ErrNonFinite) {
			t.Fatalf("non-finite input output=%v err=%v", output, err)
		}
	}
	overflow := copyInput(input)
	overflow.LogDecay[0] = 1000
	if output, err := tensor.Forward(context.Background(), nativeInput(t, g, overflow, false), tensor.DefaultLimits()); output != nil || !errors.Is(err, tensor.ErrNonFinite) {
		t.Fatalf("overflow output=%v err=%v", output, err)
	}
	dValues[0] = math.NaN()
	seeds := nativeSeeds(t, g, dValues, dState)
	if gradient, err := tensor.GradVJP(context.Background(), nativeInput(t, g, input, false), seeds[0], seeds[1], tensor.DefaultLimits()); gradient != nil || !errors.Is(err, tensor.ErrNonFinite) {
		t.Fatalf("non-finite seed gradient=%v err=%v", gradient, err)
	}
	// Forward stays finite; its VJP overflows because q*dOutput exceeds F64.
	large := training.RecurrentInput{
		Query: []float64{1e308}, Key: []float64{0}, Value: []float64{0},
		LogDecay: []float64{0}, Beta: []float64{.5}, InitialState: []float64{1e-308},
	}
	native := nativeInput(t, g, large, false)
	output, err := tensor.Forward(context.Background(), native, tensor.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	_ = output.Close()
	seeds = nativeSeeds(t, g, []float64{1e308}, []float64{0})
	if gradient, err := tensor.GradVJP(context.Background(), native, seeds[0], seeds[1], tensor.DefaultLimits()); gradient != nil || !errors.Is(err, tensor.ErrNonFinite) {
		t.Fatalf("overflow gradient=%v err=%v", gradient, err)
	}
}

var fieldNames = []string{"query", "key", "value", "log decay", "beta", "initial state"}

func fixture(g training.RecurrentGeometry) (training.RecurrentInput, []float64, []float64) {
	rows := g.Batch * g.Tokens * g.Heads
	input := training.RecurrentInput{
		Query: make([]float64, rows*g.KeyDim), Key: make([]float64, rows*g.KeyDim), Value: make([]float64, rows*g.ValueDim),
		LogDecay: make([]float64, rows), Beta: make([]float64, rows), InitialState: make([]float64, g.Batch*g.Heads*g.KeyDim*g.ValueDim),
	}
	for i := range input.Query {
		input.Query[i], input.Key[i] = math.Sin(float64(i)+.2), math.Cos(float64(i)+.7)
	}
	for row := 0; row < rows; row++ {
		qNorm, kNorm := 0.0, 0.0
		for k := 0; k < g.KeyDim; k++ {
			index := row*g.KeyDim + k
			qNorm += input.Query[index] * input.Query[index]
			kNorm += input.Key[index] * input.Key[index]
		}
		for k := 0; k < g.KeyDim; k++ {
			index := row*g.KeyDim + k
			input.Query[index] /= math.Sqrt(qNorm * float64(g.KeyDim))
			input.Key[index] /= math.Sqrt(kNorm)
		}
		input.LogDecay[row], input.Beta[row] = -.1-.03*float64(row%3), .3+.1*float64(row%4)
	}
	for i := range input.Value {
		input.Value[i] = .1 + .09*float64(i%7-3)
	}
	for i := range input.InitialState {
		input.InitialState[i] = .01 + .05*float64(i%7-3)
	}
	dValues, dState := make([]float64, len(input.Value)), make([]float64, len(input.InitialState))
	for i := range dValues {
		dValues[i] = .1 + .07*float64(i%5)
	}
	for i := range dState {
		dState[i] = -.08 + .03*float64(i%7)
	}
	return input, dValues, dState
}

func nativeInput(t *testing.T, g training.RecurrentGeometry, input training.RecurrentInput, requiresGrad bool) tensor.Input {
	t.Helper()
	b, n, h, k, v := int64(g.Batch), int64(g.Tokens), int64(g.Heads), int64(g.KeyDim), int64(g.ValueDim)
	shapes := [][]int64{{b, n, h, k}, {b, n, h, k}, {b, n, h, v}, {b, n, h}, {b, n, h}, {b, h, k, v}}
	var current []*torch.Tensor
	for i, vector := range vectors(input) {
		current = append(current, own(t)(torch.FromFloat64(vector, shapes[i], torch.CPUDevice(), requiresGrad)))
	}
	return tensor.Input{Query: current[0], Key: current[1], Value: current[2], LogDecay: current[3], Beta: current[4], InitialState: current[5]}
}

func nativeSeeds(t *testing.T, g training.RecurrentGeometry, output, state []float64) [2]*torch.Tensor {
	t.Helper()
	return [2]*torch.Tensor{
		own(t)(torch.FromFloat64(output, []int64{int64(g.Batch), int64(g.Tokens), int64(g.Heads), int64(g.ValueDim)}, torch.CPUDevice(), false)),
		own(t)(torch.FromFloat64(state, []int64{int64(g.Batch), int64(g.Heads), int64(g.KeyDim), int64(g.ValueDim)}, torch.CPUDevice(), false)),
	}
}

func own(t *testing.T) func(*torch.Tensor, error) *torch.Tensor {
	return func(current *torch.Tensor, err error) *torch.Tensor {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() {
			if err := current.Close(); err != nil {
				t.Error(err)
			}
		})
		return current
	}
}

func values(t *testing.T, current *torch.Tensor) []float64 {
	t.Helper()
	result, err := current.Float64Values()
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func tensors(input tensor.Input) []*torch.Tensor {
	return []*torch.Tensor{input.Query, input.Key, input.Value, input.LogDecay, input.Beta, input.InitialState}
}

func gradientTensors(gradient *tensor.Gradients) []*torch.Tensor {
	return tensors(tensor.Input(*gradient))
}

func vectors(input training.RecurrentInput) [][]float64 {
	return [][]float64{input.Query, input.Key, input.Value, input.LogDecay, input.Beta, input.InitialState}
}

func copyInput(input training.RecurrentInput) training.RecurrentInput {
	return training.RecurrentInput{
		Query: slices.Clone(input.Query), Key: slices.Clone(input.Key), Value: slices.Clone(input.Value),
		LogDecay: slices.Clone(input.LogDecay), Beta: slices.Clone(input.Beta), InitialState: slices.Clone(input.InitialState),
	}
}

func objective(t *testing.T, g training.RecurrentGeometry, input training.RecurrentInput, dValues, dState []float64) float64 {
	t.Helper()
	output, err := tensor.Forward(context.Background(), nativeInput(t, g, input, false), tensor.DefaultLimits())
	if err != nil {
		t.Fatal(err)
	}
	defer output.Close()
	result := 0.0
	for i, value := range values(t, output.Values) {
		result += value * dValues[i]
	}
	for i, value := range values(t, output.FinalState) {
		result += value * dState[i]
	}
	return result
}

func tokenInput(t *testing.T, input tensor.Input, token int64, state *torch.Tensor) tensor.Input {
	t.Helper()
	var current []*torch.Tensor
	for _, value := range tensors(input)[:5] {
		current = append(current, own(t)(value.Slice(1, token, token+1, 1)))
	}
	return tensor.Input{Query: current[0], Key: current[1], Value: current[2], LogDecay: current[3], Beta: current[4], InitialState: state}
}

func closeValues(t *testing.T, name string, got, want []float64, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s length got %d want %d", name, len(got), len(want))
	}
	for i := range got {
		if math.IsNaN(got[i]) || math.IsInf(got[i], 0) || math.Abs(got[i]-want[i]) > tolerance*(1+math.Abs(want[i])) {
			t.Fatalf("%s[%d] got %.17g want %.17g", name, i, got[i], want[i])
		}
	}
}

func anyNonzero(values []float64) bool {
	for _, value := range values {
		if value != 0 {
			return true
		}
	}
	return false
}
