//go:build libtorch && cgo

package sequence_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"
	"testing"

	"github.com/tayi-ai/arandu-llama/training"
	"github.com/tayi-ai/arandu-llama/training/sequence"
	"github.com/tayi-ai/arandu-llama/training/tensor"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

func TestSequenceChunksMatchIndependentReference(t *testing.T) {
	for _, tokens := range []int{2, 5} {
		for _, chunk := range []int64{1, 2, 3} {
			for _, dtype := range []torch.DType{torch.Float32, torch.Float64} {
				t.Run(fmt.Sprintf("tokens_%d_chunk_%d_dtype_%d", tokens, chunk, dtype), func(t *testing.T) {
					g := training.RecurrentGeometry{Batch: 2, Tokens: tokens, Heads: 2, KeyDim: 3, ValueDim: 2}
					compareReference(t, g, chunk, dtype)
				})
			}
		}
	}
}

func TestSequenceLongerThanTheNativeChunkLimit(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 1, Tokens: 35, Heads: 1, KeyDim: 2, ValueDim: 2}
	compareReference(t, g, sequence.DefaultLimits().ChunkTokens, torch.Float64)
}

func TestSequence1128TokenFloat32ForwardMatchesReference(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 1, Tokens: 1128, Heads: 2, KeyDim: 4, ValueDim: 4}
	input, _, _ := fixture(g, torch.Float32)
	expected, err := training.GatedDeltaForward(context.Background(), g, input, training.DefaultRecurrentOptions())
	if err != nil {
		t.Fatal(err)
	}
	native := nativeInput(t, g, input, torch.Float32, false)
	limits := sequence.DefaultLimits()
	output, err := sequence.Forward(context.Background(), native, limits)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = output.Close() })
	closeValues(t, "1128-token output", read(t, output.Values), expected.Values, 6e-5)
	closeValues(t, "1128-token state", read(t, output.FinalState), expected.FinalState, 6e-5)
	for _, value := range []*torch.Tensor{output.Values, output.FinalState} {
		finite, err := value.AllFinite()
		if err != nil || !finite {
			t.Fatalf("1128-token forward is non-finite: finite=%t err=%v", finite, err)
		}
	}
}

func compareReference(t *testing.T, g training.RecurrentGeometry, chunk int64, dtype torch.DType) {
	t.Helper()
	input, dValues, dState := fixture(g, dtype)
	expected, err := training.GatedDeltaVJP(context.Background(), g, input, dValues, dState, training.DefaultRecurrentOptions())
	if err != nil {
		t.Fatal(err)
	}
	native := nativeInput(t, g, input, dtype, true)
	seeds := nativeSeeds(t, g, dValues, dState, dtype, true)
	limits := sequence.DefaultLimits()
	limits.ChunkTokens = chunk
	output, err := sequence.Forward(context.Background(), native, limits)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = output.Close() })
	backward, err := sequence.GradVJP(context.Background(), native, seeds[0], seeds[1], limits)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = backward.Close() })
	tolerance := 3e-12
	if dtype == torch.Float32 {
		tolerance = 4e-6
	}
	closeValues(t, "forward output", read(t, output.Values), expected.Output.Values, tolerance)
	closeValues(t, "forward state", read(t, output.FinalState), expected.Output.FinalState, tolerance)
	closeValues(t, "VJP output", read(t, backward.Output.Values), expected.Output.Values, tolerance)
	closeValues(t, "VJP state", read(t, backward.Output.FinalState), expected.Output.FinalState, tolerance)
	for i, gradient := range fields(tensor.Input(*backward.Gradients)) {
		closeValues(t, fieldNames[i], read(t, gradient), vectors(expected.Gradients)[i], tolerance)
		info, err := gradient.Info()
		if err != nil || info.RequiresGrad || info.DType != dtype {
			t.Fatalf("gradient graph/dtype mismatch: %+v %v", info, err)
		}
	}
	for _, value := range []*torch.Tensor{output.Values, output.FinalState, backward.Output.Values, backward.Output.FinalState} {
		info, err := value.Info()
		if err != nil || info.RequiresGrad || info.DType != dtype {
			t.Fatalf("output graph/dtype mismatch: %+v %v", info, err)
		}
	}
	for i, value := range fields(native) {
		closeValues(t, "unchanged "+fieldNames[i], read(t, value), vectors(input)[i], 0)
		info, err := value.Info()
		if err != nil || !info.RequiresGrad {
			t.Fatalf("input requires-grad flag changed: %+v %v", info, err)
		}
	}
	closeValues(t, "unchanged output seed", read(t, seeds[0]), dValues, 0)
	closeValues(t, "unchanged final seed", read(t, seeds[1]), dState, 0)
	for _, seed := range seeds {
		info, err := seed.Info()
		if err != nil || !info.RequiresGrad {
			t.Fatalf("seed requires-grad flag changed: %+v %v", info, err)
		}
	}
}

func TestFinalStateSeedCrossesEveryChunkBoundary(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 1, Tokens: 5, Heads: 2, KeyDim: 2, ValueDim: 3}
	input, dValues, dState := fixture(g, torch.Float64)
	clear(dValues)
	expected, err := training.GatedDeltaVJP(context.Background(), g, input, dValues, dState, training.DefaultRecurrentOptions())
	if err != nil {
		t.Fatal(err)
	}
	limits := sequence.DefaultLimits()
	limits.ChunkTokens = 2
	seeds := nativeSeeds(t, g, dValues, dState, torch.Float64, false)
	result, err := sequence.GradVJP(context.Background(), nativeInput(t, g, input, torch.Float64, false), seeds[0], seeds[1], limits)
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = result.Close() })
	for i, gradient := range fields(tensor.Input(*result.Gradients)) {
		closeValues(t, fieldNames[i], read(t, gradient), vectors(expected.Gradients)[i], 3e-12)
	}
	initial := read(t, result.Gradients.InitialState)
	if !slices.ContainsFunc(initial, func(value float64) bool { return value != 0 }) {
		t.Fatal("final-state seed did not reach the nonzero initial state")
	}
}

func TestOwnershipBudgetIncludesBackwardBuffers(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 1, Tokens: 5, Heads: 1, KeyDim: 2, ValueDim: 3}
	input, dValues, dState := fixture(g, torch.Float64)
	native := nativeInput(t, g, input, torch.Float64, false)
	seeds := nativeSeeds(t, g, dValues, dState, torch.Float64, false)
	// Two overlapping output buffers hold 30 values; two states hold 12.
	// Backward must refuse this forward-only budget: it also owns checkpoints
	// and adjoints and accounts for the explicitly borrowed seed buffers.
	limits := sequence.SequenceLimits{ChunkTokens: 2, MaxTokens: 5, MaxOwnedElements: 42}
	output, err := sequence.Forward(context.Background(), native, limits)
	if err != nil {
		t.Fatal(err)
	}
	_ = output.Close()
	if backward, err := sequence.GradVJP(context.Background(), native, seeds[0], seeds[1], limits); backward != nil || !errors.Is(err, sequence.ErrLimit) {
		t.Fatalf("backward=%v err=%v; forward-only budget must be rejected", backward, err)
	}
	limits.MaxOwnedElements--
	if output, err := sequence.Forward(context.Background(), native, limits); output != nil || !errors.Is(err, sequence.ErrLimit) {
		t.Fatalf("output=%v err=%v; insufficient output budget admitted", output, err)
	}
	limits.MaxOwnedElements = 1024
	backward, err := sequence.GradVJP(context.Background(), native, seeds[0], seeds[1], limits)
	if err != nil {
		t.Fatal(err)
	}
	if err := backward.Close(); err != nil {
		t.Fatal(err)
	}
	if err := backward.Close(); err != nil {
		t.Fatal("Close is not idempotent:", err)
	}
}

func TestSequenceRejectsInvalidLimitsInputsAndSeeds(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 1, Tokens: 5, Heads: 1, KeyDim: 2, ValueDim: 3}
	input, dValues, dState := fixture(g, torch.Float64)
	native := nativeInput(t, g, input, torch.Float64, false)
	for _, limits := range []sequence.SequenceLimits{
		{}, {ChunkTokens: 0, MaxTokens: 5, MaxOwnedElements: 1024},
		{ChunkTokens: 33, MaxTokens: 5, MaxOwnedElements: 1024},
		{ChunkTokens: 2, MaxTokens: 4, MaxOwnedElements: 1024},
		{ChunkTokens: 2, MaxTokens: 5, MaxOwnedElements: 0},
	} {
		if output, err := sequence.Forward(context.Background(), native, limits); output != nil || !errors.Is(err, sequence.ErrLimit) {
			t.Fatalf("limits=%+v output=%v err=%v", limits, output, err)
		}
	}
	limits := sequence.DefaultLimits()
	bad := native
	bad.Key = nil
	if output, err := sequence.Forward(context.Background(), bad, limits); output != nil || !errors.Is(err, sequence.ErrInvalidInput) {
		t.Fatalf("nil key output=%v err=%v", output, err)
	}
	bad = native
	bad.Beta = own(t)(native.Beta.Reshape([]int64{1, 5}))
	if output, err := sequence.Forward(context.Background(), bad, limits); output != nil || !errors.Is(err, sequence.ErrInvalidInput) {
		t.Fatalf("invalid shape output=%v err=%v", output, err)
	}
	bad = native
	bad.Key = own(t)(native.Key.To(torch.CPUDevice(), torch.Float32))
	if output, err := sequence.Forward(context.Background(), bad, limits); output != nil || !errors.Is(err, sequence.ErrInvalidInput) {
		t.Fatalf("mixed dtype output=%v err=%v", output, err)
	}
	seeds := nativeSeeds(t, g, dValues, dState, torch.Float64, false)
	for _, pair := range [][2]*torch.Tensor{{nil, seeds[1]}, {seeds[0], nil}, {seeds[1], seeds[0]}} {
		if result, err := sequence.GradVJP(context.Background(), native, pair[0], pair[1], limits); result != nil || !errors.Is(err, sequence.ErrInvalidInput) {
			t.Fatalf("invalid seeds result=%v err=%v", result, err)
		}
	}
	dState[0] = math.NaN()
	seeds = nativeSeeds(t, g, dValues, dState, torch.Float64, false)
	if result, err := sequence.GradVJP(context.Background(), native, seeds[0], seeds[1], limits); result != nil || !errors.Is(err, sequence.ErrNonFinite) {
		t.Fatalf("non-finite seed result=%v err=%v", result, err)
	}
	g.Tokens = 0
	empty, _, _ := fixture(g, torch.Float64)
	if output, err := sequence.Forward(context.Background(), nativeInput(t, g, empty, torch.Float64, false), limits); output != nil || !errors.Is(err, sequence.ErrInvalidInput) {
		t.Fatalf("empty sequence output=%v err=%v", output, err)
	}
}

func TestContextAndLateChunkFailureReturnNoPartialResult(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 1, Tokens: 5, Heads: 1, KeyDim: 2, ValueDim: 3}
	input, dValues, dState := fixture(g, torch.Float64)
	native := nativeInput(t, g, input, torch.Float64, true)
	seeds := nativeSeeds(t, g, dValues, dState, torch.Float64, false)
	limits := sequence.DefaultLimits()
	limits.ChunkTokens = 2
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if result, err := sequence.Forward(ctx, native, limits); result != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled forward result=%v err=%v", result, err)
	}
	if result, err := sequence.GradVJP(ctx, native, seeds[0], seeds[1], limits); result != nil || !errors.Is(err, context.Canceled) {
		t.Fatalf("cancelled VJP result=%v err=%v", result, err)
	}
	if result, err := sequence.Forward(nil, native, limits); result != nil || !errors.Is(err, sequence.ErrInvalidInput) {
		t.Fatalf("nil context result=%v err=%v", result, err)
	}
	// All inputs are finite. The third chunk overflows after two successful
	// chunks, requiring cleanup of their output pieces and checkpoint states.
	input.LogDecay[4] = 1000
	bad := nativeInput(t, g, input, torch.Float64, true)
	if result, err := sequence.Forward(context.Background(), bad, limits); result != nil || !errors.Is(err, sequence.ErrNonFinite) {
		t.Fatalf("late forward failure result=%v err=%v", result, err)
	}
	if result, err := sequence.GradVJP(context.Background(), bad, seeds[0], seeds[1], limits); result != nil || !errors.Is(err, sequence.ErrNonFinite) {
		t.Fatalf("late VJP failure result=%v err=%v", result, err)
	}
	for i, value := range fields(bad) {
		closeValues(t, "borrowed after failure "+fieldNames[i], read(t, value), vectors(input)[i], 0)
	}
}

var fieldNames = []string{"query", "key", "value", "log decay", "beta", "initial state"}

func fixture(g training.RecurrentGeometry, dtype torch.DType) (training.RecurrentInput, []float64, []float64) {
	rows := g.Batch * g.Tokens * g.Heads
	input := training.RecurrentInput{
		Query: make([]float64, rows*g.KeyDim), Key: make([]float64, rows*g.KeyDim), Value: make([]float64, rows*g.ValueDim),
		LogDecay: make([]float64, rows), Beta: make([]float64, rows), InitialState: make([]float64, g.Batch*g.Heads*g.KeyDim*g.ValueDim),
	}
	for field, vector := range vectors(input) {
		for i := range vector {
			vector[i] = .03 + .21*math.Sin(float64(i+field)+.7)
		}
	}
	for i := range input.LogDecay {
		input.LogDecay[i], input.Beta[i] = -.1-.03*float64(i%3), .3+.08*float64(i%4)
	}
	dValues, dState := make([]float64, len(input.Value)), make([]float64, len(input.InitialState))
	for i := range dValues {
		dValues[i] = .1 + .04*float64(i%5)
	}
	for i := range dState {
		dState[i] = -.06 + .03*float64(i%7)
	}
	if dtype == torch.Float32 {
		for _, vector := range append(vectors(input), dValues, dState) {
			for i := range vector {
				vector[i] = float64(float32(vector[i]))
			}
		}
	}
	return input, dValues, dState
}

func nativeInput(t *testing.T, g training.RecurrentGeometry, input training.RecurrentInput, dtype torch.DType, requiresGrad bool) tensor.Input {
	t.Helper()
	b, n, h, k, v := int64(g.Batch), int64(g.Tokens), int64(g.Heads), int64(g.KeyDim), int64(g.ValueDim)
	shapes := [][]int64{{b, n, h, k}, {b, n, h, k}, {b, n, h, v}, {b, n, h}, {b, n, h}, {b, h, k, v}}
	var current []*torch.Tensor
	for i, vector := range vectors(input) {
		current = append(current, makeTensor(t, vector, shapes[i], dtype, requiresGrad))
	}
	return tensor.Input{Query: current[0], Key: current[1], Value: current[2], LogDecay: current[3], Beta: current[4], InitialState: current[5]}
}

func nativeSeeds(t *testing.T, g training.RecurrentGeometry, output, state []float64, dtype torch.DType, requiresGrad bool) [2]*torch.Tensor {
	t.Helper()
	return [2]*torch.Tensor{
		makeTensor(t, output, []int64{int64(g.Batch), int64(g.Tokens), int64(g.Heads), int64(g.ValueDim)}, dtype, requiresGrad),
		makeTensor(t, state, []int64{int64(g.Batch), int64(g.Heads), int64(g.KeyDim), int64(g.ValueDim)}, dtype, requiresGrad),
	}
}

func makeTensor(t *testing.T, values []float64, shape []int64, dtype torch.DType, requiresGrad bool) *torch.Tensor {
	t.Helper()
	if dtype == torch.Float64 {
		return own(t)(torch.FromFloat64(values, shape, torch.CPUDevice(), requiresGrad))
	}
	converted := make([]float32, len(values))
	for i, value := range values {
		converted[i] = float32(value)
	}
	return own(t)(torch.FromFloat32(converted, shape, torch.CPUDevice(), requiresGrad))
}

func own(t *testing.T) func(*torch.Tensor, error) *torch.Tensor {
	return func(value *torch.Tensor, err error) *torch.Tensor {
		t.Helper()
		if err != nil {
			t.Fatal(err)
		}
		t.Cleanup(func() { _ = value.Close() })
		return value
	}
}

func read(t *testing.T, value *torch.Tensor) []float64 {
	t.Helper()
	result, err := value.Float64Values()
	if err != nil {
		t.Fatal(err)
	}
	return result
}

func fields(input tensor.Input) []*torch.Tensor {
	return []*torch.Tensor{input.Query, input.Key, input.Value, input.LogDecay, input.Beta, input.InitialState}
}

func vectors(input training.RecurrentInput) [][]float64 {
	return [][]float64{input.Query, input.Key, input.Value, input.LogDecay, input.Beta, input.InitialState}
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
