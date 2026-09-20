package recurrent_test

import (
	"context"
	"errors"
	"math"
	"reflect"
	"sync/atomic"
	"testing"

	"github.com/tayi-ai/arandu-llama/training"
)

func fixture(g training.RecurrentGeometry) (training.RecurrentInput, []float64, []float64) {
	values := func(n int, shift float64) []float64 {
		out := make([]float64, n)
		for i := range out {
			out[i] = 0.25*math.Sin(float64(i+1)*0.73) + shift
		}
		return out
	}
	scalar := g.Batch * g.Tokens * g.Heads
	state := g.Batch * g.Heads * g.KeyDim * g.ValueDim
	input := training.RecurrentInput{
		Query: values(scalar*g.KeyDim, 0.1), Key: values(scalar*g.KeyDim, -0.2), Value: values(scalar*g.ValueDim, 0.4),
		LogDecay: values(scalar, -0.5), Beta: values(scalar, 0.6), InitialState: values(state, 0.15),
	}
	return input, values(scalar*g.ValueDim, -0.3), values(state, 0.2)
}

// Independent oracle expands the transition matrix instead of evaluating the
// implementation's staged residual update: S' = rho*(I-beta*k*kᵀ)*S+beta*k*vᵀ.
func oracle(g training.RecurrentGeometry, x training.RecurrentInput) ([]float64, []float64) {
	output := make([]float64, g.Batch*g.Tokens*g.Heads*g.ValueDim)
	state := append([]float64(nil), x.InitialState...)
	for b := 0; b < g.Batch; b++ {
		for h := 0; h < g.Heads; h++ {
			base := (b*g.Heads + h) * g.KeyDim * g.ValueDim
			for token := 0; token < g.Tokens; token++ {
				position := (b*g.Tokens+token)*g.Heads + h
				next := make([]float64, g.KeyDim*g.ValueDim)
				for i := 0; i < g.KeyDim; i++ {
					ki := x.Key[position*g.KeyDim+i]
					for j := 0; j < g.ValueDim; j++ {
						next[i*g.ValueDim+j] = x.Beta[position] * ki * x.Value[position*g.ValueDim+j]
						for k := 0; k < g.KeyDim; k++ {
							coefficient := -x.Beta[position] * ki * x.Key[position*g.KeyDim+k]
							if i == k {
								coefficient++
							}
							next[i*g.ValueDim+j] += math.Exp(x.LogDecay[position]) * coefficient * state[base+k*g.ValueDim+j]
						}
					}
				}
				copy(state[base:base+len(next)], next)
				for j := 0; j < g.ValueDim; j++ {
					for k := 0; k < g.KeyDim; k++ {
						output[position*g.ValueDim+j] += x.Query[position*g.KeyDim+k] * next[k*g.ValueDim+j]
					}
				}
			}
		}
	}
	return output, state
}

func objective(g training.RecurrentGeometry, x training.RecurrentInput, dy, ds []float64) float64 {
	y, state := oracle(g, x)
	value := 0.0
	for i := range y {
		value += y[i] * dy[i]
	}
	for i := range state {
		value += state[i] * ds[i]
	}
	return value
}

func vectors(input training.RecurrentInput) [][]float64 {
	return [][]float64{input.Query, input.Key, input.Value, input.LogDecay, input.Beta, input.InitialState}
}

func clone(input training.RecurrentInput) training.RecurrentInput {
	return training.RecurrentInput{
		Query: append([]float64(nil), input.Query...), Key: append([]float64(nil), input.Key...),
		Value: append([]float64(nil), input.Value...), LogDecay: append([]float64(nil), input.LogDecay...),
		Beta: append([]float64(nil), input.Beta...), InitialState: append([]float64(nil), input.InitialState...),
	}
}

func closeVector(t *testing.T, got, want []float64, tolerance float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("length %d, expected %d", len(got), len(want))
	}
	for i := range got {
		if math.IsNaN(got[i]) || math.Abs(got[i]-want[i]) > tolerance*(1+math.Abs(want[i])) {
			t.Fatalf("element %d: got %.17g, expected %.17g", i, got[i], want[i])
		}
	}
}

func TestTwoTokenAnalyticBPTTAndFinalState(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 1, Tokens: 2, Heads: 1, KeyDim: 1, ValueDim: 1}
	x := training.RecurrentInput{Query: []float64{2, -1}, Key: []float64{0.5, 0.25}, Value: []float64{4, -2},
		LogDecay: []float64{0, 0}, Beta: []float64{0.5, 0.8}, InitialState: []float64{3}}
	result, err := training.GatedDeltaVJP(context.Background(), g, x, []float64{0, 1}, []float64{2}, training.DefaultRecurrentOptions())
	if err != nil {
		t.Fatal(err)
	}
	closeVector(t, result.Output.Values, []float64{7.25, -3.04375}, 1e-14)
	closeVector(t, result.Output.FinalState, []float64{3.04375}, 1e-14)
	want := training.RecurrentInput{
		Query: []float64{0, 3.04375}, Key: []float64{0.475, -3.05}, Value: []float64{0.2375, 0.2},
		LogDecay: []float64{2.49375, 3.44375}, Beta: []float64{1.1875, -0.7265625}, InitialState: []float64{0.83125},
	}
	for i, actual := range vectors(result.Gradients) {
		closeVector(t, actual, vectors(want)[i], 1e-14)
	}
	// The first token has no output seed. Its nonzero gradients must arrive
	// through the recurrent state, not through a token-local derivative.
	if result.Gradients.Value[0] == 0 || result.Gradients.InitialState[0] == 0 {
		t.Fatal("recurrent gradient was truncated")
	}
}

func TestEveryInputAdjointAgainstIndependentFiniteDifferences(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 2, Tokens: 4, Heads: 2, KeyDim: 2, ValueDim: 3}
	for _, zeroState := range []bool{false, true} {
		x, dy, ds := fixture(g)
		if zeroState {
			clear(x.InitialState)
		}
		options := training.DefaultRecurrentOptions()
		options.CheckpointInterval = 2
		result, err := training.GatedDeltaVJP(context.Background(), g, x, dy, ds, options)
		if err != nil {
			t.Fatal(err)
		}
		y, state := oracle(g, x)
		closeVector(t, result.Output.Values, y, 1e-14)
		closeVector(t, result.Output.FinalState, state, 1e-14)
		maximumError, checked := 0.0, 0
		for field, values := range vectors(x) {
			for index, original := range values {
				const step = 1e-6
				values[index] = original + step
				plus := objective(g, x, dy, ds)
				values[index] = original - step
				minus := objective(g, x, dy, ds)
				values[index] = original
				expected := (plus - minus) / (2 * step)
				actual := vectors(result.Gradients)[field][index]
				errorMagnitude := math.Abs(actual - expected)
				maximumError = max(maximumError, errorMagnitude)
				checked++
				if !recurrentTestFinite(actual) || errorMagnitude > 2e-8*(1+math.Abs(expected)) {
					t.Fatalf("zeroState=%v field=%d index=%d: adjoint %.17g, independent finite difference %.17g", zeroState, field, index, actual, expected)
				}
			}
		}
		t.Logf("zero_initial_state=%v checked_derivatives=%d maximum_absolute_error=%.3g", zeroState, checked, maximumError)
	}
}

func recurrentTestFinite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }

func TestCheckpointIntervalsAgreeAndReturnDetachedSlices(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 2, Tokens: 7, Heads: 2, KeyDim: 3, ValueDim: 2}
	x, dy, ds := fixture(g)
	before, beforeDY, beforeDS := clone(x), append([]float64(nil), dy...), append([]float64(nil), ds...)
	var baseline training.RecurrentBackward
	for _, interval := range []int{1, 2, 3, 7, 1000} {
		options := training.DefaultRecurrentOptions()
		options.CheckpointInterval = interval
		result, err := training.GatedDeltaVJP(context.Background(), g, x, dy, ds, options)
		if err != nil {
			t.Fatal(err)
		}
		forward, err := training.GatedDeltaForward(context.Background(), g, x, options)
		if err != nil {
			t.Fatal(err)
		}
		if !reflect.DeepEqual(result.Output.Values, forward.Values) || !reflect.DeepEqual(result.Output.FinalState, forward.FinalState) {
			t.Fatal("VJP recomputation differs from forward")
		}
		if interval == 1 {
			baseline = result
		} else if !reflect.DeepEqual(result.Gradients, baseline.Gradients) || !reflect.DeepEqual(result.Output.Values, baseline.Output.Values) {
			t.Fatalf("checkpoint interval %d changed the scalar arithmetic", interval)
		}
	}
	baseline.Output.Values[0] = 100
	baseline.Output.FinalState[0] = 200
	baseline.Gradients.Query[0] = 300
	baseline.Gradients.InitialState[0] = 400
	if !reflect.DeepEqual(x, before) || !reflect.DeepEqual(dy, beforeDY) || !reflect.DeepEqual(ds, beforeDS) {
		t.Fatal("caller-owned input or seed changed")
	}
	// Caller aliases are allowed because all input slices are read-only.
	x.Key = x.Query
	aliased := clone(x)
	if _, err := training.GatedDeltaVJP(context.Background(), g, x, dy, ds, training.DefaultRecurrentOptions()); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(x, aliased) {
		t.Fatal("aliased inputs changed")
	}
}

func TestWorkspaceBoundaryAndCheckpointStorage(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 1, Tokens: 129, Heads: 2, KeyDim: 2, ValueDim: 3}
	x, dy, ds := fixture(g)
	options := training.DefaultRecurrentOptions()
	options.CheckpointInterval = 16
	result, err := training.GatedDeltaVJP(context.Background(), g, x, dy, ds, options)
	if err != nil {
		t.Fatal(err)
	}
	storage := result.Output.Storage
	if storage.CheckpointStates != 9 || storage.BlockStates != 17 || storage.RecomputedTokens != 129 {
		t.Fatalf("unexpected bounded storage: %+v", storage)
	}
	if storage.CheckpointStates+storage.BlockStates >= g.Tokens {
		t.Fatal("saved an entire state trajectory")
	}
	// Output + gradients + boundary states + temporary block + three value vectors.
	state, scalar := g.Heads*g.KeyDim*g.ValueDim, g.Tokens*g.Heads
	expectedBytes := int64(8 * (2*scalar*g.ValueDim + 2*scalar*g.KeyDim + 2*scalar + 2*state + 26*state + 3*g.ValueDim))
	if storage.WorkspaceBytes != expectedBytes {
		t.Fatalf("workspace bytes %d, expected %d", storage.WorkspaceBytes, expectedBytes)
	}
	options.MaxWorkspaceBytes = storage.WorkspaceBytes
	if _, err := training.GatedDeltaVJP(context.Background(), g, x, dy, ds, options); err != nil {
		t.Fatal("exact workspace budget rejected:", err)
	}
	options.MaxWorkspaceBytes--
	failed, err := training.GatedDeltaVJP(context.Background(), g, x, dy, ds, options)
	if err == nil || !reflect.DeepEqual(failed, training.RecurrentBackward{}) {
		t.Fatal("undersized budget returned success or partial results")
	}
	forward, err := training.GatedDeltaForward(context.Background(), g, x, training.DefaultRecurrentOptions())
	if err != nil {
		t.Fatal(err)
	}
	options.MaxWorkspaceBytes = forward.Storage.WorkspaceBytes - 1
	if _, err := training.GatedDeltaForward(context.Background(), g, x, options); err == nil {
		t.Fatal("forward ignored workspace budget")
	}
	t.Logf("tokens=%d checkpoints=%d block_states=%d owned_float64_bytes=%d", g.Tokens, storage.CheckpointStates, storage.BlockStates, storage.WorkspaceBytes)
}

func TestEmptySequenceIsStateIdentityAndZeroSeedsGiveZeroAdjoints(t *testing.T) {
	for _, tokens := range []int{0, 3} {
		g := training.RecurrentGeometry{Batch: 2, Tokens: tokens, Heads: 2, KeyDim: 2, ValueDim: 3}
		x, dy, ds := fixture(g)
		if tokens != 0 {
			clear(dy)
			clear(ds)
		}
		result, err := training.GatedDeltaVJP(context.Background(), g, x, dy, ds, training.DefaultRecurrentOptions())
		if err != nil {
			t.Fatal(err)
		}
		if tokens == 0 {
			closeVector(t, result.Output.FinalState, x.InitialState, 0)
			closeVector(t, result.Gradients.InitialState, ds, 0)
			if result.Output.Storage.CheckpointStates != 0 || result.Output.Storage.BlockStates != 0 {
				t.Fatal("empty sequence allocated a tape")
			}
		} else {
			for _, vector := range vectors(result.Gradients) {
				for _, value := range vector {
					if value != 0 {
						t.Fatal("zero cotangents produced a nonzero gradient")
					}
				}
			}
		}
	}
}

func TestRejectsMalformedAndNonfiniteInputs(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 1, Tokens: 2, Heads: 1, KeyDim: 2, ValueDim: 3}
	x, dy, ds := fixture(g)
	options := training.DefaultRecurrentOptions()
	for field := range vectors(x) {
		for _, bad := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
			invalid := clone(x)
			vectors(invalid)[field][0] = bad
			if result, err := training.GatedDeltaForward(context.Background(), g, invalid, options); err == nil || !reflect.DeepEqual(result, training.RecurrentOutput{}) {
				t.Fatalf("forward accepted nonfinite input %d", field)
			}
			if result, err := training.GatedDeltaVJP(context.Background(), g, invalid, dy, ds, options); err == nil || !reflect.DeepEqual(result, training.RecurrentBackward{}) {
				t.Fatalf("VJP accepted nonfinite input %d", field)
			}
		}
	}
	for _, invalid := range []training.RecurrentInput{
		{Query: x.Query[:1], Key: x.Key, Value: x.Value, LogDecay: x.LogDecay, Beta: x.Beta, InitialState: x.InitialState},
		{Query: x.Query, Key: nil, Value: x.Value, LogDecay: x.LogDecay, Beta: x.Beta, InitialState: x.InitialState},
		{Query: x.Query, Key: x.Key, Value: nil, LogDecay: x.LogDecay, Beta: x.Beta, InitialState: x.InitialState},
		{Query: x.Query, Key: x.Key, Value: x.Value, LogDecay: nil, Beta: x.Beta, InitialState: x.InitialState},
		{Query: x.Query, Key: x.Key, Value: x.Value, LogDecay: x.LogDecay, Beta: nil, InitialState: x.InitialState},
		{Query: x.Query, Key: x.Key, Value: x.Value, LogDecay: x.LogDecay, Beta: x.Beta, InitialState: nil},
	} {
		if _, err := training.GatedDeltaVJP(context.Background(), g, invalid, dy, ds, options); err == nil {
			t.Fatal("accepted truncated input")
		}
	}
	for _, seed := range []struct{ output, final []float64 }{{dy[:1], ds}, {dy, ds[:1]}, {append([]float64{math.Inf(1)}, dy[1:]...), ds}, {dy, append([]float64{math.NaN()}, ds[1:]...)}} {
		if _, err := training.GatedDeltaVJP(context.Background(), g, x, seed.output, seed.final, options); err == nil {
			t.Fatal("accepted malformed cotangent")
		}
	}
}

func TestRejectsShapeArithmeticAndOptionsOverflow(t *testing.T) {
	maximum := int(^uint(0) >> 1)
	for _, g := range []training.RecurrentGeometry{
		{}, {Batch: 1, Tokens: -1, Heads: 1, KeyDim: 1, ValueDim: 1},
		{Batch: maximum, Tokens: 1, Heads: 2, KeyDim: 1, ValueDim: 1},
		{Batch: 2, Tokens: maximum, Heads: 1, KeyDim: 1, ValueDim: 1},
		{Batch: 1, Tokens: maximum, Heads: 1, KeyDim: 2, ValueDim: 1},
		{Batch: 1, Tokens: maximum, Heads: 1, KeyDim: 1, ValueDim: 2},
		{Batch: 1, Tokens: maximum, Heads: 1, KeyDim: 1, ValueDim: 1},
	} {
		if _, err := training.GatedDeltaVJP(context.Background(), g, training.RecurrentInput{}, nil, nil, training.DefaultRecurrentOptions()); err == nil {
			t.Fatal("accepted invalid or overflowing shape")
		}
	}
	g := training.RecurrentGeometry{Batch: 1, Tokens: 1, Heads: 1, KeyDim: 1, ValueDim: 1}
	x, dy, ds := fixture(g)
	for _, options := range []training.RecurrentOptions{{}, {CheckpointInterval: -1, MaxWorkspaceBytes: 1024}, {CheckpointInterval: 1, MaxWorkspaceBytes: -1}} {
		if _, err := training.GatedDeltaVJP(context.Background(), g, x, dy, ds, options); err == nil {
			t.Fatal("accepted invalid workspace options")
		}
	}
}

func TestRejectsNumericalOverflowWithoutChangingInput(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 1, Tokens: 1, Heads: 1, KeyDim: 1, ValueDim: 1}
	for _, x := range []training.RecurrentInput{
		{Query: []float64{1}, Key: []float64{1}, Value: []float64{1}, LogDecay: []float64{1000}, Beta: []float64{0.5}, InitialState: []float64{0}},
		{Query: []float64{1}, Key: []float64{1}, Value: []float64{-math.MaxFloat64}, LogDecay: []float64{0}, Beta: []float64{0.5}, InitialState: []float64{math.MaxFloat64}},
		{Query: []float64{math.MaxFloat64}, Key: []float64{1}, Value: []float64{2}, LogDecay: []float64{0}, Beta: []float64{1}, InitialState: []float64{0}},
	} {
		before := clone(x)
		if result, err := training.GatedDeltaForward(context.Background(), g, x, training.DefaultRecurrentOptions()); err == nil || !reflect.DeepEqual(result, training.RecurrentOutput{}) {
			t.Fatal("accepted nonfinite forward arithmetic")
		}
		if !reflect.DeepEqual(x, before) {
			t.Fatal("failed forward mutated inputs")
		}
	}
	x := training.RecurrentInput{Query: []float64{2}, Key: []float64{1}, Value: []float64{1}, LogDecay: []float64{0}, Beta: []float64{0.5}, InitialState: []float64{0}}
	before := clone(x)
	result, err := training.GatedDeltaVJP(context.Background(), g, x, []float64{math.MaxFloat64}, []float64{0}, training.DefaultRecurrentOptions())
	if err == nil || !reflect.DeepEqual(result, training.RecurrentBackward{}) || !reflect.DeepEqual(x, before) {
		t.Fatal("backward overflow returned partial results or changed input")
	}
	x.LogDecay[0] = -1000
	result, err = training.GatedDeltaVJP(context.Background(), g, x, []float64{1}, []float64{1}, training.DefaultRecurrentOptions())
	if err != nil || result.Gradients.LogDecay[0] != 0 || result.Gradients.InitialState[0] != 0 {
		t.Fatal("finite decay underflow must produce a zero state connection", err)
	}
}

type pollingContext struct {
	context.Context
	cancel    context.CancelFunc
	remaining atomic.Int64
}

func (c *pollingContext) Err() error {
	if c.remaining.Add(-1) == 0 {
		c.cancel()
	}
	return c.Context.Err()
}

func TestCancellationBeforeAndDuringComputation(t *testing.T) {
	g := training.RecurrentGeometry{Batch: 1, Tokens: 9, Heads: 2, KeyDim: 3, ValueDim: 4}
	x, dy, ds := fixture(g)
	before := clone(x)
	if _, err := training.GatedDeltaForward(nil, g, x, training.DefaultRecurrentOptions()); err == nil {
		t.Fatal("nil context accepted")
	}
	for _, polls := range []int64{1, 20, 170, 210, 360} {
		parent, cancel := context.WithCancel(context.Background())
		ctx := &pollingContext{Context: parent, cancel: cancel}
		ctx.remaining.Store(polls)
		options := training.DefaultRecurrentOptions()
		options.CheckpointInterval = 2
		result, err := training.GatedDeltaVJP(ctx, g, x, dy, ds, options)
		cancel()
		if !errors.Is(err, context.Canceled) || !reflect.DeepEqual(result, training.RecurrentBackward{}) {
			t.Fatalf("poll %d: cancellation not propagated atomically: %v", polls, err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := training.GatedDeltaForward(ctx, g, x, training.DefaultRecurrentOptions()); !errors.Is(err, context.Canceled) {
		t.Fatal("forward ignored cancelled context")
	}
	if !reflect.DeepEqual(x, before) {
		t.Fatal("cancellation changed caller inputs")
	}
}
