package native_test

import (
	"encoding/json"
	"math"
	"reflect"
	"slices"
	"testing"

	services "github.com/tayi-ai/arandu-llama/training/native"
)

func experimentGradients(unscaled ...float32) []services.ExperimentRankGradient {
	gradients := make([]services.ExperimentRankGradient, 20)
	for rank := range gradients {
		values := make([]float32, len(unscaled))
		for index, value := range unscaled {
			values[index] = value / 1024
			if rank == 0 {
				values[index] *= 2
			}
		}
		gradients[rank] = services.ExperimentRankGradient{Rank: rank, Values: values}
	}
	return gradients
}

func experimentOptimizer(t *testing.T, parameters ...float32) *services.ExperimentAdamW {
	t.Helper()
	optimizer, err := services.NewExperimentAdamW(parameters)
	if err != nil {
		t.Fatal(err)
	}
	return optimizer
}

func experimentNear(t *testing.T, label string, actual, expected, tolerance float64) {
	t.Helper()
	if math.IsNaN(actual) || math.Abs(actual-expected) > tolerance {
		t.Fatalf("%s: got %.12g, want %.12g within %.3g", label, actual, expected, tolerance)
	}
}

func TestExperimentAdamWMatchesIndependentScalarEquations(t *testing.T) {
	optimizer := experimentOptimizer(t, 0)
	var parameter, first, second float64
	for index, gradient := range []float64{0.25, -0.5, 0.125} {
		step := index + 1
		first = 0.9*first + 0.1*gradient
		second = 0.999*second + 0.001*gradient*gradient
		correctedFirst := first / (1 - math.Pow(0.9, float64(step)))
		correctedSecond := second / (1 - math.Pow(0.999, float64(step)))
		parameter -= 2e-6 * correctedFirst / (math.Sqrt(correctedSecond) + 1e-8)
		update, err := optimizer.Update(experimentGradients(float32(gradient)))
		if err != nil {
			t.Fatal(err)
		}
		state := optimizer.Snapshot()
		experimentNear(t, "parameter", float64(state.Parameters[0]), parameter, 1e-12)
		experimentNear(t, "first moment", float64(state.FirstMoment[0]), first, 5e-9)
		experimentNear(t, "second moment", float64(state.SecondMoment[0]), second, 5e-11)
		if update.Step != step || update.ReplicaCount != 20 || update.GlobalBatch != 21 || update.NumericMode != "experiment-go-adamw-v2" {
			t.Fatal("update did not identify the native numerical mode and replica count")
		}
	}
}

func TestExperimentAdamWDivides20RankSumsByGlobalBatch21BeforeUnscaling(t *testing.T) {
	optimizer := experimentOptimizer(t, 0)
	replicas := experimentGradients(0)
	for rank := range replicas {
		replicas[rank].Values[0] = float32(rank) / (64 * 1024)
	}
	update, err := optimizer.Update(replicas)
	if err != nil {
		t.Fatal(err)
	}
	// Sum(0..19)/21/64 = 190/21/64. Dividing by replica count or omitting
	// unscaling fails this.
	expected := 190.0 / 21 / 64
	experimentNear(t, "mean gradient norm", update.GradientNorm, expected, 1e-7)
	experimentNear(t, "mean first moment", float64(optimizer.Snapshot().FirstMoment[0]), 0.1*expected, 1e-8)
	if update.ClipCoefficient != 1 {
		t.Fatal("small mean gradient was clipped")
	}
}

func TestExperimentAdamWReducesInRankOrderRegardlessOfArrival(t *testing.T) {
	ordered := experimentGradients(0, 0)
	for rank := range ordered {
		ordered[rank].Values = []float32{float32(rank+1) / 12345, float32(rank-10) / 54321}
	}
	ordered[0].Values[0] = 10000
	ordered[1].Values[0] = -10000
	permuted := slices.Clone(ordered)
	slices.Reverse(permuted)
	left, right := experimentOptimizer(t, 0, 0), experimentOptimizer(t, 0, 0)
	a, err := left.Update(ordered)
	if err != nil {
		t.Fatal(err)
	}
	b, err := right.Update(permuted)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(a, b) || !reflect.DeepEqual(left.Snapshot(), right.Snapshot()) {
		t.Fatal("arrival order changed the update")
	}
}

func TestExperimentAdamWClipsTheGlobalNormAfterUnscaling(t *testing.T) {
	optimizer := experimentOptimizer(t, 0, 0)
	update, err := optimizer.Update(experimentGradients(3, 4))
	if err != nil {
		t.Fatal(err)
	}
	experimentNear(t, "unscaled norm", update.GradientNorm, 5, 1e-12)
	experimentNear(t, "clip coefficient", update.ClipCoefficient, 1/(5+1e-12), 1e-15)
	state := optimizer.Snapshot()
	experimentNear(t, "first moment x", float64(state.FirstMoment[0]), 0.06, 1e-8)
	experimentNear(t, "first moment y", float64(state.FirstMoment[1]), 0.08, 1e-8)
	experimentNear(t, "second moment x", float64(state.SecondMoment[0]), 0.00036, 1e-10)
	experimentNear(t, "second moment y", float64(state.SecondMoment[1]), 0.00064, 2e-10)
}

func TestExperimentAdamWRejectsInputsWithoutChangingAnyState(t *testing.T) {
	cases := map[string]func([]services.ExperimentRankGradient) []services.ExperimentRankGradient{
		"missing rank":      func(g []services.ExperimentRankGradient) []services.ExperimentRankGradient { return g[:19] },
		"extra rank":        func(g []services.ExperimentRankGradient) []services.ExperimentRankGradient { return append(g, g[0]) },
		"duplicate rank":    func(g []services.ExperimentRankGradient) []services.ExperimentRankGradient { g[19].Rank = 0; return g },
		"negative rank":     func(g []services.ExperimentRankGradient) []services.ExperimentRankGradient { g[0].Rank = -1; return g },
		"rank out of range": func(g []services.ExperimentRankGradient) []services.ExperimentRankGradient { g[19].Rank = 20; return g },
		"wrong shape": func(g []services.ExperimentRankGradient) []services.ExperimentRankGradient {
			g[19].Values = g[19].Values[:1]
			return g
		},
		"NaN": func(g []services.ExperimentRankGradient) []services.ExperimentRankGradient {
			g[19].Values[1] = float32(math.NaN())
			return g
		},
		"infinity": func(g []services.ExperimentRankGradient) []services.ExperimentRankGradient {
			g[19].Values[1] = float32(math.Inf(1))
			return g
		},
		"sum overflow": func(g []services.ExperimentRankGradient) []services.ExperimentRankGradient {
			g[0].Values[1] = math.MaxFloat32
			g[1].Values[1] = math.MaxFloat32
			return g
		},
		"unscale overflow": func(g []services.ExperimentRankGradient) []services.ExperimentRankGradient {
			g[19].Values[1] = math.MaxFloat32
			return g
		},
	}
	for name, damage := range cases {
		t.Run(name, func(t *testing.T) {
			optimizer := experimentOptimizer(t, 0, 0)
			if _, err := optimizer.Update(experimentGradients(0.125, 0.25)); err != nil {
				t.Fatal(err)
			}
			before := optimizer.Snapshot()
			if _, err := optimizer.Update(damage(experimentGradients(0.25, 0.125))); err == nil {
				t.Fatal("invalid replica set admitted")
			}
			if !reflect.DeepEqual(before, optimizer.Snapshot()) {
				t.Fatal("failed update changed optimizer state")
			}
		})
	}
}

func TestExperimentAdamWStagesEveryParameterBeforeCommit(t *testing.T) {
	state := experimentOptimizer(t, 0, 0).Snapshot()
	state.Step = 1
	state.FirstMoment[1] = math.MaxFloat32
	optimizer, err := services.RestoreExperimentAdamW(state)
	if err != nil {
		t.Fatal(err)
	}
	before := optimizer.Snapshot()
	// Coordinate zero would change, but coordinate one overflows first/epsilon.
	if _, err := optimizer.Update(experimentGradients(0.5, 0)); err == nil {
		t.Fatal("overflowing AdamW arithmetic admitted")
	}
	if !reflect.DeepEqual(before, optimizer.Snapshot()) {
		t.Fatal("late arithmetic failure partially committed")
	}
}

func TestExperimentAdamWDoesNotAliasInputsSnapshotsOrFrozenBase(t *testing.T) {
	storage := []float32{123, 0, 0, 456}
	initial := slices.Clone(storage)
	optimizer := experimentOptimizer(t, storage[1:3]...)
	storage[1] = 7
	if optimizer.Snapshot().Parameters[0] != 0 {
		t.Fatal("initializer aliases caller storage")
	}
	storage[1] = initial[1]
	replicas := experimentGradients(0.125, 0.25)
	inputs := make([][]float32, len(replicas))
	for index, g := range replicas {
		inputs[index] = slices.Clone(g.Values)
	}
	if _, err := optimizer.Update(replicas); err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(storage, initial) {
		t.Fatal("frozen base or caller parameter buffer changed")
	}
	for index, g := range replicas {
		if !slices.Equal(inputs[index], g.Values) {
			t.Fatal("input gradient mutated")
		}
	}
	snapshot := optimizer.Snapshot()
	snapshot.Parameters[0] = 100
	snapshot.FirstMoment[0] = 100
	snapshot.SecondMoment[0] = 100
	if optimizer.Snapshot().Parameters[0] == 100 || optimizer.Snapshot().FirstMoment[0] == 100 || optimizer.Snapshot().SecondMoment[0] == 100 {
		t.Fatal("snapshot aliases internal state")
	}
}

func TestExperimentAdamWSnapshotReloadContinuesIdentically(t *testing.T) {
	optimizer := experimentOptimizer(t, 0, 0)
	if _, err := optimizer.Update(experimentGradients(0.125, -0.25)); err != nil {
		t.Fatal(err)
	}
	snapshot := optimizer.Snapshot()
	encoded, err := json.Marshal(snapshot)
	if err != nil {
		t.Fatal(err)
	}
	var decoded services.ExperimentAdamWState
	if err := json.Unmarshal(encoded, &decoded); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(snapshot, decoded) {
		t.Fatal("snapshot serialization changed FP32 state")
	}
	reloaded, err := services.RestoreExperimentAdamW(decoded)
	if err != nil {
		t.Fatal(err)
	}
	decoded.Parameters[0] = 77
	decoded.FirstMoment[0] = 77
	decoded.SecondMoment[0] = 77
	if _, err := optimizer.Update(experimentGradients(-0.25, 0.5)); err != nil {
		t.Fatal(err)
	}
	if _, err := reloaded.Update(experimentGradients(-0.25, 0.5)); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(optimizer.Snapshot(), reloaded.Snapshot()) {
		t.Fatal("reload changed continuation or retained caller aliases")
	}
}

func TestExperimentAdamWRejectsInvalidSnapshotsAndInitialization(t *testing.T) {
	for _, parameters := range [][]float32{nil, {float32(math.NaN())}, {float32(math.Inf(-1))}} {
		if _, err := services.NewExperimentAdamW(parameters); err == nil {
			t.Fatal("invalid initial parameters admitted")
		}
	}
	cases := map[string]func(*services.ExperimentAdamWState){
		"version":                func(s *services.ExperimentAdamWState) { s.SchemaVersion = 2 },
		"mode":                   func(s *services.ExperimentAdamWState) { s.NumericMode = "pytorch" },
		"negative step":          func(s *services.ExperimentAdamWState) { s.Step = -1 },
		"nineteenth step":        func(s *services.ExperimentAdamWState) { s.Step = 19 },
		"shape":                  func(s *services.ExperimentAdamWState) { s.FirstMoment = nil },
		"nonfinite parameter":    func(s *services.ExperimentAdamWState) { s.Parameters[0] = float32(math.Inf(1)) },
		"nonfinite moment":       func(s *services.ExperimentAdamWState) { s.FirstMoment[0] = float32(math.NaN()) },
		"negative second moment": func(s *services.ExperimentAdamWState) { s.SecondMoment[0] = -1 },
		"initial nonzero moment": func(s *services.ExperimentAdamWState) { s.FirstMoment[0] = 1 },
	}
	for name, damage := range cases {
		t.Run(name, func(t *testing.T) {
			state := experimentOptimizer(t, 0).Snapshot()
			damage(&state)
			if _, err := services.RestoreExperimentAdamW(state); err == nil {
				t.Fatal("invalid snapshot admitted")
			}
		})
	}
}

func TestExperimentAdamWStopsAt18AndZeroGradientHasNoDecay(t *testing.T) {
	optimizer := experimentOptimizer(t, 0.5, -0.75)
	for step := 1; step <= 18; step++ {
		update, err := optimizer.Update(experimentGradients(0, 0))
		if err != nil {
			t.Fatal(err)
		}
		if update.Step != step || update.GradientNorm != 0 || update.ClipCoefficient != 1 {
			t.Fatal("invalid zero-gradient update")
		}
	}
	before := optimizer.Snapshot()
	if !slices.Equal(before.Parameters, []float32{0.5, -0.75}) {
		t.Fatal("zero weight decay changed parameters")
	}
	if _, err := optimizer.Update(experimentGradients(0.5, 0.25)); err == nil {
		t.Fatal("nineteenth update admitted")
	}
	if !reflect.DeepEqual(before, optimizer.Snapshot()) {
		t.Fatal("rejected nineteenth update changed state")
	}
	reloaded, err := services.RestoreExperimentAdamW(before)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := reloaded.Update(experimentGradients(0, 0)); err == nil {
		t.Fatal("reload bypassed maximum updates")
	}
}
