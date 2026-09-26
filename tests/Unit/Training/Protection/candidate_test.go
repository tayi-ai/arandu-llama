package protection_test

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/protection"
)

func TestNonlinearRealRegressionRollsBackCertifiedLinearCandidate(t *testing.T) {
	// m(theta)=1-theta^2 has derivative zero at theta=0. The linear prediction
	// admits theta=1, whose real margin 0 violates the anchored floor 0.5.
	problem := protection.Problem{Gradient: []float64{-1}, Lambda: 1, Constraints: []protection.Constraint{{ID: "nonlinear", Margin: 1, Floor: 0.5, Jacobian: []float64{0}}}}
	solution, err := protection.Solve(context.Background(), problem, protection.DefaultConfig())
	if err != nil || solution.Step[0] != 1 {
		t.Fatalf("linear proposal: %+v %v", solution, err)
	}
	state := []float32{0}
	calls := 0
	callbacks := protection.CandidateCallbacks{
		Apply: func(_ context.Context, values []float32) error { state = append([]float32(nil), values...); return nil },
		Measure: func(context.Context) ([]protection.Margin, error) {
			calls++
			return []protection.Margin{{ID: "nonlinear", Value: 1 - float64(state[0]*state[0])}}, nil
		},
		Restore: func(_ context.Context, values []float32) error { state = append([]float32(nil), values...); return nil },
	}
	got, err := protection.EvaluateCandidate(context.Background(), state, []float32{float32(solution.Step[0])}, []protection.Floor{{ID: "nonlinear", Value: 0.5}}, callbacks)
	if !errors.Is(err, protection.ErrRejected) || got.Accepted || !got.Restored || calls != 1 || state[0] != 0 {
		t.Fatalf("failed rollback: %+v %v state=%v", got, err, state)
	}
}

func TestAcceptsAllMeasuredFloorsByIDAndPreservesInputs(t *testing.T) {
	prior, candidate := []float32{1, 2}, []float32{3, 4}
	floors := []protection.Floor{{ID: "a", Value: 5}, {ID: "b", Value: 10}}
	state := append([]float32(nil), prior...)
	restored := false
	got, err := protection.EvaluateCandidate(context.Background(), prior, candidate, floors, protection.CandidateCallbacks{
		Apply: func(_ context.Context, values []float32) error {
			state = append([]float32(nil), values...)
			values[0] = 99
			return nil
		},
		Measure: func(context.Context) ([]protection.Margin, error) {
			return []protection.Margin{{ID: "b", Value: 10}, {ID: "a", Value: 6}}, nil
		},
		Restore: func(context.Context, []float32) error { restored = true; return nil },
	})
	if err != nil || !got.Accepted || got.Restored || restored || got.Margins[0].ID != "a" || !reflect.DeepEqual(state, candidate) || !reflect.DeepEqual(candidate, []float32{3, 4}) || !reflect.DeepEqual(prior, []float32{1, 2}) {
		t.Fatalf("acceptance: %+v %v", got, err)
	}
}

func TestCandidateFailuresRestoreEvenAfterPartialApplyAndCancellation(t *testing.T) {
	failure := errors.New("injected failure")
	cases := []string{"apply_error", "measure_error", "cancel_apply", "cancel_measure", "apply_panic", "measure_panic", "floor_violation"}
	for _, name := range cases {
		t.Run(name, func(t *testing.T) {
			ctx, cancel := context.WithCancel(context.Background())
			defer cancel()
			prior, candidate, state := []float32{1}, []float32{2}, float32(1)
			floors := []protection.Floor{{ID: "a", Value: 5}}
			restores := 0
			callbacks := protection.CandidateCallbacks{
				Apply: func(_ context.Context, values []float32) error {
					state = values[0]
					// External aliases must not corrupt the rollback or anchored floor.
					prior[0], candidate[0], floors[0].Value = 90, 91, -100
					if name == "apply_error" {
						return failure
					}
					if name == "apply_panic" {
						panic("injected")
					}
					if name == "cancel_apply" {
						cancel()
					}
					return nil
				},
				Measure: func(context.Context) ([]protection.Margin, error) {
					if name == "measure_error" {
						return nil, failure
					}
					if name == "measure_panic" {
						panic("injected")
					}
					if name == "cancel_measure" {
						cancel()
					}
					return []protection.Margin{{ID: "a", Value: 4}}, nil
				},
				Restore: func(ctx context.Context, values []float32) error {
					restores++
					if ctx.Err() != nil {
						t.Fatal("rollback inherited cancellation")
					}
					if _, exists := ctx.Deadline(); !exists {
						t.Fatal("rollback has no deadline")
					}
					state = values[0]
					return nil
				},
			}
			got, err := protection.EvaluateCandidate(ctx, prior, candidate, floors, callbacks)
			if err == nil || got.Accepted || !got.Restored || restores != 1 || state != 1 {
				t.Fatalf("rollback: %+v %v state=%g", got, err, state)
			}
			want := protection.ErrRejected
			switch name {
			case "apply_error", "measure_error":
				want = failure
			case "cancel_apply", "cancel_measure":
				want = context.Canceled
			case "apply_panic", "measure_panic":
				want = protection.ErrCallbackPanic
			}
			if !errors.Is(err, want) {
				t.Fatalf("lost cause: %v", err)
			}
		})
	}
}

func TestRejectsMissingExtraDuplicateUnknownAndNonfiniteMargins(t *testing.T) {
	cases := [][]protection.Margin{
		{{ID: "a", Value: 5}},
		{{ID: "a", Value: 5}, {ID: "b", Value: 5}, {ID: "c", Value: 5}},
		{{ID: "a", Value: 5}, {ID: "a", Value: 5}},
		{{ID: "a", Value: 5}, {ID: "unknown", Value: 5}},
		{{ID: "a", Value: 5}, {ID: "b", Value: math.NaN()}},
		{{ID: "a", Value: 5}, {ID: "b", Value: math.Inf(1)}},
		{{ID: "a", Value: 5}, {ID: "b", Value: math.Inf(-1)}},
	}
	for i, margins := range cases {
		state := float32(1)
		got, err := protection.EvaluateCandidate(context.Background(), []float32{1}, []float32{2}, []protection.Floor{{ID: "a", Value: 5}, {ID: "b", Value: 5}}, protection.CandidateCallbacks{
			Apply:   func(_ context.Context, values []float32) error { state = values[0]; return nil },
			Measure: func(context.Context) ([]protection.Margin, error) { return margins, nil },
			Restore: func(_ context.Context, values []float32) error { state = values[0]; return nil },
		})
		if !errors.Is(err, protection.ErrCandidate) || got.Accepted || !got.Restored || state != 1 {
			t.Fatalf("case %d: %+v %v", i, got, err)
		}
	}
}

func TestInvalidCandidateInputsHaveNoEffects(t *testing.T) {
	calls := 0
	callbacks := protection.CandidateCallbacks{
		Apply:   func(context.Context, []float32) error { calls++; return nil },
		Measure: func(context.Context) ([]protection.Margin, error) { calls++; return nil, nil },
		Restore: func(context.Context, []float32) error { calls++; return nil },
	}
	cases := []struct {
		prior, candidate []float32
		floors           []protection.Floor
	}{
		{nil, nil, []protection.Floor{{ID: "a"}}},
		{[]float32{1}, []float32{1, 2}, []protection.Floor{{ID: "a"}}},
		{[]float32{1}, []float32{float32(math.Inf(1))}, []protection.Floor{{ID: "a"}}},
		{[]float32{float32(math.NaN())}, []float32{1}, []protection.Floor{{ID: "a"}}},
		{[]float32{1}, []float32{2}, nil},
		{[]float32{1}, []float32{2}, []protection.Floor{{ID: "a"}, {ID: "a"}}},
		{[]float32{1}, []float32{2}, []protection.Floor{{ID: ""}}},
		{[]float32{1}, []float32{2}, []protection.Floor{{ID: "a", Value: math.NaN()}}},
	}
	for i, item := range cases {
		got, err := protection.EvaluateCandidate(context.Background(), item.prior, item.candidate, item.floors, callbacks)
		if !errors.Is(err, protection.ErrCandidate) || got.Accepted || got.Restored {
			t.Fatalf("case %d: %+v %v", i, got, err)
		}
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := protection.EvaluateCandidate(ctx, []float32{1}, []float32{2}, []protection.Floor{{ID: "a"}}, callbacks); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if calls != 0 {
		t.Fatalf("invalid input caused %d effects", calls)
	}
}

func TestRollbackFailurePreservesBothCausesAndUnknownState(t *testing.T) {
	failure := errors.New("restore failed")
	for _, panicRestore := range []bool{false, true} {
		got, err := protection.EvaluateCandidate(context.Background(), []float32{1}, []float32{2}, []protection.Floor{{ID: "a", Value: 5}}, protection.CandidateCallbacks{
			Apply: func(context.Context, []float32) error { return nil },
			Measure: func(context.Context) ([]protection.Margin, error) {
				return []protection.Margin{{ID: "a", Value: 4}}, nil
			},
			Restore: func(context.Context, []float32) error {
				if panicRestore {
					panic("injected")
				}
				return failure
			},
		})
		want := failure
		if panicRestore {
			want = protection.ErrCallbackPanic
		}
		if !errors.Is(err, protection.ErrRejected) || !errors.Is(err, protection.ErrRollback) || !errors.Is(err, want) || got.Accepted || got.Restored {
			t.Fatalf("lost rollback error: %+v %v", got, err)
		}
	}
}
