package unit_test

import (
	"math"
	"testing"

	services "github.com/tayi-ai/arandu-llama/training/zerothorder"
)

func TestConstrainedUpdateRespectsAnchoredFloorAndStepBound(t *testing.T) {
	solution, err := services.SolveConstrainedUpdate(services.ConstrainedUpdate{
		Gradient: []float64{-2, 0}, Lambda: 2, StepBound: 3,
		Margins: []float64{0}, Floors: []float64{2},
		MarginEffects: [][]float64{{1, 0}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(solution.Coefficients[0]-2) > 1e-7 || math.Abs(solution.Coefficients[1]) > 1e-7 {
		t.Fatalf("coefficients = %v", solution.Coefficients)
	}
	if solution.PredictedMargins[0] < 2-1e-7 {
		t.Fatalf("predicted floor was violated: %v", solution.PredictedMargins)
	}
}

func TestConstrainedUpdateLeavesFeasibleUnconstrainedMinimumAlone(t *testing.T) {
	solution, err := services.SolveConstrainedUpdate(services.ConstrainedUpdate{
		Gradient: []float64{-2, 4}, Lambda: 2, StepBound: 4,
		Margins: []float64{3}, Floors: []float64{2},
		MarginEffects: [][]float64{{1, 0}},
	})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(solution.Coefficients[0]-1) > 1e-8 || math.Abs(solution.Coefficients[1]+2) > 1e-8 {
		t.Fatalf("coefficients = %v, want [1 -2]", solution.Coefficients)
	}
}

func TestConstrainedUpdateRefusesFloorOutsideStepBall(t *testing.T) {
	_, err := services.SolveConstrainedUpdate(services.ConstrainedUpdate{
		Gradient: []float64{0}, Lambda: 1, StepBound: 1,
		Margins: []float64{0}, Floors: []float64{2},
		MarginEffects: [][]float64{{1}}, MaxIterations: 200,
	})
	if err == nil {
		t.Fatal("infeasible anchored floor was accepted")
	}
}

func TestConstrainedUpdateRefusesContradictoryFloors(t *testing.T) {
	_, err := services.SolveConstrainedUpdate(services.ConstrainedUpdate{
		Gradient: []float64{0}, Lambda: 1, StepBound: 10,
		Margins: []float64{0, 0}, Floors: []float64{2, 2},
		MarginEffects: [][]float64{{1}, {-1}}, MaxIterations: 500,
	})
	if err == nil {
		t.Fatal("contradictory anchored floors were accepted")
	}
}

func TestOneDirectionProjectionRemainsStableAcrossLargeScaleDifference(t *testing.T) {
	solution, err := services.SolveConstrainedUpdate(services.ConstrainedUpdate{
		Gradient: []float64{214.893336175}, Lambda: 1, StepBound: 0.001,
		Margins: []float64{0.6587905883789062}, Floors: []float64{0.4940929412841797},
		MarginEffects: [][]float64{{172.1496284008026}}, Tolerance: 1e-10,
	})
	if err != nil {
		t.Fatal(err)
	}
	want := (0.4940929412841797 - 0.6587905883789062) / 172.1496284008026
	if math.Abs(solution.Coefficients[0]-want) > 1e-12 {
		t.Fatalf("coefficient = %.15g, want %.15g", solution.Coefficients[0], want)
	}
	if solution.PredictedMargins[0] < 0.4940929412841797-1e-10 {
		t.Fatalf("predicted floor was violated: %v", solution.PredictedMargins)
	}
}
