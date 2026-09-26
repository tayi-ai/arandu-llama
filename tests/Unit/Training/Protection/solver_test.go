package protection_test

import (
	"context"
	"errors"
	"math"
	"reflect"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/protection"
)

func TestAnalyticQuadraticProgramsAndIndependentKKT(t *testing.T) {
	cases := []struct {
		name      string
		problem   protection.Problem
		want      []float64
		objective float64
	}{
		{"unconstrained", protection.Problem{Gradient: []float64{-2, 4}, Lambda: 2}, []float64{1, -2}, -5},
		{"one_oblique_face", protection.Problem{Gradient: []float64{2, 4}, Lambda: 2, Constraints: []protection.Constraint{{ID: "sum", Margin: 3, Floor: 4, Jacobian: []float64{1, 1}}}}, []float64{1, 0}, 3},
		{"two_coupled_faces", protection.Problem{Gradient: []float64{2, 0}, Lambda: 1, Constraints: []protection.Constraint{{ID: "x", Floor: 0, Jacobian: []float64{1, 0}}, {ID: "sum", Floor: 1, Jacobian: []float64{1, 1}}}}, []float64{0, 1}, 0.5},
		{"inactive_face", protection.Problem{Gradient: []float64{0, 0}, Lambda: 2, Constraints: []protection.Constraint{{ID: "x", Floor: 1, Jacobian: []float64{1, 0}}, {ID: "y", Floor: 2, Jacobian: []float64{0, 1}}, {ID: "sum", Floor: 2, Jacobian: []float64{1, 1}}}}, []float64{1, 2}, 5},
		{"dependent_faces", protection.Problem{Gradient: []float64{0}, Lambda: 2, Constraints: []protection.Constraint{{ID: "x", Floor: 1, Jacobian: []float64{1}}, {ID: "twice", Floor: 2, Jacobian: []float64{2}}, {ID: "zero", Margin: 4, Floor: 3, Jacobian: []float64{0}}}}, []float64{1}, 1},
		{"equality_faces", protection.Problem{Gradient: []float64{-4}, Lambda: 2, Constraints: []protection.Constraint{{ID: "lower", Floor: 1, Jacobian: []float64{1}}, {ID: "upper", Floor: -1, Jacobian: []float64{-1}}}}, []float64{1}, -3},
	}
	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			config := protection.DefaultConfig()
			got, err := protection.Solve(context.Background(), item.problem, config)
			if err != nil {
				t.Fatal(err)
			}
			for i, value := range item.want {
				if math.Abs(got.Step[i]-value) > 2e-8 {
					t.Fatalf("step %v, want %v", got.Step, item.want)
				}
			}
			if math.Abs(got.Certificate.Objective-item.objective) > 3e-8 {
				t.Fatalf("objective %g, want %g", got.Certificate.Objective, item.objective)
			}
			assertKKT(t, item.problem, got, config)
			repeated, err := protection.Solve(context.Background(), item.problem, config)
			if err != nil || !reflect.DeepEqual(got, repeated) {
				t.Fatalf("non-deterministic result: %+v, %v", repeated, err)
			}
		})
	}
}

func assertKKT(t *testing.T, problem protection.Problem, solution protection.Solution, config protection.Config) {
	t.Helper()
	var primal, stationarity, complementarity float64
	for i, row := range problem.Constraints {
		slack := row.Margin - row.Floor
		for j, value := range row.Jacobian {
			slack += value * solution.Step[j]
		}
		if solution.Multipliers[i] < 0 {
			t.Fatal("negative dual multiplier")
		}
		primal = math.Max(primal, -slack)
		complementarity = math.Max(complementarity, math.Abs(slack*solution.Multipliers[i]))
	}
	for j, value := range solution.Step {
		residual := problem.Lambda*value + problem.Gradient[j]
		for i, row := range problem.Constraints {
			residual -= row.Jacobian[j] * solution.Multipliers[i]
		}
		stationarity = math.Max(stationarity, math.Abs(residual))
	}
	if primal > config.PrimalTolerance || stationarity > config.StationarityTolerance || complementarity > config.ComplementarityTolerance {
		t.Fatalf("independent KKT check failed: %g %g %g", primal, stationarity, complementarity)
	}
	if math.Abs(solution.Certificate.PrimalViolation-primal) > 1e-14 || math.Abs(solution.Certificate.StationarityResidual-stationarity) > 1e-14 || math.Abs(solution.Certificate.ComplementarityResidual-complementarity) > 1e-14 {
		t.Fatal("certificate differs from independent residuals")
	}
	if solution.Certificate.Work == 0 || solution.Certificate.Work > config.MaxWork || solution.Certificate.Sweeps > config.MaxSweeps {
		t.Fatal("work is not bounded")
	}
}

func TestRefusesIncompatibleAndUnconvergedProblemsWithoutCandidate(t *testing.T) {
	cases := []struct {
		name        string
		constraints []protection.Constraint
		want        error
	}{
		{"zero_impossible", []protection.Constraint{{ID: "impossible", Floor: 1, Jacobian: []float64{0}}}, protection.ErrInfeasible},
		{"incompatible", []protection.Constraint{{ID: "lower", Floor: 1, Jacobian: []float64{1}}, {ID: "upper", Floor: 0, Jacobian: []float64{-1}}}, protection.ErrNotConverged},
	}
	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			config := protection.DefaultConfig()
			config.MaxSweeps = 12
			got, err := protection.Solve(context.Background(), protection.Problem{Gradient: []float64{0}, Lambda: 1, Constraints: item.constraints}, config)
			if !errors.Is(err, item.want) || len(got.Step) != 0 || len(got.Multipliers) != 0 {
				t.Fatalf("got %+v, %v", got, err)
			}
		})
	}
	config := protection.DefaultConfig()
	config.MaxSweeps = 1
	got, err := protection.Solve(context.Background(), protection.Problem{Gradient: []float64{2, 0}, Lambda: 1, Constraints: []protection.Constraint{{ID: "x", Jacobian: []float64{1, 0}}, {ID: "sum", Floor: 1, Jacobian: []float64{1, 1}}}}, config)
	if !errors.Is(err, protection.ErrNotConverged) || len(got.Step) != 0 {
		t.Fatalf("unfinished feasible solve escaped: %+v %v", got, err)
	}
}

func TestRejectsInvalidOrNonfiniteInputsAndArithmetic(t *testing.T) {
	cases := []struct {
		name    string
		problem protection.Problem
		want    error
	}{
		{"empty", protection.Problem{Lambda: 1}, protection.ErrProblem},
		{"lambda_zero", protection.Problem{Gradient: []float64{1}}, protection.ErrProblem},
		{"lambda_negative", protection.Problem{Gradient: []float64{1}, Lambda: -1}, protection.ErrProblem},
		{"lambda_nan", protection.Problem{Gradient: []float64{1}, Lambda: math.NaN()}, protection.ErrProblem},
		{"gradient_inf", protection.Problem{Gradient: []float64{math.Inf(1)}, Lambda: 1}, protection.ErrProblem},
		{"shape", protection.Problem{Gradient: []float64{1}, Lambda: 1, Constraints: []protection.Constraint{{ID: "x"}}}, protection.ErrProblem},
		{"jacobian_nan", protection.Problem{Gradient: []float64{1}, Lambda: 1, Constraints: []protection.Constraint{{ID: "x", Jacobian: []float64{math.NaN()}}}}, protection.ErrProblem},
		{"margin_inf", protection.Problem{Gradient: []float64{1}, Lambda: 1, Constraints: []protection.Constraint{{ID: "x", Margin: math.Inf(1), Jacobian: []float64{1}}}}, protection.ErrProblem},
		{"floor_nan", protection.Problem{Gradient: []float64{1}, Lambda: 1, Constraints: []protection.Constraint{{ID: "x", Floor: math.NaN(), Jacobian: []float64{1}}}}, protection.ErrProblem},
		{"duplicate_id", protection.Problem{Gradient: []float64{1}, Lambda: 1, Constraints: []protection.Constraint{{ID: "x", Jacobian: []float64{1}}, {ID: "x", Jacobian: []float64{1}}}}, protection.ErrProblem},
		{"norm_overflow", protection.Problem{Gradient: []float64{1}, Lambda: 1, Constraints: []protection.Constraint{{ID: "x", Jacobian: []float64{math.MaxFloat64}}}}, protection.ErrNumerical},
		{"norm_underflow", protection.Problem{Gradient: []float64{1}, Lambda: 1, Constraints: []protection.Constraint{{ID: "x", Jacobian: []float64{math.SmallestNonzeroFloat64}}}}, protection.ErrNumerical},
		{"step_overflow", protection.Problem{Gradient: []float64{math.MaxFloat64}, Lambda: math.SmallestNonzeroFloat64}, protection.ErrNumerical},
	}
	for _, item := range cases {
		t.Run(item.name, func(t *testing.T) {
			got, err := protection.Solve(context.Background(), item.problem, protection.DefaultConfig())
			if !errors.Is(err, item.want) || len(got.Step) != 0 {
				t.Fatalf("got %+v, %v", got, err)
			}
		})
	}
}

func TestResourcePolicyCancellationAndInputOwnership(t *testing.T) {
	problem := protection.Problem{Gradient: []float64{-2, 4}, Lambda: 2, Constraints: []protection.Constraint{{ID: "x", Floor: 3, Jacobian: []float64{1, 1}}}}
	for _, edit := range []func(*protection.Config){
		func(c *protection.Config) { c.MaxParameters = 1 },
		func(c *protection.Config) { c.MaxCoefficients = 1 },
		func(c *protection.Config) { c.MaxWork = 1 },
	} {
		config := protection.DefaultConfig()
		edit(&config)
		got, err := protection.Solve(context.Background(), problem, config)
		if !errors.Is(err, protection.ErrLimit) || len(got.Step) != 0 {
			t.Fatalf("got %+v, %v", got, err)
		}
	}
	config := protection.DefaultConfig()
	config.StationarityTolerance = math.Inf(1)
	if _, err := protection.Solve(context.Background(), problem, config); !errors.Is(err, protection.ErrProblem) {
		t.Fatal(err)
	}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := protection.Solve(ctx, problem, protection.DefaultConfig()); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	if _, err := protection.Solve(nil, problem, protection.DefaultConfig()); !errors.Is(err, protection.ErrProblem) {
		t.Fatal(err)
	}
	controlled := &cancelDuringSolve{Context: context.Background(), remaining: 12}
	if got, err := protection.Solve(controlled, problem, protection.DefaultConfig()); !errors.Is(err, context.Canceled) || len(got.Step) != 0 {
		t.Fatalf("mid-solve cancellation: %+v %v", got, err)
	}
	_, err := protection.Solve(context.Background(), problem, protection.DefaultConfig())
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(problem.Gradient, []float64{-2, 4}) || !reflect.DeepEqual(problem.Constraints[0].Jacobian, []float64{1, 1}) {
		t.Fatal("solver mutated caller inputs")
	}
}

type cancelDuringSolve struct {
	context.Context
	remaining int
}

func (c *cancelDuringSolve) Err() error {
	c.remaining--
	if c.remaining <= 0 {
		return context.Canceled
	}
	return nil
}
