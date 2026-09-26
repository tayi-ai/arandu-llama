package protection_test

import (
	"context"
	"errors"
	"fmt"
	"math"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/protection"
)

func TestExplicitCoefficientBudgetKeepsEveryConstraint(t *testing.T) {
	const parameters, constraints = 557056, 33
	config := protection.DefaultConfig()
	if config.MaxCoefficients != 8<<20 {
		t.Fatal("default allocation budget changed")
	}
	if err := protection.ValidateBounds(config, parameters, constraints); !errors.Is(err, protection.ErrLimit) {
		t.Fatalf("default must refuse the entire pool, never truncate: %v", err)
	}
	config.MaxCoefficients = parameters * constraints
	if err := protection.ValidateBounds(config, parameters, constraints); err != nil {
		t.Fatal(err)
	}
	problem := protection.Problem{Lambda: 2, Gradient: make([]float64, parameters)}
	for i := range problem.Gradient {
		problem.Gradient[i] = -2
	}
	for i := range constraints {
		row := protection.Constraint{ID: fmt.Sprintf("pair-%d", i), Floor: 2, Jacobian: make([]float64, parameters)}
		row.Jacobian[i] = 1
		problem.Constraints = append(problem.Constraints, row)
	}
	result, err := protection.Solve(context.Background(), problem, config)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Multipliers) != constraints || len(result.Step) != parameters {
		t.Fatal("solver dropped constraints or parameters")
	}
	for i, value := range result.Step {
		want := 1.0
		if i < constraints {
			want = 2
		}
		if value != want {
			t.Fatalf("parameter %d: got %g, want %g", i, value, want)
		}
	}
	for i, value := range result.Multipliers {
		if value != 2 {
			t.Fatalf("constraint %d not active: multiplier %g", i, value)
		}
	}
	if result.Certificate.PrimalViolation != 0 || result.Certificate.StationarityResidual != 0 || result.Certificate.ComplementarityResidual != 0 {
		t.Fatalf("analytic solution certificate differs: %+v", result.Certificate)
	}
}

func TestBoundsRefuseInvalidPolicyBeforeAllocation(t *testing.T) {
	for name, change := range map[string]func(*protection.Config){
		"coefficient_ceiling": func(c *protection.Config) { c.MaxCoefficients = 32<<20 + 1 },
		"parameter_ceiling":   func(c *protection.Config) { c.MaxParameters = 1<<20 + 1 },
		"constraint_ceiling":  func(c *protection.Config) { c.MaxConstraints = 257 },
		"sweep_ceiling":       func(c *protection.Config) { c.MaxSweeps = 100001 },
		"work_ceiling":        func(c *protection.Config) { c.MaxWork = 1000000001 },
		"nonfinite_tolerance": func(c *protection.Config) { c.PrimalTolerance = math.NaN() },
	} {
		t.Run(name, func(t *testing.T) {
			config := protection.DefaultConfig()
			change(&config)
			if err := protection.ValidateBounds(config, 1, 1); !errors.Is(err, protection.ErrProblem) {
				t.Fatalf("invalid policy admitted: %v", err)
			}
		})
	}
	for _, shape := range [][2]int{{0, 1}, {1, -1}} {
		if err := protection.ValidateBounds(protection.DefaultConfig(), shape[0], shape[1]); !errors.Is(err, protection.ErrProblem) {
			t.Fatalf("invalid dimensions admitted: %v", err)
		}
	}
}
