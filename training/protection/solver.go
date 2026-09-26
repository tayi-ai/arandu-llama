// Package protection computes constrained updates and validates materialized candidates.
package protection

import (
	"context"
	"errors"
	"fmt"
	"math"
)

// ErrProblem identifies invalid dimensions, identifiers, values or solver policy.
var ErrProblem = errors.New("protection: invalid problem or policy")

// ErrLimit identifies exhausted or exceeded resource limits.
var ErrLimit = errors.New("protection: resource limit exceeded")

// ErrNumerical identifies nonfinite intermediate arithmetic.
var ErrNumerical = errors.New("protection: nonfinite arithmetic")

// ErrInfeasible identifies a constraint with a zero Jacobian and an unmet floor.
var ErrInfeasible = errors.New("protection: infeasible zero constraint")

// ErrNotConverged means no numerical KKT certificate was obtained within policy.
// It does not distinguish general infeasibility from slow convergence.
var ErrNotConverged = errors.New("protection: convergence not certified")

const (
	maxParameters   = 1 << 20
	maxConstraints  = 256
	maxCoefficients = 8 << 20
	maxSweeps       = 100000
	maxWork         = 1000000000
)

// Constraint requires Margin + Jacobian dot Step >= Floor for one unique ID.
type Constraint struct {
	ID       string
	Margin   float64
	Floor    float64
	Jacobian []float64
}

// Problem minimizes Gradient dot Step + Lambda/2 * ||Step||^2.
// Lambda must be strictly positive. Inputs must remain immutable during Solve.
type Problem struct {
	Gradient    []float64
	Lambda      float64
	Constraints []Constraint
}

// Config bounds memory, scalar visits and sweeps, and sets absolute KKT tolerances.
// All fields must be positive. Limits may only tighten the hard ceilings used by
// DefaultConfig, except MaxSweeps (ceiling 100000) and MaxWork (ceiling 1e9).
// MaxCoefficients bounds Parameters * Constraints without building a Gram matrix.
// MaxWork counts scalar vector visits, not wall time or exact CPU instructions.
type Config struct {
	MaxParameters            int
	MaxConstraints           int
	MaxCoefficients          int
	MaxSweeps                int
	MaxWork                  uint64
	PrimalTolerance          float64
	StationarityTolerance    float64
	ComplementarityTolerance float64
}

// DefaultConfig admits at most 1048576 parameters, 256 constraints and 8388608
// coefficients, with 1000 sweeps, 500000000 scalar visits and 1e-8 tolerances.
// These are numerical resource defaults, not scientifically qualified thresholds.
func DefaultConfig() Config {
	return Config{maxParameters, maxConstraints, maxCoefficients, 1000, 500000000, 1e-8, 1e-8, 1e-8}
}

// Certificate reports independently recomputed floating-point KKT residuals.
// PrimalViolation is max(0, Floor-Margin-H*a); StationarityResidual is the
// infinity norm of Lambda*a+g-H^T*u; ComplementarityResidual is max |u_i*slack_i|.
// Multipliers are nonnegative by construction. Acceptance requires all three
// residuals within Config tolerances; this is not an exact-arithmetic proof or a
// guarantee about nonlinear model margins. Candidate evaluation remains required.
type Certificate struct {
	PrimalViolation         float64
	StationarityResidual    float64
	ComplementarityResidual float64
	Objective               float64
	Sweeps                  int
	Work                    uint64
}

// Solution contains a certified step and multipliers in input constraint order.
// Solve returns an empty Solution on every error; no uncertified step escapes.
type Solution struct {
	Step        []float64
	Multipliers []float64
	Certificate Certificate
}

type workBudget struct {
	ctx         context.Context
	used, limit uint64
}

func (w *workBudget) spend(n int) error {
	if err := w.ctx.Err(); err != nil {
		return err
	}
	if uint64(n) > w.limit-w.used {
		return ErrLimit
	}
	w.used += uint64(n)
	return nil
}

func checkContext(ctx context.Context, i int) error {
	if i&4095 == 0 {
		return ctx.Err()
	}
	return nil
}

// Solve uses deterministic cyclic Hildreth dual coordinate ascent. For each row,
// u_i <- max(0, u_i + (Floor_i-Margin_i-H_i*a)/(||H_i||^2/Lambda)), followed by
// a <- a + (u_i-new minus u_i-old)*H_i/Lambda. The starting point is -g/Lambda.
// Each coordinate maximizes the concave dual with all other coordinates fixed.
// For this strictly convex objective, primal feasibility, dual nonnegativity,
// stationarity and complementary slackness characterize the unique optimum.
// Residuals are recomputed from the original problem after every complete sweep;
// a small coordinate change alone never certifies convergence. Degenerate,
// incompatible or poorly scaled systems may exhaust policy and are refused.
// The implementation has O(parameters + constraints) additional memory.
func Solve(ctx context.Context, problem Problem, config Config) (Solution, error) {
	if ctx == nil || !validConfig(config) || !finite(problem.Lambda) || problem.Lambda <= 0 || len(problem.Gradient) == 0 {
		return Solution{}, ErrProblem
	}
	n, m := len(problem.Gradient), len(problem.Constraints)
	if n > config.MaxParameters || m > config.MaxConstraints || m != 0 && n > config.MaxCoefficients/m {
		return Solution{}, ErrLimit
	}
	budget := workBudget{ctx: ctx, limit: config.MaxWork}
	if err := budget.spend(n); err != nil {
		return Solution{}, err
	}
	a := make([]float64, n)
	for j, g := range problem.Gradient {
		if err := checkContext(ctx, j); err != nil {
			return Solution{}, err
		}
		if !finite(g) {
			return Solution{}, ErrProblem
		}
		a[j] = -g / problem.Lambda
		if !finite(a[j]) {
			return Solution{}, ErrNumerical
		}
	}
	norms := make([]float64, m)
	bounds := make([]float64, m)
	ids := make(map[string]struct{}, m)
	for i, row := range problem.Constraints {
		if _, exists := ids[row.ID]; exists || row.ID == "" || len(row.ID) > 1024 || len(row.Jacobian) != n || !finite(row.Margin) || !finite(row.Floor) {
			return Solution{}, ErrProblem
		}
		ids[row.ID] = struct{}{}
		bounds[i] = row.Floor - row.Margin
		if !finite(bounds[i]) {
			return Solution{}, ErrNumerical
		}
		var err error
		norms[i], err = dot(&budget, row.Jacobian, row.Jacobian)
		if err != nil {
			return Solution{}, err
		}
		if norms[i] == 0 {
			// A nonzero row whose squared norm underflows is not a zero constraint.
			if err := budget.spend(n); err != nil {
				return Solution{}, err
			}
			for j, value := range row.Jacobian {
				if err := checkContext(ctx, j); err != nil {
					return Solution{}, err
				}
				if value != 0 {
					return Solution{}, ErrNumerical
				}
			}
			if bounds[i] > 0 {
				return Solution{}, ErrInfeasible
			}
		}
	}
	u := make([]float64, m)
	residual := make([]float64, n)
	correction := make([]float64, n)
	for sweep := 0; sweep <= config.MaxSweeps; sweep++ {
		certificate, err := certify(&budget, problem, a, u, bounds, residual, correction)
		if err != nil {
			return Solution{}, err
		}
		certificate.Sweeps, certificate.Work = sweep, budget.used
		if certificate.PrimalViolation <= config.PrimalTolerance && certificate.StationarityResidual <= config.StationarityTolerance && certificate.ComplementarityResidual <= config.ComplementarityTolerance {
			if err := ctx.Err(); err != nil {
				return Solution{}, err
			}
			return Solution{a, u, certificate}, nil
		}
		if sweep == config.MaxSweeps {
			return Solution{}, fmt.Errorf("%w: primal=%g stationarity=%g complementarity=%g after %d sweeps", ErrNotConverged, certificate.PrimalViolation, certificate.StationarityResidual, certificate.ComplementarityResidual, sweep)
		}
		for i, row := range problem.Constraints {
			if norms[i] == 0 {
				continue
			}
			projection, err := dot(&budget, row.Jacobian, a)
			if err != nil {
				return Solution{}, err
			}
			increment := (bounds[i] - projection) / norms[i] * problem.Lambda
			next := math.Max(0, u[i]+increment)
			scale := (next - u[i]) / problem.Lambda
			if !finite(increment) || !finite(next) || !finite(scale) {
				return Solution{}, ErrNumerical
			}
			if err := budget.spend(n); err != nil {
				return Solution{}, err
			}
			for j, value := range row.Jacobian {
				if err := checkContext(ctx, j); err != nil {
					return Solution{}, err
				}
				a[j] += scale * value
				if !finite(a[j]) {
					return Solution{}, ErrNumerical
				}
			}
			u[i] = next
		}
	}
	return Solution{}, ErrNotConverged
}

func certify(budget *workBudget, problem Problem, a, u, bounds, residual, correction []float64) (Certificate, error) {
	var result Certificate
	if err := budget.spend(len(a)); err != nil {
		return result, err
	}
	for j := range a {
		if err := checkContext(budget.ctx, j); err != nil {
			return result, err
		}
		residual[j], correction[j] = 0, 0
	}
	for i, row := range problem.Constraints {
		projection, err := dot(budget, row.Jacobian, a)
		if err != nil {
			return result, err
		}
		slack := projection - bounds[i]
		product := u[i] * slack
		if !finite(slack) || !finite(product) || !finite(u[i]) || u[i] < 0 {
			return result, ErrNumerical
		}
		result.PrimalViolation = math.Max(result.PrimalViolation, -slack)
		result.ComplementarityResidual = math.Max(result.ComplementarityResidual, math.Abs(product))
		if err := budget.spend(len(a)); err != nil {
			return result, err
		}
		for j, value := range row.Jacobian {
			if err := checkContext(budget.ctx, j); err != nil {
				return result, err
			}
			term := u[i] * value
			if !finite(term) {
				return result, ErrNumerical
			}
			residual[j], correction[j] = addCompensated(residual[j], correction[j], term)
		}
	}
	if err := budget.spend(len(a)); err != nil {
		return result, err
	}
	for j, value := range a {
		if err := checkContext(budget.ctx, j); err != nil {
			return result, err
		}
		r := problem.Lambda*value + problem.Gradient[j] - (residual[j] + correction[j])
		if !finite(r) {
			return result, ErrNumerical
		}
		result.StationarityResidual = math.Max(result.StationarityResidual, math.Abs(r))
	}
	linear, err := dot(budget, problem.Gradient, a)
	if err != nil {
		return result, err
	}
	squared, err := dot(budget, a, a)
	if err != nil {
		return result, err
	}
	result.Objective = linear + problem.Lambda/2*squared
	if !finite(result.Objective) {
		return result, ErrNumerical
	}
	return result, nil
}

func dot(budget *workBudget, a, b []float64) (float64, error) {
	if err := budget.spend(len(a)); err != nil {
		return 0, err
	}
	var sum, correction float64
	for i, value := range a {
		if err := checkContext(budget.ctx, i); err != nil {
			return 0, err
		}
		if !finite(value) || !finite(b[i]) {
			return 0, ErrProblem
		}
		product := value * b[i]
		if !finite(product) {
			return 0, ErrNumerical
		}
		sum, correction = addCompensated(sum, correction, product)
	}
	result := sum + correction
	if !finite(result) {
		return 0, ErrNumerical
	}
	return result, nil
}

func addCompensated(sum, correction, value float64) (float64, float64) {
	next := sum + value
	if math.Abs(sum) >= math.Abs(value) {
		correction += (sum - next) + value
	} else {
		correction += (value - next) + sum
	}
	return next, correction
}

func validConfig(c Config) bool {
	return c.MaxParameters > 0 && c.MaxParameters <= maxParameters && c.MaxConstraints > 0 && c.MaxConstraints <= maxConstraints && c.MaxCoefficients > 0 && c.MaxCoefficients <= maxCoefficients && c.MaxSweeps > 0 && c.MaxSweeps <= maxSweeps && c.MaxWork > 0 && c.MaxWork <= maxWork && finite(c.PrimalTolerance) && c.PrimalTolerance > 0 && finite(c.StationarityTolerance) && c.StationarityTolerance > 0 && finite(c.ComplementarityTolerance) && c.ComplementarityTolerance > 0
}

func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }
