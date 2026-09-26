package zerothorder

import (
	"errors"
	"math"
)

// ConstrainedUpdate defines a bounded convex update in the direction basis.
// Gradient and every row of MarginEffects share the same direction
// basis. Floors are anchored values, not recomputed after a candidate step.
type ConstrainedUpdate struct {
	Gradient      []float64
	Margins       []float64
	MarginEffects [][]float64
	Floors        []float64
	Lambda        float64
	StepBound     float64
	Tolerance     float64
	MaxIterations int
}

// ConstrainedSolution reports coefficients and their predicted protected margins.
type ConstrainedSolution struct {
	Coefficients     []float64 `json:"coefficients"`
	PredictedMargins []float64 `json:"predicted_margins"`
	Iterations       int       `json:"iterations"`
}

// SolveConstrainedUpdate projects the unconstrained minimizer -g/lambda onto
// the intersection of the anchored margin halfspaces and the configured L2
// step ball. Dykstra's algorithm gives the Euclidean projection required by
// the isotropic quadratic objective without weakening any floor.
func SolveConstrainedUpdate(problem ConstrainedUpdate) (ConstrainedSolution, error) {
	directions := len(problem.Gradient)
	if directions == 0 || problem.Lambda <= 0 || problem.StepBound <= 0 {
		return ConstrainedSolution{}, errors.New("cluster: constrained update needs directions, positive lambda and positive step bound")
	}
	if len(problem.Margins) != len(problem.Floors) || len(problem.Margins) != len(problem.MarginEffects) {
		return ConstrainedSolution{}, errors.New("cluster: margin, floor and effect row counts differ")
	}
	for _, row := range problem.MarginEffects {
		if len(row) != directions {
			return ConstrainedSolution{}, errors.New("cluster: margin effect width differs from direction count")
		}
	}
	tolerance := problem.Tolerance
	if tolerance <= 0 {
		tolerance = 1e-9
	}
	maxIterations := problem.MaxIterations
	if maxIterations <= 0 {
		maxIterations = 20_000
	}
	if directions == 1 {
		return solveOneDirection(problem, tolerance)
	}

	x := make([]float64, directions)
	for i, gradient := range problem.Gradient {
		x[i] = -gradient / problem.Lambda
	}
	sets := len(problem.MarginEffects) + 1
	corrections := make([][]float64, sets)
	for i := range corrections {
		corrections[i] = make([]float64, directions)
	}

	for iteration := 1; iteration <= maxIterations; iteration++ {
		movement := 0.0
		for set := 0; set < sets; set++ {
			y := make([]float64, directions)
			for j := range y {
				y[j] = x[j] + corrections[set][j]
			}
			projected := append([]float64(nil), y...)
			if set < len(problem.MarginEffects) {
				row := problem.MarginEffects[set]
				required := problem.Floors[set] - problem.Margins[set]
				normSquared := dot(row, row)
				if normSquared == 0 && required > tolerance {
					return ConstrainedSolution{}, errors.New("cluster: anchored floor has no measurable direction and is infeasible")
				}
				if shortfall := required - dot(row, projected); shortfall > 0 && normSquared > 0 {
					for j := range projected {
						projected[j] += shortfall * row[j] / normSquared
					}
				}
			} else if norm := math.Sqrt(dot(projected, projected)); norm > problem.StepBound {
				scale := problem.StepBound / norm
				for j := range projected {
					projected[j] *= scale
				}
			}
			for j := range x {
				corrections[set][j] = y[j] - projected[j]
				delta := math.Abs(projected[j] - x[j])
				if delta > movement {
					movement = delta
				}
			}
			x = projected
		}
		if movement <= tolerance && feasible(problem, x, tolerance) {
			return ConstrainedSolution{Coefficients: x, PredictedMargins: predictMargins(problem, x), Iterations: iteration}, nil
		}
	}
	return ConstrainedSolution{}, errors.New("cluster: anchored margin constraints are infeasible within the step bound")
}

// solveOneDirection intersects the halfspaces as an interval and projects the
// unconstrained minimum onto it exactly. Dykstra is useful for several
// directions, but in one dimension a very large gradient and a milliscale
// step ball can lose the final tolerance to repeated floating-point
// corrections even though coefficient zero proves the problem is feasible.
func solveOneDirection(problem ConstrainedUpdate, tolerance float64) (ConstrainedSolution, error) {
	lower, upper := -problem.StepBound, problem.StepBound
	for i, row := range problem.MarginEffects {
		effect := row[0]
		required := problem.Floors[i] - problem.Margins[i]
		if math.Abs(effect) <= tolerance {
			if required > tolerance {
				return ConstrainedSolution{}, errors.New("cluster: anchored floor has no measurable direction and is infeasible")
			}
			continue
		}
		boundary := required / effect
		if effect > 0 && boundary > lower {
			lower = boundary
		}
		if effect < 0 && boundary < upper {
			upper = boundary
		}
	}
	if lower > upper+tolerance {
		return ConstrainedSolution{}, errors.New("cluster: anchored margin constraints are infeasible within the step bound")
	}
	coefficient := -problem.Gradient[0] / problem.Lambda
	coefficient = math.Max(lower, math.Min(upper, coefficient))
	coefficients := []float64{coefficient}
	if !feasible(problem, coefficients, tolerance) {
		return ConstrainedSolution{}, errors.New("cluster: anchored margin constraints are infeasible within the step bound")
	}
	return ConstrainedSolution{
		Coefficients: coefficients, PredictedMargins: predictMargins(problem, coefficients), Iterations: 1,
	}, nil
}

func feasible(problem ConstrainedUpdate, coefficients []float64, tolerance float64) bool {
	if math.Sqrt(dot(coefficients, coefficients)) > problem.StepBound+tolerance {
		return false
	}
	for i, row := range problem.MarginEffects {
		if problem.Margins[i]+dot(row, coefficients) < problem.Floors[i]-tolerance {
			return false
		}
	}
	return true
}

func predictMargins(problem ConstrainedUpdate, coefficients []float64) []float64 {
	predicted := make([]float64, len(problem.Margins))
	for i, row := range problem.MarginEffects {
		predicted[i] = problem.Margins[i] + dot(row, coefficients)
	}
	return predicted
}

func dot(a, b []float64) float64 {
	total := 0.0
	for i := range a {
		total += a[i] * b[i]
	}
	return total
}
