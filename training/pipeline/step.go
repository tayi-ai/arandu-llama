// Package pipeline connects the admitted fusion objective, constrained update,
// and real-candidate protection check without owning transport or scheduling.
package pipeline

import (
	"context"
	"errors"
	"fmt"
	"math"
	"slices"

	"github.com/tayi-ai/arandu-llama/training/protection"
)

// Example identifies an immutable prompt and supervised completion.
type Example struct {
	ID            string  `json:"id"`
	DatasetDigest string  `json:"dataset_digest"`
	Tokens        []int64 `json:"tokens"`
	PromptTokens  int     `json:"prompt_tokens"`
}

// ProtectedPair defines the anchored score margin of a correct and wrong answer.
// Floor is fixed by the admitted protocol, never recalculated from a candidate.
type ProtectedPair struct {
	ID       string  `json:"id"`
	Positive Example `json:"positive"`
	Negative Example `json:"negative"`
	Floor    float64 `json:"floor"`
}

// Objective is the unscaled mean completion objective and its ordered gradient.
type Objective struct {
	Loss, HardLoss, FeatureLoss float64
	Gradient                    []float64
}

// Margin is a summed log-probability difference and its parameter Jacobian.
type Margin struct {
	Value    float64
	Jacobian []float64
}

// Model owns the concrete backend and canonical parameter order. Local and
// distributed implementations must expose the same mathematical operations.
// Methods are serialized; installing parameters invalidates cached activations.
type Model interface {
	Parameters(context.Context) ([]float32, error)
	Objective(context.Context) (Objective, error)
	Margin(context.Context, ProtectedPair, bool) (Margin, error)
	Install(context.Context, []float32) error
}

// StepConfig pins the protection pool and numerical solver budget.
type StepConfig struct {
	Lambda float64
	// MinimumGain is a frozen nonnegative decrease of the full real objective.
	MinimumGain float64
	Protection  []ProtectedPair
	Solver      protection.Config
}

// StepReceipt separates optimization convergence from real-candidate acceptance.
type StepReceipt struct {
	Objective          Objective
	CandidateObjective Objective
	Solution           protection.Solution
	Evaluation         protection.Evaluation
	Parameters         []float32
}

// Step computes the complete fusion objective, solves the margin-constrained
// update, then evaluates the actual candidate. A failed candidate is restored;
// no checkpoint or optimizer state is committed by this mathematical operation.
func Step(ctx context.Context, model Model, cfg StepConfig) (StepReceipt, error) {
	var result StepReceipt
	if ctx == nil || model == nil || len(cfg.Protection) == 0 || !finite(cfg.Lambda) || cfg.Lambda <= 0 || !finite(cfg.MinimumGain) || cfg.MinimumGain < 0 {
		return result, errors.New("pipeline: model, protection pool and positive lambda required")
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	if len(cfg.Protection) > protection.DefaultConfig().MaxConstraints {
		return result, protection.ErrLimit
	}
	cfg.Protection = slices.Clone(cfg.Protection)
	for i := range cfg.Protection {
		cfg.Protection[i].Positive.Tokens = slices.Clone(cfg.Protection[i].Positive.Tokens)
		cfg.Protection[i].Negative.Tokens = slices.Clone(cfg.Protection[i].Negative.Tokens)
	}
	seen := make(map[string]bool, len(cfg.Protection))
	for _, pair := range cfg.Protection {
		if err := ValidateProtectedPair(pair); err != nil {
			return result, err
		}
		if seen[pair.ID] {
			return result, errors.New("pipeline: duplicate protection identity")
		}
		seen[pair.ID] = true
	}
	prior, err := model.Parameters(ctx)
	if err != nil {
		return result, err
	}
	prior = slices.Clone(prior)
	if len(prior) == 0 {
		return result, errors.New("pipeline: empty parameter vector")
	}
	limits := protection.DefaultConfig()
	if cfg.Solver.MaxParameters < 1 || cfg.Solver.MaxConstraints < 1 || cfg.Solver.MaxCoefficients < 1 ||
		len(prior) > cfg.Solver.MaxParameters || len(prior) > limits.MaxParameters ||
		len(cfg.Protection) > cfg.Solver.MaxConstraints || len(cfg.Protection) > limits.MaxConstraints ||
		len(prior) > cfg.Solver.MaxCoefficients/len(cfg.Protection) || len(prior) > limits.MaxCoefficients/len(cfg.Protection) {
		return result, protection.ErrLimit
	}
	for _, value := range prior {
		if !finite(float64(value)) {
			return result, errors.New("pipeline: nonfinite prior parameter")
		}
	}
	result.Objective, err = model.Objective(ctx)
	if err != nil {
		return result, err
	}
	if !finite(result.Objective.Loss) || !finite(result.Objective.HardLoss) || !finite(result.Objective.FeatureLoss) || len(result.Objective.Gradient) != len(prior) {
		return result, errors.New("pipeline: objective or gradient geometry differs")
	}
	for _, value := range result.Objective.Gradient {
		if !finite(value) {
			return result, errors.New("pipeline: nonfinite objective gradient")
		}
	}
	problem := protection.Problem{Gradient: result.Objective.Gradient, Lambda: cfg.Lambda}
	floors := make([]protection.Floor, 0, len(cfg.Protection))
	for _, pair := range cfg.Protection {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		margin, err := model.Margin(ctx, pair, true)
		if err != nil {
			return result, fmt.Errorf("pipeline: protection linearization: %w", err)
		}
		problem.Constraints = append(problem.Constraints, protection.Constraint{ID: pair.ID, Margin: margin.Value, Floor: pair.Floor, Jacobian: margin.Jacobian})
		floors = append(floors, protection.Floor{ID: pair.ID, Value: pair.Floor})
	}
	result.Solution, err = protection.Solve(ctx, problem, cfg.Solver)
	if err != nil {
		return result, err
	}
	candidate := make([]float32, len(prior))
	for i, value := range prior {
		candidate[i] = float32(float64(value) + result.Solution.Step[i])
		if !finite(float64(candidate[i])) {
			return result, errors.New("pipeline: candidate parameter overflow")
		}
	}
	result.Evaluation, err = protection.EvaluateCandidate(ctx, prior, candidate, floors, protection.CandidateCallbacks{
		Apply: model.Install, Restore: model.Install,
		Measure: func(ctx context.Context) ([]protection.Margin, error) {
			var err error
			result.CandidateObjective, err = model.Objective(ctx)
			if err != nil {
				return nil, err
			}
			if !finite(result.CandidateObjective.Loss) || !finite(result.CandidateObjective.HardLoss) || !finite(result.CandidateObjective.FeatureLoss) || result.Objective.Loss-result.CandidateObjective.Loss < cfg.MinimumGain {
				return nil, errors.New("pipeline: real objective failed the frozen gain requirement")
			}
			margins := make([]protection.Margin, 0, len(cfg.Protection))
			for _, pair := range cfg.Protection {
				value, err := model.Margin(ctx, pair, false)
				if err != nil {
					return nil, err
				}
				margins = append(margins, protection.Margin{ID: pair.ID, Value: value.Value})
			}
			return margins, nil
		},
	})
	if err != nil {
		return result, err
	}
	if !result.Evaluation.Accepted {
		return result, errors.New("pipeline: candidate was not accepted")
	}
	result.Parameters = candidate
	return result, nil
}

// ValidateProtectedPair checks immutable token geometry before any model call.
func ValidateProtectedPair(pair ProtectedPair) error {
	if !identifier(pair.ID) || !finite(pair.Floor) {
		return errors.New("pipeline: invalid protection identity or floor")
	}
	for _, row := range []Example{pair.Positive, pair.Negative} {
		if !identifier(row.ID) || !digest(row.DatasetDigest) || len(row.Tokens) > 1<<20 || row.PromptTokens < 1 || row.PromptTokens >= len(row.Tokens) {
			return errors.New("pipeline: invalid protection completion")
		}
		for _, token := range row.Tokens {
			if token < 0 {
				return errors.New("pipeline: negative token id")
			}
		}
	}
	if pair.Positive.DatasetDigest != pair.Negative.DatasetDigest || pair.Positive.PromptTokens != pair.Negative.PromptTokens ||
		!slices.Equal(pair.Positive.Tokens[:pair.Positive.PromptTokens], pair.Negative.Tokens[:pair.Negative.PromptTokens]) || slices.Equal(pair.Positive.Tokens, pair.Negative.Tokens) {
		return errors.New("pipeline: protection pair must share prompt and contain different answers")
	}
	return nil
}

func finite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
