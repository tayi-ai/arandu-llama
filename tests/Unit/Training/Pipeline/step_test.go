package pipeline_test

import (
	"context"
	"errors"
	"math"
	"slices"
	"strings"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/pipeline"
	"github.com/tayi-ai/arandu-llama/training/protection"
)

type scalarModel struct {
	values          []float32
	gradient        []float64
	nonlinear       bool
	quadratic       bool
	calls, installs int
	cancel          context.CancelFunc
}

func (m *scalarModel) Parameters(context.Context) ([]float32, error) { return m.values, nil }
func (m *scalarModel) Objective(context.Context) (pipeline.Objective, error) {
	m.calls++
	if m.quadratic {
		x := float64(m.values[0])
		return pipeline.Objective{Gradient: []float64{2 * x}, Loss: x * x}, nil
	}
	return pipeline.Objective{Gradient: m.gradient, Loss: 1}, nil
}
func (m *scalarModel) Margin(ctx context.Context, _ pipeline.ProtectedPair, jacobian bool) (pipeline.Margin, error) {
	if err := ctx.Err(); err != nil {
		return pipeline.Margin{}, err
	}
	x := float64(m.values[0])
	if m.nonlinear {
		return pipeline.Margin{Value: 1 - x*x, Jacobian: []float64{-2 * x}}, nil
	}
	return pipeline.Margin{Value: x, Jacobian: []float64{1}}, nil
}
func (m *scalarModel) Install(ctx context.Context, v []float32) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	m.installs++
	m.values = slices.Clone(v)
	if m.installs == 1 && m.cancel != nil {
		m.cancel()
	}
	return nil
}
func stepConfig() pipeline.StepConfig {
	positive := pipeline.Example{ID: "correct", DatasetDigest: strings.Repeat("a", 64), Tokens: []int64{1, 2}, PromptTokens: 1}
	negative := positive
	negative.ID = "wrong"
	negative.Tokens = []int64{1, 3}
	return pipeline.StepConfig{Lambda: 1, Solver: protection.DefaultConfig(), Protection: []pipeline.ProtectedPair{{ID: "protected", Positive: positive, Negative: negative, Floor: 0}}}
}
func TestConstrainedStepChecksMaterializedParameters(t *testing.T) {
	m := &scalarModel{values: []float32{1}, gradient: []float64{2}}
	r, err := pipeline.Step(context.Background(), m, stepConfig())
	if err != nil || !r.Evaluation.Accepted || len(r.Parameters) != 1 || m.values[0] != 0 || r.Solution.Step[0] != -1 || m.installs != 1 {
		t.Fatalf("invalid accepted step: %+v %v", r, err)
	}
}
func TestLinearizedSuccessDoesNotAcceptNonlinearViolation(t *testing.T) {
	m := &scalarModel{values: []float32{0}, gradient: []float64{-2}, nonlinear: true}
	r, err := pipeline.Step(context.Background(), m, stepConfig())
	if err == nil || r.Evaluation.Accepted || !r.Evaluation.Restored || r.Parameters != nil || m.values[0] != 0 || m.installs != 2 {
		t.Fatalf("candidate was not rolled back: %+v %v", r, err)
	}
}
func TestCancellationAfterApplyRestoresPrior(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	m := &scalarModel{values: []float32{1}, gradient: []float64{1}, cancel: cancel}
	r, err := pipeline.Step(ctx, m, stepConfig())
	if !errors.Is(err, context.Canceled) || !r.Evaluation.Restored || m.values[0] != 1 || m.installs != 2 {
		t.Fatalf("cancel rollback differs: %+v %v", r, err)
	}
}
func TestResourceAndNonfiniteFailuresPrecedeObjective(t *testing.T) {
	for _, bad := range []string{"budget", "prior"} {
		t.Run(bad, func(t *testing.T) {
			m := &scalarModel{values: []float32{1}, gradient: []float64{1}}
			c := stepConfig()
			if bad == "budget" {
				c.Solver.MaxCoefficients = 0
			} else {
				m.values[0] = float32(math.NaN())
			}
			if _, err := pipeline.Step(context.Background(), m, c); err == nil || m.calls != 0 || m.installs != 0 {
				t.Fatalf("invalid input reached objective: %v", err)
			}
		})
	}
}
func TestProtectionRequiresSamePromptAndDistinctAnswers(t *testing.T) {
	m := &scalarModel{values: []float32{1}, gradient: []float64{1}}
	c := stepConfig()
	c.Protection[0].Negative.Tokens = []int64{4, 3}
	if _, err := pipeline.Step(context.Background(), m, c); err == nil || m.calls != 0 {
		t.Fatal("different prompt admitted")
	}
}

func TestRealObjectiveIncreaseRestoresEvenWhenLinearConstraintsPass(t *testing.T) {
	m := &scalarModel{values: []float32{1}, quadratic: true}
	c := stepConfig()
	c.Lambda = 0.1
	c.Protection[0].Floor = -100
	r, err := pipeline.Step(context.Background(), m, c)
	if err == nil || r.Evaluation.Accepted || !r.Evaluation.Restored || m.values[0] != 1 || r.CandidateObjective.Loss != 361 {
		t.Fatalf("objective degradation accepted: %+v %v", r, err)
	}
}

type reusedGradientModel struct{ scalarModel }

func (m *reusedGradientModel) Objective(context.Context) (pipeline.Objective, error) {
	m.gradient[0] = 2
	return pipeline.Objective{Loss: 1, Gradient: m.gradient}, nil
}
func (m *reusedGradientModel) Margin(_ context.Context, _ pipeline.ProtectedPair, _ bool) (pipeline.Margin, error) {
	m.gradient[0] = 1
	return pipeline.Margin{Value: float64(m.values[0]), Jacobian: m.gradient}, nil
}
func TestBackendBufferReuseCannotChangeCapturedGradient(t *testing.T) {
	m := &reusedGradientModel{scalarModel{values: []float32{2}, gradient: []float64{2}}}
	config := stepConfig()
	config.Protection[0].Floor = -100
	result, err := pipeline.Step(context.Background(), m, config)
	if err != nil || m.values[0] != 0 || result.Objective.Gradient[0] != 2 || result.CandidateObjective.Gradient[0] != 2 {
		t.Fatalf("shared backend buffer changed update: %+v %v", result, err)
	}
}
