package decoder

import (
	"context"
	"errors"

	"github.com/tayi-ai/arandu-llama/training/pipeline"
)

// RecoveryBackend computes causal recovery over an admitted ordered batch. It
// shares native parameter installation and protection margins with fusion, but
// does not contain a teacher or feature path. The loaded model is borrowed.
type RecoveryBackend struct {
	state *TrainingBackend
	batch *RecoveryBatch
}

var _ pipeline.Model = (*RecoveryBackend)(nil)

// NewRecoveryBackend binds immutable recovery data to the independently admitted
// assembly. Prepare has the same RoPE-only contract as NewTrainingBackend.
func NewRecoveryBackend(loaded *LoadedTextModel, batch *RecoveryBatch, limits Limits, prepare func(context.Context, int) (func() error, error), admission TrainingAdmission) (*RecoveryBackend, error) {
	batch = batch.snapshot()
	if batch == nil || batch.document.Student != admission.Student {
		return nil, ErrRecovery
	}
	for _, row := range batch.document.Examples {
		if int64(len(row.Tokens)) > limits.MaxTokens || int64(len(row.Tokens)-row.PromptTokens+1) > limits.LogitRows {
			return nil, ErrRecovery
		}
	}
	state, err := newTrainingState(loaded, batch.document.Examples[0], limits, prepare, admission)
	if err != nil {
		return nil, err
	}
	return &RecoveryBackend{state: state, batch: batch}, nil
}

// Parameters copies the canonical native FP32 parameter registry.
func (m *RecoveryBackend) Parameters(ctx context.Context) ([]float32, error) {
	if m == nil || m.state == nil {
		return nil, ErrRecovery
	}
	return m.state.Parameters(ctx)
}

// Margin uses summed correct-minus-wrong log probability and its exact Jacobian.
func (m *RecoveryBackend) Margin(ctx context.Context, pair pipeline.ProtectedPair, jacobian bool) (pipeline.Margin, error) {
	if m == nil || m.state == nil {
		return pipeline.Margin{}, ErrRecovery
	}
	return m.state.Margin(ctx, pair, jacobian)
}

// Install atomically replaces the complete native adapter registry.
func (m *RecoveryBackend) Install(ctx context.Context, parameters []float32) error {
	if m == nil || m.state == nil {
		return ErrRecovery
	}
	return m.state.Install(ctx, parameters)
}

// Objective sums per-example mean NLL and gradients weighted by supervised-token
// counts, then divides once. Loss and HardLoss match; FeatureLoss is zero. Native
// snapshots and rotary tables are closed between examples. No optimizer runs.
func (m *RecoveryBackend) Objective(ctx context.Context) (pipeline.Objective, error) {
	var result pipeline.Objective
	if m == nil || m.state == nil || !m.state.ready() || m.batch == nil || !m.batch.admitted {
		return result, ErrRecovery
	}
	var supervised int64
	for _, row := range m.batch.document.Examples {
		limits, close, err := m.state.prepare(ctx, row)
		if err != nil {
			return pipeline.Objective{}, err
		}
		gradient, gradErr := CompletionGradient(ctx, m.state.loaded.Model, row.Tokens, row.PromptTokens, limits, 1)
		if err := errors.Join(gradErr, close()); err != nil {
			return pipeline.Objective{}, err
		}
		if gradient.Tokens != len(row.Tokens)-row.PromptTokens || !finiteBackend(gradient.Loss) {
			return pipeline.Objective{}, ErrRecovery
		}
		flat, err := m.state.flatten(gradient.Gradients, 1)
		if err != nil {
			return pipeline.Objective{}, err
		}
		if result.Gradient == nil {
			result.Gradient = make([]float64, len(flat))
		}
		if len(flat) != len(result.Gradient) {
			return pipeline.Objective{}, ErrRecovery
		}
		weight := float64(gradient.Tokens)
		supervised += int64(gradient.Tokens)
		result.Loss += gradient.Loss * weight
		for i, value := range flat {
			result.Gradient[i] += value * weight
		}
	}
	if supervised != m.batch.document.SupervisedTokens || supervised < 1 {
		return pipeline.Objective{}, ErrRecovery
	}
	weight := float64(supervised)
	result.Loss /= weight
	result.HardLoss = result.Loss
	if !finiteBackend(result.Loss) {
		return pipeline.Objective{}, ErrRecovery
	}
	for i := range result.Gradient {
		result.Gradient[i] /= weight
		if !finiteBackend(result.Gradient[i]) {
			return pipeline.Objective{}, ErrRecovery
		}
	}
	if err := ctx.Err(); err != nil {
		return pipeline.Objective{}, err
	}
	return result, nil
}
