package ornith

import (
	"context"
	"errors"
	"math"

	"github.com/tayi-ai/arandu-llama/training/pipeline"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// TrainingBackend binds the complete objective to a loaded native student. Teachers
// and Features must come from an identity-checked, teacher-forced cache. Prepare
// installs sequence-specific rotary tables and returns their cleanup function.
// It must not change parameters. This adapter owns no model or table lifetime.
type TrainingBackend struct {
	Loaded   *LoadedTextModel
	Training pipeline.Example
	Teachers []FusionTeacher
	Features []FeatureTarget
	Limits   Limits
	Prepare  func(context.Context, int) (func() error, error)
}

var _ pipeline.Model = (*TrainingBackend)(nil)

// Parameters copies the exact FP32 registry used by the native gradient path.
func (m *TrainingBackend) Parameters(ctx context.Context) ([]float32, error) {
	if ctx == nil || m == nil || m.Loaded == nil || m.Loaded.Model == nil {
		return nil, errors.New("pipeline: native model unavailable")
	}
	var values []float32
	for _, parameter := range m.Loaded.Parameters {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		part, err := parameter.Value.Float32Values()
		if err != nil {
			return nil, err
		}
		values = append(values, part...)
	}
	return values, nil
}

func (m *TrainingBackend) prepare(ctx context.Context, row pipeline.Example) (Limits, func() error, error) {
	if ctx == nil || m == nil || m.Loaded == nil || m.Loaded.Model == nil || row.PromptTokens < 1 || row.PromptTokens >= len(row.Tokens) ||
		int64(len(row.Tokens)) > m.Limits.MaxTokens || m.Limits.MaxCheckpointBytes <= 0 {
		return Limits{}, nil, errors.New("pipeline: sequence outside admitted native bounds")
	}
	limits := m.Limits
	limits.LogitRows = int64(len(row.Tokens) - row.PromptTokens + 1)
	close := func() error { return nil }
	if m.Prepare != nil {
		var err error
		close, err = m.Prepare(ctx, len(row.Tokens))
		if err != nil {
			return Limits{}, nil, err
		}
		if close == nil {
			return Limits{}, nil, errors.New("pipeline: missing sequence cleanup")
		}
	}
	return limits, close, nil
}

// Objective computes output distillation and hidden feature transfer together.
// The full method refuses a missing teacher or feature path instead of silently
// continuing as ordinary SFT. Standalone SFT is a separate recipe phase.
func (m *TrainingBackend) Objective(ctx context.Context) (result pipeline.Objective, err error) {
	if m == nil || len(m.Teachers) == 0 || len(m.Features) == 0 {
		return result, errors.New("pipeline: fusion requires teacher and feature signals")
	}
	limits, close, err := m.prepare(ctx, m.Training)
	if err != nil {
		return result, err
	}
	defer func() { err = errors.Join(err, close()) }()
	gradient, err := FusionCompletionGradientWithFeatures(ctx, m.Loaded.Model, m.Training.Tokens, m.Training.PromptTokens, limits, 1, m.Teachers, m.Features)
	if err != nil {
		return result, err
	}
	result.Gradient, err = m.flatten(gradient.Gradients, 1)
	result.Loss, result.HardLoss, result.FeatureLoss = gradient.Loss, gradient.HardLoss, gradient.FeatureLoss
	return result, err
}

// Margin computes log P(correct|prompt) - log P(wrong|prompt), with token sums
// rather than averages across answers of different lengths. Jacobians use the
// same summed objective. Actual-candidate checks use forward only.
func (m *TrainingBackend) Margin(ctx context.Context, pair pipeline.ProtectedPair, jacobian bool) (result pipeline.Margin, err error) {
	if err := pipeline.ValidateProtectedPair(pair); err != nil {
		return result, err
	}
	for index, row := range []pipeline.Example{pair.Positive, pair.Negative} {
		limits, close, err := m.prepare(ctx, row)
		if err != nil {
			return pipeline.Margin{}, err
		}
		sign := 1.0
		if index == 1 {
			sign = -1
		}
		if jacobian {
			gradient, gradErr := CompletionGradient(ctx, m.Loaded.Model, row.Tokens, row.PromptTokens, limits, 1)
			closeErr := close()
			if err := errors.Join(gradErr, closeErr); err != nil {
				return pipeline.Margin{}, err
			}
			result.Value -= sign * gradient.Loss * float64(gradient.Tokens)
			flat, err := m.flatten(gradient.Gradients, -sign*float64(gradient.Tokens))
			if err != nil {
				return pipeline.Margin{}, err
			}
			if index == 0 {
				result.Jacobian = flat
			} else {
				if len(flat) != len(result.Jacobian) {
					return pipeline.Margin{}, errors.New("pipeline: margin gradient geometry changed")
				}
				for i, value := range flat {
					result.Jacobian[i] += value
				}
			}
		} else {
			value, scoreErr := scoreCompletion(ctx, m.Loaded.Model, row, limits)
			if err := errors.Join(scoreErr, close()); err != nil {
				return pipeline.Margin{}, err
			}
			result.Value += sign * value
		}
	}
	return result, nil
}

func (m *TrainingBackend) flatten(gradients []CompletionParameterGradient, scale float64) ([]float64, error) {
	if len(gradients) != len(m.Loaded.Parameters) {
		return nil, errors.New("pipeline: gradient registry differs")
	}
	var flat []float64
	for i, part := range gradients {
		parameter := m.Loaded.Parameters[i]
		info, err := parameter.Value.Info()
		if err != nil {
			return nil, err
		}
		if part.Name != parameter.Name || int64(len(part.ValuesF32)) != info.Elements {
			return nil, errors.New("pipeline: gradient identity differs")
		}
		for _, value := range part.ValuesF32 {
			flat = append(flat, float64(value)*scale)
		}
	}
	return flat, nil
}

// Install replaces the complete adapter atomically using the native registry.
func (m *TrainingBackend) Install(ctx context.Context, values []float32) error {
	if m == nil || m.Loaded == nil {
		return errors.New("pipeline: native model unavailable")
	}
	_, err := m.Loaded.ReplaceParameters(ctx, values)
	return err
}

func scoreCompletion(ctx context.Context, model *TextModel, row pipeline.Example, limits Limits) (result float64, err error) {
	snapshot, err := model.Forward(ctx, row.Tokens, limits)
	if err != nil {
		return 0, err
	}
	defer func() { err = errors.Join(err, snapshot.Close()) }()
	info, err := snapshot.Logits.Info()
	if err != nil {
		return 0, err
	}
	count := len(row.Tokens) - row.PromptTokens
	if len(info.Shape) != 3 || info.Shape[0] != 1 || info.Shape[1] != int64(count+1) || info.Shape[2] <= 1 || info.DType != torch.Float32 {
		return 0, errors.New("pipeline: completion score geometry differs")
	}
	values, err := snapshot.Logits.Float32Values()
	if err != nil {
		return 0, err
	}
	vocabulary := int(info.Shape[2])
	for position, token := range row.Tokens[row.PromptTokens:] {
		if err := ctx.Err(); err != nil {
			return 0, err
		}
		if token < 0 || token >= int64(vocabulary) {
			return 0, errors.New("pipeline: scoring token outside vocabulary")
		}
		logits := values[position*vocabulary : (position+1)*vocabulary]
		max := math.Inf(-1)
		for _, v := range logits {
			if !finiteBackend(float64(v)) {
				return 0, errors.New("pipeline: nonfinite scoring logit")
			}
			max = math.Max(max, float64(v))
		}
		sum := 0.0
		for _, v := range logits {
			sum += math.Exp(float64(v) - max)
		}
		result += float64(logits[token]) - max - math.Log(sum)
	}
	return result, nil
}

func finiteBackend(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
