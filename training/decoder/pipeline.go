package decoder

import (
	"context"
	"errors"
	"math"
	"slices"

	"github.com/tayi-ai/arandu-llama/training/fusioncache"
	"github.com/tayi-ai/arandu-llama/training/pipeline"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// TrainingAdmission independently binds a cache student identity to the loaded
// assembly documents and initial adapter. These pins must come from the external
// admitted manifest. Matching hashes alone does not qualify model execution.
type TrainingAdmission struct {
	Assembly AssemblyIdentity
	Student  fusioncache.ModelIdentity
}

// TrainingBackend binds admitted signals to a loaded native student. Its fields
// are private so raw teacher/feature arrays cannot bypass signal admission.
// The caller must serialize model access and preserve the borrowed model owner.
type TrainingBackend struct {
	loaded          *LoadedTextModel
	training        pipeline.Example
	teachers        []FusionTeacher
	features        []FeatureTarget
	limits          Limits
	prepareSequence func(context.Context, int) (func() error, error)
	admission       TrainingAdmission
	admitted        bool
}

// NewTrainingBackend validates the independently admitted assembly and student
// before converting immutable generic signals. Prepare may install rotary tables
// only and must return cleanup; it must not mutate model parameters or identities.
// This adapter owns neither model nor table lifetime and starts no execution.
func NewTrainingBackend(loaded *LoadedTextModel, signals *pipeline.Signals, limits Limits, prepare func(context.Context, int) (func() error, error), admission TrainingAdmission) (*TrainingBackend, error) {
	if !signals.Admitted() || signals.Student() != admission.Student {
		return nil, errors.New("pipeline: independently admitted assembly or student differs")
	}
	result, err := newTrainingState(loaded, signals.Training(), limits, prepare, admission)
	if err != nil {
		return nil, err
	}
	embedding, err := loaded.Model.Embedding.Info()
	if err != nil {
		return nil, err
	}
	for _, teacher := range signals.Teachers() {
		out := FusionTeacher{Name: teacher.Identity.Name, Weight: teacher.Weight}
		for _, position := range teacher.Positions {
			item := FusionTeacherPosition{RetainedMass: position.RetainedMass}
			for _, probability := range position.Probabilities {
				item.TopK = append(item.TopK, FusionTokenProbability{TokenID: probability.TokenID, Probability: probability.Probability})
			}
			out.Positions = append(out.Positions, item)
		}
		result.teachers = append(result.teachers, out)
	}
	for _, feature := range signals.Features() {
		if feature.Target.Tensor != "decoder_output" || feature.Target.Layer < 0 || feature.Target.Layer >= len(loaded.Model.Layers) || int64(feature.Target.Dimension) != embedding.Shape[1] {
			return nil, errors.New("pipeline: projected target differs from loaded decoder geometry")
		}
		result.features = append(result.features, FeatureTarget{Source: feature.Source, Layer: feature.Target.Layer, Position: feature.Position, Weight: feature.Weight, Values: slices.Clone(feature.Values)})
	}
	return result, nil
}

func newTrainingState(loaded *LoadedTextModel, row pipeline.Example, limits Limits, prepare func(context.Context, int) (func() error, error), admission TrainingAdmission) (*TrainingBackend, error) {
	if loaded == nil || loaded.Model == nil || loaded.Summary.Identity != admission.Assembly {
		return nil, errors.New("pipeline: independently admitted assembly or student differs")
	}
	for _, hash := range []string{admission.Assembly.IndexSHA256, admission.Assembly.ConfigSHA256, admission.Assembly.ReferenceSHA256, admission.Assembly.InitialAdapterSHA256} {
		if !validAssemblyHash(hash) {
			return nil, errors.New("pipeline: external assembly identity required")
		}
	}
	if row.PromptTokens < 1 || row.PromptTokens >= len(row.Tokens) || limits.MaxTokens < int64(len(row.Tokens)) ||
		limits.LogitRows < int64(len(row.Tokens)-row.PromptTokens+1) || limits.MaxCheckpointBytes <= 0 || len(loaded.Parameters) == 0 || loaded.Model.Embedding == nil || loaded.Model.Head == nil {
		return nil, errors.New("pipeline: invalid loaded geometry or training limits")
	}
	embedding, err := loaded.Model.Embedding.Info()
	if err != nil {
		return nil, err
	}
	head, err := loaded.Model.Head.Info()
	if err != nil {
		return nil, err
	}
	if len(embedding.Shape) != 2 || embedding.Shape[0] != int64(admission.Student.Vocabulary) || embedding.Shape[1] <= 0 || !slices.Equal(embedding.Shape, head.Shape) {
		return nil, errors.New("pipeline: loaded vocabulary or hidden width differs")
	}
	row.Tokens = slices.Clone(row.Tokens)
	return &TrainingBackend{loaded: loaded, training: row, limits: limits, prepareSequence: prepare, admission: admission, admitted: true}, nil
}

func (m *TrainingBackend) ready() bool {
	return m != nil && m.admitted && m.loaded != nil && m.loaded.Model != nil && m.loaded.Summary.Identity == m.admission.Assembly
}

var _ pipeline.Model = (*TrainingBackend)(nil)

// Parameters copies the exact FP32 registry used by the native gradient path.
func (m *TrainingBackend) Parameters(ctx context.Context) ([]float32, error) {
	if ctx == nil || !m.ready() {
		return nil, errors.New("pipeline: native model unavailable")
	}
	var values []float32
	for _, parameter := range m.loaded.Parameters {
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
	if ctx == nil || !m.ready() || row.PromptTokens < 1 || row.PromptTokens >= len(row.Tokens) ||
		int64(len(row.Tokens)) > m.limits.MaxTokens || int64(len(row.Tokens)-row.PromptTokens+1) > m.limits.LogitRows || m.limits.MaxCheckpointBytes <= 0 {
		return Limits{}, nil, errors.New("pipeline: sequence outside admitted native bounds")
	}
	if err := ctx.Err(); err != nil {
		return Limits{}, nil, err
	}
	limits := m.limits
	limits.LogitRows = int64(len(row.Tokens) - row.PromptTokens + 1)
	close := func() error { return nil }
	if m.prepareSequence != nil {
		var err error
		close, err = m.prepareSequence(ctx, len(row.Tokens))
		if err != nil {
			if close != nil {
				err = errors.Join(err, close())
			}
			return Limits{}, nil, err
		}
		if close == nil {
			return Limits{}, nil, errors.New("pipeline: missing sequence cleanup")
		}
	}
	if !m.ready() {
		return Limits{}, nil, errors.Join(errors.New("pipeline: assembly identity changed during preparation"), close())
	}
	return limits, close, nil
}

// Objective computes output distillation and hidden feature transfer together.
// The full method refuses a missing teacher or feature path instead of silently
// continuing as ordinary SFT. Standalone SFT is a separate recipe phase.
func (m *TrainingBackend) Objective(ctx context.Context) (result pipeline.Objective, err error) {
	if !m.ready() || len(m.teachers) == 0 || len(m.features) == 0 {
		return result, errors.New("pipeline: fusion requires teacher and feature signals")
	}
	limits, close, err := m.prepare(ctx, m.training)
	if err != nil {
		return result, err
	}
	defer func() { err = errors.Join(err, close()) }()
	gradient, err := FusionCompletionGradientWithFeatures(ctx, m.loaded.Model, m.training.Tokens, m.training.PromptTokens, limits, 1, m.teachers, m.features)
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
	pair.Positive.Tokens = slices.Clone(pair.Positive.Tokens)
	pair.Negative.Tokens = slices.Clone(pair.Negative.Tokens)
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
			gradient, gradErr := CompletionGradient(ctx, m.loaded.Model, row.Tokens, row.PromptTokens, limits, 1)
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
			value, scoreErr := scoreCompletion(ctx, m.loaded.Model, row, limits)
			if err := errors.Join(scoreErr, close()); err != nil {
				return pipeline.Margin{}, err
			}
			result.Value += sign * value
		}
	}
	return result, nil
}

func (m *TrainingBackend) flatten(gradients []CompletionParameterGradient, scale float64) ([]float64, error) {
	if len(gradients) != len(m.loaded.Parameters) {
		return nil, errors.New("pipeline: gradient registry differs")
	}
	var flat []float64
	for i, part := range gradients {
		parameter := m.loaded.Parameters[i]
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
	if !m.ready() {
		return errors.New("pipeline: native model unavailable")
	}
	_, err := m.loaded.ReplaceParameters(ctx, values)
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
