package ornith

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

// FusionTokenProbability is one mapped teacher probability in the Ornith
// vocabulary. Token IDs must already be mapped by the cache producer.
type FusionTokenProbability struct {
	TokenID     int64
	Probability float64
}

// FusionTeacherPosition contains the mapped top-k mass for one supervised token.
type FusionTeacherPosition struct {
	RetainedMass float64
	TopK         []FusionTokenProbability
}

// FusionTeacher is one immutable teacher signal over a completion.
type FusionTeacher struct {
	Name      string
	Weight    float64
	Positions []FusionTeacherPosition
}

// FusionCompletionResult reports the blended objective and LoRA gradients.
// HardLoss is the pure one-hot completion CE before teacher blending.
// Loss includes FeatureLoss, the weighted hidden-state MSE, when requested.
type FusionCompletionResult struct {
	Loss                 float64
	HardLoss             float64
	FeatureLoss          float64
	Tokens               int
	EffectiveTeacherMass map[string]float64
	TeacherLosses        map[string]float64
	Gradients            []CompletionParameterGradient
}

// FusionCompletionGradient applies mapped teacher top-k distributions to the
// same causal rows used by CompletionGradient. Missing teacher probability mass
// is assigned back to the hard target, preserving a complete target distribution.
func FusionCompletionGradient(ctx context.Context, model *TextModel, tokenIDs []int64, promptTokens int, limits Limits, lossScale float64, teachers []FusionTeacher) (result FusionCompletionResult, err error) {
	return FusionCompletionGradientWithFeatures(ctx, model, tokenIDs, promptTokens, limits, lossScale, teachers, nil)
}

// FusionCompletionGradientWithFeatures adds projected teacher feature MSE to
// the causal teacher/gold objective. Features use absolute input-token positions
// and decoder outputs as defined by FeatureTarget. lossScale multiplies both
// logit and feature cotangents, while all reported losses remain unscaled.
// A nil or all-zero-weight feature list reproduces FusionCompletionGradient.
func FusionCompletionGradientWithFeatures(ctx context.Context, model *TextModel, tokenIDs []int64, promptTokens int, limits Limits, lossScale float64, teachers []FusionTeacher, features []FeatureTarget) (result FusionCompletionResult, err error) {
	if ctx == nil || model == nil || len(tokenIDs) < 2 || promptTokens < 1 || promptTokens >= len(tokenIDs) {
		return result, fmt.Errorf("%w: prompt and completion geometry is invalid", ErrCompletionStep)
	}
	if math.IsNaN(lossScale) || math.IsInf(lossScale, 0) || lossScale <= 0 {
		return result, fmt.Errorf("%w: finite positive loss scale required", ErrCompletionStep)
	}
	completionTokens := len(tokenIDs) - promptTokens
	if limits.LogitRows != int64(completionTokens+1) {
		return result, fmt.Errorf("%w: logit rows must equal supervised tokens plus one", ErrCompletionStep)
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	expected, err := candidateParameters(model)
	if err != nil {
		return result, err
	}
	snapshot, err := model.Forward(ctx, tokenIDs, limits)
	if err != nil {
		return result, err
	}
	var seed *torch.Tensor
	var nativeGradients Gradients
	defer func() {
		err = errors.Join(err, nativeGradients.Close(), seed.Close(), snapshot.Close())
		if err != nil {
			result = FusionCompletionResult{}
		}
	}()

	info, err := snapshot.Logits.Info()
	if err != nil {
		return result, err
	}
	if len(info.Shape) != 3 || info.Shape[0] != 1 || info.Shape[1] != int64(completionTokens+1) ||
		info.Shape[2] <= 1 || info.DType != torch.Float32 {
		return result, fmt.Errorf("%w: completion logits geometry differs", ErrCompletionStep)
	}
	vocabulary := int(info.Shape[2])
	values, err := snapshot.Logits.Float32Values()
	if err != nil {
		return result, err
	}
	if len(values) != (completionTokens+1)*vocabulary {
		return result, fmt.Errorf("%w: completion logit payload differs", ErrCompletionStep)
	}
	stats, err := fusionCotangent(values, vocabulary, tokenIDs[promptTokens:], lossScale, teachers)
	if err != nil {
		return result, err
	}
	result.Loss = stats.loss
	result.HardLoss = stats.hardLoss
	result.Tokens = completionTokens
	result.EffectiveTeacherMass = stats.effectiveMass
	result.TeacherLosses = stats.teacherLosses

	seed, err = torch.FromFloat32(values, info.Shape, info.Device, false)
	if err != nil {
		return result, err
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	nativeGradients, result.FeatureLoss, err = model.VJPWithFeatures(ctx, snapshot, seed, features, lossScale)
	if err != nil {
		return result, err
	}
	result.Loss += result.FeatureLoss
	if math.IsNaN(result.Loss) || math.IsInf(result.Loss, 0) {
		return result, fmt.Errorf("%w: nonfinite combined fusion loss", ErrCompletionStep)
	}
	if len(nativeGradients) != len(expected) {
		return result, fmt.Errorf("%w: incomplete parameter gradient set", ErrCompletionStep)
	}
	for index, gradient := range nativeGradients {
		if err := ctx.Err(); err != nil {
			return result, err
		}
		info, err := gradient.Value.Info()
		if err != nil {
			return result, err
		}
		if gradient.Name != expected[index].name || info.DType != torch.Float32 || info.RequiresGrad ||
			info.Device != expected[index].info.Device || !sameShape(info.Shape, expected[index].info.Shape) {
			return result, fmt.Errorf("%w: gradient name, geometry, precision or placement differs", ErrCompletionStep)
		}
		copied, err := gradient.Value.Float32Values()
		if err != nil {
			return result, err
		}
		for _, value := range copied {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return result, fmt.Errorf("%w: nonfinite parameter gradient", ErrCompletionStep)
			}
		}
		result.Gradients = append(result.Gradients, CompletionParameterGradient{
			Name: gradient.Name, Shape: info.Shape, ValuesF32: copied,
		})
	}
	return result, ctx.Err()
}

type fusionStats struct {
	loss          float64
	hardLoss      float64
	effectiveMass map[string]float64
	teacherLosses map[string]float64
}

func fusionCotangent(logits []float32, vocabulary int, targets []int64, lossScale float64, teachers []FusionTeacher) (fusionStats, error) {
	stats := fusionStats{
		effectiveMass: make(map[string]float64, len(teachers)),
		teacherLosses: make(map[string]float64, len(teachers)),
	}
	if vocabulary < 2 || len(targets) == 0 || len(logits) != (len(targets)+1)*vocabulary ||
		math.IsNaN(lossScale) || math.IsInf(lossScale, 0) || lossScale <= 0 {
		return stats, fmt.Errorf("%w: invalid fusion geometry", ErrCompletionStep)
	}
	totalWeight := 0.0
	seenTeachers := make(map[string]bool, len(teachers))
	for _, teacher := range teachers {
		if teacher.Name == "" || seenTeachers[teacher.Name] || teacher.Weight < 0 ||
			math.IsNaN(teacher.Weight) || math.IsInf(teacher.Weight, 0) ||
			len(teacher.Positions) != len(targets) {
			return fusionStats{}, fmt.Errorf("%w: invalid fusion teacher", ErrCompletionStep)
		}
		seenTeachers[teacher.Name] = true
		totalWeight += teacher.Weight
	}
	if totalWeight > 1+1e-12 {
		return fusionStats{}, fmt.Errorf("%w: teacher weights exceed one", ErrCompletionStep)
	}
	for _, target := range targets {
		if target < 0 || target >= int64(vocabulary) {
			return fusionStats{}, fmt.Errorf("%w: target token outside vocabulary", ErrCompletionStep)
		}
	}
	for _, value := range logits[:len(targets)*vocabulary] {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return fusionStats{}, fmt.Errorf("%w: nonfinite fusion logit", ErrCompletionStep)
		}
	}
	factor := lossScale / float64(len(targets))
	for row, target := range targets {
		offset := row * vocabulary
		maximum := float64(logits[offset])
		for column := 1; column < vocabulary; column++ {
			maximum = math.Max(maximum, float64(logits[offset+column]))
		}
		sum := 0.0
		for column := 0; column < vocabulary; column++ {
			sum += math.Exp(float64(logits[offset+column]) - maximum)
		}
		if !(sum > 0) || math.IsNaN(sum) || math.IsInf(sum, 0) {
			return fusionStats{}, fmt.Errorf("%w: invalid fusion softmax normalization", ErrCompletionStep)
		}
		logSum := math.Log(sum)
		hardLogProbability := float64(logits[offset+int(target)]) - maximum - logSum
		stats.hardLoss -= hardLogProbability

		targetMass := make(map[int64]float64, 1+64*len(teachers))
		hardMass := 1 - totalWeight
		mappedMassTotal := 0.0
		for _, teacher := range teachers {
			position := teacher.Positions[row]
			if position.RetainedMass < 0 || position.RetainedMass > 1 ||
				math.IsNaN(position.RetainedMass) || math.IsInf(position.RetainedMass, 0) {
				return fusionStats{}, fmt.Errorf("%w: invalid retained teacher mass", ErrCompletionStep)
			}
			sumMass := 0.0
			seenTokens := make(map[int64]bool, len(position.TopK))
			teacherCE := 0.0
			for _, item := range position.TopK {
				if item.TokenID < 0 || item.TokenID >= int64(vocabulary) || seenTokens[item.TokenID] ||
					item.Probability < 0 || math.IsNaN(item.Probability) || math.IsInf(item.Probability, 0) {
					return fusionStats{}, fmt.Errorf("%w: invalid mapped teacher token", ErrCompletionStep)
				}
				seenTokens[item.TokenID] = true
				sumMass += item.Probability
				logProbability := float64(logits[offset+int(item.TokenID)]) - maximum - logSum
				teacherCE -= item.Probability * logProbability
				targetMass[item.TokenID] += teacher.Weight * item.Probability
			}
			if math.Abs(sumMass-position.RetainedMass) > 1e-6 {
				return fusionStats{}, fmt.Errorf("%w: teacher retained mass differs from mapped top-k", ErrCompletionStep)
			}
			mappedMassTotal += teacher.Weight * sumMass
			hardMass += teacher.Weight * (1 - position.RetainedMass)
			stats.effectiveMass[teacher.Name] += teacher.Weight * position.RetainedMass
			stats.teacherLosses[teacher.Name] += teacherCE
		}
		targetMass[target] += hardMass
		targetTotal := hardMass + mappedMassTotal
		if math.Abs(targetTotal-1) > 1e-6 {
			return fusionStats{}, fmt.Errorf("%w: fusion target mass does not sum to one", ErrCompletionStep)
		}

		rowLoss := 0.0
		for column := 0; column < vocabulary; column++ {
			probability := math.Exp(float64(logits[offset+column])-maximum) / sum
			mass := targetMass[int64(column)]
			if mass != 0 {
				logProbability := float64(logits[offset+column]) - maximum - logSum
				rowLoss -= mass * logProbability
			}
			gradient := probability - mass
			rounded := float32(gradient * factor)
			if math.IsNaN(gradient) || math.IsInf(gradient, 0) ||
				math.IsNaN(float64(rounded)) || math.IsInf(float64(rounded), 0) {
				return fusionStats{}, fmt.Errorf("%w: fusion derivative is not finite FP32", ErrCompletionStep)
			}
			logits[offset+column] = rounded
		}
		stats.loss += rowLoss
	}
	clear(logits[len(targets)*vocabulary:])
	count := float64(len(targets))
	stats.loss /= count
	stats.hardLoss /= count
	for name := range stats.effectiveMass {
		stats.effectiveMass[name] /= count
		stats.teacherLosses[name] /= count
	}
	if math.IsNaN(stats.loss) || math.IsInf(stats.loss, 0) ||
		math.IsNaN(stats.hardLoss) || math.IsInf(stats.hardLoss, 0) {
		return fusionStats{}, fmt.Errorf("%w: nonfinite fusion loss", ErrCompletionStep)
	}
	return stats, nil
}
