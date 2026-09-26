package decoder

import (
	"context"
	"errors"
	"fmt"
	"math"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

// ErrCompletionStep identifies invalid autoregressive completion training data.
var ErrCompletionStep = errors.New("decoder: completion calculation rejected")

// CompletionParameterGradient owns a detached Go FP32 copy of one trainable
// adapter cotangent. Names and ordering follow the model's frozen PEFT layout.
type CompletionParameterGradient struct {
	Name      string
	Shape     []int64
	ValuesF32 []float32
}

// CompletionGradientResult contains the mean supervised-token negative log
// likelihood and the ordered LoRA gradients that produce that loss.
type CompletionGradientResult struct {
	Loss      float64
	Tokens    int
	Gradients []CompletionParameterGradient
}

// CompletionGradient computes exact causal-LM cross entropy over the final
// completion tokens only. tokenIDs contains prompt+completion. promptTokens is
// the first supervised token index. The model sees the entire sequence and the
// final target token is predicted from its preceding row, matching standard
// shifted causal-LM labels with -100 over the prompt.
//
// limits.LogitRows must equal completionTokens+1: one preceding row is needed
// to predict the first completion token, and the final row is intentionally
// unscored after the causal shift. lossScale multiplies only the cotangent;
// the reported Loss remains the unscaled mean NLL.
func CompletionGradient(ctx context.Context, model *TextModel, tokenIDs []int64, promptTokens int, limits Limits, lossScale float64) (result CompletionGradientResult, err error) {
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
			result = CompletionGradientResult{}
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
	result.Loss, err = completionCotangent(values, vocabulary, tokenIDs[promptTokens:], lossScale)
	if err != nil {
		return result, err
	}
	result.Tokens = completionTokens
	seed, err = torch.FromFloat32(values, info.Shape, info.Device, false)
	if err != nil {
		return result, err
	}
	if err := ctx.Err(); err != nil {
		return result, err
	}
	nativeGradients, err = model.VJP(ctx, snapshot, seed)
	if err != nil {
		return result, err
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

func completionCotangent(logits []float32, vocabulary int, targets []int64, lossScale float64) (float64, error) {
	if vocabulary < 2 || len(targets) == 0 || len(logits) != (len(targets)+1)*vocabulary ||
		math.IsNaN(lossScale) || math.IsInf(lossScale, 0) || lossScale <= 0 {
		return 0, fmt.Errorf("%w: invalid completion cotangent geometry", ErrCompletionStep)
	}
	for _, target := range targets {
		if target < 0 || target >= int64(vocabulary) {
			return 0, fmt.Errorf("%w: target token outside vocabulary", ErrCompletionStep)
		}
	}
	for _, value := range logits[:len(targets)*vocabulary] {
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return 0, fmt.Errorf("%w: nonfinite completion logit", ErrCompletionStep)
		}
	}

	factor := lossScale / float64(len(targets))
	totalLoss := 0.0
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
			return 0, fmt.Errorf("%w: invalid completion softmax normalization", ErrCompletionStep)
		}
		logSum := math.Log(sum)
		targetLogit := float64(logits[offset+int(target)])
		totalLoss += maximum + logSum - targetLogit
		for column := 0; column < vocabulary; column++ {
			probability := math.Exp(float64(logits[offset+column])-maximum) / sum
			gradient := probability * factor
			if column == int(target) {
				gradient -= factor
			}
			rounded := float32(gradient)
			if math.IsNaN(gradient) || math.IsInf(gradient, 0) ||
				math.IsNaN(float64(rounded)) || math.IsInf(float64(rounded), 0) {
				return 0, fmt.Errorf("%w: completion derivative is not finite FP32", ErrCompletionStep)
			}
			logits[offset+column] = rounded
		}
	}
	clear(logits[len(targets)*vocabulary:])
	loss := totalLoss / float64(len(targets))
	if math.IsNaN(loss) || math.IsInf(loss, 0) {
		return 0, fmt.Errorf("%w: nonfinite completion loss", ErrCompletionStep)
	}
	return loss, nil
}
