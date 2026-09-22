package layers

import (
	"context"
	"errors"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

// DecoderWeights describes one frozen Qwen3.5 decoder. Exactly one attention
// variant is supplied. LoRA belongs only to a full-attention variant.
type DecoderWeights struct {
	InputNorm, PostAttentionNorm, Gate, Up, Down *torch.Tensor
	Full                                         *AttentionWeights
	Linear                                       *LinearAttentionWeights
}

// DecoderConfig carries explicit limits and precision-independent dimensions.
type DecoderConfig struct {
	Epsilon          float64
	MaxInputElements int64
	Full             AttentionConfig
	Linear           LinearAttentionConfig
}

// DecoderGradients owns the input cotangent and, for a full-attention block with
// LoRA, its four parameter cotangents. The base weights are never updated.
type DecoderGradients struct {
	Input, QueryA, QueryB, ValueA, ValueB *torch.Tensor
}

// Close releases all owned gradient handles.
func (g *DecoderGradients) Close() error {
	if g == nil {
		return nil
	}
	return errors.Join(g.Input.Close(), g.QueryA.Close(), g.QueryB.Close(), g.ValueA.Close(), g.ValueB.Close())
}

// DecoderForward computes a complete decoder block and returns detached output.
// The caller checkpoints the input and invokes DecoderVJP when backpropagating.
// Attention and MLP arithmetic run in Float32; residual checkpoints retain the
// admitted input dtype.
func DecoderForward(ctx context.Context, x *torch.Tensor, weights DecoderWeights, adapter *AttentionLoRA, cosine, sine *torch.Tensor, config DecoderConfig) (*torch.Tensor, error) {
	if err := decoderValidate(ctx, x, weights, adapter, config); err != nil {
		return nil, err
	}
	var s scope
	defer s.close()
	input := s.run(func() (*torch.Tensor, error) { return x.Detach() })
	output, _, _ := decoderGraph(ctx, &s, input, weights, adapter, cosine, sine, config, false)
	output = s.run(func() (*torch.Tensor, error) { return output.Detach() })
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	return s.result(output)
}

// DecoderVJP recomputes one decoder block and propagates the complete input
// gradient through both residual and attention paths. Recurrent attention uses
// explicit state VJPs; it is not treated as a stop-gradient layer. Inputs and
// parameters remain caller-owned and are not changed.
func DecoderVJP(ctx context.Context, x *torch.Tensor, weights DecoderWeights, adapter *AttentionLoRA, cosine, sine, cotangent *torch.Tensor, config DecoderConfig) (*DecoderGradients, error) {
	if err := decoderValidate(ctx, x, weights, adapter, config); err != nil {
		return nil, err
	}
	if cotangent == nil {
		return nil, errors.New("layers: decoder requires an output cotangent")
	}
	xInfo, _ := x.Info()
	dInfo, err := cotangent.Info()
	if err != nil {
		return nil, err
	}
	if !equalShape(xInfo.Shape, dInfo.Shape) || xInfo.DType != dInfo.DType || xInfo.Device != dInfo.Device {
		return nil, errors.New("layers: decoder cotangent geometry or placement differs")
	}
	finite, err := cotangent.AllFinite()
	if err != nil || !finite {
		return nil, errors.Join(errors.New("layers: decoder cotangent must be finite"), err)
	}
	var s scope
	defer s.close()
	input := s.run(func() (*torch.Tensor, error) { return x.Detach() })
	input = s.run(func() (*torch.Tensor, error) { return input.SetRequiresGrad(true) })
	output, attentionInput, attentionOutput := decoderGraph(ctx, &s, input, weights, adapter, cosine, sine, config, true)
	if s.err != nil {
		return nil, s.err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	inputs := []*torch.Tensor{input}
	if weights.Linear != nil {
		inputs = append(inputs, attentionOutput)
	} else if adapter != nil {
		inputs = append(inputs, adapter.QueryA, adapter.QueryB, adapter.ValueA, adapter.ValueB)
	}
	// Recurrent attentionOutput is a detached leaf. The residual/MLP graph
	// shares only input with the separate input-normalization graph, so its
	// saved tensors can be released before recomputing the attention VJP.
	gradients, err := torch.Grad([]*torch.Tensor{output}, inputs, []*torch.Tensor{cotangent}, false, false)
	if err != nil {
		return nil, err
	}
	for _, gradient := range gradients {
		s.tensors = append(s.tensors, gradient)
	}
	inputGradient := gradients[0]
	if weights.Linear != nil {
		attentionGradient := s.run(func() (*torch.Tensor, error) {
			return LinearAttentionVJP(ctx, attentionInput, *weights.Linear, gradients[1], config.Linear)
		})
		if s.err != nil {
			return nil, s.err
		}
		throughNormalization, err := torch.Grad([]*torch.Tensor{attentionInput}, []*torch.Tensor{input}, []*torch.Tensor{attentionGradient}, false, false)
		if err != nil {
			return nil, err
		}
		s.tensors = append(s.tensors, throughNormalization...)
		inputGradient = s.run(func() (*torch.Tensor, error) { return inputGradient.Add(throughNormalization[0]) })
	}
	if s.err != nil {
		return nil, s.err
	}
	result := &DecoderGradients{}
	result.Input, err = s.result(inputGradient)
	if err != nil {
		return nil, err
	}
	if adapter != nil {
		result.QueryA, _ = s.result(gradients[1])
		result.QueryB, _ = s.result(gradients[2])
		result.ValueA, _ = s.result(gradients[3])
		result.ValueB, _ = s.result(gradients[4])
	}
	for _, gradient := range []*torch.Tensor{result.Input, result.QueryA, result.QueryB, result.ValueA, result.ValueB} {
		if gradient == nil {
			continue
		}
		finite, err := gradient.AllFinite()
		if err != nil || !finite {
			_ = result.Close()
			return nil, errors.Join(errors.New("layers: non-finite decoder gradient"), err)
		}
	}
	if err := ctx.Err(); err != nil {
		_ = result.Close()
		return nil, err
	}
	return result, nil
}

func decoderGraph(ctx context.Context, s *scope, input *torch.Tensor, weights DecoderWeights, adapter *AttentionLoRA, cosine, sine *torch.Tensor, config DecoderConfig, backward bool) (output, attentionInput, attentionOutput *torch.Tensor) {
	info, err := input.Info()
	if err != nil {
		s.err = err
		return nil, nil, nil
	}
	// Attention is admitted in Float32. Promote before RMSNorm so a finite
	// normalized activation is never rounded through Float16 and turned into
	// an infinity before entering either attention implementation. The
	// residual and MLP paths retain the qualified checkpoint storage dtype.
	attentionSource := s.run(func() (*torch.Tensor, error) { return input.To(info.Device, torch.Float32) })
	normalized := s.run(func() (*torch.Tensor, error) { return RMSNorm(attentionSource, weights.InputNorm, config.Epsilon) })
	attentionInput = normalized
	if weights.Full != nil {
		attentionOutput = s.run(func() (*torch.Tensor, error) {
			return FullAttention(attentionInput, *weights.Full, adapter, cosine, sine, config.Full)
		})
	} else {
		attentionOutput = s.run(func() (*torch.Tensor, error) {
			return ForwardLinearAttention(ctx, attentionInput, *weights.Linear, config.Linear)
		})
		if backward {
			attentionOutput = s.run(func() (*torch.Tensor, error) { return attentionOutput.SetRequiresGrad(true) })
		}
	}
	attentionStorage := s.run(func() (*torch.Tensor, error) { return attentionOutput.To(info.Device, info.DType) })
	residual := s.run(func() (*torch.Tensor, error) { return input.Add(attentionStorage) })
	normalized = s.run(func() (*torch.Tensor, error) { return RMSNorm(residual, weights.PostAttentionNorm, config.Epsilon) })
	feedForward := s.run(func() (*torch.Tensor, error) {
		return FeedForwardPromoted(normalized, weights.Gate, weights.Up, weights.Down)
	})
	feedForwardStorage := s.run(func() (*torch.Tensor, error) { return feedForward.To(info.Device, info.DType) })
	output = s.run(func() (*torch.Tensor, error) { return residual.Add(feedForwardStorage) })
	return output, attentionInput, attentionOutput
}

func decoderValidate(ctx context.Context, x *torch.Tensor, weights DecoderWeights, adapter *AttentionLoRA, config DecoderConfig) error {
	if ctx == nil || x == nil {
		return errors.New("layers: decoder requires context and input")
	}
	if err := ctx.Err(); err != nil {
		return err
	}
	info, err := x.Info()
	if err != nil {
		return err
	}
	if len(info.Shape) != 3 || info.Elements <= 0 || config.MaxInputElements <= 0 || info.Elements > config.MaxInputElements ||
		(info.DType != torch.Float16 && info.DType != torch.Float32) || !positiveFinite(config.Epsilon) {
		return errors.New("layers: decoder input geometry, precision or budget is invalid")
	}
	if (weights.Full == nil) == (weights.Linear == nil) || (weights.Linear != nil && adapter != nil) {
		return errors.New("layers: decoder requires exactly one attention variant; LoRA is full-attention only")
	}
	for _, weight := range []*torch.Tensor{weights.InputNorm, weights.PostAttentionNorm, weights.Gate, weights.Up, weights.Down} {
		if weight == nil {
			return errors.New("layers: missing frozen decoder weight")
		}
		w, err := weight.Info()
		if err != nil {
			return err
		}
		if w.RequiresGrad || w.Device != info.Device || w.DType != info.DType {
			return errors.New("layers: decoder storage weights must be frozen and match input placement and dtype")
		}
	}
	return nil
}
