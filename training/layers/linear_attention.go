package layers

import (
	"context"
	"errors"
	"math"

	"github.com/tayi-ai/arandu-llama/training/sequence"
	"github.com/tayi-ai/arandu-llama/training/tensor"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// LinearAttentionWeights contains unconverted, frozen Qwen3.5 checkpoint
// tensors. Projections use [out,in]. Convolution is [2*Hk*K+Hv*V,1,4],
// ALog and DTBias are [Hv], and Norm is [V] (a direct multiplier, not 1+w).
type LinearAttentionWeights struct {
	QKV, Z, Beta, Alpha, Convolution *torch.Tensor
	ALog, DTBias, Norm, Output       *torch.Tensor
}

// LinearAttentionConfig defines head geometry and explicit allocation limits.
// Epsilon belongs to gated RMS normalization; Q/K L2 normalization uses the
// HF constant 1e-6. Zero Sequence uses sequence.DefaultLimits().
// MaxWorkingElements limits a conservative projection/intermediate element
// estimate, separately from Sequence's recurrence budget. Zero selects 256Mi
// elements. Neither budget measures private native allocator workspaces.
type LinearAttentionConfig struct {
	KeyHeads, ValueHeads, KeyDimension, ValueDimension int64
	Epsilon                                            float64
	MaxWorkingElements                                 int64
	Sequence                                           sequence.SequenceLimits
}

type linearAttentionGeometry struct {
	batch, tokens, hidden, keys, values, channels, stateElements int64
	device                                                       torch.Device
}

// ForwardLinearAttention computes the complete causal, bias-free Qwen3.5
// linear-attention block in FP32, starting from zero convolution and GDN state.
// x is [B,T,D]; all arguments are borrowed and immutable. The caller owns the
// detached result. This block has no LoRA weights and no model-training claim.
func ForwardLinearAttention(ctx context.Context, x *torch.Tensor, weights LinearAttentionWeights, config LinearAttentionConfig) (*torch.Tensor, error) {
	geometry, config, err := validateLinearAttention(ctx, x, weights, nil, config)
	if err != nil {
		return nil, err
	}
	s := linearAttentionScope{ctx: ctx}
	defer s.close()
	input := s.run(func() (*torch.Tensor, error) { return x.Detach() })
	recurrence, gate := projectLinearAttention(&s, input, weights, geometry, config)
	if s.err != nil {
		return nil, s.err
	}
	output, err := sequence.Forward(ctx, recurrence, config.Sequence)
	if err != nil {
		return nil, err
	}
	defer output.Close()
	result := finishLinearAttention(&s, output.Values, gate, weights, geometry, config)
	result = s.run(func() (*torch.Tensor, error) { return result.Detach() })
	return s.result(result)
}

// LinearAttentionVJP returns the complete detached input cotangent for dy.
// It recomputes projections, invokes checkpointed recurrence VJP, and joins
// the recurrence and output-gate paths. Frozen parameters receive no gradients.
// No caller gradient flag, storage, or autograd graph is changed. Cancellation
// is checked between native operations; an already-running kernel cannot be
// preempted through this API.
func LinearAttentionVJP(ctx context.Context, x *torch.Tensor, weights LinearAttentionWeights, dy *torch.Tensor, config LinearAttentionConfig) (*torch.Tensor, error) {
	if dy == nil {
		return nil, errors.New("layers: linear attention VJP requires a cotangent")
	}
	geometry, config, err := validateLinearAttention(ctx, x, weights, dy, config)
	if err != nil {
		return nil, err
	}
	s := linearAttentionScope{ctx: ctx}
	defer s.close()
	input := s.run(func() (*torch.Tensor, error) { return x.Detach() })
	input = s.run(func() (*torch.Tensor, error) { return input.SetRequiresGrad(true) })
	recurrence, gate := projectLinearAttention(&s, input, weights, geometry, config)
	if s.err != nil {
		return nil, s.err
	}
	forward, err := sequence.Forward(ctx, recurrence, config.Sequence)
	if err != nil {
		return nil, err
	}
	defer forward.Close()
	outerScope := linearAttentionScope{ctx: ctx}
	defer outerScope.close()
	core := outerScope.run(func() (*torch.Tensor, error) { return forward.Values.Detach() })
	core = outerScope.run(func() (*torch.Tensor, error) { return core.SetRequiresGrad(true) })
	result := finishLinearAttention(&outerScope, core, gate, weights, geometry, config)
	// The Z/gate branch and recurrence projections share only the input leaf.
	// Release the output graph after its VJP; the projection graph remains
	// available for the recurrence cotangents below.
	outer := outerScope.grad([]*torch.Tensor{result}, []*torch.Tensor{core, input}, []*torch.Tensor{dy}, false)
	if outerScope.err != nil {
		return nil, outerScope.err
	}
	for _, gradient := range outer {
		owned, err := outerScope.scope.result(gradient)
		if err != nil {
			return nil, err
		}
		s.tensors = append(s.tensors, owned)
	}
	// Keep only the two cotangents. In particular, core aliases must close
	// before the first forward's Values storage can be released.
	outerScope.close()
	_ = forward.Close()
	backward, err := sequence.GradVJP(ctx, recurrence, outer[0], recurrence.InitialState, config.Sequence)
	if err != nil {
		return nil, err
	}
	defer backward.Close()
	// This caller consumes only adjoints, not the recomputed forward result.
	_ = backward.Output.Close()
	g := backward.Gradients
	inner := s.grad(
		[]*torch.Tensor{recurrence.Query, recurrence.Key, recurrence.Value, recurrence.LogDecay, recurrence.Beta},
		[]*torch.Tensor{input}, []*torch.Tensor{g.Query, g.Key, g.Value, g.LogDecay, g.Beta}, false)
	if s.err != nil {
		return nil, s.err
	}
	result = s.run(func() (*torch.Tensor, error) { return inner[0].Add(outer[1]) })
	result = s.run(func() (*torch.Tensor, error) { return result.Detach() })
	return s.result(result)
}

// Every intermediate is checked, including pre-normalization squares and
// exponentials: finite output must not hide an earlier FP32 overflow.
type linearAttentionScope struct {
	scope
	ctx context.Context
}

func (s *linearAttentionScope) check() bool {
	if s.err == nil {
		s.err = s.ctx.Err()
	}
	return s.err == nil
}

func (s *linearAttentionScope) finite(value *torch.Tensor) {
	if !s.check() {
		return
	}
	finite, err := value.AllFinite()
	if err != nil {
		s.err = err
	} else if !finite {
		s.err = errors.New("layers: nonfinite linear-attention intermediate")
	}
	s.check()
}

func (s *linearAttentionScope) run(operation func() (*torch.Tensor, error)) *torch.Tensor {
	if !s.check() {
		return nil
	}
	value := s.scope.run(operation)
	if s.err == nil {
		s.finite(value)
	}
	return value
}

func (s *linearAttentionScope) grad(outputs, inputs, seeds []*torch.Tensor, retain bool) []*torch.Tensor {
	if !s.check() {
		return nil
	}
	gradients, err := torch.Grad(outputs, inputs, seeds, retain, false)
	// Own every returned handle even if a later check fails.
	s.tensors = append(s.tensors, gradients...)
	if err != nil {
		s.err = err
	}
	for _, gradient := range gradients {
		s.finite(gradient)
	}
	return gradients
}

func (s *linearAttentionScope) result(value *torch.Tensor) (*torch.Tensor, error) {
	s.check()
	return s.scope.result(value)
}

func validateLinearAttention(ctx context.Context, x *torch.Tensor, weights LinearAttentionWeights, dy *torch.Tensor, config LinearAttentionConfig) (linearAttentionGeometry, LinearAttentionConfig, error) {
	var geometry linearAttentionGeometry
	fail := func(message string) (linearAttentionGeometry, LinearAttentionConfig, error) {
		return geometry, config, errors.New("layers: " + message)
	}
	if ctx == nil || x == nil {
		return fail("linear attention requires context and input")
	}
	if err := ctx.Err(); err != nil {
		return geometry, config, err
	}
	info, err := x.Info()
	if err != nil {
		return geometry, config, err
	}
	if len(info.Shape) != 3 || info.DType != torch.Float32 || info.Shape[0] <= 0 || info.Shape[1] <= 0 || info.Shape[2] <= 0 {
		return fail("linear attention input must be nonempty [B,T,D] Float32")
	}
	if config.KeyHeads <= 0 || config.ValueHeads <= 0 || config.KeyDimension <= 0 || config.ValueDimension <= 0 || config.ValueHeads%config.KeyHeads != 0 ||
		!positiveFinite(config.Epsilon) || !positiveFinite(float64(float32(config.Epsilon))) {
		return fail("invalid linear-attention geometry or epsilon")
	}
	if config.MaxWorkingElements == 0 {
		config.MaxWorkingElements = 256 << 20
	}
	if config.MaxWorkingElements <= 0 {
		return fail("invalid linear-attention working budget")
	}
	if config.Sequence == (sequence.SequenceLimits{}) {
		config.Sequence = sequence.DefaultLimits()
	}
	if config.Sequence.ChunkTokens <= 0 || config.Sequence.ChunkTokens > tensor.MaxChunkTokens || config.Sequence.MaxTokens <= 0 ||
		config.Sequence.MaxOwnedElements <= 0 || info.Shape[1] > config.Sequence.MaxTokens {
		return fail("invalid or exceeded linear-attention sequence limits")
	}
	geometry.batch, geometry.tokens, geometry.hidden, geometry.device = info.Shape[0], info.Shape[1], info.Shape[2], info.Device
	geometry.keys, err = linearAttentionProduct(config.KeyHeads, config.KeyDimension)
	if err != nil {
		return geometry, config, err
	}
	geometry.values, err = linearAttentionProduct(config.ValueHeads, config.ValueDimension)
	if err != nil {
		return geometry, config, err
	}
	if geometry.keys > (math.MaxInt64-geometry.values)/2 {
		return fail("linear-attention channel count overflows")
	}
	geometry.channels = 2*geometry.keys + geometry.values
	geometry.stateElements, err = linearAttentionProduct(geometry.batch, config.ValueHeads, config.KeyDimension, config.ValueDimension)
	if err != nil {
		return geometry, config, err
	}
	// Conservative sum for all owned projection/gradient intermediates. State
	// recomputation is additionally bounded inside sequence; weights are borrowed.
	remaining := config.MaxWorkingElements
	for _, dimensions := range [][]int64{
		{32, geometry.batch, geometry.tokens, geometry.hidden},
		{32, geometry.batch, geometry.tokens, geometry.channels},
		{32, geometry.batch, geometry.tokens, geometry.values},
		{64, geometry.batch, geometry.tokens, config.ValueHeads, config.KeyDimension},
		{32, geometry.batch, geometry.tokens, config.ValueHeads},
		{4, geometry.stateElements},
	} {
		count, countErr := linearAttentionProduct(dimensions...)
		if countErr != nil || count > remaining {
			return fail("linear-attention working budget exceeded")
		}
		remaining -= count
	}
	for _, expected := range []struct {
		name   string
		value  *torch.Tensor
		shape  []int64
		frozen bool
	}{
		{"input", x, info.Shape, false},
		{"QKV", weights.QKV, []int64{geometry.channels, geometry.hidden}, true},
		{"Z", weights.Z, []int64{geometry.values, geometry.hidden}, true},
		{"Beta", weights.Beta, []int64{config.ValueHeads, geometry.hidden}, true},
		{"Alpha", weights.Alpha, []int64{config.ValueHeads, geometry.hidden}, true},
		{"Convolution", weights.Convolution, []int64{geometry.channels, 1, 4}, true},
		{"ALog", weights.ALog, []int64{config.ValueHeads}, true},
		{"DTBias", weights.DTBias, []int64{config.ValueHeads}, true},
		{"Norm", weights.Norm, []int64{config.ValueDimension}, true},
		{"Output", weights.Output, []int64{geometry.hidden, geometry.values}, true},
		{"cotangent", dy, info.Shape, false},
	} {
		if err := ctx.Err(); err != nil {
			return geometry, config, err
		}
		if expected.name == "cotangent" && expected.value == nil {
			continue
		}
		if expected.value == nil {
			return fail("missing linear-attention " + expected.name)
		}
		actual, infoErr := expected.value.Info()
		if infoErr != nil {
			return geometry, config, infoErr
		}
		if !equalShape(actual.Shape, expected.shape) || actual.DType != torch.Float32 || actual.Device != geometry.device || (expected.frozen && actual.RequiresGrad) {
			return fail("invalid linear-attention " + expected.name + " shape, dtype, device or frozen flag")
		}
		finite, finiteErr := expected.value.AllFinite()
		if finiteErr != nil {
			return geometry, config, finiteErr
		}
		if !finite {
			return fail("nonfinite linear-attention " + expected.name)
		}
	}
	return geometry, config, ctx.Err()
}

func linearAttentionProduct(dimensions ...int64) (int64, error) {
	product := int64(1)
	for _, dimension := range dimensions {
		if dimension <= 0 || product > math.MaxInt64/dimension {
			return 0, errors.New("layers: linear-attention dimensions overflow")
		}
		product *= dimension
	}
	// A zero-state host allocation must also fit the platform's byte count.
	if uint64(product) > uint64(^uint(0)>>1)/4 {
		return 0, errors.New("layers: linear-attention byte count overflows")
	}
	return product, nil
}

func projectLinearAttention(s *linearAttentionScope, x *torch.Tensor, weights LinearAttentionWeights, geometry linearAttentionGeometry, config LinearAttentionConfig) (tensor.Input, *torch.Tensor) {
	mixed := s.run(func() (*torch.Tensor, error) { return Linear(x, weights.QKV) })
	var convolution *torch.Tensor
	for tap := int64(0); tap < 4 && s.check(); tap++ {
		lag := 3 - tap
		if lag >= geometry.tokens {
			continue // The left padding is zero, with no trainable conv weights.
		}
		shifted := mixed
		if lag > 0 {
			prefix := s.run(func() (*torch.Tensor, error) { return mixed.Slice(1, 0, lag, 1) })
			prefix = s.run(func() (*torch.Tensor, error) { return prefix.Scale(0) })
			past := s.run(func() (*torch.Tensor, error) { return mixed.Slice(1, 0, geometry.tokens-lag, 1) })
			shifted = s.run(func() (*torch.Tensor, error) { return torch.Cat([]*torch.Tensor{prefix, past}, 1) })
		}
		kernel := s.run(func() (*torch.Tensor, error) { return weights.Convolution.Select(1, 0) })
		kernel = s.run(func() (*torch.Tensor, error) { return kernel.Select(1, tap) })
		term := s.run(func() (*torch.Tensor, error) { return shifted.Mul(kernel) })
		if convolution == nil {
			convolution = term
		} else {
			convolution = s.run(func() (*torch.Tensor, error) { return convolution.Add(term) })
		}
	}
	convolution = s.run(func() (*torch.Tensor, error) { return convolution.SiLU() })
	query := s.run(func() (*torch.Tensor, error) { return convolution.Slice(2, 0, geometry.keys, 1) })
	key := s.run(func() (*torch.Tensor, error) { return convolution.Slice(2, geometry.keys, 2*geometry.keys, 1) })
	value := s.run(func() (*torch.Tensor, error) { return convolution.Slice(2, 2*geometry.keys, geometry.channels, 1) })
	query = s.run(func() (*torch.Tensor, error) {
		return query.Reshape([]int64{geometry.batch, geometry.tokens, config.KeyHeads, config.KeyDimension})
	})
	key = s.run(func() (*torch.Tensor, error) {
		return key.Reshape([]int64{geometry.batch, geometry.tokens, config.KeyHeads, config.KeyDimension})
	})
	value = s.run(func() (*torch.Tensor, error) {
		return value.Reshape([]int64{geometry.batch, geometry.tokens, config.ValueHeads, config.ValueDimension})
	})
	epsilon := s.run(func() (*torch.Tensor, error) {
		return torch.FromFloat32([]float32{1e-6}, []int64{1}, geometry.device, false)
	})
	normalize := func(input *torch.Tensor) *torch.Tensor {
		squares := s.run(func() (*torch.Tensor, error) { return input.Mul(input) })
		squares = s.run(func() (*torch.Tensor, error) { return squares.Sum([]int64{-1}, true) })
		squares = s.run(func() (*torch.Tensor, error) { return squares.Add(epsilon) })
		inverse := s.run(func() (*torch.Tensor, error) { return squares.RSqrt() })
		return s.run(func() (*torch.Tensor, error) { return input.Mul(inverse) })
	}
	query, key = normalize(query), normalize(key)
	if config.ValueHeads != config.KeyHeads {
		repeat := func(input *torch.Tensor) *torch.Tensor {
			parts := make([]*torch.Tensor, 0, config.ValueHeads)
			for head := int64(0); head < config.KeyHeads && s.check(); head++ {
				part := s.run(func() (*torch.Tensor, error) { return input.Select(2, head) })
				for count := int64(0); count < config.ValueHeads/config.KeyHeads; count++ {
					parts = append(parts, part)
				}
			}
			return s.run(func() (*torch.Tensor, error) { return torch.Stack(parts, 2) })
		}
		query, key = repeat(query), repeat(key)
	}
	query = s.run(func() (*torch.Tensor, error) { return query.Scale(1 / math.Sqrt(float64(config.KeyDimension))) })
	beta := s.run(func() (*torch.Tensor, error) { return Linear(x, weights.Beta) })
	beta = s.run(func() (*torch.Tensor, error) { return beta.Sigmoid() })
	decay := s.run(func() (*torch.Tensor, error) { return Linear(x, weights.Alpha) })
	decay = s.run(func() (*torch.Tensor, error) { return decay.Add(weights.DTBias) })
	decay = s.run(func() (*torch.Tensor, error) { return decay.Softplus() })
	negativeA := s.run(func() (*torch.Tensor, error) { return weights.ALog.Exp() })
	negativeA = s.run(func() (*torch.Tensor, error) { return negativeA.Scale(-1) })
	decay = s.run(func() (*torch.Tensor, error) { return decay.Mul(negativeA) })
	gate := s.run(func() (*torch.Tensor, error) { return Linear(x, weights.Z) })
	gate = s.run(func() (*torch.Tensor, error) {
		return gate.Reshape([]int64{geometry.batch, geometry.tokens, config.ValueHeads, config.ValueDimension})
	})
	initial := s.run(func() (*torch.Tensor, error) {
		return torch.FromFloat32(make([]float32, geometry.stateElements), []int64{geometry.batch, config.ValueHeads, config.KeyDimension, config.ValueDimension}, geometry.device, false)
	})
	return tensor.Input{Query: query, Key: key, Value: value, LogDecay: decay, Beta: beta, InitialState: initial}, gate
}

func finishLinearAttention(s *linearAttentionScope, core, gate *torch.Tensor, weights LinearAttentionWeights, geometry linearAttentionGeometry, config LinearAttentionConfig) *torch.Tensor {
	value := s.run(func() (*torch.Tensor, error) { return GatedRMSNorm(core, weights.Norm, gate, config.Epsilon) })
	value = s.run(func() (*torch.Tensor, error) {
		return value.Reshape([]int64{geometry.batch, geometry.tokens, geometry.values})
	})
	return s.run(func() (*torch.Tensor, error) { return Linear(value, weights.Output) })
}
