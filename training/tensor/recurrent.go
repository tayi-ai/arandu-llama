// Package tensor implements bounded model operations in Go using optional
// native tensor primitives. It does not execute a model or start training jobs.
package tensor

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

// ErrInvalidInput identifies an invalid shape, dtype, placement, or input handle.
var ErrInvalidInput = errors.New("gated delta: invalid input")

// ErrLimit means the chunk exceeds its explicit admission limits.
var ErrLimit = errors.New("gated delta: chunk exceeds admission limits")

// ErrNonFinite means an input, output, or gradient contains NaN or infinity.
var ErrNonFinite = errors.New("gated delta: non-finite tensor")

const (
	// MaxChunkTokens is the absolute nonempty chunk length supported by this API.
	MaxChunkTokens int64 = 32
	// MaxWorkingElements bounds the arithmetic admission estimate. It is not a
	// measurement of native allocator, autograd metadata, or kernel workspace.
	MaxWorkingElements int64 = 64 << 20
)

// Limits bounds one chunk. Both values must be positive and may only lower the
// absolute caps. WorkingElements admits 8*(T+2)*state_elements plus eight times
// the input and output elements. This deliberately conservative element budget
// does not replace measurement of native peak memory on a target device.
type Limits struct {
	Tokens          int64
	WorkingElements int64
}

// DefaultLimits returns the absolute chunk caps; callers can impose lower caps.
func DefaultLimits() Limits {
	return Limits{Tokens: MaxChunkTokens, WorkingElements: MaxWorkingElements}
}

// Input contains six same-dtype, same-device tensors. Query and Key have shape
// [B,T,H,K], Value [B,T,H,V], LogDecay and Beta [B,T,H], and InitialState
// [B,H,K,V]. Every dimension must be positive. Query and Key are already L2
// normalized; Query already includes its scale; Beta is already activated.
// No normalization, scaling, sigmoid, or head replication is implicit. Callers
// must keep tensor values and handles unchanged until the operation returns.
type Input struct {
	Query, Key, Value, LogDecay, Beta, InitialState *torch.Tensor
}

// Output owns detached Values [B,T,H,V] and FinalState [B,H,K,V]. Closing the
// input handles does not invalidate these tensors. Call Close to release them.
type Output struct {
	Values, FinalState *torch.Tensor
}

// Close releases both owned output handles and is idempotent.
func (o *Output) Close() error {
	if o == nil {
		return nil
	}
	return errors.Join(o.Values.Close(), o.FinalState.Close())
}

// Gradients owns the six input cotangents, in the same shapes as Input. These
// tensors have no retained autograd graph and do not accumulate into .grad.
type Gradients Input

// Close releases all owned gradient handles and is idempotent.
func (g *Gradients) Close() error {
	if g == nil {
		return nil
	}
	return closeTensors(Input(*g).tensors())
}

type geometry struct{ batch, tokens, heads, key, value int64 }

// Forward computes P=exp(g)*S, e=v-Pᵀk, u=beta*e, S=P+k*uᵀ, o=Sᵀq.
// It detaches borrowed inputs before computation, so repeated forward chunks do
// not retain an autograd tape. Values are computed in the input dtype, with no
// implicit precision promotion. Context is checked between native calls; it
// cannot interrupt a native kernel already in progress.
func Forward(ctx context.Context, input Input, limits Limits) (*Output, error) {
	g, _, err := validate(ctx, input, limits)
	if err != nil {
		return nil, err
	}
	borrowed, err := detach(input, false)
	if err != nil {
		return nil, err
	}
	defer closeTensors(borrowed.tensors())
	return forward(ctx, borrowed, g)
}

// GradVJP recomputes one bounded chunk and returns the vector-Jacobian product
// seeded by dValues and dFinalState. Both finite cotangents are required, even
// when one is zero. Detached input leaves prevent retaining earlier chunks;
// propagate the returned InitialState gradient into the previous chunk's
// dFinalState. A caller managing a long sequence retains its own checkpoints
// and recomputes chunks instead of retaining a sequence-wide native tape.
// The inputs' requires-grad flags, values, and existing graphs are unchanged.
func GradVJP(ctx context.Context, input Input, dValues, dFinalState *torch.Tensor, limits Limits) (*Gradients, error) {
	g, info, err := validate(ctx, input, limits)
	if err != nil {
		return nil, err
	}
	if err := validateTensor(ctx, "output cotangent", dValues, []int64{g.batch, g.tokens, g.heads, g.value}, info); err != nil {
		return nil, err
	}
	if err := validateTensor(ctx, "final state cotangent", dFinalState, []int64{g.batch, g.heads, g.key, g.value}, info); err != nil {
		return nil, err
	}
	leaves, err := detach(input, true)
	if err != nil {
		return nil, err
	}
	defer closeTensors(leaves.tensors())
	output, err := forward(ctx, leaves, g)
	if err != nil {
		return nil, err
	}
	defer output.Close()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	gradients, err := torch.Grad(
		[]*torch.Tensor{output.Values, output.FinalState}, leaves.tensors(),
		[]*torch.Tensor{dValues, dFinalState}, false, false,
	)
	if err != nil {
		return nil, fmt.Errorf("gated delta: native VJP: %w", err)
	}
	if len(gradients) != 6 {
		_ = closeTensors(gradients)
		return nil, errors.New("gated delta: native VJP returned an invalid gradient count")
	}
	for i, gradient := range gradients {
		if err := finite(ctx, inputNames[i]+" gradient", gradient); err != nil {
			_ = closeTensors(gradients)
			return nil, err
		}
	}
	result := Gradients(fromTensors(gradients))
	return &result, nil
}

func forward(ctx context.Context, input Input, g geometry) (result *Output, err error) {
	state := input.InitialState
	ownedState := false
	values := make([]*torch.Tensor, 0, g.tokens)
	defer func() {
		_ = closeTensors(values)
		if result == nil && ownedState {
			_ = state.Close()
		}
	}()
	for token := int64(0); token < g.tokens; token++ {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		next, value, err := step(ctx, input, state, g, token)
		if err != nil {
			return nil, err
		}
		if ownedState {
			_ = state.Close()
		}
		state, ownedState = next, true
		values = append(values, value)
	}
	combined, err := torch.Cat(values, 1)
	if err != nil {
		return nil, err
	}
	if err := finite(ctx, "output", combined); err != nil {
		_ = combined.Close()
		return nil, err
	}
	if err := finite(ctx, "final state", state); err != nil {
		_ = combined.Close()
		return nil, err
	}
	return &Output{Values: combined, FinalState: state}, nil
}

// scope releases every intermediate handle; native autograd separately retains
// only the tensors it needs. Failed operations stop subsequent native calls.
type scope struct {
	ctx   context.Context
	owned []*torch.Tensor
	err   error
}

func (s *scope) call(operation func() (*torch.Tensor, error)) *torch.Tensor {
	if s.err != nil {
		return nil
	}
	if s.err = s.ctx.Err(); s.err != nil {
		return nil
	}
	tensor, err := operation()
	if err != nil {
		s.err = err
		return nil
	}
	s.owned = append(s.owned, tensor)
	return tensor
}

func (s *scope) token(tensor *torch.Tensor, token int64, shape []int64) *torch.Tensor {
	view := s.call(func() (*torch.Tensor, error) { return tensor.Slice(1, token, token+1, 1) })
	return s.call(func() (*torch.Tensor, error) { return view.Reshape(shape) })
}

func (s *scope) closeExcept(keep ...*torch.Tensor) {
	for _, tensor := range s.owned {
		if !slices.Contains(keep, tensor) {
			_ = tensor.Close()
		}
	}
}

func step(ctx context.Context, input Input, state *torch.Tensor, g geometry, token int64) (next, value *torch.Tensor, err error) {
	s := scope{ctx: ctx}
	defer func() { s.closeExcept(next, value) }()
	q := s.token(input.Query, token, []int64{g.batch, g.heads, g.key, 1})
	k := s.token(input.Key, token, []int64{g.batch, g.heads, g.key, 1})
	v := s.token(input.Value, token, []int64{g.batch, g.heads, g.value, 1})
	logDecay := s.token(input.LogDecay, token, []int64{g.batch, g.heads, 1, 1})
	beta := s.token(input.Beta, token, []int64{g.batch, g.heads, 1, 1})
	decay := s.call(func() (*torch.Tensor, error) { return logDecay.Exp() })
	p := s.call(func() (*torch.Tensor, error) { return state.Mul(decay) })
	pTranspose := s.call(func() (*torch.Tensor, error) { return p.Transpose(-2, -1) })
	prediction := s.call(func() (*torch.Tensor, error) { return pTranspose.MatMul(k) })
	e := s.call(func() (*torch.Tensor, error) { return v.Sub(prediction) })
	u := s.call(func() (*torch.Tensor, error) { return beta.Mul(e) })
	uTranspose := s.call(func() (*torch.Tensor, error) { return u.Transpose(-2, -1) })
	update := s.call(func() (*torch.Tensor, error) { return k.MatMul(uTranspose) })
	newState := s.call(func() (*torch.Tensor, error) { return p.Add(update) })
	stateTranspose := s.call(func() (*torch.Tensor, error) { return newState.Transpose(-2, -1) })
	column := s.call(func() (*torch.Tensor, error) { return stateTranspose.MatMul(q) })
	output := s.call(func() (*torch.Tensor, error) { return column.Reshape([]int64{g.batch, 1, g.heads, g.value}) })
	if s.err != nil {
		return nil, nil, s.err
	}
	return newState, output, nil
}

var inputNames = []string{"query", "key", "value", "log decay", "beta", "initial state"}

func (input Input) tensors() []*torch.Tensor {
	return []*torch.Tensor{input.Query, input.Key, input.Value, input.LogDecay, input.Beta, input.InitialState}
}

func fromTensors(tensors []*torch.Tensor) Input {
	return Input{tensors[0], tensors[1], tensors[2], tensors[3], tensors[4], tensors[5]}
}

func closeTensors(tensors []*torch.Tensor) error {
	var err error
	for _, tensor := range tensors {
		err = errors.Join(err, tensor.Close())
	}
	return err
}

func detach(input Input, gradients bool) (Input, error) {
	var owned []*torch.Tensor
	for _, tensor := range input.tensors() {
		leaf, err := tensor.Detach()
		if err == nil && gradients {
			var enabled *torch.Tensor
			enabled, err = leaf.SetRequiresGrad(true)
			_ = leaf.Close()
			leaf = enabled
		}
		if err != nil {
			_ = closeTensors(owned)
			return Input{}, err
		}
		owned = append(owned, leaf)
	}
	return fromTensors(owned), nil
}

func validate(ctx context.Context, input Input, limits Limits) (geometry, torch.Info, error) {
	var g geometry
	var base torch.Info
	if ctx == nil {
		return g, base, fmt.Errorf("%w: context is required", ErrInvalidInput)
	}
	if err := ctx.Err(); err != nil {
		return g, base, err
	}
	if limits.Tokens <= 0 || limits.Tokens > MaxChunkTokens || limits.WorkingElements <= 0 || limits.WorkingElements > MaxWorkingElements {
		return g, base, ErrLimit
	}
	infos := make([]torch.Info, 6)
	for i, tensor := range input.tensors() {
		info, err := tensor.Info()
		if err != nil {
			return g, base, fmt.Errorf("%w: %s metadata: %v", ErrInvalidInput, inputNames[i], err)
		}
		infos[i] = info
	}
	base = infos[0]
	if len(base.Shape) != 4 || len(infos[2].Shape) != 4 {
		return g, base, fmt.Errorf("%w: query and value must have rank four", ErrInvalidInput)
	}
	if !slices.Contains([]torch.DType{torch.Float32, torch.Float64, torch.Float16, torch.BFloat16}, base.DType) {
		return g, base, fmt.Errorf("%w: a floating dtype is required", ErrInvalidInput)
	}
	g = geometry{base.Shape[0], base.Shape[1], base.Shape[2], base.Shape[3], infos[2].Shape[3]}
	if g.batch <= 0 || g.tokens <= 0 || g.heads <= 0 || g.key <= 0 || g.value <= 0 {
		return g, base, fmt.Errorf("%w: dimensions must be positive", ErrInvalidInput)
	}
	if g.tokens > limits.Tokens {
		return g, base, ErrLimit
	}
	shapes := [][]int64{
		{g.batch, g.tokens, g.heads, g.key}, {g.batch, g.tokens, g.heads, g.key},
		{g.batch, g.tokens, g.heads, g.value}, {g.batch, g.tokens, g.heads},
		{g.batch, g.tokens, g.heads}, {g.batch, g.heads, g.key, g.value},
	}
	for i, info := range infos {
		if !slices.Equal(info.Shape, shapes[i]) || info.DType != base.DType || info.Device != base.Device {
			return g, base, fmt.Errorf("%w: %s shape, dtype, or device mismatch", ErrInvalidInput, inputNames[i])
		}
	}
	used := int64(0)
	if !admit(&used, limits.WorkingElements, 8, g.tokens+2, infos[5].Elements) {
		return g, base, ErrLimit
	}
	for _, info := range infos {
		if !admit(&used, limits.WorkingElements, 8, info.Elements) {
			return g, base, ErrLimit
		}
	}
	if !admit(&used, limits.WorkingElements, 8, infos[2].Elements) {
		return g, base, ErrLimit
	}
	for i, tensor := range input.tensors() {
		if err := finite(ctx, inputNames[i], tensor); err != nil {
			return g, base, err
		}
	}
	return g, base, nil
}

func admit(used *int64, limit int64, factors ...int64) bool {
	product := int64(1)
	for _, factor := range factors {
		if factor <= 0 || product > limit/factor {
			return false
		}
		product *= factor
	}
	if *used > limit-product {
		return false
	}
	*used += product
	return true
}

func validateTensor(ctx context.Context, name string, tensor *torch.Tensor, shape []int64, base torch.Info) error {
	info, err := tensor.Info()
	if err != nil {
		return fmt.Errorf("%w: %s metadata: %v", ErrInvalidInput, name, err)
	}
	if !slices.Equal(info.Shape, shape) || info.DType != base.DType || info.Device != base.Device {
		return fmt.Errorf("%w: %s shape, dtype, or device mismatch", ErrInvalidInput, name)
	}
	return finite(ctx, name, tensor)
}

func finite(ctx context.Context, name string, tensor *torch.Tensor) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	ok, err := tensor.AllFinite()
	if err != nil {
		return fmt.Errorf("gated delta: %s finite check: %w", name, err)
	}
	if !ok {
		return fmt.Errorf("%w: %s", ErrNonFinite, name)
	}
	return ctx.Err()
}
