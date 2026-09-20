// Package sequence connects bounded gated-delta chunks with explicit state
// checkpoints. It does not retain an autograd graph across chunk boundaries.
package sequence

import (
	"context"
	"errors"
	"fmt"
	"slices"

	"github.com/tayi-ai/arandu-llama/training/tensor"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// ErrInvalidInput identifies invalid sequence shapes, handles, or placement.
var ErrInvalidInput = tensor.ErrInvalidInput

// ErrLimit identifies a sequence or chunk admission limit violation.
var ErrLimit = tensor.ErrLimit

// ErrNonFinite identifies non-finite inputs, outputs, or gradients.
var ErrNonFinite = tensor.ErrNonFinite

// SequenceLimits bounds the sequence and its manager-owned tensor payload.
// ChunkTokens must be 1..tensor.MaxChunkTokens. MaxTokens and MaxOwnedElements
// must be positive. The element budget includes outputs, checkpoint states,
// all input adjoints, state cotangents, concatenation temporaries, and the
// declared output/final-state seeds even though those seeds are borrowed.
// Caller-owned primal inputs, Go metadata, native allocator overhead, and
// kernel/autograd workspace inside a chunk are excluded. Chunk work is subject
// to tensor.DefaultLimits independently; this budget is not measured peak RAM.
type SequenceLimits struct {
	ChunkTokens, MaxTokens, MaxOwnedElements int64
}

// DefaultLimits uses eight-token chunks, at most 4096 tokens, and an admission
// budget of 268435456 tensor elements. This is an element count, not a byte cap.
func DefaultLimits() SequenceLimits {
	return SequenceLimits{ChunkTokens: 8, MaxTokens: 4096, MaxOwnedElements: 256 << 20}
}

// Backward owns the forward output and every input adjoint. Its tensors are
// detached, and their layout is unchanged from tensor.Input/tensor.Output.
type Backward struct {
	Output    *tensor.Output
	Gradients *tensor.Gradients
}

// Close releases the owned output and gradient handles and is idempotent.
func (b *Backward) Close() error {
	if b == nil {
		return nil
	}
	return errors.Join(b.Output.Close(), b.Gradients.Close())
}

type geometry struct {
	batch, tokens, heads, key, value, chunks int64
	info                                     torch.Info
}

// Forward evaluates a complete nonempty sequence in bounded chunks. Inputs use
// tensor.Input layouts, are borrowed and immutable until return, and must all
// be float32 or float64 on the same device. Query scaling, normalization, and
// beta activation have already happened. Only the current state and output
// pieces survive each chunk; the returned tensors own detached storage.
func Forward(ctx context.Context, input tensor.Input, limits SequenceLimits) (*tensor.Output, error) {
	g, err := validate(ctx, input, limits, false)
	if err != nil {
		return nil, err
	}
	owned := newArena()
	defer owned.closeAll()
	detached, err := detach(input, owned)
	if err != nil {
		return nil, err
	}
	output, _, err := forward(ctx, detached, g, limits, false, owned)
	if err != nil {
		return nil, err
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	owned.release(output.Values, output.FinalState)
	return output, nil
}

// GradVJP evaluates the forward sequence while retaining only chunk boundary
// states and output pieces. It then recomputes each chunk in reverse, seeds it
// with both output and state cotangents, and concatenates all five sequence
// adjoints in their original token order. InitialState is the complete adjoint
// of the first state. Both finite seeds are explicit and borrowed; nil does not
// mean zero. No caller-owned tensor, gradient flag, or graph is mutated.
func GradVJP(ctx context.Context, input tensor.Input, dValues, dFinalState *torch.Tensor, limits SequenceLimits) (*Backward, error) {
	g, err := validate(ctx, input, limits, true)
	if err != nil {
		return nil, err
	}
	if err := match(ctx, "output seed", dValues, []int64{g.batch, g.tokens, g.heads, g.value}, g.info); err != nil {
		return nil, err
	}
	if err := match(ctx, "final-state seed", dFinalState, []int64{g.batch, g.heads, g.key, g.value}, g.info); err != nil {
		return nil, err
	}
	owned := newArena()
	defer owned.closeAll()
	detached, err := detach(input, owned)
	if err != nil {
		return nil, err
	}
	valueSeed, err := owned.take(dValues.Detach())
	if err != nil {
		return nil, err
	}
	stateSeed, err := owned.take(dFinalState.Detach())
	if err != nil {
		return nil, err
	}
	output, checkpoints, err := forward(ctx, detached, g, limits, true, owned)
	if err != nil {
		return nil, err
	}
	var pieces [5][]*torch.Tensor
	for i := range pieces {
		pieces[i] = make([]*torch.Tensor, int(g.chunks))
	}
	for chunk := g.chunks - 1; chunk >= 0; chunk-- {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		start := chunk * limits.ChunkTokens
		end := start + min(limits.ChunkTokens, g.tokens-start)
		gradient, err := reverseChunk(ctx, detached, checkpoints[chunk], valueSeed, stateSeed, start, end, limits)
		if err != nil {
			return nil, err
		}
		fields := inputTensors(tensor.Input(*gradient))
		for i, value := range fields {
			owned.add(value)
			if i < 5 {
				pieces[i][chunk] = value
			}
		}
		owned.close(stateSeed)
		stateSeed = gradient.InitialState
		// The initial checkpoint is a detached borrowed input handle. It is also
		// owned by this arena, so closing the view never closes the caller's handle.
		owned.close(checkpoints[chunk])
	}
	var complete [6]*torch.Tensor
	for i := range pieces {
		if err := ctx.Err(); err != nil {
			return nil, err
		}
		complete[i], err = owned.take(torch.Cat(pieces[i], 1))
		if err != nil {
			return nil, err
		}
		owned.close(pieces[i]...)
	}
	complete[5] = stateSeed
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	gradient := tensor.Gradients(fromTensors(complete[:]))
	owned.release(output.Values, output.FinalState)
	owned.release(complete[:]...)
	return &Backward{Output: output, Gradients: &gradient}, nil
}

func forward(ctx context.Context, input tensor.Input, g geometry, limits SequenceLimits, saveStates bool, owned *arena) (*tensor.Output, []*torch.Tensor, error) {
	state := input.InitialState
	values := make([]*torch.Tensor, 0, int(g.chunks))
	var checkpoints []*torch.Tensor
	if saveStates {
		checkpoints = make([]*torch.Tensor, 0, int(g.chunks))
	}
	for start := int64(0); start < g.tokens; {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		end := start + min(limits.ChunkTokens, g.tokens-start)
		if saveStates {
			checkpoints = append(checkpoints, state)
		}
		output, err := forwardChunk(ctx, input, state, start, end, limits)
		if err != nil {
			return nil, nil, err
		}
		owned.add(output.Values, output.FinalState)
		values = append(values, output.Values)
		if !saveStates && state != input.InitialState {
			owned.close(state)
		}
		state, start = output.FinalState, end
	}
	combined, err := owned.take(torch.Cat(values, 1))
	if err != nil {
		return nil, nil, err
	}
	owned.close(values...)
	return &tensor.Output{Values: combined, FinalState: state}, checkpoints, nil
}

func forwardChunk(ctx context.Context, input tensor.Input, state *torch.Tensor, start, end int64, limits SequenceLimits) (*tensor.Output, error) {
	views := newArena()
	defer views.closeAll()
	chunk, err := sliceInput(input, state, start, end, views)
	if err != nil {
		return nil, err
	}
	return tensor.Forward(ctx, chunk, chunkLimits(limits))
}

func reverseChunk(ctx context.Context, input tensor.Input, state, dValues, dState *torch.Tensor, start, end int64, limits SequenceLimits) (*tensor.Gradients, error) {
	views := newArena()
	defer views.closeAll()
	chunk, err := sliceInput(input, state, start, end, views)
	if err != nil {
		return nil, err
	}
	seed, err := views.take(dValues.Slice(1, start, end, 1))
	if err != nil {
		return nil, err
	}
	return tensor.GradVJP(ctx, chunk, seed, dState, chunkLimits(limits))
}

func chunkLimits(limits SequenceLimits) tensor.Limits {
	result := tensor.DefaultLimits()
	result.Tokens = limits.ChunkTokens
	return result
}

func sliceInput(input tensor.Input, state *torch.Tensor, start, end int64, owned *arena) (tensor.Input, error) {
	fields := inputTensors(input)
	for i := 0; i < 5; i++ {
		view, err := owned.take(fields[i].Slice(1, start, end, 1))
		if err != nil {
			return tensor.Input{}, err
		}
		fields[i] = view
	}
	fields[5] = state
	return fromTensors(fields), nil
}

func detach(input tensor.Input, owned *arena) (tensor.Input, error) {
	fields := inputTensors(input)
	for i, value := range fields {
		detached, err := owned.take(value.Detach())
		if err != nil {
			return tensor.Input{}, err
		}
		fields[i] = detached
	}
	return fromTensors(fields), nil
}

type arena struct{ handles map[*torch.Tensor]struct{} }

func newArena() *arena { return &arena{handles: make(map[*torch.Tensor]struct{})} }

func (a *arena) add(values ...*torch.Tensor) {
	for _, value := range values {
		a.handles[value] = struct{}{}
	}
}

func (a *arena) take(value *torch.Tensor, err error) (*torch.Tensor, error) {
	if err != nil {
		return nil, err
	}
	a.add(value)
	return value, nil
}

func (a *arena) release(values ...*torch.Tensor) {
	for _, value := range values {
		delete(a.handles, value)
	}
}

func (a *arena) close(values ...*torch.Tensor) {
	for _, value := range values {
		if _, owned := a.handles[value]; owned {
			_ = value.Close()
			delete(a.handles, value)
		}
	}
}

func (a *arena) closeAll() {
	for value := range a.handles {
		_ = value.Close()
		delete(a.handles, value)
	}
}

func inputTensors(input tensor.Input) []*torch.Tensor {
	return []*torch.Tensor{input.Query, input.Key, input.Value, input.LogDecay, input.Beta, input.InitialState}
}

func fromTensors(values []*torch.Tensor) tensor.Input {
	return tensor.Input{Query: values[0], Key: values[1], Value: values[2], LogDecay: values[3], Beta: values[4], InitialState: values[5]}
}

func validate(ctx context.Context, input tensor.Input, limits SequenceLimits, backward bool) (geometry, error) {
	var g geometry
	if ctx == nil {
		return g, fmt.Errorf("%w: sequence context is required", ErrInvalidInput)
	}
	if err := ctx.Err(); err != nil {
		return g, err
	}
	if limits.ChunkTokens <= 0 || limits.ChunkTokens > tensor.MaxChunkTokens || limits.MaxTokens <= 0 || limits.MaxOwnedElements <= 0 {
		return g, ErrLimit
	}
	fields := inputTensors(input)
	infos := make([]torch.Info, len(fields))
	for i, value := range fields {
		info, err := value.Info()
		if err != nil {
			return g, fmt.Errorf("%w: sequence input %d: %v", ErrInvalidInput, i, err)
		}
		infos[i] = info
	}
	base := infos[0]
	if len(base.Shape) != 4 || len(infos[2].Shape) != 4 || (base.DType != torch.Float32 && base.DType != torch.Float64) {
		return g, fmt.Errorf("%w: sequence requires rank-four query/value and float32 or float64", ErrInvalidInput)
	}
	g = geometry{batch: base.Shape[0], tokens: base.Shape[1], heads: base.Shape[2], key: base.Shape[3], value: infos[2].Shape[3], info: base}
	if g.batch <= 0 || g.tokens <= 0 || g.heads <= 0 || g.key <= 0 || g.value <= 0 {
		return g, fmt.Errorf("%w: sequence dimensions must be positive", ErrInvalidInput)
	}
	if g.tokens > limits.MaxTokens {
		return g, ErrLimit
	}
	g.chunks = 1 + (g.tokens-1)/limits.ChunkTokens
	if g.chunks > int64(int(^uint(0)>>1)) {
		return g, ErrLimit
	}
	shapes := [][]int64{
		{g.batch, g.tokens, g.heads, g.key}, {g.batch, g.tokens, g.heads, g.key},
		{g.batch, g.tokens, g.heads, g.value}, {g.batch, g.tokens, g.heads},
		{g.batch, g.tokens, g.heads}, {g.batch, g.heads, g.key, g.value},
	}
	for i, info := range infos {
		if !slices.Equal(info.Shape, shapes[i]) || info.DType != base.DType || info.Device != base.Device || info.Elements <= 0 {
			return g, fmt.Errorf("%w: sequence input %d shape/dtype/device mismatch", ErrInvalidInput, i)
		}
	}
	if !admit(infos, g.chunks, limits.MaxOwnedElements, backward) {
		return g, ErrLimit
	}
	for i, value := range fields {
		if err := finite(ctx, fmt.Sprintf("sequence input %d", i), value); err != nil {
			return g, err
		}
	}
	return g, nil
}

// Each phase is admitted independently; their storage is not added together.
// Products and sums are checked against the caller's budget before allocation.
func admit(infos []torch.Info, chunks, limit int64, backward bool) bool {
	value, state := infos[2].Elements, infos[5].Elements
	forward := int64(0)
	if !backward {
		return add(&forward, limit, 2, value) && add(&forward, limit, 2, state)
	}
	// Concatenated output + output pieces + N states + explicit seeds.
	if !add(&forward, limit, 3, value) || !add(&forward, limit, chunks, state) || !add(&forward, limit, 1, state) {
		return false
	}
	// Full output + checkpoints/final state + both rolling state adjoints +
	// all sequence gradient pieces + the borrowed seeds.
	reverse := int64(0)
	if !add(&reverse, limit, 2, value) || !add(&reverse, limit, chunks, state) || !add(&reverse, limit, 3, state) {
		return false
	}
	largest := int64(0)
	for _, info := range infos[:5] {
		if !add(&reverse, limit, 1, info.Elements) {
			return false
		}
		largest = max(largest, info.Elements)
	}
	// One concatenated field overlaps its pieces; all other fields are closed
	// immediately after concatenation. All checkpoint buffers are gone here.
	concatenate := int64(0)
	if !add(&concatenate, limit, 2, value) || !add(&concatenate, limit, 3, state) || !add(&concatenate, limit, 1, largest) {
		return false
	}
	for _, info := range infos[:5] {
		if !add(&concatenate, limit, 1, info.Elements) {
			return false
		}
	}
	return true
}

func add(total *int64, limit, count, elements int64) bool {
	if count <= 0 || elements <= 0 || count > limit/elements {
		return false
	}
	product := count * elements
	if *total > limit-product {
		return false
	}
	*total += product
	return true
}

func match(ctx context.Context, name string, value *torch.Tensor, shape []int64, base torch.Info) error {
	info, err := value.Info()
	if err != nil {
		return fmt.Errorf("%w: %s: %v", ErrInvalidInput, name, err)
	}
	if !slices.Equal(info.Shape, shape) || info.DType != base.DType || info.Device != base.Device {
		return fmt.Errorf("%w: %s shape/dtype/device mismatch", ErrInvalidInput, name)
	}
	return finite(ctx, name, value)
}

func finite(ctx context.Context, name string, value *torch.Tensor) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	ok, err := value.AllFinite()
	if err != nil {
		return fmt.Errorf("sequence %s: %w", name, err)
	}
	if !ok {
		return fmt.Errorf("%w: sequence %s", ErrNonFinite, name)
	}
	return ctx.Err()
}
