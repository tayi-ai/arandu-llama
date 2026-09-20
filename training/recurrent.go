// Package training provides numerical references for differentiable model operations.
package training

import (
	"context"
	"errors"
	"fmt"
	"math"
)

// RecurrentGeometry defines independent sequences with equally sized heads.
// Tokens may be zero; every other dimension must be positive.
type RecurrentGeometry struct {
	Batch, Tokens, Heads, KeyDim, ValueDim int
}

// RecurrentInput holds float64 reference values in row-major order. Query/Key
// use [batch,tokens,heads,key], Value uses [batch,tokens,heads,value], LogDecay
// and Beta use [batch,tokens,heads], and InitialState uses [batch,heads,key,value].
// Query and Key are already normalized, Query already includes its scale, and
// Beta is already activated. No normalization, sigmoid or head repetition is
// implicit. All slices, including an explicit zero initial state, are required.
type RecurrentInput struct {
	Query, Key, Value, LogDecay, Beta, InitialState []float64
}

// RecurrentOptions bounds owned float64 storage, excluding caller-owned inputs
// and fixed Go object overhead. A positive checkpoint interval controls the
// number of saved boundary states and the maximum recomputed block length.
type RecurrentOptions struct {
	CheckpointInterval int
	MaxWorkspaceBytes  int64
}

// DefaultRecurrentOptions permits 256 MiB of owned float64 storage and recomputes
// blocks of at most 64 tokens. This CPU reference is not a GPU admission policy.
func DefaultRecurrentOptions() RecurrentOptions {
	return RecurrentOptions{CheckpointInterval: 64, MaxWorkspaceBytes: 256 << 20}
}

// RecurrentStorage describes the peak float64 storage reserved by the call.
// CheckpointStates counts sequence-wide boundaries, not individual heads.
// BlockStates includes the initial state of the one recomputed block.
type RecurrentStorage struct {
	WorkspaceBytes   int64
	CheckpointStates int
	BlockStates      int
	RecomputedTokens int
}

// RecurrentOutput owns Values [batch,tokens,heads,value] and FinalState
// [batch,heads,key,value]. Its slices never alias input slices.
type RecurrentOutput struct {
	Values, FinalState []float64
	Storage            RecurrentStorage
}

// RecurrentBackward owns the forward result and all input adjoints. Gradients
// uses the same layout as RecurrentInput; InitialState is its full adjoint.
type RecurrentBackward struct {
	Output    RecurrentOutput
	Gradients RecurrentInput
}

type recurrentSizes struct {
	query, value, scalar, state int
	storage                     RecurrentStorage
}

// GatedDeltaForward computes P=exp(g)*S, u=beta*(v-Pᵀk), S=P+k*uᵀ,
// and o=Sᵀq in a fixed scalar float64 order. It is a numerical reference, not
// a claim of equivalence to a fused or chunked floating-point kernel. Callers
// must keep all input slices unchanged until the function returns.
func GatedDeltaForward(ctx context.Context, geometry RecurrentGeometry, input RecurrentInput, options RecurrentOptions) (RecurrentOutput, error) {
	sizes, err := recurrentValidate(ctx, geometry, input, options, false)
	if err != nil {
		return RecurrentOutput{}, err
	}
	output := RecurrentOutput{
		Values: make([]float64, sizes.value), FinalState: append([]float64(nil), input.InitialState...), Storage: sizes.storage,
	}
	u := make([]float64, geometry.ValueDim)
	for token := 0; token < geometry.Tokens; token++ {
		if err := recurrentStep(ctx, geometry, input, token, output.FinalState, output.Values, u); err != nil {
			return RecurrentOutput{}, err
		}
	}
	if err := ctx.Err(); err != nil {
		return RecurrentOutput{}, err
	}
	return output, nil
}

// GatedDeltaVJP computes the vector-Jacobian product seeded by dOutput and
// dFinalState, both explicit finite slices with the corresponding output shape.
// It differentiates every token, including initial/final state connections.
// Only checkpoint boundaries and one recomputed block are retained; no complete
// per-token tape is retained across blocks. Smaller intervals save more
// checkpoints; larger intervals require a larger temporary block. All owned
// storage is checked against MaxWorkspaceBytes before allocation. Errors return
// no partial result and never mutate caller-owned inputs or seeds.
func GatedDeltaVJP(ctx context.Context, geometry RecurrentGeometry, input RecurrentInput, dOutput, dFinalState []float64, options RecurrentOptions) (RecurrentBackward, error) {
	sizes, err := recurrentValidate(ctx, geometry, input, options, true)
	if err != nil {
		return RecurrentBackward{}, err
	}
	if err := recurrentVector(ctx, "output adjoint", dOutput, sizes.value); err != nil {
		return RecurrentBackward{}, err
	}
	if err := recurrentVector(ctx, "final state adjoint", dFinalState, sizes.state); err != nil {
		return RecurrentBackward{}, err
	}
	result := RecurrentBackward{
		Output: RecurrentOutput{Values: make([]float64, sizes.value), FinalState: append([]float64(nil), input.InitialState...), Storage: sizes.storage},
		Gradients: RecurrentInput{
			Query: make([]float64, sizes.query), Key: make([]float64, sizes.query), Value: make([]float64, sizes.value),
			LogDecay: make([]float64, sizes.scalar), Beta: make([]float64, sizes.scalar), InitialState: append([]float64(nil), dFinalState...),
		},
	}
	checkpoints := make([]float64, sizes.storage.CheckpointStates*sizes.state)
	block := make([]float64, sizes.storage.BlockStates*sizes.state)
	u, e, du := make([]float64, geometry.ValueDim), make([]float64, geometry.ValueDim), make([]float64, geometry.ValueDim)
	for token := 0; token < geometry.Tokens; token++ {
		if token%options.CheckpointInterval == 0 {
			boundary := token / options.CheckpointInterval
			copy(checkpoints[boundary*sizes.state:(boundary+1)*sizes.state], result.Output.FinalState)
		}
		if err := recurrentStep(ctx, geometry, input, token, result.Output.FinalState, result.Output.Values, u); err != nil {
			return RecurrentBackward{}, err
		}
	}
	for boundary := sizes.storage.CheckpointStates - 1; boundary >= 0; boundary-- {
		start := boundary * options.CheckpointInterval
		span := min(options.CheckpointInterval, geometry.Tokens-start)
		copy(block[:sizes.state], checkpoints[boundary*sizes.state:(boundary+1)*sizes.state])
		for offset := 0; offset < span; offset++ {
			state := block[(offset+1)*sizes.state : (offset+2)*sizes.state]
			copy(state, block[offset*sizes.state:(offset+1)*sizes.state])
			if err := recurrentStep(ctx, geometry, input, start+offset, state, nil, u); err != nil {
				return RecurrentBackward{}, err
			}
		}
		for offset := span - 1; offset >= 0; offset-- {
			previous := block[offset*sizes.state : (offset+1)*sizes.state]
			state := block[(offset+1)*sizes.state : (offset+2)*sizes.state]
			if err := recurrentReverse(ctx, geometry, input, start+offset, previous, state, dOutput, &result.Gradients, u, e, du); err != nil {
				return RecurrentBackward{}, err
			}
		}
	}
	if err := ctx.Err(); err != nil {
		return RecurrentBackward{}, err
	}
	return result, nil
}

func recurrentValidate(ctx context.Context, g RecurrentGeometry, input RecurrentInput, options RecurrentOptions, backward bool) (recurrentSizes, error) {
	var sizes recurrentSizes
	if ctx == nil {
		return sizes, errors.New("recurrent: nil context")
	}
	if err := ctx.Err(); err != nil {
		return sizes, err
	}
	if g.Batch <= 0 || g.Tokens < 0 || g.Heads <= 0 || g.KeyDim <= 0 || g.ValueDim <= 0 ||
		options.CheckpointInterval <= 0 || options.MaxWorkspaceBytes <= 0 {
		return sizes, errors.New("recurrent: invalid geometry or workspace options")
	}
	var ok bool
	if sizes.state, ok = recurrentProduct(g.Batch, g.Heads, g.KeyDim, g.ValueDim); !ok {
		return sizes, errors.New("recurrent: state shape overflows")
	}
	if sizes.scalar, ok = recurrentProduct(g.Batch, g.Tokens, g.Heads); !ok {
		return sizes, errors.New("recurrent: sequence shape overflows")
	}
	if sizes.query, ok = recurrentProduct(sizes.scalar, g.KeyDim); !ok {
		return sizes, errors.New("recurrent: query shape overflows")
	}
	if sizes.value, ok = recurrentProduct(sizes.scalar, g.ValueDim); !ok {
		return sizes, errors.New("recurrent: value shape overflows")
	}
	// Validate the full peak payload using checked integer arithmetic. No slice
	// allocation occurs until every dimension, limit and input has been checked.
	counts := []int{sizes.value, sizes.state, g.ValueDim}
	if backward {
		if g.Tokens > 0 {
			sizes.storage.CheckpointStates = (g.Tokens-1)/options.CheckpointInterval + 1
			blockLength := min(options.CheckpointInterval, g.Tokens)
			if blockLength == int(^uint(0)>>1) {
				return sizes, errors.New("recurrent: block length overflows")
			}
			sizes.storage.BlockStates = blockLength + 1
			sizes.storage.RecomputedTokens = g.Tokens
		}
		checkpointElements, checkpointOK := recurrentProduct(sizes.storage.CheckpointStates, sizes.state)
		blockElements, blockOK := recurrentProduct(sizes.storage.BlockStates, sizes.state)
		if !checkpointOK || !blockOK {
			return sizes, errors.New("recurrent: checkpoint shape overflows")
		}
		counts = append(counts, sizes.query, sizes.query, sizes.value, sizes.scalar, sizes.scalar, sizes.state,
			checkpointElements, blockElements, g.ValueDim, g.ValueDim)
	}
	for _, count := range counts {
		if int64(count) > (math.MaxInt64-sizes.storage.WorkspaceBytes)/8 {
			return sizes, errors.New("recurrent: workspace size overflows")
		}
		sizes.storage.WorkspaceBytes += int64(count) * 8
	}
	if sizes.storage.WorkspaceBytes > options.MaxWorkspaceBytes {
		return sizes, errors.New("recurrent: workspace limit exceeded")
	}
	for _, vector := range []struct {
		name string
		data []float64
		size int
	}{
		{"query", input.Query, sizes.query}, {"key", input.Key, sizes.query}, {"value", input.Value, sizes.value},
		{"log decay", input.LogDecay, sizes.scalar}, {"beta", input.Beta, sizes.scalar}, {"initial state", input.InitialState, sizes.state},
	} {
		if err := recurrentVector(ctx, vector.name, vector.data, vector.size); err != nil {
			return sizes, err
		}
	}
	return sizes, nil
}

func recurrentProduct(values ...int) (int, bool) {
	product := 1
	for _, value := range values {
		if value != 0 && product > int(^uint(0)>>1)/value {
			return 0, false
		}
		product *= value
	}
	return product, true
}

func recurrentVector(ctx context.Context, name string, values []float64, size int) error {
	if len(values) != size {
		return fmt.Errorf("recurrent: %s shape mismatch", name)
	}
	for index, value := range values {
		if index&255 == 0 {
			if err := ctx.Err(); err != nil {
				return err
			}
		}
		if !recurrentFinite(value) {
			return fmt.Errorf("recurrent: nonfinite %s", name)
		}
	}
	return nil
}

func recurrentFinite(value float64) bool {
	return !math.IsNaN(value) && !math.IsInf(value, 0)
}

func recurrentStep(ctx context.Context, g RecurrentGeometry, input RecurrentInput, token int, state, output, u []float64) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	for batch := 0; batch < g.Batch; batch++ {
		for head := 0; head < g.Heads; head++ {
			position := (batch*g.Tokens+token)*g.Heads + head
			stateOffset := (batch*g.Heads + head) * g.KeyDim * g.ValueDim
			q := input.Query[position*g.KeyDim : (position+1)*g.KeyDim]
			k := input.Key[position*g.KeyDim : (position+1)*g.KeyDim]
			v := input.Value[position*g.ValueDim : (position+1)*g.ValueDim]
			s := state[stateOffset : stateOffset+g.KeyDim*g.ValueDim]
			rho, beta := math.Exp(input.LogDecay[position]), input.Beta[position]
			if !recurrentFinite(rho) {
				return errors.New("recurrent: decay exponential overflow")
			}
			for j := 0; j < g.ValueDim; j++ {
				memory := 0.0
				for i := 0; i < g.KeyDim; i++ {
					if i&255 == 0 {
						if err := ctx.Err(); err != nil {
							return err
						}
					}
					memory += (rho * s[i*g.ValueDim+j]) * k[i]
				}
				u[j] = beta * (v[j] - memory)
				if !recurrentFinite(memory) || !recurrentFinite(u[j]) {
					return errors.New("recurrent: state projection overflow")
				}
			}
			for j := 0; j < g.ValueDim; j++ {
				value := 0.0
				for i := 0; i < g.KeyDim; i++ {
					index := i*g.ValueDim + j
					if i&255 == 0 {
						if err := ctx.Err(); err != nil {
							return err
						}
					}
					s[index] = rho*s[index] + k[i]*u[j]
					if !recurrentFinite(s[index]) {
						return errors.New("recurrent: state update overflow")
					}
					value += s[index] * q[i]
				}
				if !recurrentFinite(value) {
					return errors.New("recurrent: output overflow")
				}
				if output != nil {
					output[position*g.ValueDim+j] = value
				}
			}
		}
	}
	return nil
}

func recurrentReverse(ctx context.Context, g RecurrentGeometry, input RecurrentInput, token int, previous, state, dOutput []float64, gradients *RecurrentInput, u, e, du []float64) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	for batch := 0; batch < g.Batch; batch++ {
		for head := 0; head < g.Heads; head++ {
			position := (batch*g.Tokens+token)*g.Heads + head
			stateOffset := (batch*g.Heads + head) * g.KeyDim * g.ValueDim
			q := input.Query[position*g.KeyDim : (position+1)*g.KeyDim]
			k := input.Key[position*g.KeyDim : (position+1)*g.KeyDim]
			v := input.Value[position*g.ValueDim : (position+1)*g.ValueDim]
			dy := dOutput[position*g.ValueDim : (position+1)*g.ValueDim]
			before := previous[stateOffset : stateOffset+g.KeyDim*g.ValueDim]
			after := state[stateOffset : stateOffset+g.KeyDim*g.ValueDim]
			carry := gradients.InitialState[stateOffset : stateOffset+g.KeyDim*g.ValueDim]
			rho, beta := math.Exp(input.LogDecay[position]), input.Beta[position]
			betaGradient := 0.0
			for j := 0; j < g.ValueDim; j++ {
				memory, deltaGradient := 0.0, 0.0
				for i := 0; i < g.KeyDim; i++ {
					index := i*g.ValueDim + j
					if i&255 == 0 {
						if err := ctx.Err(); err != nil {
							return err
						}
					}
					memory += (rho * before[index]) * k[i]
					deltaGradient += (carry[index] + q[i]*dy[j]) * k[i]
				}
				e[j], du[j] = v[j]-memory, deltaGradient
				u[j] = beta * e[j]
				gradients.Value[position*g.ValueDim+j] = beta * du[j]
				betaGradient += du[j] * e[j]
				if !recurrentFinite(e[j]) || !recurrentFinite(u[j]) || !recurrentFinite(du[j]) ||
					!recurrentFinite(gradients.Value[position*g.ValueDim+j]) || !recurrentFinite(betaGradient) {
					return errors.New("recurrent: value or beta adjoint overflow")
				}
			}
			gradients.Beta[position] = betaGradient
			decayGradient := 0.0
			for i := 0; i < g.KeyDim; i++ {
				queryGradient, keyGradient := 0.0, 0.0
				for j := 0; j < g.ValueDim; j++ {
					index := i*g.ValueDim + j
					if j&255 == 0 {
						if err := ctx.Err(); err != nil {
							return err
						}
					}
					p := rho * before[index]
					adjoint := carry[index] + q[i]*dy[j]
					dr := -beta * du[j]
					queryGradient += after[index] * dy[j]
					keyGradient += adjoint*u[j] + p*dr
					dp := adjoint + k[i]*dr
					decayGradient += dp * p
					carry[index] = rho * dp
					if !recurrentFinite(carry[index]) || !recurrentFinite(decayGradient) {
						return errors.New("recurrent: state or decay adjoint overflow")
					}
				}
				if !recurrentFinite(queryGradient) || !recurrentFinite(keyGradient) {
					return errors.New("recurrent: query or key adjoint overflow")
				}
				gradients.Query[position*g.KeyDim+i] = queryGradient
				gradients.Key[position*g.KeyDim+i] = keyGradient
			}
			gradients.LogDecay[position] = decayGradient
		}
	}
	return nil
}
