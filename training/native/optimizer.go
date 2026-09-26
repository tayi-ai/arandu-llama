package native

import (
	"errors"
	"fmt"
	"math"
	"slices"
	"sync"
)

// These constants describe the native Go recipe, not configurable defaults.
// The arithmetic is explicitly FP32 and does not claim PyTorch bitwise parity.
const (
	ExperimentAdamWNumericMode  = "experiment-go-adamw-v2"
	ExperimentAdamWReplicas     = 20
	ExperimentAdamWGlobalBatch  = 21
	ExperimentAdamWMaxUpdates   = 18
	ExperimentAdamWLearningRate = 2e-6
	ExperimentAdamWBeta1        = 0.9
	ExperimentAdamWBeta2        = 0.999
	ExperimentAdamWEpsilon      = 1e-8
	ExperimentAdamWWeightDecay  = 0.0
	ExperimentAdamWGradientClip = 1.0
	ExperimentAdamWLossScale    = 1.0 / 1024.0
)

// ExperimentRankGradient contains one replica's sum of scaled example gradients.
// Exactly one rank contributes two sequential examples; the other 19 contribute
// one each. The application owns that coverage; values alone cannot attest it.
// Values must follow the same flat trainable-parameter order on every rank.
type ExperimentRankGradient struct {
	Rank   int       `json:"rank"`
	Values []float32 `json:"values"`
}

// ExperimentAdamWState owns a portable snapshot of parameters and optimizer moments.
// Tensor names, shapes and the frozen base identity belong to the backend receipt.
type ExperimentAdamWState struct {
	SchemaVersion int       `json:"schema_version"`
	NumericMode   string    `json:"numeric_mode"`
	Step          int       `json:"step"`
	Parameters    []float32 `json:"parameters"`
	FirstMoment   []float32 `json:"first_moment"`
	SecondMoment  []float32 `json:"second_moment"`
}

// ExperimentAdamWUpdate records the reduction and clipping applied by one update.
type ExperimentAdamWUpdate struct {
	NumericMode     string  `json:"numeric_mode"`
	Step            int     `json:"step"`
	ReplicaCount    int     `json:"replica_count"`
	GlobalBatch     int     `json:"global_batch"`
	GradientNorm    float64 `json:"gradient_norm"`
	ClipCoefficient float64 `json:"clip_coefficient"`
}

// ExperimentAdamW owns only trainable FP32 parameters and their two FP32 moments.
// It accepts no frozen base and performs no I/O, allocation on a GPU or process
// launch. Update and Snapshot are serialized; caller gradient buffers must not
// be modified concurrently with Update.
type ExperimentAdamW struct {
	mu           sync.Mutex
	parameters   []float32
	firstMoment  []float32
	secondMoment []float32
	step         int
}

// NewExperimentAdamW copies finite initial parameters and initializes zero moments.
func NewExperimentAdamW(parameters []float32) (*ExperimentAdamW, error) {
	if len(parameters) == 0 {
		return nil, errors.New("experiment AdamW: parameters must not be empty")
	}
	for index, value := range parameters {
		if !experimentFinite32(value) {
			return nil, fmt.Errorf("experiment AdamW: nonfinite initial parameter at %d", index)
		}
	}
	return &ExperimentAdamW{parameters: slices.Clone(parameters), firstMoment: make([]float32, len(parameters)), secondMoment: make([]float32, len(parameters))}, nil
}

// RestoreExperimentAdamW validates a snapshot and copies all buffers independently.
// A step-zero snapshot must have zero moments; a completed snapshot may be read
// but cannot perform a nineteenth update.
func RestoreExperimentAdamW(state ExperimentAdamWState) (*ExperimentAdamW, error) {
	if state.SchemaVersion != 1 || state.NumericMode != ExperimentAdamWNumericMode || state.Step < 0 || state.Step > ExperimentAdamWMaxUpdates {
		return nil, errors.New("experiment AdamW: unsupported snapshot identity or step")
	}
	count := len(state.Parameters)
	if count == 0 || len(state.FirstMoment) != count || len(state.SecondMoment) != count {
		return nil, errors.New("experiment AdamW: snapshot shape mismatch")
	}
	for index, parameter := range state.Parameters {
		first, second := state.FirstMoment[index], state.SecondMoment[index]
		if !experimentFinite32(parameter) || !experimentFinite32(first) || !experimentFinite32(second) || second < 0 {
			return nil, fmt.Errorf("experiment AdamW: invalid snapshot value at %d", index)
		}
		if state.Step == 0 && (first != 0 || second != 0) {
			return nil, errors.New("experiment AdamW: initial snapshot has nonzero moments")
		}
	}
	return &ExperimentAdamW{parameters: slices.Clone(state.Parameters), firstMoment: slices.Clone(state.FirstMoment),
		secondMoment: slices.Clone(state.SecondMoment), step: state.Step}, nil
}

// Snapshot returns detached buffers that callers may serialize or modify.
func (optimizer *ExperimentAdamW) Snapshot() ExperimentAdamWState {
	if optimizer == nil {
		return ExperimentAdamWState{}
	}
	optimizer.mu.Lock()
	defer optimizer.mu.Unlock()
	return ExperimentAdamWState{SchemaVersion: 1, NumericMode: ExperimentAdamWNumericMode, Step: optimizer.step,
		Parameters: slices.Clone(optimizer.parameters), FirstMoment: slices.Clone(optimizer.firstMoment), SecondMoment: slices.Clone(optimizer.secondMoment)}
}

// Update reduces exactly ranks 0..19 in that order, divides the sum by 21
// examples (not 20 replicas), and unscales the mean. Local sums are FP32; this
// changes reduction grouping from v1 and does not claim bitwise equivalence.
// It computes the global L2 norm in float64,
// clips once, then applies bias-corrected AdamW with zero weight decay.
//
// Reduction, clipping, moments and parameter updates are staged and checked
// before any optimizer state changes. Rejection leaves every buffer and Step
// unchanged. No caller-owned slice is mutated or retained.
func (optimizer *ExperimentAdamW) Update(replicas []ExperimentRankGradient) (ExperimentAdamWUpdate, error) {
	if optimizer == nil {
		return ExperimentAdamWUpdate{}, errors.New("experiment AdamW: optimizer is nil")
	}
	optimizer.mu.Lock()
	defer optimizer.mu.Unlock()
	fail := func(message string) (ExperimentAdamWUpdate, error) {
		return ExperimentAdamWUpdate{}, errors.New("experiment AdamW: " + message)
	}
	count := len(optimizer.parameters)
	if count == 0 || len(optimizer.firstMoment) != count || len(optimizer.secondMoment) != count {
		return fail("optimizer is not initialized")
	}
	if optimizer.step >= ExperimentAdamWMaxUpdates {
		return fail("all 18 updates are already complete")
	}
	if len(replicas) != ExperimentAdamWReplicas {
		return fail("exactly 20 replica gradient sums are required")
	}
	var ordered [ExperimentAdamWReplicas][]float32
	for _, replica := range replicas {
		if replica.Rank < 0 || replica.Rank >= ExperimentAdamWReplicas || ordered[replica.Rank] != nil {
			return fail("replica rank is invalid or duplicated")
		}
		if len(replica.Values) != count {
			return fail("replica gradient shape mismatch")
		}
		for _, value := range replica.Values {
			if !experimentFinite32(value) {
				return fail("replica gradient contains a nonfinite value")
			}
		}
		ordered[replica.Rank] = replica.Values
	}
	gradients := make([]float32, count)
	normSquared := float64(0)
	for index := range gradients {
		sum := float32(0)
		for rank := 0; rank < ExperimentAdamWReplicas; rank++ {
			if ordered[rank] == nil {
				return fail("replica rank is missing")
			}
			sum = float32(sum + ordered[rank][index])
			if !experimentFinite32(sum) {
				return fail("FP32 gradient reduction overflow")
			}
		}
		mean := float32(sum / float32(ExperimentAdamWGlobalBatch))
		gradient := float32(mean / float32(ExperimentAdamWLossScale))
		if !experimentFinite32(gradient) {
			return fail("FP32 gradient unscaling overflow")
		}
		gradients[index] = gradient
		normSquared += float64(gradient) * float64(gradient)
	}
	norm := math.Sqrt(normSquared)
	if math.IsNaN(norm) || math.IsInf(norm, 0) {
		return fail("gradient norm overflow")
	}
	clip := math.Min(1, ExperimentAdamWGradientClip/(norm+1e-12))
	nextStep := optimizer.step + 1
	bias1 := 1 - math.Pow(ExperimentAdamWBeta1, float64(nextStep))
	bias2 := 1 - math.Pow(ExperimentAdamWBeta2, float64(nextStep))
	stepSize := float32(ExperimentAdamWLearningRate / bias1)
	secondBiasRoot := float32(math.Sqrt(bias2))
	parameters := make([]float32, count)
	firstMoment := make([]float32, count)
	secondMoment := make([]float32, count)
	for index, gradient := range gradients {
		gradient = float32(gradient * float32(clip))
		// Explicit casts fix the rounding points of this native numerical mode.
		first := float32(float32(float32(ExperimentAdamWBeta1)*optimizer.firstMoment[index]) + float32(float32(1-ExperimentAdamWBeta1)*gradient))
		square := float32(gradient * gradient)
		second := float32(float32(float32(ExperimentAdamWBeta2)*optimizer.secondMoment[index]) + float32(float32(1-ExperimentAdamWBeta2)*square))
		denominator := float32(float32(math.Sqrt(float64(second))) / secondBiasRoot)
		denominator = float32(denominator + float32(ExperimentAdamWEpsilon))
		delta := float32(stepSize * float32(first/denominator))
		parameter := float32(optimizer.parameters[index] - delta)
		if !experimentFinite32(gradient) || !experimentFinite32(first) || !experimentFinite32(second) || second < 0 ||
			!experimentFinite32(denominator) || denominator <= 0 || !experimentFinite32(delta) || !experimentFinite32(parameter) {
			return fail("FP32 optimizer arithmetic overflow")
		}
		parameters[index], firstMoment[index], secondMoment[index] = parameter, first, second
	}
	optimizer.parameters, optimizer.firstMoment, optimizer.secondMoment = parameters, firstMoment, secondMoment
	optimizer.step = nextStep
	return ExperimentAdamWUpdate{NumericMode: ExperimentAdamWNumericMode, Step: nextStep, ReplicaCount: ExperimentAdamWReplicas,
		GlobalBatch: ExperimentAdamWGlobalBatch, GradientNorm: norm, ClipCoefficient: clip}, nil
}

func experimentFinite32(value float32) bool {
	return !math.IsNaN(float64(value)) && !math.IsInf(float64(value), 0)
}
