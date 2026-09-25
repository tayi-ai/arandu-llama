package optim

import (
	"errors"
	"math"
)

// ErrAdamW identifies an invalid optimizer state, gradient or configuration.
var ErrAdamW = errors.New("optim: adamw update rejected")

// AdamWConfig fixes the numerical recipe for an entire run.
type AdamWConfig struct {
	LearningRate, Beta1, Beta2, Epsilon, WeightDecay, MaxGradientNorm float64
}

// AdamWState contains the complete resumable FP32 state for one ordered vector.
// Parameter names, base identity and dataset cursor belong to the job manifest.
type AdamWState struct {
	Step       uint64
	Parameters []float32
	First      []float32
	Second     []float32
}

// AdamWReceipt describes the accepted update without changing its recipe.
type AdamWReceipt struct {
	Step             uint64
	GradientNorm     float64
	ClipCoefficient  float64
	ChangedParameter int
}

// NewAdamWState copies an admitted parameter vector and starts zero moments.
func NewAdamWState(parameters []float32) (AdamWState, error) {
	if len(parameters) == 0 || !allFinite(parameters) {
		return AdamWState{}, ErrAdamW
	}
	return AdamWState{Parameters: append([]float32(nil), parameters...), First: make([]float32, len(parameters)), Second: make([]float32, len(parameters))}, nil
}

// ValidateAdamWState refuses incomplete, nonfinite or negative-moment state.
func ValidateAdamWState(state AdamWState) error {
	n := len(state.Parameters)
	if n == 0 || len(state.First) != n || len(state.Second) != n || !allFinite(state.Parameters) || !allFinite(state.First) || !allFinite(state.Second) {
		return ErrAdamW
	}
	for _, value := range state.Second {
		if value < 0 {
			return ErrAdamW
		}
	}
	return nil
}

// UpdateAdamW returns a new state; a rejected update leaves the old state intact.
// The norm and clipping cover the entire ordered gradient vector once.
func UpdateAdamW(state AdamWState, gradient []float32, config AdamWConfig) (AdamWState, AdamWReceipt, error) {
	if err := ValidateAdamWState(state); err != nil {
		return AdamWState{}, AdamWReceipt{}, err
	}
	if len(gradient) != len(state.Parameters) || !allFinite(gradient) || !validConfig(config) || state.Step == math.MaxUint64 {
		return AdamWState{}, AdamWReceipt{}, ErrAdamW
	}
	var sum float64
	for _, value := range gradient {
		sum += float64(value) * float64(value)
	}
	norm := math.Sqrt(sum)
	if !finite(norm) || norm == 0 {
		return AdamWState{}, AdamWReceipt{}, ErrAdamW
	}
	clip := math.Min(1, config.MaxGradientNorm/(norm+1e-12))
	step := state.Step + 1
	denom1, denom2 := 1-math.Pow(config.Beta1, float64(step)), 1-math.Pow(config.Beta2, float64(step))
	if !finite(denom1) || !finite(denom2) || denom1 <= 0 || denom2 <= 0 {
		return AdamWState{}, AdamWReceipt{}, ErrAdamW
	}
	next := AdamWState{Step: step, Parameters: make([]float32, len(gradient)), First: make([]float32, len(gradient)), Second: make([]float32, len(gradient))}
	receipt := AdamWReceipt{Step: step, GradientNorm: norm, ClipCoefficient: clip}
	for i, value := range gradient {
		g := float32(float64(value) * clip)
		next.First[i] = float32(config.Beta1*float64(state.First[i]) + (1-config.Beta1)*float64(g))
		next.Second[i] = float32(config.Beta2*float64(state.Second[i]) + (1-config.Beta2)*float64(g*g))
		mean := float64(next.First[i]) / denom1
		variance := float64(next.Second[i]) / denom2
		updated := float64(state.Parameters[i])*(1-config.LearningRate*config.WeightDecay) - config.LearningRate*mean/(math.Sqrt(variance)+config.Epsilon)
		next.Parameters[i] = float32(updated)
		if !finite(updated) || !finite(float64(next.Parameters[i])) || !finite(float64(next.First[i])) || !finite(float64(next.Second[i])) || next.Second[i] < 0 {
			return AdamWState{}, AdamWReceipt{}, ErrAdamW
		}
		if math.Float32bits(next.Parameters[i]) != math.Float32bits(state.Parameters[i]) {
			receipt.ChangedParameter++
		}
	}
	if receipt.ChangedParameter == 0 {
		return AdamWState{}, AdamWReceipt{}, ErrAdamW
	}
	return next, receipt, nil
}

func validConfig(c AdamWConfig) bool {
	return finite(c.LearningRate) && c.LearningRate > 0 && finite(c.Beta1) && c.Beta1 >= 0 && c.Beta1 < 1 && finite(c.Beta2) && c.Beta2 >= 0 && c.Beta2 < 1 && finite(c.Epsilon) && c.Epsilon > 0 && finite(c.WeightDecay) && c.WeightDecay >= 0 && finite(c.MaxGradientNorm) && c.MaxGradientNorm > 0
}

func allFinite(values []float32) bool {
	for _, value := range values {
		if !finite(float64(value)) {
			return false
		}
	}
	return true
}

func finite(value float64) bool { return !math.IsNaN(value) && !math.IsInf(value, 0) }
