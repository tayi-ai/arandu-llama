package decoder

import (
	"context"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

// ErrFeatureTarget identifies invalid projected hidden-state supervision.
var ErrFeatureTarget = errors.New("decoder: feature target rejected")

// FeatureTarget is one immutable teacher vector already projected into the
// student's hidden width. Layer is the zero-based decoder whose output is
// supervised, before the final norm; Position is an absolute input-token index.
// Source identifies the teacher. Source/Layer/Position must be unique, while
// different sources may supervise the same student row independently.
//
// The objective is sum(Weight * sum((hidden-Values)^2) / hiddenWidth). Weights
// are finite, nonnegative and are not renormalized across sources or positions.
// A caller wanting an average over positions must include that factor in Weight.
// Zero-weight targets are validated but add neither loss nor gradient. Token,
// layer, projection and teacher provenance must be admitted by the cache owner;
// this calculation validates geometry and arithmetic, not semantic alignment.
type FeatureTarget struct {
	Source   string
	Layer    int
	Position int64
	Weight   float64
	Values   []float32
}

type featureCotangent struct {
	position int64
	values   []float64
}

// VJPWithFeatures adds projected feature MSE to a caller-supplied logit
// cotangent. The logit cotangent is used unchanged; lossScale scales only the
// additional feature derivatives. The returned feature loss is unscaled.
// Each feature derivative is injected at its decoder output during checkpointed
// reverse traversal, so it affects only that decoder and earlier parameters.
// Features before the first adapter can contribute loss but no adapter gradient.
// Working memory includes sparse host derivatives and dense host/native seed
// buffers for the current layer; it is additional to Limits.MaxCheckpointBytes.
// The snapshot and all targets remain caller-owned and must not be mutated.
// Failure returns no gradients and zero loss; the snapshot remains reusable.
func (m *TextModel) VJPWithFeatures(ctx context.Context, snapshot *Snapshot, logitCotangent *torch.Tensor, targets []FeatureTarget, lossScale float64) (Gradients, float64, error) {
	if err := m.validateSnapshot(ctx, snapshot, logitCotangent); err != nil {
		return nil, 0, err
	}
	seeds, loss, err := prepareFeatureCotangents(ctx, snapshot, targets, lossScale)
	if err != nil {
		return nil, 0, err
	}
	gradients, err := m.vjp(ctx, snapshot, logitCotangent, seeds)
	if err != nil {
		return nil, 0, err
	}
	return gradients, loss, nil
}

func prepareFeatureCotangents(ctx context.Context, snapshot *Snapshot, targets []FeatureTarget, lossScale float64) (map[int][]featureCotangent, float64, error) {
	if !(lossScale > 0) || math.IsInf(lossScale, 0) {
		return nil, 0, fmt.Errorf("%w: finite positive loss scale required", ErrFeatureTarget)
	}
	type identity struct {
		source   string
		layer    int
		position int64
	}
	seen := make(map[identity]bool, len(targets))
	// Validate every target before reading an activation or calculating a loss.
	for _, target := range targets {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		id := identity{target.Source, target.Layer, target.Position}
		if strings.TrimSpace(target.Source) == "" || seen[id] || target.Layer < 0 || target.Layer >= len(snapshot.states)-1 ||
			target.Weight < 0 || math.IsNaN(target.Weight) || math.IsInf(target.Weight, 0) {
			return nil, 0, fmt.Errorf("%w: source, layer, weight or duplicate position", ErrFeatureTarget)
		}
		seen[id] = true
		info, err := snapshot.states[target.Layer+1].Info()
		if err != nil {
			return nil, 0, err
		}
		if len(info.Shape) != 3 || info.Shape[0] != 1 || info.Shape[2] <= 0 ||
			(info.DType != torch.Float16 && info.DType != torch.Float32) || info.RequiresGrad ||
			target.Position < 0 || target.Position >= info.Shape[1] || int64(len(target.Values)) != info.Shape[2] {
			return nil, 0, fmt.Errorf("%w: hidden width or token position differs", ErrFeatureTarget)
		}
		for _, value := range target.Values {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
				return nil, 0, fmt.Errorf("%w: nonfinite teacher value", ErrFeatureTarget)
			}
		}
	}
	seeds := make(map[int][]featureCotangent)
	loss := 0.0
	for _, target := range targets {
		if err := ctx.Err(); err != nil {
			return nil, 0, err
		}
		if target.Weight == 0 {
			continue
		}
		// states[layer+1] is the detached decoder output, possibly placed on
		// the next layer's device without changing its storage dtype or values.
		row, err := snapshot.states[target.Layer+1].Slice(1, target.Position, target.Position+1, 1)
		if err != nil {
			return nil, 0, err
		}
		hidden, readErr := row.Float32Values()
		if err := errors.Join(readErr, row.Close()); err != nil {
			return nil, 0, err
		}
		derivative := make([]float64, len(hidden))
		weight := target.Weight / float64(len(hidden))
		for column, value := range hidden {
			difference := float64(value) - float64(target.Values[column])
			loss += weight * difference * difference
			derivative[column] = 2 * weight * difference * lossScale
			if math.IsNaN(loss) || math.IsInf(loss, 0) || math.IsNaN(derivative[column]) || math.IsInf(derivative[column], 0) {
				return nil, 0, fmt.Errorf("%w: nonfinite feature loss or derivative", ErrFeatureTarget)
			}
		}
		seeds[target.Layer] = append(seeds[target.Layer], featureCotangent{target.Position, derivative})
	}
	return seeds, loss, nil
}

func addFeatureCotangents(ctx context.Context, current *torch.Tensor, rows []featureCotangent) (result *torch.Tensor, err error) {
	var seed, placed *torch.Tensor
	defer func() {
		err = errors.Join(err, seed.Close(), placed.Close())
		if err != nil {
			err = errors.Join(err, result.Close())
			result = nil
		}
	}()
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	info, err := current.Info()
	if err != nil {
		return nil, err
	}
	// Sum teacher contributions in FP64 before one cast to boundary precision.
	byPosition := make(map[int64][]float64, len(rows))
	for _, row := range rows {
		accumulated := byPosition[row.position]
		if accumulated == nil {
			accumulated = make([]float64, len(row.values))
			byPosition[row.position] = accumulated
		}
		for column, value := range row.values {
			accumulated[column] += value
		}
	}
	values := make([]float32, info.Elements)
	for position, row := range byPosition {
		for column, value := range row {
			rounded := float32(value)
			if math.IsNaN(value) || math.IsInf(value, 0) || math.IsNaN(float64(rounded)) || math.IsInf(float64(rounded), 0) {
				return nil, fmt.Errorf("%w: feature cotangent exceeds Float32", ErrFeatureTarget)
			}
			values[position*info.Shape[2]+int64(column)] = rounded
		}
	}
	seed, err = torch.FromFloat32(values, info.Shape, info.Device, false)
	if err != nil {
		return nil, err
	}
	placed, err = seed.To(info.Device, info.DType)
	if err != nil {
		return nil, err
	}
	combined, err := current.Add(placed)
	if err != nil {
		return nil, err
	}
	finite, err := combined.AllFinite()
	if err != nil || !finite {
		return nil, errors.Join(fmt.Errorf("%w: nonfinite combined feature cotangent", ErrFeatureTarget), err, combined.Close())
	}
	return combined, nil
}
