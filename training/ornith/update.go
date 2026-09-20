package ornith

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"

	"github.com/tayi-ai/arandu-llama/training/layers"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// ErrParameters reports an invalid adapter registry, ownership, or FP32 vector.
var ErrParameters = errors.New("ornith: parameter replacement rejected")

// ErrParameterRelease reports a native release failure after a committed
// replacement. The returned digest is nonempty; the new parameters are active.
// Failed releases remain owned for Close to retry. The caller must abort its job
// instead of treating this as an uncommitted update or retrying the replacement.
var ErrParameterRelease = errors.New("ornith: parameters committed but old handle release failed")

// ReplaceParameters atomically installs the 557056 ordered FP32 adapter values.
// Names/shapes follow InitializeAdapter: layers 3,7,...31, q/v projection, A/B.
// It validates all values and current registry metadata before native allocation,
// preserves each parameter's device, and materializes all 32 new leaves before
// changing any model reference. Base tensors and initializer receipts are intact.
// The returned digest hashes each UTF-8 name then its little-endian FP32 bytes.
//
// The caller must serialize this method with Forward, VJP, Close and any other
// access to model tensors, and must not mutate values during the call. Successful
// replacement invalidates older forward snapshots. Validation, preparation or
// cancellation failure returns an empty digest with every old reference intact.
// Cancellation is checked between native operations, not inside a running one.
// Native release errors after commit have the distinct ErrParameterRelease state.
// An adapter-only fixture may transfer its exact 32 referenced handles with an
// empty owned registry; otherwise every old parameter must already be owned once.
func (m *LoadedTextModel) ReplaceParameters(ctx context.Context, values []float32) (digest string, err error) {
	if ctx == nil || m == nil || m.Model == nil || len(m.Model.Layers) != 32 || len(m.Parameters) != 32 ||
		len(values) != 557056 || m.Model.generation == math.MaxUint64 {
		return "", ErrParameters
	}
	if err := ctx.Err(); err != nil {
		return "", err
	}
	// Reject the entire incoming vector before any new native tensor exists.
	for i, value := range values {
		if i&4095 == 0 {
			if err := ctx.Err(); err != nil {
				return "", err
			}
		}
		if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) {
			return "", fmt.Errorf("%w: nonfinite value", ErrParameters)
		}
	}
	_, expected := assemblyGeometry()
	devices, oldIndex, err := m.validateParameterRegistry(ctx, expected)
	if err != nil {
		return "", err
	}
	// Encode once after admission; this also fixes the aggregate's byte order.
	raw := make([]byte, len(values)*4)
	for i, value := range values {
		binary.LittleEndian.PutUint32(raw[i*4:], math.Float32bits(value))
	}
	prepared := make([]InitialParameter, 0, len(expected))
	committed := false
	defer func() {
		if !committed {
			for _, parameter := range prepared {
				err = errors.Join(err, parameter.Value.Close())
			}
		}
	}()
	aggregate := sha256.New()
	start := int64(0)
	for i, item := range expected {
		if err := ctx.Err(); err != nil {
			return "", err
		}
		content := raw[start : start+item.Bytes]
		leaf, err := torch.FromBytes(content, item.Shape, torch.Float32, devices[i], true)
		if err != nil {
			return "", err
		}
		prepared = append(prepared, InitialParameter{Name: item.ReferenceName, Value: leaf})
		if err := ctx.Err(); err != nil {
			return "", err
		}
		info, err := leaf.Info()
		if err != nil || info.DType != torch.Float32 || !info.RequiresGrad || info.Device != devices[i] ||
			!sameShape(info.Shape, item.Shape) || info.Elements*4 != item.Bytes {
			return "", errors.Join(ErrParameters, err)
		}
		if err := ctx.Err(); err != nil {
			return "", err
		}
		materialized, err := leaf.Bytes()
		if err != nil {
			return "", err
		}
		if !bytes.Equal(materialized, content) {
			return "", fmt.Errorf("%w: materialized FP32 bytes differ", ErrParameters)
		}
		_, _ = aggregate.Write([]byte(item.ReferenceName))
		_, _ = aggregate.Write(materialized)
		start += item.Bytes
	}
	if start != int64(len(raw)) {
		return "", ErrParameters
	}
	var nextAdapters [8]*layers.AttentionLoRA
	for i := range nextAdapters {
		base := i * 4
		nextAdapters[i] = &layers.AttentionLoRA{QueryA: prepared[base].Value, QueryB: prepared[base+1].Value,
			ValueA: prepared[base+2].Value, ValueB: prepared[base+3].Value, Alpha: 8}
	}
	nextOwned := make([]*torch.Tensor, 0, len(m.owned)+32)
	if len(m.owned) == 0 {
		for _, parameter := range prepared {
			nextOwned = append(nextOwned, parameter.Value)
		}
	} else {
		for _, value := range m.owned {
			if index, ok := oldIndex[*value]; ok {
				nextOwned = append(nextOwned, prepared[index].Value)
			} else {
				nextOwned = append(nextOwned, value)
			}
		}
	}
	digest = hex.EncodeToString(aggregate.Sum(nil))
	if err := ctx.Err(); err != nil {
		return "", err
	}
	// Commit consists only of Go assignments; no fallible native operation or
	// cancellation point can expose a partially replaced parameter registry.
	old := m.Parameters
	for i, adapter := range nextAdapters {
		m.Model.Layers[3+i*4].Adapter = adapter
	}
	m.Parameters, m.owned = prepared, nextOwned
	m.Model.generation++
	committed = true
	var releaseErrors []error
	for _, parameter := range old {
		if closeErr := parameter.Value.Close(); closeErr != nil {
			m.owned = append(m.owned, parameter.Value)
			releaseErrors = append(releaseErrors, closeErr)
		}
	}
	if len(releaseErrors) != 0 {
		return digest, errors.Join(ErrParameterRelease, errors.Join(releaseErrors...))
	}
	return digest, nil
}

func (m *LoadedTextModel) validateParameterRegistry(ctx context.Context, expected []AssemblyTensor) ([]torch.Device, map[torch.Tensor]int, error) {
	devices := make([]torch.Device, len(expected))
	oldIndex := make(map[torch.Tensor]int, len(expected))
	for i, layer := range m.Model.Layers {
		if (i%4 == 3) != (layer.Adapter != nil) || layer.Adapter != nil && layer.Adapter.Alpha != 8 {
			return nil, nil, fmt.Errorf("%w: full-attention adapter layout differs", ErrParameters)
		}
	}
	for i, item := range expected {
		if err := ctx.Err(); err != nil {
			return nil, nil, err
		}
		parameter := m.Parameters[i]
		layer := m.Model.Layers[3+(i/4)*4]
		a := layer.Adapter
		references := [4]*torch.Tensor{a.QueryA, a.QueryB, a.ValueA, a.ValueB}
		if parameter.Name != item.ReferenceName || parameter.Value == nil || references[i%4] != parameter.Value {
			return nil, nil, fmt.Errorf("%w: parameter names/order/references differ", ErrParameters)
		}
		if _, duplicate := oldIndex[*parameter.Value]; duplicate {
			return nil, nil, fmt.Errorf("%w: parameter handle reused", ErrParameters)
		}
		info, err := parameter.Value.Info()
		if err != nil || info.DType != torch.Float32 || !info.RequiresGrad || info.Device != layer.Device ||
			!sameShape(info.Shape, item.Shape) || info.Elements*4 != item.Bytes {
			return nil, nil, errors.Join(ErrParameters, err)
		}
		devices[i], oldIndex[*parameter.Value] = info.Device, i
	}
	if len(m.owned) != 0 {
		seen := make(map[torch.Tensor]bool, len(m.owned))
		for _, value := range m.owned {
			if value == nil || seen[*value] {
				return nil, nil, fmt.Errorf("%w: owned tensor handle absent or duplicated", ErrParameters)
			}
			seen[*value] = true
		}
		for value := range oldIndex {
			if !seen[value] {
				return nil, nil, fmt.Errorf("%w: parameter handle is not owned", ErrParameters)
			}
		}
	}
	// Releasing an old adapter must never close a handle also borrowed as base
	// or rotary data. Distinct native views remain independently owned handles.
	base := []*torch.Tensor{m.Model.Embedding, m.Model.FinalNorm, m.Model.Head}
	for _, layer := range m.Model.Layers {
		w := layer.Weights
		base = append(base, w.InputNorm, w.PostAttentionNorm, w.Gate, w.Up, w.Down, layer.Cosine, layer.Sine)
		if f := w.Full; f != nil {
			base = append(base, f.Query, f.Key, f.Value, f.Output, f.QueryNorm, f.KeyNorm)
		}
		if l := w.Linear; l != nil {
			base = append(base, l.QKV, l.Z, l.Beta, l.Alpha, l.Convolution, l.ALog, l.DTBias, l.Norm, l.Output)
		}
	}
	for _, value := range base {
		if value != nil {
			if _, alias := oldIndex[*value]; alias {
				return nil, nil, fmt.Errorf("%w: parameter handle aliases frozen or rotary data", ErrParameters)
			}
		}
	}
	return devices, oldIndex, nil
}
