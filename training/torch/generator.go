package torch

import (
	"errors"
	"math"
	"unsafe"
)

// ErrGeneratorClosed means a generator is uninitialized or has been closed.
var ErrGeneratorClosed = errors.New("torch: generator is closed or uninitialized")

type generatorState struct{ handle unsafe.Pointer }

// Generator owns a private native CPU random stream. Close is required and is
// idempotent even if the Go value is copied. Calls and Close are serialized with
// tensor operations. Creating or using this stream does not seed or consume the
// default generator. Reproducibility across releases and devices is not implied.
type Generator struct{ state *generatorState }

// NewCPUGenerator creates an independent CPU stream with an explicit 64-bit seed.
func NewCPUGenerator(seed uint64) (*Generator, error) {
	if !nativeEnabled {
		return nil, ErrUnavailable
	}
	calls.Lock()
	defer calls.Unlock()
	handle, err := nativeGeneratorCreate(seed)
	if err != nil {
		return nil, err
	}
	if handle == nil {
		return nil, errors.New("torch: native operation returned a null generator")
	}
	return &Generator{state: &generatorState{handle: handle}}, nil
}

// Uniform creates an owned CPU tensor using this stream and the explicit dtype.
// The tensor does not require gradients. Bounds must be finite with low <= high;
// native dtype range checks also apply. Values follow the native uniform kernel,
// including its rounding for Float16 and BFloat16. The caller admits allocation
// size and selects any subsequent device transfer or gradient requirement.
func (g *Generator) Uniform(shape []int64, low, high float64, dtype DType) (*Tensor, error) {
	if !nativeEnabled {
		return nil, ErrUnavailable
	}
	elements, err := checkedShape(shape)
	if err != nil {
		return nil, err
	}
	width, err := dtypeWidth(dtype)
	if err != nil {
		return nil, err
	}
	if dtype == Int64 || dtype == Bool {
		return nil, errors.New("torch: uniform requires a floating dtype")
	}
	if elements > math.MaxInt64/width {
		return nil, errors.New("torch: uniform storage size overflows")
	}
	if math.IsNaN(low) || math.IsNaN(high) || math.IsInf(low, 0) || math.IsInf(high, 0) || low > high {
		return nil, errors.New("torch: uniform bounds must be finite and ordered")
	}
	calls.Lock()
	defer calls.Unlock()
	if g == nil || g.state == nil || g.state.handle == nil {
		return nil, ErrGeneratorClosed
	}
	handle, err := nativeGeneratorUniform(g.state.handle, shape, low, high, dtype)
	return wrap(handle, err)
}

// Close releases the private stream without invalidating previously drawn tensors.
func (g *Generator) Close() error {
	calls.Lock()
	defer calls.Unlock()
	if g == nil || g.state == nil || g.state.handle == nil {
		return nil
	}
	if err := nativeGeneratorClose(g.state.handle); err != nil {
		return err
	}
	g.state.handle = nil
	return nil
}
