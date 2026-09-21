// Package torch exposes optional native tensor operations and reverse-mode
// differentiation. It does not implement a model, loss, optimizer or job runner.
// Build with libtorch and cgo, supplying matching LibTorch headers and libraries.
package torch

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"sync"
	"unsafe"
)

// ErrUnavailable means this build does not contain the optional LibTorch bridge.
var ErrUnavailable = errors.New("torch: native bridge requires the libtorch build tag and cgo")

// ErrClosed means an operation used an uninitialized or closed tensor.
var ErrClosed = errors.New("torch: tensor is closed or uninitialized")

// DType identifies an explicitly represented scalar type.
type DType int

const (
	// Float32 is IEEE binary32.
	Float32 DType = iota
	// Float64 is IEEE binary64.
	Float64
	// Float16 is IEEE binary16.
	Float16
	// BFloat16 is the bfloat16 representation.
	BFloat16
	// Int64 is a signed 64-bit integer.
	Int64
	// Bool uses one byte per boolean.
	Bool
)

// Device identifies CPU or one explicitly indexed CUDA device.
type Device struct {
	Kind  string
	Index int
}

// CPUDevice selects CPU storage and computation.
func CPUDevice() Device { return Device{Kind: "cpu"} }

// CUDADevice selects an indexed CUDA device; constructors validate the index.
func CUDADevice(index int) Device { return Device{Kind: "cuda", Index: index} }

// Info describes a tensor without copying its values to the host.
type Info struct {
	Shape        []int64
	DType        DType
	Device       Device
	RequiresGrad bool
	Elements     int64
}

type tensorState struct{ handle unsafe.Pointer }

// Tensor owns one native tensor handle. Close releases that handle, and is
// idempotent even if the Go value was copied. Views can share native storage;
// closing a parent does not invalidate a view or an autograd graph that owns it.
// Every returned tensor must be closed. There is no finalizer or implicit I/O.
type Tensor struct{ state *tensorState }

// Native calls are serialized so Close cannot race a call using the same handle.
// Native kernels can still dispatch asynchronously; LibTorch owns their tensors.
var calls sync.Mutex

// Enabled reports build availability, not numerical or GPU qualification.
func Enabled() bool { return nativeEnabled }

// HeaderVersion returns the LibTorch version against which this bridge compiled.
// It does not verify the identity of dynamically loaded libraries.
func HeaderVersion() (string, error) { return nativeVersion() }

// Version returns the compiled LibTorch header version, not a library hash.
func Version() (string, error) { return HeaderVersion() }

// FromFloat32 copies values into independent native storage with the given shape.
func FromFloat32(values []float32, shape []int64, device Device, requiresGrad bool) (*Tensor, error) {
	return create(unsafe.Pointer(unsafe.SliceData(values)), int64(len(values))*4, values, shape, Float32, device, requiresGrad)
}

// FromFloat64 copies values into independent native storage with the given shape.
func FromFloat64(values []float64, shape []int64, device Device, requiresGrad bool) (*Tensor, error) {
	return create(unsafe.Pointer(unsafe.SliceData(values)), int64(len(values))*8, values, shape, Float64, device, requiresGrad)
}

// FromBytes copies contiguous little-endian scalar bytes without changing dtype.
// NaNs and infinities are preserved. Callers are responsible for their admission.
func FromBytes(values []byte, shape []int64, dtype DType, device Device, requiresGrad bool) (*Tensor, error) {
	return create(unsafe.Pointer(unsafe.SliceData(values)), int64(len(values)), values, shape, dtype, device, requiresGrad)
}

func create(data unsafe.Pointer, size int64, owner any, shape []int64, dtype DType, device Device, requiresGrad bool) (*Tensor, error) {
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
	if elements > math.MaxInt64/width || elements*width != size {
		return nil, errors.New("torch: shape and dtype do not match input bytes")
	}
	if _, err := deviceIndex(device); err != nil {
		return nil, err
	}
	if requiresGrad && (dtype == Int64 || dtype == Bool) {
		return nil, errors.New("torch: integer and boolean tensors cannot require gradients")
	}
	calls.Lock()
	defer calls.Unlock()
	handle, err := nativeCreate(data, size, owner, shape, dtype, device, requiresGrad)
	return wrap(handle, err)
}

func checkedShape(shape []int64) (int64, error) {
	if len(shape) > 32 {
		return 0, errors.New("torch: rank exceeds 32")
	}
	elements := int64(1)
	for _, dimension := range shape {
		if dimension < 0 || dimension > 0 && elements > math.MaxInt64/dimension {
			return 0, errors.New("torch: invalid or overflowing shape")
		}
		elements *= dimension
	}
	return elements, nil
}

func dtypeWidth(dtype DType) (int64, error) {
	switch dtype {
	case Float32:
		return 4, nil
	case Float64, Int64:
		return 8, nil
	case Float16, BFloat16:
		return 2, nil
	case Bool:
		return 1, nil
	default:
		return 0, errors.New("torch: unsupported dtype")
	}
}

func deviceIndex(device Device) (int, error) {
	if device.Kind == "cpu" && device.Index == 0 {
		return -1, nil
	}
	if device.Kind == "cuda" && device.Index >= 0 && device.Index <= 127 {
		return device.Index, nil
	}
	return 0, errors.New("torch: expected cpu or cuda with an index from 0 to 127")
}

func wrap(handle unsafe.Pointer, err error) (*Tensor, error) {
	if err != nil {
		return nil, err
	}
	if handle == nil {
		return nil, errors.New("torch: native operation returned a null tensor")
	}
	return &Tensor{state: &tensorState{handle: handle}}, nil
}

func handles(tensors []*Tensor) ([]unsafe.Pointer, error) {
	result := make([]unsafe.Pointer, len(tensors))
	for i, tensor := range tensors {
		if tensor == nil || tensor.state == nil || tensor.state.handle == nil {
			return nil, ErrClosed
		}
		result[i] = tensor.state.handle
	}
	return result, nil
}

// Close releases this native handle. It does not free tensors retained by a graph.
func (t *Tensor) Close() error {
	calls.Lock()
	defer calls.Unlock()
	if t == nil || t.state == nil || t.state.handle == nil {
		return nil
	}
	err := nativeClose(t.state.handle)
	if err == nil {
		t.state.handle = nil
	}
	return err
}

// Info returns detached shape and placement metadata without reading values.
func (t *Tensor) Info() (Info, error) {
	calls.Lock()
	defer calls.Unlock()
	h, err := handles([]*Tensor{t})
	if err != nil {
		return Info{}, err
	}
	return nativeInfo(h[0])
}

type operation int

const (
	opMatMul operation = iota
	opAdd
	opMul
	opScale
	opReshape
	opTranspose
	opSlice
	opCat
	opSum
	opMean
	opExp
	opLog
	opSigmoid
	opSiLU
	opSoftplus
	opRSqrt
	opSoftmax
	opTo
	opClone
	opDetach
	opRequiresGrad
	opTril
	opMaskedFill
	opSub
	opUnsqueeze
	opSqueeze
	opStack
	opSelect
	opAllFinite
	opIndexSelect
	opPow
	opReciprocal
	opCos
	opSin
	opAbs
	opAMax
	opClampMin
)

func apply(op operation, tensors []*Tensor, integers []int64, scalar float64) (*Tensor, error) {
	if !nativeEnabled {
		return nil, ErrUnavailable
	}
	calls.Lock()
	defer calls.Unlock()
	h, err := handles(tensors)
	if err != nil {
		return nil, err
	}
	return wrap(nativeApply(op, h, integers, scalar))
}

// MatMul computes a matrix product using LibTorch broadcasting rules.
func (t *Tensor) MatMul(other *Tensor) (*Tensor, error) {
	return apply(opMatMul, []*Tensor{t, other}, nil, 0)
}

// Add adds tensors with LibTorch broadcasting rules.
func (t *Tensor) Add(other *Tensor) (*Tensor, error) {
	return apply(opAdd, []*Tensor{t, other}, nil, 0)
}

// Sub subtracts tensors with LibTorch broadcasting rules.
func (t *Tensor) Sub(other *Tensor) (*Tensor, error) {
	return apply(opSub, []*Tensor{t, other}, nil, 0)
}

// Mul multiplies tensors elementwise with LibTorch broadcasting rules.
func (t *Tensor) Mul(other *Tensor) (*Tensor, error) {
	return apply(opMul, []*Tensor{t, other}, nil, 0)
}

// Scale multiplies every element by a finite scalar.
func (t *Tensor) Scale(value float64) (*Tensor, error) {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return nil, errors.New("torch: scale must be finite")
	}
	return apply(opScale, []*Tensor{t}, nil, value)
}

// Reshape preserves the element count and requires explicit nonnegative dimensions.
func (t *Tensor) Reshape(shape []int64) (*Tensor, error) {
	if _, err := checkedShape(shape); err != nil {
		return nil, err
	}
	return apply(opReshape, []*Tensor{t}, shape, 0)
}

// Transpose exchanges two dimensions and returns a view.
func (t *Tensor) Transpose(first, second int64) (*Tensor, error) {
	return apply(opTranspose, []*Tensor{t}, []int64{first, second}, 0)
}

// Slice returns a view from start to end, excluding end, with a positive step.
func (t *Tensor) Slice(dimension, start, end, step int64) (*Tensor, error) {
	if step <= 0 {
		return nil, errors.New("torch: slice step must be positive")
	}
	return apply(opSlice, []*Tensor{t}, []int64{dimension, start, end, step}, 0)
}

// Cat concatenates tensors along one dimension.
func Cat(tensors []*Tensor, dimension int64) (*Tensor, error) {
	if len(tensors) == 0 {
		return nil, errors.New("torch: concatenation needs tensors")
	}
	return apply(opCat, tensors, []int64{dimension}, 0)
}

// Stack joins equally shaped tensors along a new dimension.
func Stack(tensors []*Tensor, dimension int64) (*Tensor, error) {
	if len(tensors) == 0 {
		return nil, errors.New("torch: stacking needs tensors")
	}
	return apply(opStack, tensors, []int64{dimension}, 0)
}

// Unsqueeze inserts a size-one dimension.
func (t *Tensor) Unsqueeze(dimension int64) (*Tensor, error) {
	return apply(opUnsqueeze, []*Tensor{t}, []int64{dimension}, 0)
}

// Squeeze removes one size-one dimension.
func (t *Tensor) Squeeze(dimension int64) (*Tensor, error) {
	return apply(opSqueeze, []*Tensor{t}, []int64{dimension}, 0)
}

// Select selects one index and removes that dimension from the result.
func (t *Tensor) Select(dimension, index int64) (*Tensor, error) {
	return apply(opSelect, []*Tensor{t}, []int64{dimension, index}, 0)
}

// IndexSelect selects entries along one dimension using an Int64 index tensor.
// Repeated indices retain their multiplicity in the resulting input gradient.
func (t *Tensor) IndexSelect(dimension int64, indices *Tensor) (*Tensor, error) {
	return apply(opIndexSelect, []*Tensor{t, indices}, []int64{dimension}, 0)
}

func reduction(op operation, t *Tensor, dimensions []int64, keepDim bool) (*Tensor, error) {
	if len(dimensions) > 32 {
		return nil, errors.New("torch: too many reduction dimensions")
	}
	ints := make([]int64, 1, len(dimensions)+1)
	if keepDim {
		ints[0] = 1
	}
	ints = append(ints, dimensions...)
	return apply(op, []*Tensor{t}, ints, 0)
}

// Sum sums dimensions; an empty dimensions slice means all dimensions.
func (t *Tensor) Sum(dimensions []int64, keepDim bool) (*Tensor, error) {
	return reduction(opSum, t, dimensions, keepDim)
}

// Mean averages dimensions; an empty dimensions slice means all dimensions.
func (t *Tensor) Mean(dimensions []int64, keepDim bool) (*Tensor, error) {
	return reduction(opMean, t, dimensions, keepDim)
}

// Exp computes the elementwise exponential.
func (t *Tensor) Exp() (*Tensor, error) { return apply(opExp, []*Tensor{t}, nil, 0) }

// Log computes the elementwise natural logarithm.
func (t *Tensor) Log() (*Tensor, error) { return apply(opLog, []*Tensor{t}, nil, 0) }

// Sigmoid computes the logistic sigmoid.
func (t *Tensor) Sigmoid() (*Tensor, error) { return apply(opSigmoid, []*Tensor{t}, nil, 0) }

// Pow raises each element to a tensor exponent using native broadcasting.
func (t *Tensor) Pow(exponent *Tensor) (*Tensor, error) {
	return apply(opPow, []*Tensor{t, exponent}, nil, 0)
}

// Reciprocal computes the elementwise multiplicative inverse.
func (t *Tensor) Reciprocal() (*Tensor, error) { return apply(opReciprocal, []*Tensor{t}, nil, 0) }

// Cos computes elementwise cosine in radians.
func (t *Tensor) Cos() (*Tensor, error) { return apply(opCos, []*Tensor{t}, nil, 0) }

// Sin computes elementwise sine in radians.
func (t *Tensor) Sin() (*Tensor, error) { return apply(opSin, []*Tensor{t}, nil, 0) }

// Abs computes the elementwise absolute value.
func (t *Tensor) Abs() (*Tensor, error) { return apply(opAbs, []*Tensor{t}, nil, 0) }

// AMax returns the maximum values across dimensions; an empty dimensions slice
// reduces every dimension, matching Sum and Mean.
func (t *Tensor) AMax(dimensions []int64, keepDim bool) (*Tensor, error) {
	return reduction(opAMax, t, dimensions, keepDim)
}

// ClampMin replaces values below a finite scalar floor.
func (t *Tensor) ClampMin(value float64) (*Tensor, error) {
	if math.IsNaN(value) || math.IsInf(value, 0) {
		return nil, errors.New("torch: clamp minimum must be finite")
	}
	return apply(opClampMin, []*Tensor{t}, nil, value)
}

// SiLU computes x times sigmoid(x).
func (t *Tensor) SiLU() (*Tensor, error) { return apply(opSiLU, []*Tensor{t}, nil, 0) }

// Softplus computes softplus with beta=1 and threshold=20.
func (t *Tensor) Softplus() (*Tensor, error) { return apply(opSoftplus, []*Tensor{t}, nil, 0) }

// RSqrt computes the elementwise reciprocal square root.
func (t *Tensor) RSqrt() (*Tensor, error) { return apply(opRSqrt, []*Tensor{t}, nil, 0) }

// Softmax normalizes exponentials along one dimension in the input dtype.
func (t *Tensor) Softmax(dimension int64) (*Tensor, error) {
	return apply(opSoftmax, []*Tensor{t}, []int64{dimension}, 0)
}

// To copies to the requested device and dtype while preserving autograd history.
func (t *Tensor) To(device Device, dtype DType) (*Tensor, error) {
	index, err := deviceIndex(device)
	if err != nil {
		return nil, err
	}
	if _, err := dtypeWidth(dtype); err != nil {
		return nil, err
	}
	return apply(opTo, []*Tensor{t}, []int64{int64(index), int64(dtype)}, 0)
}

// Clone copies storage and preserves autograd history.
func (t *Tensor) Clone() (*Tensor, error) { return apply(opClone, []*Tensor{t}, nil, 0) }

// Detach shares storage but removes the result from the autograd graph.
func (t *Tensor) Detach() (*Tensor, error) { return apply(opDetach, []*Tensor{t}, nil, 0) }

// SetRequiresGrad changes a leaf tensor's gradient flag and returns a new owned
// handle to that same tensor. Native autograd rejects invalid non-leaf changes.
func (t *Tensor) SetRequiresGrad(enabled bool) (*Tensor, error) {
	flag := int64(0)
	if enabled {
		flag = 1
	}
	return apply(opRequiresGrad, []*Tensor{t}, []int64{flag}, 0)
}

// Tril retains the lower triangle, offset by diagonal, and zeros the upper part.
func (t *Tensor) Tril(diagonal int64) (*Tensor, error) {
	return apply(opTril, []*Tensor{t}, []int64{diagonal}, 0)
}

// MaskedFill replaces true mask entries with value. Negative infinity is allowed
// for attention masks; NaN is rejected.
func (t *Tensor) MaskedFill(mask *Tensor, value float64) (*Tensor, error) {
	if math.IsNaN(value) {
		return nil, errors.New("torch: mask value cannot be NaN")
	}
	return apply(opMaskedFill, []*Tensor{t, mask}, nil, value)
}

// Grad returns the vector-Jacobian product for each input without accumulating
// into .grad fields. gradOutputs must explicitly match outputs, including scalar
// outputs. Unused or nondifferentiable inputs are errors. Each result is owned.
// Unless retainGraph or createGraph is true, backward frees saved graph buffers.
func Grad(outputs, inputs, gradOutputs []*Tensor, retainGraph, createGraph bool) ([]*Tensor, error) {
	if !nativeEnabled {
		return nil, ErrUnavailable
	}
	if len(outputs) == 0 || len(inputs) == 0 || len(outputs) != len(gradOutputs) {
		return nil, errors.New("torch: grad needs inputs and matching outputs and cotangents")
	}
	calls.Lock()
	defer calls.Unlock()
	o, err := handles(outputs)
	if err != nil {
		return nil, err
	}
	i, err := handles(inputs)
	if err != nil {
		return nil, err
	}
	g, err := handles(gradOutputs)
	if err != nil {
		return nil, err
	}
	native, err := nativeGrad(o, i, g, retainGraph || createGraph, createGraph)
	if err != nil {
		return nil, err
	}
	result := make([]*Tensor, len(native))
	for index, handle := range native {
		result[index] = &Tensor{state: &tensorState{handle: handle}}
	}
	return result, nil
}

func (t *Tensor) copyValues(dtype DType, raw bool) ([]byte, Info, error) {
	calls.Lock()
	defer calls.Unlock()
	h, err := handles([]*Tensor{t})
	if err != nil {
		return nil, Info{}, err
	}
	info, err := nativeInfo(h[0])
	if err != nil {
		return nil, Info{}, err
	}
	if raw {
		dtype = info.DType
	}
	width, err := dtypeWidth(dtype)
	if err != nil {
		return nil, Info{}, err
	}
	if info.Elements < 0 || info.Elements > int64(int(^uint(0)>>1))/width {
		return nil, Info{}, fmt.Errorf("torch: tensor values exceed host address space")
	}
	data := make([]byte, int(info.Elements*width))
	if err := nativeCopy(h[0], dtype, data); err != nil {
		return nil, Info{}, err
	}
	return data, info, nil
}

// Bytes copies contiguous little-endian values to Go without changing dtype.
// It synchronizes device copies and allocates the full tensor size on the host.
func (t *Tensor) Bytes() ([]byte, error) {
	data, _, err := t.copyValues(0, true)
	return data, err
}

// Float32Values copies and converts the complete tensor to host float32 values.
func (t *Tensor) Float32Values() ([]float32, error) {
	data, _, err := t.copyValues(Float32, false)
	if err != nil {
		return nil, err
	}
	values := make([]float32, len(data)/4)
	for i := range values {
		values[i] = math.Float32frombits(binary.LittleEndian.Uint32(data[i*4:]))
	}
	return values, nil
}

// Float64Values copies and converts the complete tensor to host float64 values.
func (t *Tensor) Float64Values() ([]float64, error) {
	data, _, err := t.copyValues(Float64, false)
	if err != nil {
		return nil, err
	}
	values := make([]float64, len(data)/8)
	for i := range values {
		values[i] = math.Float64frombits(binary.LittleEndian.Uint64(data[i*8:]))
	}
	return values, nil
}

// AllFinite checks every value natively and copies only the boolean result.
func (t *Tensor) AllFinite() (bool, error) {
	result, err := apply(opAllFinite, []*Tensor{t}, nil, 0)
	if err != nil {
		return false, err
	}
	defer result.Close()
	data, err := result.Bytes()
	if err != nil {
		return false, err
	}
	return len(data) == 1 && data[0] != 0, nil
}
