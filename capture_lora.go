package llama

/*
#include <stdlib.h>
#include "wrapper.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

// Observing the LoRA arithmetic of one projection.
//
// Every control that leaves B at zero measures a contribution of zero, and an
// engine applying a hundred times the right value passes all of them. So does a
// set of adapters compared against one another: a uniform factor survives every
// equality between two routes to the same contribution, because it multiplies
// both sides. Settling it needs the numbers the graph actually produced,
// compared against arithmetic done somewhere else.
//
// What comes back is the whole chain for one projection and one sequence, in
// the order llama.cpp computes it, so a caller can say which operation is the
// first to disagree rather than only that the answer is wrong.

// LoRACaptureVersion names the procedure and the engine revision. The nodes are
// found by operation and by the name of the weight operand, and the pinned
// commit decides what that graph looks like; either moving changes what is
// observed. The suffix is the gitlink, as CaptureVersion's is.
const LoRACaptureVersion = "arandu-llama.capture-lora.v1/llama.cpp.90c26fcd"

// LoRACapture is one projection's arithmetic, row-major with one row per token.
//
// Ratio is U over UPre, element by element, wherever UPre is not zero: the
// scale llama.cpp applied, read off the graph rather than inferred from a
// comparison of two adapters. It is the field that closes a uniform factor,
// because it does not cancel.
type LoRACapture struct {
	// X is the projection's input, NIn floats per token.
	X []float32
	// H is A*X, Rank floats per token. Empty when no adapter was applied.
	H []float32
	// UPre is B*H before the scale, NOut floats per token. Empty without an adapter.
	UPre []float32
	// U is the scaled contribution, NOut floats per token. Empty without an adapter.
	U []float32
	// YBase is W*X, the projection without the adapter, NOut floats per token.
	YBase []float32
	// Y is YBase + U, NOut floats per token. Empty without an adapter.
	Y []float32

	Module  string
	NTokens int
	NIn     int
	Rank    int
	NOut    int
	// Scale is the applied scale as the caller asked for it: adapter_scale
	// times alpha over rank. What the engine did with it is in U and UPre.
	Scale   float32
	Version string
}

// LoRARow returns the k-th token's row of one of LoRACapture's fields, which
// are flat and row-major. Capture.Row is the same idea for the other capture,
// but that one knows its own width and this one cannot: the six fields have
// three different widths.
func LoRARow(data []float32, k, width int) []float32 {
	if width <= 0 || k < 0 || (k+1)*width > len(data) {
		return nil
	}
	return data[k*width : (k+1)*width]
}

// CaptureLoRA runs one decode on a context of its own and reads back the six
// tensors of a projection's LoRA arithmetic.
//
// The context is created and destroyed inside the call. It has to be: cb_eval
// is reachable only through llama_context_params at creation and llama.h offers
// no setter afterwards. That is also what makes the cost acceptable -- the
// per-split synchronisation the callback forces lasts one decode, not the life
// of a context that serves inference. The production scoring path is untouched.
//
// adapter may be nil, and then only X and YBase come back. nIn, rank and nOut
// are the shapes the caller read from the adapter and the model; they size the
// buffers, and the C side refuses a length that is not exactly right.
func (m *Model) CaptureLoRA(
	module string, adapter *Adapter, scale float32,
	tokens []int32, nCtx, nIn, rank, nOut int,
) (*LoRACapture, error) {
	m.mu.RLock()
	closed := m.closed
	m.mu.RUnlock()
	if closed {
		return nil, fmt.Errorf("model is closed")
	}
	if module == "" {
		return nil, fmt.Errorf("a module name is required")
	}
	if len(tokens) == 0 {
		return nil, fmt.Errorf("at least one token required to capture, got 0")
	}
	if nCtx < len(tokens) {
		return nil, fmt.Errorf("context of %d is shorter than the %d tokens to capture", nCtx, len(tokens))
	}
	if nIn <= 0 || nOut <= 0 {
		return nil, fmt.Errorf("the projection's shapes must be positive, got n_in=%d n_out=%d", nIn, nOut)
	}
	if adapter != nil && rank <= 0 {
		return nil, fmt.Errorf("an adapter needs a positive rank, got %d", rank)
	}
	for i, token := range tokens {
		if token < 0 {
			return nil, fmt.Errorf("invalid token %d at index %d", token, i)
		}
	}

	n := len(tokens)
	capture := &LoRACapture{
		Module: module, NTokens: n, NIn: nIn, NOut: nOut,
		Scale: scale, Version: LoRACaptureVersion,
	}
	capture.X = make([]float32, n*nIn)
	capture.YBase = make([]float32, n*nOut)

	// Every buffer is pinned before its address goes into the struct.
	//
	// The struct itself is Go memory, and cgo forbids Go memory passed to C
	// from containing Go pointers: the check runs on the way in and aborts with
	// "argument of cgo function has Go pointer to unpinned Go pointer". It is
	// not intermittent and does not wait for a collection -- x and y_base are
	// always filled, and two pointers are enough to trip it. Pinning is what
	// makes the addresses legal for the duration of the call.
	var pinner runtime.Pinner
	defer pinner.Unpin()

	var request C.llama_wrapper_lora_capture
	pinner.Pin(&capture.X[0])
	request.x = (*C.float)(unsafe.Pointer(&capture.X[0]))
	request.x_floats = C.longlong(len(capture.X))
	pinner.Pin(&capture.YBase[0])
	request.y_base = (*C.float)(unsafe.Pointer(&capture.YBase[0]))
	request.y_base_floats = C.longlong(len(capture.YBase))

	var adapterPtr unsafe.Pointer
	if adapter != nil {
		adapter.mu.RLock()
		adapterClosed := adapter.closed
		adapterPtr = adapter.adapterPtr
		adapter.mu.RUnlock()
		if adapterClosed || adapterPtr == nil {
			return nil, fmt.Errorf("adapter is closed")
		}
		capture.Rank = rank
		capture.H = make([]float32, n*rank)
		capture.UPre = make([]float32, n*nOut)
		capture.U = make([]float32, n*nOut)
		capture.Y = make([]float32, n*nOut)
		pinner.Pin(&capture.H[0])
		request.h = (*C.float)(unsafe.Pointer(&capture.H[0]))
		request.h_floats = C.longlong(len(capture.H))
		pinner.Pin(&capture.UPre[0])
		request.u_pre = (*C.float)(unsafe.Pointer(&capture.UPre[0]))
		request.u_pre_floats = C.longlong(len(capture.UPre))
		pinner.Pin(&capture.U[0])
		request.u = (*C.float)(unsafe.Pointer(&capture.U[0]))
		request.u_floats = C.longlong(len(capture.U))
		pinner.Pin(&capture.Y[0])
		request.y = (*C.float)(unsafe.Pointer(&capture.Y[0]))
		request.y_floats = C.longlong(len(capture.Y))
	}

	cModule := C.CString(module)
	defer C.free(unsafe.Pointer(cModule))
	cTokens := make([]C.int, n)
	for i, token := range tokens {
		cTokens[i] = C.int(token)
	}

	result := C.llama_wrapper_capture_lora(
		m.modelPtr, cModule, adapterPtr, C.float(scale),
		&cTokens[0], C.int(n), C.int(nCtx), &request,
	)
	// The C side writes into the slices above for the length of the call, so
	// nothing they belong to may be collected before it returns.
	runtime.KeepAlive(capture)
	runtime.KeepAlive(cTokens)
	runtime.KeepAlive(m)
	if adapter != nil {
		runtime.KeepAlive(adapter)
	}
	if result != 0 {
		return nil, fmt.Errorf("capturing the LoRA arithmetic: %s", C.GoString(C.llama_wrapper_last_error()))
	}
	return capture, nil
}

// AppliedScale is U divided by UPre wherever UPre is not zero, which is the
// scale the engine applied.
//
// It reports the first element it could divide, how many it could, and whether
// every one of them agreed to within tolerance. A uniform factor on the
// contribution shows up here and nowhere else: every equality between two
// adapters cancels it, and this does not.
func (c *LoRACapture) AppliedScale(tolerance float64) (scale float64, agreed int, uniform bool) {
	if len(c.U) == 0 || len(c.U) != len(c.UPre) {
		return 0, 0, false
	}
	first := 0.0
	uniform = true
	for i := range c.U {
		if c.UPre[i] == 0 {
			continue
		}
		ratio := float64(c.U[i]) / float64(c.UPre[i])
		if agreed == 0 {
			first = ratio
		} else if diff := ratio - first; diff > tolerance || diff < -tolerance {
			uniform = false
		}
		agreed++
	}
	return first, agreed, uniform && agreed > 0
}
