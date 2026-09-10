package llama

/*
#include "wrapper.h"
*/
import "C"

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"io"
	"math"
	"runtime"
	"unsafe"
)

// Capturing the final representation of a token sequence.
//
// The objective this exists for trains a quantised model plus an adapter to
// match, token by token, the representation a high-precision reference
// produces for the same tokens: the hidden state after the final normalisation
// and before the output projection. A loss can say the two models disagree; a
// representation says where, at which position, in which direction. The
// quantity compared is therefore a matrix, one row per requested position, and
// the two sides of the comparison have to be able to prove they are looking at
// the same rows of the same tensor produced by the same model -- which is what
// the identity fields on Capture are for.

// CaptureVersion names the procedure and the engine revision that produced a
// capture: which tensor, that every window row is an output row, the digest
// layouts, and the pinned llama.cpp commit. Either moving changes the numbers,
// so a cache keyed without it would hand a trainer a target from a different
// model. tests/Unit checks the suffix against `git ls-tree HEAD llama.cpp`.
const CaptureVersion = "arandu-llama.capture-final.v1/llama.cpp.90c26fcd"

// CaptureTensor is the graph tensor the rows come from at the pinned commit:
// the output of the final RMS norm, before the output projection. At that
// commit it is `res->t_embd`, which llama.cpp copies out when embeddings are
// on and nothing is pooled.
const CaptureTensor = "result_norm"

// CaptureDType is the element type of Capture.Data.
const CaptureDType = "f32"

// Capture is the per-token final representation of one token sequence at the
// requested positions: one float32 row of Width per position, the hidden state
// after the final normalisation and before the output projection. It is per
// token by construction; a pooled sequence embedding does not satisfy the
// objective this exists for, because it cannot say at which position the two
// models disagree.
//
// Cost: len(Positions) x Width x 4 bytes -- 16,777,216 bytes (16 MiB) for
// SmolLM3-3B (Width 2048) at 2048 positions -- plus, transiently inside
// llama.cpp per decode window, window x (n_vocab + Width) x 4 bytes
// (136,037,376 bytes at the 261-row window the 128 MiB logit cap gives
// SmolLM3), because logits are reserved per output row in embeddings mode too.
type Capture struct {
	Data           []float32 // row-major, len == len(Positions)*Width; Row(k) is Positions[k]
	Positions      []int32   // copy of the request, strictly increasing
	Width          int       // n_embd_out of the model; floats per row
	Tensor         string    // CaptureTensor
	DType          string    // CaptureDType
	Quantisation   string    // Model.Describe at capture time, never named by the caller
	Policy         string    // adapter digests in applied order, or "base" -- the label a reading carries
	Scales         []float32 // scale per applied adapter, same order; nil for "base"
	SnapshotDigest string    // SnapshotDigest(Quantisation, Policy, Scales)
	TokenDigest    string    // TokenDigest(tokens, Positions)
	Version        string    // CaptureVersion
}

// Shape returns (n_embd, n_positions) = (Width, len(Positions)).
func (c *Capture) Shape() (nEmbd, nPositions int) {
	return c.Width, len(c.Positions)
}

// Row returns the representation at Positions[k] as a sub-slice of Data; it
// panics outside [0, len(Positions)) like any slice would.
func (c *Capture) Row(k int) []float32 {
	if k < 0 || k >= len(c.Positions) {
		panic(fmt.Sprintf("llama: capture row %d outside [0, %d)", k, len(c.Positions)))
	}
	return c.Data[k*c.Width : (k+1)*c.Width]
}

// PositionsOf turns a mask (true = capture this position) into the strictly
// increasing position list CaptureFinal takes. This is how a mask over the
// sequence is expressed, and it is therefore inside TokenDigest.
func PositionsOf(mask []bool) []int32 {
	positions := make([]int32, 0, len(mask))
	for i, keep := range mask {
		if keep {
			positions = append(positions, int32(i))
		}
	}
	return positions
}

// TokenDigest is the cache key of what was captured: sha256 hex over the ASCII
// tag CaptureVersion + "\n", uint32 LE len(tokens), each token int32 LE,
// uint32 LE len(positions), each position int32 LE. Length-prefixed, so two
// requests that differ in one token or one position never share it, and two
// requests whose concatenations happen to coincide do not either.
func TokenDigest(tokens, positions []int32) string {
	h := sha256.New()
	h.Write([]byte(CaptureVersion + "\n"))
	writeInt32s(h, tokens)
	writeInt32s(h, positions)
	return hex.EncodeToString(h.Sum(nil))
}

// SnapshotDigest is the cache key of what produced a capture: sha256 hex over
// the same tag, the quantisation and the policy label (each length-prefixed),
// uint32 LE len(scales) and each scale as math.Float32bits LE. +1 and -1 on the
// same adapter differ; base and any adapter differ; a version bump moves every
// key. Two captures of different policies can therefore never be confused, and
// that is the property a reference cache is keyed on.
func SnapshotDigest(quantisation, policy string, scales []float32) string {
	h := sha256.New()
	h.Write([]byte(CaptureVersion + "\n"))
	writeString(h, quantisation)
	writeString(h, policy)
	var buffer [4]byte
	binary.LittleEndian.PutUint32(buffer[:], uint32(len(scales)))
	h.Write(buffer[:])
	for _, scale := range scales {
		binary.LittleEndian.PutUint32(buffer[:], math.Float32bits(scale))
		h.Write(buffer[:])
	}
	return hex.EncodeToString(h.Sum(nil))
}

func writeInt32s(h io.Writer, values []int32) {
	var buffer [4]byte
	binary.LittleEndian.PutUint32(buffer[:], uint32(len(values)))
	h.Write(buffer[:])
	for _, value := range values {
		binary.LittleEndian.PutUint32(buffer[:], uint32(value))
		h.Write(buffer[:])
	}
}

func writeString(h io.Writer, value string) {
	var buffer [4]byte
	binary.LittleEndian.PutUint32(buffer[:], uint32(len(value)))
	h.Write(buffer[:])
	h.Write([]byte(value))
}

// CaptureFinal decodes tokens from an empty KV cache under the adapters
// currently applied and returns the final representation at positions: the
// hidden state after the final normalisation and before the output projection,
// which is the tensor named CaptureTensor at the pinned llama.cpp commit.
//
// The rows are read through llama.cpp's own copy of that tensor rather than a
// scheduler callback: the embeddings flag is switched on for the decode and
// restored afterwards, and the rows it hands back were measured bitwise equal
// to the tensor a callback observes. Every row of every decode window is an
// output row -- embeddings mode overrides a partial selection anyway, and a
// subset decode differs from the full rows by up to 1.9e-6, so which rows are
// computed is part of CaptureVersion. The positions choose which rows are
// copied out.
//
// The KV cache and the prefix bookkeeping are cleared first, as Score does. A
// cache filled under another adapter would capture a mixture of two models,
// and the whole point of the capture is to name which model it came from. The
// consequence for the caller is the same as for Score: an interleaved Generate
// on this context re-decodes its prompt.
//
// Refused before crossing into C: closed context or model; no tokens; no
// positions; a token id below zero; a position outside [0, len(tokens));
// positions not strictly increasing; an applied adapter whose digest is unknown
// (a capture that cannot name its policy cannot be matched against a
// reference). Refused in C: a token outside the vocabulary, a sequence longer
// than the context, a context whose pooling type is not NONE, a non-finite
// value anywhere in a captured row.
//
// Cost: len(positions) x Width x 4 bytes -- 16,777,216 bytes for SmolLM3-3B
// (Width 2048) at 2048 positions -- plus the transient per-window buffer
// described on Capture.
//
// Thread safety: holds the context's write lock for the whole call, like
// Score, so the policy the result is labelled with is the one the rows were
// computed under. Tokenise before calling; the mutex is not reentrant.
func (c *Context) CaptureFinal(tokens, positions []int32) (*Capture, error) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return nil, fmt.Errorf("context is closed")
	}
	if c.model == nil {
		return nil, fmt.Errorf("model is closed")
	}
	c.model.mu.RLock()
	modelClosed := c.model.closed
	c.model.mu.RUnlock()
	if modelClosed {
		return nil, fmt.Errorf("model is closed")
	}

	if len(tokens) == 0 {
		return nil, fmt.Errorf("at least one token required to capture, got 0")
	}
	if len(positions) == 0 {
		return nil, fmt.Errorf("at least one position required to capture, got 0")
	}
	for i, token := range tokens {
		if token < 0 {
			return nil, fmt.Errorf("invalid token %d at index %d", token, i)
		}
	}
	for k, position := range positions {
		if position < 0 || int(position) >= len(tokens) {
			return nil, fmt.Errorf("invalid position %d at index %d for %d tokens", position, k, len(tokens))
		}
		if k > 0 && position <= positions[k-1] {
			return nil, fmt.Errorf("positions must be strictly increasing: index %d is %d after %d", k, position, positions[k-1])
		}
	}

	// Labelled under the lock the rows are computed under. The policy is
	// refused when an adapter has no digest, for the reason readings are: a
	// capture labelled "unknown" matches every other capture labelled "unknown".
	policy, err := c.appliedPolicyLocked()
	if err != nil {
		return nil, err
	}
	// Describe takes the model's own lock, not the context's.
	quantisation, err := c.model.Describe()
	if err != nil {
		return nil, err
	}
	width := int(C.llama_wrapper_model_n_embd_out(c.model.modelPtr))
	if width <= 0 {
		return nil, fmt.Errorf("capture failed: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	cTokens := make([]C.int, len(tokens))
	for i, token := range tokens {
		cTokens[i] = C.int(token)
	}
	cPositions := make([]C.int, len(positions))
	for k, position := range positions {
		cPositions[k] = C.int(position)
	}
	out := make([]float32, len(positions)*width)

	result := C.llama_wrapper_capture_final(
		c.contextPtr,
		&cTokens[0],
		C.int(len(tokens)),
		&cPositions[0],
		C.int(len(positions)),
		C.bool(c.config.embeddings),
		(*C.float)(unsafe.Pointer(&out[0])),
		C.longlong(len(out)),
	)
	// The C side reads the token and position arrays and writes the output for
	// the length of the call and retains none of them.
	runtime.KeepAlive(cTokens)
	runtime.KeepAlive(cPositions)
	runtime.KeepAlive(out)

	if result != 0 {
		return nil, fmt.Errorf("capture failed: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	captured := append([]int32(nil), positions...)
	var scales []float32
	if len(c.appliedScales) > 0 {
		scales = append([]float32(nil), c.appliedScales...)
	}
	return &Capture{
		Data:           out,
		Positions:      captured,
		Width:          width,
		Tensor:         CaptureTensor,
		DType:          CaptureDType,
		Quantisation:   quantisation,
		Policy:         policy,
		Scales:         scales,
		SnapshotDigest: SnapshotDigest(quantisation, policy, scales),
		TokenDigest:    TokenDigest(tokens, captured),
		Version:        CaptureVersion,
	}, nil
}
