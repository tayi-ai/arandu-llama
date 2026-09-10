package llama

/*
#include "wrapper.h"
#include <stdlib.h>
*/
import "C"

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"os"
	"runtime"
	"sync"
	"unsafe"
)

// Adapter represents a LoRA adapter loaded against a specific model.
//
// An adapter carries the low-rank tensors that displace the base weights during
// a forward pass, leaving the weights themselves untouched. It is loaded once
// from the model whose architecture and tensor shapes it matches, and is then
// applied to, or removed from, any context created from that same model.
//
// This is what zeroth-order optimisation needs and ordinary inference does not:
// a perturbation that moves between one forward pass and the next whilst the
// quantised base stays resident on the cards, because reloading the base for
// every step would cost more than the step.
//
// Thread safety: Adapter is safe for concurrent use. The context it is applied
// to is not - every decode on that context reads the adapter's tensors, so
// removing or freeing an adapter whilst work is in flight on such a context
// corrupts whatever that work returns.
//
// Unlike Model and Context, Adapter carries no finaliser. A context holds the
// raw handle rather than a Go reference this package can see, so a finaliser
// would be free to run whilst a context still points at the tensors. An adapter
// that is never closed leaks memory; an adapter freed whilst applied is read
// after the free, and that is the worse of the two:
//
//	adapter, _ := model.LoadAdapter("rank4.gguf")
//	defer adapter.Close()
type Adapter struct {
	adapterPtr unsafe.Pointer // llama_adapter_lora*
	model      *Model         // parent, checked before the handle reaches a context
	// path and digest are what a measurement taken under this adapter is labelled
	// with. A reading that cannot say which adapter produced it cannot be compared
	// with any other, and comparison is the only thing a reading is for. The
	// digest is read once, at load, because the file is the only thing that
	// identifies an adapter unambiguously -- two exports of the same training run
	// have different bytes only if they are different training runs.
	path   string
	digest string
	mu     sync.RWMutex
	closed bool
}

// Path is where this adapter was loaded from.
func (a *Adapter) Path() string { return a.path }

// Digest is the sha256 of the file this adapter was loaded from.
func (a *Adapter) Digest() string { return a.digest }

// LoadAdapter loads a LoRA adapter from a GGUF file against this model.
//
// The adapter is read against the model it will be applied to, so a mismatch in
// architecture or tensor shape is refused now rather than surfacing later as a
// plausible-looking loss. Loading does not modify the model or any existing
// context: an adapter takes effect only where Context.SetAdapters applies it,
// which is what lets one adapter serve every context created from the model.
//
// The returned adapter must not outlive the model it was loaded against, and
// must be released with Close.
//
// Thread safety: Model is thread-safe, so adapters may be loaded concurrently.
//
// See also: Context.SetAdapters to apply one, Context.ClearAdapters to remove
// every adapter from a context.
//
// Example:
//
//	adapter, err := model.LoadAdapter("rank4.gguf")
//	if err != nil {
//	    return err
//	}
//	defer adapter.Close()
func (m *Model) LoadAdapter(path string) (*Adapter, error) {
	if path == "" {
		return nil, fmt.Errorf("adapter path cannot be empty")
	}

	m.mu.RLock()
	if m.closed {
		m.mu.RUnlock()
		return nil, fmt.Errorf("model is closed")
	}
	modelPtr := m.modelPtr
	m.mu.RUnlock()

	cPath := C.CString(path)
	defer C.free(unsafe.Pointer(cPath))

	adapterPtr := C.llama_wrapper_adapter_load(modelPtr, cPath)
	if adapterPtr == nil {
		return nil, fmt.Errorf("failed to load adapter: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	// The digest is read after the adapter loaded, so a file that llama.cpp
	// refused is never hashed, and a hash failure does not lose an adapter that
	// is otherwise usable -- it is labelled unknown and said so. What refuses is
	// the reading: Context.appliedPolicy will not label a measurement with a
	// digest that every unhashable adapter shares.
	digest := unknownDigest
	if sum, err := digestOfFile(path); err == nil {
		digest = sum
	}
	return &Adapter{
		adapterPtr: adapterPtr,
		model:      m,
		path:       path,
		digest:     digest,
	}, nil
}

// Close frees the adapter and its tensors.
//
// This method is idempotent - multiple calls are safe and subsequent calls
// return immediately without error.
//
// Remove the adapter from every context that has it applied, with
// Context.SetAdapters or Context.ClearAdapters, before closing it. A context
// keeps the raw handle, so the next decode after the free reads released memory
// and reports a number rather than an error.
//
// After Close, the adapter can no longer be applied to a context.
func (a *Adapter) Close() error {
	a.mu.Lock()
	defer a.mu.Unlock()

	if a.closed {
		return nil
	}

	// llama_model_free destroys every adapter loaded against the model, so a
	// model closed first has already freed this handle: freeing it again is a
	// double free, and a double free corrupts the allocator for the process.
	// Closing out of order is the caller error the type documentation warns
	// about, but it is not one worth answering with a crash in the middle of a
	// measurement
	modelClosed := false
	if a.model != nil {
		a.model.mu.RLock()
		modelClosed = a.model.closed
		a.model.mu.RUnlock()
	}

	if a.adapterPtr != nil && !modelClosed {
		C.llama_wrapper_adapter_free(a.adapterPtr)
	}
	a.adapterPtr = nil

	a.closed = true
	return nil
}

// SetAdapters applies a set of adapters to this context, one scale each.
//
// The call replaces the context's entire adapter set: anything applied before
// and absent from adapters is removed. Each scale multiplies the contribution of
// the adapter at the same index, so a zeroth-order probe moves the policy by
// re-applying the same handle with a different scale rather than by reloading
// tensors between passes.
//
// Every adapter must have been loaded from the model this context was created
// from. Applying an adapter whose tensor shapes belong to a different model is
// undefined behaviour inside llama.cpp rather than an error it reports, so the
// parent model is checked here instead.
//
// Changing the set invalidates the KV cache. The cached keys and values were
// computed under the previous adapter, and decoding on top of them scores a
// mixture of two models. Context.Score clears the cache before it decodes for
// exactly this reason; Generate does not, because prefix reuse is what makes it
// fast, so do not generate on a context whose adapters moved since the cache was
// filled - score on a context reserved for scoring, or create a fresh one.
//
// Thread safety: Context is NOT thread-safe. This call is serialised against
// generation on the same context, but a caller that swaps adapters from one
// goroutine whilst another decodes gets whichever set arrived first.
//
// Examples:
//
//	// One adapter at full strength
//	err := ctx.SetAdapters([]*llama.Adapter{adapter}, []float32{1.0})
//
//	// The negative half of a zeroth-order pair
//	err := ctx.SetAdapters([]*llama.Adapter{adapter}, []float32{-1.0})
func (c *Context) SetAdapters(adapters []*Adapter, scales []float32) error {
	c.mu.Lock()
	defer c.mu.Unlock()

	if c.closed {
		return fmt.Errorf("context is closed")
	}

	// The C side indexes both arrays to n_adapters and cannot tell that one is
	// shorter, so a mismatch here reads past the end of it
	if len(adapters) != len(scales) {
		return fmt.Errorf("adapter count mismatch: %d adapters, %d scales", len(adapters), len(scales))
	}

	// Check if model is closed
	if c.model == nil {
		return fmt.Errorf("model is closed")
	}
	c.model.mu.RLock()
	modelClosed := c.model.closed
	c.model.mu.RUnlock()
	if modelClosed {
		return fmt.Errorf("model is closed")
	}

	adapterPtrs := make([]unsafe.Pointer, len(adapters))
	for i, adapter := range adapters {
		if adapter == nil {
			return fmt.Errorf("adapter %d is nil", i)
		}

		adapter.mu.RLock()
		if adapter.closed {
			adapter.mu.RUnlock()
			return fmt.Errorf("adapter %d is closed", i)
		}
		// A handle from another model passes every check llama.cpp makes and
		// produces a loss that looks like a measurement
		if adapter.model != c.model {
			adapter.mu.RUnlock()
			return fmt.Errorf("adapter %d belongs to a different model", i)
		}
		adapterPtrs[i] = adapter.adapterPtr
		adapter.mu.RUnlock()
	}

	cScales := make([]C.float, len(scales))
	for i, scale := range scales {
		cScales[i] = C.float(scale)
	}

	// Zero adapters is the documented way to remove every adapter, and taking
	// the address of the first element of an empty slice panics
	var cAdapters *unsafe.Pointer
	var cScalesPtr *C.float
	if len(adapters) > 0 {
		cAdapters = &adapterPtrs[0]
		cScalesPtr = &cScales[0]
	}

	applied := C.llama_wrapper_adapters_set(c.contextPtr, cAdapters, cScalesPtr, C.int(len(adapters)))
	// Remembered, so a reading taken afterwards can say what produced it. Without
	// this the base model and an unrecorded adapter are indistinguishable in a
	// table, and that table is what a comparison is read from.
	if applied >= 0 {
		c.applied = append(c.applied[:0], adapters...)
	}
	// The C side copies both arrays and retains neither, but it reads them for
	// the length of the call, and a collected scale array applies noise
	runtime.KeepAlive(adapterPtrs)
	runtime.KeepAlive(cScales)

	if applied < 0 {
		return fmt.Errorf("failed to set adapters: %s", C.GoString(C.llama_wrapper_last_error()))
	}

	return nil
}

// ClearAdapters removes every adapter from this context.
//
// The context returns to the base weights. Call this after a zeroth-order probe:
// an engine left perturbed is a model that the next measurement reads without
// anybody intending it, and the difference is small enough to pass for noise.
//
// Clearing invalidates the KV cache for the same reason applying does - the
// cached keys and values were computed under the adapter that is now gone. See
// SetAdapters.
//
// Example:
//
//	defer ctx.ClearAdapters()
func (c *Context) ClearAdapters() error {
	return c.SetAdapters(nil, nil)
}

// Meta reads one GGUF metadata value from the adapter.
//
// The key that matters is "adapter.lora.alpha". llama.cpp multiplies by
// adapter_scale * alpha / rank when the file declares a non-zero alpha, and by
// adapter_scale alone when it does not, whilst PEFT applied alpha / r. So an
// adapter that lost the key during conversion is applied at a different
// magnitude than the trainer applied it, every weight is displaced by one
// constant factor, and the loss that comes out belongs to a different model.
//
// That failure reports nothing. It produces a plausible number, and a plausible
// number is what gets compared against a measured reference and believed, which
// is why this is worth a call at load rather than a comment.
//
// An absent key returns an error rather than an empty string: a caller that
// cannot distinguish "the key says nothing" from "there is no key" cannot make
// the check that matters.
func (a *Adapter) Meta(key string) (string, error) {
	a.mu.RLock()
	defer a.mu.RUnlock()

	if a.closed {
		return "", fmt.Errorf("adapter is closed")
	}
	if key == "" {
		return "", fmt.Errorf("metadata key cannot be empty")
	}

	cKey := C.CString(key)
	defer C.free(unsafe.Pointer(cKey))

	// 1 KiB holds any scalar this is used for; a value longer than that is not a
	// scalar and not what this reads.
	buffer := make([]C.char, 1024)
	written := C.llama_wrapper_adapter_meta(a.adapterPtr, cKey, &buffer[0], C.int(len(buffer)))
	runtime.KeepAlive(buffer)
	if written < 0 {
		return "", fmt.Errorf("adapter declares no %s", key)
	}
	// llama_adapter_meta_val_str reports the length of the whole value, not the
	// length it managed to write, so a value that did not fit comes back as a
	// count past the end of the buffer. Handing that count to GoStringN reads off
	// the end of Go memory; truncating to what fits would be worse than the read,
	// because a prefix of "adapter.lora.alpha" is a different number and arrives
	// as a plausible loss rather than as a failure
	if int(written) >= len(buffer) {
		return "", fmt.Errorf("adapter value for %s is %d bytes, longer than the %d-byte buffer", key, int(written), len(buffer))
	}
	return C.GoStringN(&buffer[0], written), nil
}

// unknownDigest is what an adapter is labelled with when its file could not be
// hashed. It is a named constant so the one place that refuses to build a
// policy label out of it cannot drift from the one place that writes it.
const unknownDigest = "unknown"

// digestOfFile is what labels an adapter.
func digestOfFile(path string) (string, error) {
	file, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer file.Close()
	sum := sha256.New()
	if _, err := io.Copy(sum, file); err != nil {
		return "", err
	}
	return hex.EncodeToString(sum.Sum(nil)), nil
}
