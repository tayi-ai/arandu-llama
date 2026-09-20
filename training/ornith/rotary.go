package ornith

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"strings"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

// ErrRotarySpec identifies invalid token limits or a missing reference digest.
var ErrRotarySpec = errors.New("ornith: invalid text rotary specification")

// ErrRotaryIdentity means the native inverse frequencies differ from the
// mandatory expected identity. No target-device tables are allocated afterward.
var ErrRotaryIdentity = errors.New("ornith: rotary frequency identity differs")

// Rotary owns text-only rotary tables. Half-width tables are expanded by the
// attention calculation. Multimodal coordinates and cache offsets are excluded.
type Rotary struct{ Cosine, Sine *torch.Tensor }

// Close releases both tables and is idempotent, including a nil receiver.
func (r *Rotary) Close() error {
	if r == nil {
		return nil
	}
	return errors.Join(r.Cosine.Close(), r.Sine.Close())
}

// TextRotary reproduces the fixed checkpoint's default RoPE frequencies and
// text positions 0..tokens-1. The required reference digest checks the 32 FP32
// inverse frequencies before target-device allocation. Trigonometric calculations
// run on the specified device; the FP16-output/FP32-attention casts reproduce
// the frozen model hooks. This is not a multimodal position-ID implementation.
// Source: Transformers v5.16.1 Qwen3_5TextRotaryEmbedding, lines 109-135.
// All intermediate values must be finite. Cancellation is checked between
// native operations; an operation already running cannot be preempted here.
func TextRotary(ctx context.Context, tokens int, device torch.Device, expectedFrequencySHA256 string) (_ *Rotary, err error) {
	if ctx == nil || tokens < 1 || tokens > 4096 {
		return nil, fmt.Errorf("%w: context required and tokens must be in [1,4096]", ErrRotarySpec)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	expected, err := hex.DecodeString(expectedFrequencySHA256)
	if err != nil || len(expected) != 32 || strings.ToLower(expectedFrequencySHA256) != expectedFrequencySHA256 {
		return nil, fmt.Errorf("%w: reference digest must be 64 lowercase hex digits", ErrRotarySpec)
	}
	var owned []*torch.Tensor
	defer func() {
		for _, value := range owned {
			_ = value.Close()
		}
	}()
	keep := func(operation func() (*torch.Tensor, error)) *torch.Tensor {
		if err == nil {
			err = ctx.Err()
		}
		if err != nil {
			return nil
		}
		t, e := operation()
		if e != nil {
			_ = t.Close()
			err = e
			return nil
		}
		owned = append(owned, t)
		finite, e := t.AllFinite()
		if e != nil || !finite {
			err = errors.Join(errors.New("ornith: nonfinite rotary intermediate"), e)
			return nil
		}
		err = ctx.Err()
		return t
	}
	base := keep(func() (*torch.Tensor, error) {
		return torch.FromFloat32([]float32{10000000}, []int64{1}, torch.CPUDevice(), false)
	})
	exponents := make([]float32, 32)
	for i := range exponents {
		exponents[i] = float32(i*2) / 64
	}
	powers := keep(func() (*torch.Tensor, error) {
		return torch.FromFloat32(exponents, []int64{32}, torch.CPUDevice(), false)
	})
	frequencies := keep(func() (*torch.Tensor, error) { return base.Pow(powers) })
	frequencies = keep(func() (*torch.Tensor, error) { return frequencies.Reciprocal() })
	if err != nil {
		return nil, err
	}
	raw, err := frequencies.Bytes()
	if err != nil {
		return nil, err
	}
	digest := sha256.Sum256(raw)
	if !bytes.Equal(digest[:], expected) {
		return nil, fmt.Errorf("%w: %x", ErrRotaryIdentity, digest)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	frequencies = keep(func() (*torch.Tensor, error) { return frequencies.To(device, torch.Float32) })
	frequencies = keep(func() (*torch.Tensor, error) { return frequencies.Reshape([]int64{1, 32}) })
	positions := make([]float32, tokens)
	for i := range positions {
		positions[i] = float32(i)
	}
	position := keep(func() (*torch.Tensor, error) {
		return torch.FromFloat32(positions, []int64{int64(tokens), 1}, device, false)
	})
	angles := keep(func() (*torch.Tensor, error) { return position.MatMul(frequencies) })
	cosine := keep(func() (*torch.Tensor, error) { return angles.Cos() })
	sine := keep(func() (*torch.Tensor, error) { return angles.Sin() })
	cosine = keep(func() (*torch.Tensor, error) { return cosine.To(device, torch.Float16) })
	sine = keep(func() (*torch.Tensor, error) { return sine.To(device, torch.Float16) })
	if err != nil {
		return nil, err
	}
	result := &Rotary{}
	defer func() {
		if err != nil {
			_ = result.Close()
		}
	}()
	result.Cosine, err = cosine.To(device, torch.Float32)
	if err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	result.Sine, err = sine.To(device, torch.Float32)
	if err != nil {
		return nil, err
	}
	if err = ctx.Err(); err != nil {
		return nil, err
	}
	for _, value := range []*torch.Tensor{result.Cosine, result.Sine} {
		info, e := value.Info()
		if e != nil {
			return nil, e
		}
		if info.DType != torch.Float32 || info.Device != device || info.RequiresGrad || !sameShape(info.Shape, []int64{int64(tokens), 32}) {
			return nil, errors.New("ornith: rotary table metadata differs")
		}
		finite, e := value.AllFinite()
		if e != nil || !finite {
			return nil, errors.Join(errors.New("ornith: nonfinite rotary table"), e)
		}
		if err = ctx.Err(); err != nil {
			return nil, err
		}
	}
	return result, nil
}
