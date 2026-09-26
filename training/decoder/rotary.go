package decoder

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"strings"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

// ErrRotarySpec identifies invalid token limits or a missing reference digest.
var ErrRotarySpec = errors.New("decoder: invalid text rotary specification")

// ErrRotaryIdentity means the native inverse frequencies differ from the
// mandatory expected identity. No target-device tables are allocated afterward.
var ErrRotaryIdentity = errors.New("decoder: rotary frequency identity differs")

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

// RotarySpec pins frequencies, text position limits and the intermediate dtype.
// HalfPrecision preserves an admitted Float16 trigonometric intermediate.
type RotarySpec struct {
	Theta                float64
	Dimension, MaxTokens int
	HalfPrecision        bool
	ExpectedSHA256       string
}

// TextRotary builds caller-admitted frequencies and text positions 0..tokens-1.
// The frequency digest is checked before target-device table allocation.
func TextRotary(ctx context.Context, tokens int, device torch.Device, spec RotarySpec) (_ *Rotary, err error) {
	if ctx == nil || tokens < 1 || tokens > spec.MaxTokens || spec.MaxTokens < 1 || spec.MaxTokens > 1<<20 || spec.Dimension < 2 || spec.Dimension > 1<<20 || spec.Dimension%2 != 0 || spec.Theta <= 0 || spec.Theta > math.MaxFloat32 || math.IsNaN(spec.Theta) || math.IsInf(spec.Theta, 0) {
		return nil, fmt.Errorf("%w: context and admitted rotary bounds required", ErrRotarySpec)
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	expected, err := hex.DecodeString(spec.ExpectedSHA256)
	if err != nil || len(expected) != 32 || strings.ToLower(spec.ExpectedSHA256) != spec.ExpectedSHA256 {
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
			err = errors.Join(errors.New("decoder: nonfinite rotary intermediate"), e)
			return nil
		}
		err = ctx.Err()
		return t
	}
	base := keep(func() (*torch.Tensor, error) {
		return torch.FromFloat32([]float32{float32(spec.Theta)}, []int64{1}, torch.CPUDevice(), false)
	})
	exponents := make([]float32, spec.Dimension/2)
	for i := range exponents {
		exponents[i] = float32(i*2) / float32(spec.Dimension)
	}
	powers := keep(func() (*torch.Tensor, error) {
		return torch.FromFloat32(exponents, []int64{int64(spec.Dimension / 2)}, torch.CPUDevice(), false)
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
	frequencies = keep(func() (*torch.Tensor, error) { return frequencies.Reshape([]int64{1, int64(spec.Dimension / 2)}) })
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
	if spec.HalfPrecision {
		cosine = keep(func() (*torch.Tensor, error) { return cosine.To(device, torch.Float16) })
		sine = keep(func() (*torch.Tensor, error) { return sine.To(device, torch.Float16) })
	}
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
		if info.DType != torch.Float32 || info.Device != device || info.RequiresGrad || !sameShape(info.Shape, []int64{int64(tokens), int64(spec.Dimension / 2)}) {
			return nil, errors.New("decoder: rotary table metadata differs")
		}
		finite, e := value.AllFinite()
		if e != nil || !finite {
			return nil, errors.Join(errors.New("decoder: nonfinite rotary table"), e)
		}
		if err = ctx.Err(); err != nil {
			return nil, err
		}
	}
	return result, nil
}
