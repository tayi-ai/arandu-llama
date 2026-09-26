//go:build libtorch && cgo

// Package fixture contains synthetic, independently materialized CPU references.
package fixture

import (
	"context"
	"crypto/sha256"
	"fmt"
	"math"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/decoder"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

// Spec supplies synthetic values for the large shape boundary tests.
func Spec(t *testing.T) decoder.InitialAdapterSpec {
	s := decoder.InitialAdapterSpec{Seed: 19, PreludeBlocks: 3, PreludeWidth: 5, PreludeLow: .1, PreludeHigh: 2}
	for layer := 3; layer < 32; layer += 4 {
		for _, projection := range []struct {
			name   string
			output int64
		}{{"q_proj", 8192}, {"v_proj", 1024}} {
			s.Projections = append(s.Projections, decoder.InitialProjection{Name: fmt.Sprintf("base_model.model.model.language_model.layers.%d.self_attn.%s", layer, projection.name), Input: 4096, Output: projection.output, Rank: 4})
		}
	}
	s.ExpectedSHA256 = Digest(t, s)
	return s
}

// Digest independently consumes the CPU reference stream and hashes its result.
func Digest(t *testing.T, s decoder.InitialAdapterSpec) string {
	t.Helper()
	g, err := torch.NewCPUGenerator(s.Seed)
	if err != nil {
		t.Fatal(err)
	}
	defer g.Close()
	draw := func(shape []int64, low, high float64) []byte {
		v, e := g.Uniform(shape, low, high, torch.Float32)
		if e != nil {
			t.Fatal(e)
		}
		defer v.Close()
		b, e := v.Bytes()
		if e != nil {
			t.Fatal(e)
		}
		return b
	}
	for i := 0; i < s.PreludeBlocks; i++ {
		draw([]int64{s.PreludeWidth}, s.PreludeLow, s.PreludeHigh)
	}
	h := sha256.New()
	for _, p := range s.Projections {
		a := 1 / math.Sqrt(float64(p.Input))
		b := 1 / math.Sqrt(float64(p.Rank))
		draw([]int64{p.Rank, p.Input}, -a, a)
		draw([]int64{p.Output, p.Rank}, -b, b)
		h.Write([]byte(p.Name + ".lora_A.default.weight"))
		h.Write(draw([]int64{p.Rank, p.Input}, -a, a))
		h.Write([]byte(p.Name + ".lora_B.default.weight"))
		h.Write(make([]byte, p.Output*p.Rank*4))
	}
	return fmt.Sprintf("%x", h.Sum(nil))
}

// Initial returns caller-owned synthetic tensors.
func Initial(t *testing.T) *decoder.InitialAdapter {
	t.Helper()
	a, e := decoder.InitializeAdapter(context.Background(), Spec(t))
	if e != nil {
		t.Fatal(e)
	}
	return a
}
