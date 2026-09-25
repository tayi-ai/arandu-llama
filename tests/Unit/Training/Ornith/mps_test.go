//go:build libtorch && cgo && darwin

package ornith_test

import (
	"context"
	"math"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/ornith"
	"github.com/tayi-ai/arandu-llama/training/torch"
)

func TestMPSFullDecoderCompletionGradientMatchesCPU(t *testing.T) {
	if !torch.MPSAvailable() {
		t.Skip("MPS device is unavailable")
	}
	cpu := newFixture(t, torch.Float32)
	metal := newFixtureOnDevice(t, torch.Float32, torch.MPSDevice())
	cpuResult, err := ornith.CompletionGradient(context.Background(), cpu.model, cpu.tokens, 2, cpu.limits, 1)
	if err != nil {
		t.Fatal(err)
	}
	metalResult, err := ornith.CompletionGradient(context.Background(), metal.model, metal.tokens, 2, metal.limits, 1)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(cpuResult.Loss-metalResult.Loss) > 1e-3 || len(cpuResult.Gradients) != 32 || len(metalResult.Gradients) != 32 {
		t.Fatalf("CPU/MPS completion differs: cpu=%g metal=%g gradients=%d/%d", cpuResult.Loss, metalResult.Loss, len(cpuResult.Gradients), len(metalResult.Gradients))
	}
	var nonzero bool
	for i, gradient := range metalResult.Gradients {
		if gradient.Name != cpuResult.Gradients[i].Name || len(gradient.ValuesF32) != len(cpuResult.Gradients[i].ValuesF32) {
			t.Fatalf("gradient layout differs at %d", i)
		}
		for j, value := range gradient.ValuesF32 {
			if math.IsNaN(float64(value)) || math.IsInf(float64(value), 0) || math.Abs(float64(value-cpuResult.Gradients[i].ValuesF32[j])) > 1e-2 {
				t.Fatalf("MPS gradient[%d][%d] differs: %g", i, j, value)
			}
			nonzero = nonzero || value != 0
		}
	}
	if !nonzero {
		t.Fatal("MPS completion has no trainable gradient")
	}
}
