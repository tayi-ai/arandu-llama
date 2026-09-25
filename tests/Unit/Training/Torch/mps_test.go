//go:build libtorch && cgo && darwin

package torch_test

import (
	"math"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

func TestMPSMatMulAutogradAndRoundTrip(t *testing.T) {
	if !torch.MPSAvailable() {
		t.Skip("MPS device is unavailable")
	}
	keep := scope(t)
	w := keep(torch.FromFloat32([]float32{1, 2, 3, 4}, []int64{2, 2}, torch.MPSDevice(), true))
	x := keep(torch.FromFloat32([]float32{5, 6}, []int64{2}, torch.MPSDevice(), false))
	info, err := w.Info()
	if err != nil {
		t.Fatal(err)
	}
	if info.Device != torch.MPSDevice() || !info.RequiresGrad {
		t.Fatalf("unexpected MPS metadata: %+v", info)
	}
	y := keep(w.MatMul(x))
	actual, err := y.Float32Values()
	if err != nil {
		t.Fatal(err)
	}
	if len(actual) != 2 || actual[0] != 17 || actual[1] != 39 {
		t.Fatalf("MPS forward: %v", actual)
	}
	seed := keep(torch.FromFloat32([]float32{1, 1}, []int64{2}, torch.MPSDevice(), false))
	gradients, err := torch.Grad([]*torch.Tensor{y}, []*torch.Tensor{w}, []*torch.Tensor{seed}, false, false)
	if err != nil {
		t.Fatal(err)
	}
	gradient := keep(gradients[0], nil)
	got, err := gradient.Float32Values()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != 4 {
		t.Fatalf("MPS gradient length: %d", len(got))
	}
	for i, want := range []float32{5, 6, 5, 6} {
		if math.Abs(float64(got[i]-want)) > 1e-5 {
			t.Fatalf("MPS gradient[%d] = %g, want %g", i, got[i], want)
		}
	}
	if _, err := torch.FromFloat32([]float32{1}, []int64{1}, torch.Device{Kind: "mps", Index: 1}, false); err == nil {
		t.Fatal("indexed MPS device was accepted")
	}
}

func TestMPSMemoryTelemetry(t *testing.T) {
	if !torch.MPSAvailable() {
		t.Skip("MPS device is unavailable")
	}
	stats, err := torch.ReadMPSMemory()
	if err != nil {
		t.Fatal(err)
	}
	if stats.RecommendedMaxBytes == 0 || stats.DriverAllocatedBytes < stats.CurrentAllocatedBytes {
		t.Fatalf("invalid MPS allocator reading: %+v", stats)
	}
}
