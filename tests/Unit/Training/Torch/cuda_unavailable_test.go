//go:build !libtorch || !libtorch_cuda || !cgo

package torch_test

import (
	"errors"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

func TestCUDAUnavailableBuildDoesNotClaimTelemetryOrLimits(t *testing.T) {
	if torch.CUDAEnabled() {
		t.Fatal("unlinked CUDA bridge advertised availability")
	}
	if count, err := torch.CUDADeviceCount(); count != 0 || !errors.Is(err, torch.ErrCUDAUnavailable) {
		t.Fatalf("unsupported count returned success: %d %v", count, err)
	}
	for _, device := range []int{0, 1, 127} {
		for _, fraction := range []float64{.8, 1} {
			if err := torch.SetCUDAMemoryFraction(device, fraction); !errors.Is(err, torch.ErrCUDAUnavailable) {
				t.Fatalf("unsupported cap returned success: %d %g %v", device, fraction, err)
			}
		}
		if stats, err := torch.CUDAMemory(device); !errors.Is(err, torch.ErrCUDAUnavailable) || stats != (torch.CUDAMemoryStats{}) {
			t.Fatalf("unsupported memory returned measurements: %+v %v", stats, err)
		}
		if err := torch.CUDASynchronize(device); !errors.Is(err, torch.ErrCUDAUnavailable) {
			t.Fatalf("unsupported synchronization returned success: %v", err)
		}
	}
}
