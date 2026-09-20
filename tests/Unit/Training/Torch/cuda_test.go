package torch_test

import (
	"errors"
	"math"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

func TestCUDARejectsInvalidArgumentsBeforeBackendDispatch(t *testing.T) {
	for _, device := range []int{-1, -128, 128, 65536} {
		if err := torch.SetCUDAMemoryFraction(device, .8); !errors.Is(err, torch.ErrCUDAArgument) {
			t.Fatalf("invalid device fraction: %d %v", device, err)
		}
		if stats, err := torch.CUDAMemory(device); !errors.Is(err, torch.ErrCUDAArgument) || stats != (torch.CUDAMemoryStats{}) {
			t.Fatalf("invalid device memory: %d %+v %v", device, stats, err)
		}
		if err := torch.CUDASynchronize(device); !errors.Is(err, torch.ErrCUDAArgument) {
			t.Fatalf("invalid device synchronization: %d %v", device, err)
		}
	}
	for _, fraction := range []float64{-1, 0, math.SmallestNonzeroFloat64 * -1, 1.0000000001, math.Inf(1), math.Inf(-1), math.NaN()} {
		if err := torch.SetCUDAMemoryFraction(0, fraction); !errors.Is(err, torch.ErrCUDAArgument) {
			t.Fatalf("invalid fraction reached native backend: %g %v", fraction, err)
		}
	}
}
