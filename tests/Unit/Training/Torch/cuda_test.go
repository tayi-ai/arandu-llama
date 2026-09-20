package torch_test

import (
	"errors"
	"math"
	"testing"

	"github.com/tayi-ai/arandu-llama/training/torch"
)

func TestCUDADeviceTransferSurvivesImmediateSourceClose(t *testing.T) {
	if !torch.CUDAEnabled() {
		t.Skip("CUDA bridge is not enabled")
	}
	count, err := torch.CUDADeviceCount()
	if err != nil {
		t.Fatal(err)
	}
	if count < 2 {
		t.Skip("two CUDA devices are required")
	}
	const elements = 1 << 20
	values := make([]float32, elements)
	for index := range values {
		values[index] = float32(index%2047-1023) / 128
	}
	check := func(stage string, iteration int, actual []float32) {
		t.Helper()
		for index := range values {
			if math.Float32bits(actual[index]) != math.Float32bits(values[index]) {
				t.Fatalf("%s iteration %d index %d: got %g want exactly %g", stage, iteration, index, actual[index], values[index])
			}
		}
	}
	for iteration := range 8 {
		source, err := torch.FromFloat32(values, []int64{elements}, torch.CUDADevice(0), false)
		if err != nil {
			t.Fatal(err)
		}
		sourceValues, err := source.Float32Values()
		if err != nil {
			_ = source.Close()
			t.Fatal(err)
		}
		check("source", iteration, sourceValues)
		destination, err := source.To(torch.CUDADevice(1), torch.Float32)
		if err != nil {
			_ = source.Close()
			t.Fatal(err)
		}
		beforeClose, err := destination.Float32Values()
		if err != nil {
			_ = source.Close()
			_ = destination.Close()
			t.Fatal(err)
		}
		check("destination_before_source_close", iteration, beforeClose)
		if err := source.Close(); err != nil {
			_ = destination.Close()
			t.Fatal(err)
		}
		actual, err := destination.Float32Values()
		if closeErr := destination.Close(); err == nil {
			err = closeErr
		}
		if err != nil {
			t.Fatal(err)
		}
		check("destination", iteration, actual)
	}
}

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
