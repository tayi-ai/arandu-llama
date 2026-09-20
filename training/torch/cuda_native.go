//go:build libtorch && libtorch_cuda && cgo

package torch

/*
// Keep CUDA registration hooks linked even when the linker drops unused DSOs.
#cgo linux LDFLAGS: -Wl,--no-as-needed -ltorch_cuda -Wl,--as-needed -lc10_cuda -lcudart
#cgo !linux LDFLAGS: -ltorch_cuda -lc10_cuda -lcudart
#include "bridge_cuda.h"
*/
import "C"

import "fmt"

const nativeCUDAEnabled = true

func nativeCUDADeviceCount() (int, error) {
	var count C.int
	var message [2048]C.char
	if C.tayi_torch_cuda_device_count(&count, &message[0], C.size_t(len(message))) != 0 {
		return 0, fmt.Errorf("torch: CUDA: %s", C.GoString(&message[0]))
	}
	return int(count), nil
}

func nativeCUDASetMemoryFraction(device int, fraction float64) error {
	var message [2048]C.char
	if C.tayi_torch_cuda_set_memory_fraction(C.int(device), C.double(fraction), &message[0], C.size_t(len(message))) != 0 {
		return fmt.Errorf("torch: CUDA: %s", C.GoString(&message[0]))
	}
	return nil
}

func nativeCUDAMemory(device int) (CUDAMemoryStats, error) {
	var value C.tayi_torch_cuda_memory
	var message [2048]C.char
	if C.tayi_torch_cuda_memory_stats(C.int(device), &value, &message[0], C.size_t(len(message))) != 0 {
		return CUDAMemoryStats{}, fmt.Errorf("torch: CUDA: %s", C.GoString(&message[0]))
	}
	backend := C.GoString(&value.allocator_backend[0])
	enabled := value.allocator_enabled != 0
	return CUDAMemoryStats{Device: device, FreeBytes: int64(value.free_bytes), TotalBytes: int64(value.total_bytes),
		AllocatedBytes: int64(value.allocated_bytes), ReservedBytes: int64(value.reserved_bytes),
		PeakAllocatedBytes: int64(value.peak_allocated_bytes), PeakReservedBytes: int64(value.peak_reserved_bytes),
		AllocatorBackend: backend, AllocatorEnabled: enabled,
		AllocatorStatsValid: enabled && backend == "native", AllocatorFraction: float64(value.allocator_fraction)}, nil
}

func nativeCUDASynchronize(device int) error {
	var message [2048]C.char
	if C.tayi_torch_cuda_synchronize(C.int(device), &message[0], C.size_t(len(message))) != 0 {
		return fmt.Errorf("torch: CUDA: %s", C.GoString(&message[0]))
	}
	return nil
}
