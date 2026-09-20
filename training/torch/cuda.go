package torch

import (
	"errors"
	"fmt"
	"math"
)

// ErrCUDAUnavailable means this build lacks the optional CUDA telemetry bridge.
var ErrCUDAUnavailable = errors.New("torch: CUDA telemetry requires libtorch, libtorch_cuda and cgo")

// ErrCUDAArgument means a CUDA request failed validation before any native call.
var ErrCUDAArgument = errors.New("torch: invalid CUDA argument")

// CUDAMemoryStats separates device-wide CUDA memory from this process's caching
// allocator counters. FreeBytes and TotalBytes come from cudaMemGetInfo, not
// NVML. AllocatorStatsValid is true only for the enabled native caching allocator;
// alternative allocator backends can omit counters or report zeros. Peaks are
// cumulative and are never reset here. These counters exclude allocations made
// outside this allocator, including CUDA contexts and external library storage.
type CUDAMemoryStats struct {
	Device              int     `json:"device"`
	FreeBytes           int64   `json:"free_bytes"`
	TotalBytes          int64   `json:"total_bytes"`
	AllocatedBytes      int64   `json:"allocated_bytes"`
	ReservedBytes       int64   `json:"reserved_bytes"`
	PeakAllocatedBytes  int64   `json:"peak_allocated_bytes"`
	PeakReservedBytes   int64   `json:"peak_reserved_bytes"`
	AllocatorBackend    string  `json:"allocator_backend"`
	AllocatorEnabled    bool    `json:"allocator_enabled"`
	AllocatorStatsValid bool    `json:"allocator_stats_valid"`
	AllocatorFraction   float64 `json:"allocator_fraction"`
}

// CUDAEnabled reports build availability only. It neither queries a device nor
// establishes that a CUDA driver, usable hardware or sufficient memory exists.
func CUDAEnabled() bool { return nativeCUDAEnabled }

// CUDADeviceCount queries the CUDA runtime for the number of visible devices.
// Runtime and driver failures return errors rather than a successful zero count.
func CUDADeviceCount() (int, error) {
	if !nativeCUDAEnabled {
		return 0, ErrCUDAUnavailable
	}
	calls.Lock()
	defer calls.Unlock()
	return nativeCUDADeviceCount()
}

// SetCUDAMemoryFraction explicitly limits this process's caching allocator on
// one device to a fraction of total CUDA-visible memory. The fraction must be
// finite and satisfy 0 < fraction <= 1. Call before loading model tensors; this
// does not release existing allocations or cap total process GPU memory. It may
// initialize the CUDA context. The prior thread device is restored on return.
func SetCUDAMemoryFraction(device int, fraction float64) error {
	if err := validateCUDADevice(device); err != nil {
		return err
	}
	if math.IsNaN(fraction) || math.IsInf(fraction, 0) || fraction <= 0 || fraction > 1 {
		return fmt.Errorf("%w: memory fraction must be finite and within (0, 1]", ErrCUDAArgument)
	}
	if !nativeCUDAEnabled {
		return ErrCUDAUnavailable
	}
	calls.Lock()
	defer calls.Unlock()
	return nativeCUDASetMemoryFraction(device, fraction)
}

// CUDAMemory reads device-wide free/total bytes and process allocator counters.
// It may initialize CUDA but never resets peaks, empties caches or synchronizes.
// The individual readings are not an atomic snapshot of asynchronous workloads.
// The prior thread device is restored. An error returns an empty statistics value.
func CUDAMemory(device int) (CUDAMemoryStats, error) {
	if err := validateCUDADevice(device); err != nil {
		return CUDAMemoryStats{}, err
	}
	if !nativeCUDAEnabled {
		return CUDAMemoryStats{}, ErrCUDAUnavailable
	}
	calls.Lock()
	defer calls.Unlock()
	return nativeCUDAMemory(device)
}

// CUDASynchronize explicitly waits for work on one CUDA device and surfaces
// asynchronous errors. It may initialize CUDA and restores the prior thread
// device. The call has no cancellation or implicit deadline.
func CUDASynchronize(device int) error {
	if err := validateCUDADevice(device); err != nil {
		return err
	}
	if !nativeCUDAEnabled {
		return ErrCUDAUnavailable
	}
	calls.Lock()
	defer calls.Unlock()
	return nativeCUDASynchronize(device)
}

func validateCUDADevice(device int) error {
	if device < 0 || device > 127 {
		return fmt.Errorf("%w: device index must be between 0 and 127", ErrCUDAArgument)
	}
	return nil
}
