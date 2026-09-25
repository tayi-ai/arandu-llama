package torch

import "errors"

// ErrMPSUnavailable means this host has no usable Metal tensor device.
var ErrMPSUnavailable = errors.New("torch: MPS device unavailable")

// MPSMemoryStats reports current PyTorch tensor bytes, driver-reserved bytes,
// and the device's recommended working-set bytes. These are instantaneous
// readings, not a measured peak or a complete process-memory budget.
type MPSMemoryStats struct {
	CurrentAllocatedBytes uint64
	DriverAllocatedBytes  uint64
	RecommendedMaxBytes   uint64
}

// ReadMPSMemory samples the optional LibTorch MPS allocator on this host.
func ReadMPSMemory() (MPSMemoryStats, error) {
	if !nativeEnabled {
		return MPSMemoryStats{}, ErrUnavailable
	}
	if !MPSAvailable() {
		return MPSMemoryStats{}, ErrMPSUnavailable
	}
	return nativeMPSMemory()
}
