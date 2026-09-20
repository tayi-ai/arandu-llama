//go:build !libtorch || !libtorch_cuda || !cgo

package torch

const nativeCUDAEnabled = false

func nativeCUDADeviceCount() (int, error)            { return 0, ErrCUDAUnavailable }
func nativeCUDASetMemoryFraction(int, float64) error { return ErrCUDAUnavailable }
func nativeCUDAMemory(int) (CUDAMemoryStats, error)  { return CUDAMemoryStats{}, ErrCUDAUnavailable }
func nativeCUDASynchronize(int) error                { return ErrCUDAUnavailable }
