//go:build !libtorch || !cgo

package torch

import "unsafe"

const nativeEnabled = false

func nativeVersion() (string, error)           { return "", ErrUnavailable }
func nativeMPSAvailable() bool                 { return false }
func nativeMPSMemory() (MPSMemoryStats, error) { return MPSMemoryStats{}, ErrUnavailable }
func nativeCreate(unsafe.Pointer, int64, any, []int64, DType, Device, bool) (unsafe.Pointer, error) {
	return nil, ErrUnavailable
}
func nativeApply(operation, []unsafe.Pointer, []int64, float64) (unsafe.Pointer, error) {
	return nil, ErrUnavailable
}
func nativeInfo(unsafe.Pointer) (Info, error)        { return Info{}, ErrUnavailable }
func nativeCopy(unsafe.Pointer, DType, []byte) error { return ErrUnavailable }
func nativeGrad([]unsafe.Pointer, []unsafe.Pointer, []unsafe.Pointer, bool, bool) ([]unsafe.Pointer, error) {
	return nil, ErrUnavailable
}
func nativeClose(unsafe.Pointer) error                     { return ErrUnavailable }
func nativeGeneratorCreate(uint64) (unsafe.Pointer, error) { return nil, ErrUnavailable }
func nativeGeneratorUniform(unsafe.Pointer, []int64, float64, float64, DType) (unsafe.Pointer, error) {
	return nil, ErrUnavailable
}
func nativeGeneratorClose(unsafe.Pointer) error { return ErrUnavailable }
