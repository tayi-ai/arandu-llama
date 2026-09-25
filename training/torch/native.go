//go:build libtorch && cgo

package torch

/*
#cgo CXXFLAGS: -std=c++20
#cgo LDFLAGS: -ltorch -ltorch_cpu -lc10
#cgo linux LDFLAGS: -lstdc++
#cgo darwin LDFLAGS: -lc++
#include "bridge.h"
*/
import "C"

import (
	"fmt"
	"runtime"
	"unsafe"
)

const nativeEnabled = true

func nativeVersion() (string, error) { return C.GoString(C.tayi_torch_header_version()), nil }

func nativeMPSAvailable() bool { return C.tayi_torch_mps_available() != 0 }

func resultValue(result C.tayi_torch_result) (unsafe.Pointer, error) {
	if result.tensor == nil {
		return nil, fmt.Errorf("torch: %s", C.GoString(&result.error[0]))
	}
	return unsafe.Pointer(result.tensor), nil
}

func nativeCreate(data unsafe.Pointer, size int64, owner any, shape []int64, dtype DType, device Device, requiresGrad bool) (unsafe.Pointer, error) {
	index, _ := deviceIndex(device)
	result := C.tayi_torch_create(data, C.int64_t(size), (*C.int64_t)(unsafe.Pointer(unsafe.SliceData(shape))),
		C.size_t(len(shape)), C.int(dtype), C.int(index), boolean(requiresGrad))
	runtime.KeepAlive(owner)
	runtime.KeepAlive(shape)
	return resultValue(result)
}

func nativeApply(operation operation, handles []unsafe.Pointer, integers []int64, scalar float64) (unsafe.Pointer, error) {
	result := C.tayi_torch_apply(C.int(operation), (**C.tayi_torch_tensor)(unsafe.Pointer(unsafe.SliceData(handles))),
		C.size_t(len(handles)), (*C.int64_t)(unsafe.Pointer(unsafe.SliceData(integers))), C.size_t(len(integers)), C.double(scalar))
	runtime.KeepAlive(handles)
	runtime.KeepAlive(integers)
	return resultValue(result)
}

func nativeInfo(handle unsafe.Pointer) (Info, error) {
	var info C.tayi_torch_info
	var message [2048]C.char
	if C.tayi_torch_metadata((*C.tayi_torch_tensor)(handle), &info, &message[0], C.size_t(len(message))) != 0 {
		return Info{}, fmt.Errorf("torch: %s", C.GoString(&message[0]))
	}
	result := Info{Shape: make([]int64, int(info.rank)), DType: DType(info.dtype),
		Device: CPUDevice(), RequiresGrad: info.requires_grad != 0, Elements: int64(info.elements)}
	for index := range result.Shape {
		result.Shape[index] = int64(info.shape[index])
	}
	if info.device >= 0 {
		result.Device = CUDADevice(int(info.device))
	} else if info.device == -2 {
		result.Device = MPSDevice()
	}
	return result, nil
}

func nativeCopy(handle unsafe.Pointer, dtype DType, data []byte) error {
	var message [2048]C.char
	code := C.tayi_torch_copy((*C.tayi_torch_tensor)(handle), C.int(dtype), unsafe.Pointer(unsafe.SliceData(data)),
		C.int64_t(len(data)), &message[0], C.size_t(len(message)))
	runtime.KeepAlive(data)
	if code != 0 {
		return fmt.Errorf("torch: %s", C.GoString(&message[0]))
	}
	return nil
}

func nativeGrad(outputs, inputs, cotangents []unsafe.Pointer, retainGraph, createGraph bool) ([]unsafe.Pointer, error) {
	gradients := make([]unsafe.Pointer, len(inputs))
	var message [2048]C.char
	code := C.tayi_torch_grad((**C.tayi_torch_tensor)(unsafe.Pointer(unsafe.SliceData(outputs))), C.size_t(len(outputs)),
		(**C.tayi_torch_tensor)(unsafe.Pointer(unsafe.SliceData(inputs))), C.size_t(len(inputs)),
		(**C.tayi_torch_tensor)(unsafe.Pointer(unsafe.SliceData(cotangents))), boolean(retainGraph), boolean(createGraph),
		(**C.tayi_torch_tensor)(unsafe.Pointer(unsafe.SliceData(gradients))), &message[0], C.size_t(len(message)))
	runtime.KeepAlive(outputs)
	runtime.KeepAlive(inputs)
	runtime.KeepAlive(cotangents)
	if code != 0 {
		return nil, fmt.Errorf("torch: %s", C.GoString(&message[0]))
	}
	return gradients, nil
}

func nativeClose(handle unsafe.Pointer) error {
	var message [2048]C.char
	if C.tayi_torch_close((*C.tayi_torch_tensor)(handle), &message[0], C.size_t(len(message))) != 0 {
		return fmt.Errorf("torch: %s", C.GoString(&message[0]))
	}
	return nil
}

func nativeGeneratorCreate(seed uint64) (unsafe.Pointer, error) {
	result := C.tayi_torch_generator_create(C.uint64_t(seed))
	if result.generator == nil {
		return nil, fmt.Errorf("torch: %s", C.GoString(&result.error[0]))
	}
	return unsafe.Pointer(result.generator), nil
}

func nativeGeneratorUniform(handle unsafe.Pointer, shape []int64, low, high float64, dtype DType) (unsafe.Pointer, error) {
	result := C.tayi_torch_generator_uniform((*C.tayi_torch_generator)(handle),
		(*C.int64_t)(unsafe.Pointer(unsafe.SliceData(shape))), C.size_t(len(shape)), C.double(low), C.double(high), C.int(dtype))
	runtime.KeepAlive(shape)
	return resultValue(result)
}

func nativeGeneratorClose(handle unsafe.Pointer) error {
	var message [2048]C.char
	if C.tayi_torch_generator_close((*C.tayi_torch_generator)(handle), &message[0], C.size_t(len(message))) != 0 {
		return fmt.Errorf("torch: %s", C.GoString(&message[0]))
	}
	return nil
}

func boolean(value bool) C.int {
	if value {
		return 1
	}
	return 0
}
