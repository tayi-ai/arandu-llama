//go:build libtorch && libtorch_cuda && cgo

#include "bridge_cuda.h"
#include <ATen/Context.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAFunctions.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime_api.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {
void error_text(char *output, size_t capacity, const char *message) noexcept {
    if (!output || capacity == 0) return;
    std::strncpy(output, message, capacity - 1);
    output[capacity - 1] = '\0';
}

template<class F> int checked(char *error, size_t capacity, F function) noexcept {
    if (error && capacity) error[0] = '\0';
    try { function(); return 0; }
    catch (const c10::Error &failure) { error_text(error, capacity, failure.what_without_backtrace()); }
    catch (const std::exception &failure) { error_text(error, capacity, failure.what()); }
    catch (...) { error_text(error, capacity, "unknown CUDA exception"); }
    return -1;
}

int device_count() {
    int count = 0;
    C10_CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

c10::DeviceIndex checked_device(int device) {
    if (device < 0 || device > 127) throw std::invalid_argument("invalid CUDA device index");
    if (device >= device_count()) throw std::invalid_argument("CUDA device is not visible");
    return static_cast<c10::DeviceIndex>(device);
}

void initialize() {
    // Standalone LibTorch must initialize its CUDA hooks/allocator explicitly.
    at::globalContext().lazyInitDevice(c10::DeviceType::CUDA);
}
}

extern "C" int tayi_torch_cuda_device_count(int *count, char *error, size_t capacity) {
    return checked(error, capacity, [&] {
        if (!count) throw std::invalid_argument("null device count output");
        *count = device_count();
    });
}

extern "C" int tayi_torch_cuda_set_memory_fraction(int device, double fraction, char *error, size_t capacity) {
    return checked(error, capacity, [&] {
        if (!std::isfinite(fraction) || fraction <= 0 || fraction > 1)
            throw std::invalid_argument("memory fraction must be finite and within (0, 1]");
        const auto index = checked_device(device);
        const c10::cuda::CUDAGuard guard(index);
        initialize();
        if (!c10::cuda::CUDACachingAllocator::isEnabled())
            throw std::runtime_error("CUDA caching allocator is disabled; memory fraction cannot be enforced");
        c10::cuda::CUDACachingAllocator::setMemoryFraction(fraction, index);
    });
}

extern "C" int tayi_torch_cuda_memory_stats(int device, tayi_torch_cuda_memory *stats, char *error, size_t capacity) {
    return checked(error, capacity, [&] {
        if (!stats) throw std::invalid_argument("null memory statistics output");
        const auto index = checked_device(device);
        const c10::cuda::CUDAGuard guard(index);
        initialize();
        size_t free_bytes = 0, total_bytes = 0;
        C10_CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        if (total_bytes > static_cast<size_t>(std::numeric_limits<int64_t>::max()) || free_bytes > total_bytes)
            throw std::runtime_error("CUDA memory statistics exceed the represented range");
        const auto memory = c10::cuda::CUDACachingAllocator::getDeviceStats(index);
        constexpr auto all = static_cast<size_t>(c10::CachingAllocator::StatType::AGGREGATE);
        const auto backend = c10::cuda::CUDACachingAllocator::name();
        if (backend.size() >= sizeof(stats->allocator_backend))
            throw std::runtime_error("CUDA allocator backend name exceeds the represented range");
        tayi_torch_cuda_memory value{};
        value.free_bytes = static_cast<int64_t>(free_bytes);
        value.total_bytes = static_cast<int64_t>(total_bytes);
        value.allocated_bytes = memory.allocated_bytes[all].current;
        value.reserved_bytes = memory.reserved_bytes[all].current;
        value.peak_allocated_bytes = memory.allocated_bytes[all].peak;
        value.peak_reserved_bytes = memory.reserved_bytes[all].peak;
        value.allocator_fraction = c10::cuda::CUDACachingAllocator::getMemoryFraction(index);
        value.allocator_enabled = c10::cuda::CUDACachingAllocator::isEnabled() ? 1 : 0;
        std::memcpy(value.allocator_backend, backend.c_str(), backend.size() + 1);
        *stats = value;
    });
}

extern "C" int tayi_torch_cuda_synchronize(int device, char *error, size_t capacity) {
    return checked(error, capacity, [&] {
        const auto index = checked_device(device);
        const c10::cuda::CUDAGuard guard(index);
        initialize();
        c10::cuda::device_synchronize();
    });
}
