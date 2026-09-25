//go:build libtorch && cgo

#include "bridge.h"
#include <torch/torch.h>
#ifdef __APPLE__
#include <torch/mps.h>
#include <ATen/detail/MPSHooksInterface.h>
#endif
#include <torch/version.h>
#include <ATen/CPUGeneratorImpl.h>
#include <c10/core/DeviceGuard.h>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

struct tayi_torch_tensor { torch::Tensor value; };
struct tayi_torch_generator { at::Generator value; };

extern "C" int tayi_torch_mps_available(void) {
#ifdef __APPLE__
    return torch::mps::is_available() ? 1 : 0;
#else
    return 0;
#endif
}

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
    catch (...) { error_text(error, capacity, "unknown native exception"); }
    return -1;
}

torch::ScalarType scalar_type(int dtype) {
    switch (dtype) {
        case 0: return torch::kFloat32;
        case 1: return torch::kFloat64;
        case 2: return torch::kFloat16;
        case 3: return torch::kBFloat16;
        case 4: return torch::kInt64;
        case 5: return torch::kBool;
        default: throw std::invalid_argument("unsupported scalar dtype");
    }
}

int dtype_code(torch::ScalarType dtype) {
    switch (dtype) {
        case torch::kFloat32: return 0;
        case torch::kFloat64: return 1;
        case torch::kFloat16: return 2;
        case torch::kBFloat16: return 3;
        case torch::kInt64: return 4;
        case torch::kBool: return 5;
        default: throw std::invalid_argument("unsupported result dtype");
    }
}

torch::Device device_type(int device) {
    if (device == -1) return torch::Device(torch::kCPU);
    if (device == -2) return torch::Device(torch::kMPS);
    if (device < 0 || device > 127) throw std::invalid_argument("invalid device index");
    return torch::Device(torch::kCUDA, static_cast<c10::DeviceIndex>(device));
}

const torch::Tensor &tensor(tayi_torch_tensor *handle) {
    if (!handle || !handle->value.defined()) throw std::invalid_argument("null or undefined tensor");
    return handle->value;
}

void little_endian() {
    const uint16_t marker = 1;
    if (*reinterpret_cast<const unsigned char *>(&marker) != 1)
        throw std::invalid_argument("only little-endian hosts are supported");
}

std::vector<int64_t> dimensions(const int64_t *shape, size_t rank) {
    if (rank > 32 || (rank && !shape)) throw std::invalid_argument("invalid rank or shape pointer");
    std::vector<int64_t> result;
    if (rank) result.assign(shape, shape + rank);
    int64_t elements = 1;
    for (int64_t size : result) {
        if (size < 0 || (size && elements > std::numeric_limits<int64_t>::max() / size))
            throw std::invalid_argument("negative or overflowing shape");
        elements *= size;
    }
    return result;
}

std::vector<torch::Tensor> tensor_list(tayi_torch_tensor *const *handles, size_t count) {
    if (!count || !handles) throw std::invalid_argument("empty tensor list");
    std::vector<torch::Tensor> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) result.push_back(tensor(handles[i]));
    return result;
}
}

extern "C" int tayi_torch_mps_memory(uint64_t *current, uint64_t *driver,
    uint64_t *recommended, char *error, size_t capacity) {
    return checked(error, capacity, [&] {
        if (!current || !driver || !recommended) throw std::invalid_argument("MPS memory outputs required");
#ifdef __APPLE__
        if (!torch::mps::is_available()) throw std::runtime_error("MPS unavailable");
        const auto &hooks = at::detail::getMPSHooks();
        *current = hooks.getCurrentAllocatedMemory();
        *driver = hooks.getDriverAllocatedMemory();
        *recommended = hooks.getRecommendedMaxMemory();
#else
        throw std::runtime_error("MPS memory is unavailable on this platform");
#endif
    });
}

extern "C" tayi_torch_generator_result tayi_torch_generator_create(uint64_t seed) {
    tayi_torch_generator_result result{};
    checked(result.error, sizeof(result.error), [&] {
        result.generator = new tayi_torch_generator{at::detail::createCPUGenerator(seed)};
    });
    return result;
}

extern "C" tayi_torch_result tayi_torch_generator_uniform(tayi_torch_generator *generator,
        const int64_t *shape, size_t rank, double low, double high, int dtype) {
    tayi_torch_result result{};
    checked(result.error, sizeof(result.error), [&] {
        if (!generator || !generator->value.defined())
            throw std::invalid_argument("null or undefined CPU generator");
        if (generator->value.device().type() != torch::kCPU)
            throw std::invalid_argument("uniform generator must use CPU");
        if (!std::isfinite(low) || !std::isfinite(high) || low > high)
            throw std::invalid_argument("uniform bounds must be finite and ordered");
        const auto scalar = scalar_type(dtype);
        if (!c10::isFloatingType(scalar))
            throw std::invalid_argument("uniform requires a floating dtype");
        auto value = torch::empty(dimensions(shape, rank),
            torch::TensorOptions().dtype(scalar).device(torch::kCPU).requires_grad(false));
        value.uniform_(low, high, generator->value);
        result.tensor = new tayi_torch_tensor{std::move(value)};
    });
    return result;
}

extern "C" int tayi_torch_generator_close(tayi_torch_generator *generator, char *error, size_t capacity) {
    return checked(error, capacity, [&] { delete generator; });
}

extern "C" tayi_torch_result tayi_torch_create(const void *data, int64_t bytes,
        const int64_t *shape, size_t rank, int dtype, int device, int requires_grad) {
    tayi_torch_result result{};
    checked(result.error, sizeof(result.error), [&] {
        little_endian();
        const auto sizes = dimensions(shape, rank);
        const auto destination = device_type(device);
        c10::OptionalDeviceGuard device_guard;
        if (destination.is_cuda()) device_guard.reset_device(destination);
        auto options = torch::TensorOptions().dtype(scalar_type(dtype)).device(torch::kCPU);
        int64_t elements = 1;
        for (int64_t size : sizes) elements *= size;
        const int64_t width = c10::elementSize(scalar_type(dtype));
        if (bytes < 0 || elements > std::numeric_limits<int64_t>::max()/width ||
            bytes != elements * width || (bytes && !data))
            throw std::invalid_argument("shape and dtype do not match input bytes");
        // Copy before returning: native tensors must never retain Go pointers.
        auto value = torch::empty(sizes, options);
        if (bytes) std::memcpy(value.data_ptr(), data, static_cast<size_t>(bytes));
        value = value.to(destination);
        value.requires_grad_(requires_grad != 0);
        result.tensor = new tayi_torch_tensor{std::move(value)};
    });
    return result;
}

extern "C" tayi_torch_result tayi_torch_apply(int operation, tayi_torch_tensor *const *inputs,
        size_t count, const int64_t *integers, size_t integer_count, double scalar) {
    tayi_torch_result result{};
    checked(result.error, sizeof(result.error), [&] {
        auto values = tensor_list(inputs, count);
        if (integer_count && !integers) throw std::invalid_argument("null integer arguments");
        const auto &a = values[0];
        c10::OptionalDeviceGuard device_guard;
        if (a.device().is_cuda()) device_guard.reset_device(a.device());
        auto integer = [&](size_t i) -> int64_t {
            if (i >= integer_count) throw std::invalid_argument("missing integer argument");
            return integers[i];
        };
        auto second = [&]() -> const torch::Tensor & {
            if (count != 2) throw std::invalid_argument("expected two input tensors");
            return values[1];
        };
        torch::Tensor output;
        // Operation numbers are shared with tensor.go; no model logic lives here.
        switch (operation) {
            case 0: output = torch::matmul(a, second()); break;
            case 1: output = a + second(); break;
            case 2: output = a * second(); break;
            case 3: output = a * scalar; break;
            case 4: output = a.reshape(dimensions(integers, integer_count)); break;
            case 5: output = a.transpose(integer(0), integer(1)); break;
            case 6: output = a.slice(integer(0), integer(1), integer(2), integer(3)); break;
            case 7: output = torch::cat(values, integer(0)); break;
            case 8:
            case 9: {
                const bool keep = integer(0) != 0;
                std::vector<int64_t> dims;
                if (integer_count > 1) dims.assign(integers + 1, integers + integer_count);
                else for (int64_t i = 0; i < a.dim(); ++i) dims.push_back(i);
                output = operation == 8 ? a.sum(dims, keep) : a.mean(dims, keep);
                break;
            }
            case 10: output = a.exp(); break;
            case 11: output = a.log(); break;
            case 12: output = a.sigmoid(); break;
            case 13: output = torch::silu(a); break;
            case 14: output = torch::softplus(a, 1, 20); break;
            case 15: output = a.rsqrt(); break;
            case 16: output = a.softmax(integer(0)); break;
            case 17: {
                const auto destination = device_type(static_cast<int>(integer(0)));
                const auto dtype = scalar_type(static_cast<int>(integer(1)));
                // Some qualified T10 pairs return a zero-filled result for a
                // direct peer copy. Stage only cross-GPU placement through
                // host memory; both copies remain in the autograd graph.
                if (a.device().is_cuda() && destination.is_cuda() && a.device() != destination) {
                    auto host = a.to(torch::TensorOptions().device(torch::kCPU).dtype(dtype), false, false);
                    output = host.to(torch::TensorOptions().device(destination).dtype(dtype), false, false);
                } else {
                    output = a.to(torch::TensorOptions().device(destination).dtype(dtype), false, false);
                }
                break;
            }
            case 18: output = a.clone(); break;
            case 19: output = a.detach(); break;
            case 20: output = a; output.requires_grad_(integer(0) != 0); break;
            case 21: output = a.tril(integer(0)); break;
            case 22: output = a.masked_fill(second(), scalar); break;
            case 23: output = a - second(); break;
            case 24: output = a.unsqueeze(integer(0)); break;
            case 25: output = a.squeeze(integer(0)); break;
            case 26: output = torch::stack(values, integer(0)); break;
            case 27: output = a.select(integer(0), integer(1)); break;
            case 28: output = a.isfinite().all(); break;
            case 29: output = a.index_select(integer(0), second()); break;
            case 30: output = a.pow(second()); break;
            case 31: output = a.reciprocal(); break;
            case 32: output = a.cos(); break;
            case 33: output = a.sin(); break;
            case 34: output = a.abs(); break;
            case 35: {
                const bool keep = integer(0) != 0;
                std::vector<int64_t> dims;
                if (integer_count > 1) dims.assign(integers + 1, integers + integer_count);
                else for (int64_t i = 0; i < a.dim(); ++i) dims.push_back(i);
                output = a.amax(dims, keep);
                break;
            }
            case 36: output = a.clamp_min(scalar); break;
            default: throw std::invalid_argument("unknown native tensor operation");
        }
        if (output.dim() > 32) throw std::invalid_argument("result rank exceeds 32");
        result.tensor = new tayi_torch_tensor{std::move(output)};
    });
    return result;
}

extern "C" int tayi_torch_metadata(tayi_torch_tensor *handle, tayi_torch_info *info,
        char *error, size_t capacity) {
    return checked(error, capacity, [&] {
        if (!info) throw std::invalid_argument("null metadata destination");
        const auto &value = tensor(handle);
        if (value.dim() > 32) throw std::invalid_argument("rank exceeds 32");
        if (!value.device().is_cpu() && !value.device().is_cuda() && !value.device().is_mps())
            throw std::invalid_argument("unsupported result device");
        *info = {};
        info->rank = static_cast<int>(value.dim());
        for (int i = 0; i < info->rank; ++i) info->shape[i] = value.size(i);
        info->elements = value.numel();
        info->dtype = dtype_code(value.scalar_type());
        info->device = value.device().is_cpu() ? -1 : value.device().is_mps() ? -2 : value.device().index();
        info->requires_grad = value.requires_grad() ? 1 : 0;
    });
}

extern "C" int tayi_torch_copy(tayi_torch_tensor *handle, int dtype, void *data, int64_t bytes,
        char *error, size_t capacity) {
    return checked(error, capacity, [&] {
        little_endian();
        const auto &input = tensor(handle);
        c10::OptionalDeviceGuard device_guard;
        if (input.device().is_cuda()) device_guard.reset_device(input.device());
        const int64_t width = c10::elementSize(scalar_type(dtype));
        if (bytes < 0 || input.numel() > std::numeric_limits<int64_t>::max()/width ||
            bytes != input.numel()*width || (bytes && !data))
            throw std::invalid_argument("invalid copy destination size");
        // Value extraction is diagnostic and never attaches the Go buffer to a graph.
        auto value = input.detach().to(torch::TensorOptions().device(torch::kCPU).dtype(scalar_type(dtype))).contiguous();
        if (bytes) std::memcpy(data, value.const_data_ptr(), static_cast<size_t>(bytes));
    });
}

extern "C" int tayi_torch_grad(tayi_torch_tensor *const *outputs, size_t output_count,
        tayi_torch_tensor *const *inputs, size_t input_count, tayi_torch_tensor *const *cotangents,
        int retain_graph, int create_graph, tayi_torch_tensor **gradients, char *error, size_t capacity) {
    return checked(error, capacity, [&] {
        if (!gradients) throw std::invalid_argument("null gradient destination");
        auto output_values = tensor_list(outputs, output_count);
        c10::OptionalDeviceGuard device_guard;
        if (output_values[0].device().is_cuda()) device_guard.reset_device(output_values[0].device());
        auto result = torch::autograd::grad(output_values, tensor_list(inputs, input_count),
            tensor_list(cotangents, output_count), retain_graph != 0, create_graph != 0, false);
        std::vector<std::unique_ptr<tayi_torch_tensor>> owned;
        owned.reserve(input_count);
        for (auto &value : result) {
            if (!value.defined()) throw std::runtime_error("undefined input gradient");
            owned.push_back(std::make_unique<tayi_torch_tensor>(tayi_torch_tensor{std::move(value)}));
        }
        if (owned.size() != input_count) throw std::runtime_error("gradient count mismatch");
        for (size_t i = 0; i < input_count; ++i) gradients[i] = owned[i].release();
    });
}

extern "C" int tayi_torch_close(tayi_torch_tensor *handle, char *error, size_t capacity) {
    return checked(error, capacity, [&] {
        const auto device = tensor(handle).device();
        c10::OptionalDeviceGuard device_guard;
        if (device.is_cuda()) device_guard.reset_device(device);
        delete handle;
    });
}

extern "C" const char *tayi_torch_header_version(void) { return TORCH_VERSION; }
