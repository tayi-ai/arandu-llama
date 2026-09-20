#ifndef TAYI_TORCH_CUDA_BRIDGE_H
#define TAYI_TORCH_CUDA_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    int64_t free_bytes;
    int64_t total_bytes;
    int64_t allocated_bytes;
    int64_t reserved_bytes;
    int64_t peak_allocated_bytes;
    int64_t peak_reserved_bytes;
    double allocator_fraction;
    int allocator_enabled;
    char allocator_backend[128];
} tayi_torch_cuda_memory;

int tayi_torch_cuda_device_count(int *count, char *error, size_t capacity);
int tayi_torch_cuda_set_memory_fraction(int device, double fraction, char *error, size_t capacity);
int tayi_torch_cuda_memory_stats(int device, tayi_torch_cuda_memory *stats, char *error, size_t capacity);
int tayi_torch_cuda_synchronize(int device, char *error, size_t capacity);

#ifdef __cplusplus
}
#endif
#endif
