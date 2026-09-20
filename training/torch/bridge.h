#ifndef TAYI_TORCH_BRIDGE_H
#define TAYI_TORCH_BRIDGE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct tayi_torch_tensor tayi_torch_tensor;
typedef struct tayi_torch_generator tayi_torch_generator;
typedef struct {
    tayi_torch_tensor *tensor;
    char error[2048];
} tayi_torch_result;

typedef struct {
    tayi_torch_generator *generator;
    char error[2048];
} tayi_torch_generator_result;

typedef struct {
    int64_t shape[32];
    int64_t elements;
    int rank;
    int dtype;
    int device;
    int requires_grad;
} tayi_torch_info;

tayi_torch_result tayi_torch_create(const void *data, int64_t bytes, const int64_t *shape,
    size_t rank, int dtype, int device, int requires_grad);
tayi_torch_result tayi_torch_apply(int operation, tayi_torch_tensor *const *inputs,
    size_t count, const int64_t *integers, size_t integer_count, double scalar);
int tayi_torch_metadata(tayi_torch_tensor *tensor, tayi_torch_info *info, char *error, size_t capacity);
int tayi_torch_copy(tayi_torch_tensor *tensor, int dtype, void *data, int64_t bytes, char *error, size_t capacity);
int tayi_torch_grad(tayi_torch_tensor *const *outputs, size_t output_count,
    tayi_torch_tensor *const *inputs, size_t input_count, tayi_torch_tensor *const *cotangents,
    int retain_graph, int create_graph, tayi_torch_tensor **gradients, char *error, size_t capacity);
int tayi_torch_close(tayi_torch_tensor *tensor, char *error, size_t capacity);
const char *tayi_torch_header_version(void);
tayi_torch_generator_result tayi_torch_generator_create(uint64_t seed);
tayi_torch_result tayi_torch_generator_uniform(tayi_torch_generator *generator,
    const int64_t *shape, size_t rank, double low, double high, int dtype);
int tayi_torch_generator_close(tayi_torch_generator *generator, char *error, size_t capacity);

#ifdef __cplusplus
}
#endif
#endif
