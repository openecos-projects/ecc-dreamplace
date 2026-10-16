#include "cuda_runtime.h"
#include <c10/cuda/CUDAException.h>
#include "utility/src/utils.cuh"
#include "pin2pin_attraction/src/functional_cuda.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
void pin2pinAttractionCudaForwardLauncher(
    const T *x, const T *y, const int *pairs, const T *weights,
    int num_pairs, T *partials, T *total, cudaStream_t stream) {
    constexpr int threads = 256;
    int blocks = (num_pairs + threads - 1) / threads;
    pin2pinAttractionCudaForward<T, threads><<<blocks, threads, 0, stream>>>(
        x, y, pairs, weights, num_pairs, partials);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    pin2pinAttractionCudaReduce<T, threads><<<1, threads, 0, stream>>>(
        partials, blocks, total);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void pin2pinAttractionCudaBackwardLauncher(
    const T *x, const T *y, const int *pairs, const T *weights,
    int num_ends, const int *sorted_pins, const int64_t *sorted_ends,
    const T *grad, T *gx, T *gy, cudaStream_t stream) {
    constexpr int threads = 128;
    constexpr int warps = threads / 32;
    int blocks = (num_ends + warps - 1) / warps;
    pin2pinAttractionCudaBackward<<<blocks, threads, 0, stream>>>(
        x, y, pairs, weights, num_ends, sorted_pins, sorted_ends, grad, gx, gy);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define REGISTER_KERNEL_LAUNCHERS(T) \
template void pin2pinAttractionCudaForwardLauncher<T>( \
    const T *, const T *, const int *, const T *, int, T *, T *, cudaStream_t); \
template void pin2pinAttractionCudaBackwardLauncher<T>( \
    const T *, const T *, const int *, const T *, int, const int *, \
    const int64_t *, const T *, T *, T *, cudaStream_t)

REGISTER_KERNEL_LAUNCHERS(float);
REGISTER_KERNEL_LAUNCHERS(double);

DREAMPLACE_END_NAMESPACE
