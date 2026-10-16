#include <cmath>
#include <cstdint>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

#include "buffer_interpolate_math.h"

namespace {
template <typename scalar_t>
__global__ void interpolate(const scalar_t* index, const scalar_t* cap, const scalar_t* delay,
    const scalar_t* slew, scalar_t* values, scalar_t* clipped, int64_t* lo, int64_t* hi,
    scalar_t* alpha, scalar_t* metadata, int32_t* flags, int rows, int sizes) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) {
    const int invalid = interpolateRow<scalar_t>(row, index, cap, delay, slew, values, clipped, lo, hi, alpha, metadata, rows, sizes);
    if (invalid) atomicOr(flags, invalid);
  }
}

template <typename scalar_t>
__global__ void finishMetadata(const int32_t* flags, scalar_t* metadata) {
  metadata[0] = scalar_t(*flags);
}

template <typename scalar_t>
__global__ void interpolateBackward(const scalar_t* index, const scalar_t* cap, const scalar_t* delay,
    const scalar_t* slew, const int64_t* lo, const int64_t* hi, const scalar_t* alpha,
    const scalar_t* grad_values, const scalar_t* grad_clipped, const scalar_t* grad_alpha,
    scalar_t* grad_index, scalar_t* grad_cap, scalar_t* grad_delay, scalar_t* grad_slew,
    int rows, int sizes) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < rows) interpolateBackwardRow<scalar_t>(row, index, cap, delay, slew, lo, hi, alpha, grad_values, grad_clipped, grad_alpha, grad_index, grad_cap, grad_delay, grad_slew, rows, sizes);
}
}  // namespace

template <typename scalar_t>
void bufferInterpolateForwardLauncher(const scalar_t* index, const scalar_t* cap, const scalar_t* delay,
    const scalar_t* slew, scalar_t* values, scalar_t* clipped, int64_t* lo, int64_t* hi,
    scalar_t* alpha, scalar_t* metadata, int32_t* flags, int rows, int sizes, cudaStream_t stream) {
  if (rows) interpolate<scalar_t><<<(rows + 127) / 128, 128, 0, stream>>>(index, cap, delay, slew,
      values, clipped, lo, hi, alpha, metadata, flags, rows, sizes);
  finishMetadata<scalar_t><<<1, 1, 0, stream>>>(flags, metadata);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void bufferInterpolateBackwardLauncher(const scalar_t* index, const scalar_t* cap, const scalar_t* delay,
    const scalar_t* slew, const int64_t* lo, const int64_t* hi, const scalar_t* alpha,
    const scalar_t* grad_values, const scalar_t* grad_clipped, const scalar_t* grad_alpha,
    scalar_t* grad_index, scalar_t* grad_cap, scalar_t* grad_delay, scalar_t* grad_slew,
    int rows, int sizes, cudaStream_t stream) {
  if (!rows) return;
  interpolateBackward<scalar_t><<<(rows + 127) / 128, 128, 0, stream>>>(index, cap, delay, slew,
      lo, hi, alpha, grad_values, grad_clipped, grad_alpha, grad_index, grad_cap, grad_delay,
      grad_slew, rows, sizes);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define INSTANTIATE(scalar_t) \
  template void bufferInterpolateForwardLauncher<scalar_t>(const scalar_t*, const scalar_t*, \
      const scalar_t*, const scalar_t*, scalar_t*, scalar_t*, int64_t*, int64_t*, scalar_t*, \
      scalar_t*, int32_t*, int, int, cudaStream_t); \
  template void bufferInterpolateBackwardLauncher<scalar_t>(const scalar_t*, const scalar_t*, \
      const scalar_t*, const scalar_t*, const int64_t*, const int64_t*, const scalar_t*, \
      const scalar_t*, const scalar_t*, const scalar_t*, scalar_t*, scalar_t*, scalar_t*, \
      scalar_t*, int, int, cudaStream_t)
INSTANTIATE(float);
INSTANTIATE(double);
