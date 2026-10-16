#include <cmath>
#include <cstdint>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include "net_aat_math.h"

namespace {

template <typename scalar_t>
__global__ void netForward(const int64_t* arcs, int columns, int count, int pins,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    const scalar_t* rd, const scalar_t* fd, const scalar_t* ri, const scalar_t* fi,
    scalar_t* updates, scalar_t* jac, scalar_t eps) {
  const int arc = blockIdx.x * blockDim.x + threadIdx.x;
  if (arc < count) netForwardRow<scalar_t>(arc, arcs, columns, count, pins, ra, fa, rs, fs, rd, fd, ri, fi, updates, jac, eps);
}

template <typename scalar_t>
__global__ void netBackward(const int64_t* input_order, const int64_t* input_offsets,
    const int64_t* output_order, const int64_t* output_offsets,
    const scalar_t* jac, const scalar_t* gradient,
    scalar_t* ra, scalar_t* fa, scalar_t* rs, scalar_t* fs,
    scalar_t* rd, scalar_t* fd, scalar_t* ri, scalar_t* fi, int pins,
    bool rise_slew_active, bool fall_slew_active) {
  const int pin = blockIdx.x * blockDim.x + threadIdx.x;
  if (pin < pins) netBackwardRow<scalar_t>(pin, input_order, input_offsets, output_order, output_offsets, jac, gradient, ra, fa, rs, fs, rd, fd, ri, fi, pins, rise_slew_active, fall_slew_active);
}
}  // namespace

template <typename scalar_t>
void netAatForwardLauncher(const int64_t* arcs, int columns, int count, int pins,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    const scalar_t* rd, const scalar_t* fd, const scalar_t* ri, const scalar_t* fi,
    scalar_t* updates, scalar_t* jac, scalar_t eps, cudaStream_t stream) {
  if (count == 0) return;
  netForward<scalar_t><<<(count + 127) / 128, 128, 0, stream>>>(arcs, columns, count, pins,
      ra, fa, rs, fs, rd, fd, ri, fi, updates, jac, eps);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void netAatBackwardLauncher(const int64_t* input_order, const int64_t* input_offsets,
    const int64_t* output_order, const int64_t* output_offsets,
    const scalar_t* jac, const scalar_t* gradient,
    scalar_t* ra, scalar_t* fa, scalar_t* rs, scalar_t* fs,
    scalar_t* rd, scalar_t* fd, scalar_t* ri, scalar_t* fi, int pins,
    bool rise_slew_active, bool fall_slew_active, cudaStream_t stream) {
  if (pins == 0) return;
  netBackward<scalar_t><<<(pins + 127) / 128, 128, 0, stream>>>(input_order, input_offsets,
      output_order, output_offsets, jac, gradient, ra, fa, rs, fs, rd, fd, ri, fi, pins,
      rise_slew_active, fall_slew_active);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define INSTANTIATE(scalar_t) \
  template void netAatForwardLauncher<scalar_t>(const int64_t*, int, int, int, \
      const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*, \
      const scalar_t*, const scalar_t*, const scalar_t*, scalar_t*, scalar_t*, scalar_t, cudaStream_t); \
  template void netAatBackwardLauncher<scalar_t>(const int64_t*, const int64_t*, const int64_t*, \
      const int64_t*, const scalar_t*, const scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, \
      scalar_t*, scalar_t*, scalar_t*, scalar_t*, int, bool, bool, cudaStream_t)
INSTANTIATE(float);
INSTANTIATE(double);
