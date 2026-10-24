#include <cmath>
#include <cstdint>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include "cell_aat_math.h"
#include "lut_query_cuda.cuh"

namespace {

template <typename scalar_t>
__global__ void cellForward(
    const int64_t* arcs, int columns, int count, int pins,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    const scalar_t* rc, const scalar_t* fc, TimingLutTable<scalar_t> rd,
    TimingLutTable<scalar_t> fd, TimingLutTable<scalar_t> rt, TimingLutTable<scalar_t> ft,
    scalar_t* updates, scalar_t* derivatives, bool smooth, scalar_t alpha) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) cellForwardRow<scalar_t>(i, arcs, columns, count, pins, ra, fa, rs, fs, rc, fc, rd, fd, rt, ft, updates, derivatives, smooth, alpha);
}

template <typename scalar_t>
__global__ void cellValues(
    const int64_t* arcs, int columns, int count, int pins,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    const scalar_t* rc, const scalar_t* fc, TimingLutTable<scalar_t> rd,
    TimingLutTable<scalar_t> fd, TimingLutTable<scalar_t> rt, TimingLutTable<scalar_t> ft,
    scalar_t* updates, bool smooth, scalar_t alpha) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) cellValueRow<scalar_t>(i, arcs, columns, count, pins, ra, fa, rs, fs, rc, fc, rd, fd, rt, ft, updates, smooth, alpha);
}

template <typename scalar_t>
__global__ void cellBackward(
    const int64_t* arcs, int columns, const scalar_t* jac, const scalar_t* gradient,
    scalar_t* ra, scalar_t* fa, scalar_t* rs, scalar_t* fs, scalar_t* rc, scalar_t* fc,
    int count) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  scalar_t result[6] = {0, 0, 0, 0, 0, 0};
  for (int o = 0; o < 8; ++o) {
    for (int c = 0; c < 6; ++c) result[c] += gradient[i * 8 + o] * jac[i * 48 + o * 6 + c];
  }
  const int64_t in = arcs[i * columns], out = arcs[i * columns + 1];
  atomicAdd(ra + in, result[0]); atomicAdd(fa + in, result[1]);
  atomicAdd(rs + in, result[2]); atomicAdd(fs + in, result[3]);
  atomicAdd(rc + out, result[4]); atomicAdd(fc + out, result[5]);
}

template <typename scalar_t>
__global__ void cellGroupedBackward(
    const int64_t* input_order, const int64_t* input_offsets,
    const int64_t* output_order, const int64_t* output_offsets,
    const scalar_t* jac, const scalar_t* gradient,
    scalar_t* ra, scalar_t* fa, scalar_t* rs, scalar_t* fs, scalar_t* rc, scalar_t* fc,
    int pins) {
  const int pin = blockIdx.x * blockDim.x + threadIdx.x;
  if (pin < pins) cellGroupedBackwardRow<scalar_t>(pin, input_order, input_offsets, output_order, output_offsets, jac, gradient, ra, fa, rs, fs, rc, fc, pins);
}
}  // namespace

template <typename scalar_t>
void cellAatForwardLauncher(
    const int64_t* arcs, int columns, int count, int pins,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    const scalar_t* rc, const scalar_t* fc, TimingLutTable<scalar_t> rd,
    TimingLutTable<scalar_t> fd, TimingLutTable<scalar_t> rt, TimingLutTable<scalar_t> ft,
    scalar_t* updates, scalar_t* derivatives, bool smooth, scalar_t alpha, cudaStream_t stream) {
  if (count == 0) return;
  cellForward<scalar_t><<<(count + 127) / 128, 128, 0, stream>>>(
      arcs, columns, count, pins, ra, fa, rs, fs, rc, fc, rd, fd, rt, ft,
      updates, derivatives, smooth, alpha);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void cellAatValuesLauncher(
    const int64_t* arcs, int columns, int count, int pins,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    const scalar_t* rc, const scalar_t* fc, TimingLutTable<scalar_t> rd,
    TimingLutTable<scalar_t> fd, TimingLutTable<scalar_t> rt, TimingLutTable<scalar_t> ft,
    scalar_t* updates, bool smooth, scalar_t alpha, cudaStream_t stream) {
  if (count == 0) return;
  cellValues<scalar_t><<<(count + 127) / 128, 128, 0, stream>>>(
      arcs, columns, count, pins, ra, fa, rs, fs, rc, fc, rd, fd, rt, ft,
      updates, smooth, alpha);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void cellAatBackwardLauncher(
    const int64_t* arcs, int columns, const scalar_t* jac, const scalar_t* gradient,
    scalar_t* ra, scalar_t* fa, scalar_t* rs, scalar_t* fs, scalar_t* rc, scalar_t* fc,
    int count, cudaStream_t stream) {
  if (count == 0) return;
  cellBackward<scalar_t><<<(count + 127) / 128, 128, 0, stream>>>(
      arcs, columns, jac, gradient, ra, fa, rs, fs, rc, fc, count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void cellAatGroupedBackwardLauncher(
    const int64_t* input_order, const int64_t* input_offsets,
    const int64_t* output_order, const int64_t* output_offsets,
    const scalar_t* jac, const scalar_t* gradient,
    scalar_t* ra, scalar_t* fa, scalar_t* rs, scalar_t* fs, scalar_t* rc, scalar_t* fc,
    int pins, cudaStream_t stream) {
  if (pins == 0) return;
  cellGroupedBackward<scalar_t><<<(pins + 127) / 128, 128, 0, stream>>>(
      input_order, input_offsets, output_order, output_offsets, jac, gradient,
      ra, fa, rs, fs, rc, fc, pins);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define INSTANTIATE(scalar_t) \
  template void cellAatForwardLauncher<scalar_t>(const int64_t*, int, int, int, \
      const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*, \
      TimingLutTable<scalar_t>, TimingLutTable<scalar_t>, TimingLutTable<scalar_t>, TimingLutTable<scalar_t>, \
      scalar_t*, scalar_t*, bool, scalar_t, cudaStream_t); \
  template void cellAatValuesLauncher<scalar_t>(const int64_t*, int, int, int, \
      const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*, \
      TimingLutTable<scalar_t>, TimingLutTable<scalar_t>, TimingLutTable<scalar_t>, TimingLutTable<scalar_t>, \
      scalar_t*, bool, scalar_t, cudaStream_t); \
  template void cellAatBackwardLauncher<scalar_t>(const int64_t*, int, const scalar_t*, const scalar_t*, \
      scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, int, cudaStream_t); \
  template void cellAatGroupedBackwardLauncher<scalar_t>(const int64_t*, const int64_t*, \
      const int64_t*, const int64_t*, const scalar_t*, const scalar_t*, scalar_t*, scalar_t*, \
      scalar_t*, scalar_t*, scalar_t*, scalar_t*, int, cudaStream_t)
INSTANTIATE(float);
INSTANTIATE(double);
