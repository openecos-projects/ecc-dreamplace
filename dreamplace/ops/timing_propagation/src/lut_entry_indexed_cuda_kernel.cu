#include <cmath>
#include <cstdint>

#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include "lut_query_cuda.cuh"

namespace {
template <typename scalar_t>
__global__ void lookupIndexed(
    const scalar_t* trans, const scalar_t* caps, const int64_t* indices,
    const scalar_t* trans_axes, const scalar_t* cap_axes, const scalar_t* values,
    const int64_t* trans_dims, const int64_t* cap_dims,
    scalar_t* output, scalar_t* grad_trans, scalar_t* grad_caps,
    int batch, int rows, int trans_stride, int cap_stride, int value_stride,
    bool extrapolate_1d) {
  const int q = blockIdx.x * blockDim.x + threadIdx.x;
  if (q >= batch) return;
  const TimingLutTable<scalar_t> table{
      values, trans_axes, cap_axes, trans_dims, cap_dims,
      rows, value_stride, trans_stride, cap_stride};
  const auto query = timingLutQuery(table, indices[q], trans[q], caps[q], extrapolate_1d);
  output[q] = query.value;
  grad_trans[q] = query.dt;
  grad_caps[q] = query.dc;
}
}  // namespace

template <typename scalar_t>
void lutIndexedCudaLauncher(
    const scalar_t* trans, const scalar_t* caps, const int64_t* indices,
    const scalar_t* trans_axes, const scalar_t* cap_axes, const scalar_t* values,
    const int64_t* trans_dims, const int64_t* cap_dims,
    scalar_t* output, scalar_t* grad_trans, scalar_t* grad_caps,
    int batch, int rows, int trans_stride, int cap_stride, int value_stride,
    bool extrapolate_1d, cudaStream_t stream) {
  if (batch == 0) return;
  lookupIndexed<scalar_t><<<(batch + 127) / 128, 128, 0, stream>>>(
      trans, caps, indices, trans_axes, cap_axes, values, trans_dims, cap_dims,
      output, grad_trans, grad_caps, batch, rows, trans_stride, cap_stride, value_stride, extrapolate_1d);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define INSTANTIATE(scalar_t) \
  template void lutIndexedCudaLauncher<scalar_t>( \
      const scalar_t*, const scalar_t*, const int64_t*, const scalar_t*, \
      const scalar_t*, const scalar_t*, const int64_t*, const int64_t*, \
      scalar_t*, scalar_t*, scalar_t*, int, int, int, int, int, bool, cudaStream_t)
INSTANTIATE(float);
INSTANTIATE(double);
