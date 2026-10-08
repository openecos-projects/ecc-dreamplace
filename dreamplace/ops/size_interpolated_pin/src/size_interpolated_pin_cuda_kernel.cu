#include <stdint.h>
#include <math.h>

#include "cuda_runtime.h"

namespace {
constexpr double kDenomEps = 1e-12;

template <typename scalar_t>
__device__ __forceinline__ scalar_t clamp_value(scalar_t x, scalar_t lo, scalar_t hi) {
  return min(max(x, lo), hi);
}

__device__ __forceinline__ int32_t clamp_int(int32_t x, int32_t lo, int32_t hi) {
  return max(lo, min(x, hi));
}

template <typename scalar_t>
__device__ __forceinline__ void interpolate_one(
    scalar_t size_value,
    const scalar_t* sizes,
    const int64_t* libpin_ids,
    int32_t dim,
    const scalar_t* flat_values,
    scalar_t* interp,
    scalar_t* deriv) {
  if (dim <= 0) {
    *interp = scalar_t(0);
    *deriv = scalar_t(0);
    return;
  }
  if (dim == 1) {
    int64_t id = libpin_ids[0];
    *interp = id >= 0 ? flat_values[id] : scalar_t(0);
    *deriv = scalar_t(0);
    return;
  }
  scalar_t x_min = sizes[0];
  scalar_t x_max = sizes[dim - 1];
  scalar_t x = clamp_value(size_value, x_min, x_max);
  int32_t high = 1;
  while (high < dim && !(x < sizes[high])) {
    ++high;
  }
  high = clamp_int(high, 1, dim - 1);
  int32_t low = high - 1;
  scalar_t x0 = sizes[low];
  scalar_t x1 = sizes[high];
  int64_t id0 = libpin_ids[low];
  int64_t id1 = libpin_ids[high];
  scalar_t y0 = id0 >= 0 ? flat_values[id0] : scalar_t(0);
  scalar_t y1 = id1 >= 0 ? flat_values[id1] : scalar_t(0);
  scalar_t denom = x1 - x0;
  if (fabs((double)denom) < kDenomEps) {
    *interp = y0;
    *deriv = scalar_t(0);
    return;
  }
  scalar_t slope = (y1 - y0) / denom;
  *interp = y0 + (x - x0) * slope;
  *deriv = (size_value >= x_min && size_value <= x_max) ? slope : scalar_t(0);
}

template <typename scalar_t>
__global__ void sizeInterpolatedPinForwardKernel(
    const scalar_t* local_sizes,
    const scalar_t* vt_probs,
    const scalar_t* candidate_sizes,
    const int64_t* candidate_libpin_ids,
    const int32_t* actual_dims,
    const scalar_t* flat_pin_values,
    const scalar_t* current_values_active,
    scalar_t* output,
    int32_t pin_count,
    int32_t vt_count,
    int32_t kmax,
    int32_t q_count,
    int32_t num_flat_libpins) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  int32_t total = pin_count * q_count;
  if (idx >= total) {
    return;
  }
  int32_t p = idx % pin_count;
  int32_t q = idx / pin_count;
  scalar_t weighted_sum = scalar_t(0);
  scalar_t total_weight = scalar_t(0);
  for (int32_t vt = 0; vt < vt_count; ++vt) {
    int32_t dim = actual_dims[p * vt_count + vt];
    if (dim <= 0) {
      continue;
    }
    scalar_t weight = vt_probs[p * vt_count + vt];
    if (weight == scalar_t(0)) {
      continue;
    }
    const scalar_t* sizes = candidate_sizes + (p * vt_count + vt) * kmax;
    const int64_t* ids = candidate_libpin_ids + (p * vt_count + vt) * kmax;
    scalar_t interp = scalar_t(0);
    scalar_t deriv = scalar_t(0);
    interpolate_one(
        local_sizes[p],
        sizes,
        ids,
        dim,
        flat_pin_values + q * num_flat_libpins,
        &interp,
        &deriv);
    weighted_sum += weight * interp;
    total_weight += weight;
  }
  if (total_weight > scalar_t(0)) {
    output[q * pin_count + p] = weighted_sum / max(total_weight, static_cast<scalar_t>(kDenomEps));
  } else {
    output[q * pin_count + p] = current_values_active[q * pin_count + p];
  }
}

template <typename scalar_t>
__global__ void sizeInterpolatedPinBackwardKernel(
    const scalar_t* grad_output,
    const scalar_t* local_sizes,
    const scalar_t* vt_probs,
    const scalar_t* candidate_sizes,
    const int64_t* candidate_libpin_ids,
    const int32_t* actual_dims,
    const scalar_t* flat_pin_values,
    scalar_t* grad_local_sizes,
    scalar_t* grad_vt_probs,
    int32_t pin_count,
    int32_t vt_count,
    int32_t kmax,
    int32_t q_count,
    int32_t num_flat_libpins) {
  int32_t p = blockIdx.x * blockDim.x + threadIdx.x;
  if (p >= pin_count) {
    return;
  }
  scalar_t grad_size = scalar_t(0);
  for (int32_t q = 0; q < q_count; ++q) {
    scalar_t weighted_sum = scalar_t(0);
    scalar_t total_weight = scalar_t(0);
    for (int32_t vt = 0; vt < vt_count; ++vt) {
      int32_t dim = actual_dims[p * vt_count + vt];
      if (dim <= 0) {
        continue;
      }
      const scalar_t* sizes = candidate_sizes + (p * vt_count + vt) * kmax;
      const int64_t* ids = candidate_libpin_ids + (p * vt_count + vt) * kmax;
      scalar_t interp = scalar_t(0);
      scalar_t deriv = scalar_t(0);
      interpolate_one(
          local_sizes[p],
          sizes,
          ids,
          dim,
          flat_pin_values + q * num_flat_libpins,
          &interp,
          &deriv);
      scalar_t weight = vt_probs[p * vt_count + vt];
      weighted_sum += weight * interp;
      total_weight += weight;
    }
    if (total_weight <= scalar_t(0)) {
      continue;
    }
    scalar_t inv_total = scalar_t(1) / max(total_weight, static_cast<scalar_t>(kDenomEps));
    scalar_t g = grad_output[q * pin_count + p];
    for (int32_t vt = 0; vt < vt_count; ++vt) {
      int32_t dim = actual_dims[p * vt_count + vt];
      if (dim <= 0) {
        continue;
      }
      const scalar_t* sizes = candidate_sizes + (p * vt_count + vt) * kmax;
      const int64_t* ids = candidate_libpin_ids + (p * vt_count + vt) * kmax;
      scalar_t interp = scalar_t(0);
      scalar_t deriv = scalar_t(0);
      interpolate_one(
          local_sizes[p],
          sizes,
          ids,
          dim,
          flat_pin_values + q * num_flat_libpins,
          &interp,
          &deriv);
      scalar_t weight = vt_probs[p * vt_count + vt];
      grad_size += g * weight * deriv * inv_total;
      grad_vt_probs[p * vt_count + vt] +=
          g * (interp * total_weight - weighted_sum) * inv_total * inv_total;
    }
  }
  grad_local_sizes[p] = grad_size;
}
}  // namespace

template <typename scalar_t>
void sizeInterpolatedPinForwardCudaLauncher(
    const scalar_t* local_sizes,
    const scalar_t* vt_probs,
    const scalar_t* candidate_sizes,
    const int64_t* candidate_libpin_ids,
    const int32_t* actual_dims,
    const scalar_t* flat_pin_values,
    const scalar_t* current_values_active,
    scalar_t* output,
    int32_t pin_count,
    int32_t vt_count,
    int32_t kmax,
    int32_t q_count,
    int32_t num_flat_libpins) {
  int32_t total = pin_count * q_count;
  int32_t threads = 256;
  int32_t blocks = (total + threads - 1) / threads;
  sizeInterpolatedPinForwardKernel<scalar_t><<<blocks, threads>>>(
      local_sizes,
      vt_probs,
      candidate_sizes,
      candidate_libpin_ids,
      actual_dims,
      flat_pin_values,
      current_values_active,
      output,
      pin_count,
      vt_count,
      kmax,
      q_count,
      num_flat_libpins);
}

template <typename scalar_t>
void sizeInterpolatedPinBackwardCudaLauncher(
    const scalar_t* grad_output,
    const scalar_t* local_sizes,
    const scalar_t* vt_probs,
    const scalar_t* candidate_sizes,
    const int64_t* candidate_libpin_ids,
    const int32_t* actual_dims,
    const scalar_t* flat_pin_values,
    scalar_t* grad_local_sizes,
    scalar_t* grad_vt_probs,
    int32_t pin_count,
    int32_t vt_count,
    int32_t kmax,
    int32_t q_count,
    int32_t num_flat_libpins) {
  int32_t threads = 256;
  int32_t blocks = (pin_count + threads - 1) / threads;
  sizeInterpolatedPinBackwardKernel<scalar_t><<<blocks, threads>>>(
      grad_output,
      local_sizes,
      vt_probs,
      candidate_sizes,
      candidate_libpin_ids,
      actual_dims,
      flat_pin_values,
      grad_local_sizes,
      grad_vt_probs,
      pin_count,
      vt_count,
      kmax,
      q_count,
      num_flat_libpins);
}

template void sizeInterpolatedPinForwardCudaLauncher<float>(
    const float*,
    const float*,
    const float*,
    const int64_t*,
    const int32_t*,
    const float*,
    const float*,
    float*,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t);

template void sizeInterpolatedPinForwardCudaLauncher<double>(
    const double*,
    const double*,
    const double*,
    const int64_t*,
    const int32_t*,
    const double*,
    const double*,
    double*,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t);

template void sizeInterpolatedPinBackwardCudaLauncher<float>(
    const float*,
    const float*,
    const float*,
    const float*,
    const int64_t*,
    const int32_t*,
    const float*,
    float*,
    float*,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t);

template void sizeInterpolatedPinBackwardCudaLauncher<double>(
    const double*,
    const double*,
    const double*,
    const double*,
    const int64_t*,
    const int32_t*,
    const double*,
    double*,
    double*,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t);
