#include <stdint.h>
#include <math.h>

#include "cuda_runtime.h"

namespace {
constexpr double kDenomEps = 1e-12;

template <typename scalar_t>
__device__ __forceinline__ int32_t upper_bound_device(const scalar_t* table, int32_t size, scalar_t value) {
  int32_t first = 0;
  int32_t count = size;
  while (count > 0) {
    int32_t step = count / 2;
    int32_t it = first + step;
    if (!(value < table[it])) {
      first = it + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first;
}

__device__ __forceinline__ int32_t clamp_int(int32_t value, int32_t lo, int32_t hi) {
  return max(lo, min(value, hi));
}

template <typename scalar_t>
__device__ void lut2d_indices(
    scalar_t tin,
    scalar_t cin,
    const scalar_t* trans_table,
    const scalar_t* cap_table,
    int32_t trans_dim,
    int32_t cap_dim,
    int32_t* trans_idx_low,
    int32_t* trans_idx_high,
    int32_t* cap_idx_low,
    int32_t* cap_idx_high) {
  scalar_t trans_min = trans_table[0];
  scalar_t trans_max = trans_table[trans_dim - 1];
  scalar_t cap_min = cap_table[0];
  scalar_t cap_max = cap_table[cap_dim - 1];

  int32_t trans_idx_padded = upper_bound_device(trans_table, trans_dim, tin);
  int32_t cap_idx_padded = upper_bound_device(cap_table, cap_dim, cin);
  int32_t tih = tin < trans_min ? 1 : (tin >= trans_max ? trans_dim - 1 : trans_idx_padded);
  int32_t cih = cin < cap_min ? 1 : (cin >= cap_max ? cap_dim - 1 : cap_idx_padded);
  tih = clamp_int(tih, 1, trans_dim - 1);
  cih = clamp_int(cih, 1, cap_dim - 1);
  *trans_idx_high = tih;
  *cap_idx_high = cih;
  *trans_idx_low = tih - 1;
  *cap_idx_low = cih - 1;
}

template <typename scalar_t>
__global__ void lut2dForwardKernel(
    const scalar_t* input_trans,
    const scalar_t* output_caps,
    const scalar_t* trans_tables_batch,
    const scalar_t* cap_tables_batch,
    const scalar_t* lut_values_batch,
    const int32_t* trans_dims_actual,
    const int32_t* cap_dims_actual,
    scalar_t* output,
    int32_t batch_size,
    int32_t padded_trans_dim,
    int32_t padded_cap_dim,
    int32_t padded_lut_size) {
  int32_t batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= batch_size) {
    return;
  }
  scalar_t tin = input_trans[batch_idx];
  scalar_t cin = output_caps[batch_idx];
  int32_t trans_dim = trans_dims_actual[batch_idx];
  int32_t cap_dim = cap_dims_actual[batch_idx];
  if (trans_dim < 2 || cap_dim < 2) {
    output[batch_idx] = 0;
    return;
  }
  const scalar_t* trans_table = trans_tables_batch + batch_idx * padded_trans_dim;
  const scalar_t* cap_table = cap_tables_batch + batch_idx * padded_cap_dim;
  const scalar_t* lut_values = lut_values_batch + batch_idx * padded_lut_size;

  int32_t til, tih, cil, cih;
  lut2d_indices(tin, cin, trans_table, cap_table, trans_dim, cap_dim, &til, &tih, &cil, &cih);
  scalar_t t0 = trans_table[til];
  scalar_t t1 = trans_table[tih];
  scalar_t c0 = cap_table[cil];
  scalar_t c1 = cap_table[cih];
  int32_t idx00 = til * cap_dim + cil;
  int32_t idx01 = til * cap_dim + cih;
  int32_t idx10 = tih * cap_dim + cil;
  int32_t idx11 = tih * cap_dim + cih;
  scalar_t v00 = lut_values[idx00];
  scalar_t v01 = lut_values[idx01];
  scalar_t v10 = lut_values[idx10];
  scalar_t v11 = lut_values[idx11];
  scalar_t t_interval = t1 - t0;
  scalar_t c_interval = c1 - c0;
  bool is_t_degenerate = fabs((double)t_interval) < kDenomEps;
  bool is_c_degenerate = fabs((double)c_interval) < kDenomEps;
  if (is_t_degenerate && is_c_degenerate) {
    output[batch_idx] = v00;
  } else if (is_t_degenerate) {
    scalar_t factor = (cin - c0) / c_interval;
    output[batch_idx] = v00 + factor * (v01 - v00);
  } else if (is_c_degenerate) {
    scalar_t factor = (tin - t0) / t_interval;
    output[batch_idx] = v00 + factor * (v10 - v00);
  } else {
    scalar_t denom = t_interval * c_interval;
    scalar_t wa = (t1 - tin) * (c1 - cin);
    scalar_t wb = (t1 - tin) * (cin - c0);
    scalar_t wc = (tin - t0) * (c1 - cin);
    scalar_t wd = (tin - t0) * (cin - c0);
    output[batch_idx] = (v00 * wa + v01 * wb + v10 * wc + v11 * wd) / denom;
  }
}

template <typename scalar_t>
__global__ void lut2dBackwardKernel(
    const scalar_t* grad_output,
    const scalar_t* input_trans,
    const scalar_t* output_caps,
    const scalar_t* trans_tables_batch,
    const scalar_t* cap_tables_batch,
    const scalar_t* lut_values_batch,
    const int32_t* trans_dims_actual,
    const int32_t* cap_dims_actual,
    scalar_t* grad_input_trans,
    scalar_t* grad_output_caps,
    scalar_t* grad_lut_values_batch,
    int32_t batch_size,
    int32_t padded_trans_dim,
    int32_t padded_cap_dim,
    int32_t padded_lut_size) {
  int32_t batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= batch_size) {
    return;
  }
  scalar_t tin = input_trans[batch_idx];
  scalar_t cin = output_caps[batch_idx];
  int32_t trans_dim = trans_dims_actual[batch_idx];
  int32_t cap_dim = cap_dims_actual[batch_idx];
  if (trans_dim < 2 || cap_dim < 2) {
    return;
  }
  const scalar_t* trans_table = trans_tables_batch + batch_idx * padded_trans_dim;
  const scalar_t* cap_table = cap_tables_batch + batch_idx * padded_cap_dim;
  const scalar_t* lut_values = lut_values_batch + batch_idx * padded_lut_size;
  scalar_t* grad_lut_values = grad_lut_values_batch + batch_idx * padded_lut_size;

  int32_t til, tih, cil, cih;
  lut2d_indices(tin, cin, trans_table, cap_table, trans_dim, cap_dim, &til, &tih, &cil, &cih);
  scalar_t t0 = trans_table[til];
  scalar_t t1 = trans_table[tih];
  scalar_t c0 = cap_table[cil];
  scalar_t c1 = cap_table[cih];
  int32_t idx00 = til * cap_dim + cil;
  int32_t idx01 = til * cap_dim + cih;
  int32_t idx10 = tih * cap_dim + cil;
  int32_t idx11 = tih * cap_dim + cih;
  scalar_t v00 = lut_values[idx00];
  scalar_t v01 = lut_values[idx01];
  scalar_t v10 = lut_values[idx10];
  scalar_t v11 = lut_values[idx11];
  scalar_t t_interval = t1 - t0;
  scalar_t c_interval = c1 - c0;
  bool is_t_degenerate = fabs((double)t_interval) < kDenomEps;
  bool is_c_degenerate = fabs((double)c_interval) < kDenomEps;
  scalar_t g = grad_output[batch_idx];
  if (is_t_degenerate || is_c_degenerate) {
    return;
  }
  scalar_t denom = t_interval * c_interval;
  scalar_t inv_denom = scalar_t(1) / denom;
  scalar_t wa = (t1 - tin) * (c1 - cin);
  scalar_t wb = (t1 - tin) * (cin - c0);
  scalar_t wc = (tin - t0) * (c1 - cin);
  scalar_t wd = (tin - t0) * (cin - c0);
  grad_lut_values[idx00] = g * wa * inv_denom;
  grad_lut_values[idx01] = g * wb * inv_denom;
  grad_lut_values[idx10] = g * wc * inv_denom;
  grad_lut_values[idx11] = g * wd * inv_denom;
  grad_input_trans[batch_idx] =
      g * ((v10 - v00) * (c1 - cin) + (v11 - v01) * (cin - c0)) * inv_denom;
  grad_output_caps[batch_idx] =
      g * ((v01 - v00) * (t1 - tin) + (v11 - v10) * (tin - t0)) * inv_denom;
}

template <typename scalar_t>
__global__ void lut2dBuildCoeffKernel(
    const scalar_t* trans_tables,
    const scalar_t* cap_tables,
    const scalar_t* lut_values_batch,
    const int32_t* trans_dims_actual,
    const int32_t* cap_dims_actual,
    scalar_t* coeff,
    int32_t num_luts,
    int32_t padded_trans_dim,
    int32_t padded_cap_dim,
    int32_t padded_lut_size,
    int32_t trans_cells,
    int32_t cap_cells,
    int32_t total_cells) {
  int32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= total_cells) {
    return;
  }
  int32_t ci = idx % cap_cells;
  int32_t tmp = idx / cap_cells;
  int32_t ti = tmp % trans_cells;
  int32_t lut = tmp / trans_cells;
  if (lut >= num_luts || ti >= trans_dims_actual[lut] - 1 || ci >= cap_dims_actual[lut] - 1) {
    return;
  }
  const scalar_t* trans_table = trans_tables + lut * padded_trans_dim;
  const scalar_t* cap_table = cap_tables + lut * padded_cap_dim;
  const scalar_t* lut_values = lut_values_batch + lut * padded_lut_size;
  scalar_t t0 = trans_table[ti];
  scalar_t t1 = trans_table[ti + 1];
  scalar_t c0 = cap_table[ci];
  scalar_t c1 = cap_table[ci + 1];
  scalar_t denom = (t1 - t0) * (c1 - c0);
  if (fabs((double)denom) < kDenomEps) {
    return;
  }
  int32_t cap_dim = cap_dims_actual[lut];
  scalar_t v00 = lut_values[ti * cap_dim + ci];
  scalar_t v01 = lut_values[ti * cap_dim + ci + 1];
  scalar_t v10 = lut_values[(ti + 1) * cap_dim + ci];
  scalar_t v11 = lut_values[(ti + 1) * cap_dim + ci + 1];
  scalar_t* out = coeff + (((lut * trans_cells + ti) * cap_cells + ci) * 4);
  out[0] = (v00 - v01 - v10 + v11) / denom;
  out[1] = (-v00 * c1 + v01 * c0 + v10 * c1 - v11 * c0) / denom;
  out[2] = (-v00 * t1 + v01 * t1 + v10 * t0 - v11 * t0) / denom;
  out[3] = (v00 * t1 * c1 - v01 * t1 * c0 - v10 * t0 * c1 + v11 * t0 * c0) / denom;
}

template <typename scalar_t>
__global__ void lut2dCoeffForwardKernel(
    const scalar_t* input_trans,
    const scalar_t* output_caps,
    const scalar_t* trans_tables_batch,
    const scalar_t* cap_tables_batch,
    const scalar_t* coeff_batch,
    const int32_t* trans_dims_actual,
    const int32_t* cap_dims_actual,
    scalar_t* output,
    int32_t batch_size,
    int32_t padded_trans_dim,
    int32_t padded_cap_dim,
    int32_t trans_cells,
    int32_t cap_cells) {
  int32_t batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= batch_size) {
    return;
  }
  int32_t trans_dim = trans_dims_actual[batch_idx];
  int32_t cap_dim = cap_dims_actual[batch_idx];
  if (trans_dim < 2 || cap_dim < 2) {
    output[batch_idx] = 0;
    return;
  }
  scalar_t tin = input_trans[batch_idx];
  scalar_t cin = output_caps[batch_idx];
  int32_t til, tih, cil, cih;
  lut2d_indices(
      tin, cin, trans_tables_batch + batch_idx * padded_trans_dim,
      cap_tables_batch + batch_idx * padded_cap_dim, trans_dim, cap_dim,
      &til, &tih, &cil, &cih);
  const scalar_t* coeff = coeff_batch + (((batch_idx * trans_cells + til) * cap_cells + cil) * 4);
  output[batch_idx] = coeff[0] * tin * cin + coeff[1] * tin + coeff[2] * cin + coeff[3];
}

template <typename scalar_t>
__global__ void lut2dCoeffBackwardKernel(
    const scalar_t* grad_output,
    const scalar_t* input_trans,
    const scalar_t* output_caps,
    const scalar_t* trans_tables_batch,
    const scalar_t* cap_tables_batch,
    const scalar_t* coeff_batch,
    const int32_t* trans_dims_actual,
    const int32_t* cap_dims_actual,
    scalar_t* grad_input_trans,
    scalar_t* grad_output_caps,
    int32_t batch_size,
    int32_t padded_trans_dim,
    int32_t padded_cap_dim,
    int32_t trans_cells,
    int32_t cap_cells) {
  int32_t batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (batch_idx >= batch_size) {
    return;
  }
  int32_t trans_dim = trans_dims_actual[batch_idx];
  int32_t cap_dim = cap_dims_actual[batch_idx];
  if (trans_dim < 2 || cap_dim < 2) {
    return;
  }
  scalar_t tin = input_trans[batch_idx];
  scalar_t cin = output_caps[batch_idx];
  int32_t til, tih, cil, cih;
  lut2d_indices(
      tin, cin, trans_tables_batch + batch_idx * padded_trans_dim,
      cap_tables_batch + batch_idx * padded_cap_dim, trans_dim, cap_dim,
      &til, &tih, &cil, &cih);
  const scalar_t* coeff = coeff_batch + (((batch_idx * trans_cells + til) * cap_cells + cil) * 4);
  scalar_t g = grad_output[batch_idx];
  grad_input_trans[batch_idx] = g * (coeff[0] * cin + coeff[1]);
  grad_output_caps[batch_idx] = g * (coeff[0] * tin + coeff[2]);
}
}  // namespace

template <typename scalar_t>
void lut2dForwardCudaLauncher(
    const scalar_t* input_trans_ptr,
    const scalar_t* output_caps_ptr,
    const scalar_t* trans_tables_batch_ptr,
    const scalar_t* cap_tables_batch_ptr,
    const scalar_t* lut_values_batch_ptr,
    const int32_t* trans_dims_actual_ptr,
    const int32_t* cap_dims_actual_ptr,
    scalar_t* output_ptr,
    int32_t batch_size,
    int32_t padded_trans_dim,
    int32_t padded_cap_dim,
    int32_t padded_lut_size) {
  int thread_count = 256;
  int block_count = (batch_size + thread_count - 1) / thread_count;
  lut2dForwardKernel<scalar_t><<<block_count, thread_count>>>(
      input_trans_ptr, output_caps_ptr, trans_tables_batch_ptr, cap_tables_batch_ptr,
      lut_values_batch_ptr, trans_dims_actual_ptr, cap_dims_actual_ptr, output_ptr,
      batch_size, padded_trans_dim, padded_cap_dim, padded_lut_size);
}

template <typename scalar_t>
void lut2dBackwardCudaLauncher(
    const scalar_t* grad_output_ptr,
    const scalar_t* input_trans_ptr,
    const scalar_t* output_caps_ptr,
    const scalar_t* trans_tables_batch_ptr,
    const scalar_t* cap_tables_batch_ptr,
    const scalar_t* lut_values_batch_ptr,
    const int32_t* trans_dims_actual_ptr,
    const int32_t* cap_dims_actual_ptr,
    scalar_t* grad_input_trans_ptr,
    scalar_t* grad_output_caps_ptr,
    scalar_t* grad_lut_values_batch_ptr,
    int32_t batch_size,
    int32_t padded_trans_dim,
    int32_t padded_cap_dim,
    int32_t padded_lut_size) {
  int thread_count = 256;
  int block_count = (batch_size + thread_count - 1) / thread_count;
  lut2dBackwardKernel<scalar_t><<<block_count, thread_count>>>(
      grad_output_ptr, input_trans_ptr, output_caps_ptr, trans_tables_batch_ptr,
      cap_tables_batch_ptr, lut_values_batch_ptr, trans_dims_actual_ptr,
      cap_dims_actual_ptr, grad_input_trans_ptr, grad_output_caps_ptr,
      grad_lut_values_batch_ptr, batch_size, padded_trans_dim, padded_cap_dim,
      padded_lut_size);
}

template <typename scalar_t>
void lut2dBuildCoeffCudaLauncher(
    const scalar_t* trans_tables_batch_ptr,
    const scalar_t* cap_tables_batch_ptr,
    const scalar_t* lut_values_batch_ptr,
    const int32_t* trans_dims_actual_ptr,
    const int32_t* cap_dims_actual_ptr,
    scalar_t* coeff_ptr,
    int32_t num_luts,
    int32_t padded_trans_dim,
    int32_t padded_cap_dim,
    int32_t padded_lut_size,
    int32_t trans_cells,
    int32_t cap_cells) {
  int32_t total_cells = num_luts * trans_cells * cap_cells;
  int thread_count = 256;
  int block_count = (total_cells + thread_count - 1) / thread_count;
  lut2dBuildCoeffKernel<scalar_t><<<block_count, thread_count>>>(
      trans_tables_batch_ptr, cap_tables_batch_ptr, lut_values_batch_ptr,
      trans_dims_actual_ptr, cap_dims_actual_ptr, coeff_ptr, num_luts,
      padded_trans_dim, padded_cap_dim, padded_lut_size, trans_cells, cap_cells,
      total_cells);
}

template <typename scalar_t>
void lut2dCoeffForwardCudaLauncher(
    const scalar_t* input_trans_ptr,
    const scalar_t* output_caps_ptr,
    const scalar_t* trans_tables_batch_ptr,
    const scalar_t* cap_tables_batch_ptr,
    const scalar_t* coeff_batch_ptr,
    const int32_t* trans_dims_actual_ptr,
    const int32_t* cap_dims_actual_ptr,
    scalar_t* output_ptr,
    int32_t batch_size,
    int32_t padded_trans_dim,
    int32_t padded_cap_dim,
    int32_t trans_cells,
    int32_t cap_cells) {
  int thread_count = 256;
  int block_count = (batch_size + thread_count - 1) / thread_count;
  lut2dCoeffForwardKernel<scalar_t><<<block_count, thread_count>>>(
      input_trans_ptr, output_caps_ptr, trans_tables_batch_ptr, cap_tables_batch_ptr,
      coeff_batch_ptr, trans_dims_actual_ptr, cap_dims_actual_ptr, output_ptr,
      batch_size, padded_trans_dim, padded_cap_dim, trans_cells, cap_cells);
}

template <typename scalar_t>
void lut2dCoeffBackwardCudaLauncher(
    const scalar_t* grad_output_ptr,
    const scalar_t* input_trans_ptr,
    const scalar_t* output_caps_ptr,
    const scalar_t* trans_tables_batch_ptr,
    const scalar_t* cap_tables_batch_ptr,
    const scalar_t* coeff_batch_ptr,
    const int32_t* trans_dims_actual_ptr,
    const int32_t* cap_dims_actual_ptr,
    scalar_t* grad_input_trans_ptr,
    scalar_t* grad_output_caps_ptr,
    int32_t batch_size,
    int32_t padded_trans_dim,
    int32_t padded_cap_dim,
    int32_t trans_cells,
    int32_t cap_cells) {
  int thread_count = 256;
  int block_count = (batch_size + thread_count - 1) / thread_count;
  lut2dCoeffBackwardKernel<scalar_t><<<block_count, thread_count>>>(
      grad_output_ptr, input_trans_ptr, output_caps_ptr, trans_tables_batch_ptr,
      cap_tables_batch_ptr, coeff_batch_ptr, trans_dims_actual_ptr,
      cap_dims_actual_ptr, grad_input_trans_ptr, grad_output_caps_ptr, batch_size,
      padded_trans_dim, padded_cap_dim, trans_cells, cap_cells);
}

#define REGISTER_LUT2D_KERNEL(type)                                                \
  template void lut2dForwardCudaLauncher<type>(                                    \
      const type*, const type*, const type*, const type*, const type*,             \
      const int32_t*, const int32_t*, type*, int32_t, int32_t, int32_t, int32_t);  \
  template void lut2dBackwardCudaLauncher<type>(                                   \
      const type*, const type*, const type*, const type*, const type*,             \
      const type*, const int32_t*, const int32_t*, type*, type*, type*, int32_t,   \
      int32_t, int32_t, int32_t);

REGISTER_LUT2D_KERNEL(float);
REGISTER_LUT2D_KERNEL(double);

#define REGISTER_LUT2D_COEFF_KERNEL(type)                                          \
  template void lut2dBuildCoeffCudaLauncher<type>(                                 \
      const type*, const type*, const type*, const int32_t*, const int32_t*,       \
      type*, int32_t, int32_t, int32_t, int32_t, int32_t, int32_t);                \
  template void lut2dCoeffForwardCudaLauncher<type>(                               \
      const type*, const type*, const type*, const type*, const type*,             \
      const int32_t*, const int32_t*, type*, int32_t, int32_t, int32_t, int32_t,   \
      int32_t);                                                                    \
  template void lut2dCoeffBackwardCudaLauncher<type>(                              \
      const type*, const type*, const type*, const type*, const type*, const type*, \
      const int32_t*, const int32_t*, type*, type*, int32_t, int32_t, int32_t,     \
      int32_t, int32_t);

REGISTER_LUT2D_COEFF_KERNEL(float);
REGISTER_LUT2D_COEFF_KERNEL(double);
