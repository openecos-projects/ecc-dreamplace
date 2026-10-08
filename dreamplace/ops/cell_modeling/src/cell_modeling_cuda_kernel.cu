#include <stdint.h>

#include "cuda_runtime.h"
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

namespace {
constexpr double kSizeEps = 1e-9;
constexpr int64_t kLutBoundaryClamp = 0;
constexpr int64_t kLutBoundaryExtrapolate = 1;

__device__ __forceinline__ int64_t clamp_index_cuda(int64_t value, int64_t max_value) {
  if (value < 0) {
    return 0;
  }
  if (value > max_value) {
    return max_value;
  }
  return value;
}

template <typename T>
__global__ void cellModelingPoly12ForwardKernel(
    const T* coeff,
    const int64_t* main_id,
    const int64_t* arc_offset,
    const T* vt,
    const T* size,
    const T* input_slew,
    const T* out_cap,
    int64_t num_items,
    int64_t num_main,
    int64_t num_arc,
    T* output) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  int64_t mt = clamp_index_cuda(main_id[i], num_main - 1);
  int64_t ao = clamp_index_cuda(arc_offset[i], num_arc - 1);
  const T* w = coeff + ((mt * num_arc + ao) * 12);
  T inv_size = T(1) / (size[i] + T(kSizeEps));
  T sl = input_slew[i];
  T ca = out_cap[i];
  T v = vt[i];
  output[i] = w[0] * sl + w[1] * ca + w[2] * inv_size +
              w[3] * ca * inv_size + w[4] * v +
              w[5] * sl * ca + w[6] * sl * inv_size +
              w[7] * sl * v + w[8] * ca * v +
              w[9] * inv_size * v + w[10] * inv_size * inv_size +
              w[11];
}

template <typename T>
__global__ void cellModelingPoly12BackwardKernel(
    const T* grad_output,
    const T* coeff,
    const int64_t* main_id,
    const int64_t* arc_offset,
    const T* vt,
    const T* size,
    const T* input_slew,
    const T* out_cap,
    int64_t num_items,
    int64_t num_main,
    int64_t num_arc,
    T* grad_vt,
    T* grad_size,
    T* grad_slew,
    T* grad_cap) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  int64_t mt = clamp_index_cuda(main_id[i], num_main - 1);
  int64_t ao = clamp_index_cuda(arc_offset[i], num_arc - 1);
  const T* w = coeff + ((mt * num_arc + ao) * 12);
  T g = grad_output[i];
  T inv_size = T(1) / (size[i] + T(kSizeEps));
  T d_inv_d_size = -inv_size * inv_size;
  T sl = input_slew[i];
  T ca = out_cap[i];
  T v = vt[i];

  T d_value_d_vt = w[4] + w[7] * sl + w[8] * ca + w[9] * inv_size;
  T d_value_d_slew = w[0] + w[5] * ca + w[6] * inv_size + w[7] * v;
  T d_value_d_cap = w[1] + w[3] * inv_size + w[5] * sl + w[8] * v;
  T d_value_d_inv = w[2] + w[3] * ca + w[6] * sl + w[9] * v +
                    T(2) * w[10] * inv_size;

  grad_vt[i] = g * d_value_d_vt;
  grad_size[i] = g * d_value_d_inv * d_inv_d_size;
  grad_slew[i] = g * d_value_d_slew;
  grad_cap[i] = g * d_value_d_cap;
}

template <typename T>
__device__ __forceinline__ int64_t upper_bound_cuda(const T* table, int64_t size, T value) {
  int64_t first = 0;
  int64_t count = size;
  while (count > 0) {
    int64_t step = count / 2;
    int64_t it = first + step;
    if (!(value < table[it])) {
      first = it + 1;
      count -= step + 1;
    } else {
      count = step;
    }
  }
  return first;
}

template <typename T>
__device__ void lut2d_value_and_grads_cuda(
    T input_slew,
    T out_cap,
    const T* trans_table,
    const T* cap_table,
    const T* values,
    int64_t trans_dim,
    int64_t cap_dim,
    T* value,
    T* d_slew,
    T* d_cap,
    T* weights,
    int64_t* corner_indices,
    int64_t lut_boundary_mode) {
  constexpr double kDenomEps = 1e-12;
  if (trans_dim < 2 || cap_dim < 2) {
    *value = T(0);
    *d_slew = T(0);
    *d_cap = T(0);
    for (int k = 0; k < 4; ++k) {
      weights[k] = T(0);
      corner_indices[k] = 0;
    }
    return;
  }
  T trans_min = trans_table[0];
  T trans_max = trans_table[trans_dim - 1];
  T cap_min = cap_table[0];
  T cap_max = cap_table[cap_dim - 1];
  T x = input_slew;
  T y = out_cap;
  if (lut_boundary_mode == kLutBoundaryClamp) {
    x = min(max(input_slew, trans_min), trans_max);
    y = min(max(out_cap, cap_min), cap_max);
  }
  int64_t trans_high = upper_bound_cuda(trans_table, trans_dim, x);
  int64_t cap_high = upper_bound_cuda(cap_table, cap_dim, y);
  if (lut_boundary_mode == kLutBoundaryExtrapolate) {
    if (x < trans_min) {
      trans_high = 1;
    } else if (x >= trans_max) {
      trans_high = trans_dim - 1;
    }
    if (y < cap_min) {
      cap_high = 1;
    } else if (y >= cap_max) {
      cap_high = cap_dim - 1;
    }
  }
  trans_high = max((int64_t)1, min(trans_high, trans_dim - 1));
  cap_high = max((int64_t)1, min(cap_high, cap_dim - 1));
  int64_t trans_low = trans_high - 1;
  int64_t cap_low = cap_high - 1;
  T t0 = trans_table[trans_low];
  T t1 = trans_table[trans_high];
  T c0 = cap_table[cap_low];
  T c1 = cap_table[cap_high];
  int64_t idx00 = trans_low * cap_dim + cap_low;
  int64_t idx01 = trans_low * cap_dim + cap_high;
  int64_t idx10 = trans_high * cap_dim + cap_low;
  int64_t idx11 = trans_high * cap_dim + cap_high;
  T v00 = values[idx00];
  T v01 = values[idx01];
  T v10 = values[idx10];
  T v11 = values[idx11];
  T t_interval = t1 - t0;
  T c_interval = c1 - c0;
  bool t_degenerate = fabs((double)t_interval) < kDenomEps;
  bool c_degenerate = fabs((double)c_interval) < kDenomEps;
  corner_indices[0] = idx00;
  corner_indices[1] = idx01;
  corner_indices[2] = idx10;
  corner_indices[3] = idx11;
  if (t_degenerate && c_degenerate) {
    *value = v00;
    *d_slew = T(0);
    *d_cap = T(0);
    weights[0] = T(1);
    weights[1] = weights[2] = weights[3] = T(0);
    return;
  }
  if (t_degenerate) {
    T factor = (y - c0) / c_interval;
    *value = v00 + factor * (v01 - v00);
    *d_slew = T(0);
    *d_cap = (v01 - v00) / c_interval;
    weights[0] = T(1) - factor;
    weights[1] = factor;
    weights[2] = weights[3] = T(0);
    return;
  }
  if (c_degenerate) {
    T factor = (x - t0) / t_interval;
    *value = v00 + factor * (v10 - v00);
    *d_slew = (v10 - v00) / t_interval;
    *d_cap = T(0);
    weights[0] = T(1) - factor;
    weights[2] = factor;
    weights[1] = weights[3] = T(0);
    return;
  }
  T inv_den = T(1) / (t_interval * c_interval);
  T wa = (t1 - x) * (c1 - y);
  T wb = (t1 - x) * (y - c0);
  T wc = (x - t0) * (c1 - y);
  T wd = (x - t0) * (y - c0);
  weights[0] = wa * inv_den;
  weights[1] = wb * inv_den;
  weights[2] = wc * inv_den;
  weights[3] = wd * inv_den;
  *value = v00 * weights[0] + v01 * weights[1] + v10 * weights[2] + v11 * weights[3];
  *d_slew = ((v10 - v00) * (c1 - y) + (v11 - v01) * (y - c0)) * inv_den;
  *d_cap = ((v01 - v00) * (t1 - x) + (v11 - v10) * (x - t0)) * inv_den;
}

template <typename T>
__global__ void cellModelingPiecewiseSizeForwardKernel(
    const T* size_table,
    const int64_t* arc_table,
    const int64_t* size_count,
    const T* trans_tables,
    const T* cap_tables,
    const T* lut_values,
    const int64_t* trans_dims,
    const int64_t* cap_dims,
    const T* size,
    const T* slew,
    const T* cap,
    int64_t num_items,
    int64_t max_sizes,
    int64_t num_arcs,
    int64_t trans_stride,
    int64_t cap_stride,
    int64_t value_stride,
    int64_t lut_boundary_mode,
    T* output) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  int64_t count = size_count[i];
  if (count <= 0) {
    output[i] = T(0);
    return;
  }
  const T* row_sizes = size_table + i * max_sizes;
  const int64_t* row_arcs = arc_table + i * max_sizes;
  auto eval_arc = [&](int64_t arc_idx) -> T {
    if (arc_idx < 0 || arc_idx >= num_arcs) {
      return T(0);
    }
    T value, dslew, dcap, weights[4];
    int64_t corners[4];
    lut2d_value_and_grads_cuda(
        slew[i], cap[i], trans_tables + arc_idx * trans_stride,
        cap_tables + arc_idx * cap_stride, lut_values + arc_idx * value_stride,
        trans_dims[arc_idx], cap_dims[arc_idx], &value, &dslew, &dcap, weights, corners,
        lut_boundary_mode);
    return value;
  };
  if (count <= 1) {
    output[i] = eval_arc(row_arcs[0]);
    return;
  }
  int64_t max_idx = count - 1;
  T size_min = row_sizes[0];
  T size_max = row_sizes[max_idx];
  T size_clamped = min(max(size[i], size_min), size_max);
  int64_t high = upper_bound_cuda(row_sizes, count, size_clamped);
  high = max((int64_t)1, min(high, max_idx));
  int64_t low = high - 1;
  T size_low = row_sizes[low];
  T size_high = row_sizes[high];
  T value_low = eval_arc(row_arcs[low]);
  T value_high = row_arcs[low] == row_arcs[high] ? value_low : eval_arc(row_arcs[high]);
  T denom = size_high - size_low;
  if (!isfinite((double)size_low) || !isfinite((double)size_high) || !isfinite((double)denom) ||
      fabs((double)denom) < 1e-12) {
    output[i] = value_low;
  } else {
    T factor = (size_clamped - size_low) / denom;
    output[i] = value_low + factor * (value_high - value_low);
  }
}

template <typename T>
__global__ void cellModelingPiecewiseSizeBackwardKernel(
    const T* grad_output,
    const T* size_table,
    const int64_t* arc_table,
    const int64_t* size_count,
    const T* trans_tables,
    const T* cap_tables,
    const T* lut_values,
    const int64_t* trans_dims,
    const int64_t* cap_dims,
    const T* size,
    const T* slew,
    const T* cap,
    int64_t num_items,
    int64_t max_sizes,
    int64_t num_arcs,
    int64_t trans_stride,
    int64_t cap_stride,
    int64_t value_stride,
    int64_t lut_boundary_mode,
    T* grad_size,
    T* grad_slew,
    T* grad_cap,
    T* grad_lut_values) {
  int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= num_items) {
    return;
  }
  int64_t count = size_count[i];
  if (count <= 0) {
    return;
  }
  const T* row_sizes = size_table + i * max_sizes;
  const int64_t* row_arcs = arc_table + i * max_sizes;
  auto eval_arc = [&](int64_t arc_idx, T* value, T* dslew, T* dcap, T* weights, int64_t* corners) {
    if (arc_idx < 0 || arc_idx >= num_arcs) {
      *value = *dslew = *dcap = T(0);
      for (int k = 0; k < 4; ++k) {
        weights[k] = T(0);
        corners[k] = 0;
      }
      return;
    }
    lut2d_value_and_grads_cuda(
        slew[i], cap[i], trans_tables + arc_idx * trans_stride,
        cap_tables + arc_idx * cap_stride, lut_values + arc_idx * value_stride,
        trans_dims[arc_idx], cap_dims[arc_idx], value, dslew, dcap, weights, corners,
        lut_boundary_mode);
  };
  T g = grad_output[i];
  if (count <= 1) {
    int64_t arc = row_arcs[0];
    T value, dslew, dcap, weights[4];
    int64_t corners[4];
    eval_arc(arc, &value, &dslew, &dcap, weights, corners);
    grad_slew[i] = g * dslew;
    grad_cap[i] = g * dcap;
    if (arc >= 0 && arc < num_arcs) {
      for (int k = 0; k < 4; ++k) {
        atomicAdd(grad_lut_values + arc * value_stride + corners[k], g * weights[k]);
      }
    }
    return;
  }
  int64_t max_idx = count - 1;
  T size_min = row_sizes[0];
  T size_max = row_sizes[max_idx];
  T size_clamped = min(max(size[i], size_min), size_max);
  int64_t high = upper_bound_cuda(row_sizes, count, size_clamped);
  high = max((int64_t)1, min(high, max_idx));
  int64_t low = high - 1;
  T size_low = row_sizes[low];
  T size_high = row_sizes[high];
  int64_t arc_low = row_arcs[low];
  int64_t arc_high = row_arcs[high];
  T value_low, dslew_low, dcap_low, weights_low[4];
  T value_high, dslew_high, dcap_high, weights_high[4];
  int64_t corners_low[4], corners_high[4];
  eval_arc(arc_low, &value_low, &dslew_low, &dcap_low, weights_low, corners_low);
  if (arc_low == arc_high) {
    value_high = value_low;
    dslew_high = dslew_low;
    dcap_high = dcap_low;
    for (int k = 0; k < 4; ++k) {
      weights_high[k] = weights_low[k];
      corners_high[k] = corners_low[k];
    }
  } else {
    eval_arc(arc_high, &value_high, &dslew_high, &dcap_high, weights_high, corners_high);
  }
  T denom = size_high - size_low;
  if (!isfinite((double)size_low) || !isfinite((double)size_high) || !isfinite((double)denom) ||
      fabs((double)denom) < 1e-12) {
    grad_slew[i] = g * dslew_low;
    grad_cap[i] = g * dcap_low;
    if (arc_low >= 0 && arc_low < num_arcs) {
      for (int k = 0; k < 4; ++k) {
        atomicAdd(grad_lut_values + arc_low * value_stride + corners_low[k], g * weights_low[k]);
      }
    }
    return;
  }
  T factor = (size_clamped - size_low) / denom;
  bool inside = size[i] >= size_min && size[i] <= size_max;
  grad_size[i] = inside ? g * (value_high - value_low) / denom : T(0);
  grad_slew[i] = g * ((T(1) - factor) * dslew_low + factor * dslew_high);
  grad_cap[i] = g * ((T(1) - factor) * dcap_low + factor * dcap_high);
  if (arc_low >= 0 && arc_low < num_arcs) {
    for (int k = 0; k < 4; ++k) {
      atomicAdd(grad_lut_values + arc_low * value_stride + corners_low[k],
                g * (T(1) - factor) * weights_low[k]);
    }
  }
  if (arc_high >= 0 && arc_high < num_arcs) {
    for (int k = 0; k < 4; ++k) {
      atomicAdd(grad_lut_values + arc_high * value_stride + corners_high[k],
                g * factor * weights_high[k]);
    }
  }
}
}  // namespace

template <typename T>
int cellModelingPoly12ForwardCudaLauncher(
    const T* coeff,
    const int64_t* main_id,
    const int64_t* arc_offset,
    const T* vt,
    const T* size,
    const T* input_slew,
    const T* out_cap,
    int64_t num_items,
    int64_t num_main,
    int64_t num_arc,
    T* output) {
  int thread_count = 256;
  int block_count = (num_items + thread_count - 1) / thread_count;
  cellModelingPoly12ForwardKernel<T><<<block_count, thread_count>>>(
      coeff, main_id, arc_offset, vt, size, input_slew, out_cap, num_items,
      num_main, num_arc, output);
  return 0;
}

template <typename T>
int cellModelingPoly12BackwardCudaLauncher(
    const T* grad_output,
    const T* coeff,
    const int64_t* main_id,
    const int64_t* arc_offset,
    const T* vt,
    const T* size,
    const T* input_slew,
    const T* out_cap,
    int64_t num_items,
    int64_t num_main,
    int64_t num_arc,
    T* grad_vt,
    T* grad_size,
    T* grad_slew,
    T* grad_cap) {
  int thread_count = 256;
  int block_count = (num_items + thread_count - 1) / thread_count;
  cellModelingPoly12BackwardKernel<T><<<block_count, thread_count>>>(
      grad_output, coeff, main_id, arc_offset, vt, size, input_slew, out_cap,
      num_items, num_main, num_arc, grad_vt, grad_size, grad_slew, grad_cap);
  return 0;
}

template <typename T>
int cellModelingPiecewiseSizeForwardCudaLauncher(
    const T* size_table,
    const int64_t* arc_table,
    const int64_t* size_count,
    const T* trans_tables,
    const T* cap_tables,
    const T* lut_values,
    const int64_t* trans_dims,
    const int64_t* cap_dims,
    const T* size,
    const T* slew,
    const T* cap,
    int64_t num_items,
    int64_t max_sizes,
    int64_t num_arcs,
    int64_t trans_stride,
    int64_t cap_stride,
    int64_t value_stride,
    int64_t lut_boundary_mode,
    T* output) {
  int thread_count = 256;
  int block_count = (num_items + thread_count - 1) / thread_count;
  cellModelingPiecewiseSizeForwardKernel<T><<<block_count, thread_count>>>(
      size_table, arc_table, size_count, trans_tables, cap_tables, lut_values,
      trans_dims, cap_dims, size, slew, cap, num_items, max_sizes, num_arcs,
      trans_stride, cap_stride, value_stride, lut_boundary_mode, output);
  return 0;
}

template <typename T>
int cellModelingPiecewiseSizeBackwardCudaLauncher(
    const T* grad_output,
    const T* size_table,
    const int64_t* arc_table,
    const int64_t* size_count,
    const T* trans_tables,
    const T* cap_tables,
    const T* lut_values,
    const int64_t* trans_dims,
    const int64_t* cap_dims,
    const T* size,
    const T* slew,
    const T* cap,
    int64_t num_items,
    int64_t max_sizes,
    int64_t num_arcs,
    int64_t trans_stride,
    int64_t cap_stride,
    int64_t value_stride,
    int64_t lut_boundary_mode,
    T* grad_size,
    T* grad_slew,
    T* grad_cap,
    T* grad_lut_values) {
  int thread_count = 256;
  int block_count = (num_items + thread_count - 1) / thread_count;
  cellModelingPiecewiseSizeBackwardKernel<T><<<block_count, thread_count>>>(
      grad_output, size_table, arc_table, size_count, trans_tables, cap_tables,
      lut_values, trans_dims, cap_dims, size, slew, cap, num_items, max_sizes,
      num_arcs, trans_stride, cap_stride, value_stride, lut_boundary_mode, grad_size,
      grad_slew, grad_cap, grad_lut_values);
  return 0;
}

#define REGISTER_KERNEL_LAUNCHER(type)                                           \
  template int cellModelingPoly12ForwardCudaLauncher<type>(                      \
      const type* coeff, const int64_t* main_id, const int64_t* arc_offset,      \
      const type* vt, const type* size, const type* input_slew,                  \
      const type* out_cap, int64_t num_items, int64_t num_main,                  \
      int64_t num_arc, type* output);                                            \
  template int cellModelingPoly12BackwardCudaLauncher<type>(                     \
      const type* grad_output, const type* coeff, const int64_t* main_id,        \
      const int64_t* arc_offset, const type* vt, const type* size,               \
      const type* input_slew, const type* out_cap, int64_t num_items,            \
      int64_t num_main, int64_t num_arc, type* grad_vt, type* grad_size,         \
      type* grad_slew, type* grad_cap);                                          \
  template int cellModelingPiecewiseSizeForwardCudaLauncher<type>(               \
      const type* size_table, const int64_t* arc_table,                          \
      const int64_t* size_count, const type* trans_tables,                       \
      const type* cap_tables, const type* lut_values,                            \
      const int64_t* trans_dims, const int64_t* cap_dims, const type* size,      \
      const type* slew, const type* cap, int64_t num_items, int64_t max_sizes,   \
      int64_t num_arcs, int64_t trans_stride, int64_t cap_stride,                \
      int64_t value_stride, int64_t lut_boundary_mode, type* output);            \
  template int cellModelingPiecewiseSizeBackwardCudaLauncher<type>(              \
      const type* grad_output, const type* size_table, const int64_t* arc_table, \
      const int64_t* size_count, const type* trans_tables,                       \
      const type* cap_tables, const type* lut_values,                            \
      const int64_t* trans_dims, const int64_t* cap_dims, const type* size,      \
      const type* slew, const type* cap, int64_t num_items, int64_t max_sizes,   \
      int64_t num_arcs, int64_t trans_stride, int64_t cap_stride,                \
      int64_t value_stride, int64_t lut_boundary_mode, type* grad_size,          \
      type* grad_slew, type* grad_cap, type* grad_lut_values);

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
