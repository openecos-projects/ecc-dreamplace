#include <algorithm>

#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

namespace {
constexpr double kSizeEps = 1e-9;
constexpr int64_t kLutBoundaryClamp = 0;
constexpr int64_t kLutBoundaryExtrapolate = 1;

template <typename T>
inline int64_t clamp_index(int64_t value, int64_t max_value) {
  return std::max<int64_t>(0, std::min<int64_t>(value, max_value));
}

template <typename T>
void poly12_forward_cpu_kernel(
    const T* coeff,
    const int64_t* main_id,
    const int64_t* arc_offset,
    const T* vt,
    const T* size,
    const T* slew,
    const T* cap,
    int64_t num_items,
    int64_t num_main,
    int64_t num_arc,
    int num_threads,
    T* output) {
#pragma omp parallel for num_threads(num_threads)
  for (int64_t i = 0; i < num_items; ++i) {
    int64_t mt = clamp_index<T>(main_id[i], num_main - 1);
    int64_t ao = clamp_index<T>(arc_offset[i], num_arc - 1);
    const T* w = coeff + ((mt * num_arc + ao) * 12);
    T s = size[i];
    T inv_size = T(1) / (s + T(kSizeEps));
    T sl = slew[i];
    T ca = cap[i];
    T v = vt[i];
    output[i] = w[0] * sl + w[1] * ca + w[2] * inv_size +
                w[3] * ca * inv_size + w[4] * v +
                w[5] * sl * ca + w[6] * sl * inv_size +
                w[7] * sl * v + w[8] * ca * v +
                w[9] * inv_size * v + w[10] * inv_size * inv_size +
                w[11];
  }
}

template <typename T>
void poly12_backward_cpu_kernel(
    const T* grad_output,
    const T* coeff,
    const int64_t* main_id,
    const int64_t* arc_offset,
    const T* vt,
    const T* size,
    const T* slew,
    const T* cap,
    int64_t num_items,
    int64_t num_main,
    int64_t num_arc,
    int num_threads,
    T* grad_vt,
    T* grad_size,
    T* grad_slew,
    T* grad_cap) {
#pragma omp parallel for num_threads(num_threads)
  for (int64_t i = 0; i < num_items; ++i) {
    int64_t mt = clamp_index<T>(main_id[i], num_main - 1);
    int64_t ao = clamp_index<T>(arc_offset[i], num_arc - 1);
    const T* w = coeff + ((mt * num_arc + ao) * 12);
    T g = grad_output[i];
    T s = size[i];
    T inv_size = T(1) / (s + T(kSizeEps));
    T d_inv_d_size = -inv_size * inv_size;
    T sl = slew[i];
    T ca = cap[i];
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
}

template <typename T>
inline int64_t upper_bound_cpu(const T* table, int64_t size, T value) {
  return std::upper_bound(table, table + size, value) - table;
}

template <typename T>
inline void lut2d_value_and_grads(
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
    for (int i = 0; i < 4; ++i) {
      weights[i] = T(0);
      corner_indices[i] = 0;
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
    x = std::min(std::max(input_slew, trans_min), trans_max);
    y = std::min(std::max(out_cap, cap_min), cap_max);
  }
  int64_t trans_high = upper_bound_cpu(trans_table, trans_dim, x);
  int64_t cap_high = upper_bound_cpu(cap_table, cap_dim, y);
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
  trans_high = std::max<int64_t>(1, std::min<int64_t>(trans_high, trans_dim - 1));
  cap_high = std::max<int64_t>(1, std::min<int64_t>(cap_high, cap_dim - 1));
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
  bool t_degenerate = std::abs(t_interval) < kDenomEps;
  bool c_degenerate = std::abs(c_interval) < kDenomEps;
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
void piecewise_size_forward_cpu_kernel(
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
    int num_threads,
    int64_t lut_boundary_mode,
    T* output) {
#pragma omp parallel for num_threads(num_threads)
  for (int64_t i = 0; i < num_items; ++i) {
    int64_t count = size_count[i];
    if (count <= 0) {
      output[i] = T(0);
      continue;
    }
    const T* row_sizes = size_table + i * max_sizes;
    const int64_t* row_arcs = arc_table + i * max_sizes;
    auto eval_arc = [&](int64_t arc_idx) {
      T value = T(0), dslew = T(0), dcap = T(0), weights[4];
      int64_t corners[4];
      if (arc_idx < 0 || arc_idx >= num_arcs) {
        return T(0);
      }
      lut2d_value_and_grads<T>(
          slew[i], cap[i], trans_tables + arc_idx * trans_stride,
          cap_tables + arc_idx * cap_stride, lut_values + arc_idx * value_stride,
          trans_dims[arc_idx], cap_dims[arc_idx], &value, &dslew, &dcap, weights, corners,
          lut_boundary_mode);
      return value;
    };
    if (count <= 1) {
      output[i] = eval_arc(row_arcs[0]);
      continue;
    }
    int64_t max_idx = count - 1;
    T size_min = row_sizes[0];
    T size_max = row_sizes[max_idx];
    T size_clamped = std::min(std::max(size[i], size_min), size_max);
    int64_t high = upper_bound_cpu(row_sizes, count, size_clamped);
    high = std::max<int64_t>(1, std::min<int64_t>(high, max_idx));
    int64_t low = high - 1;
    T size_low = row_sizes[low];
    T size_high = row_sizes[high];
    T value_low = eval_arc(row_arcs[low]);
    T value_high = row_arcs[low] == row_arcs[high] ? value_low : eval_arc(row_arcs[high]);
    T denom = size_high - size_low;
    if (!std::isfinite(size_low) || !std::isfinite(size_high) || !std::isfinite(denom) ||
        std::abs(denom) < T(1e-12)) {
      output[i] = value_low;
    } else {
      T factor = (size_clamped - size_low) / denom;
      output[i] = value_low + factor * (value_high - value_low);
    }
  }
}

template <typename T>
void piecewise_size_backward_cpu_kernel(
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
    int num_threads,
    int64_t lut_boundary_mode,
    T* grad_size,
    T* grad_slew,
    T* grad_cap,
    T* grad_lut_values) {
#pragma omp parallel for num_threads(num_threads)
  for (int64_t i = 0; i < num_items; ++i) {
    int64_t count = size_count[i];
    if (count <= 0) {
      continue;
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
      lut2d_value_and_grads<T>(
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
#pragma omp atomic
          grad_lut_values[arc * value_stride + corners[k]] += g * weights[k];
        }
      }
      continue;
    }
    int64_t max_idx = count - 1;
    T size_min = row_sizes[0];
    T size_max = row_sizes[max_idx];
    T size_clamped = std::min(std::max(size[i], size_min), size_max);
    int64_t high = upper_bound_cpu(row_sizes, count, size_clamped);
    high = std::max<int64_t>(1, std::min<int64_t>(high, max_idx));
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
    if (!std::isfinite(size_low) || !std::isfinite(size_high) || !std::isfinite(denom) ||
        std::abs(denom) < T(1e-12)) {
      grad_slew[i] = g * dslew_low;
      grad_cap[i] = g * dcap_low;
      if (arc_low >= 0 && arc_low < num_arcs) {
        for (int k = 0; k < 4; ++k) {
#pragma omp atomic
          grad_lut_values[arc_low * value_stride + corners_low[k]] += g * weights_low[k];
        }
      }
      continue;
    }
    T factor = (size_clamped - size_low) / denom;
    bool inside = size[i] >= size_min && size[i] <= size_max;
    grad_size[i] = inside ? g * (value_high - value_low) / denom : T(0);
    grad_slew[i] = g * ((T(1) - factor) * dslew_low + factor * dslew_high);
    grad_cap[i] = g * ((T(1) - factor) * dcap_low + factor * dcap_high);
    if (arc_low >= 0 && arc_low < num_arcs) {
      for (int k = 0; k < 4; ++k) {
#pragma omp atomic
        grad_lut_values[arc_low * value_stride + corners_low[k]] += g * (T(1) - factor) * weights_low[k];
      }
    }
    if (arc_high >= 0 && arc_high < num_arcs) {
      for (int k = 0; k < 4; ++k) {
#pragma omp atomic
        grad_lut_values[arc_high * value_stride + corners_high[k]] += g * factor * weights_high[k];
      }
    }
  }
}
}  // namespace

at::Tensor cell_modeling_poly12_forward(
    at::Tensor coeff,
    at::Tensor main_id,
    at::Tensor arc_offset,
    at::Tensor vt,
    at::Tensor size,
    at::Tensor input_slew,
    at::Tensor out_cap) {
  CHECK_CPU(coeff);
  CHECK_CONTIGUOUS(coeff);
  CHECK_FLAT_CPU(main_id);
  CHECK_CONTIGUOUS(main_id);
  CHECK_FLAT_CPU(arc_offset);
  CHECK_CONTIGUOUS(arc_offset);
  CHECK_FLAT_CPU(vt);
  CHECK_CONTIGUOUS(vt);
  CHECK_FLAT_CPU(size);
  CHECK_CONTIGUOUS(size);
  CHECK_FLAT_CPU(input_slew);
  CHECK_CONTIGUOUS(input_slew);
  CHECK_FLAT_CPU(out_cap);
  CHECK_CONTIGUOUS(out_cap);
  AT_ASSERTM(coeff.dim() == 3 && coeff.size(2) == 12, "coeff must have shape [main, arc, 12]");
  AT_ASSERTM(main_id.numel() == arc_offset.numel(), "main_id and arc_offset size mismatch");
  AT_ASSERTM(main_id.numel() == vt.numel(), "main_id and vt size mismatch");
  AT_ASSERTM(main_id.numel() == size.numel(), "main_id and size size mismatch");
  AT_ASSERTM(main_id.numel() == input_slew.numel(), "main_id and input_slew size mismatch");
  AT_ASSERTM(main_id.numel() == out_cap.numel(), "main_id and out_cap size mismatch");

  auto output = at::empty_like(input_slew);
  DREAMPLACE_DISPATCH_FLOATING_TYPES(input_slew, "cell_modeling_poly12_forward", [&] {
    poly12_forward_cpu_kernel<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(coeff, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(main_id, int64_t),
        DREAMPLACE_TENSOR_DATA_PTR(arc_offset, int64_t),
        DREAMPLACE_TENSOR_DATA_PTR(vt, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(size, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(input_slew, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(out_cap, scalar_t),
        input_slew.numel(),
        coeff.size(0),
        coeff.size(1),
        at::get_num_threads(),
        DREAMPLACE_TENSOR_DATA_PTR(output, scalar_t));
  });
  return output;
}

std::vector<at::Tensor> cell_modeling_poly12_backward(
    at::Tensor grad_output,
    at::Tensor coeff,
    at::Tensor main_id,
    at::Tensor arc_offset,
    at::Tensor vt,
    at::Tensor size,
    at::Tensor input_slew,
    at::Tensor out_cap) {
  CHECK_FLAT_CPU(grad_output);
  CHECK_CONTIGUOUS(grad_output);
  CHECK_CPU(coeff);
  CHECK_CONTIGUOUS(coeff);
  CHECK_FLAT_CPU(main_id);
  CHECK_CONTIGUOUS(main_id);
  CHECK_FLAT_CPU(arc_offset);
  CHECK_CONTIGUOUS(arc_offset);
  CHECK_FLAT_CPU(vt);
  CHECK_CONTIGUOUS(vt);
  CHECK_FLAT_CPU(size);
  CHECK_CONTIGUOUS(size);
  CHECK_FLAT_CPU(input_slew);
  CHECK_CONTIGUOUS(input_slew);
  CHECK_FLAT_CPU(out_cap);
  CHECK_CONTIGUOUS(out_cap);
  AT_ASSERTM(coeff.dim() == 3 && coeff.size(2) == 12, "coeff must have shape [main, arc, 12]");

  auto grad_vt = at::empty_like(vt);
  auto grad_size = at::empty_like(size);
  auto grad_slew = at::empty_like(input_slew);
  auto grad_cap = at::empty_like(out_cap);
  DREAMPLACE_DISPATCH_FLOATING_TYPES(input_slew, "cell_modeling_poly12_backward", [&] {
    poly12_backward_cpu_kernel<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(grad_output, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(coeff, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(main_id, int64_t),
        DREAMPLACE_TENSOR_DATA_PTR(arc_offset, int64_t),
        DREAMPLACE_TENSOR_DATA_PTR(vt, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(size, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(input_slew, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(out_cap, scalar_t),
        input_slew.numel(),
        coeff.size(0),
        coeff.size(1),
        at::get_num_threads(),
        DREAMPLACE_TENSOR_DATA_PTR(grad_vt, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(grad_size, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(grad_slew, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(grad_cap, scalar_t));
  });
  return {grad_vt, grad_size, grad_slew, grad_cap};
}

at::Tensor cell_modeling_piecewise_size_forward(
    at::Tensor size_table,
    at::Tensor arc_table,
    at::Tensor size_count,
    at::Tensor trans_tables,
    at::Tensor cap_tables,
    at::Tensor lut_values,
    at::Tensor trans_dims,
    at::Tensor cap_dims,
    at::Tensor size,
    at::Tensor input_slew,
    at::Tensor out_cap,
    int64_t lut_boundary_mode) {
  CHECK_CPU(size_table);
  CHECK_CONTIGUOUS(size_table);
  CHECK_CPU(arc_table);
  CHECK_CONTIGUOUS(arc_table);
  CHECK_FLAT_CPU(size_count);
  CHECK_CONTIGUOUS(size_count);
  CHECK_CPU(trans_tables);
  CHECK_CONTIGUOUS(trans_tables);
  CHECK_CPU(cap_tables);
  CHECK_CONTIGUOUS(cap_tables);
  CHECK_CPU(lut_values);
  CHECK_CONTIGUOUS(lut_values);
  CHECK_FLAT_CPU(trans_dims);
  CHECK_CONTIGUOUS(trans_dims);
  CHECK_FLAT_CPU(cap_dims);
  CHECK_CONTIGUOUS(cap_dims);
  CHECK_FLAT_CPU(size);
  CHECK_CONTIGUOUS(size);
  CHECK_FLAT_CPU(input_slew);
  CHECK_CONTIGUOUS(input_slew);
  CHECK_FLAT_CPU(out_cap);
  CHECK_CONTIGUOUS(out_cap);
  auto output = at::zeros_like(size);
  DREAMPLACE_DISPATCH_FLOATING_TYPES(size, "cell_modeling_piecewise_size_forward", [&] {
    piecewise_size_forward_cpu_kernel<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(size_table, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(arc_table, int64_t),
        DREAMPLACE_TENSOR_DATA_PTR(size_count, int64_t),
        DREAMPLACE_TENSOR_DATA_PTR(trans_tables, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(cap_tables, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(lut_values, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(trans_dims, int64_t),
        DREAMPLACE_TENSOR_DATA_PTR(cap_dims, int64_t),
        DREAMPLACE_TENSOR_DATA_PTR(size, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(input_slew, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(out_cap, scalar_t),
        size.numel(),
        size_table.size(1),
        trans_tables.size(0),
        trans_tables.size(1),
        cap_tables.size(1),
        lut_values.size(1),
        at::get_num_threads(),
        lut_boundary_mode,
        DREAMPLACE_TENSOR_DATA_PTR(output, scalar_t));
  });
  return output;
}

std::vector<at::Tensor> cell_modeling_piecewise_size_backward(
    at::Tensor grad_output,
    at::Tensor size_table,
    at::Tensor arc_table,
    at::Tensor size_count,
    at::Tensor trans_tables,
    at::Tensor cap_tables,
    at::Tensor lut_values,
    at::Tensor trans_dims,
    at::Tensor cap_dims,
    at::Tensor size,
    at::Tensor input_slew,
    at::Tensor out_cap,
    int64_t lut_boundary_mode) {
  CHECK_FLAT_CPU(grad_output);
  CHECK_CONTIGUOUS(grad_output);
  auto grad_size = at::zeros_like(size);
  auto grad_slew = at::zeros_like(input_slew);
  auto grad_cap = at::zeros_like(out_cap);
  auto grad_lut_values = at::zeros_like(lut_values);
  DREAMPLACE_DISPATCH_FLOATING_TYPES(size, "cell_modeling_piecewise_size_backward", [&] {
    piecewise_size_backward_cpu_kernel<scalar_t>(
        DREAMPLACE_TENSOR_DATA_PTR(grad_output, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(size_table, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(arc_table, int64_t),
        DREAMPLACE_TENSOR_DATA_PTR(size_count, int64_t),
        DREAMPLACE_TENSOR_DATA_PTR(trans_tables, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(cap_tables, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(lut_values, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(trans_dims, int64_t),
        DREAMPLACE_TENSOR_DATA_PTR(cap_dims, int64_t),
        DREAMPLACE_TENSOR_DATA_PTR(size, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(input_slew, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(out_cap, scalar_t),
        size.numel(),
        size_table.size(1),
        trans_tables.size(0),
        trans_tables.size(1),
        cap_tables.size(1),
        lut_values.size(1),
        at::get_num_threads(),
        lut_boundary_mode,
        DREAMPLACE_TENSOR_DATA_PTR(grad_size, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(grad_slew, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(grad_cap, scalar_t),
        DREAMPLACE_TENSOR_DATA_PTR(grad_lut_values, scalar_t));
  });
  return {grad_size, grad_slew, grad_cap, grad_lut_values};
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &DREAMPLACE_NAMESPACE::cell_modeling_poly12_forward,
        "Cell modeling poly12 forward");
  m.def("backward", &DREAMPLACE_NAMESPACE::cell_modeling_poly12_backward,
        "Cell modeling poly12 backward");
  m.def("piecewise_size_forward", &DREAMPLACE_NAMESPACE::cell_modeling_piecewise_size_forward,
        "Cell modeling piecewise size forward");
  m.def("piecewise_size_backward", &DREAMPLACE_NAMESPACE::cell_modeling_piecewise_size_backward,
        "Cell modeling piecewise size backward");
}
