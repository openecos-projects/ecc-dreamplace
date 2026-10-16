#include "utility/src/torch.h"
#include "utility/src/utils.h"

#include <algorithm>
#include <cmath>
#include <torch/extension.h>
#include <vector>

namespace {
constexpr double kDenomEps = 1e-12;

template <typename scalar_t>
inline int32_t find_high_index(
    scalar_t size_value,
    const scalar_t* sizes,
    int32_t dim) {
  scalar_t x_min = sizes[0];
  scalar_t x_max = sizes[dim - 1];
  scalar_t x = std::min(std::max(size_value, x_min), x_max);
  int32_t high = std::upper_bound(sizes, sizes + dim, x) - sizes;
  high = std::max<int32_t>(1, std::min<int32_t>(high, dim - 1));
  return high;
}

template <typename scalar_t>
inline void interpolate_one(
    scalar_t size_value,
    const scalar_t* sizes,
    const int64_t* libpin_ids,
    int32_t dim,
    const scalar_t* flat_values,
    scalar_t* interp,
    scalar_t* d_interp_d_size) {
  if (dim <= 0) {
    *interp = scalar_t(0);
    *d_interp_d_size = scalar_t(0);
    return;
  }
  if (dim == 1) {
    int64_t id = libpin_ids[0];
    *interp = id >= 0 ? flat_values[id] : scalar_t(0);
    *d_interp_d_size = scalar_t(0);
    return;
  }
  int32_t high = find_high_index(size_value, sizes, dim);
  int32_t low = high - 1;
  scalar_t x_min = sizes[0];
  scalar_t x_max = sizes[dim - 1];
  scalar_t x = std::min(std::max(size_value, x_min), x_max);
  scalar_t x0 = sizes[low];
  scalar_t x1 = sizes[high];
  int64_t id0 = libpin_ids[low];
  int64_t id1 = libpin_ids[high];
  scalar_t y0 = id0 >= 0 ? flat_values[id0] : scalar_t(0);
  scalar_t y1 = id1 >= 0 ? flat_values[id1] : scalar_t(0);
  scalar_t denom = x1 - x0;
  if (std::abs(static_cast<double>(denom)) < kDenomEps) {
    *interp = y0;
    *d_interp_d_size = scalar_t(0);
    return;
  }
  scalar_t slope = (y1 - y0) / denom;
  *interp = y0 + (x - x0) * slope;
  *d_interp_d_size = (size_value >= x_min && size_value <= x_max) ? slope : scalar_t(0);
}

void check_common_inputs(
    const at::Tensor& local_sizes,
    const at::Tensor& vt_probs,
    const at::Tensor& candidate_sizes,
    const at::Tensor& candidate_libpin_ids,
    const at::Tensor& actual_dims,
    const at::Tensor& flat_pin_values,
    const at::Tensor& current_values_active) {
  CHECK_FLAT(local_sizes);
  CHECK_CONTIGUOUS(local_sizes);
  CHECK_CONTIGUOUS(vt_probs);
  CHECK_CONTIGUOUS(candidate_sizes);
  CHECK_CONTIGUOUS(candidate_libpin_ids);
  CHECK_CONTIGUOUS(actual_dims);
  CHECK_CONTIGUOUS(flat_pin_values);
  CHECK_CONTIGUOUS(current_values_active);
  TORCH_CHECK(local_sizes.dim() == 1, "local_sizes must be 1D");
  TORCH_CHECK(vt_probs.dim() == 2, "vt_probs must be 2D");
  TORCH_CHECK(candidate_sizes.dim() == 3, "candidate_sizes must be 3D");
  TORCH_CHECK(candidate_libpin_ids.dim() == 3, "candidate_libpin_ids must be 3D");
  TORCH_CHECK(actual_dims.dim() == 2, "actual_dims must be 2D");
  TORCH_CHECK(flat_pin_values.dim() == 2, "flat_pin_values must be 2D");
  TORCH_CHECK(current_values_active.dim() == 2, "current_values_active must be 2D");
  TORCH_CHECK(candidate_libpin_ids.scalar_type() == at::kLong, "candidate_libpin_ids must be int64");
  TORCH_CHECK(actual_dims.scalar_type() == at::kInt, "actual_dims must be int32");
  TORCH_CHECK(local_sizes.scalar_type() == vt_probs.scalar_type(), "local_sizes/vt_probs dtype mismatch");
  TORCH_CHECK(local_sizes.scalar_type() == candidate_sizes.scalar_type(), "candidate_sizes dtype mismatch");
  TORCH_CHECK(local_sizes.scalar_type() == flat_pin_values.scalar_type(), "flat_pin_values dtype mismatch");
  TORCH_CHECK(local_sizes.scalar_type() == current_values_active.scalar_type(), "current_values_active dtype mismatch");
  int64_t pin_count = local_sizes.numel();
  int64_t vt_count = vt_probs.size(1);
  TORCH_CHECK(vt_probs.size(0) == pin_count, "vt_probs pin dimension mismatch");
  TORCH_CHECK(candidate_sizes.size(0) == pin_count, "candidate_sizes pin dimension mismatch");
  TORCH_CHECK(candidate_sizes.size(1) == vt_count, "candidate_sizes VT dimension mismatch");
  TORCH_CHECK(candidate_libpin_ids.sizes() == candidate_sizes.sizes(), "candidate_libpin_ids shape mismatch");
  TORCH_CHECK(actual_dims.size(0) == pin_count, "actual_dims pin dimension mismatch");
  TORCH_CHECK(actual_dims.size(1) == vt_count, "actual_dims VT dimension mismatch");
  TORCH_CHECK(current_values_active.size(0) == flat_pin_values.size(0), "property dimension mismatch");
  TORCH_CHECK(current_values_active.size(1) == pin_count, "current_values_active pin dimension mismatch");
}
}  // namespace

at::Tensor size_interpolated_pin_forward_cpp(
    at::Tensor local_sizes,
    at::Tensor vt_probs,
    at::Tensor candidate_sizes,
    at::Tensor candidate_libpin_ids,
    at::Tensor actual_dims,
    at::Tensor flat_pin_values,
    at::Tensor current_values_active) {
  CHECK_CPU(local_sizes);
  CHECK_CPU(vt_probs);
  CHECK_CPU(candidate_sizes);
  CHECK_CPU(candidate_libpin_ids);
  CHECK_CPU(actual_dims);
  CHECK_CPU(flat_pin_values);
  CHECK_CPU(current_values_active);
  check_common_inputs(
      local_sizes,
      vt_probs,
      candidate_sizes,
      candidate_libpin_ids,
      actual_dims,
      flat_pin_values,
      current_values_active);

  auto output = current_values_active.clone();
  int64_t pin_count = local_sizes.numel();
  int64_t vt_count = vt_probs.size(1);
  int64_t kmax = candidate_sizes.size(2);
  int64_t q_count = flat_pin_values.size(0);
  int64_t num_flat_libpins = flat_pin_values.size(1);
  AT_DISPATCH_FLOATING_TYPES(local_sizes.scalar_type(), "sizeInterpolatedPinForwardCpp", [&] {
    const scalar_t* local_sizes_ptr = local_sizes.data_ptr<scalar_t>();
    const scalar_t* vt_probs_ptr = vt_probs.data_ptr<scalar_t>();
    const scalar_t* candidate_sizes_ptr = candidate_sizes.data_ptr<scalar_t>();
    const int64_t* candidate_libpin_ids_ptr = candidate_libpin_ids.data_ptr<int64_t>();
    const int32_t* actual_dims_ptr = actual_dims.data_ptr<int32_t>();
    const scalar_t* flat_values_ptr = flat_pin_values.data_ptr<scalar_t>();
    scalar_t* output_ptr = output.data_ptr<scalar_t>();
#pragma omp parallel for collapse(2) num_threads(at::get_num_threads())
    for (int64_t q = 0; q < q_count; ++q) {
      for (int64_t p = 0; p < pin_count; ++p) {
        scalar_t weighted_sum = scalar_t(0);
        scalar_t total_weight = scalar_t(0);
        for (int64_t vt = 0; vt < vt_count; ++vt) {
          int32_t dim = actual_dims_ptr[p * vt_count + vt];
          if (dim <= 0) {
            continue;
          }
          scalar_t weight = vt_probs_ptr[p * vt_count + vt];
          if (weight == scalar_t(0)) {
            continue;
          }
          const scalar_t* sizes = candidate_sizes_ptr + (p * vt_count + vt) * kmax;
          const int64_t* ids = candidate_libpin_ids_ptr + (p * vt_count + vt) * kmax;
          scalar_t interp = scalar_t(0);
          scalar_t deriv = scalar_t(0);
          interpolate_one(
              local_sizes_ptr[p],
              sizes,
              ids,
              dim,
              flat_values_ptr + q * num_flat_libpins,
              &interp,
              &deriv);
          weighted_sum += weight * interp;
          total_weight += weight;
        }
        if (total_weight > scalar_t(0)) {
          output_ptr[q * pin_count + p] = weighted_sum / std::max(total_weight, static_cast<scalar_t>(kDenomEps));
        }
      }
    }
  });
  return output;
}

std::vector<at::Tensor> size_interpolated_pin_backward_cpp(
    at::Tensor grad_output,
    at::Tensor local_sizes,
    at::Tensor vt_probs,
    at::Tensor candidate_sizes,
    at::Tensor candidate_libpin_ids,
    at::Tensor actual_dims,
    at::Tensor flat_pin_values,
    at::Tensor current_values_active) {
  CHECK_CPU(grad_output);
  CHECK_CPU(local_sizes);
  CHECK_CPU(vt_probs);
  CHECK_CPU(candidate_sizes);
  CHECK_CPU(candidate_libpin_ids);
  CHECK_CPU(actual_dims);
  CHECK_CPU(flat_pin_values);
  CHECK_CPU(current_values_active);
  CHECK_CONTIGUOUS(grad_output);
  check_common_inputs(
      local_sizes,
      vt_probs,
      candidate_sizes,
      candidate_libpin_ids,
      actual_dims,
      flat_pin_values,
      current_values_active);
  auto grad_local_sizes = at::zeros_like(local_sizes);
  auto grad_vt_probs = at::zeros_like(vt_probs);
  int64_t pin_count = local_sizes.numel();
  int64_t vt_count = vt_probs.size(1);
  int64_t kmax = candidate_sizes.size(2);
  int64_t q_count = flat_pin_values.size(0);
  int64_t num_flat_libpins = flat_pin_values.size(1);
  AT_DISPATCH_FLOATING_TYPES(local_sizes.scalar_type(), "sizeInterpolatedPinBackwardCpp", [&] {
    const scalar_t* grad_output_ptr = grad_output.data_ptr<scalar_t>();
    const scalar_t* local_sizes_ptr = local_sizes.data_ptr<scalar_t>();
    const scalar_t* vt_probs_ptr = vt_probs.data_ptr<scalar_t>();
    const scalar_t* candidate_sizes_ptr = candidate_sizes.data_ptr<scalar_t>();
    const int64_t* candidate_libpin_ids_ptr = candidate_libpin_ids.data_ptr<int64_t>();
    const int32_t* actual_dims_ptr = actual_dims.data_ptr<int32_t>();
    const scalar_t* flat_values_ptr = flat_pin_values.data_ptr<scalar_t>();
    scalar_t* grad_local_sizes_ptr = grad_local_sizes.data_ptr<scalar_t>();
    scalar_t* grad_vt_probs_ptr = grad_vt_probs.data_ptr<scalar_t>();
#pragma omp parallel for num_threads(at::get_num_threads())
    for (int64_t p = 0; p < pin_count; ++p) {
      scalar_t local_grad_size = scalar_t(0);
      for (int64_t q = 0; q < q_count; ++q) {
        scalar_t interp_values[64];
        scalar_t deriv_values[64];
        scalar_t weighted_sum = scalar_t(0);
        scalar_t total_weight = scalar_t(0);
        for (int64_t vt = 0; vt < vt_count; ++vt) {
          scalar_t interp = scalar_t(0);
          scalar_t deriv = scalar_t(0);
          int32_t dim = actual_dims_ptr[p * vt_count + vt];
          if (dim > 0) {
            const scalar_t* sizes = candidate_sizes_ptr + (p * vt_count + vt) * kmax;
            const int64_t* ids = candidate_libpin_ids_ptr + (p * vt_count + vt) * kmax;
            interpolate_one(
                local_sizes_ptr[p],
                sizes,
                ids,
                dim,
                flat_values_ptr + q * num_flat_libpins,
                &interp,
                &deriv);
          }
          if (vt < 64) {
            interp_values[vt] = interp;
            deriv_values[vt] = deriv;
          }
          scalar_t weight = vt_probs_ptr[p * vt_count + vt];
          weighted_sum += weight * interp;
          total_weight += dim > 0 ? weight : scalar_t(0);
        }
        if (total_weight <= scalar_t(0)) {
          continue;
        }
        scalar_t inv_total = scalar_t(1) / std::max(total_weight, static_cast<scalar_t>(kDenomEps));
        scalar_t g = grad_output_ptr[q * pin_count + p];
        for (int64_t vt = 0; vt < vt_count; ++vt) {
          int32_t dim = actual_dims_ptr[p * vt_count + vt];
          if (dim <= 0) {
            continue;
          }
          scalar_t interp = vt < 64 ? interp_values[vt] : scalar_t(0);
          scalar_t deriv = vt < 64 ? deriv_values[vt] : scalar_t(0);
          if (vt >= 64) {
            const scalar_t* sizes = candidate_sizes_ptr + (p * vt_count + vt) * kmax;
            const int64_t* ids = candidate_libpin_ids_ptr + (p * vt_count + vt) * kmax;
            interpolate_one(
                local_sizes_ptr[p],
                sizes,
                ids,
                dim,
                flat_values_ptr + q * num_flat_libpins,
                &interp,
                &deriv);
          }
          scalar_t weight = vt_probs_ptr[p * vt_count + vt];
          local_grad_size += g * weight * deriv * inv_total;
          grad_vt_probs_ptr[p * vt_count + vt] +=
              g * (interp * total_weight - weighted_sum) * inv_total * inv_total;
        }
      }
      grad_local_sizes_ptr[p] = local_grad_size;
    }
  });
  return {
      grad_local_sizes,
      grad_vt_probs,
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor(),
      at::Tensor()};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &size_interpolated_pin_forward_cpp, "size-interpolated pin forward");
  m.def("backward", &size_interpolated_pin_backward_cpp, "size-interpolated pin backward");
}
