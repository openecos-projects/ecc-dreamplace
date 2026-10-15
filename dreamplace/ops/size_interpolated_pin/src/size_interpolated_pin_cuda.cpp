#include "utility/src/torch.h"
#include "utility/src/utils.h"

#include <torch/extension.h>
#include <vector>

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
    int32_t num_flat_libpins);

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
    int32_t num_flat_libpins);

namespace {
void check_inputs(
    const at::Tensor& local_sizes,
    const at::Tensor& vt_probs,
    const at::Tensor& candidate_sizes,
    const at::Tensor& candidate_libpin_ids,
    const at::Tensor& actual_dims,
    const at::Tensor& flat_pin_values,
    const at::Tensor& current_values_active) {
  CHECK_FLAT_CUDA(local_sizes);
  CHECK_CUDA(vt_probs);
  CHECK_CUDA(candidate_sizes);
  CHECK_CUDA(candidate_libpin_ids);
  CHECK_CUDA(actual_dims);
  CHECK_CUDA(flat_pin_values);
  CHECK_CUDA(current_values_active);
  CHECK_CONTIGUOUS(local_sizes);
  CHECK_CONTIGUOUS(vt_probs);
  CHECK_CONTIGUOUS(candidate_sizes);
  CHECK_CONTIGUOUS(candidate_libpin_ids);
  CHECK_CONTIGUOUS(actual_dims);
  CHECK_CONTIGUOUS(flat_pin_values);
  CHECK_CONTIGUOUS(current_values_active);
  TORCH_CHECK(candidate_libpin_ids.scalar_type() == at::kLong, "candidate_libpin_ids must be int64");
  TORCH_CHECK(actual_dims.scalar_type() == at::kInt, "actual_dims must be int32");
  TORCH_CHECK(local_sizes.scalar_type() == vt_probs.scalar_type(), "local_sizes/vt_probs dtype mismatch");
  TORCH_CHECK(local_sizes.scalar_type() == candidate_sizes.scalar_type(), "candidate_sizes dtype mismatch");
  TORCH_CHECK(local_sizes.scalar_type() == flat_pin_values.scalar_type(), "flat_pin_values dtype mismatch");
  TORCH_CHECK(local_sizes.scalar_type() == current_values_active.scalar_type(), "current_values_active dtype mismatch");
}
}  // namespace

at::Tensor size_interpolated_pin_forward_cuda(
    at::Tensor local_sizes,
    at::Tensor vt_probs,
    at::Tensor candidate_sizes,
    at::Tensor candidate_libpin_ids,
    at::Tensor actual_dims,
    at::Tensor flat_pin_values,
    at::Tensor current_values_active) {
  check_inputs(
      local_sizes,
      vt_probs,
      candidate_sizes,
      candidate_libpin_ids,
      actual_dims,
      flat_pin_values,
      current_values_active);
  auto output = at::empty_like(current_values_active);
  int32_t pin_count = local_sizes.numel();
  int32_t vt_count = vt_probs.size(1);
  int32_t kmax = candidate_sizes.size(2);
  int32_t q_count = flat_pin_values.size(0);
  int32_t num_flat_libpins = flat_pin_values.size(1);
  AT_DISPATCH_FLOATING_TYPES(local_sizes.scalar_type(), "sizeInterpolatedPinForwardCuda", [&] {
    sizeInterpolatedPinForwardCudaLauncher<scalar_t>(
        local_sizes.data_ptr<scalar_t>(),
        vt_probs.data_ptr<scalar_t>(),
        candidate_sizes.data_ptr<scalar_t>(),
        candidate_libpin_ids.data_ptr<int64_t>(),
        actual_dims.data_ptr<int32_t>(),
        flat_pin_values.data_ptr<scalar_t>(),
        current_values_active.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(),
        pin_count,
        vt_count,
        kmax,
        q_count,
        num_flat_libpins);
  });
  return output;
}

std::vector<at::Tensor> size_interpolated_pin_backward_cuda(
    at::Tensor grad_output,
    at::Tensor local_sizes,
    at::Tensor vt_probs,
    at::Tensor candidate_sizes,
    at::Tensor candidate_libpin_ids,
    at::Tensor actual_dims,
    at::Tensor flat_pin_values,
    at::Tensor current_values_active) {
  (void)current_values_active;
  CHECK_CUDA(grad_output);
  CHECK_CONTIGUOUS(grad_output);
  check_inputs(
      local_sizes,
      vt_probs,
      candidate_sizes,
      candidate_libpin_ids,
      actual_dims,
      flat_pin_values,
      current_values_active);
  auto grad_local_sizes = at::zeros_like(local_sizes);
  auto grad_vt_probs = at::zeros_like(vt_probs);
  int32_t pin_count = local_sizes.numel();
  int32_t vt_count = vt_probs.size(1);
  int32_t kmax = candidate_sizes.size(2);
  int32_t q_count = flat_pin_values.size(0);
  int32_t num_flat_libpins = flat_pin_values.size(1);
  AT_DISPATCH_FLOATING_TYPES(local_sizes.scalar_type(), "sizeInterpolatedPinBackwardCuda", [&] {
    sizeInterpolatedPinBackwardCudaLauncher<scalar_t>(
        grad_output.data_ptr<scalar_t>(),
        local_sizes.data_ptr<scalar_t>(),
        vt_probs.data_ptr<scalar_t>(),
        candidate_sizes.data_ptr<scalar_t>(),
        candidate_libpin_ids.data_ptr<int64_t>(),
        actual_dims.data_ptr<int32_t>(),
        flat_pin_values.data_ptr<scalar_t>(),
        grad_local_sizes.data_ptr<scalar_t>(),
        grad_vt_probs.data_ptr<scalar_t>(),
        pin_count,
        vt_count,
        kmax,
        q_count,
        num_flat_libpins);
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
  m.def("forward", &size_interpolated_pin_forward_cuda, "size-interpolated pin CUDA forward");
  m.def("backward", &size_interpolated_pin_backward_cuda, "size-interpolated pin CUDA backward");
}
