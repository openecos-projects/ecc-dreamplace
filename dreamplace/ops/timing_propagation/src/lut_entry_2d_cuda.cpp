#include "utility/src/torch.h"
#include "utility/src/utils.h"

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
    int32_t padded_lut_size);

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
    int32_t padded_lut_size);

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
    int32_t cap_cells);

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
    int32_t cap_cells);

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
    int32_t cap_cells);

at::Tensor lut_2d_forward_cuda(
    at::Tensor input_trans,
    at::Tensor output_caps,
    at::Tensor trans_tables_batch,
    at::Tensor cap_tables_batch,
    at::Tensor lut_values_batch,
    at::Tensor trans_dims_actual,
    at::Tensor cap_dims_actual) {
  CHECK_FLAT_CUDA(input_trans);
  CHECK_CONTIGUOUS(input_trans);
  CHECK_FLAT_CUDA(output_caps);
  CHECK_CONTIGUOUS(output_caps);
  CHECK_CUDA(trans_tables_batch);
  CHECK_CONTIGUOUS(trans_tables_batch);
  CHECK_CUDA(cap_tables_batch);
  CHECK_CONTIGUOUS(cap_tables_batch);
  CHECK_CUDA(lut_values_batch);
  CHECK_CONTIGUOUS(lut_values_batch);
  CHECK_FLAT_CUDA(trans_dims_actual);
  CHECK_CONTIGUOUS(trans_dims_actual);
  CHECK_FLAT_CUDA(cap_dims_actual);
  CHECK_CONTIGUOUS(cap_dims_actual);
  TORCH_CHECK(trans_dims_actual.scalar_type() == at::kInt, "trans_dims_actual must be int32");
  TORCH_CHECK(cap_dims_actual.scalar_type() == at::kInt, "cap_dims_actual must be int32");

  auto output = at::zeros_like(input_trans);
  int32_t batch_size = input_trans.numel();
  int32_t padded_trans_dim = trans_tables_batch.size(1);
  int32_t padded_cap_dim = cap_tables_batch.size(1);
  int32_t padded_lut_size = lut_values_batch.size(1);
  AT_DISPATCH_FLOATING_TYPES(input_trans.scalar_type(), "lut2dForwardCudaLauncher", [&] {
    lut2dForwardCudaLauncher<scalar_t>(
        input_trans.data_ptr<scalar_t>(),
        output_caps.data_ptr<scalar_t>(),
        trans_tables_batch.data_ptr<scalar_t>(),
        cap_tables_batch.data_ptr<scalar_t>(),
        lut_values_batch.data_ptr<scalar_t>(),
        trans_dims_actual.data_ptr<int32_t>(),
        cap_dims_actual.data_ptr<int32_t>(),
        output.data_ptr<scalar_t>(),
        batch_size,
        padded_trans_dim,
        padded_cap_dim,
        padded_lut_size);
  });
  return output;
}

std::vector<at::Tensor> lut_2d_backward_cuda(
    at::Tensor grad_output,
    at::Tensor input_trans,
    at::Tensor output_caps,
    at::Tensor trans_tables_batch,
    at::Tensor cap_tables_batch,
    at::Tensor lut_values_batch,
    at::Tensor trans_dims_actual,
    at::Tensor cap_dims_actual) {
  CHECK_FLAT_CUDA(grad_output);
  CHECK_CONTIGUOUS(grad_output);
  CHECK_FLAT_CUDA(input_trans);
  CHECK_CONTIGUOUS(input_trans);
  CHECK_FLAT_CUDA(output_caps);
  CHECK_CONTIGUOUS(output_caps);
  CHECK_CUDA(trans_tables_batch);
  CHECK_CONTIGUOUS(trans_tables_batch);
  CHECK_CUDA(cap_tables_batch);
  CHECK_CONTIGUOUS(cap_tables_batch);
  CHECK_CUDA(lut_values_batch);
  CHECK_CONTIGUOUS(lut_values_batch);
  CHECK_FLAT_CUDA(trans_dims_actual);
  CHECK_CONTIGUOUS(trans_dims_actual);
  CHECK_FLAT_CUDA(cap_dims_actual);
  CHECK_CONTIGUOUS(cap_dims_actual);

  auto grad_input_trans = at::zeros_like(input_trans);
  auto grad_output_caps = at::zeros_like(output_caps);
  auto grad_lut_values_batch = at::zeros_like(lut_values_batch);
  int32_t batch_size = input_trans.numel();
  int32_t padded_trans_dim = trans_tables_batch.size(1);
  int32_t padded_cap_dim = cap_tables_batch.size(1);
  int32_t padded_lut_size = lut_values_batch.size(1);
  AT_DISPATCH_FLOATING_TYPES(input_trans.scalar_type(), "lut2dBackwardCudaLauncher", [&] {
    lut2dBackwardCudaLauncher<scalar_t>(
        grad_output.data_ptr<scalar_t>(),
        input_trans.data_ptr<scalar_t>(),
        output_caps.data_ptr<scalar_t>(),
        trans_tables_batch.data_ptr<scalar_t>(),
        cap_tables_batch.data_ptr<scalar_t>(),
        lut_values_batch.data_ptr<scalar_t>(),
        trans_dims_actual.data_ptr<int32_t>(),
        cap_dims_actual.data_ptr<int32_t>(),
        grad_input_trans.data_ptr<scalar_t>(),
        grad_output_caps.data_ptr<scalar_t>(),
        grad_lut_values_batch.data_ptr<scalar_t>(),
        batch_size,
        padded_trans_dim,
        padded_cap_dim,
        padded_lut_size);
  });
  return {grad_input_trans, grad_output_caps, at::Tensor(), at::Tensor(),
          grad_lut_values_batch, at::Tensor(), at::Tensor()};
}

at::Tensor lut_2d_build_coefficients_cuda(
    at::Tensor trans_tables_batch,
    at::Tensor cap_tables_batch,
    at::Tensor lut_values_batch,
    at::Tensor trans_dims_actual,
    at::Tensor cap_dims_actual) {
  CHECK_CUDA(trans_tables_batch);
  CHECK_CONTIGUOUS(trans_tables_batch);
  CHECK_CUDA(cap_tables_batch);
  CHECK_CONTIGUOUS(cap_tables_batch);
  CHECK_CUDA(lut_values_batch);
  CHECK_CONTIGUOUS(lut_values_batch);
  CHECK_FLAT_CUDA(trans_dims_actual);
  CHECK_CONTIGUOUS(trans_dims_actual);
  CHECK_FLAT_CUDA(cap_dims_actual);
  CHECK_CONTIGUOUS(cap_dims_actual);
  TORCH_CHECK(trans_dims_actual.scalar_type() == at::kInt, "trans_dims_actual must be int32");
  TORCH_CHECK(cap_dims_actual.scalar_type() == at::kInt, "cap_dims_actual must be int32");

  int32_t num_luts = trans_tables_batch.size(0);
  int32_t trans_cells = std::max<int32_t>(0, trans_tables_batch.size(1) - 1);
  int32_t cap_cells = std::max<int32_t>(0, cap_tables_batch.size(1) - 1);
  auto coeff = at::zeros({num_luts, trans_cells, cap_cells, 4}, trans_tables_batch.options());
  AT_DISPATCH_FLOATING_TYPES(trans_tables_batch.scalar_type(), "lut2dBuildCoeffCudaLauncher", [&] {
    lut2dBuildCoeffCudaLauncher<scalar_t>(
        trans_tables_batch.data_ptr<scalar_t>(),
        cap_tables_batch.data_ptr<scalar_t>(),
        lut_values_batch.data_ptr<scalar_t>(),
        trans_dims_actual.data_ptr<int32_t>(),
        cap_dims_actual.data_ptr<int32_t>(),
        coeff.data_ptr<scalar_t>(),
        num_luts,
        trans_tables_batch.size(1),
        cap_tables_batch.size(1),
        lut_values_batch.size(1),
        trans_cells,
        cap_cells);
  });
  return coeff;
}

at::Tensor lut_2d_coeff_forward_cuda(
    at::Tensor input_trans,
    at::Tensor output_caps,
    at::Tensor trans_tables_batch,
    at::Tensor cap_tables_batch,
    at::Tensor coeff_batch,
    at::Tensor trans_dims_actual,
    at::Tensor cap_dims_actual) {
  CHECK_FLAT_CUDA(input_trans);
  CHECK_CONTIGUOUS(input_trans);
  CHECK_FLAT_CUDA(output_caps);
  CHECK_CONTIGUOUS(output_caps);
  CHECK_CUDA(trans_tables_batch);
  CHECK_CONTIGUOUS(trans_tables_batch);
  CHECK_CUDA(cap_tables_batch);
  CHECK_CONTIGUOUS(cap_tables_batch);
  CHECK_CUDA(coeff_batch);
  CHECK_CONTIGUOUS(coeff_batch);
  CHECK_FLAT_CUDA(trans_dims_actual);
  CHECK_CONTIGUOUS(trans_dims_actual);
  CHECK_FLAT_CUDA(cap_dims_actual);
  CHECK_CONTIGUOUS(cap_dims_actual);
  TORCH_CHECK(trans_dims_actual.scalar_type() == at::kInt, "trans_dims_actual must be int32");
  TORCH_CHECK(cap_dims_actual.scalar_type() == at::kInt, "cap_dims_actual must be int32");

  auto output = at::zeros_like(input_trans);
  AT_DISPATCH_FLOATING_TYPES(input_trans.scalar_type(), "lut2dCoeffForwardCudaLauncher", [&] {
    lut2dCoeffForwardCudaLauncher<scalar_t>(
        input_trans.data_ptr<scalar_t>(),
        output_caps.data_ptr<scalar_t>(),
        trans_tables_batch.data_ptr<scalar_t>(),
        cap_tables_batch.data_ptr<scalar_t>(),
        coeff_batch.data_ptr<scalar_t>(),
        trans_dims_actual.data_ptr<int32_t>(),
        cap_dims_actual.data_ptr<int32_t>(),
        output.data_ptr<scalar_t>(),
        input_trans.numel(),
        trans_tables_batch.size(1),
        cap_tables_batch.size(1),
        coeff_batch.size(1),
        coeff_batch.size(2));
  });
  return output;
}

std::vector<at::Tensor> lut_2d_coeff_backward_cuda(
    at::Tensor grad_output,
    at::Tensor input_trans,
    at::Tensor output_caps,
    at::Tensor trans_tables_batch,
    at::Tensor cap_tables_batch,
    at::Tensor coeff_batch,
    at::Tensor trans_dims_actual,
    at::Tensor cap_dims_actual) {
  CHECK_FLAT_CUDA(grad_output);
  CHECK_CONTIGUOUS(grad_output);
  CHECK_FLAT_CUDA(input_trans);
  CHECK_CONTIGUOUS(input_trans);
  CHECK_FLAT_CUDA(output_caps);
  CHECK_CONTIGUOUS(output_caps);
  auto grad_input_trans = at::zeros_like(input_trans);
  auto grad_output_caps = at::zeros_like(output_caps);
  AT_DISPATCH_FLOATING_TYPES(input_trans.scalar_type(), "lut2dCoeffBackwardCudaLauncher", [&] {
    lut2dCoeffBackwardCudaLauncher<scalar_t>(
        grad_output.data_ptr<scalar_t>(),
        input_trans.data_ptr<scalar_t>(),
        output_caps.data_ptr<scalar_t>(),
        trans_tables_batch.data_ptr<scalar_t>(),
        cap_tables_batch.data_ptr<scalar_t>(),
        coeff_batch.data_ptr<scalar_t>(),
        trans_dims_actual.data_ptr<int32_t>(),
        cap_dims_actual.data_ptr<int32_t>(),
        grad_input_trans.data_ptr<scalar_t>(),
        grad_output_caps.data_ptr<scalar_t>(),
        input_trans.numel(),
        trans_tables_batch.size(1),
        cap_tables_batch.size(1),
        coeff_batch.size(1),
        coeff_batch.size(2));
  });
  return {grad_input_trans, grad_output_caps, at::Tensor(), at::Tensor(), at::Tensor(),
          at::Tensor(), at::Tensor()};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &lut_2d_forward_cuda, "2D LUT forward (CUDA)");
  m.def("backward", &lut_2d_backward_cuda, "2D LUT backward (CUDA)");
  m.def("build_coefficients", &lut_2d_build_coefficients_cuda, "2D LUT coefficient build (CUDA)");
  m.def("coeff_forward", &lut_2d_coeff_forward_cuda, "2D LUT coefficient forward (CUDA)");
  m.def("coeff_backward", &lut_2d_coeff_backward_cuda, "2D LUT coefficient backward (CUDA)");
}
