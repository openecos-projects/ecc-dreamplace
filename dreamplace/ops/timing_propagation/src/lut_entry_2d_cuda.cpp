#include "utility/src/torch.h"
#include "utility/src/utils.h"
#include <c10/cuda/CUDAGuard.h>

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
  const c10::cuda::CUDAGuard guard(input_trans.device());
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
  const c10::cuda::CUDAGuard guard(input_trans.device());
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
  const c10::cuda::CUDAGuard guard(trans_tables_batch.device());
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
  const c10::cuda::CUDAGuard guard(input_trans.device());
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
  const c10::cuda::CUDAGuard guard(input_trans.device());
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

std::vector<at::Tensor> lut_indexed_cuda(
    at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor);
std::vector<at::Tensor> buffer_lut_indexed_cuda(
    at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor);

std::vector<at::Tensor> smooth_scatter_cuda(at::Tensor, at::Tensor, at::Tensor, double, bool,
                                        std::vector<at::Tensor>);
std::vector<at::Tensor> hard_scatter_forward_cuda(at::Tensor, at::Tensor, at::Tensor, bool);
std::vector<at::Tensor> hard_scatter_backward_cuda(at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, bool);
std::vector<at::Tensor> cell_aat_forward_cuda(at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, std::vector<at::Tensor>, bool, double);
std::vector<at::Tensor> cell_aat_backward_cuda(at::Tensor, at::Tensor, at::Tensor, int64_t,
                                           std::vector<at::Tensor>);
std::vector<at::Tensor> net_aat_forward_cuda(at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, double);
std::vector<at::Tensor> net_aat_backward_cuda(at::Tensor, at::Tensor, at::Tensor, int64_t,
    std::vector<at::Tensor>, bool, bool);
std::vector<at::Tensor> buffer_interpolate_forward_cuda(at::Tensor, at::Tensor, at::Tensor, at::Tensor);
std::vector<at::Tensor> buffer_interpolate_backward_cuda(at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor, at::Tensor);
std::vector<at::Tensor> timing_lanes_forward_cuda(at::Tensor, std::vector<at::Tensor>,
    std::vector<at::Tensor>, bool, double, std::vector<at::Tensor>);
std::vector<at::Tensor> timing_lanes_backward_cuda(at::Tensor, at::Tensor, at::Tensor, at::Tensor,
    std::vector<at::Tensor>, bool);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("timing_lanes_forward", &timing_lanes_forward_cuda, "Grouped arrival/slew lanes (CUDA)");
  m.def("timing_lanes_backward", &timing_lanes_backward_cuda, "Arrival/slew lane adjoints (CUDA)");
  m.def("buffer_interpolate_forward", &buffer_interpolate_forward_cuda, "Buffer drive-strength interpolation (CUDA)");
  m.def("buffer_interpolate_backward", &buffer_interpolate_backward_cuda, "Buffer interpolation derivatives (CUDA)");
  m.def("net_aat_forward", &net_aat_forward_cuda, "Net arrival/slew updates (CUDA)");
  m.def("net_aat_backward", &net_aat_backward_cuda, "Deterministic net pin derivatives (CUDA)");
  m.def("hard_scatter_forward", &hard_scatter_forward_cuda, "Slew max and Torch tie counts (CUDA)");
  m.def("hard_scatter_backward", &hard_scatter_backward_cuda, "Slew max first-order adjoints (CUDA)");
  m.def("buffer_indexed_forward", &buffer_lut_indexed_cuda, "Indexed buffer LUT query with 1D extrapolation (CUDA)");
  m.def("cell_aat_forward", &cell_aat_forward_cuda, "Fixed-master cell AAT forward (CUDA)");
  m.def("cell_aat_backward", &cell_aat_backward_cuda, "Fixed-master cell AAT backward (CUDA)",
        pybind11::arg("arcs"), pybind11::arg("jac"), pybind11::arg("gradient"), pybind11::arg("pins"),
        pybind11::arg("groups") = std::vector<at::Tensor>{});
  m.def("smooth_scatter_forward", &smooth_scatter_cuda, "Timing scatter log-sum-exp with first derivatives (CUDA)",
        pybind11::arg("dest"), pybind11::arg("index"), pybind11::arg("src"), pybind11::arg("tau"),
        pybind11::arg("include_self"), pybind11::arg("groups") = std::vector<at::Tensor>{});
  m.def("indexed_forward", &lut_indexed_cuda, "Indexed scalar/1D/2D LUT query with input derivatives (CUDA)");
  m.def("forward", &lut_2d_forward_cuda, "2D LUT forward (CUDA)");
  m.def("backward", &lut_2d_backward_cuda, "2D LUT backward (CUDA)");
  m.def("build_coefficients", &lut_2d_build_coefficients_cuda, "2D LUT coefficient build (CUDA)");
  m.def("coeff_forward", &lut_2d_coeff_forward_cuda, "2D LUT coefficient forward (CUDA)");
  m.def("coeff_backward", &lut_2d_coeff_backward_cuda, "2D LUT coefficient backward (CUDA)");
}
