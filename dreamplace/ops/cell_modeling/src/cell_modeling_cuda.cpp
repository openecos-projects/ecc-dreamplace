#include "utility/src/torch.h"
#include "utility/src/utils.h"

DREAMPLACE_BEGIN_NAMESPACE

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
    T* output);

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
    T* grad_cap);

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
    T* output);

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
    T* grad_lut_values);

at::Tensor cell_modeling_poly12_forward(
    at::Tensor coeff,
    at::Tensor main_id,
    at::Tensor arc_offset,
    at::Tensor vt,
    at::Tensor size,
    at::Tensor input_slew,
    at::Tensor out_cap) {
  CHECK_CUDA(coeff);
  CHECK_CONTIGUOUS(coeff);
  CHECK_FLAT_CUDA(main_id);
  CHECK_CONTIGUOUS(main_id);
  CHECK_FLAT_CUDA(arc_offset);
  CHECK_CONTIGUOUS(arc_offset);
  CHECK_FLAT_CUDA(vt);
  CHECK_CONTIGUOUS(vt);
  CHECK_FLAT_CUDA(size);
  CHECK_CONTIGUOUS(size);
  CHECK_FLAT_CUDA(input_slew);
  CHECK_CONTIGUOUS(input_slew);
  CHECK_FLAT_CUDA(out_cap);
  CHECK_CONTIGUOUS(out_cap);
  AT_ASSERTM(coeff.dim() == 3 && coeff.size(2) == 12, "coeff must have shape [main, arc, 12]");

  auto output = at::empty_like(input_slew);
  DREAMPLACE_DISPATCH_FLOATING_TYPES(input_slew, "cellModelingPoly12ForwardCudaLauncher", [&] {
    cellModelingPoly12ForwardCudaLauncher<scalar_t>(
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
  CHECK_FLAT_CUDA(grad_output);
  CHECK_CONTIGUOUS(grad_output);
  CHECK_CUDA(coeff);
  CHECK_CONTIGUOUS(coeff);
  CHECK_FLAT_CUDA(main_id);
  CHECK_CONTIGUOUS(main_id);
  CHECK_FLAT_CUDA(arc_offset);
  CHECK_CONTIGUOUS(arc_offset);
  CHECK_FLAT_CUDA(vt);
  CHECK_CONTIGUOUS(vt);
  CHECK_FLAT_CUDA(size);
  CHECK_CONTIGUOUS(size);
  CHECK_FLAT_CUDA(input_slew);
  CHECK_CONTIGUOUS(input_slew);
  CHECK_FLAT_CUDA(out_cap);
  CHECK_CONTIGUOUS(out_cap);
  AT_ASSERTM(coeff.dim() == 3 && coeff.size(2) == 12, "coeff must have shape [main, arc, 12]");

  auto grad_vt = at::empty_like(vt);
  auto grad_size = at::empty_like(size);
  auto grad_slew = at::empty_like(input_slew);
  auto grad_cap = at::empty_like(out_cap);
  DREAMPLACE_DISPATCH_FLOATING_TYPES(input_slew, "cellModelingPoly12BackwardCudaLauncher", [&] {
    cellModelingPoly12BackwardCudaLauncher<scalar_t>(
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
  CHECK_CUDA(size_table);
  CHECK_CONTIGUOUS(size_table);
  CHECK_CUDA(arc_table);
  CHECK_CONTIGUOUS(arc_table);
  CHECK_FLAT_CUDA(size_count);
  CHECK_CONTIGUOUS(size_count);
  CHECK_CUDA(trans_tables);
  CHECK_CONTIGUOUS(trans_tables);
  CHECK_CUDA(cap_tables);
  CHECK_CONTIGUOUS(cap_tables);
  CHECK_CUDA(lut_values);
  CHECK_CONTIGUOUS(lut_values);
  CHECK_FLAT_CUDA(trans_dims);
  CHECK_CONTIGUOUS(trans_dims);
  CHECK_FLAT_CUDA(cap_dims);
  CHECK_CONTIGUOUS(cap_dims);
  CHECK_FLAT_CUDA(size);
  CHECK_CONTIGUOUS(size);
  CHECK_FLAT_CUDA(input_slew);
  CHECK_CONTIGUOUS(input_slew);
  CHECK_FLAT_CUDA(out_cap);
  CHECK_CONTIGUOUS(out_cap);
  auto output = at::zeros_like(size);
  DREAMPLACE_DISPATCH_FLOATING_TYPES(size, "cellModelingPiecewiseSizeForwardCudaLauncher", [&] {
    cellModelingPiecewiseSizeForwardCudaLauncher<scalar_t>(
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
  CHECK_FLAT_CUDA(grad_output);
  CHECK_CONTIGUOUS(grad_output);
  auto grad_size = at::zeros_like(size);
  auto grad_slew = at::zeros_like(input_slew);
  auto grad_cap = at::zeros_like(out_cap);
  auto grad_lut_values = at::zeros_like(lut_values);
  DREAMPLACE_DISPATCH_FLOATING_TYPES(size, "cellModelingPiecewiseSizeBackwardCudaLauncher", [&] {
    cellModelingPiecewiseSizeBackwardCudaLauncher<scalar_t>(
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
        "Cell modeling poly12 forward (CUDA)");
  m.def("backward", &DREAMPLACE_NAMESPACE::cell_modeling_poly12_backward,
        "Cell modeling poly12 backward (CUDA)");
  m.def("piecewise_size_forward", &DREAMPLACE_NAMESPACE::cell_modeling_piecewise_size_forward,
        "Cell modeling piecewise size forward (CUDA)");
  m.def("piecewise_size_backward", &DREAMPLACE_NAMESPACE::cell_modeling_piecewise_size_backward,
        "Cell modeling piecewise size backward (CUDA)");
}
