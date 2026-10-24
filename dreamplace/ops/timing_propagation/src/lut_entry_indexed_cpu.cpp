#include "utility/src/cpu_tensor.h"
#include "timing_cpu_checks.h"
#include "utility/src/torch.h"

#include "lut_query.h"
#include <ATen/Parallel.h>
#include <omp.h>

template <typename T>
void lutIndexedCpuLauncher(const T* trans, const T* caps, const int64_t* index,
    const T* trans_axis, const T* cap_axis, const T* values, const int64_t* nt,
    const int64_t* nc, T* out, T* dt, T* dc, int count, int rows, int ts, int cs, int vs,
    bool extrapolate) {
  const TimingLutTable<T> table{values, trans_axis, cap_axis, nt, nc, rows, vs, ts, cs};
  #pragma omp parallel for if(count >= 2048) num_threads(at::get_num_threads()) schedule(static)
  for (int i = 0; i < count; ++i) {
    const auto query = timingLutQuery(table, index[i], trans[i], caps[i], extrapolate);
    out[i] = query.value; dt[i] = query.dt; dc[i] = query.dc;
  }
}

std::vector<at::Tensor> lut_indexed_cpu_impl(
    at::Tensor trans, at::Tensor caps, at::Tensor indices,
    at::Tensor trans_axes, at::Tensor cap_axes, at::Tensor values,
    at::Tensor trans_dims, at::Tensor cap_dims, bool extrapolate_1d) {
  for (const auto& tensor : {trans, caps, indices, trans_axes, cap_axes, values, trans_dims, cap_dims}) {
    TORCH_CHECK(tensor.is_cpu() && tensor.is_contiguous(), "indexed LUT inputs must be contiguous CPU tensors");
    TORCH_CHECK(tensor.device() == trans.device(), "indexed LUT device mismatch");
  }
  for (const auto& tensor : {caps, trans_axes, cap_axes, values}) {
    TORCH_CHECK(tensor.scalar_type() == trans.scalar_type(), "indexed LUT scalar type mismatch");
  }
  for (const auto& tensor : {indices, trans_dims, cap_dims}) {
    TORCH_CHECK(tensor.scalar_type() == at::kLong, "indexed LUT indices/dimensions must be int64");
  }
  TORCH_CHECK(trans.dim() == 1 && caps.sizes() == trans.sizes() && indices.sizes() == trans.sizes(),
              "indexed LUT query shape mismatch");
  TORCH_CHECK(trans_axes.dim() == 2 && cap_axes.dim() == 2 && values.dim() == 2,
              "indexed LUT tables must be matrices");
  const auto rows = values.size(0);
  TORCH_CHECK(trans_axes.size(0) == rows && cap_axes.size(0) == rows &&
              trans_dims.numel() == rows && cap_dims.numel() == rows, "indexed LUT table row mismatch");
  checkCpuLutRows(indices, trans_axes, cap_axes, values, trans_dims, cap_dims);
  auto output = fullyWrittenCpuTensor(trans.sizes(), trans.options()), dt = fullyWrittenCpuTensor(trans.sizes(), trans.options()), dc = fullyWrittenCpuTensor(trans.sizes(), trans.options());
  AT_DISPATCH_FLOATING_TYPES(trans.scalar_type(), "lutIndexedCpu", [&] {
    lutIndexedCpuLauncher<scalar_t>(
        trans.data_ptr<scalar_t>(), caps.data_ptr<scalar_t>(), indices.data_ptr<int64_t>(),
        trans_axes.data_ptr<scalar_t>(), cap_axes.data_ptr<scalar_t>(), values.data_ptr<scalar_t>(),
        trans_dims.data_ptr<int64_t>(), cap_dims.data_ptr<int64_t>(),
        output.data_ptr<scalar_t>(), dt.data_ptr<scalar_t>(), dc.data_ptr<scalar_t>(),
        trans.numel(), rows, trans_axes.size(1), cap_axes.size(1), values.size(1), extrapolate_1d);
  });
  return {output, dt, dc};
}

std::vector<at::Tensor> lut_indexed_cpu(
    at::Tensor trans, at::Tensor caps, at::Tensor indices, at::Tensor trans_axes,
    at::Tensor cap_axes, at::Tensor values, at::Tensor trans_dims, at::Tensor cap_dims) {
  return lut_indexed_cpu_impl(trans, caps, indices, trans_axes, cap_axes, values, trans_dims, cap_dims, false);
}

std::vector<at::Tensor> buffer_lut_indexed_cpu(
    at::Tensor trans, at::Tensor caps, at::Tensor indices, at::Tensor trans_axes,
    at::Tensor cap_axes, at::Tensor values, at::Tensor trans_dims, at::Tensor cap_dims) {
  return lut_indexed_cpu_impl(trans, caps, indices, trans_axes, cap_axes, values, trans_dims, cap_dims, true);
}
