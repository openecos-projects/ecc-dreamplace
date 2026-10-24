#include "utility/src/cpu_tensor.h"
#include "utility/src/torch.h"

#include "buffer_interpolate_math.h"
#include <ATen/Parallel.h>
#include <omp.h>

template <typename scalar_t>
void bufferInterpolateForwardCpuLauncher(const scalar_t* index, const scalar_t* cap, const scalar_t* delay,
    const scalar_t* slew, scalar_t* values, scalar_t* clipped, int64_t* lo, int64_t* hi,
    scalar_t* alpha, scalar_t* metadata, int32_t* flags, int rows, int sizes) {
  int invalid = 0;
  #pragma omp parallel for if(rows >= 2048) num_threads(at::get_num_threads()) schedule(static) reduction(|:invalid)
  for (int row = 0; row < rows; ++row) {
    invalid |= interpolateRow<scalar_t>(row, index, cap, delay, slew, values, clipped, lo, hi, alpha, metadata, rows, sizes);
  }
  *flags = invalid;
  metadata[0] = scalar_t(invalid);
}
template <typename scalar_t>
void bufferInterpolateBackwardCpuLauncher(const scalar_t* index, const scalar_t* cap, const scalar_t* delay,
    const scalar_t* slew, const int64_t* lo, const int64_t* hi, const scalar_t* alpha,
    const scalar_t* grad_values, const scalar_t* grad_clipped, const scalar_t* grad_alpha,
    scalar_t* grad_index, scalar_t* grad_cap, scalar_t* grad_delay, scalar_t* grad_slew,
    int rows, int sizes) {
  #pragma omp parallel for if(rows >= 2048) num_threads(at::get_num_threads()) schedule(static)
  for (int row = 0; row < rows; ++row) {
    interpolateBackwardRow<scalar_t>(row, index, cap, delay, slew, lo, hi, alpha, grad_values, grad_clipped, grad_alpha, grad_index, grad_cap, grad_delay, grad_slew, rows, sizes);
  }
}

namespace {
void requireTensor(at::Tensor value, at::Tensor reference, at::ScalarType dtype) {
  TORCH_CHECK(value.is_cpu() && value.device() == reference.device() && value.is_contiguous() &&
              value.scalar_type() == dtype, "buffer interpolation tensor mismatch");
}

void checkTables(at::Tensor index, const std::vector<at::Tensor>& tables) {
  requireTensor(index, index, index.scalar_type());
  TORCH_CHECK(index.dim() == 1, "buffer interpolation index must be a vector");
  for (const auto& table : tables) {
    requireTensor(table, index, index.scalar_type());
    TORCH_CHECK(table.dim() == 2 && table.size(0) == index.numel() && table.size(1) >= 2 &&
                table.sizes() == tables[0].sizes(), "buffer interpolation table shape mismatch");
  }
}
}  // namespace

std::vector<at::Tensor> buffer_interpolate_forward_cpu(
    at::Tensor index, at::Tensor cap, at::Tensor delay, at::Tensor slew) {
  checkTables(index, {cap, delay, slew});
  auto values = fullyWrittenCpuTensor({index.numel(), 3}, index.options());
  auto clipped = fullyWrittenCpuTensor(index.sizes(), index.options()), alpha = fullyWrittenCpuTensor(index.sizes(), index.options());
  auto lo = fullyWrittenCpuTensor(index.sizes(), index.options().dtype(at::kLong));
  auto hi = fullyWrittenCpuTensor(lo.sizes(), lo.options());
  auto metadata = fullyWrittenCpuTensor({index.numel() + 1}, index.options());
  auto flags = at::zeros({1}, index.options().dtype(at::kInt));
  AT_DISPATCH_FLOATING_TYPES(index.scalar_type(), "bufferInterpolateForward", [&] {
    bufferInterpolateForwardCpuLauncher<scalar_t>(index.data_ptr<scalar_t>(), cap.data_ptr<scalar_t>(),
        delay.data_ptr<scalar_t>(), slew.data_ptr<scalar_t>(), values.data_ptr<scalar_t>(),
        clipped.data_ptr<scalar_t>(), lo.data_ptr<int64_t>(), hi.data_ptr<int64_t>(),
        alpha.data_ptr<scalar_t>(), metadata.data_ptr<scalar_t>(), flags.data_ptr<int32_t>(),
        index.numel(), cap.size(1));
  });
  return {values, clipped, lo, hi, alpha, metadata};
}

std::vector<at::Tensor> buffer_interpolate_backward_cpu(
    at::Tensor index, at::Tensor cap, at::Tensor delay, at::Tensor slew,
    at::Tensor lo, at::Tensor hi, at::Tensor alpha,
    at::Tensor grad_values, at::Tensor grad_clipped, at::Tensor grad_alpha) {
  checkTables(index, {cap, delay, slew});
  for (const auto& value : {lo, hi}) {
    requireTensor(value, index, at::kLong);
    TORCH_CHECK(value.sizes() == index.sizes(), "buffer interpolation index adjoint shape mismatch");
  }
  for (const auto& value : {alpha, grad_clipped, grad_alpha}) {
    requireTensor(value, index, index.scalar_type());
    TORCH_CHECK(value.sizes() == index.sizes(), "buffer interpolation scalar adjoint shape mismatch");
  }
  requireTensor(grad_values, index, index.scalar_type());
  TORCH_CHECK(grad_values.dim() == 2 && grad_values.size(0) == index.numel() && grad_values.size(1) == 3,
              "buffer interpolation value adjoint shape mismatch");
  auto grad_index = fullyWrittenCpuTensor(index.sizes(), index.options()), grad_cap = fullyWrittenCpuTensor(cap.sizes(), cap.options());
  auto grad_delay = fullyWrittenCpuTensor(delay.sizes(), delay.options()), grad_slew = fullyWrittenCpuTensor(slew.sizes(), slew.options());
  AT_DISPATCH_FLOATING_TYPES(index.scalar_type(), "bufferInterpolateBackward", [&] {
    bufferInterpolateBackwardCpuLauncher<scalar_t>(index.data_ptr<scalar_t>(), cap.data_ptr<scalar_t>(),
        delay.data_ptr<scalar_t>(), slew.data_ptr<scalar_t>(), lo.data_ptr<int64_t>(), hi.data_ptr<int64_t>(),
        alpha.data_ptr<scalar_t>(), grad_values.data_ptr<scalar_t>(), grad_clipped.data_ptr<scalar_t>(),
        grad_alpha.data_ptr<scalar_t>(), grad_index.data_ptr<scalar_t>(), grad_cap.data_ptr<scalar_t>(),
        grad_delay.data_ptr<scalar_t>(), grad_slew.data_ptr<scalar_t>(), index.numel(), cap.size(1));
  });
  return {grad_index, grad_cap, grad_delay, grad_slew};
}
