#include "utility/src/torch.h"
#include "utility/src/cuda_tensor.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

template <typename scalar_t>
void bufferInterpolateForwardLauncher(const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*,
    scalar_t*, scalar_t*, int64_t*, int64_t*, scalar_t*, scalar_t*, int32_t*, int, int, cudaStream_t);
template <typename scalar_t>
void bufferInterpolateBackwardLauncher(const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*,
    const int64_t*, const int64_t*, const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*,
    scalar_t*, scalar_t*, scalar_t*, scalar_t*, int, int, cudaStream_t);

namespace {
void requireTensor(at::Tensor value, at::Tensor reference, at::ScalarType dtype) {
  TORCH_CHECK(value.is_cuda() && value.device() == reference.device() && value.is_contiguous() &&
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

std::vector<at::Tensor> buffer_interpolate_forward_cuda(
    at::Tensor index, at::Tensor cap, at::Tensor delay, at::Tensor slew) {
  checkTables(index, {cap, delay, slew});
  const c10::cuda::CUDAGuard guard(index.device());
  auto values = fullyWrittenCudaTensor({index.numel(), 3}, index.options());
  auto clipped = fullyWrittenCudaTensor(index.sizes(), index.options()), alpha = fullyWrittenCudaTensor(index.sizes(), index.options());
  auto lo = fullyWrittenCudaTensor(index.sizes(), index.options().dtype(at::kLong));
  auto hi = fullyWrittenCudaTensor(lo.sizes(), lo.options());
  auto metadata = fullyWrittenCudaTensor({index.numel() + 1}, index.options());
  auto flags = at::zeros({1}, index.options().dtype(at::kInt));
  const auto stream = c10::cuda::getCurrentCUDAStream(index.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES(index.scalar_type(), "bufferInterpolateForward", [&] {
    bufferInterpolateForwardLauncher<scalar_t>(index.data_ptr<scalar_t>(), cap.data_ptr<scalar_t>(),
        delay.data_ptr<scalar_t>(), slew.data_ptr<scalar_t>(), values.data_ptr<scalar_t>(),
        clipped.data_ptr<scalar_t>(), lo.data_ptr<int64_t>(), hi.data_ptr<int64_t>(),
        alpha.data_ptr<scalar_t>(), metadata.data_ptr<scalar_t>(), flags.data_ptr<int32_t>(),
        index.numel(), cap.size(1), stream);
  });
  return {values, clipped, lo, hi, alpha, metadata};
}

std::vector<at::Tensor> buffer_interpolate_backward_cuda(
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
  const c10::cuda::CUDAGuard guard(index.device());
  auto grad_index = fullyWrittenCudaTensor(index.sizes(), index.options()), grad_cap = fullyWrittenCudaTensor(cap.sizes(), cap.options());
  auto grad_delay = fullyWrittenCudaTensor(delay.sizes(), delay.options()), grad_slew = fullyWrittenCudaTensor(slew.sizes(), slew.options());
  const auto stream = c10::cuda::getCurrentCUDAStream(index.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES(index.scalar_type(), "bufferInterpolateBackward", [&] {
    bufferInterpolateBackwardLauncher<scalar_t>(index.data_ptr<scalar_t>(), cap.data_ptr<scalar_t>(),
        delay.data_ptr<scalar_t>(), slew.data_ptr<scalar_t>(), lo.data_ptr<int64_t>(), hi.data_ptr<int64_t>(),
        alpha.data_ptr<scalar_t>(), grad_values.data_ptr<scalar_t>(), grad_clipped.data_ptr<scalar_t>(),
        grad_alpha.data_ptr<scalar_t>(), grad_index.data_ptr<scalar_t>(), grad_cap.data_ptr<scalar_t>(),
        grad_delay.data_ptr<scalar_t>(), grad_slew.data_ptr<scalar_t>(), index.numel(), cap.size(1), stream);
  });
  return {grad_index, grad_cap, grad_delay, grad_slew};
}
