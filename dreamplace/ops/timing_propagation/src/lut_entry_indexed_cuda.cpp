#include "utility/src/torch.h"
#include "utility/src/cuda_tensor.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

template <typename scalar_t>
void lutIndexedCudaLauncher(
    const scalar_t*, const scalar_t*, const int64_t*, const scalar_t*,
    const scalar_t*, const scalar_t*, const int64_t*, const int64_t*,
    scalar_t*, scalar_t*, scalar_t*, int, int, int, int, int, bool, cudaStream_t);

static std::vector<at::Tensor> lut_indexed_cuda_impl(
    at::Tensor trans, at::Tensor caps, at::Tensor indices,
    at::Tensor trans_axes, at::Tensor cap_axes, at::Tensor values,
    at::Tensor trans_dims, at::Tensor cap_dims, bool extrapolate_1d) {
  for (const auto& tensor : {trans, caps, indices, trans_axes, cap_axes, values, trans_dims, cap_dims}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), "indexed LUT inputs must be contiguous CUDA tensors");
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
  const c10::cuda::CUDAGuard guard(trans.device());
  auto output = fullyWrittenCudaTensor(trans.sizes(), trans.options()), dt = fullyWrittenCudaTensor(trans.sizes(), trans.options()), dc = fullyWrittenCudaTensor(trans.sizes(), trans.options());
  const auto stream = c10::cuda::getCurrentCUDAStream(trans.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES(trans.scalar_type(), "lutIndexedCuda", [&] {
    lutIndexedCudaLauncher<scalar_t>(
        trans.data_ptr<scalar_t>(), caps.data_ptr<scalar_t>(), indices.data_ptr<int64_t>(),
        trans_axes.data_ptr<scalar_t>(), cap_axes.data_ptr<scalar_t>(), values.data_ptr<scalar_t>(),
        trans_dims.data_ptr<int64_t>(), cap_dims.data_ptr<int64_t>(),
        output.data_ptr<scalar_t>(), dt.data_ptr<scalar_t>(), dc.data_ptr<scalar_t>(),
        trans.numel(), rows, trans_axes.size(1), cap_axes.size(1), values.size(1), extrapolate_1d, stream);
  });
  return {output, dt, dc};
}

std::vector<at::Tensor> lut_indexed_cuda(
    at::Tensor trans, at::Tensor caps, at::Tensor indices, at::Tensor trans_axes,
    at::Tensor cap_axes, at::Tensor values, at::Tensor trans_dims, at::Tensor cap_dims) {
  return lut_indexed_cuda_impl(trans, caps, indices, trans_axes, cap_axes, values, trans_dims, cap_dims, false);
}

std::vector<at::Tensor> buffer_lut_indexed_cuda(
    at::Tensor trans, at::Tensor caps, at::Tensor indices, at::Tensor trans_axes,
    at::Tensor cap_axes, at::Tensor values, at::Tensor trans_dims, at::Tensor cap_dims) {
  return lut_indexed_cuda_impl(trans, caps, indices, trans_axes, cap_axes, values, trans_dims, cap_dims, true);
}
