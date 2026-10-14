#include "utility/src/torch.h"
#include "utility/src/cuda_tensor.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <limits>

template <typename T>
void hardScatterForwardLauncher(const T*, const int64_t*, const T*, const T*,
                                T*, int32_t*, int, int, cudaStream_t);
template <typename T>
void hardScatterBackwardLauncher(const T*, const int64_t*, const T*, const T*,
                                 const T*, const int32_t*, const T*, T*, T*,
                                 int, int, bool, cudaStream_t);

namespace {
void check_inputs(const at::Tensor& dest, const at::Tensor& index, const at::Tensor& src) {
  for (const auto& tensor : {dest, index, src}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous() && tensor.dim() == 1,
                "hard scatter expects contiguous CUDA vectors");
    TORCH_CHECK(tensor.device() == dest.device(), "hard scatter device mismatch");
  }
  TORCH_CHECK(index.scalar_type() == at::kLong && index.sizes() == src.sizes(),
              "hard scatter source/index mismatch");
  TORCH_CHECK(dest.scalar_type() == src.scalar_type(), "hard scatter scalar type mismatch");
  TORCH_CHECK(dest.numel() <= std::numeric_limits<int>::max() &&
              src.numel() <= std::numeric_limits<int>::max(), "hard scatter index domain exceeded");
}
}

std::vector<at::Tensor> hard_scatter_forward_cuda(
    at::Tensor dest, at::Tensor index, at::Tensor src, bool include_self) {
  check_inputs(dest, index, src);
  const c10::cuda::CUDAGuard guard(dest.device());
  // Keep Torch's forward reduction, including untouched destinations and NaNs.
  auto output = dest.clone();
  output.scatter_reduce_(0, index, src, "amax", include_self);
  auto counts = fullyWrittenCudaTensor(dest.sizes(), dest.options());
  auto touched = fullyWrittenCudaTensor(dest.sizes(), dest.options().dtype(at::kInt));
  const auto stream = c10::cuda::getCurrentCUDAStream(dest.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES(dest.scalar_type(), "hardScatterForward", [&] {
    hardScatterForwardLauncher<scalar_t>(
        dest.data_ptr<scalar_t>(), index.data_ptr<int64_t>(), src.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(), counts.data_ptr<scalar_t>(), touched.data_ptr<int32_t>(),
        dest.numel(), src.numel(), stream);
  });
  return {output, counts, touched};
}

std::vector<at::Tensor> hard_scatter_backward_cuda(
    at::Tensor dest, at::Tensor index, at::Tensor src, at::Tensor output,
    at::Tensor counts, at::Tensor touched, at::Tensor gradient, bool include_self) {
  check_inputs(dest, index, src);
  for (const auto& tensor : {output, counts, touched, gradient}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.device() == dest.device() &&
                tensor.is_contiguous() && tensor.sizes() == dest.sizes(),
                "hard scatter adjoint vector mismatch");
  }
  TORCH_CHECK(output.scalar_type() == dest.scalar_type() && counts.scalar_type() == dest.scalar_type() &&
              gradient.scalar_type() == dest.scalar_type() && touched.scalar_type() == at::kInt,
              "hard scatter adjoint dtype mismatch");
  const c10::cuda::CUDAGuard guard(dest.device());
  auto grad_dest = fullyWrittenCudaTensor(dest.sizes(), dest.options()), grad_src = fullyWrittenCudaTensor(src.sizes(), src.options());
  const auto stream = c10::cuda::getCurrentCUDAStream(dest.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES(dest.scalar_type(), "hardScatterBackward", [&] {
    hardScatterBackwardLauncher<scalar_t>(
        dest.data_ptr<scalar_t>(), index.data_ptr<int64_t>(), src.data_ptr<scalar_t>(),
        output.data_ptr<scalar_t>(), counts.data_ptr<scalar_t>(), touched.data_ptr<int32_t>(),
        gradient.data_ptr<scalar_t>(), grad_dest.data_ptr<scalar_t>(), grad_src.data_ptr<scalar_t>(),
        dest.numel(), src.numel(), include_self, stream);
  });
  return {grad_dest, grad_src};
}
