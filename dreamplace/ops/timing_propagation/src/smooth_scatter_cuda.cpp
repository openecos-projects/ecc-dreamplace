#include "utility/src/torch.h"
#include "utility/src/cuda_tensor.h"
#include <cmath>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "timing_scatter_groups.h"

template <typename scalar_t>
void smoothScatterCudaLauncher(
    const scalar_t*, const int64_t*, const scalar_t*, const scalar_t*,
    scalar_t*, scalar_t*, scalar_t*, scalar_t*, int, int, scalar_t, bool, cudaStream_t);
template <typename scalar_t>
void smoothScatterGroupedCudaLauncher(const scalar_t*, const int64_t*, const scalar_t*,
    const scalar_t*, const int64_t*, const int64_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*,
    int, int, scalar_t, bool, cudaStream_t);

std::vector<at::Tensor> smooth_scatter_cuda(
    at::Tensor dest, at::Tensor index, at::Tensor src, double tau, bool include_self,
    std::vector<at::Tensor> prepared_groups) {
  TORCH_CHECK(tau > 0, "smooth scatter temperature must be positive");
  for (const auto& tensor : {dest, index, src}) {
    TORCH_CHECK(tensor.is_cuda() && tensor.is_contiguous(), "smooth scatter expects contiguous CUDA tensors");
    TORCH_CHECK(tensor.device() == dest.device(), "smooth scatter device mismatch");
    TORCH_CHECK(tensor.dim() == 1, "smooth scatter expects vectors");
  }
  TORCH_CHECK(index.scalar_type() == at::kLong && index.sizes() == src.sizes(),
              "smooth scatter source/index mismatch");
  TORCH_CHECK(dest.scalar_type() == src.scalar_type(), "smooth scatter scalar type mismatch");
  const c10::cuda::CUDAGuard guard(dest.device());
  auto maxima = include_self ? dest.clone() : at::full_like(dest, -INFINITY);
  maxima.scatter_reduce_(0, index, src, "amax", true);
  auto sums = fullyWrittenCudaTensor(dest.sizes(), dest.options()), output = fullyWrittenCudaTensor(dest.sizes(), dest.options());
  auto dest_weight = fullyWrittenCudaTensor(dest.sizes(), dest.options()), src_weight = fullyWrittenCudaTensor(src.sizes(), src.options());
  const bool deterministic = at::globalContext().deterministicAlgorithms();
  TimingScatterGroups groups;
  if (deterministic) groups = timingScatterGroups(index, dest.numel(), prepared_groups);
  const auto stream = c10::cuda::getCurrentCUDAStream(dest.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES(dest.scalar_type(), "smoothScatterCuda", [&] {
    if (deterministic) {
      smoothScatterGroupedCudaLauncher<scalar_t>(dest.data_ptr<scalar_t>(), index.data_ptr<int64_t>(),
          src.data_ptr<scalar_t>(), maxima.data_ptr<scalar_t>(), groups.order.data_ptr<int64_t>(),
          groups.offsets.data_ptr<int64_t>(), sums.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
          dest_weight.data_ptr<scalar_t>(), src_weight.data_ptr<scalar_t>(),
          dest.numel(), src.numel(), static_cast<scalar_t>(tau), include_self, stream);
    } else {
      smoothScatterCudaLauncher<scalar_t>(
          dest.data_ptr<scalar_t>(), index.data_ptr<int64_t>(), src.data_ptr<scalar_t>(),
          maxima.data_ptr<scalar_t>(), sums.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(),
          dest_weight.data_ptr<scalar_t>(), src_weight.data_ptr<scalar_t>(),
          dest.numel(), src.numel(), static_cast<scalar_t>(tau), include_self, stream);
    }
  });
  return {output, dest_weight, src_weight};
}
