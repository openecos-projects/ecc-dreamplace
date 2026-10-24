#include "utility/src/torch.h"
#include "utility/src/cuda_tensor.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "timing_scatter_groups.h"

template <typename scalar_t>
void netAatForwardLauncher(const int64_t*, int, int, int,
    const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*,
    const scalar_t*, const scalar_t*, const scalar_t*, scalar_t*, scalar_t*, scalar_t, cudaStream_t);
template <typename scalar_t>
void netAatBackwardLauncher(const int64_t*, const int64_t*, const int64_t*, const int64_t*,
    const scalar_t*, const scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*,
    scalar_t*, scalar_t*, scalar_t*, scalar_t*, int, bool, bool, cudaStream_t);

namespace {
void requireTensor(at::Tensor value, at::Tensor reference, at::ScalarType dtype) {
  TORCH_CHECK(value.is_cuda() && value.device() == reference.device() && value.is_contiguous(),
              "net AAT expects contiguous tensors on one CUDA device");
  TORCH_CHECK(value.scalar_type() == dtype, "net AAT dtype mismatch");
}
}  // namespace

std::vector<at::Tensor> net_aat_forward_cuda(
    at::Tensor arcs, at::Tensor ra, at::Tensor fa, at::Tensor rs, at::Tensor fs,
    at::Tensor rd, at::Tensor fd, at::Tensor ri, at::Tensor fi, double eps) {
  requireTensor(arcs, ra, at::kLong);
  TORCH_CHECK(arcs.dim() == 2 && arcs.size(1) >= 2 && eps > 0,
              "net AAT arc/epsilon mismatch");
  for (const auto& pin : {ra, fa, rs, fs, rd, fd, ri, fi}) {
    requireTensor(pin, ra, ra.scalar_type());
    TORCH_CHECK(pin.dim() == 1 && pin.sizes() == ra.sizes(), "net AAT pin shape mismatch");
  }
  const c10::cuda::CUDAGuard guard(ra.device());
  auto updates = fullyWrittenCudaTensor({arcs.size(0), 4}, ra.options());
  auto jac = fullyWrittenCudaTensor({arcs.size(0), 6}, ra.options());
  const auto stream = c10::cuda::getCurrentCUDAStream(ra.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES(ra.scalar_type(), "netAatForward", [&] {
    netAatForwardLauncher<scalar_t>(arcs.data_ptr<int64_t>(), arcs.size(1), arcs.size(0), ra.numel(),
        ra.data_ptr<scalar_t>(), fa.data_ptr<scalar_t>(), rs.data_ptr<scalar_t>(), fs.data_ptr<scalar_t>(),
        rd.data_ptr<scalar_t>(), fd.data_ptr<scalar_t>(), ri.data_ptr<scalar_t>(), fi.data_ptr<scalar_t>(),
        updates.data_ptr<scalar_t>(), jac.data_ptr<scalar_t>(), scalar_t(eps), stream);
  });
  return {updates, jac};
}

std::vector<at::Tensor> net_aat_backward_cuda(
    at::Tensor arcs, at::Tensor jac, at::Tensor gradient, int64_t pins,
    std::vector<at::Tensor> prepared_groups, bool rise_slew_active, bool fall_slew_active) {
  requireTensor(arcs, jac, at::kLong);
  requireTensor(gradient, jac, jac.scalar_type());
  requireTensor(jac, jac, jac.scalar_type());
  TORCH_CHECK(jac.dim() == 2 && jac.size(0) == arcs.size(0) && jac.size(1) == 6 &&
              gradient.dim() == 2 && gradient.size(0) == arcs.size(0) && gradient.size(1) == 4,
              "net AAT adjoint shape mismatch");
  const c10::cuda::CUDAGuard guard(jac.device());
  auto inputs = timingScatterGroups(arcs.select(1, 0), pins, prepared_groups);
  auto outputs = timingScatterGroups(arcs.select(1, 1), pins, prepared_groups, 2);
  std::vector<at::Tensor> result;
  for (int i = 0; i < 8; ++i) result.push_back(fullyWrittenCudaTensor({pins}, jac.options()));
  const auto stream = c10::cuda::getCurrentCUDAStream(jac.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES(jac.scalar_type(), "netAatBackward", [&] {
    netAatBackwardLauncher<scalar_t>(inputs.order.data_ptr<int64_t>(), inputs.offsets.data_ptr<int64_t>(),
        outputs.order.data_ptr<int64_t>(), outputs.offsets.data_ptr<int64_t>(),
        jac.data_ptr<scalar_t>(), gradient.data_ptr<scalar_t>(), result[0].data_ptr<scalar_t>(),
        result[1].data_ptr<scalar_t>(), result[2].data_ptr<scalar_t>(), result[3].data_ptr<scalar_t>(),
        result[4].data_ptr<scalar_t>(), result[5].data_ptr<scalar_t>(), result[6].data_ptr<scalar_t>(),
        result[7].data_ptr<scalar_t>(), pins, rise_slew_active, fall_slew_active, stream);
  });
  return result;
}
