#include "utility/src/torch.h"
#include "utility/src/utils.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
cudaError_t launchSegmentEnergyGradientCUDA(
    const T*, const T*, const T*, const T*, const T*, const T*, const T*, const bool*,
    int, int, int, T, T, T, T, T*, T*, T*, cudaStream_t);

std::vector<at::Tensor> segment_energy_backward_cuda(
    at::Tensor pos, at::Tensor size_x, at::Tensor size_y,
    at::Tensor ratio, at::Tensor weight, at::Tensor psi, at::Tensor psi_v,
    at::Tensor directions,
    double xl, double yl, double bin_x, double bin_y) {
  CHECK_FLAT_CUDA(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  const int count = pos.numel() / 2;
  for (const auto& tensor : {size_x, size_y, ratio, weight}) {
    CHECK_FLAT_CUDA(tensor);
    CHECK_CONTIGUOUS(tensor);
    TORCH_CHECK(tensor.numel() == count && tensor.scalar_type() == pos.scalar_type()
                && tensor.device() == pos.device(), "segment arrays must match position");
  }
  CHECK_CUDA(psi);
  CHECK_CONTIGUOUS(psi);
  TORCH_CHECK(psi.dim() == 2 && psi.scalar_type() == pos.scalar_type()
              && psi.device() == pos.device(), "demand adjoint must match position device and dtype");
  CHECK_FLAT_CUDA(directions);
  CHECK_CONTIGUOUS(directions);
  TORCH_CHECK(directions.scalar_type() == at::ScalarType::Bool && directions.device() == pos.device()
              && (directions.numel() == 0 || directions.numel() == count), "invalid direction array");
  CHECK_CUDA(psi_v);
  CHECK_CONTIGUOUS(psi_v);
  TORCH_CHECK(psi_v.sizes() == psi.sizes() && psi_v.scalar_type() == pos.scalar_type()
              && psi_v.device() == pos.device(), "directional adjoint maps must match");
  TORCH_CHECK(bin_x > 0 && bin_y > 0, "bin sizes must be positive");
  c10::cuda::CUDAGuard device_guard(pos.device());
  auto gp = at::empty_like(pos), gx = at::empty_like(size_x), gy = at::empty_like(size_y);
  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "segment_energy_backward_cuda", [&] {
    C10_CUDA_CHECK(launchSegmentEnergyGradientCUDA(
        pos.data_ptr<scalar_t>(), size_x.data_ptr<scalar_t>(), size_y.data_ptr<scalar_t>(),
        ratio.data_ptr<scalar_t>(), weight.data_ptr<scalar_t>(), psi.data_ptr<scalar_t>(),
        psi_v.data_ptr<scalar_t>(), directions.numel() ? directions.data_ptr<bool>() : nullptr,
        count, int(psi.size(0)), int(psi.size(1)), scalar_t(xl), scalar_t(yl),
        scalar_t(bin_x), scalar_t(bin_y), gp.data_ptr<scalar_t>(), gx.data_ptr<scalar_t>(),
        gy.data_ptr<scalar_t>(), c10::cuda::getCurrentCUDAStream()));
  });
  return {gp, gx, gy};
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("backward", &DREAMPLACE_NAMESPACE::segment_energy_backward_cuda,
        "Complete rectangle energy derivative on CUDA");
}
