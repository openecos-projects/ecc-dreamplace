#include "utility/src/torch.h"
#include "utility/src/utils.h"
#include "segment_energy_gradient.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
void launchSegmentEnergyGradient(const T* positions, const T* sx, const T* sy,
                                const T* ratios, const T* weights, const T* adjoint,
                                const T* adjoint_v, const bool* directions,
                                int count, int nx, int ny, T xl, T yl, T bin_x, T bin_y,
                                T* gp, T* gx, T* gy) {
#pragma omp parallel for num_threads(at::get_num_threads()) schedule(static)
  for (int i = 0; i < count; ++i) {
    T gradient[4];
    const T* source = directions && !directions[i] ? adjoint_v : adjoint;
    segmentEnergyGradient(positions[i], positions[i + count], sx[i], sy[i],
                          ratios[i], weights[i], source, nx, ny,
                          xl, yl, bin_x, bin_y, gradient);
    gp[i] = gradient[0];
    gp[i + count] = gradient[1];
    gx[i] = gradient[2];
    gy[i] = gradient[3];
  }
}

std::vector<at::Tensor> segment_energy_backward(
    at::Tensor pos, at::Tensor size_x, at::Tensor size_y,
    at::Tensor ratio, at::Tensor weight, at::Tensor psi, at::Tensor psi_v,
    at::Tensor directions,
    double xl, double yl, double bin_x, double bin_y) {
  CHECK_FLAT_CPU(pos);
  CHECK_EVEN(pos);
  CHECK_CONTIGUOUS(pos);
  const int count = pos.numel() / 2;
  for (const auto& tensor : {size_x, size_y, ratio, weight}) {
    CHECK_FLAT_CPU(tensor);
    CHECK_CONTIGUOUS(tensor);
    TORCH_CHECK(tensor.numel() == count && tensor.scalar_type() == pos.scalar_type(),
                "segment array must match position count and dtype");
  }
  CHECK_CPU(psi);
  CHECK_CONTIGUOUS(psi);
  TORCH_CHECK(psi.dim() == 2 && psi.scalar_type() == pos.scalar_type(),
              "demand adjoint must be a 2D map matching position dtype");
  CHECK_FLAT_CPU(directions);
  CHECK_CONTIGUOUS(directions);
  TORCH_CHECK(directions.scalar_type() == at::ScalarType::Bool &&
              (directions.numel() == 0 || directions.numel() == count), "invalid direction array");
  CHECK_CPU(psi_v);
  CHECK_CONTIGUOUS(psi_v);
  TORCH_CHECK(psi_v.sizes() == psi.sizes() && psi_v.scalar_type() == pos.scalar_type(),
              "directional adjoint maps must match");
  TORCH_CHECK(bin_x > 0 && bin_y > 0, "bin sizes must be positive");
  auto grad_pos = at::empty_like(pos);
  auto grad_x = at::empty_like(size_x);
  auto grad_y = at::empty_like(size_y);
  const int nx = psi.size(0), ny = psi.size(1);
  DREAMPLACE_DISPATCH_FLOATING_TYPES(pos, "segment_energy_backward", [&] {
    const auto* positions = pos.data_ptr<scalar_t>();
    const auto* sx = size_x.data_ptr<scalar_t>();
    const auto* sy = size_y.data_ptr<scalar_t>();
    const auto* ratios = ratio.data_ptr<scalar_t>();
    const auto* weights = weight.data_ptr<scalar_t>();
    const auto* adjoint = psi.data_ptr<scalar_t>();
    auto* gp = grad_pos.data_ptr<scalar_t>();
    auto* gx = grad_x.data_ptr<scalar_t>();
    auto* gy = grad_y.data_ptr<scalar_t>();
    launchSegmentEnergyGradient(positions, sx, sy, ratios, weights, adjoint,
                                psi_v.data_ptr<scalar_t>(),
                                directions.numel() ? directions.data_ptr<bool>() : nullptr,
                                count, nx, ny, scalar_t(xl), scalar_t(yl),
                                scalar_t(bin_x), scalar_t(bin_y), gp, gx, gy);
  });
  return {grad_pos, grad_x, grad_y};
}

DREAMPLACE_END_NAMESPACE

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("backward", &DREAMPLACE_NAMESPACE::segment_energy_backward,
        "Complete rectangle energy derivative through position and dimensions");
}
