#include "utility/src/cpu_tensor.h"
#include "timing_cpu_checks.h"
#include "utility/src/torch.h"
#include "timing_scatter_groups.h"

#include "net_aat_math.h"
#include <ATen/Parallel.h>
#include <omp.h>

template <typename scalar_t>
void netAatForwardCpuLauncher(const int64_t* arcs, int columns, int count, int pins,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    const scalar_t* rd, const scalar_t* fd, const scalar_t* ri, const scalar_t* fi,
    scalar_t* updates, scalar_t* jac, scalar_t eps) {
  #pragma omp parallel for if(count >= 2048) num_threads(at::get_num_threads()) schedule(static)
  for (int arc = 0; arc < count; ++arc) {
    netForwardRow<scalar_t>(arc, arcs, columns, count, pins, ra, fa, rs, fs, rd, fd, ri, fi, updates, jac, eps);
  }
}
template <typename scalar_t>
void netAatBackwardCpuLauncher(const int64_t* input_order, const int64_t* input_offsets,
    const int64_t* output_order, const int64_t* output_offsets,
    const scalar_t* jac, const scalar_t* gradient,
    scalar_t* ra, scalar_t* fa, scalar_t* rs, scalar_t* fs,
    scalar_t* rd, scalar_t* fd, scalar_t* ri, scalar_t* fi, int pins,
    bool rise_slew_active, bool fall_slew_active) {
  #pragma omp parallel for if(pins >= 2048) num_threads(at::get_num_threads()) schedule(static)
  for (int pin = 0; pin < pins; ++pin) {
    netBackwardRow<scalar_t>(pin, input_order, input_offsets, output_order, output_offsets, jac, gradient, ra, fa, rs, fs, rd, fd, ri, fi, pins, rise_slew_active, fall_slew_active);
  }
}

namespace {
void requireTensor(at::Tensor value, at::Tensor reference, at::ScalarType dtype) {
  TORCH_CHECK(value.is_cpu() && value.device() == reference.device() && value.is_contiguous(),
              "net AAT expects contiguous tensors on one CPU device");
  TORCH_CHECK(value.scalar_type() == dtype, "net AAT dtype mismatch");
}
}  // namespace

std::vector<at::Tensor> net_aat_forward_cpu(
    at::Tensor arcs, at::Tensor ra, at::Tensor fa, at::Tensor rs, at::Tensor fs,
    at::Tensor rd, at::Tensor fd, at::Tensor ri, at::Tensor fi, double eps) {
  requireTensor(arcs, ra, at::kLong);
  TORCH_CHECK(arcs.dim() == 2 && arcs.size(1) >= 2 && eps > 0,
              "net AAT arc/epsilon mismatch");
  for (const auto& pin : {ra, fa, rs, fs, rd, fd, ri, fi}) {
    requireTensor(pin, ra, ra.scalar_type());
    TORCH_CHECK(pin.dim() == 1 && pin.sizes() == ra.sizes(), "net AAT pin shape mismatch");
  }
  checkCpuArcIndices(arcs, ra.numel());
  auto updates = fullyWrittenCpuTensor({arcs.size(0), 4}, ra.options());
  auto jac = fullyWrittenCpuTensor({arcs.size(0), 6}, ra.options());
  AT_DISPATCH_FLOATING_TYPES(ra.scalar_type(), "netAatForward", [&] {
    netAatForwardCpuLauncher<scalar_t>(arcs.data_ptr<int64_t>(), arcs.size(1), arcs.size(0), ra.numel(),
        ra.data_ptr<scalar_t>(), fa.data_ptr<scalar_t>(), rs.data_ptr<scalar_t>(), fs.data_ptr<scalar_t>(),
        rd.data_ptr<scalar_t>(), fd.data_ptr<scalar_t>(), ri.data_ptr<scalar_t>(), fi.data_ptr<scalar_t>(),
        updates.data_ptr<scalar_t>(), jac.data_ptr<scalar_t>(), scalar_t(eps));
  });
  return {updates, jac};
}

std::vector<at::Tensor> net_aat_backward_cpu(
    at::Tensor arcs, at::Tensor jac, at::Tensor gradient, int64_t pins,
    std::vector<at::Tensor> prepared_groups, bool rise_slew_active, bool fall_slew_active) {
  requireTensor(arcs, jac, at::kLong);
  requireTensor(gradient, jac, jac.scalar_type());
  requireTensor(jac, jac, jac.scalar_type());
  TORCH_CHECK(jac.dim() == 2 && jac.size(0) == arcs.size(0) && jac.size(1) == 6 &&
              gradient.dim() == 2 && gradient.size(0) == arcs.size(0) && gradient.size(1) == 4,
              "net AAT adjoint shape mismatch");
  checkCpuArcIndices(arcs, pins);
  auto inputs = timingScatterGroups(arcs.select(1, 0), pins, prepared_groups);
  auto outputs = timingScatterGroups(arcs.select(1, 1), pins, prepared_groups, 2);
  std::vector<at::Tensor> result;
  for (int i = 0; i < 8; ++i) result.push_back(fullyWrittenCpuTensor({pins}, jac.options()));
  AT_DISPATCH_FLOATING_TYPES(jac.scalar_type(), "netAatBackward", [&] {
    netAatBackwardCpuLauncher<scalar_t>(inputs.order.data_ptr<int64_t>(), inputs.offsets.data_ptr<int64_t>(),
        outputs.order.data_ptr<int64_t>(), outputs.offsets.data_ptr<int64_t>(),
        jac.data_ptr<scalar_t>(), gradient.data_ptr<scalar_t>(), result[0].data_ptr<scalar_t>(),
        result[1].data_ptr<scalar_t>(), result[2].data_ptr<scalar_t>(), result[3].data_ptr<scalar_t>(),
        result[4].data_ptr<scalar_t>(), result[5].data_ptr<scalar_t>(), result[6].data_ptr<scalar_t>(),
        result[7].data_ptr<scalar_t>(), pins, rise_slew_active, fall_slew_active);
  });
  return result;
}
