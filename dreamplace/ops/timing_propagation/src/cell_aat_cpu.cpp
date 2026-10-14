#include "utility/src/cpu_tensor.h"
#include "timing_cpu_checks.h"
#include "utility/src/torch.h"
#include "lut_query.h"
#include "timing_scatter_groups.h"

#include "cell_aat_math.h"
#include <ATen/Parallel.h>
#include <omp.h>

template <typename scalar_t>
void cellAatForwardCpuLauncher(
    const int64_t* arcs, int columns, int count, int pins,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    const scalar_t* rc, const scalar_t* fc, TimingLutTable<scalar_t> rd,
    TimingLutTable<scalar_t> fd, TimingLutTable<scalar_t> rt, TimingLutTable<scalar_t> ft,
    scalar_t* updates, scalar_t* derivatives, bool smooth, scalar_t alpha) {
  #pragma omp parallel for if(count >= 2048) num_threads(at::get_num_threads()) schedule(static)
  for (int i = 0; i < count; ++i) {
    cellForwardRow<scalar_t>(i, arcs, columns, count, pins, ra, fa, rs, fs, rc, fc, rd, fd, rt, ft, updates, derivatives, smooth, alpha);
  }
}
template <typename scalar_t>
void cellAatGroupedBackwardCpuLauncher(
    const int64_t* input_order, const int64_t* input_offsets,
    const int64_t* output_order, const int64_t* output_offsets,
    const scalar_t* jac, const scalar_t* gradient,
    scalar_t* ra, scalar_t* fa, scalar_t* rs, scalar_t* fs, scalar_t* rc, scalar_t* fc,
    int pins) {
  #pragma omp parallel for if(pins >= 2048) num_threads(at::get_num_threads()) schedule(static)
  for (int pin = 0; pin < pins; ++pin) {
    cellGroupedBackwardRow<scalar_t>(pin, input_order, input_offsets, output_order, output_offsets, jac, gradient, ra, fa, rs, fs, rc, fc, pins);
  }
}

namespace {
void requireTensor(at::Tensor value, at::Tensor reference, at::ScalarType dtype) {
  TORCH_CHECK(value.is_cpu() && value.device() == reference.device() && value.is_contiguous(),
              "cell AAT expects contiguous tensors on one CPU device");
  TORCH_CHECK(value.scalar_type() == dtype, "cell AAT dtype mismatch");
}

template <typename scalar_t>
TimingLutTable<scalar_t> table(const std::vector<at::Tensor>& values, int offset) {
  return {values[offset].data_ptr<scalar_t>(), values[offset + 1].data_ptr<scalar_t>(),
      values[offset + 2].data_ptr<scalar_t>(), values[offset + 3].data_ptr<int64_t>(),
      values[offset + 4].data_ptr<int64_t>(), static_cast<int>(values[offset].size(0)),
      static_cast<int>(values[offset].size(1)), static_cast<int>(values[offset + 1].size(1)),
      static_cast<int>(values[offset + 2].size(1))};
}
}  // namespace

std::vector<at::Tensor> cell_aat_forward_cpu(
    at::Tensor arcs, at::Tensor ra, at::Tensor fa, at::Tensor rs, at::Tensor fs,
    at::Tensor rc, at::Tensor fc, std::vector<at::Tensor> tables, bool smooth, double alpha) {
  requireTensor(arcs, ra, at::kLong);
  TORCH_CHECK(arcs.dim() == 2 && arcs.size(1) >= 5, "cell AAT arcs need at least five columns");
  for (const auto& pin : {ra, fa, rs, fs, rc, fc}) {
    requireTensor(pin, ra, ra.scalar_type());
    TORCH_CHECK(pin.dim() == 1 && pin.sizes() == ra.sizes(), "cell AAT pin shape mismatch");
  }
  TORCH_CHECK(tables.size() == 20 && alpha > 0, "cell AAT LUT/temperature mismatch");
  for (int i = 0; i < 20; ++i) {
    requireTensor(tables[i], ra, i % 5 < 3 ? ra.scalar_type() : at::kLong);
    TORCH_CHECK(tables[i].dim() == (i % 5 < 3 ? 2 : 1), "cell AAT LUT shape mismatch");
    TORCH_CHECK(tables[i].size(0) == tables[i - i % 5].size(0), "cell AAT LUT row mismatch");
  }
  checkCpuArcIndices(arcs, ra.numel());
  for (int offset = 0; offset < 20; offset += 5) {
    checkCpuLutRows(arcs.select(1, 3), tables[offset + 1], tables[offset + 2],
                    tables[offset], tables[offset + 3], tables[offset + 4]);
  }
  auto updates = fullyWrittenCpuTensor({arcs.size(0), 8}, ra.options());
  auto jac = fullyWrittenCpuTensor({arcs.size(0), 8, 6}, ra.options());
  AT_DISPATCH_FLOATING_TYPES(ra.scalar_type(), "cellAatForward", [&] {
    cellAatForwardCpuLauncher<scalar_t>(arcs.data_ptr<int64_t>(), arcs.size(1), arcs.size(0), ra.numel(),
        ra.data_ptr<scalar_t>(), fa.data_ptr<scalar_t>(), rs.data_ptr<scalar_t>(), fs.data_ptr<scalar_t>(),
        rc.data_ptr<scalar_t>(), fc.data_ptr<scalar_t>(), table<scalar_t>(tables, 0),
        table<scalar_t>(tables, 5), table<scalar_t>(tables, 10), table<scalar_t>(tables, 15),
        updates.data_ptr<scalar_t>(), jac.data_ptr<scalar_t>(), smooth, scalar_t(alpha));
  });
  return {updates, jac};
}

std::vector<at::Tensor> cell_aat_backward_cpu(
    at::Tensor arcs, at::Tensor jac, at::Tensor gradient, int64_t pins,
    std::vector<at::Tensor> prepared_groups) {
  requireTensor(arcs, jac, at::kLong);
  requireTensor(gradient, jac, jac.scalar_type());
  requireTensor(jac, jac, jac.scalar_type());
  TORCH_CHECK(jac.dim() == 3 && jac.size(0) == arcs.size(0) && jac.size(1) == 8 && jac.size(2) == 6,
              "cell AAT Jacobian shape mismatch");
  TORCH_CHECK(gradient.dim() == 2 && gradient.size(0) == arcs.size(0) && gradient.size(1) == 8,
              "cell AAT gradient shape mismatch");
  checkCpuArcIndices(arcs, pins);
  auto inputs = timingScatterGroups(arcs.select(1, 0), pins, prepared_groups);
  auto outputs = timingScatterGroups(arcs.select(1, 1), pins, prepared_groups, 2);
  std::vector<at::Tensor> result;
  for (int i = 0; i < 6; ++i) {
    result.push_back(fullyWrittenCpuTensor({pins}, jac.options()));
  }
  AT_DISPATCH_FLOATING_TYPES(jac.scalar_type(), "cellAatBackward", [&] {
      cellAatGroupedBackwardCpuLauncher<scalar_t>(inputs.order.data_ptr<int64_t>(), inputs.offsets.data_ptr<int64_t>(),
          outputs.order.data_ptr<int64_t>(), outputs.offsets.data_ptr<int64_t>(),
          jac.data_ptr<scalar_t>(), gradient.data_ptr<scalar_t>(), result[0].data_ptr<scalar_t>(),
          result[1].data_ptr<scalar_t>(), result[2].data_ptr<scalar_t>(), result[3].data_ptr<scalar_t>(),
          result[4].data_ptr<scalar_t>(), result[5].data_ptr<scalar_t>(), pins);

  });
  return result;
}
