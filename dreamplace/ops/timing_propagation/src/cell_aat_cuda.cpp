#include "utility/src/torch.h"
#include "utility/src/cuda_tensor.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include "lut_query_cuda.cuh"
#include "timing_scatter_groups.h"

template <typename scalar_t>
void cellAatForwardLauncher(const int64_t*, int, int, int,
    const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*,
    TimingLutTable<scalar_t>, TimingLutTable<scalar_t>, TimingLutTable<scalar_t>, TimingLutTable<scalar_t>,
    scalar_t*, scalar_t*, bool, scalar_t, cudaStream_t);
template <typename scalar_t>
void cellAatBackwardLauncher(const int64_t*, int, const scalar_t*, const scalar_t*,
    scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, int, cudaStream_t);
template <typename scalar_t>
void cellAatGroupedBackwardLauncher(const int64_t*, const int64_t*, const int64_t*, const int64_t*,
    const scalar_t*, const scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*,
    int, cudaStream_t);

namespace {
void requireTensor(at::Tensor value, at::Tensor reference, at::ScalarType dtype) {
  TORCH_CHECK(value.is_cuda() && value.device() == reference.device() && value.is_contiguous(),
              "cell AAT expects contiguous tensors on one CUDA device");
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

std::vector<at::Tensor> cell_aat_forward_cuda(
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
  const c10::cuda::CUDAGuard guard(ra.device());
  auto updates = fullyWrittenCudaTensor({arcs.size(0), 8}, ra.options());
  auto jac = fullyWrittenCudaTensor({arcs.size(0), 8, 6}, ra.options());
  const auto stream = c10::cuda::getCurrentCUDAStream(ra.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES(ra.scalar_type(), "cellAatForward", [&] {
    cellAatForwardLauncher<scalar_t>(arcs.data_ptr<int64_t>(), arcs.size(1), arcs.size(0), ra.numel(),
        ra.data_ptr<scalar_t>(), fa.data_ptr<scalar_t>(), rs.data_ptr<scalar_t>(), fs.data_ptr<scalar_t>(),
        rc.data_ptr<scalar_t>(), fc.data_ptr<scalar_t>(), table<scalar_t>(tables, 0),
        table<scalar_t>(tables, 5), table<scalar_t>(tables, 10), table<scalar_t>(tables, 15),
        updates.data_ptr<scalar_t>(), jac.data_ptr<scalar_t>(), smooth, scalar_t(alpha), stream);
  });
  return {updates, jac};
}

std::vector<at::Tensor> cell_aat_backward_cuda(
    at::Tensor arcs, at::Tensor jac, at::Tensor gradient, int64_t pins,
    std::vector<at::Tensor> prepared_groups) {
  requireTensor(arcs, jac, at::kLong);
  requireTensor(gradient, jac, jac.scalar_type());
  requireTensor(jac, jac, jac.scalar_type());
  TORCH_CHECK(jac.dim() == 3 && jac.size(0) == arcs.size(0) && jac.size(1) == 8 && jac.size(2) == 6,
              "cell AAT Jacobian shape mismatch");
  TORCH_CHECK(gradient.dim() == 2 && gradient.size(0) == arcs.size(0) && gradient.size(1) == 8,
              "cell AAT gradient shape mismatch");
  const c10::cuda::CUDAGuard guard(jac.device());
  const bool deterministic = at::globalContext().deterministicAlgorithms();
  TimingScatterGroups inputs, outputs;
  if (deterministic) {
    inputs = timingScatterGroups(arcs.select(1, 0), pins, prepared_groups);
    outputs = timingScatterGroups(arcs.select(1, 1), pins, prepared_groups, 2);
  }
  std::vector<at::Tensor> result;
  for (int i = 0; i < 6; ++i) {
    result.push_back(deterministic ? fullyWrittenCudaTensor({pins}, jac.options()) : at::zeros({pins}, jac.options()));
  }
  const auto stream = c10::cuda::getCurrentCUDAStream(jac.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES(jac.scalar_type(), "cellAatBackward", [&] {
    if (deterministic) {
      cellAatGroupedBackwardLauncher<scalar_t>(inputs.order.data_ptr<int64_t>(), inputs.offsets.data_ptr<int64_t>(),
          outputs.order.data_ptr<int64_t>(), outputs.offsets.data_ptr<int64_t>(),
          jac.data_ptr<scalar_t>(), gradient.data_ptr<scalar_t>(), result[0].data_ptr<scalar_t>(),
          result[1].data_ptr<scalar_t>(), result[2].data_ptr<scalar_t>(), result[3].data_ptr<scalar_t>(),
          result[4].data_ptr<scalar_t>(), result[5].data_ptr<scalar_t>(), pins, stream);
    } else {
      cellAatBackwardLauncher<scalar_t>(arcs.data_ptr<int64_t>(), arcs.size(1), jac.data_ptr<scalar_t>(),
          gradient.data_ptr<scalar_t>(), result[0].data_ptr<scalar_t>(), result[1].data_ptr<scalar_t>(),
          result[2].data_ptr<scalar_t>(), result[3].data_ptr<scalar_t>(), result[4].data_ptr<scalar_t>(),
          result[5].data_ptr<scalar_t>(), arcs.size(0), stream);
    }
  });
  return result;
}
