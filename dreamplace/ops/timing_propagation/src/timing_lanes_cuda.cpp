#include "utility/src/torch.h"
#include "utility/src/cuda_tensor.h"
#include "timing_lanes_cuda.h"
#include "timing_scatter_groups.h"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

namespace {
void requireVector(const at::Tensor& value, const at::Tensor& reference, int64_t count) {
  TORCH_CHECK(value.is_cuda() && value.device() == reference.device() &&
              value.scalar_type() == reference.scalar_type() &&
              value.dim() == 1 && value.numel() == count, "timing lane vector mismatch");
}

template <typename T>
TimingLaneInputs<T> inputs(const std::vector<at::Tensor>& tensors) {
  TimingLaneInputs<T> result;
  for (int lane = 0; lane < 4; ++lane) {
    result.values[lane] = tensors[lane].numel() ? tensors[lane].data_ptr<T>() : nullptr;
    result.strides[lane] = tensors[lane].stride(0);
  }
  return result;
}

template <typename T>
TimingLaneOutputs<T> outputs(const std::vector<at::Tensor>& tensors, int offset = 0) {
  TimingLaneOutputs<T> result;
  for (int lane = 0; lane < 4; ++lane) result.values[lane] = tensors[offset + lane].data_ptr<T>();
  return result;
}

void requireIndex(const at::Tensor& index, const at::Tensor& reference) {
  TORCH_CHECK(index.is_cuda() && index.device() == reference.device() &&
              index.scalar_type() == at::kLong && index.dim() == 1 && index.is_contiguous(),
              "timing lane index mismatch");
}
}  // namespace

std::vector<at::Tensor> timing_lanes_forward_cuda(at::Tensor index,
    std::vector<at::Tensor> dest, std::vector<at::Tensor> src, bool smooth, double tau,
    std::vector<at::Tensor> prepared_groups) {
  TORCH_CHECK(dest.size() == 4 && src.size() == 4 && (!smooth || tau > 0),
              "timing lanes require four vectors and positive smooth temperature");
  const auto pins = dest[0].numel();
  requireIndex(index, dest[0]);
  for (const auto& value : dest) requireVector(value, dest[0], pins);
  for (const auto& value : src) requireVector(value, dest[0], index.numel());
  const c10::cuda::CUDAGuard guard(dest[0].device());
  auto groups = timingScatterGroups(index, pins, prepared_groups);
  std::vector<at::Tensor> result;
  for (int lane = 0; lane < 4; ++lane) result.push_back(fullyWrittenCudaTensor({pins}, dest[0].options()));
  result.push_back(fullyWrittenCudaTensor({pins, 4}, dest[0].options()));
  result.push_back(fullyWrittenCudaTensor({pins, 4}, dest[0].options()));
  result.push_back(fullyWrittenCudaTensor({index.numel(), 4}, dest[0].options()));
  const auto stream = c10::cuda::getCurrentCUDAStream(dest[0].get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES(dest[0].scalar_type(), "timingLanesForward", [&] {
    timingLanesForwardLauncher<scalar_t>(inputs<scalar_t>(dest), inputs<scalar_t>(src),
        groups.order.data_ptr<int64_t>(), groups.offsets.data_ptr<int64_t>(), outputs<scalar_t>(result),
        result[4].data_ptr<scalar_t>(), result[5].data_ptr<scalar_t>(), result[6].data_ptr<scalar_t>(),
        pins, smooth, scalar_t(tau), stream);
  });
  return result;
}

std::vector<at::Tensor> timing_lanes_backward_cuda(at::Tensor index, at::Tensor denominator,
    at::Tensor dest_factor, at::Tensor src_factor, std::vector<at::Tensor> gradient, bool smooth) {
  requireIndex(index, denominator);
  TORCH_CHECK(gradient.size() == 4 && denominator.dim() == 2 && denominator.size(1) == 4,
              "timing lane adjoint shape mismatch");
  const auto pins = denominator.size(0);
  for (const auto& value : {denominator, dest_factor, src_factor}) {
    TORCH_CHECK(value.is_cuda() && value.device() == denominator.device() &&
                value.scalar_type() == denominator.scalar_type() && value.is_contiguous(),
                "timing lane adjoint tensor mismatch");
  }
  TORCH_CHECK(dest_factor.sizes() == denominator.sizes() && src_factor.dim() == 2 &&
              src_factor.size(0) == index.numel() && src_factor.size(1) == 4,
              "timing lane adjoint factor mismatch");
  for (const auto& value : gradient) requireVector(value, denominator, value.numel() ? pins : 0);
  const c10::cuda::CUDAGuard guard(denominator.device());
  std::vector<at::Tensor> result;
  for (int lane = 0; lane < 8; ++lane) {
    result.push_back(fullyWrittenCudaTensor({lane < 4 ? pins : index.numel()}, denominator.options()));
  }
  const auto stream = c10::cuda::getCurrentCUDAStream(denominator.get_device()).stream();
  AT_DISPATCH_FLOATING_TYPES(denominator.scalar_type(), "timingLanesBackward", [&] {
    timingLanesBackwardLauncher<scalar_t>(index.data_ptr<int64_t>(), denominator.data_ptr<scalar_t>(),
        dest_factor.data_ptr<scalar_t>(), src_factor.data_ptr<scalar_t>(), inputs<scalar_t>(gradient),
        outputs<scalar_t>(result), outputs<scalar_t>(result, 4), pins, index.numel(), smooth, stream);
  });
  return result;
}
