#pragma once

#include "utility/src/torch.h"

struct TimingScatterGroups {
  at::Tensor order;
  at::Tensor offsets;
};

inline TimingScatterGroups timingScatterGroups(
    at::Tensor index, int64_t destinations, const std::vector<at::Tensor>& prepared = {}, int offset = 0) {
  if (!prepared.empty()) {
    TORCH_CHECK(offset + 2 <= prepared.size(), "timing scatter group pair missing");
    auto order = prepared[offset], offsets = prepared[offset + 1];
    for (const auto& tensor : {order, offsets}) {
      TORCH_CHECK(tensor.device() == index.device() && tensor.scalar_type() == at::kLong &&
                  tensor.dim() == 1 && tensor.is_contiguous(), "timing scatter group tensor mismatch");
    }
    TORCH_CHECK(order.numel() == index.numel() && offsets.numel() == destinations + 1,
                "timing scatter group shape mismatch");
    return {order, offsets};
  }
  auto order = at::argsort(index, /*stable=*/true, 0, false);
  auto counts = at::bincount(index, {}, destinations);
  auto offsets = at::cat({at::zeros({1}, index.options()), counts.cumsum(0)});
  return {order, offsets};
}
