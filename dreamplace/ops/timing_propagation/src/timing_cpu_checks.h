#pragma once

#include "utility/src/torch.h"

// Validate integer domains before entering OpenMP: exceptions may not escape a
// parallel region. Metadata is small and immutable during each native call.
inline void checkCpuArcIndices(const at::Tensor& arcs, int64_t pins) {
  TORCH_CHECK(arcs.dim() == 2 && arcs.size(1) >= 2 && pins >= 0, "timing arc shape mismatch");
  const auto* ids = arcs.data_ptr<int64_t>();
  for (int64_t row = 0; row < arcs.size(0); ++row) {
    for (int col = 0; col < 2; ++col) {
      const auto pin = ids[row * arcs.size(1) + col];
      TORCH_CHECK(pin >= 0 && pin < pins, "timing arc pin out of range");
    }
  }
}

inline void checkCpuLutRows(const at::Tensor& index, const at::Tensor& trans,
    const at::Tensor& caps, const at::Tensor& values, const at::Tensor& nt, const at::Tensor& nc) {
  const auto rows = values.size(0);
  const auto* ids = index.data_ptr<int64_t>();
  for (int64_t i = 0; i < index.numel(); ++i) {
    TORCH_CHECK(ids[i * index.stride(0)] >= -rows && ids[i * index.stride(0)] < rows,
                "timing LUT row out of range");
  }
  const auto* td = nt.data_ptr<int64_t>(); const auto* cd = nc.data_ptr<int64_t>();
  for (int64_t row = 0; row < rows; ++row) {
    TORCH_CHECK(td[row] >= 0 && td[row] <= trans.size(1) && cd[row] >= 0 && cd[row] <= caps.size(1),
                "timing LUT axis dimension out of range");
    const auto count = std::max<int64_t>(td[row], 1) * std::max<int64_t>(cd[row], 1);
    TORCH_CHECK((td[row] == 0 && cd[row] == 0) || count <= values.size(1),
                "timing LUT value dimension out of range");
  }
}
