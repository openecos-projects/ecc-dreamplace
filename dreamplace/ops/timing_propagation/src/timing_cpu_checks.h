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

inline void checkCpuLutRowIndices(const at::Tensor& index, int64_t rows) {
  const auto* ids = index.data_ptr<int64_t>();
  const auto stride = index.stride(0);
  const auto count = index.numel();
  for (int64_t i = 0; i < count; ++i) {
    const auto id = ids[i * stride];
    TORCH_CHECK(id >= -rows && id < rows, "timing LUT row out of range");
  }
}

inline void checkCpuLutTableDims(const at::Tensor& trans,
    const at::Tensor& caps, const at::Tensor& values, const at::Tensor& nt, const at::Tensor& nc) {
  struct ValidatedLutKey {
    const int64_t* nt_ptr = nullptr;
    const int64_t* nc_ptr = nullptr;
    uint32_t nt_version = 0;
    uint32_t nc_version = 0;
    int64_t rows = -1;
    int64_t trans_cols = -1;
    int64_t caps_cols = -1;
    int64_t val_cols = -1;
  };
  thread_local ValidatedLutKey cache[8];
  thread_local int cache_slot = 0;

  const auto rows = values.size(0);
  const auto* td = nt.data_ptr<int64_t>();
  const auto* cd = nc.data_ptr<int64_t>();
  const uint32_t nt_ver = nt._version();
  const uint32_t nc_ver = nc._version();
  const auto trans_cols = trans.size(1);
  const auto caps_cols = caps.size(1);
  const auto val_cols = values.size(1);

  for (const auto& entry : cache) {
    if (entry.nt_ptr == td && entry.nc_ptr == cd &&
        entry.nt_version == nt_ver && entry.nc_version == nc_ver &&
        entry.rows == rows && entry.trans_cols == trans_cols &&
        entry.caps_cols == caps_cols && entry.val_cols == val_cols) {
      return;
    }
  }

  for (int64_t row = 0; row < rows; ++row) {
    TORCH_CHECK(td[row] >= 0 && td[row] <= trans_cols && cd[row] >= 0 && cd[row] <= caps_cols,
                "timing LUT axis dimension out of range");
    const auto count = std::max<int64_t>(td[row], 1) * std::max<int64_t>(cd[row], 1);
    TORCH_CHECK((td[row] == 0 && cd[row] == 0) || count <= val_cols,
                "timing LUT value dimension out of range");
  }
  cache[cache_slot] = {td, cd, nt_ver, nc_ver, rows, trans_cols, caps_cols, val_cols};
  cache_slot = (cache_slot + 1) & 7;
}

inline void checkCpuLutRows(const at::Tensor& index, const at::Tensor& trans,
    const at::Tensor& caps, const at::Tensor& values, const at::Tensor& nt, const at::Tensor& nc) {
  checkCpuLutRowIndices(index, values.size(0));
  checkCpuLutTableDims(trans, caps, values, nt, nc);
}

inline void checkCpuTargetLutRows(
    const at::Tensor& target_arc_ids, const at::Tensor& valid_arcs, int64_t row_count) {
  const auto* targets = target_arc_ids.data_ptr<int64_t>();
  const auto* valid = valid_arcs.data_ptr<bool>();
  const auto total = target_arc_ids.numel();
  for (int64_t i = 0; i < total; ++i) {
    if (valid[i]) {
      TORCH_CHECK(targets[i] >= 0 && targets[i] < row_count,
                  "cell action target row is out of range");
    }
  }
}
