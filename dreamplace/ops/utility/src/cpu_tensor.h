#pragma once

#include "utility/src/torch.h"
#include <ATen/EmptyTensor.h>

// Producer-owned storage: write every element before it is read or returned.
// As on CUDA, avoid a redundant deterministic fill without changing global state.
inline at::Tensor fullyWrittenCpuTensor(at::IntArrayRef shape, const at::TensorOptions& options) {
  return at::Tensor(at::detail::empty_cpu(shape, options));
}
