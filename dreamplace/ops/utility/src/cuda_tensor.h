#pragma once

#include "utility/src/torch.h"
#include <ATen/cuda/EmptyTensor.h>

// Producer-owned storage: the caller must write every element before it is
// read or returned. Avoid a diagnostic fill followed by that same full write,
// without changing Torch's process-wide deterministic memory-fill setting.
inline at::Tensor fullyWrittenCudaTensor(at::IntArrayRef shape, const at::TensorOptions& options) {
  return at::Tensor(at::detail::empty_cuda(shape, options));
}
