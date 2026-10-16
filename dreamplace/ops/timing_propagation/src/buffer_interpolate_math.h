#pragma once

#include "timing_math.h"

template <typename scalar_t>
TIMING_INLINE int interpolateRow(int row, const scalar_t* index, const scalar_t* cap, const scalar_t* delay,
    const scalar_t* slew, scalar_t* values, scalar_t* clipped, int64_t* lo, int64_t* hi,
    scalar_t* alpha, scalar_t* metadata, int rows, int sizes) {
  const scalar_t raw = index[row];
  const bool nan = isnan(raw);
  const scalar_t bounded = nan ? scalar_t(0) : max(scalar_t(0), min(scalar_t(sizes - 1), raw));
  const int lower = floor(bounded), upper = ceil(bounded);
  const scalar_t weight = bounded - scalar_t(lower);
  clipped[row] = nan ? raw : bounded;
  lo[row] = lower; hi[row] = upper; alpha[row] = nan ? raw : weight;
  metadata[row + 1] = raw;
  const scalar_t* tables[3] = {cap, delay, slew};
  int invalid = nan ? 8 : 0;
  for (int field = 0; field < 3; ++field) {
    for (int col = 0; col < sizes; ++col) {
      if (!isfinite(tables[field][row * sizes + col])) invalid |= 1 << field;
    }
    values[row * 3 + field] = (scalar_t(1) - weight) * tables[field][row * sizes + lower]
        + weight * tables[field][row * sizes + upper];
  }
  return invalid;
}
template <typename scalar_t>
TIMING_INLINE void interpolateBackwardRow(int row, const scalar_t* index, const scalar_t* cap, const scalar_t* delay,
    const scalar_t* slew, const int64_t* lo, const int64_t* hi, const scalar_t* alpha,
    const scalar_t* grad_values, const scalar_t* grad_clipped, const scalar_t* grad_alpha,
    scalar_t* grad_index, scalar_t* grad_cap, scalar_t* grad_delay, scalar_t* grad_slew,
    int rows, int sizes) {
  const scalar_t* tables[3] = {cap, delay, slew};
  scalar_t* gradients[3] = {grad_cap, grad_delay, grad_slew};
  scalar_t grad_weight = grad_alpha[row];
  for (int field = 2; field >= 0; --field) {
    const scalar_t grad = grad_values[row * 3 + field];
    // Keep the two gather contributions separate even when lo==hi: 0*inf
    // must retain Torch's non-finite derivative, not become inf via merging.
    const scalar_t low_grad = (scalar_t(1) - alpha[row]) * grad;
    const scalar_t high_grad = alpha[row] * grad;
    for (int col = 0; col < sizes; ++col) {
      scalar_t value = 0;
      if (col == lo[row]) value += low_grad;
      if (col == hi[row]) value += high_grad;
      gradients[field][row * sizes + col] = value;
    }
    grad_weight += tables[field][row * sizes + hi[row]] * grad
        - tables[field][row * sizes + lo[row]] * grad;
  }
  grad_index[row] = index[row] >= 0 && index[row] <= sizes - 1
      ? grad_weight + grad_clipped[row] : scalar_t(0);
}
