#pragma once

#include "timing_math.h"
#include "timing_lanes.h"

template <typename T>
TIMING_INLINE T laneContribution(T value, T maximum, T tau) {
  return isfinite(maximum) ? exp((value - maximum) / tau) : T(value == maximum);
}

template <typename T>
TIMING_INLINE void forwardRow(int pin, TimingLaneInputs<T> dest, TimingLaneInputs<T> src,
    const int64_t* order, const int64_t* offsets, TimingLaneOutputs<T> output,
    T* denominator, T* dest_factor, T* src_factor, int pins, bool smooth, T tau) {
  for (int lane = 0; lane < 4; ++lane) {
    const T initial = dest.values[lane][pin * dest.strides[lane]];
    T maximum = initial;
    for (int64_t row = offsets[pin]; row < offsets[pin + 1]; ++row) {
      const T value = src.values[lane][order[row] * src.strides[lane]];
      maximum = isnan(maximum) ? maximum : (isnan(value) ? value : max(maximum, value));
    }
    const bool lse = smooth && lane < 2;
    T sum = lse && isinf(maximum) && maximum < 0 ? T(1) : T(0);
    for (int64_t row = offsets[pin]; row < offsets[pin + 1]; ++row) {
      const T value = src.values[lane][order[row] * src.strides[lane]];
      sum += lse ? laneContribution(value, maximum, tau) : T(value == maximum);
    }
    sum += lse ? laneContribution(initial, maximum, tau) : T(initial == maximum);
    denominator[pin * 4 + lane] = sum;
    dest_factor[pin * 4 + lane] = lse ? laneContribution(initial, maximum, tau) / sum : T(initial == maximum);
    output.values[lane][pin] = lse && isfinite(maximum)
        ? maximum + tau * log(max(sum, T(1e-30))) : maximum;
    for (int64_t row = offsets[pin]; row < offsets[pin + 1]; ++row) {
      const int64_t arc = order[row];
      const T value = src.values[lane][arc * src.strides[lane]];
      src_factor[arc * 4 + lane] = lse ? laneContribution(value, maximum, tau) / sum : T(value == maximum);
    }
  }
}
template <typename T>
TIMING_INLINE void backwardRow(int row, const int64_t* index, const T* denominator,
    const T* dest_factor, const T* src_factor, TimingLaneInputs<T> gradient,
    TimingLaneOutputs<T> grad_dest, TimingLaneOutputs<T> grad_src, int pins, int arcs, bool smooth) {
  for (int lane = 0; lane < 4; ++lane) {
    const bool lse = smooth && lane < 2;
    if (row < pins) {
      T value = 0;
      if (gradient.values[lane]) {
        value = gradient.values[lane][row * gradient.strides[lane]];
        if (!lse) value /= denominator[row * 4 + lane];
        value *= dest_factor[row * 4 + lane];
      }
      grad_dest.values[lane][row] = value;
    }
    if (row < arcs) {
      const int64_t pin = index[row];
      T value = 0;
      if (gradient.values[lane]) {
        value = gradient.values[lane][pin * gradient.strides[lane]];
        if (!lse) value /= denominator[pin * 4 + lane];
        value *= src_factor[row * 4 + lane];
      }
      grad_src.values[lane][row] = value;
    }
  }
}
