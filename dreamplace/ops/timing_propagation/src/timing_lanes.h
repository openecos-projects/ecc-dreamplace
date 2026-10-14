#pragma once

#include <cstdint>

template <typename T>
struct TimingLaneInputs {
  const T* values[4];
  int64_t strides[4];
};

template <typename T>
struct TimingLaneOutputs {
  T* values[4];
};
