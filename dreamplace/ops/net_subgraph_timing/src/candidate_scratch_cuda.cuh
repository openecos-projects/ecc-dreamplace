#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace candidate_scratch {
// Each pointer belongs to a separately allocated tensor. Combining their
// initialization does not introduce shared storage or version counters.
template <typename T>
struct Forward {
  int64_t* candidate_index;
  T* nodes[7];
  T* sinks[6];
};

template <typename T>
__global__ void initialize_forward(Forward<T> scratch, int nodes, int sinks) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nodes) {
    scratch.candidate_index[i] = -1;
    #pragma unroll
    for (int field = 0; field < 7; ++field) scratch.nodes[field][i] = T(0);
  }
  if (i < sinks) {
    #pragma unroll
    for (int field = 0; field < 6; ++field) scratch.sinks[field][i] = T(0);
  }
}

template <typename T>
struct Gradient {
  T* drivers[2];
  T* candidates[4];
};

template <typename T>
__global__ void initialize_gradient(Gradient<T> scratch, int nets, int candidates) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < nets) {
    #pragma unroll
    for (int field = 0; field < 2; ++field) scratch.drivers[field][i] = T(0);
  }
  if (i < candidates) {
    #pragma unroll
    for (int field = 0; field < 4; ++field) scratch.candidates[field][i] = T(0);
  }
}
}  // namespace candidate_scratch
