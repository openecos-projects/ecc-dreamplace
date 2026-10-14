#include "timing_lanes_cuda.h"
#include <cmath>
#include <c10/cuda/CUDAException.h>

#include "timing_lanes_math.h"

namespace {
template <typename T>
__global__ void forward(TimingLaneInputs<T> dest, TimingLaneInputs<T> src,
    const int64_t* order, const int64_t* offsets, TimingLaneOutputs<T> output,
    T* denominator, T* dest_factor, T* src_factor, int pins, bool smooth, T tau) {
  const int pin = blockIdx.x * blockDim.x + threadIdx.x;
  if (pin < pins) forwardRow<T>(pin, dest, src, order, offsets, output, denominator, dest_factor, src_factor, pins, smooth, tau);
}

template <typename T>
__global__ void backward(const int64_t* index, const T* denominator,
    const T* dest_factor, const T* src_factor, TimingLaneInputs<T> gradient,
    TimingLaneOutputs<T> grad_dest, TimingLaneOutputs<T> grad_src, int pins, int arcs, bool smooth) {
  const int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row < max(pins, arcs)) backwardRow<T>(row, index, denominator, dest_factor, src_factor, gradient, grad_dest, grad_src, pins, arcs, smooth);
}
}  // namespace

template <typename T>
void timingLanesForwardLauncher(TimingLaneInputs<T> dest, TimingLaneInputs<T> src,
    const int64_t* order, const int64_t* offsets, TimingLaneOutputs<T> output,
    T* denominator, T* dest_factor, T* src_factor, int pins, bool smooth, T tau, cudaStream_t stream) {
  if (!pins) return;
  forward<T><<<(pins + 127) / 128, 128, 0, stream>>>(dest, src, order, offsets, output,
      denominator, dest_factor, src_factor, pins, smooth, tau);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void timingLanesBackwardLauncher(const int64_t* index, const T* denominator,
    const T* dest_factor, const T* src_factor, TimingLaneInputs<T> gradient,
    TimingLaneOutputs<T> grad_dest, TimingLaneOutputs<T> grad_src, int pins, int arcs,
    bool smooth, cudaStream_t stream) {
  const int count = max(pins, arcs);
  if (!count) return;
  backward<T><<<(count + 127) / 128, 128, 0, stream>>>(index, denominator, dest_factor, src_factor,
      gradient, grad_dest, grad_src, pins, arcs, smooth);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define INSTANTIATE(T) \
  template void timingLanesForwardLauncher<T>(TimingLaneInputs<T>, TimingLaneInputs<T>, \
      const int64_t*, const int64_t*, TimingLaneOutputs<T>, T*, T*, T*, int, bool, T, cudaStream_t); \
  template void timingLanesBackwardLauncher<T>(const int64_t*, const T*, const T*, const T*, \
      TimingLaneInputs<T>, TimingLaneOutputs<T>, TimingLaneOutputs<T>, int, int, bool, cudaStream_t)
INSTANTIATE(float);
INSTANTIATE(double);
