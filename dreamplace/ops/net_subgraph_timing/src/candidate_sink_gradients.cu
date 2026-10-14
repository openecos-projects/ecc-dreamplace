#include <cstdint>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {
template <typename scalar_t>
__global__ void seedSinks(
    const int64_t* topology, const int64_t* starts, const int64_t* sinks,
    const int64_t* order, const int64_t* offsets,
    const scalar_t* slew_out, const scalar_t* grad_arrival,
    const scalar_t* grad_slew, const scalar_t* grad_load,
    const scalar_t* grad_delay, const scalar_t* grad_impulse,
    scalar_t* lin, scalar_t* arrival_out, scalar_t* slew, int nets) {
  const int net = blockIdx.x * blockDim.x + threadIdx.x;
  if (net >= nets || offsets[net] == offsets[net + 1]) return;
  const int64_t root = topology[starts[net]];
  for (int64_t row = offsets[net]; row < offsets[net + 1]; ++row) {
    const int64_t source = order[row], node = sinks[source];
    arrival_out[node] += grad_arrival[source];
    arrival_out[node] += grad_delay[source];
    arrival_out[root] -= grad_delay[source];
    slew[node] += grad_slew[source];
    slew[node] += scalar_t(2) * slew_out[node] * grad_impulse[source];
    slew[root] -= scalar_t(2) * slew_out[root] * grad_impulse[source];
    lin[node] += grad_load[source];
  }
}
}  // namespace

template <typename scalar_t>
void candidateSinkGradientsLauncher(
    const int64_t* topology, const int64_t* starts, const int64_t* sinks,
    const int64_t* order, const int64_t* offsets,
    const scalar_t* slew_out, const scalar_t* grad_arrival,
    const scalar_t* grad_slew, const scalar_t* grad_load,
    const scalar_t* grad_delay, const scalar_t* grad_impulse,
    scalar_t* lin, scalar_t* arrival_out, scalar_t* slew, int nets, cudaStream_t stream) {
  if (nets == 0) return;
  seedSinks<scalar_t><<<(nets + 127) / 128, 128, 0, stream>>>(topology, starts, sinks, order, offsets,
      slew_out, grad_arrival, grad_slew, grad_load, grad_delay, grad_impulse,
      lin, arrival_out, slew, nets);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define INSTANTIATE(scalar_t) \
  template void candidateSinkGradientsLauncher<scalar_t>( \
      const int64_t*, const int64_t*, const int64_t*, const int64_t*, const int64_t*, \
      const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*, \
      const scalar_t*, scalar_t*, scalar_t*, scalar_t*, int, cudaStream_t)
INSTANTIATE(float);
INSTANTIATE(double);
