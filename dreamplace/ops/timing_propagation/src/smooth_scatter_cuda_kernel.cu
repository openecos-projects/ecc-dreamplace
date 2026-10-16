#include <cmath>
#include <cstdint>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>

namespace {
template <typename scalar_t>
__device__ scalar_t contribution(scalar_t value, scalar_t maximum, scalar_t tau) {
  return isfinite(maximum) ? exp((value - maximum) / tau)
                          : scalar_t(value == maximum);
}

template <typename scalar_t>
__global__ void initializeSums(
    const scalar_t* dest, const scalar_t* maxima, scalar_t* sums,
    int count, scalar_t tau, bool include_self) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) {
    // Torch's max backward counts the initial -inf destination in -inf ties,
    // including when include_self=false. Preserve that reference derivative.
    const scalar_t initial_tie = isinf(maxima[i]) && maxima[i] < 0 ? scalar_t(1) : scalar_t(0);
    sums[i] = initial_tie + (include_self ? contribution(dest[i], maxima[i], tau) : scalar_t(0));
  }
}

template <typename scalar_t>
__global__ void accumulateSources(
    const int64_t* index, const scalar_t* src, const scalar_t* maxima,
    scalar_t* sums, int count, scalar_t tau) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) atomicAdd(sums + index[i], contribution(src[i], maxima[index[i]], tau));
}

template <typename scalar_t>
__global__ void groupedSums(
    const scalar_t* dest, const scalar_t* src, const scalar_t* maxima,
    const int64_t* order, const int64_t* offsets, scalar_t* sums,
    int destinations, scalar_t tau, bool include_self) {
  const int group = blockIdx.x * blockDim.x + threadIdx.x;
  if (group >= destinations) return;
  const scalar_t initial_tie = isinf(maxima[group]) && maxima[group] < 0 ? scalar_t(1) : scalar_t(0);
  scalar_t sum = initial_tie;
  for (int64_t row = offsets[group]; row < offsets[group + 1]; ++row) {
    sum += contribution(src[order[row]], maxima[group], tau);
  }
  if (include_self) sum += contribution(dest[group], maxima[group], tau);
  sums[group] = sum;
}

template <typename scalar_t>
__global__ void finishScatter(
    const scalar_t* dest, const int64_t* index, const scalar_t* src,
    const scalar_t* maxima, const scalar_t* sums, scalar_t* output,
    scalar_t* dest_weight, scalar_t* src_weight,
    int destinations, int sources, scalar_t tau, bool include_self) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < destinations) {
    const scalar_t sum = sums[i];
    const scalar_t initial_tie = isinf(maxima[i]) && maxima[i] < 0 ? scalar_t(1) : scalar_t(0);
    if (!include_self && sum == initial_tie && !isnan(maxima[i])) {
      output[i] = dest[i];
      dest_weight[i] = 1;
    } else {
      output[i] = isfinite(maxima[i]) ? maxima[i] + tau * log(max(sum, scalar_t(1e-30))) : maxima[i];
      dest_weight[i] = include_self ? contribution(dest[i], maxima[i], tau) / sum : scalar_t(0);
    }
  }
  if (i < sources) {
    const int64_t group = index[i];
    src_weight[i] = contribution(src[i], maxima[group], tau) / sums[group];
  }
}
}  // namespace

template <typename scalar_t>
void smoothScatterCudaLauncher(
    const scalar_t* dest, const int64_t* index, const scalar_t* src,
    const scalar_t* maxima, scalar_t* sums, scalar_t* output,
    scalar_t* dest_weight, scalar_t* src_weight,
    int destinations, int sources, scalar_t tau, bool include_self, cudaStream_t stream) {
  if (destinations == 0) return;
  initializeSums<scalar_t><<<(destinations + 127) / 128, 128, 0, stream>>>(
      dest, maxima, sums, destinations, tau, include_self);
  if (sources != 0) {
    accumulateSources<scalar_t><<<(sources + 127) / 128, 128, 0, stream>>>(
        index, src, maxima, sums, sources, tau);
  }
  const int count = max(destinations, sources);
  finishScatter<scalar_t><<<(count + 127) / 128, 128, 0, stream>>>(
      dest, index, src, maxima, sums, output, dest_weight, src_weight,
      destinations, sources, tau, include_self);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename scalar_t>
void smoothScatterGroupedCudaLauncher(
    const scalar_t* dest, const int64_t* index, const scalar_t* src,
    const scalar_t* maxima, const int64_t* order, const int64_t* offsets,
    scalar_t* sums, scalar_t* output, scalar_t* dest_weight, scalar_t* src_weight,
    int destinations, int sources, scalar_t tau, bool include_self, cudaStream_t stream) {
  if (destinations == 0) return;
  groupedSums<scalar_t><<<(destinations + 127) / 128, 128, 0, stream>>>(
      dest, src, maxima, order, offsets, sums, destinations, tau, include_self);
  const int count = max(destinations, sources);
  finishScatter<scalar_t><<<(count + 127) / 128, 128, 0, stream>>>(
      dest, index, src, maxima, sums, output, dest_weight, src_weight,
      destinations, sources, tau, include_self);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define INSTANTIATE(scalar_t) \
  template void smoothScatterCudaLauncher<scalar_t>( \
      const scalar_t*, const int64_t*, const scalar_t*, const scalar_t*, \
      scalar_t*, scalar_t*, scalar_t*, scalar_t*, int, int, scalar_t, bool, cudaStream_t); \
  template void smoothScatterGroupedCudaLauncher<scalar_t>(const scalar_t*, const int64_t*, \
      const scalar_t*, const scalar_t*, const int64_t*, const int64_t*, scalar_t*, scalar_t*, \
      scalar_t*, scalar_t*, int, int, scalar_t, bool, cudaStream_t)
INSTANTIATE(float);
INSTANTIATE(double);
