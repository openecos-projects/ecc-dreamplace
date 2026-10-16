#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <cstdint>

namespace {
template <typename T>
__global__ void initializeCounts(const T* dest, const T* output, T* counts,
                                 int32_t* touched, int destinations) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < destinations) {
    // Torch counts matching self values even when include_self=false. Its
    // backward subsequently zeros self gradients for overwritten groups.
    counts[i] = T(dest[i] == output[i]);
    touched[i] = 0;
  }
}

template <typename T>
__global__ void accumulateCounts(const int64_t* index, const T* src,
                                 const T* output, T* counts, int32_t* touched, int sources) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < sources) {
    const int64_t group = index[i];
    atomicAdd(counts + group, T(src[i] == output[group]));
    atomicExch(touched + group, 1);
  }
}

template <typename T>
__global__ void scatterBackward(const T* dest, const int64_t* index, const T* src,
                               const T* output, const T* counts, const int32_t* touched,
                               const T* gradient, T* grad_dest, T* grad_src,
                               int destinations, int sources, bool include_self) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < destinations) {
    grad_dest[i] = !include_self && touched[i] ? T(0)
        : (gradient[i] / counts[i]) * T(dest[i] == output[i]);
  }
  if (i < sources) {
    const int64_t group = index[i];
    grad_src[i] = (gradient[group] / counts[group]) * T(src[i] == output[group]);
  }
}
}

template <typename T>
void hardScatterForwardLauncher(const T* dest, const int64_t* index, const T* src,
                                const T* output, T* counts, int32_t* touched,
                                int destinations, int sources, cudaStream_t stream) {
  if (destinations) initializeCounts<T><<<(destinations + 127) / 128, 128, 0, stream>>>(
      dest, output, counts, touched, destinations);
  if (sources) accumulateCounts<T><<<(sources + 127) / 128, 128, 0, stream>>>(
      index, src, output, counts, touched, sources);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template <typename T>
void hardScatterBackwardLauncher(const T* dest, const int64_t* index, const T* src,
                                 const T* output, const T* counts, const int32_t* touched,
                                 const T* gradient, T* grad_dest, T* grad_src,
                                 int destinations, int sources, bool include_self, cudaStream_t stream) {
  const int count = max(destinations, sources);
  if (count) scatterBackward<T><<<(count + 127) / 128, 128, 0, stream>>>(
      dest, index, src, output, counts, touched, gradient, grad_dest, grad_src,
      destinations, sources, include_self);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

#define INSTANTIATE(T) \
  template void hardScatterForwardLauncher<T>(const T*, const int64_t*, const T*, const T*, \
                                              T*, int32_t*, int, int, cudaStream_t); \
  template void hardScatterBackwardLauncher<T>(const T*, const int64_t*, const T*, const T*, \
                                               const T*, const int32_t*, const T*, T*, T*, \
                                               int, int, bool, cudaStream_t);
INSTANTIATE(float)
INSTANTIATE(double)
#undef INSTANTIATE
