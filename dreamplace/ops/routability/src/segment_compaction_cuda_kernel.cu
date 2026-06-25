#include <stdio.h>

#include "cuda_runtime.h"
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void compactSegmentForwardKernel(const T* seg1_llx,
                                            const T* seg1_lly,
                                            const T* seg2_llx,
                                            const T* seg2_lly,
                                            const long* seg1_indices,
                                            const long* seg2_indices,
                                            int seg1_count, int seg2_count,
                                            T* segment_llx,
                                            T* segment_lly) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < seg1_count) {
    const long src = seg1_indices[i];
    segment_llx[i] = seg1_llx[src];
    segment_lly[i] = seg1_lly[src];
  }
  if (i < seg2_count) {
    const long src = seg2_indices[i];
    const int dst = seg1_count + i;
    segment_llx[dst] = seg2_llx[src];
    segment_lly[dst] = seg2_lly[src];
  }
}

template <typename T>
__global__ void compactSegmentBackwardKernel(const T* grad_segment_llx,
                                             const T* grad_segment_lly,
                                             const long* seg1_indices,
                                             const long* seg2_indices,
                                             int seg1_count, int seg2_count,
                                             T* grad_seg1_llx,
                                             T* grad_seg1_lly,
                                             T* grad_seg2_llx,
                                             T* grad_seg2_lly) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < seg1_count) {
    const long dst = seg1_indices[i];
    grad_seg1_llx[dst] = grad_segment_llx[i];
    grad_seg1_lly[dst] = grad_segment_lly[i];
  }
  if (i < seg2_count) {
    const long dst = seg2_indices[i];
    const int src = seg1_count + i;
    grad_seg2_llx[dst] = grad_segment_llx[src];
    grad_seg2_lly[dst] = grad_segment_lly[src];
  }
}

template <typename T>
int compactSegmentForwardCudaLauncher(const T* seg1_llx, const T* seg1_lly,
                                      const T* seg2_llx, const T* seg2_lly,
                                      const long* seg1_indices,
                                      const long* seg2_indices,
                                      int seg1_count, int seg2_count,
                                      T* segment_llx, T* segment_lly) {
  const int thread_count = 512;
  const int max_count = max(seg1_count, seg2_count);
  if (max_count > 0) {
    compactSegmentForwardKernel<<<ceilDiv(max_count, thread_count),
                                  thread_count>>>(
        seg1_llx, seg1_lly, seg2_llx, seg2_lly, seg1_indices, seg2_indices,
        seg1_count, seg2_count, segment_llx, segment_lly);
  }
  return 0;
}

template <typename T>
int compactSegmentBackwardCudaLauncher(const T* grad_segment_llx,
                                       const T* grad_segment_lly,
                                       const long* seg1_indices,
                                       const long* seg2_indices,
                                       int seg1_count, int seg2_count,
                                       T* grad_seg1_llx, T* grad_seg1_lly,
                                       T* grad_seg2_llx, T* grad_seg2_lly) {
  const int thread_count = 512;
  const int max_count = max(seg1_count, seg2_count);
  if (max_count > 0) {
    compactSegmentBackwardKernel<<<ceilDiv(max_count, thread_count),
                                   thread_count>>>(
        grad_segment_llx, grad_segment_lly, seg1_indices, seg2_indices,
        seg1_count, seg2_count, grad_seg1_llx, grad_seg1_lly, grad_seg2_llx,
        grad_seg2_lly);
  }
  return 0;
}

#define REGISTER_KERNEL_LAUNCHER(T)                                             \
  template int compactSegmentForwardCudaLauncher<T>(                            \
      const T* seg1_llx, const T* seg1_lly, const T* seg2_llx,                  \
      const T* seg2_lly, const long* seg1_indices,                              \
      const long* seg2_indices, int seg1_count, int seg2_count,                 \
      T* segment_llx, T* segment_lly);                                          \
  template int compactSegmentBackwardCudaLauncher<T>(                           \
      const T* grad_segment_llx, const T* grad_segment_lly,                     \
      const long* seg1_indices, const long* seg2_indices, int seg1_count,       \
      int seg2_count, T* grad_seg1_llx, T* grad_seg1_lly,                       \
      T* grad_seg2_llx, T* grad_seg2_lly);

REGISTER_KERNEL_LAUNCHER(float);
REGISTER_KERNEL_LAUNCHER(double);

DREAMPLACE_END_NAMESPACE
