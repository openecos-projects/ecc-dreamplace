#ifndef GPUPLACE_RUDY_SMOOTH_FUNCTIONAL_H
#define GPUPLACE_RUDY_SMOOTH_FUNCTIONAL_H

#include <cstdint>
#include "utility/src/utils.cuh"

DREAMPLACE_BEGIN_NAMESPACE

// Every block uses the same binary tree; no floating-point atomic additions.
template <typename T, int Threads>
__device__ T pin2pinBlockSum(T value) {
    __shared__ T values[Threads];
    values[threadIdx.x] = value;
    __syncthreads();
    for (int stride = Threads / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            values[threadIdx.x] += values[threadIdx.x + stride];
        }
        __syncthreads();
    }
    return values[0];
}

template <typename T, int Threads>
__global__ void pin2pinAttractionCudaForward(
    const T *pin_pos_x, const T *pin_pos_y, const int *pairs,
    const T *weights, int num_pairs, T *partials) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    T distance = 0;
    if (idx < num_pairs) {
        int p1 = pairs[2 * idx], p2 = pairs[2 * idx + 1];
        T dx = pin_pos_x[p1] - pin_pos_x[p2];
        T dy = pin_pos_y[p1] - pin_pos_y[p2];
        distance = weights[idx] * (dx * dx + dy * dy);
    }
    T sum = pin2pinBlockSum<T, Threads>(distance);
    if (threadIdx.x == 0) partials[blockIdx.x] = sum;
}

template <typename T, int Threads>
__global__ void pin2pinAttractionCudaReduce(
    const T *partials, int count, T *total_distance) {
    T sum = 0;
    for (int i = threadIdx.x; i < count; i += Threads) sum += partials[i];
    sum = pin2pinBlockSum<T, Threads>(sum);
    if (threadIdx.x == 0) *total_distance = sum;
}

template <typename T>
__global__ void pin2pinAttractionCudaBackward(
    const T *pin_pos_x, const T *pin_pos_y, const int *pairs,
    const T *weights, int num_ends, const int *sorted_pins,
    const int64_t *sorted_ends, const T *grad_tensor,
    T *grad_x_tensor, T *grad_y_tensor) {
    // One warp owns a segment head. Stable sorting preserves endpoint order
    // within each pin; lane-strided sums and the warp tree have fixed order.
    int lane = threadIdx.x % 32;
    int start = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    if (start >= num_ends) return;
    int pin = sorted_pins[start];
    if (start > 0 && sorted_pins[start - 1] == pin) return;
    T gx = 0, gy = 0;
    for (int i = start + lane; i < num_ends && sorted_pins[i] == pin; i += 32) {
        int64_t end = sorted_ends[i];
        int pair = end / 2;
        int p1 = pairs[2 * pair], p2 = pairs[2 * pair + 1];
        T scale = 2 * (*grad_tensor) * weights[pair];
        T dx = scale * (pin_pos_x[p1] - pin_pos_x[p2]);
        T dy = scale * (pin_pos_y[p1] - pin_pos_y[p2]);
        gx += (end % 2) ? -dx : dx;
        gy += (end % 2) ? -dy : dy;
    }
    for (int offset = 16; offset > 0; offset /= 2) {
        gx += __shfl_down_sync(0xffffffff, gx, offset);
        gy += __shfl_down_sync(0xffffffff, gy, offset);
    }
    if (lane == 0) {
        grad_x_tensor[pin] = gx;
        grad_y_tensor[pin] = gy;
    }
}

DREAMPLACE_END_NAMESPACE

#endif
