#include <cuda_runtime.h>
#include "utility/src/namespace.h"
#include "segment_energy_gradient.h"

DREAMPLACE_BEGIN_NAMESPACE

template <typename T>
__global__ void segmentEnergyGradientKernel(
    const T* positions, const T* sx, const T* sy, const T* ratios,
    const T* weights, const T* psi, const T* psi_v, const bool* directions,
    int count, int nx, int ny,
    T xl, T yl, T bin_x, T bin_y, T* gp, T* gx, T* gy) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  T gradient[4];
  const T* source = directions && !directions[i] ? psi_v : psi;
  segmentEnergyGradient(positions[i], positions[i + count], sx[i], sy[i],
                        ratios[i], weights[i], source, nx, ny, xl, yl, bin_x, bin_y, gradient);
  gp[i] = gradient[0];
  gp[i + count] = gradient[1];
  gx[i] = gradient[2];
  gy[i] = gradient[3];
}

template <typename T>
cudaError_t launchSegmentEnergyGradientCUDA(
    const T* pos, const T* sx, const T* sy, const T* ratio, const T* weight,
    const T* psi, const T* psi_v, const bool* directions,
    int count, int nx, int ny, T xl, T yl, T bin_x, T bin_y,
    T* gp, T* gx, T* gy, cudaStream_t stream) {
  if (count == 0) return cudaSuccess;
  segmentEnergyGradientKernel<<<(count + 127) / 128, 128, 0, stream>>>(
      pos, sx, sy, ratio, weight, psi, psi_v, directions,
      count, nx, ny, xl, yl, bin_x, bin_y, gp, gx, gy);
  return cudaGetLastError();
}

template cudaError_t launchSegmentEnergyGradientCUDA<float>(
    const float*, const float*, const float*, const float*, const float*, const float*, const float*, const bool*,
    int, int, int, float, float, float, float, float*, float*, float*, cudaStream_t);
template cudaError_t launchSegmentEnergyGradientCUDA<double>(
    const double*, const double*, const double*, const double*, const double*, const double*, const double*, const bool*,
    int, int, int, double, double, double, double, double*, double*, double*, cudaStream_t);

DREAMPLACE_END_NAMESPACE
