#ifndef DREAMPLACE_SEGMENT_ENERGY_GRADIENT_H
#define DREAMPLACE_SEGMENT_ENERGY_GRADIENT_H

#include <cmath>

#ifdef __CUDACC__
#define ROUTE_HD __host__ __device__
#else
#define ROUTE_HD
#endif

DREAMPLACE_BEGIN_NAMESPACE

template <typename T> ROUTE_HD inline T routeMin(T a, T b) { return a < b ? a : b; }
template <typename T> ROUTE_HD inline T routeMax(T a, T b) { return a > b ? a : b; }

template <typename T>
ROUTE_HD inline T routeOverlap(T c, T size, T lower, T bin_size) {
  return routeMax(T(0), routeMin(c + size, lower + bin_size) - routeMax(c, lower));
}

template <typename T>
ROUTE_HD inline T routeVerticalEdge(T edge, bool right, T x, T y, T width, T height,
                                   const T* psi, int nx, int ny, T xl, T yl,
                                   T bin_x, T bin_y) {
  if (edge < xl || edge > xl + nx * bin_x) return T(0);
  int k = int(floor((edge - xl) / bin_x));
  if (right && edge == xl + k * bin_x) --k;
  if (k < 0 || k >= nx) return T(0);
  const T lower = xl + k * bin_x;
  if (routeOverlap(x, width, lower, bin_x) <= 0) return T(0);
  const T tie = right ? lower + bin_x : lower;
  const T factor = edge == tie ? T(0.5) : T(1);
  int lo = routeMax(int(floor((y - yl) / bin_y)), 0);
  int hi = routeMin(int(floor((y + height - yl) / bin_y)) + 1, ny);
  T result = 0;
  for (int j = lo; j < hi; ++j)
    result += psi[k * ny + j] * routeOverlap(y, height, yl + j * bin_y, bin_y);
  return factor * result;
}

template <typename T>
ROUTE_HD inline T routeHorizontalEdge(T edge, bool top, T x, T y, T width, T height,
                                     const T* psi, int nx, int ny, T xl, T yl,
                                     T bin_x, T bin_y) {
  if (edge < yl || edge > yl + ny * bin_y) return T(0);
  int j = int(floor((edge - yl) / bin_y));
  if (top && edge == yl + j * bin_y) --j;
  if (j < 0 || j >= ny) return T(0);
  const T lower = yl + j * bin_y;
  if (routeOverlap(y, height, lower, bin_y) <= 0) return T(0);
  const T tie = top ? lower + bin_y : lower;
  const T factor = edge == tie ? T(0.5) : T(1);
  int lo = routeMax(int(floor((x - xl) / bin_x)), 0);
  int hi = routeMin(int(floor((x + width - xl) / bin_x)) + 1, nx);
  T result = 0;
  for (int k = lo; k < hi; ++k)
    result += psi[k * ny + j] * routeOverlap(x, width, xl + k * bin_x, bin_x);
  return factor * result;
}

template <typename T>
ROUTE_HD inline void segmentEnergyGradient(T x, T y, T width, T height,
                                         T ratio, T weight, const T* psi,
                                         int nx, int ny, T xl, T yl,
                                         T bin_x, T bin_y, T* out) {
  for (int j = 0; j < 4; ++j) out[j] = 0;
  if (width <= 0 || height <= 0 || x + width <= xl || y + height <= yl ||
      x >= xl + nx * bin_x || y >= yl + ny * bin_y) return;
  T left = routeVerticalEdge(x, false, x, y, width, height, psi, nx, ny, xl, yl, bin_x, bin_y);
  T right = routeVerticalEdge(x + width, true, x, y, width, height, psi, nx, ny, xl, yl, bin_x, bin_y);
  T bottom = routeHorizontalEdge(y, false, x, y, width, height, psi, nx, ny, xl, yl, bin_x, bin_y);
  T top = routeHorizontalEdge(y + height, true, x, y, width, height, psi, nx, ny, xl, yl, bin_x, bin_y);
  out[0] = ratio * (right - left);
  out[1] = ratio * (top - bottom);
  out[2] = ratio * right;
  out[3] = ratio * top;
  if (width * height < T(1e-10)) {
    T integral = 0;
    int lo_x = routeMax(int(floor((x - xl) / bin_x)), 0);
    int hi_x = routeMin(int(floor((x + width - xl) / bin_x)) + 1, nx);
    int lo_y = routeMax(int(floor((y - yl) / bin_y)), 0);
    int hi_y = routeMin(int(floor((y + height - yl) / bin_y)) + 1, ny);
    for (int k = lo_x; k < hi_x; ++k)
      for (int j = lo_y; j < hi_y; ++j)
        integral += psi[k * ny + j] * routeOverlap(x, width, xl + k * bin_x, bin_x)
                    * routeOverlap(y, height, yl + j * bin_y, bin_y);
    out[2] += weight * height / T(1e-10) * integral;
    out[3] += weight * width / T(1e-10) * integral;
  }
}

DREAMPLACE_END_NAMESPACE
#undef ROUTE_HD
#endif
