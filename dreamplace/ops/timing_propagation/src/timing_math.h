#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>

#ifdef __CUDACC__
#define TIMING_INLINE __device__ inline
#define TIMING_ASSERT CUDA_KERNEL_ASSERT
#else
#define TIMING_INLINE inline
#define TIMING_ASSERT assert
using std::max;
using std::min;
using std::isnan;
using std::isinf;
using std::isfinite;
using std::exp;
using std::log;
using std::sqrt;
using std::floor;
using std::ceil;
#endif
