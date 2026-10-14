#pragma once

#include <cmath>
#include <cstdint>
#include "timing_math.h"

template <typename scalar_t>
struct TimingLutTable {
  const scalar_t *values, *trans, *caps;
  const int64_t *trans_dims, *cap_dims;
  int rows, value_stride, trans_stride, cap_stride;
};

template <typename scalar_t>
struct TimingLutQuery {
  scalar_t value, dt, dc;
};

template <typename scalar_t>
TIMING_INLINE int timingLutBracket(const scalar_t* axis, int count, scalar_t x) {
  int lo = 0, hi = count;
  while (lo < hi) {
    const int mid = (lo + hi) / 2;
    if (x < axis[mid]) hi = mid;
    else lo = mid + 1;
  }
  return max(1, min(lo, count - 1));
}

template <typename scalar_t>
TIMING_INLINE TimingLutQuery<scalar_t> timingLutQuery(
    TimingLutTable<scalar_t> table, int64_t row, scalar_t t, scalar_t c,
    bool extrapolate_1d = false) {
  if (row < 0) row += table.rows;
  TIMING_ASSERT(row >= 0 && row < table.rows);
  const int nt = table.trans_dims[row], nc = table.cap_dims[row];
  scalar_t value = 0, dt = 0, dc = 0;
  const scalar_t* v = table.values + row * table.value_stride;
  const scalar_t* tx = table.trans + row * table.trans_stride;
  const scalar_t* cx = table.caps + row * table.cap_stride;
  if (nt <= 1 && nc <= 1) {
    if (nt > 0 || nc > 0) value = v[0];
  } else if (nt <= 1 || nc <= 1) {
    const bool use_trans = nt > 1;
    const scalar_t x = use_trans ? t : c;
    const scalar_t* axis = use_trans ? tx : cx;
    const int hi = timingLutBracket(axis, use_trans ? nt : nc, x), lo = hi - 1;
    const scalar_t interval = axis[hi] - axis[lo];
    value = v[lo];
    if (fabs(static_cast<double>(interval)) >= 1e-12) {
      const scalar_t position = (x - axis[lo]) / interval;
      const scalar_t fraction = extrapolate_1d ? position : max(scalar_t(0), min(scalar_t(1), position));
      value += fraction * (v[hi] - v[lo]);
      const scalar_t slope = extrapolate_1d || (x >= axis[lo] && x <= axis[hi])
          ? (v[hi] - v[lo]) / interval : scalar_t(0);
      if (use_trans) dt = slope;
      else dc = slope;
    }
  } else {
    const int th = timingLutBracket(tx, nt, t), tl = th - 1;
    const int ch = timingLutBracket(cx, nc, c), cl = ch - 1;
    const scalar_t t0 = tx[tl], t1 = tx[th], c0 = cx[cl], c1 = cx[ch];
    const scalar_t v00 = v[tl * nc + cl], v01 = v[tl * nc + ch];
    const scalar_t v10 = v[th * nc + cl], v11 = v[th * nc + ch];
    const scalar_t ti = t1 - t0, ci = c1 - c0;
    const bool td = fabs(static_cast<double>(ti)) < 1e-12;
    const bool cd = fabs(static_cast<double>(ci)) < 1e-12;
    if (td && cd) value = v00;
    else if (td) {
      dc = (v01 - v00) / ci;
      value = v00 + (c - c0) * dc;
    } else if (cd) {
      dt = (v10 - v00) / ti;
      value = v00 + (t - t0) * dt;
    } else {
      const scalar_t denom = ti * ci;
      const scalar_t wa = (t1 - t) * (c1 - c);
      const scalar_t wb = (t1 - t) * (c - c0);
      const scalar_t wc = (t - t0) * (c1 - c);
      const scalar_t wd = (t - t0) * (c - c0);
      value = (v00 * wa + v01 * wb + v10 * wc + v11 * wd) / denom;
      dt = ((v10 - v00) * (c1 - c) + (v11 - v01) * (c - c0)) / denom;
      dc = ((v01 - v00) * (t1 - t) + (v11 - v10) * (t - t0)) / denom;
    }
  }
  const bool finite = isfinite(value);
  return {finite ? value : scalar_t(0), finite ? dt : scalar_t(0), finite ? dc : scalar_t(0)};
}
