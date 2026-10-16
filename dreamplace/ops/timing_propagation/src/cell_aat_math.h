#pragma once

#include "lut_query.h"

template <typename scalar_t>
TIMING_INLINE scalar_t pairMaximum(
    scalar_t a, scalar_t b, bool smooth, scalar_t alpha, scalar_t& wa, scalar_t& wb) {
  if (isnan(a) || isnan(b)) {
    wa = wb = smooth ? scalar_t(NAN) : scalar_t(1);
    return scalar_t(NAN);
  }
  if (!smooth) {
    wa = a > b ? 1 : (a == b ? scalar_t(.5) : 0);
    wb = 1 - wa;
    return max(a, b);
  }
  const scalar_t x = alpha * a, y = alpha * b, maximum = max(x, y);
  scalar_t total;
  if (isinf(maximum)) total = maximum;
  else total = maximum + log(exp(x - maximum) + exp(y - maximum));
  wa = exp(x - total);
  wb = exp(y - total);
  return total / alpha;
}

template <typename scalar_t>
TIMING_INLINE void queryJacobian(
    scalar_t* jac, int output, TimingLutQuery<scalar_t> query,
    int slew_column, int cap_column) {
  jac[output * 6 + slew_column] = query.dt;
  jac[output * 6 + cap_column] = query.dc;
}

template <typename scalar_t>
TIMING_INLINE void cellForwardRow(int i,
    const int64_t* arcs, int columns, int count, int pins,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    const scalar_t* rc, const scalar_t* fc, TimingLutTable<scalar_t> rd,
    TimingLutTable<scalar_t> fd, TimingLutTable<scalar_t> rt, TimingLutTable<scalar_t> ft,
    scalar_t* updates, scalar_t* derivatives, bool smooth, scalar_t alpha) {
  const int64_t* arc = arcs + i * columns;
  const int64_t in = arc[0], out = arc[1], row = arc[3], sense = arc[4];
  TIMING_ASSERT(in >= 0 && in < pins && out >= 0 && out < pins);
  scalar_t* v = updates + i * 8;
  scalar_t* j = derivatives + i * 48;
  for (int k = 0; k < 48; ++k) j[k] = 0;
  for (int k = 4; k < 8; ++k) v[k] = -INFINITY;
  if (sense == 1 || sense == -1) {
    const bool positive = sense == 1;
    const scalar_t r_slew = positive ? rs[in] : fs[in];
    const scalar_t f_slew = positive ? fs[in] : rs[in];
    const auto dr = timingLutQuery(rd, row, r_slew, rc[out]);
    const auto df = timingLutQuery(fd, row, f_slew, fc[out]);
    const auto tr = timingLutQuery(rt, row, r_slew, rc[out]);
    const auto tf = timingLutQuery(ft, row, f_slew, fc[out]);
    v[0] = (positive ? ra[in] : fa[in]) + dr.value;
    v[1] = (positive ? fa[in] : ra[in]) + df.value;
    v[2] = tr.value; v[3] = tf.value;
    const int rise_slew = positive ? 2 : 3, fall_slew = positive ? 3 : 2;
    const int rise_delay = positive ? 4 : 5, fall_delay = positive ? 7 : 6;
    j[positive ? 0 : 1] = 1;
    j[6 + (positive ? 1 : 0)] = 1;
    queryJacobian(j, 0, dr, rise_slew, 4);
    queryJacobian(j, 1, df, fall_slew, 5);
    queryJacobian(j, 2, tr, rise_slew, 4);
    queryJacobian(j, 3, tf, fall_slew, 5);
    v[rise_delay] = dr.value; v[fall_delay] = df.value;
    queryJacobian(j, rise_delay, dr, rise_slew, 4);
    queryJacobian(j, fall_delay, df, fall_slew, 5);
  } else {
    const auto drr = timingLutQuery(rd, row, rs[in], rc[out]);
    const auto dfr = timingLutQuery(rd, row, fs[in], rc[out]);
    const auto dff = timingLutQuery(fd, row, fs[in], fc[out]);
    const auto drf = timingLutQuery(fd, row, rs[in], fc[out]);
    const auto trr = timingLutQuery(rt, row, rs[in], rc[out]);
    const auto tfr = timingLutQuery(rt, row, fs[in], rc[out]);
    const auto tff = timingLutQuery(ft, row, fs[in], fc[out]);
    const auto trf = timingLutQuery(ft, row, rs[in], fc[out]);
    scalar_t a, b;
    v[0] = pairMaximum(ra[in] + drr.value, fa[in] + dfr.value, smooth, alpha, a, b);
    j[0] = a; j[1] = b; j[2] = a * drr.dt; j[3] = b * dfr.dt;
    j[4] = a * drr.dc + b * dfr.dc;
    v[1] = pairMaximum(fa[in] + dff.value, ra[in] + drf.value, smooth, alpha, a, b);
    j[6] = b; j[7] = a; j[8] = b * drf.dt; j[9] = a * dff.dt;
    j[11] = a * dff.dc + b * drf.dc;
    v[2] = pairMaximum(trr.value, tfr.value, smooth, alpha, a, b);
    j[14] = a * trr.dt; j[15] = b * tfr.dt; j[16] = a * trr.dc + b * tfr.dc;
    v[3] = pairMaximum(tff.value, trf.value, smooth, alpha, a, b);
    j[20] = b * trf.dt; j[21] = a * tff.dt; j[23] = a * tff.dc + b * trf.dc;
    v[4] = drr.value; v[5] = dfr.value; v[6] = drf.value; v[7] = dff.value;
    queryJacobian(j, 4, drr, 2, 4);
    queryJacobian(j, 5, dfr, 3, 4);
    queryJacobian(j, 6, drf, 2, 5);
    queryJacobian(j, 7, dff, 3, 5);
  }
}
template <typename scalar_t>
TIMING_INLINE void cellGroupedBackwardRow(int pin,
    const int64_t* input_order, const int64_t* input_offsets,
    const int64_t* output_order, const int64_t* output_offsets,
    const scalar_t* jac, const scalar_t* gradient,
    scalar_t* ra, scalar_t* fa, scalar_t* rs, scalar_t* fs, scalar_t* rc, scalar_t* fc,
    int pins) {
  scalar_t input_result[4] = {0, 0, 0, 0};
  scalar_t output_result[2] = {0, 0};
  for (int64_t row = input_offsets[pin]; row < input_offsets[pin + 1]; ++row) {
    const int64_t arc = input_order[row];
    scalar_t local[4] = {0, 0, 0, 0};
    for (int o = 0; o < 8; ++o) {
      for (int c = 0; c < 4; ++c) local[c] += gradient[arc * 8 + o] * jac[arc * 48 + o * 6 + c];
    }
    for (int c = 0; c < 4; ++c) input_result[c] += local[c];
  }
  for (int64_t row = output_offsets[pin]; row < output_offsets[pin + 1]; ++row) {
    const int64_t arc = output_order[row];
    scalar_t local[2] = {0, 0};
    for (int o = 0; o < 8; ++o) {
      for (int c = 0; c < 2; ++c) local[c] += gradient[arc * 8 + o] * jac[arc * 48 + o * 6 + c + 4];
    }
    for (int c = 0; c < 2; ++c) output_result[c] += local[c];
  }
  ra[pin] = input_result[0]; fa[pin] = input_result[1];
  rs[pin] = input_result[2]; fs[pin] = input_result[3];
  rc[pin] = output_result[0]; fc[pin] = output_result[1];
}
