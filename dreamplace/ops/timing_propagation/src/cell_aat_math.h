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
TIMING_INLINE scalar_t pairEarly(
    scalar_t a, scalar_t b, bool smooth, scalar_t alpha, scalar_t& wa, scalar_t& wb) {
  if (isinf(a) && a > 0 && isinf(b) && b > 0) {
    wa = wb = 0;
    return INFINITY;
  }
  return -pairMaximum(-a, -b, smooth, alpha, wa, wb);
}

template <typename scalar_t>
TIMING_INLINE scalar_t maskedArrivalPair(
    scalar_t a, scalar_t b, bool use_a, bool use_b, bool smooth,
    scalar_t alpha, bool early, scalar_t& wa, scalar_t& wb) {
  if (!use_a || !use_b) {
    wa = use_a ? 1 : 0;
    wb = use_b ? 1 : 0;
    return use_a ? a : use_b ? b : early ? scalar_t(INFINITY) : scalar_t(-2e8);
  }
  return early ? pairEarly(a, b, smooth, alpha, wa, wb)
               : pairMaximum(a, b, smooth, alpha, wa, wb);
}

template <typename scalar_t>
TIMING_INLINE void queryJacobian(
    scalar_t* jac, int output, TimingLutQuery<scalar_t> query,
    int slew_column, int cap_column) {
  jac[output * 6 + slew_column] = query.dt;
  jac[output * 6 + cap_column] = query.dc;
}

template <typename scalar_t>
TIMING_INLINE TimingLutQuery<scalar_t> cellLutQuery(
    TimingLutTable<scalar_t> table, int64_t row, scalar_t slew, scalar_t cap, bool early) {
  if (early && !isfinite(slew)) return {scalar_t(INFINITY), scalar_t(0), scalar_t(0)};
  return timingLutQuery(table, row, slew, cap);
}

template <typename scalar_t>
TIMING_INLINE void cellForwardRow(int i,
    const int64_t* arcs, int columns, int count, int pins,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    const scalar_t* rc, const scalar_t* fc, TimingLutTable<scalar_t> rd,
    TimingLutTable<scalar_t> fd, TimingLutTable<scalar_t> rt, TimingLutTable<scalar_t> ft,
    scalar_t* updates, scalar_t* derivatives, bool smooth, scalar_t alpha, bool early = false) {
  const int64_t* arc = arcs + i * columns;
  const int64_t in = arc[0], out = arc[1], row = arc[3], sense = arc[4];
  const int mask = columns > 8 ? arc[8] : 15;
  TIMING_ASSERT(in >= 0 && in < pins && out >= 0 && out < pins);
  scalar_t* v = updates + i * 8;
  scalar_t* j = derivatives + i * 48;
  for (int k = 0; k < 48; ++k) j[k] = 0;
  for (int k = 4; k < 8; ++k) v[k] = -INFINITY;
  if (sense == 1 || sense == -1) {
    const bool positive = sense == 1;
    const scalar_t r_slew = positive ? rs[in] : fs[in];
    const scalar_t f_slew = positive ? fs[in] : rs[in];
    const auto dr = cellLutQuery(rd, row, r_slew, rc[out], early);
    const auto df = cellLutQuery(fd, row, f_slew, fc[out], early);
    const auto tr = cellLutQuery(rt, row, r_slew, rc[out], early);
    const auto tf = cellLutQuery(ft, row, f_slew, fc[out], early);
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
    for (int output = 0; output < 2; ++output) {
      const int input = positive ? output : 1 - output;
      if (!(mask & (1 << (2 * input + output)))) {
        v[output] = early ? scalar_t(INFINITY) : scalar_t(-2e8);
        for (int column = 0; column < 6; ++column) j[output * 6 + column] = 0;
      }
    }
  } else {
    const auto drr = cellLutQuery(rd, row, rs[in], rc[out], early);
    const auto dfr = cellLutQuery(rd, row, fs[in], rc[out], early);
    const auto dff = cellLutQuery(fd, row, fs[in], fc[out], early);
    const auto drf = cellLutQuery(fd, row, rs[in], fc[out], early);
    const auto trr = cellLutQuery(rt, row, rs[in], rc[out], early);
    const auto tfr = cellLutQuery(rt, row, fs[in], rc[out], early);
    const auto tff = cellLutQuery(ft, row, fs[in], fc[out], early);
    const auto trf = cellLutQuery(ft, row, rs[in], fc[out], early);
    scalar_t a, b;
    v[0] = maskedArrivalPair(ra[in] + drr.value, fa[in] + dfr.value, mask & 1, mask & 4, smooth, alpha, early, a, b);
    j[0] = a; j[1] = b; j[2] = a * drr.dt; j[3] = b * dfr.dt;
    j[4] = a * drr.dc + b * dfr.dc;
    v[1] = maskedArrivalPair(fa[in] + dff.value, ra[in] + drf.value, mask & 8, mask & 2, smooth, alpha, early, a, b);
    j[6] = b; j[7] = a; j[8] = b * drf.dt; j[9] = a * dff.dt;
    j[11] = a * dff.dc + b * drf.dc;
    v[2] = early ? pairEarly(trr.value, tfr.value, false, alpha, a, b)
                 : pairMaximum(trr.value, tfr.value, smooth, alpha, a, b);
    j[14] = a * trr.dt; j[15] = b * tfr.dt; j[16] = a * trr.dc + b * tfr.dc;
    v[3] = early ? pairEarly(tff.value, trf.value, false, alpha, a, b)
                 : pairMaximum(tff.value, trf.value, smooth, alpha, a, b);
    j[20] = b * trf.dt; j[21] = a * tff.dt; j[23] = a * tff.dc + b * trf.dc;
    v[4] = drr.value; v[5] = dfr.value; v[6] = drf.value; v[7] = dff.value;
    queryJacobian(j, 4, drr, 2, 4);
    queryJacobian(j, 5, dfr, 3, 4);
    queryJacobian(j, 6, drf, 2, 5);
    queryJacobian(j, 7, dff, 3, 5);
  }
}

template <typename scalar_t>
TIMING_INLINE void cellValueRowAtLoad(int64_t in, int64_t out, int64_t row,
    int64_t sense, int pins,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    scalar_t rc_load, scalar_t fc_load, TimingLutTable<scalar_t> rd,
    TimingLutTable<scalar_t> fd, TimingLutTable<scalar_t> rt, TimingLutTable<scalar_t> ft,
    scalar_t* v, bool smooth, scalar_t alpha, int mask = 15) {
  TIMING_ASSERT(in >= 0 && in < pins && out >= 0 && out < pins);
  for (int k = 4; k < 8; ++k) v[k] = -INFINITY;
  if (sense == 1 || sense == -1) {
    const bool positive = sense == 1;
    const scalar_t r_slew = positive ? rs[in] : fs[in];
    const scalar_t f_slew = positive ? fs[in] : rs[in];
    const auto dr = timingLutQuery(rd, row, r_slew, rc_load);
    const auto df = timingLutQuery(fd, row, f_slew, fc_load);
    const auto tr = timingLutQuery(rt, row, r_slew, rc_load);
    const auto tf = timingLutQuery(ft, row, f_slew, fc_load);
    v[0] = (mask & (positive ? 1 : 4)) ? (positive ? ra[in] : fa[in]) + dr.value : scalar_t(-2e8);
    v[1] = (mask & (positive ? 8 : 2)) ? (positive ? fa[in] : ra[in]) + df.value : scalar_t(-2e8);
    v[2] = tr.value; v[3] = tf.value;
    v[positive ? 4 : 5] = dr.value;
    v[positive ? 7 : 6] = df.value;
  } else {
    const auto drr = timingLutQuery(rd, row, rs[in], rc_load);
    const auto dfr = timingLutQuery(rd, row, fs[in], rc_load);
    const auto dff = timingLutQuery(fd, row, fs[in], fc_load);
    const auto drf = timingLutQuery(fd, row, rs[in], fc_load);
    const auto trr = timingLutQuery(rt, row, rs[in], rc_load);
    const auto tfr = timingLutQuery(rt, row, fs[in], rc_load);
    const auto tff = timingLutQuery(ft, row, fs[in], fc_load);
    const auto trf = timingLutQuery(ft, row, rs[in], fc_load);
    scalar_t a, b;
    v[0] = maskedArrivalPair(ra[in] + drr.value, fa[in] + dfr.value, mask & 1, mask & 4, smooth, alpha, false, a, b);
    v[1] = maskedArrivalPair(fa[in] + dff.value, ra[in] + drf.value, mask & 8, mask & 2, smooth, alpha, false, a, b);
    v[2] = pairMaximum(trr.value, tfr.value, smooth, alpha, a, b);
    v[3] = pairMaximum(tff.value, trf.value, smooth, alpha, a, b);
    v[4] = drr.value; v[5] = dfr.value; v[6] = drf.value; v[7] = dff.value;
  }
}

template <typename scalar_t>
TIMING_INLINE void cellValueRowAt(int64_t in, int64_t out, int64_t row,
    int64_t sense, int pins,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    const scalar_t* rc, const scalar_t* fc, TimingLutTable<scalar_t> rd,
    TimingLutTable<scalar_t> fd, TimingLutTable<scalar_t> rt, TimingLutTable<scalar_t> ft,
    scalar_t* values, bool smooth, scalar_t alpha, int mask = 15) {
  cellValueRowAtLoad(in, out, row, sense, pins, ra, fa, rs, fs, rc[out], fc[out],
      rd, fd, rt, ft, values, smooth, alpha, mask);
}

template <typename scalar_t>
TIMING_INLINE void cellValueRow(int i,
    const int64_t* arcs, int columns, int count, int pins,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    const scalar_t* rc, const scalar_t* fc, TimingLutTable<scalar_t> rd,
    TimingLutTable<scalar_t> fd, TimingLutTable<scalar_t> rt, TimingLutTable<scalar_t> ft,
    scalar_t* updates, bool smooth, scalar_t alpha) {
  const int64_t* arc = arcs + i * columns;
  cellValueRowAt<scalar_t>(arc[0], arc[1], arc[3], arc[4], pins,
      ra, fa, rs, fs, rc, fc, rd, fd, rt, ft,
      updates + i * 8, smooth, alpha, columns > 8 ? arc[8] : 15);
}

template <typename scalar_t>
TIMING_INLINE void cellValueRow4AtLoad(int64_t in, int64_t out, int64_t row,
    int64_t sense, int pins,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    scalar_t rc_load, scalar_t fc_load, TimingLutTable<scalar_t> rd,
    TimingLutTable<scalar_t> fd, TimingLutTable<scalar_t> rt, TimingLutTable<scalar_t> ft,
    scalar_t* values, bool smooth, scalar_t alpha, int mask = 15) {
  scalar_t all[8];
  cellValueRowAtLoad(in, out, row, sense, pins, ra, fa, rs, fs, rc_load, fc_load,
      rd, fd, rt, ft, all, smooth, alpha, mask);
  for (int k = 0; k < 4; ++k) values[k] = all[k];
}

template <typename scalar_t>
TIMING_INLINE void cellValueRow4At(int64_t in, int64_t out, int64_t row,
    int64_t sense, int pins,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    const scalar_t* rc, const scalar_t* fc, TimingLutTable<scalar_t> rd,
    TimingLutTable<scalar_t> fd, TimingLutTable<scalar_t> rt, TimingLutTable<scalar_t> ft,
    scalar_t* values, bool smooth, scalar_t alpha, int mask = 15) {
  cellValueRow4AtLoad(in, out, row, sense, pins, ra, fa, rs, fs, rc[out], fc[out],
      rd, fd, rt, ft, values, smooth, alpha, mask);
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
