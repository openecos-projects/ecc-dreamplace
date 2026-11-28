// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Shared CPU timing math. These functions take raw pointers so tensor
// wrappers and graph-owned std::vector state execute the same operations.
#include "cell_aat_math.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace dreamplace {
namespace timing {

template <typename scalar_t>
void hardMaximum(scalar_t& destination, scalar_t value) {
  if (std::isnan(destination) || std::isnan(value)) {
    destination = std::numeric_limits<scalar_t>::quiet_NaN();
  } else if (value > destination) {
    destination = value;
  }
}

template <typename scalar_t>
scalar_t hardMinimum(scalar_t left, scalar_t right) {
  if (std::isnan(left) || std::isnan(right)) {
    return std::numeric_limits<scalar_t>::quiet_NaN();
  }
  return left < right ? left : right;
}

template <typename scalar_t>
void aggregateCellRows(const int64_t* arc_ptr, const scalar_t* update_ptr,
                       int64_t count, int64_t columns,
                       scalar_t* ra, scalar_t* fa, scalar_t* rs, scalar_t* fs) {
  for (int64_t index = 0; index < count; ++index) {
    const int64_t output = arc_ptr[index * columns + 1];
    const scalar_t* update = update_ptr + index * 8;
    hardMaximum(ra[output], update[0]);
    hardMaximum(fa[output], update[1]);
    hardMaximum(rs[output], update[2]);
    hardMaximum(fs[output], update[3]);
  }
}

template <typename scalar_t>
void aggregateNetRows(const int64_t* arc_ptr, const scalar_t* update_ptr,
                      int64_t count, int64_t columns,
                      scalar_t* ra, scalar_t* fa, scalar_t* rs, scalar_t* fs) {
  for (int64_t index = 0; index < count; ++index) {
    const int64_t output = arc_ptr[index * columns + 1];
    const scalar_t* update = update_ptr + index * 4;
    hardMaximum(ra[output], update[0]);
    hardMaximum(fa[output], update[1]);
    hardMaximum(rs[output], update[2]);
    hardMaximum(fs[output], update[3]);
  }
}

template <typename scalar_t>
void cellRatRows(const int64_t* arc_ptr, int64_t count, int64_t columns,
                 const scalar_t* rr, const scalar_t* fr,
                 const scalar_t* drr, const scalar_t* dfr,
                 const scalar_t* drf, const scalar_t* dff,
                 scalar_t* out_r, scalar_t* out_f) {
  for (int64_t index = 0; index < count; ++index) {
    const int64_t* arc = arc_ptr + index * columns;
    const int64_t input = arc[0];
    const int64_t output = arc[1];
    scalar_t rise_value;
    scalar_t fall_value;
    if (arc[4] == 1) {
      rise_value = rr[output] - drr[index];
      fall_value = fr[output] - dff[index];
    } else if (arc[4] == -1) {
      rise_value = fr[output] - drf[index];
      fall_value = rr[output] - dfr[index];
    } else {
      rise_value = hardMinimum(rr[output] - drr[index],
                               fr[output] - drf[index]);
      fall_value = hardMinimum(rr[output] - dfr[index],
                               fr[output] - dff[index]);
    }
    out_r[input] = hardMinimum(out_r[input], rise_value);
    out_f[input] = hardMinimum(out_f[input], fall_value);
  }
}

template <typename scalar_t>
void netRatRows(const int64_t* arc_ptr, int64_t count, int64_t columns,
                const scalar_t* rr, const scalar_t* fr,
                const scalar_t* delay_rise, const scalar_t* delay_fall,
                scalar_t* out_r, scalar_t* out_f) {
  for (int64_t index = 0; index < count; ++index) {
    const int64_t* arc = arc_ptr + index * columns;
    const int64_t driver = arc[0];
    const int64_t sink = arc[1];
    out_r[driver] = hardMinimum(out_r[driver], rr[sink] - delay_rise[sink]);
    out_f[driver] = hardMinimum(out_f[driver], fr[sink] - delay_fall[sink]);
  }
}

// Raw per-check candidates are also needed for path selection. Do not use the
// already reduced/propagated pin RAT as that check's original deadline.
template <typename scalar_t>
std::pair<scalar_t, scalar_t> setupRatRow(
    const int64_t* arc, scalar_t rise_seed, scalar_t fall_seed,
    const scalar_t* clk_r, const scalar_t* clk_f,
    const scalar_t* data_r, const scalar_t* data_f,
    TimingLutTable<scalar_t> r_setup, TimingLutTable<scalar_t> f_setup,
    bool smooth_pairwise, double alpha) {
  const int64_t clock_pin = arc[0];
  const int64_t data_pin = arc[1];
  const int64_t row = arc[3];
  const auto rr_time = timingLutQuery(
      r_setup, row, clk_r[clock_pin], data_r[data_pin]);
  const auto ff_time = timingLutQuery(
      f_setup, row, clk_f[clock_pin], data_f[data_pin]);
  const auto fr_time = timingLutQuery(
      r_setup, row, clk_f[clock_pin], data_r[data_pin]);
  const auto rf_time = timingLutQuery(
      f_setup, row, clk_r[clock_pin], data_f[data_pin]);
  scalar_t rise_time;
  scalar_t fall_time;
  if (arc[4] == 1) {
    rise_time = rr_time.value;
    fall_time = ff_time.value;
  } else if (arc[4] == -1) {
    rise_time = fr_time.value;
    fall_time = rf_time.value;
  } else {
    scalar_t rise_weight_a, rise_weight_b, fall_weight_a, fall_weight_b;
    rise_time = pairMaximum(rr_time.value, fr_time.value, smooth_pairwise,
                            scalar_t(alpha), rise_weight_a, rise_weight_b);
    fall_time = pairMaximum(ff_time.value, rf_time.value, smooth_pairwise,
                            scalar_t(alpha), fall_weight_a, fall_weight_b);
  }
  return {rise_seed - rise_time, fall_seed - fall_time};
}

template <typename scalar_t>
void setupRatRows(const int64_t* arc_ptr, int64_t count, int64_t columns,
                  const scalar_t* rr, const scalar_t* fr,
                  const scalar_t* clk_r, const scalar_t* clk_f,
                  const scalar_t* data_r, const scalar_t* data_f,
                  TimingLutTable<scalar_t> r_setup, TimingLutTable<scalar_t> f_setup,
                  const bool* valid_ptr, bool smooth_pairwise, double alpha,
                  scalar_t* out_r, scalar_t* out_f) {
  const bool has_valid = valid_ptr != nullptr;
  for (int64_t index = 0; index < count; ++index) {
    const int64_t* arc = arc_ptr + index * columns;
    const int64_t data_pin = arc[1];
    const auto candidate = setupRatRow(
        arc, rr[data_pin], fr[data_pin], clk_r, clk_f, data_r, data_f,
        r_setup, f_setup, smooth_pairwise, alpha);
    if (!has_valid || valid_ptr[index * 2]) {
      out_r[data_pin] = hardMinimum(out_r[data_pin], candidate.first);
    }
    if (!has_valid || valid_ptr[index * 2 + 1]) {
      out_f[data_pin] = hardMinimum(out_f[data_pin], candidate.second);
    }
  }
}

template <typename scalar_t>
void recoveryRatRows(const int64_t* arc_ptr, int64_t count, int64_t columns, int64_t pins,
                     const scalar_t* rr, const scalar_t* fr,
                     const scalar_t* data_r, const scalar_t* data_f,
                     const scalar_t* clock_r, const scalar_t* clock_f,
                     const scalar_t* reference_r, const scalar_t* reference_f,
                     TimingLutTable<scalar_t> r_check, TimingLutTable<scalar_t> f_check,
                     const bool* valid_ptr, scalar_t* out_r, scalar_t* out_f,
                     bool capture_deadline = false,
                     const scalar_t* deadline_r = nullptr, const scalar_t* deadline_f = nullptr,
                     const bool* keep_setup = nullptr) {
  std::vector<unsigned char> touched_r(pins, 0), touched_f(pins, 0);
  for (int64_t index = 0; index < count; ++index) {
    const int64_t* arc = arc_ptr + index * columns;
    const int64_t clock_pin = arc[0];
    const int64_t endpoint_pin = arc[1];
    const int64_t row = arc[3];
    const auto capture_r = capture_deadline ? scalar_t(0) : timingLutQuery(
        r_check, row, clock_r[clock_pin], reference_r[index]).value;
    const auto capture_f = capture_deadline ? scalar_t(0) : timingLutQuery(
        f_check, row, clock_f[clock_pin], reference_f[index]).value;
    const auto anchor_r = capture_deadline ? deadline_r[endpoint_pin] : rr[endpoint_pin];
    const auto anchor_f = capture_deadline ? deadline_f[endpoint_pin] : fr[endpoint_pin];
    const auto live_r = timingLutQuery(
        r_check, row, clock_r[clock_pin], data_r[endpoint_pin]);
    const auto live_f = timingLutQuery(
        f_check, row, clock_f[clock_pin], data_f[endpoint_pin]);
    if (valid_ptr[index * 2]) {
      const scalar_t candidate = anchor_r + capture_r - live_r.value;
      if (!touched_r[endpoint_pin] && !(capture_deadline && keep_setup[endpoint_pin * 2])) {
        out_r[endpoint_pin] = candidate;
        touched_r[endpoint_pin] = 1;
      } else {
        out_r[endpoint_pin] = hardMinimum(out_r[endpoint_pin], candidate);
      }
    }
    if (valid_ptr[index * 2 + 1]) {
      const scalar_t candidate = anchor_f + capture_f - live_f.value;
      if (!touched_f[endpoint_pin] && !(capture_deadline && keep_setup[endpoint_pin * 2 + 1])) {
        out_f[endpoint_pin] = candidate;
        touched_f[endpoint_pin] = 1;
      } else {
        out_f[endpoint_pin] = hardMinimum(out_f[endpoint_pin], candidate);
      }
    }
  }
}

template <typename scalar_t>
void clk2qRows(const int64_t* arc_ptr, int64_t count, int64_t columns, int64_t pins,
               const scalar_t* cap_r, const scalar_t* cap_f,
               const scalar_t* clk_r, const scalar_t* clk_f,
               TimingLutTable<scalar_t> delay_rise, TimingLutTable<scalar_t> delay_fall,
               TimingLutTable<scalar_t> tran_rise, TimingLutTable<scalar_t> tran_fall,
               bool smooth_pairwise, double alpha,
               scalar_t* out_ra, scalar_t* out_fa, scalar_t* out_rs, scalar_t* out_fs) {
  std::vector<unsigned char> touched(pins, 0);
  for (int64_t index = 0; index < count; ++index) {
    const int64_t* arc = arc_ptr + index * columns;
    const int64_t clock_pin = arc[0];
    const int64_t output = arc[1];
    const int64_t row = arc[3];
    const bool rising_edge = arc[5] == 1 || arc[5] == 0;
    const bool falling_edge = arc[5] == -1 || arc[5] == 0;
    scalar_t rise_update = -std::numeric_limits<scalar_t>::infinity();
    scalar_t fall_update = -std::numeric_limits<scalar_t>::infinity();
    scalar_t rise_transition = 0;
    scalar_t fall_transition = 0;
    if (arc[4] == 1) {
      if (rising_edge) {
        rise_update = timingLutQuery(delay_rise, row, clk_r[clock_pin], cap_r[output]).value;
        rise_transition = timingLutQuery(tran_rise, row, clk_r[clock_pin], cap_r[output]).value;
      }
      if (falling_edge) {
        fall_update = timingLutQuery(delay_fall, row, clk_f[clock_pin], cap_f[output]).value;
        fall_transition = timingLutQuery(tran_fall, row, clk_f[clock_pin], cap_f[output]).value;
      }
    } else if (arc[4] == -1) {
      if (rising_edge) {
        fall_update = timingLutQuery(delay_fall, row, clk_r[clock_pin], cap_f[output]).value;
        fall_transition = timingLutQuery(tran_fall, row, clk_r[clock_pin], cap_f[output]).value;
      }
      if (falling_edge) {
        rise_update = timingLutQuery(delay_rise, row, clk_f[clock_pin], cap_r[output]).value;
        rise_transition = timingLutQuery(tran_rise, row, clk_f[clock_pin], cap_r[output]).value;
      }
    } else {
      scalar_t rr = -std::numeric_limits<scalar_t>::infinity();
      scalar_t rf = -std::numeric_limits<scalar_t>::infinity();
      scalar_t fr = -std::numeric_limits<scalar_t>::infinity();
      scalar_t ff = -std::numeric_limits<scalar_t>::infinity();
      scalar_t rr_tran = 0, rf_tran = 0, fr_tran = 0, ff_tran = 0;
      if (rising_edge) {
        rr = timingLutQuery(delay_rise, row, clk_r[clock_pin], cap_r[output]).value;
        rf = timingLutQuery(delay_fall, row, clk_r[clock_pin], cap_f[output]).value;
        rr_tran = timingLutQuery(tran_rise, row, clk_r[clock_pin], cap_r[output]).value;
        rf_tran = timingLutQuery(tran_fall, row, clk_r[clock_pin], cap_f[output]).value;
      }
      if (falling_edge) {
        fr = timingLutQuery(delay_rise, row, clk_f[clock_pin], cap_r[output]).value;
        ff = timingLutQuery(delay_fall, row, clk_f[clock_pin], cap_f[output]).value;
        fr_tran = timingLutQuery(tran_rise, row, clk_f[clock_pin], cap_r[output]).value;
        ff_tran = timingLutQuery(tran_fall, row, clk_f[clock_pin], cap_f[output]).value;
      }
      scalar_t rise_a, rise_b, fall_a, fall_b;
      rise_update = pairMaximum(rr, fr, smooth_pairwise, scalar_t(alpha), rise_a, rise_b);
      fall_update = pairMaximum(rf, ff, smooth_pairwise, scalar_t(alpha), fall_a, fall_b);
      rise_transition = pairMaximum(rr_tran, fr_tran, smooth_pairwise, scalar_t(alpha), rise_a, rise_b);
      fall_transition = pairMaximum(rf_tran, ff_tran, smooth_pairwise, scalar_t(alpha), fall_a, fall_b);
    }
    if (!touched[output]) {
      out_ra[output] = rise_update;
      out_fa[output] = fall_update;
      out_rs[output] = rise_transition;
      out_fs[output] = fall_transition;
      touched[output] = 1;
    } else {
      hardMaximum(out_ra[output], rise_update);
      hardMaximum(out_fa[output], fall_update);
      hardMaximum(out_rs[output], rise_transition);
      hardMaximum(out_fs[output], fall_transition);
    }
  }
}

}  // namespace timing
}  // namespace dreamplace
