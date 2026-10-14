#pragma once

#include "lut_query.h"

template <typename scalar_t>
TIMING_INLINE void netForwardRow(int arc, const int64_t* arcs, int columns, int count, int pins,
    const scalar_t* ra, const scalar_t* fa, const scalar_t* rs, const scalar_t* fs,
    const scalar_t* rd, const scalar_t* fd, const scalar_t* ri, const scalar_t* fi,
    scalar_t* updates, scalar_t* jac, scalar_t eps) {
  const int64_t in = arcs[arc * columns], out = arcs[arc * columns + 1];
  TIMING_ASSERT(in >= 0 && in < pins && out >= 0 && out < pins);
  const scalar_t r_square = rs[in] * rs[in] + ri[out];
  const scalar_t f_square = fs[in] * fs[in] + fi[out];
  const scalar_t r_slew = sqrt(isnan(r_square) ? r_square : max(r_square, eps));
  const scalar_t f_slew = sqrt(isnan(f_square) ? f_square : max(f_square, eps));
  updates[arc * 4] = ra[in] + rd[out];
  updates[arc * 4 + 1] = fa[in] + fd[out];
  updates[arc * 4 + 2] = r_slew;
  updates[arc * 4 + 3] = f_slew;
  jac[arc * 6] = scalar_t(2) * rs[in];
  jac[arc * 6 + 1] = scalar_t(2) * fs[in];
  jac[arc * 6 + 2] = scalar_t(2) * r_slew;
  jac[arc * 6 + 3] = scalar_t(2) * f_slew;
  jac[arc * 6 + 4] = scalar_t(r_square >= eps);
  jac[arc * 6 + 5] = scalar_t(f_square >= eps);
}
template <typename scalar_t>
TIMING_INLINE void netBackwardRow(int pin, const int64_t* input_order, const int64_t* input_offsets,
    const int64_t* output_order, const int64_t* output_offsets,
    const scalar_t* jac, const scalar_t* gradient,
    scalar_t* ra, scalar_t* fa, scalar_t* rs, scalar_t* fs,
    scalar_t* rd, scalar_t* fd, scalar_t* ri, scalar_t* fi, int pins,
    bool rise_slew_active, bool fall_slew_active) {
  scalar_t input[4] = {0, 0, 0, 0}, output[4] = {0, 0, 0, 0};
  for (int64_t row = input_offsets[pin]; row < input_offsets[pin + 1]; ++row) {
    const int64_t arc = input_order[row];
    input[0] += gradient[arc * 4]; input[1] += gradient[arc * 4 + 1];
    // Clamp backward selects zero before pow backward. A zero multiplier alone
    // would incorrectly pass a NaN adjoint into a clamped impulse input.
    if (rise_slew_active) {
      const scalar_t grad_square = jac[arc * 6 + 4] ? gradient[arc * 4 + 2] / jac[arc * 6 + 2] : 0;
      input[2] += grad_square * jac[arc * 6];
    }
    if (fall_slew_active) {
      const scalar_t grad_square = jac[arc * 6 + 5] ? gradient[arc * 4 + 3] / jac[arc * 6 + 3] : 0;
      input[3] += grad_square * jac[arc * 6 + 1];
    }
  }
  for (int64_t row = output_offsets[pin]; row < output_offsets[pin + 1]; ++row) {
    const int64_t arc = output_order[row];
    output[0] += gradient[arc * 4]; output[1] += gradient[arc * 4 + 1];
    if (rise_slew_active && jac[arc * 6 + 4]) output[2] += gradient[arc * 4 + 2] / jac[arc * 6 + 2];
    if (fall_slew_active && jac[arc * 6 + 5]) output[3] += gradient[arc * 4 + 3] / jac[arc * 6 + 3];
  }
  ra[pin] = input[0]; fa[pin] = input[1]; rs[pin] = input[2]; fs[pin] = input[3];
  rd[pin] = output[0]; fd[pin] = output[1]; ri[pin] = output[2]; fi[pin] = output[3];
}
