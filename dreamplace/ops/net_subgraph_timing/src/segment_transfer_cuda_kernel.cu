#include <math.h>
#include <stdint.h>

#include <c10/cuda/CUDAException.h>
#include "cuda_runtime.h"

namespace {

constexpr int32_t kMaxBackwardRepeaterCount = 16;

template <typename scalar_t>
__device__ __forceinline__ int32_t find_axis_hi(
    const scalar_t* axis,
    const int32_t count,
    const scalar_t point) {
  if (count <= 1) {
    return 0;
  }
  if (point < axis[0]) {
    return 1;
  }
  if (point >= axis[count - 1]) {
    return count - 1;
  }
  int32_t lo = 0;
  int32_t hi = count;
  while (hi - lo > 1) {
    const int32_t mid = (hi + lo) / 2;
    if (point >= axis[mid]) {
      lo = mid;
    } else {
      hi = mid;
    }
  }
  return hi;
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t clamp_value(
    const scalar_t value,
    const scalar_t low,
    const scalar_t high) {
  return min(max(value, low), high);
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t interp_size_1d(
    const scalar_t* values,
    const int32_t size_count,
    const scalar_t bsu) {
  if (size_count <= 1) {
    return values[0];
  }
  const scalar_t clipped =
      clamp_value(bsu, scalar_t(0), static_cast<scalar_t>(size_count - 1));
  const int32_t lo = static_cast<int32_t>(floor(static_cast<double>(clipped)));
  const int32_t hi = min(lo + 1, size_count - 1);
  const scalar_t alpha = clipped - static_cast<scalar_t>(lo);
  return (scalar_t(1) - alpha) * values[lo] + alpha * values[hi];
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t interp_size_1d_slope(
    const scalar_t* values,
    const int32_t size_count,
    const scalar_t bsu) {
  if (size_count <= 1) {
    return scalar_t(0);
  }
  if (bsu < scalar_t(0) || bsu > static_cast<scalar_t>(size_count - 1)) {
    return scalar_t(0);
  }
  const scalar_t clipped =
      clamp_value(bsu, scalar_t(0), static_cast<scalar_t>(size_count - 1));
  const int32_t lo = static_cast<int32_t>(floor(static_cast<double>(clipped)));
  const int32_t hi = min(lo + 1, size_count - 1);
  return values[hi] - values[lo];
}

template <typename scalar_t>
__device__ __forceinline__ scalar_t interp_count_1d(
    const scalar_t* values,
    const int32_t max_count,
    const scalar_t z) {
  const scalar_t clipped =
      clamp_value(z, scalar_t(0), static_cast<scalar_t>(max_count));
  const int32_t lo = static_cast<int32_t>(floor(static_cast<double>(clipped)));
  const int32_t hi = min(lo + 1, max_count);
  const scalar_t alpha = clipped - static_cast<scalar_t>(lo);
  return (scalar_t(1) - alpha) * values[lo] + alpha * values[hi];
}

template <typename scalar_t>
struct LutValueAndGrad {
  scalar_t value;
  scalar_t grad_bsu;
  scalar_t grad_slew;
  scalar_t grad_load;
};

template <typename scalar_t>
__device__ __forceinline__ scalar_t lookup_lut_3d(
    const scalar_t* table,
    const int32_t size_count,
    const int32_t slew_count,
    const int32_t load_count,
    const scalar_t* slew_axis,
    const scalar_t* load_axis,
    const scalar_t bsu,
    const scalar_t input_slew,
    const scalar_t output_load) {
  const int32_t slew_hi = find_axis_hi(slew_axis, slew_count, input_slew);
  const int32_t slew_lo = slew_count <= 1 ? 0 : slew_hi - 1;
  const int32_t load_hi = find_axis_hi(load_axis, load_count, output_load);
  const int32_t load_lo = load_count <= 1 ? 0 : load_hi - 1;
  const scalar_t slew_alpha =
      slew_count <= 1
          ? scalar_t(0)
          : (input_slew - slew_axis[slew_lo]) /
                (slew_axis[slew_hi] - slew_axis[slew_lo]);
  const scalar_t load_alpha =
      load_count <= 1
          ? scalar_t(0)
          : (output_load - load_axis[load_lo]) /
                (load_axis[load_hi] - load_axis[load_lo]);

  const scalar_t clipped_bsu =
      clamp_value(bsu, scalar_t(0), static_cast<scalar_t>(size_count - 1));
  const int32_t size_lo = static_cast<int32_t>(floor(static_cast<double>(clipped_bsu)));
  const int32_t size_hi = min(size_lo + 1, size_count - 1);
  const scalar_t size_alpha = clipped_bsu - static_cast<scalar_t>(size_lo);

  auto at = [&](const int32_t size, const int32_t slew, const int32_t load) {
    return table[(size * slew_count + slew) * load_count + load];
  };
  auto interp_size = [&](const int32_t size) {
    const scalar_t v00 = at(size, slew_lo, load_lo);
    const scalar_t v01 = at(size, slew_lo, load_hi);
    const scalar_t v10 = at(size, slew_hi, load_lo);
    const scalar_t v11 = at(size, slew_hi, load_hi);
    const scalar_t v0 = (scalar_t(1) - load_alpha) * v00 + load_alpha * v01;
    const scalar_t v1 = (scalar_t(1) - load_alpha) * v10 + load_alpha * v11;
    return (scalar_t(1) - slew_alpha) * v0 + slew_alpha * v1;
  };
  const scalar_t lo_value = interp_size(size_lo);
  const scalar_t hi_value = interp_size(size_hi);
  return (scalar_t(1) - size_alpha) * lo_value + size_alpha * hi_value;
}

template <typename scalar_t>
__device__ __forceinline__ LutValueAndGrad<scalar_t> lookup_lut_3d_with_grad(
    const scalar_t* table,
    const int32_t size_count,
    const int32_t slew_count,
    const int32_t load_count,
    const scalar_t* slew_axis,
    const scalar_t* load_axis,
    const scalar_t bsu,
    const scalar_t input_slew,
    const scalar_t output_load) {
  const int32_t slew_hi = find_axis_hi(slew_axis, slew_count, input_slew);
  const int32_t slew_lo = slew_count <= 1 ? 0 : slew_hi - 1;
  const int32_t load_hi = find_axis_hi(load_axis, load_count, output_load);
  const int32_t load_lo = load_count <= 1 ? 0 : load_hi - 1;
  const scalar_t slew_den =
      slew_count <= 1 ? scalar_t(1) : slew_axis[slew_hi] - slew_axis[slew_lo];
  const scalar_t load_den =
      load_count <= 1 ? scalar_t(1) : load_axis[load_hi] - load_axis[load_lo];
  const scalar_t slew_alpha =
      slew_count <= 1 ? scalar_t(0) : (input_slew - slew_axis[slew_lo]) / slew_den;
  const scalar_t load_alpha =
      load_count <= 1 ? scalar_t(0) : (output_load - load_axis[load_lo]) / load_den;

  const scalar_t clipped_bsu =
      clamp_value(bsu, scalar_t(0), static_cast<scalar_t>(size_count - 1));
  const int32_t size_lo = static_cast<int32_t>(floor(static_cast<double>(clipped_bsu)));
  const int32_t size_hi = min(size_lo + 1, size_count - 1);
  const scalar_t size_alpha = clipped_bsu - static_cast<scalar_t>(size_lo);

  auto at = [&](const int32_t size, const int32_t slew, const int32_t load) {
    return table[(size * slew_count + slew) * load_count + load];
  };
  auto interp_size = [&](const int32_t size) {
    const scalar_t v00 = at(size, slew_lo, load_lo);
    const scalar_t v01 = at(size, slew_lo, load_hi);
    const scalar_t v10 = at(size, slew_hi, load_lo);
    const scalar_t v11 = at(size, slew_hi, load_hi);
    const scalar_t v0 = (scalar_t(1) - load_alpha) * v00 + load_alpha * v01;
    const scalar_t v1 = (scalar_t(1) - load_alpha) * v10 + load_alpha * v11;
    return (scalar_t(1) - slew_alpha) * v0 + slew_alpha * v1;
  };
  auto interp_size_grad_slew = [&](const int32_t size) {
    if (slew_count <= 1) {
      return scalar_t(0);
    }
    const scalar_t v00 = at(size, slew_lo, load_lo);
    const scalar_t v01 = at(size, slew_lo, load_hi);
    const scalar_t v10 = at(size, slew_hi, load_lo);
    const scalar_t v11 = at(size, slew_hi, load_hi);
    const scalar_t lo_load = (scalar_t(1) - load_alpha) * v00 + load_alpha * v01;
    const scalar_t hi_load = (scalar_t(1) - load_alpha) * v10 + load_alpha * v11;
    return (hi_load - lo_load) / slew_den;
  };
  auto interp_size_grad_load = [&](const int32_t size) {
    if (load_count <= 1) {
      return scalar_t(0);
    }
    const scalar_t v00 = at(size, slew_lo, load_lo);
    const scalar_t v01 = at(size, slew_lo, load_hi);
    const scalar_t v10 = at(size, slew_hi, load_lo);
    const scalar_t v11 = at(size, slew_hi, load_hi);
    const scalar_t lo_slew = (scalar_t(1) - slew_alpha) * v00 + slew_alpha * v10;
    const scalar_t hi_slew = (scalar_t(1) - slew_alpha) * v01 + slew_alpha * v11;
    return (hi_slew - lo_slew) / load_den;
  };

  const scalar_t lo_value = interp_size(size_lo);
  const scalar_t hi_value = interp_size(size_hi);
  const scalar_t value = (scalar_t(1) - size_alpha) * lo_value + size_alpha * hi_value;
  const scalar_t grad_bsu =
      (bsu < scalar_t(0) || bsu > static_cast<scalar_t>(size_count - 1))
          ? scalar_t(0)
          : hi_value - lo_value;
  const scalar_t grad_slew =
      (scalar_t(1) - size_alpha) * interp_size_grad_slew(size_lo) +
      size_alpha * interp_size_grad_slew(size_hi);
  const scalar_t grad_load =
      (scalar_t(1) - size_alpha) * interp_size_grad_load(size_lo) +
      size_alpha * interp_size_grad_load(size_hi);
  return {value, grad_bsu, grad_slew, grad_load};
}

template <typename scalar_t>
struct TransferStateGrad {
  scalar_t delay;
  scalar_t output_slew;
  scalar_t grad_input_slew;
  scalar_t grad_downstream_load;
  scalar_t grad_bsu;
  scalar_t grad_resistance;
  scalar_t grad_capacitance;
};

template <typename scalar_t>
__device__ TransferStateGrad<scalar_t> transfer_state_value_and_grad(
    const scalar_t parent_slew,
    const scalar_t downstream_load,
    const scalar_t resistance,
    const scalar_t capacitance,
    const int32_t count,
    const scalar_t* fractions,
    const scalar_t bsu,
    const scalar_t* buffer_input_cap_by_size,
    const int32_t size_count,
    const scalar_t* slew_axis,
    const int32_t slew_count,
    const scalar_t* load_axis,
    const int32_t load_count,
    const scalar_t* delay_lut,
    const scalar_t* output_slew_lut,
    const scalar_t grad_delay,
    const scalar_t grad_output_slew) {
  const scalar_t log10_value = static_cast<scalar_t>(2.302585092994046);
  const scalar_t epsilon = static_cast<scalar_t>(1e-30);
  auto wire_slew = [&](const scalar_t slew, const scalar_t delay) {
    const scalar_t delta = log10_value * delay;
    return sqrt(slew * slew + delta * delta);
  };
  auto backprop_wire = [&](
      const scalar_t in_slew,
      const scalar_t wire_delay,
      const scalar_t out_slew,
      const scalar_t grad_out_slew,
      scalar_t& grad_in_slew,
      scalar_t& grad_wire_delay) {
    const scalar_t denom = max(out_slew, epsilon);
    grad_in_slew += grad_out_slew * in_slew / denom;
    grad_wire_delay +=
        grad_out_slew * (log10_value * log10_value) * wire_delay / denom;
  };

  if (count == 0) {
    const scalar_t delay = resistance * (downstream_load + scalar_t(0.5) * capacitance);
    const scalar_t out_slew = wire_slew(parent_slew, delay);
    scalar_t grad_parent_slew = scalar_t(0);
    scalar_t grad_wire_delay = grad_delay;
    backprop_wire(
        parent_slew,
        delay,
        out_slew,
        grad_output_slew,
        grad_parent_slew,
        grad_wire_delay);
    return {
        delay,
        out_slew,
        grad_parent_slew,
        grad_wire_delay * resistance,
        scalar_t(0),
        grad_wire_delay * (downstream_load + scalar_t(0.5) * capacitance),
        grad_wire_delay * scalar_t(0.5) * resistance,
    };
  }

  scalar_t wire_delays[kMaxBackwardRepeaterCount + 1];
  scalar_t wire_input_slews[kMaxBackwardRepeaterCount + 1];
  scalar_t wire_output_slews[kMaxBackwardRepeaterCount + 1];
  scalar_t buffer_input_slews[kMaxBackwardRepeaterCount];
  scalar_t buffer_output_loads[kMaxBackwardRepeaterCount];

  const scalar_t buffer_input_cap =
      interp_size_1d(buffer_input_cap_by_size, size_count, bsu);
  const scalar_t buffer_input_cap_grad =
      interp_size_1d_slope(buffer_input_cap_by_size, size_count, bsu);
  scalar_t total_delay = scalar_t(0);
  scalar_t slew = parent_slew;
  for (int32_t step = 0; step < count; ++step) {
    const scalar_t fraction = fractions[step];
    const scalar_t step_resistance = resistance * fraction;
    const scalar_t step_capacitance = capacitance * fraction;
    const scalar_t wire_delay =
        step_resistance * (buffer_input_cap + scalar_t(0.5) * step_capacitance);
    wire_delays[step] = wire_delay;
    wire_input_slews[step] = slew;
    slew = wire_slew(slew, wire_delay);
    wire_output_slews[step] = slew;
    buffer_input_slews[step] = slew;

    const bool has_next_buffer = count > (step + 1);
    const scalar_t next_fraction = fractions[step + 1];
    const scalar_t next_capacitance = capacitance * next_fraction;
    const scalar_t output_load =
        (has_next_buffer ? buffer_input_cap : downstream_load) + next_capacitance;
    buffer_output_loads[step] = output_load;
    const scalar_t buffer_delay = lookup_lut_3d(
        delay_lut,
        size_count,
        slew_count,
        load_count,
        slew_axis,
        load_axis,
        bsu,
        slew,
        output_load);
    const scalar_t buffer_slew = lookup_lut_3d(
        output_slew_lut,
        size_count,
        slew_count,
        load_count,
        slew_axis,
        load_axis,
        bsu,
        slew,
        output_load);
    total_delay += wire_delay + buffer_delay;
    slew = buffer_slew;
  }

  const scalar_t final_fraction = fractions[count];
  const scalar_t final_resistance = resistance * final_fraction;
  const scalar_t final_capacitance = capacitance * final_fraction;
  const scalar_t downstream_wire_delay =
      final_resistance * (downstream_load + scalar_t(0.5) * final_capacitance);
  wire_delays[count] = downstream_wire_delay;
  wire_input_slews[count] = slew;
  const scalar_t output_slew = wire_slew(slew, downstream_wire_delay);
  wire_output_slews[count] = output_slew;
  const scalar_t delay = total_delay + downstream_wire_delay;

  scalar_t grad_slew_state = scalar_t(0);
  scalar_t grad_downstream_load = scalar_t(0);
  scalar_t grad_buffer_input_cap = scalar_t(0);
  scalar_t grad_bsu = scalar_t(0);
  scalar_t grad_resistance = scalar_t(0);
  scalar_t grad_capacitance = scalar_t(0);
  scalar_t grad_final_wire_delay = grad_delay;
  backprop_wire(
      wire_input_slews[count],
      downstream_wire_delay,
      output_slew,
      grad_output_slew,
      grad_slew_state,
      grad_final_wire_delay);
  grad_downstream_load += grad_final_wire_delay * final_resistance;
  grad_resistance += grad_final_wire_delay * final_fraction *
      (downstream_load + scalar_t(0.5) * capacitance * final_fraction);
  grad_capacitance += grad_final_wire_delay * scalar_t(0.5) *
      resistance * final_fraction * final_fraction;

  for (int32_t step = count - 1; step >= 0; --step) {
    const bool has_next_buffer = count > (step + 1);
    const auto delay_lookup = lookup_lut_3d_with_grad(
        delay_lut,
        size_count,
        slew_count,
        load_count,
        slew_axis,
        load_axis,
        bsu,
        buffer_input_slews[step],
        buffer_output_loads[step]);
    const auto slew_lookup = lookup_lut_3d_with_grad(
        output_slew_lut,
        size_count,
        slew_count,
        load_count,
        slew_axis,
        load_axis,
        bsu,
        buffer_input_slews[step],
        buffer_output_loads[step]);

    const scalar_t grad_buffer_delay = grad_delay;
    const scalar_t grad_buffer_slew = grad_slew_state;
    scalar_t grad_buffer_input_slew =
        grad_buffer_delay * delay_lookup.grad_slew +
        grad_buffer_slew * slew_lookup.grad_slew;
    const scalar_t grad_output_load =
        grad_buffer_delay * delay_lookup.grad_load +
        grad_buffer_slew * slew_lookup.grad_load;
    grad_bsu +=
        grad_buffer_delay * delay_lookup.grad_bsu +
        grad_buffer_slew * slew_lookup.grad_bsu;
    if (has_next_buffer) {
      grad_buffer_input_cap += grad_output_load;
    } else {
      grad_downstream_load += grad_output_load;
    }
    grad_capacitance += grad_output_load * fractions[step + 1];

    scalar_t grad_wire_delay = grad_delay;
    scalar_t grad_previous_slew = scalar_t(0);
    backprop_wire(
        wire_input_slews[step],
        wire_delays[step],
        wire_output_slews[step],
        grad_buffer_input_slew,
        grad_previous_slew,
        grad_wire_delay);
    grad_slew_state = grad_previous_slew;
    const scalar_t fraction = fractions[step];
    const scalar_t step_resistance = resistance * fraction;
    grad_buffer_input_cap += grad_wire_delay * step_resistance;
    grad_resistance += grad_wire_delay * fraction *
        (buffer_input_cap + scalar_t(0.5) * capacitance * fraction);
    grad_capacitance += grad_wire_delay * scalar_t(0.5) *
        resistance * fraction * fraction;
  }

  grad_bsu += grad_buffer_input_cap * buffer_input_cap_grad;
  return {
      delay,
      output_slew,
      grad_slew_state,
      grad_downstream_load,
      grad_bsu,
      grad_resistance,
      grad_capacitance,
  };
}

template <typename scalar_t>
__global__ void segmentTransferForwardKernel(
    const scalar_t* input_arrival,
    const scalar_t* input_slew,
    const scalar_t* downstream_load,
    const scalar_t* edge_resistance,
    const scalar_t* edge_capacitance,
    const int64_t* repeater_count,
    const scalar_t* split_fractions,
    const scalar_t* bsu_index,
    const scalar_t* upstream_retained_cap,
    const scalar_t* buffer_input_cap_by_size,
    const scalar_t* buffer_slew_axis,
    const scalar_t* buffer_load_axis,
    const scalar_t* buffer_delay_lut,
    const scalar_t* buffer_output_slew_lut,
    scalar_t* upstream_visible_load,
    scalar_t* segment_delay,
    scalar_t* output_arrival,
    scalar_t* output_slew,
    scalar_t* first_buffer_input_slew,
    scalar_t* first_buffer_output_load,
    scalar_t* first_buffer_delay,
    scalar_t* first_buffer_output_slew,
    int32_t sample_count,
    int32_t nmax,
    int32_t size_count,
    int32_t slew_count,
    int32_t load_count) {
  const int32_t sample = blockIdx.x * blockDim.x + threadIdx.x;
  if (sample >= sample_count) {
    return;
  }
  const scalar_t log10_value = static_cast<scalar_t>(2.302585092994046);
  auto wire_slew = [&](const scalar_t slew, const scalar_t delay) {
    const scalar_t delta = log10_value * delay;
    return sqrt(slew * slew + delta * delta);
  };

  const int64_t count_i64 = repeater_count[sample];
  const int32_t count =
      count_i64 < 0 ? 0 : static_cast<int32_t>(min(count_i64, static_cast<int64_t>(nmax)));
  const scalar_t resistance = edge_resistance[sample];
  const scalar_t capacitance = edge_capacitance[sample];
  const scalar_t load = downstream_load[sample];
  const scalar_t no_buffer_delay = resistance * (load + scalar_t(0.5) * capacitance);
  const scalar_t no_buffer_slew = wire_slew(input_slew[sample], no_buffer_delay);
  if (count == 0) {
    upstream_visible_load[sample] = load + capacitance;
    segment_delay[sample] = no_buffer_delay;
    output_arrival[sample] = input_arrival[sample] + no_buffer_delay;
    output_slew[sample] = no_buffer_slew;
    return;
  }

  const scalar_t bsu = bsu_index[sample];
  const scalar_t buffer_input_cap =
      interp_size_1d(buffer_input_cap_by_size, size_count, bsu);
  scalar_t total_delay = scalar_t(0);
  scalar_t slew = input_slew[sample];
  for (int32_t step = 0; step < count; ++step) {
    const scalar_t step_fraction = split_fractions[sample * (nmax + 1) + step];
    const scalar_t step_resistance = resistance * step_fraction;
    const scalar_t step_capacitance = capacitance * step_fraction;
    const scalar_t wire_delay =
        step_resistance * (buffer_input_cap + scalar_t(0.5) * step_capacitance);
    total_delay += wire_delay;
    slew = wire_slew(slew, wire_delay);
    const bool has_next_buffer = count > (step + 1);
    const scalar_t next_fraction = split_fractions[sample * (nmax + 1) + step + 1];
    const scalar_t next_capacitance = capacitance * next_fraction;
    const scalar_t output_load =
        (has_next_buffer ? buffer_input_cap : load) + next_capacitance;
    const scalar_t buffer_delay = lookup_lut_3d(
        buffer_delay_lut,
        size_count,
        slew_count,
        load_count,
        buffer_slew_axis,
        buffer_load_axis,
        bsu,
        slew,
        output_load);
    const scalar_t buffer_slew = lookup_lut_3d(
        buffer_output_slew_lut,
        size_count,
        slew_count,
        load_count,
        buffer_slew_axis,
        buffer_load_axis,
        bsu,
        slew,
        output_load);
    if (step == 0) {
      first_buffer_input_slew[sample] = slew;
      first_buffer_output_load[sample] = output_load;
      first_buffer_delay[sample] = buffer_delay;
      first_buffer_output_slew[sample] = buffer_slew;
    }
    total_delay += buffer_delay;
    slew = buffer_slew;
  }
  const scalar_t final_fraction = split_fractions[sample * (nmax + 1) + count];
  const scalar_t final_resistance = resistance * final_fraction;
  const scalar_t final_capacitance = capacitance * final_fraction;
  const scalar_t downstream_wire_delay =
      final_resistance * (load + scalar_t(0.5) * final_capacitance);
  const scalar_t delay = total_delay + downstream_wire_delay;
  upstream_visible_load[sample] =
      buffer_input_cap + capacitance * split_fractions[sample * (nmax + 1)] +
      upstream_retained_cap[sample];
  segment_delay[sample] = delay;
  output_arrival[sample] = input_arrival[sample] + delay;
  output_slew[sample] = wire_slew(slew, downstream_wire_delay);
}

template <typename scalar_t>
__global__ void segmentCountCapOverlayKernel(
    const int64_t* edge_parent_compact_id,
    const int64_t* edge_child_compact_id,
    const int64_t* edge_to_segment_id,
    const scalar_t* edge_capacitance,
    const scalar_t* z_value,
    const scalar_t* parent_cap_fraction,
    const scalar_t* child_cap_fraction,
    scalar_t* effective_node_cap,
    int32_t edge_count,
    int32_t segment_count,
    int32_t max_count) {
  const int32_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= edge_count) {
    return;
  }
  const int64_t parent = edge_parent_compact_id[edge];
  const int64_t child = edge_child_compact_id[edge];
  const int64_t seg = edge_to_segment_id[edge];
  scalar_t parent_fraction = scalar_t(1);
  scalar_t child_fraction = scalar_t(1);
  if (seg >= 0 && seg < segment_count) {
    const scalar_t z = z_value[seg];
    parent_fraction = interp_count_1d(
        parent_cap_fraction + seg * (max_count + 1),
        max_count,
        z);
    child_fraction = interp_count_1d(
        child_cap_fraction + seg * (max_count + 1),
        max_count,
        z);
  }
  const scalar_t half_cap = scalar_t(0.5) * edge_capacitance[edge];
  atomicAdd(effective_node_cap + parent, half_cap * parent_fraction);
  atomicAdd(effective_node_cap + child, half_cap * child_fraction);
}

template <typename scalar_t>
__global__ void segmentCountLoadKernel(
    const int64_t* net_topo_start,
    const int64_t* edge_start,
    const int64_t* edge_child_compact_id,
    const int64_t* edge_to_segment_id,
    const scalar_t* z_value,
    const scalar_t* load_input_cap,
    const scalar_t* effective_node_cap,
    scalar_t* node_load,
    scalar_t* segment_upstream_cap,
    int32_t net_count,
    int32_t max_count) {
  const int32_t net = blockIdx.x * blockDim.x + threadIdx.x;
  if (net >= net_count) {
    return;
  }
  const int64_t begin = net_topo_start[net];
  const int64_t end = net_topo_start[net + 1];
  for (int64_t compact = end - 1; compact >= begin; --compact) {
    scalar_t children_load = scalar_t(0);
    for (int64_t edge = edge_start[compact]; edge < edge_start[compact + 1]; ++edge) {
      const int64_t child = edge_child_compact_id[edge];
      const int64_t seg = edge_to_segment_id[edge];
      scalar_t edge_input = node_load[child];
      if (seg >= 0) {
        const scalar_t z = clamp_value(z_value[seg], scalar_t(0), static_cast<scalar_t>(max_count));
        const int32_t lo = static_cast<int32_t>(floor(static_cast<double>(z)));
        const int32_t hi = min(lo + 1, max_count);
        const scalar_t alpha = z - static_cast<scalar_t>(lo);
        const scalar_t cap = load_input_cap[seg];
        const scalar_t lo_value = (lo == 0) ? node_load[child] : cap;
        const scalar_t hi_value = (hi == 0) ? node_load[child] : cap;
        edge_input = (scalar_t(1) - alpha) * lo_value + alpha * hi_value;
        if (segment_upstream_cap != nullptr) {
          segment_upstream_cap[seg] = edge_input;
        }
      }
      children_load += edge_input;
    }
    node_load[compact] = effective_node_cap[compact] + children_load;
  }
}

template <typename scalar_t>
__global__ void segmentCountTimingKernel(
    const int64_t* net_topo_start,
    const int64_t* edge_start,
    const int64_t* edge_child_compact_id,
    const int64_t* edge_to_segment_id,
    const scalar_t* edge_resistance,
    const scalar_t* edge_capacitance,
    const scalar_t* driver_arrival,
    const scalar_t* driver_slew,
    const scalar_t* z_value,
    const scalar_t* bsu_index,
    const scalar_t* load_input_cap,
    const scalar_t* buffer_input_cap_by_size,
    const scalar_t* buffer_slew_axis,
    const scalar_t* buffer_load_axis,
    const scalar_t* buffer_delay_lut,
    const scalar_t* buffer_output_slew_lut,
    const scalar_t* child_cap_fraction,
    const scalar_t* segment_sub_resistance_fraction,
    const scalar_t* node_load,
    scalar_t* node_arrival,
    scalar_t* node_slew,
    scalar_t* segment_delay,
    scalar_t* segment_output_slew,
    int32_t net_count,
    int32_t max_count,
    int32_t size_count,
    int32_t slew_count,
    int32_t load_count) {
  const int32_t net = blockIdx.x * blockDim.x + threadIdx.x;
  if (net >= net_count) {
    return;
  }
  const scalar_t log10_value = static_cast<scalar_t>(2.302585092994046);
  auto wire_slew = [&](const scalar_t slew, const scalar_t delay) {
    const scalar_t delta = log10_value * delay;
    return sqrt(slew * slew + delta * delta);
  };

  const int64_t begin = net_topo_start[net];
  const int64_t end = net_topo_start[net + 1];
  if (begin == end) {
    return;
  }
  const int32_t sub_stride_count = max_count + 1;
  const int32_t sub_stride_segment = sub_stride_count * sub_stride_count;
  node_arrival[begin] = driver_arrival[net];
  node_slew[begin] = driver_slew[net];
  for (int64_t compact = begin; compact < end; ++compact) {
    for (int64_t edge = edge_start[compact]; edge < edge_start[compact + 1]; ++edge) {
      const int64_t child = edge_child_compact_id[edge];
      const int64_t seg = edge_to_segment_id[edge];
      const scalar_t resistance = edge_resistance[edge];
      const scalar_t capacitance = edge_capacitance[edge];
      scalar_t edge_delay = scalar_t(0);
      scalar_t child_slew = node_slew[compact];
      if (seg < 0) {
        const scalar_t wire_delta = resistance * node_load[child];
        edge_delay = wire_delta;
        child_slew = wire_slew(node_slew[compact], wire_delta);
      } else {
        const scalar_t bsu = bsu_index[seg];
        const scalar_t buffer_input_cap =
            interp_size_1d(buffer_input_cap_by_size, size_count, bsu);
        scalar_t delay_lo = scalar_t(0);
        scalar_t delay_hi = scalar_t(0);
        scalar_t slew_lo = scalar_t(0);
        scalar_t slew_hi = scalar_t(0);
        const scalar_t z = clamp_value(z_value[seg], scalar_t(0), static_cast<scalar_t>(max_count));
        const int32_t lo = static_cast<int32_t>(floor(static_cast<double>(z)));
        const int32_t hi = min(lo + 1, max_count);
        const scalar_t alpha = z - static_cast<scalar_t>(lo);
        const scalar_t child_lo = child_cap_fraction[seg * (max_count + 1) + lo];
        const scalar_t child_hi = child_cap_fraction[seg * (max_count + 1) + hi];
        const scalar_t child_fraction =
            (scalar_t(1) - alpha) * child_lo + alpha * child_hi;
        const scalar_t downstream_load =
            node_load[child] - scalar_t(0.5) * capacitance * child_fraction;
        for (int32_t select = 0; select < 2; ++select) {
          const int32_t count = (select == 0) ? lo : hi;
          scalar_t delay_state = scalar_t(0);
          scalar_t slew_state = node_slew[compact];
          const int32_t sub_base = seg * sub_stride_segment + count * sub_stride_count;
          if (count == 0) {
            delay_state =
                resistance * (downstream_load + scalar_t(0.5) * capacitance);
            slew_state = wire_slew(node_slew[compact], delay_state);
          } else {
            for (int32_t step = 0; step < count; ++step) {
              const scalar_t fraction = segment_sub_resistance_fraction[sub_base + step];
              const scalar_t step_resistance = resistance * fraction;
              const scalar_t step_capacitance = capacitance * fraction;
              const scalar_t wire_delay =
                  step_resistance * (buffer_input_cap + scalar_t(0.5) * step_capacitance);
              delay_state += wire_delay;
              slew_state = wire_slew(slew_state, wire_delay);
              const bool has_next_buffer = count > (step + 1);
              const scalar_t next_fraction = segment_sub_resistance_fraction[sub_base + step + 1];
              const scalar_t next_capacitance = capacitance * next_fraction;
              const scalar_t output_load =
                  (has_next_buffer ? buffer_input_cap : downstream_load) + next_capacitance;
              const scalar_t buffer_delay = lookup_lut_3d(
                  buffer_delay_lut,
                  size_count,
                  slew_count,
                  load_count,
                  buffer_slew_axis,
                  buffer_load_axis,
                  bsu,
                  slew_state,
                  output_load);
              const scalar_t buffer_slew = lookup_lut_3d(
                  buffer_output_slew_lut,
                  size_count,
                  slew_count,
                  load_count,
                  buffer_slew_axis,
                  buffer_load_axis,
                  bsu,
                  slew_state,
                  output_load);
              delay_state += buffer_delay;
              slew_state = buffer_slew;
            }
            const scalar_t final_fraction = segment_sub_resistance_fraction[sub_base + count];
            const scalar_t final_resistance = resistance * final_fraction;
            const scalar_t final_capacitance = capacitance * final_fraction;
            const scalar_t downstream_wire_delay =
                final_resistance *
                (downstream_load + scalar_t(0.5) * final_capacitance);
            delay_state += downstream_wire_delay;
            slew_state = wire_slew(slew_state, downstream_wire_delay);
          }
          if (select == 0) {
            delay_lo = delay_state;
            slew_lo = slew_state;
          } else {
            delay_hi = delay_state;
            slew_hi = slew_state;
          }
        }
        edge_delay = (scalar_t(1) - alpha) * delay_lo + alpha * delay_hi;
        child_slew = (scalar_t(1) - alpha) * slew_lo + alpha * slew_hi;
        segment_delay[seg] = edge_delay;
        segment_output_slew[seg] = child_slew;
      }
      node_arrival[child] = node_arrival[compact] + edge_delay;
      node_slew[child] = child_slew;
    }
  }
}

template <typename scalar_t>
__global__ void segmentCountGatherSinkKernel(
    const int64_t* sink_node_compact_id,
    const scalar_t* node_arrival,
    const scalar_t* node_slew,
    const scalar_t* node_load,
    scalar_t* sink_arrival,
    scalar_t* sink_slew,
    scalar_t* sink_load,
    int32_t sink_count) {
  const int32_t sink = blockIdx.x * blockDim.x + threadIdx.x;
  if (sink >= sink_count) {
    return;
  }
  const int64_t compact = sink_node_compact_id[sink];
  sink_arrival[sink] = node_arrival[compact];
  sink_slew[sink] = node_slew[compact];
  sink_load[sink] = node_load[compact];
}

template <typename scalar_t>
__global__ void segmentCountSeedSinkGradKernel(
    const int64_t* sink_node_compact_id,
    const scalar_t* grad_sink_arrival,
    const scalar_t* grad_sink_slew,
    const scalar_t* grad_sink_load,
    scalar_t* grad_arrival,
    scalar_t* grad_slew,
    scalar_t* grad_load,
    int32_t sink_count) {
  const int32_t sink = blockIdx.x * blockDim.x + threadIdx.x;
  if (sink >= sink_count) {
    return;
  }
  const int64_t compact = sink_node_compact_id[sink];
  atomicAdd(grad_arrival + compact, grad_sink_arrival[sink]);
  atomicAdd(grad_slew + compact, grad_sink_slew[sink]);
  atomicAdd(grad_load + compact, grad_sink_load[sink]);
}

template <typename scalar_t>
__global__ void segmentCountSeedRootGradKernel(
    const int64_t* net_topo_start,
    const scalar_t* grad_driver_net_cap,
    scalar_t* grad_load,
    int32_t net_count) {
  const int32_t net = blockIdx.x * blockDim.x + threadIdx.x;
  if (net >= net_count) {
    return;
  }
  const int64_t root = net_topo_start[net];
  atomicAdd(grad_load + root, grad_driver_net_cap[net]);
}

template <typename scalar_t>
__global__ void segmentCountTimingBackwardKernel(
    const int64_t* net_topo_start,
    const int64_t* edge_start,
    const int64_t* edge_child_compact_id,
    const int64_t* edge_to_segment_id,
    const scalar_t* edge_resistance,
    const scalar_t* edge_capacitance,
    const scalar_t* driver_slew,
    const scalar_t* z_value,
    const scalar_t* bsu_index,
    const scalar_t* load_input_cap,
    const scalar_t* buffer_input_cap_by_size,
    const scalar_t* buffer_slew_axis,
    const scalar_t* buffer_load_axis,
    const scalar_t* buffer_delay_lut,
    const scalar_t* buffer_output_slew_lut,
    const scalar_t* child_cap_fraction,
    const scalar_t* segment_sub_resistance_fraction,
    const scalar_t* node_load,
    const scalar_t* node_slew,
    const scalar_t* grad_segment_delay,
    const scalar_t* grad_segment_output_slew,
    scalar_t* grad_load,
    scalar_t* grad_arrival,
    scalar_t* grad_slew,
    scalar_t* grad_z,
    scalar_t* grad_bsu,
    scalar_t* grad_driver_arrival,
    scalar_t* grad_driver_slew,
    scalar_t* grad_edge_resistance,
    scalar_t* grad_edge_capacitance,
    int32_t net_count,
    int32_t max_count,
    int32_t size_count,
    int32_t slew_count,
    int32_t load_count) {
  const int32_t net = blockIdx.x * blockDim.x + threadIdx.x;
  if (net >= net_count) {
    return;
  }
  const scalar_t log10_value = static_cast<scalar_t>(2.302585092994046);
  const scalar_t epsilon = static_cast<scalar_t>(1e-30);
  const int32_t sub_stride_count = max_count + 1;
  const int32_t sub_stride_segment = sub_stride_count * sub_stride_count;
  scalar_t delay_states[kMaxBackwardRepeaterCount + 1];
  scalar_t slew_states[kMaxBackwardRepeaterCount + 1];

  const int64_t begin = net_topo_start[net];
  const int64_t end = net_topo_start[net + 1];
  if (begin == end) {
    return;
  }
  for (int64_t compact = end - 1; compact >= begin; --compact) {
    for (int64_t edge = edge_start[compact]; edge < edge_start[compact + 1]; ++edge) {
      const int64_t child = edge_child_compact_id[edge];
      const int64_t seg = edge_to_segment_id[edge];
      scalar_t grad_edge_delay = grad_arrival[child];
      scalar_t grad_child_slew = grad_slew[child];
      grad_arrival[compact] += grad_arrival[child];
      if (seg >= 0) {
        grad_edge_delay += grad_segment_delay[seg];
        grad_child_slew += grad_segment_output_slew[seg];
      }
      const scalar_t resistance = edge_resistance[edge];
      const scalar_t child_load = node_load[child];
      const scalar_t parent_slew = node_slew[compact];
      if (seg < 0) {
        const scalar_t wire_delta = resistance * child_load;
        const scalar_t child_slew_value = max(node_slew[child], epsilon);
        scalar_t grad_wire_delay = grad_edge_delay;
        grad_slew[compact] += grad_child_slew * parent_slew / child_slew_value;
        grad_wire_delay +=
            grad_child_slew * (log10_value * log10_value) * wire_delta / child_slew_value;
        grad_load[child] += grad_wire_delay * resistance;
        grad_edge_resistance[edge] += grad_wire_delay * child_load;
        continue;
      }

      const scalar_t z =
          clamp_value(z_value[seg], scalar_t(0), static_cast<scalar_t>(max_count));
      const int32_t lo = static_cast<int32_t>(floor(static_cast<double>(z)));
      const int32_t hi = min(lo + 1, max_count);
      const scalar_t alpha = z - static_cast<scalar_t>(lo);
      const scalar_t child_lo = child_cap_fraction[seg * (max_count + 1) + lo];
      const scalar_t child_hi = child_cap_fraction[seg * (max_count + 1) + hi];
      const scalar_t child_fraction =
          (scalar_t(1) - alpha) * child_lo + alpha * child_hi;
      const scalar_t downstream_load =
          child_load - scalar_t(0.5) * edge_capacitance[edge] * child_fraction;
      const scalar_t bsu = bsu_index[seg];
      const int32_t sub_base_zero = seg * sub_stride_segment;
      for (int32_t count = 0; count <= max_count; ++count) {
        const int32_t sub_base = sub_base_zero + count * sub_stride_count;
        const auto state = transfer_state_value_and_grad<scalar_t>(
            parent_slew,
            downstream_load,
            resistance,
            edge_capacitance[edge],
            count,
            segment_sub_resistance_fraction + sub_base,
            bsu,
            buffer_input_cap_by_size,
            size_count,
            buffer_slew_axis,
            slew_count,
            buffer_load_axis,
            load_count,
            buffer_delay_lut,
            buffer_output_slew_lut,
            scalar_t(0),
            scalar_t(0));
        delay_states[count] = state.delay;
        slew_states[count] = state.output_slew;
      }

      grad_z[seg] += grad_edge_delay * (delay_states[hi] - delay_states[lo]);
      grad_z[seg] += grad_child_slew * (slew_states[hi] - slew_states[lo]);

      scalar_t grad_downstream_load = scalar_t(0);
      for (int32_t count_index = 0; count_index < 2; ++count_index) {
        const int32_t count = count_index == 0 ? lo : hi;
        const scalar_t weight = count_index == 0 ? (scalar_t(1) - alpha) : alpha;
        if (weight == scalar_t(0)) {
          continue;
        }
        const int32_t sub_base = sub_base_zero + count * sub_stride_count;
        const auto vjp = transfer_state_value_and_grad<scalar_t>(
            parent_slew,
            downstream_load,
            resistance,
            edge_capacitance[edge],
            count,
            segment_sub_resistance_fraction + sub_base,
            bsu,
            buffer_input_cap_by_size,
            size_count,
            buffer_slew_axis,
            slew_count,
            buffer_load_axis,
            load_count,
            buffer_delay_lut,
            buffer_output_slew_lut,
            grad_edge_delay * weight,
            grad_child_slew * weight);
        grad_slew[compact] += vjp.grad_input_slew;
        grad_downstream_load += vjp.grad_downstream_load;
        grad_bsu[seg] += vjp.grad_bsu;
        grad_edge_resistance[edge] += vjp.grad_resistance;
        grad_edge_capacitance[edge] += vjp.grad_capacitance;
      }
      grad_load[child] += grad_downstream_load;
      grad_edge_capacitance[edge] -=
          scalar_t(0.5) * child_fraction * grad_downstream_load;
      grad_z[seg] -= scalar_t(0.5) * edge_capacitance[edge] *
                     (child_hi - child_lo) * grad_downstream_load;
    }
  }
  grad_driver_arrival[net] += grad_arrival[begin];
  grad_driver_slew[net] += grad_slew[begin];
}

template <typename scalar_t>
__global__ void segmentCountLoadBackwardKernel(
    const int64_t* net_topo_start,
    const int64_t* edge_start,
    const int64_t* edge_child_compact_id,
    const int64_t* edge_to_segment_id,
    const scalar_t* z_value,
    const scalar_t* load_input_cap,
    const scalar_t* node_load,
    const scalar_t* grad_segment_upstream_cap,
    scalar_t* grad_load,
    scalar_t* grad_eff_cap,
    scalar_t* grad_z,
    scalar_t* grad_input_cap,
    int32_t net_count,
    int32_t max_count) {
  const int32_t net = blockIdx.x * blockDim.x + threadIdx.x;
  if (net >= net_count) {
    return;
  }
  const int64_t begin = net_topo_start[net];
  const int64_t end = net_topo_start[net + 1];
  for (int64_t compact = begin; compact < end; ++compact) {
    grad_eff_cap[compact] += grad_load[compact];
    for (int64_t edge = edge_start[compact]; edge < edge_start[compact + 1]; ++edge) {
      const int64_t child = edge_child_compact_id[edge];
      const int64_t seg = edge_to_segment_id[edge];
      scalar_t grad_edge_input = grad_load[compact];
      if (seg >= 0) {
        grad_edge_input += grad_segment_upstream_cap[seg];
      }
      if (seg < 0) {
        grad_load[child] += grad_edge_input;
        continue;
      }
      const scalar_t z =
          clamp_value(z_value[seg], scalar_t(0), static_cast<scalar_t>(max_count));
      const int32_t lo = static_cast<int32_t>(floor(static_cast<double>(z)));
      const int32_t hi = min(lo + 1, max_count);
      const scalar_t alpha = z - static_cast<scalar_t>(lo);
      const scalar_t input_cap = load_input_cap[seg];
      const scalar_t child_load = node_load[child];
      const scalar_t lo_value = (lo == 0) ? child_load : input_cap;
      const scalar_t hi_value = (hi == 0) ? child_load : input_cap;
      grad_z[seg] += grad_edge_input * (hi_value - lo_value);
      if (lo == 0) {
        grad_load[child] += grad_edge_input * (scalar_t(1) - alpha);
      } else {
        grad_input_cap[seg] += grad_edge_input * (scalar_t(1) - alpha);
      }
      if (hi == 0) {
        grad_load[child] += grad_edge_input * alpha;
      } else {
        grad_input_cap[seg] += grad_edge_input * alpha;
      }
    }
  }
}

template <typename scalar_t>
__global__ void segmentCountCapBackwardKernel(
    const int64_t* edge_parent_compact_id,
    const int64_t* edge_child_compact_id,
    const int64_t* edge_to_segment_id,
    const scalar_t* edge_capacitance,
    const scalar_t* z_value,
    const scalar_t* parent_cap_fraction,
    const scalar_t* child_cap_fraction,
    const scalar_t* grad_eff_cap,
    scalar_t* grad_z,
    scalar_t* grad_edge_capacitance,
    int32_t edge_count,
    int32_t segment_count,
    int32_t max_count) {
  const int32_t edge = blockIdx.x * blockDim.x + threadIdx.x;
  if (edge >= edge_count) {
    return;
  }
  const int64_t seg = edge_to_segment_id[edge];
  if (seg >= segment_count) {
    return;
  }
  const int64_t parent = edge_parent_compact_id[edge];
  const int64_t child = edge_child_compact_id[edge];
  if (seg < 0) {
    if (grad_edge_capacitance != nullptr) {
      grad_edge_capacitance[edge] += scalar_t(0.5) *
          (grad_eff_cap[parent] + grad_eff_cap[child]);
    }
    return;
  }
  const scalar_t z =
      clamp_value(z_value[seg], scalar_t(0), static_cast<scalar_t>(max_count));
  const int32_t lo = static_cast<int32_t>(floor(static_cast<double>(z)));
  const int32_t hi = min(lo + 1, max_count);
  const scalar_t alpha = z - static_cast<scalar_t>(lo);
  const scalar_t parent_lo = parent_cap_fraction[seg * (max_count + 1) + lo];
  const scalar_t parent_hi = parent_cap_fraction[seg * (max_count + 1) + hi];
  const scalar_t child_lo = child_cap_fraction[seg * (max_count + 1) + lo];
  const scalar_t child_hi = child_cap_fraction[seg * (max_count + 1) + hi];
  const scalar_t half_cap = scalar_t(0.5) * edge_capacitance[edge];
  const scalar_t parent_fraction =
      (scalar_t(1) - alpha) * parent_lo + alpha * parent_hi;
  const scalar_t child_fraction =
      (scalar_t(1) - alpha) * child_lo + alpha * child_hi;
  if (grad_edge_capacitance != nullptr) {
    grad_edge_capacitance[edge] += scalar_t(0.5) *
        (grad_eff_cap[parent] * parent_fraction +
         grad_eff_cap[child] * child_fraction);
  }
  const scalar_t delta =
      grad_eff_cap[parent] * half_cap * (parent_hi - parent_lo) +
      grad_eff_cap[child] * half_cap * (child_hi - child_lo);
  atomicAdd(grad_z + seg, delta);
}

template <typename scalar_t>
__global__ void segmentCountClampBackwardKernel(
    const scalar_t* z_value,
    const scalar_t* bsu_index,
    scalar_t* grad_z,
    scalar_t* grad_bsu,
    int32_t segment_count,
    int32_t max_count,
    int32_t size_count) {
  const int32_t seg = blockIdx.x * blockDim.x + threadIdx.x;
  if (seg >= segment_count) {
    return;
  }
  if (z_value[seg] < scalar_t(0) ||
      z_value[seg] > static_cast<scalar_t>(max_count)) {
    grad_z[seg] = scalar_t(0);
  }
  if (bsu_index[seg] < scalar_t(0) ||
      bsu_index[seg] > static_cast<scalar_t>(size_count - 1)) {
    grad_bsu[seg] = scalar_t(0);
  }
}

}  // namespace

template <typename scalar_t>
void segmentTransferForwardCudaLauncher(
    const scalar_t* input_arrival,
    const scalar_t* input_slew,
    const scalar_t* downstream_load,
    const scalar_t* edge_resistance,
    const scalar_t* edge_capacitance,
    const int64_t* repeater_count,
    const scalar_t* split_fractions,
    const scalar_t* bsu_index,
    const scalar_t* upstream_retained_cap,
    const scalar_t* buffer_input_cap_by_size,
    const scalar_t* buffer_slew_axis,
    const scalar_t* buffer_load_axis,
    const scalar_t* buffer_delay_lut,
    const scalar_t* buffer_output_slew_lut,
    scalar_t* upstream_visible_load,
    scalar_t* segment_delay,
    scalar_t* output_arrival,
    scalar_t* output_slew,
    scalar_t* first_buffer_input_slew,
    scalar_t* first_buffer_output_load,
    scalar_t* first_buffer_delay,
    scalar_t* first_buffer_output_slew,
    int32_t sample_count,
    int32_t nmax,
    int32_t size_count,
    int32_t slew_count,
    int32_t load_count,
    cudaStream_t stream) {
  const int32_t threads = 256;
  const int32_t blocks = (sample_count + threads - 1) / threads;
  segmentTransferForwardKernel<scalar_t><<<blocks, threads, 0, stream>>>(
      input_arrival,
      input_slew,
      downstream_load,
      edge_resistance,
      edge_capacitance,
      repeater_count,
      split_fractions,
      bsu_index,
      upstream_retained_cap,
      buffer_input_cap_by_size,
      buffer_slew_axis,
      buffer_load_axis,
      buffer_delay_lut,
      buffer_output_slew_lut,
      upstream_visible_load,
      segment_delay,
      output_arrival,
      output_slew,
      first_buffer_input_slew,
      first_buffer_output_load,
      first_buffer_delay,
      first_buffer_output_slew,
      sample_count,
      nmax,
      size_count,
      slew_count,
      load_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void segmentTransferForwardCudaLauncher<float>(
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const int64_t*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    float*,
    float*,
    float*,
    float*,
    float*,
    float*,
    float*,
    float*,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    cudaStream_t);

template void segmentTransferForwardCudaLauncher<double>(
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const int64_t*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    cudaStream_t);

template <typename scalar_t>
void segmentCountTransferForwardCudaLauncher(
    const int64_t* net_topo_start,
    const int64_t* edge_start,
    const int64_t* edge_parent_compact_id,
    const int64_t* edge_child_compact_id,
    const scalar_t* edge_resistance,
    const scalar_t* edge_capacitance,
    const scalar_t* node_capacitance,
    const int64_t* edge_to_segment_id,
    const scalar_t* driver_arrival,
    const scalar_t* driver_slew,
    const scalar_t* z_value,
    const scalar_t* bsu_index,
    const scalar_t* load_input_cap,
    const scalar_t* buffer_input_cap_by_size,
    const scalar_t* buffer_slew_axis,
    const scalar_t* buffer_load_axis,
    const scalar_t* buffer_delay_lut,
    const scalar_t* buffer_output_slew_lut,
    const scalar_t* parent_cap_fraction,
    const scalar_t* child_cap_fraction,
    const scalar_t* segment_sub_resistance_fraction,
    const int64_t* sink_node_compact_id,
    scalar_t* effective_node_cap,
    scalar_t* node_load,
    scalar_t* node_arrival,
    scalar_t* node_slew,
    scalar_t* segment_delay,
    scalar_t* segment_output_slew,
    scalar_t* segment_upstream_cap,
    scalar_t* sink_arrival,
    scalar_t* sink_slew,
    scalar_t* sink_load,
    int32_t net_count,
    int32_t node_count,
    int32_t edge_count,
    int32_t segment_count,
    int32_t sink_count,
    int32_t max_count,
    int32_t size_count,
    int32_t slew_count,
    int32_t load_count,
    cudaStream_t stream) {
  const int32_t threads = 128;
  C10_CUDA_CHECK(cudaMemcpyAsync(
      effective_node_cap,
      node_capacitance,
      static_cast<size_t>(node_count) * sizeof(scalar_t),
      cudaMemcpyDeviceToDevice,
      stream));
  C10_CUDA_CHECK(cudaMemsetAsync(node_load, 0, static_cast<size_t>(node_count) * sizeof(scalar_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(node_arrival, 0, static_cast<size_t>(node_count) * sizeof(scalar_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(node_slew, 0, static_cast<size_t>(node_count) * sizeof(scalar_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(segment_delay, 0, static_cast<size_t>(segment_count) * sizeof(scalar_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(segment_output_slew, 0, static_cast<size_t>(segment_count) * sizeof(scalar_t), stream));
  C10_CUDA_CHECK(cudaMemsetAsync(segment_upstream_cap, 0, static_cast<size_t>(segment_count) * sizeof(scalar_t), stream));

  const int32_t edge_blocks = (edge_count + threads - 1) / threads;
  segmentCountCapOverlayKernel<scalar_t><<<edge_blocks, threads, 0, stream>>>(
      edge_parent_compact_id,
      edge_child_compact_id,
      edge_to_segment_id,
      edge_capacitance,
      z_value,
      parent_cap_fraction,
      child_cap_fraction,
      effective_node_cap,
      edge_count,
      segment_count,
      max_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int32_t net_blocks = (net_count + threads - 1) / threads;
  segmentCountLoadKernel<scalar_t><<<net_blocks, threads, 0, stream>>>(
      net_topo_start,
      edge_start,
      edge_child_compact_id,
      edge_to_segment_id,
      z_value,
      load_input_cap,
      effective_node_cap,
      node_load,
      segment_upstream_cap,
      net_count,
      max_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  segmentCountTimingKernel<scalar_t><<<net_blocks, threads, 0, stream>>>(
      net_topo_start,
      edge_start,
      edge_child_compact_id,
      edge_to_segment_id,
      edge_resistance,
      edge_capacitance,
      driver_arrival,
      driver_slew,
      z_value,
      bsu_index,
      load_input_cap,
      buffer_input_cap_by_size,
      buffer_slew_axis,
      buffer_load_axis,
      buffer_delay_lut,
      buffer_output_slew_lut,
      child_cap_fraction,
      segment_sub_resistance_fraction,
      node_load,
      node_arrival,
      node_slew,
      segment_delay,
      segment_output_slew,
      net_count,
      max_count,
      size_count,
      slew_count,
      load_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int32_t sink_blocks = (sink_count + threads - 1) / threads;
  segmentCountGatherSinkKernel<scalar_t><<<sink_blocks, threads, 0, stream>>>(
      sink_node_compact_id,
      node_arrival,
      node_slew,
      node_load,
      sink_arrival,
      sink_slew,
      sink_load,
      sink_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void segmentCountTransferForwardCudaLauncher<float>(
    const int64_t*,
    const int64_t*,
    const int64_t*,
    const int64_t*,
    const float*,
    const float*,
    const float*,
    const int64_t*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const int64_t*,
    float*,
    float*,
    float*,
    float*,
    float*,
    float*,
    float*,
    float*,
    float*,
    float*,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    cudaStream_t);

template void segmentCountTransferForwardCudaLauncher<double>(
    const int64_t*,
    const int64_t*,
    const int64_t*,
    const int64_t*,
    const double*,
    const double*,
    const double*,
    const int64_t*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const int64_t*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    cudaStream_t);

template <typename scalar_t>
void segmentCountCapForwardCudaLauncher(
    const int64_t* net_topo_start,
    const int64_t* edge_start,
    const int64_t* edge_parent_compact_id,
    const int64_t* edge_child_compact_id,
    const scalar_t* edge_capacitance,
    const scalar_t* node_capacitance,
    const int64_t* edge_to_segment_id,
    const scalar_t* z_value,
    const scalar_t* load_input_cap,
    const scalar_t* parent_cap_fraction,
    const scalar_t* child_cap_fraction,
    scalar_t* effective_node_cap,
    scalar_t* node_load,
    int32_t net_count,
    int32_t node_count,
    int32_t edge_count,
    int32_t segment_count,
    int32_t max_count,
    cudaStream_t stream) {
  const int32_t threads = 128;
  C10_CUDA_CHECK(cudaMemcpyAsync(
      effective_node_cap,
      node_capacitance,
      static_cast<size_t>(node_count) * sizeof(scalar_t),
      cudaMemcpyDeviceToDevice,
      stream));
  C10_CUDA_CHECK(cudaMemsetAsync(
      node_load,
      0,
      static_cast<size_t>(node_count) * sizeof(scalar_t),
      stream));
  const int32_t edge_blocks = (edge_count + threads - 1) / threads;
  segmentCountCapOverlayKernel<scalar_t><<<edge_blocks, threads, 0, stream>>>(
      edge_parent_compact_id,
      edge_child_compact_id,
      edge_to_segment_id,
      edge_capacitance,
      z_value,
      parent_cap_fraction,
      child_cap_fraction,
      effective_node_cap,
      edge_count,
      segment_count,
      max_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int32_t net_blocks = (net_count + threads - 1) / threads;
  segmentCountLoadKernel<scalar_t><<<net_blocks, threads, 0, stream>>>(
      net_topo_start,
      edge_start,
      edge_child_compact_id,
      edge_to_segment_id,
      z_value,
      load_input_cap,
      effective_node_cap,
      node_load,
      nullptr,
      net_count,
      max_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void segmentCountCapForwardCudaLauncher<float>(
    const int64_t*,
    const int64_t*,
    const int64_t*,
    const int64_t*,
    const float*,
    const float*,
    const int64_t*,
    const float*,
    const float*,
    const float*,
    const float*,
    float*,
    float*,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    cudaStream_t);

template void segmentCountCapForwardCudaLauncher<double>(
    const int64_t*,
    const int64_t*,
    const int64_t*,
    const int64_t*,
    const double*,
    const double*,
    const int64_t*,
    const double*,
    const double*,
    const double*,
    const double*,
    double*,
    double*,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    cudaStream_t);

template <typename scalar_t>
void segmentCountTransferBackwardCudaLauncher(
    const int64_t* net_topo_start,
    const int64_t* edge_start,
    const int64_t* edge_parent_compact_id,
    const int64_t* edge_child_compact_id,
    const scalar_t* edge_resistance,
    const scalar_t* edge_capacitance,
    const int64_t* edge_to_segment_id,
    const scalar_t* driver_slew,
    const scalar_t* z_value,
    const scalar_t* bsu_index,
    const scalar_t* load_input_cap,
    const scalar_t* buffer_input_cap_by_size,
    const scalar_t* buffer_slew_axis,
    const scalar_t* buffer_load_axis,
    const scalar_t* buffer_delay_lut,
    const scalar_t* buffer_output_slew_lut,
    const scalar_t* parent_cap_fraction,
    const scalar_t* child_cap_fraction,
    const scalar_t* segment_sub_resistance_fraction,
    const int64_t* sink_node_compact_id,
    const scalar_t* node_load,
    const scalar_t* node_slew,
    const scalar_t* effective_node_cap,
    const scalar_t* grad_segment_delay,
    const scalar_t* grad_segment_output_slew,
    const scalar_t* grad_segment_upstream_cap,
    const scalar_t* grad_sink_arrival,
    const scalar_t* grad_sink_slew,
    const scalar_t* grad_sink_load,
    scalar_t* grad_load,
    scalar_t* grad_arrival,
    scalar_t* grad_slew,
    scalar_t* grad_eff_cap,
    scalar_t* grad_z,
    scalar_t* grad_bsu,
    scalar_t* grad_load_input_cap,
    scalar_t* grad_driver_arrival,
    scalar_t* grad_driver_slew,
    scalar_t* grad_edge_resistance,
    scalar_t* grad_edge_capacitance,
    int32_t net_count,
    int32_t node_count,
    int32_t edge_count,
    int32_t segment_count,
    int32_t sink_count,
    int32_t max_count,
    int32_t size_count,
    int32_t slew_count,
    int32_t load_count,
    cudaStream_t stream) {
  const int32_t threads = 128;

  const int32_t sink_blocks = (sink_count + threads - 1) / threads;
  segmentCountSeedSinkGradKernel<scalar_t><<<sink_blocks, threads, 0, stream>>>(
      sink_node_compact_id,
      grad_sink_arrival,
      grad_sink_slew,
      grad_sink_load,
      grad_arrival,
      grad_slew,
      grad_load,
      sink_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int32_t net_blocks = (net_count + threads - 1) / threads;
  segmentCountTimingBackwardKernel<scalar_t><<<net_blocks, threads, 0, stream>>>(
      net_topo_start,
      edge_start,
      edge_child_compact_id,
      edge_to_segment_id,
      edge_resistance,
      edge_capacitance,
      driver_slew,
      z_value,
      bsu_index,
      load_input_cap,
      buffer_input_cap_by_size,
      buffer_slew_axis,
      buffer_load_axis,
      buffer_delay_lut,
      buffer_output_slew_lut,
      child_cap_fraction,
      segment_sub_resistance_fraction,
      node_load,
      node_slew,
      grad_segment_delay,
      grad_segment_output_slew,
      grad_load,
      grad_arrival,
      grad_slew,
      grad_z,
      grad_bsu,
      grad_driver_arrival,
      grad_driver_slew,
      grad_edge_resistance,
      grad_edge_capacitance,
      net_count,
      max_count,
      size_count,
      slew_count,
      load_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  segmentCountLoadBackwardKernel<scalar_t><<<net_blocks, threads, 0, stream>>>(
      net_topo_start,
      edge_start,
      edge_child_compact_id,
      edge_to_segment_id,
      z_value,
      load_input_cap,
      node_load,
      grad_segment_upstream_cap,
      grad_load,
      grad_eff_cap,
      grad_z,
      grad_load_input_cap,
      net_count,
      max_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int32_t edge_blocks = (edge_count + threads - 1) / threads;
  segmentCountCapBackwardKernel<scalar_t><<<edge_blocks, threads, 0, stream>>>(
      edge_parent_compact_id,
      edge_child_compact_id,
      edge_to_segment_id,
      edge_capacitance,
      z_value,
      parent_cap_fraction,
      child_cap_fraction,
      grad_eff_cap,
      grad_z,
      grad_edge_capacitance,
      edge_count,
      segment_count,
      max_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const int32_t segment_blocks = (segment_count + threads - 1) / threads;
  segmentCountClampBackwardKernel<scalar_t><<<segment_blocks, threads, 0, stream>>>(
      z_value,
      bsu_index,
      grad_z,
      grad_bsu,
      segment_count,
      max_count,
      size_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void segmentCountTransferBackwardCudaLauncher<float>(
    const int64_t*,
    const int64_t*,
    const int64_t*,
    const int64_t*,
    const float*,
    const float*,
    const int64_t*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const int64_t*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    float*,
    float*,
    float*,
    float*,
    float*,
    float*,
    float*,
    float*,
    float*,
    float*,
    float*,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    cudaStream_t);

template void segmentCountTransferBackwardCudaLauncher<double>(
    const int64_t*,
    const int64_t*,
    const int64_t*,
    const int64_t*,
    const double*,
    const double*,
    const int64_t*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const int64_t*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    double*,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    cudaStream_t);

template <typename scalar_t>
void segmentCountCapBackwardCudaLauncher(
    const int64_t* net_topo_start,
    const int64_t* edge_start,
    const int64_t* edge_parent_compact_id,
    const int64_t* edge_child_compact_id,
    const scalar_t* edge_capacitance,
    const int64_t* edge_to_segment_id,
    const scalar_t* z_value,
    const scalar_t* load_input_cap,
    const scalar_t* parent_cap_fraction,
    const scalar_t* child_cap_fraction,
    const scalar_t* node_load,
    const scalar_t* grad_driver_net_cap,
    scalar_t* grad_load,
    scalar_t* grad_effective_node_cap,
    const scalar_t* grad_segment_upstream_cap,
    scalar_t* grad_z,
    scalar_t* grad_load_input_cap,
    int32_t net_count,
    int32_t edge_count,
    int32_t segment_count,
    int32_t max_count,
    cudaStream_t stream) {
  const int32_t threads = 128;
  const int32_t net_blocks = (net_count + threads - 1) / threads;
  segmentCountSeedRootGradKernel<scalar_t><<<net_blocks, threads, 0, stream>>>(
      net_topo_start,
      grad_driver_net_cap,
      grad_load,
      net_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  segmentCountLoadBackwardKernel<scalar_t><<<net_blocks, threads, 0, stream>>>(
      net_topo_start,
      edge_start,
      edge_child_compact_id,
      edge_to_segment_id,
      z_value,
      load_input_cap,
      node_load,
      grad_segment_upstream_cap,
      grad_load,
      grad_effective_node_cap,
      grad_z,
      grad_load_input_cap,
      net_count,
      max_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  const int32_t edge_blocks = (edge_count + threads - 1) / threads;
  segmentCountCapBackwardKernel<scalar_t><<<edge_blocks, threads, 0, stream>>>(
      edge_parent_compact_id,
      edge_child_compact_id,
      edge_to_segment_id,
      edge_capacitance,
      z_value,
      parent_cap_fraction,
      child_cap_fraction,
      grad_effective_node_cap,
      grad_z,
      nullptr,
      edge_count,
      segment_count,
      max_count);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template void segmentCountCapBackwardCudaLauncher<float>(
    const int64_t*,
    const int64_t*,
    const int64_t*,
    const int64_t*,
    const float*,
    const int64_t*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    const float*,
    float*,
    float*,
    const float*,
    float*,
    float*,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    cudaStream_t);

template void segmentCountCapBackwardCudaLauncher<double>(
    const int64_t*,
    const int64_t*,
    const int64_t*,
    const int64_t*,
    const double*,
    const int64_t*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    const double*,
    double*,
    double*,
    const double*,
    double*,
    double*,
    int32_t,
    int32_t,
    int32_t,
    int32_t,
    cudaStream_t);

template <typename scalar_t>
__global__ void candidateIndexByNodeKernel(
    const int64_t* candidate_node_id,
    int64_t* candidate_index_by_node,
    int32_t candidate_count,
    int32_t node_count) {
  const int32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) {
    return;
  }
  const int64_t node = candidate_node_id[candidate];
  if (node >= 0 && node < node_count) {
    candidate_index_by_node[node] = candidate;
  }
}

template <typename scalar_t>
__global__ void candidateNetSubgraphForwardKernel(
    const int64_t* net_flat_topo_sort,
    const int64_t* net_flat_topo_sort_start,
    const int64_t* pin_fa,
    const int64_t* flat_pin_to_start,
    const int64_t* flat_pin_to,
    const scalar_t* edge_resistance,
    const scalar_t* node_capacitance,
    const scalar_t* edge_capacitance,
    const scalar_t* driver_arrival,
    const scalar_t* driver_slew,
    const int64_t* candidate_index_by_node,
    const scalar_t* candidate_bu,
    const scalar_t* buffer_input_cap,
    const scalar_t* buffer_delay,
    const scalar_t* buffer_output_slew,
    scalar_t* effective_node_cap,
    scalar_t* lout,
    scalar_t* lin,
    scalar_t* arrival_in,
    scalar_t* arrival_out,
    scalar_t* slew_in,
    scalar_t* slew_out,
    int32_t net_count) {
  const int32_t net = blockIdx.x * blockDim.x + threadIdx.x;
  if (net >= net_count) {
    return;
  }
  const int64_t begin = net_flat_topo_sort_start[net];
  const int64_t end = net_flat_topo_sort_start[net + 1];
  if (begin == end) {
    return;
  }

  for (int64_t pos = begin; pos < end; ++pos) {
    const int64_t node = net_flat_topo_sort[pos];
    effective_node_cap[node] = node_capacitance[node];
  }
  for (int64_t pos = begin; pos < end; ++pos) {
    const int64_t node = net_flat_topo_sort[pos];
    const int64_t parent = pin_fa[node];
    if (parent >= 0) {
      const scalar_t half_cap = scalar_t(0.5) * edge_capacitance[node];
      effective_node_cap[parent] += half_cap;
      effective_node_cap[node] += half_cap;
    }
  }

  for (int64_t pos = end; pos-- > begin;) {
    const int64_t node = net_flat_topo_sort[pos];
    scalar_t children_load = scalar_t(0);
    for (int64_t edge = flat_pin_to_start[node];
         edge < flat_pin_to_start[node + 1];
         ++edge) {
      children_load += lin[flat_pin_to[edge]];
    }
    const scalar_t node_lout = effective_node_cap[node] + children_load;
    const int64_t candidate = candidate_index_by_node[node];
    const scalar_t bu = candidate >= 0 ? candidate_bu[candidate] : scalar_t(0);
    const scalar_t input_cap =
        candidate >= 0 ? buffer_input_cap[candidate] : scalar_t(0);
    lout[node] = node_lout;
    lin[node] = (scalar_t(1) - bu) * node_lout + bu * input_cap;
  }

  const scalar_t log10_value = scalar_t(2.302585092994046);
  const int64_t root = net_flat_topo_sort[begin];
  arrival_in[root] = driver_arrival[net];
  arrival_out[root] = driver_arrival[net];
  slew_in[root] = driver_slew[net];
  slew_out[root] = driver_slew[net];
  for (int64_t pos = begin; pos < end; ++pos) {
    const int64_t node = net_flat_topo_sort[pos];
    const int64_t parent = pin_fa[node];
    if (parent >= 0) {
      const scalar_t wire_delta = edge_resistance[node] * lin[node];
      arrival_in[node] = arrival_out[parent] + wire_delta;
      const scalar_t slew_delta = log10_value * wire_delta;
      slew_in[node] = sqrt(
          slew_out[parent] * slew_out[parent] + slew_delta * slew_delta);
    }
    const int64_t candidate = candidate_index_by_node[node];
    const scalar_t bu = candidate >= 0 ? candidate_bu[candidate] : scalar_t(0);
    const scalar_t delay = candidate >= 0 ? buffer_delay[candidate] : scalar_t(0);
    const scalar_t output_slew =
        candidate >= 0 ? buffer_output_slew[candidate] : scalar_t(0);
    arrival_out[node] = arrival_in[node] + bu * delay;
    slew_out[node] =
        (scalar_t(1) - bu) * slew_in[node] + bu * output_slew;
  }
}

template <typename scalar_t>
__global__ void candidateNetSubgraphFixedBsuForwardKernel(
    const int64_t* net_flat_topo_sort,
    const int64_t* net_flat_topo_sort_start,
    const int64_t* pin_fa,
    const int64_t* flat_pin_to_start,
    const int64_t* flat_pin_to,
    const scalar_t* edge_resistance,
    const scalar_t* node_capacitance,
    const scalar_t* edge_capacitance,
    const scalar_t* driver_arrival,
    const scalar_t* driver_slew,
    const int64_t* candidate_index_by_node,
    const scalar_t* candidate_bu,
    const scalar_t* buffer_input_cap,
    const scalar_t* probe_buffer_delay,
    const scalar_t* probe_buffer_output_slew,
    scalar_t* effective_node_cap,
    scalar_t* lout,
    scalar_t* lin,
    scalar_t* probe_arrival_in,
    scalar_t* probe_arrival_out,
    scalar_t* probe_slew_in,
    scalar_t* probe_slew_out,
    int32_t net_count) {
  const int32_t net = blockIdx.x * blockDim.x + threadIdx.x;
  if (net >= net_count) {
    return;
  }
  const int64_t begin = net_flat_topo_sort_start[net];
  const int64_t end = net_flat_topo_sort_start[net + 1];
  if (begin == end) {
    return;
  }

  for (int64_t pos = begin; pos < end; ++pos) {
    const int64_t node = net_flat_topo_sort[pos];
    effective_node_cap[node] = node_capacitance[node];
  }
  for (int64_t pos = begin; pos < end; ++pos) {
    const int64_t node = net_flat_topo_sort[pos];
    const int64_t parent = pin_fa[node];
    if (parent >= 0) {
      const scalar_t half_cap = scalar_t(0.5) * edge_capacitance[node];
      effective_node_cap[parent] += half_cap;
      effective_node_cap[node] += half_cap;
    }
  }
  for (int64_t pos = end; pos-- > begin;) {
    const int64_t node = net_flat_topo_sort[pos];
    scalar_t children_load = scalar_t(0);
    for (int64_t edge = flat_pin_to_start[node];
         edge < flat_pin_to_start[node + 1];
         ++edge) {
      children_load += lin[flat_pin_to[edge]];
    }
    const scalar_t node_lout = effective_node_cap[node] + children_load;
    const int64_t candidate = candidate_index_by_node[node];
    const scalar_t bu = candidate >= 0 ? candidate_bu[candidate] : scalar_t(0);
    const scalar_t input_cap =
        candidate >= 0 ? buffer_input_cap[candidate] : scalar_t(0);
    lout[node] = node_lout;
    lin[node] = (scalar_t(1) - bu) * node_lout + bu * input_cap;
  }

  const scalar_t log10_value = scalar_t(2.302585092994046);
  const int64_t root = net_flat_topo_sort[begin];
  probe_arrival_in[root] = driver_arrival[net];
  probe_arrival_out[root] = driver_arrival[net];
  probe_slew_in[root] = driver_slew[net];
  probe_slew_out[root] = driver_slew[net];
  for (int64_t pos = begin; pos < end; ++pos) {
    const int64_t node = net_flat_topo_sort[pos];
    const int64_t parent = pin_fa[node];
    if (parent >= 0) {
      const scalar_t wire_delta = edge_resistance[node] * lin[node];
      probe_arrival_in[node] = probe_arrival_out[parent] + wire_delta;
      const scalar_t slew_delta = log10_value * wire_delta;
      probe_slew_in[node] = sqrt(
          probe_slew_out[parent] * probe_slew_out[parent] +
          slew_delta * slew_delta);
    }
    const int64_t candidate = candidate_index_by_node[node];
    const scalar_t bu = candidate >= 0 ? candidate_bu[candidate] : scalar_t(0);
    const scalar_t delay =
        candidate >= 0 ? probe_buffer_delay[candidate] : scalar_t(0);
    const scalar_t output_slew =
        candidate >= 0 ? probe_buffer_output_slew[candidate] : scalar_t(0);
    probe_arrival_out[node] = probe_arrival_in[node] + bu * delay;
    probe_slew_out[node] =
        (scalar_t(1) - bu) * probe_slew_in[node] + bu * output_slew;
  }

}

template <typename scalar_t>
__global__ void candidateFixedBsuLookupKernel(
    const int64_t* candidate_node_id,
    const scalar_t* lout,
    const scalar_t* probe_slew_in,
    const scalar_t* upstream_retained_cap,
    const scalar_t* buffer_slew_axis,
    const scalar_t* buffer_load_axis,
    const scalar_t* buffer_delay_lut,
    const scalar_t* buffer_output_slew_lut,
    int32_t fixed_bsu_index,
    int32_t size_count,
    int32_t slew_count,
    int32_t load_count,
    scalar_t* candidate_input_slew,
    scalar_t* candidate_output_load,
    scalar_t* buffer_delay,
    scalar_t* buffer_output_slew,
    scalar_t* buffer_delay_grad_slew,
    scalar_t* buffer_delay_grad_load,
    scalar_t* buffer_output_slew_grad_slew,
    scalar_t* buffer_output_slew_grad_load,
    int32_t candidate_count) {
  const int32_t candidate = blockIdx.x * blockDim.x + threadIdx.x;
  if (candidate >= candidate_count) {
    return;
  }
  const int64_t node = candidate_node_id[candidate];
  const scalar_t input_slew = probe_slew_in[node];
  const scalar_t output_load = lout[node] - upstream_retained_cap[candidate];
  const scalar_t bsu = static_cast<scalar_t>(fixed_bsu_index);
  const auto delay_lookup = lookup_lut_3d_with_grad(
      buffer_delay_lut,
      size_count,
      slew_count,
      load_count,
      buffer_slew_axis,
      buffer_load_axis,
      bsu,
      input_slew,
      output_load);
  const auto slew_lookup = lookup_lut_3d_with_grad(
      buffer_output_slew_lut,
      size_count,
      slew_count,
      load_count,
      buffer_slew_axis,
      buffer_load_axis,
      bsu,
      input_slew,
      output_load);
  candidate_input_slew[candidate] = input_slew;
  candidate_output_load[candidate] = output_load;
  buffer_delay[candidate] = delay_lookup.value;
  buffer_output_slew[candidate] = slew_lookup.value;
  buffer_delay_grad_slew[candidate] = delay_lookup.grad_slew;
  buffer_delay_grad_load[candidate] = delay_lookup.grad_load;
  buffer_output_slew_grad_slew[candidate] = slew_lookup.grad_slew;
  buffer_output_slew_grad_load[candidate] = slew_lookup.grad_load;
}

template <typename scalar_t>
__global__ void candidateNetSubgraphTimingForwardKernel(
    const int64_t* net_flat_topo_sort,
    const int64_t* net_flat_topo_sort_start,
    const int64_t* pin_fa,
    const scalar_t* edge_resistance,
    const scalar_t* driver_arrival,
    const scalar_t* driver_slew,
    const int64_t* candidate_index_by_node,
    const scalar_t* candidate_bu,
    const scalar_t* buffer_delay,
    const scalar_t* buffer_output_slew,
    const scalar_t* lin,
    scalar_t* arrival_in,
    scalar_t* arrival_out,
    scalar_t* slew_in,
    scalar_t* slew_out,
    int32_t net_count) {
  const int32_t net = blockIdx.x * blockDim.x + threadIdx.x;
  if (net >= net_count) {
    return;
  }
  const int64_t begin = net_flat_topo_sort_start[net];
  const int64_t end = net_flat_topo_sort_start[net + 1];
  if (begin == end) {
    return;
  }
  const scalar_t log10_value = scalar_t(2.302585092994046);
  const int64_t root = net_flat_topo_sort[begin];
  arrival_in[root] = driver_arrival[net];
  arrival_out[root] = driver_arrival[net];
  slew_in[root] = driver_slew[net];
  slew_out[root] = driver_slew[net];
  for (int64_t pos = begin; pos < end; ++pos) {
    const int64_t node = net_flat_topo_sort[pos];
    const int64_t parent = pin_fa[node];
    if (parent >= 0) {
      const scalar_t wire_delta = edge_resistance[node] * lin[node];
      arrival_in[node] = arrival_out[parent] + wire_delta;
      const scalar_t slew_delta = log10_value * wire_delta;
      slew_in[node] = sqrt(
          slew_out[parent] * slew_out[parent] + slew_delta * slew_delta);
    }
    const int64_t candidate = candidate_index_by_node[node];
    const scalar_t bu = candidate >= 0 ? candidate_bu[candidate] : scalar_t(0);
    const scalar_t delay = candidate >= 0 ? buffer_delay[candidate] : scalar_t(0);
    const scalar_t output_slew =
        candidate >= 0 ? buffer_output_slew[candidate] : scalar_t(0);
    arrival_out[node] = arrival_in[node] + bu * delay;
    slew_out[node] =
        (scalar_t(1) - bu) * slew_in[node] + bu * output_slew;
  }
}

template <typename scalar_t>
__global__ void candidateNetSubgraphGatherSinkKernel(
    const int64_t* net_flat_topo_sort,
    const int64_t* net_flat_topo_sort_start,
    const int64_t* sink_node_id,
    const int64_t* sink_net_index,
    const scalar_t* effective_node_cap,
    const scalar_t* lin,
    const scalar_t* arrival_out,
    const scalar_t* slew_out,
    scalar_t* sink_arrival,
    scalar_t* sink_slew,
    scalar_t* sink_load,
    scalar_t* sink_cap,
    scalar_t* sink_net_delay,
    scalar_t* sink_net_impulse,
    int32_t sink_count) {
  const int32_t sink = blockIdx.x * blockDim.x + threadIdx.x;
  if (sink >= sink_count) {
    return;
  }
  const int64_t node = sink_node_id[sink];
  const int64_t net = sink_net_index[sink];
  const int64_t root = net_flat_topo_sort[net_flat_topo_sort_start[net]];
  sink_arrival[sink] = arrival_out[node];
  sink_slew[sink] = slew_out[node];
  sink_load[sink] = lin[node];
  sink_cap[sink] = effective_node_cap[node];
  sink_net_delay[sink] = arrival_out[node] - arrival_out[root];
  sink_net_impulse[sink] =
      slew_out[node] * slew_out[node] - slew_out[root] * slew_out[root];
}

template <typename scalar_t>
__global__ void candidateNetSubgraphSeedSinkGradKernel(
    const int64_t* net_flat_topo_sort,
    const int64_t* net_flat_topo_sort_start,
    const int64_t* sink_node_id,
    const int64_t* sink_net_index,
    const scalar_t* slew_out,
    const scalar_t* grad_sink_arrival,
    const scalar_t* grad_sink_slew,
    const scalar_t* grad_sink_load,
    const scalar_t* grad_sink_net_delay,
    const scalar_t* grad_sink_net_impulse,
    scalar_t* grad_lin,
    scalar_t* grad_arrival_out,
    scalar_t* grad_slew_out,
    int32_t sink_count) {
  const int32_t sink = blockIdx.x * blockDim.x + threadIdx.x;
  if (sink >= sink_count) {
    return;
  }
  const int64_t node = sink_node_id[sink];
  const int64_t net = sink_net_index[sink];
  const int64_t root = net_flat_topo_sort[net_flat_topo_sort_start[net]];
  atomicAdd(grad_arrival_out + node, grad_sink_arrival[sink]);
  atomicAdd(grad_slew_out + node, grad_sink_slew[sink]);
  atomicAdd(grad_lin + node, grad_sink_load[sink]);
  const scalar_t delay_grad = grad_sink_net_delay[sink];
  atomicAdd(grad_arrival_out + node, delay_grad);
  atomicAdd(grad_arrival_out + root, -delay_grad);
  const scalar_t impulse_grad = grad_sink_net_impulse[sink];
  atomicAdd(grad_slew_out + node, scalar_t(2) * slew_out[node] * impulse_grad);
  atomicAdd(grad_slew_out + root, -scalar_t(2) * slew_out[root] * impulse_grad);
}

template <typename scalar_t>
__global__ void candidateNetSubgraphBackwardKernel(
    const int64_t* net_flat_topo_sort,
    const int64_t* net_flat_topo_sort_start,
    const int64_t* pin_fa,
    const int64_t* flat_pin_to_start,
    const int64_t* flat_pin_to,
    const scalar_t* edge_resistance,
    const int64_t* candidate_index_by_node,
    const scalar_t* candidate_bu,
    const scalar_t* buffer_input_cap,
    const scalar_t* buffer_delay,
    const scalar_t* buffer_output_slew,
    const scalar_t* lin,
    const scalar_t* lout,
    const scalar_t* slew_in,
    const scalar_t* slew_out,
    scalar_t* grad_lout,
    scalar_t* grad_lin,
    scalar_t* grad_arrival_in,
    scalar_t* grad_arrival_out,
    scalar_t* grad_slew_in,
    scalar_t* grad_slew_out,
    scalar_t* grad_driver_arrival,
    scalar_t* grad_driver_slew,
    scalar_t* grad_candidate_bu,
    scalar_t* grad_buffer_input_cap,
    scalar_t* grad_buffer_delay,
    scalar_t* grad_buffer_output_slew,
    int32_t net_count) {
  const int32_t net = blockIdx.x * blockDim.x + threadIdx.x;
  if (net >= net_count) {
    return;
  }
  const int64_t begin = net_flat_topo_sort_start[net];
  const int64_t end = net_flat_topo_sort_start[net + 1];
  if (begin == end) {
    return;
  }
  const scalar_t log10_value = scalar_t(2.302585092994046);
  const scalar_t log10_squared = log10_value * log10_value;

  for (int64_t pos = end; pos-- > begin;) {
    const int64_t node = net_flat_topo_sort[pos];
    const int64_t candidate = candidate_index_by_node[node];
    const scalar_t bu = candidate >= 0 ? candidate_bu[candidate] : scalar_t(0);
    const scalar_t delay = candidate >= 0 ? buffer_delay[candidate] : scalar_t(0);
    const scalar_t output_slew =
        candidate >= 0 ? buffer_output_slew[candidate] : scalar_t(0);

    grad_arrival_in[node] += grad_arrival_out[node];
    if (candidate >= 0) {
      grad_candidate_bu[candidate] += grad_arrival_out[node] * delay;
      grad_buffer_delay[candidate] += grad_arrival_out[node] * bu;
    }
    grad_slew_in[node] += grad_slew_out[node] * (scalar_t(1) - bu);
    if (candidate >= 0) {
      grad_candidate_bu[candidate] +=
          grad_slew_out[node] * (output_slew - slew_in[node]);
      grad_buffer_output_slew[candidate] += grad_slew_out[node] * bu;
    }

    const int64_t parent = pin_fa[node];
    if (parent >= 0) {
      const scalar_t resistance = edge_resistance[node];
      const scalar_t wire_delta = resistance * lin[node];
      grad_arrival_out[parent] += grad_arrival_in[node];
      grad_lin[node] += grad_arrival_in[node] * resistance;
      const scalar_t slew_value = slew_in[node];
      if (fabs(slew_value) > scalar_t(0)) {
        grad_slew_out[parent] +=
            grad_slew_in[node] * slew_out[parent] / slew_value;
        const scalar_t grad_wire_delta =
            grad_slew_in[node] * log10_squared * wire_delta / slew_value;
        grad_lin[node] += grad_wire_delta * resistance;
      }
    } else {
      grad_driver_arrival[net] += grad_arrival_in[node];
      grad_driver_slew[net] += grad_slew_in[node];
    }
  }

  for (int64_t pos = begin; pos < end; ++pos) {
    const int64_t node = net_flat_topo_sort[pos];
    const int64_t candidate = candidate_index_by_node[node];
    const scalar_t bu = candidate >= 0 ? candidate_bu[candidate] : scalar_t(0);
    if (candidate >= 0) {
      grad_candidate_bu[candidate] +=
          grad_lin[node] * (buffer_input_cap[candidate] - lout[node]);
      grad_buffer_input_cap[candidate] += grad_lin[node] * bu;
      grad_lout[node] += grad_lin[node] * (scalar_t(1) - bu);
    } else {
      grad_lout[node] += grad_lin[node];
    }
    const scalar_t children_grad = grad_lout[node];
    for (int64_t edge = flat_pin_to_start[node];
         edge < flat_pin_to_start[node + 1];
         ++edge) {
      grad_lin[flat_pin_to[edge]] += children_grad;
    }
  }
}

template <typename scalar_t>
void candidateNetSubgraphForwardCudaLauncher(
    const int64_t* net_flat_topo_sort,
    const int64_t* net_flat_topo_sort_start,
    const int64_t* pin_fa,
    const int64_t* flat_pin_to_start,
    const int64_t* flat_pin_to,
    const scalar_t* edge_resistance,
    const scalar_t* node_capacitance,
    const scalar_t* edge_capacitance,
    const scalar_t* driver_arrival,
    const scalar_t* driver_slew,
    const int64_t* candidate_node_id,
    const scalar_t* candidate_bu,
    const scalar_t* buffer_input_cap,
    const scalar_t* buffer_delay,
    const scalar_t* buffer_output_slew,
    const int64_t* sink_node_id,
    const int64_t* sink_net_index,
    int64_t* candidate_index_by_node,
    scalar_t* effective_node_cap,
    scalar_t* lout,
    scalar_t* lin,
    scalar_t* arrival_in,
    scalar_t* arrival_out,
    scalar_t* slew_in,
    scalar_t* slew_out,
    scalar_t* sink_arrival,
    scalar_t* sink_slew,
    scalar_t* sink_load,
    scalar_t* sink_cap,
    scalar_t* sink_net_delay,
    scalar_t* sink_net_impulse,
    int32_t net_count,
    int32_t node_count,
    int32_t candidate_count,
    int32_t sink_count,
    cudaStream_t stream) {
  const int32_t threads = 128;
  const int32_t candidate_blocks = (candidate_count + threads - 1) / threads;
  if (candidate_blocks > 0) {
    candidateIndexByNodeKernel<scalar_t><<<candidate_blocks, threads, 0, stream>>>(
        candidate_node_id,
        candidate_index_by_node,
        candidate_count,
        node_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const int32_t net_blocks = (net_count + threads - 1) / threads;
  if (net_blocks > 0) {
    candidateNetSubgraphForwardKernel<scalar_t><<<net_blocks, threads, 0, stream>>>(
        net_flat_topo_sort,
        net_flat_topo_sort_start,
        pin_fa,
        flat_pin_to_start,
        flat_pin_to,
        edge_resistance,
        node_capacitance,
        edge_capacitance,
        driver_arrival,
        driver_slew,
        candidate_index_by_node,
        candidate_bu,
        buffer_input_cap,
        buffer_delay,
        buffer_output_slew,
        effective_node_cap,
        lout,
        lin,
        arrival_in,
        arrival_out,
        slew_in,
        slew_out,
        net_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const int32_t sink_blocks = (sink_count + threads - 1) / threads;
  if (sink_blocks > 0) {
    candidateNetSubgraphGatherSinkKernel<scalar_t><<<sink_blocks, threads, 0, stream>>>(
        net_flat_topo_sort,
        net_flat_topo_sort_start,
        sink_node_id,
        sink_net_index,
        effective_node_cap,
        lin,
        arrival_out,
        slew_out,
        sink_arrival,
        sink_slew,
        sink_load,
        sink_cap,
        sink_net_delay,
        sink_net_impulse,
        sink_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <typename scalar_t>
void candidateNetSubgraphFixedBsuForwardCudaLauncher(
    const int64_t* net_flat_topo_sort,
    const int64_t* net_flat_topo_sort_start,
    const int64_t* pin_fa,
    const int64_t* flat_pin_to_start,
    const int64_t* flat_pin_to,
    const scalar_t* edge_resistance,
    const scalar_t* node_capacitance,
    const scalar_t* edge_capacitance,
    const scalar_t* driver_arrival,
    const scalar_t* driver_slew,
    const int64_t* candidate_node_id,
    const scalar_t* candidate_bu,
    const scalar_t* buffer_input_cap,
    const scalar_t* probe_buffer_delay,
    const scalar_t* probe_buffer_output_slew,
    const scalar_t* upstream_retained_cap,
    const scalar_t* buffer_slew_axis,
    const scalar_t* buffer_load_axis,
    const scalar_t* buffer_delay_lut,
    const scalar_t* buffer_output_slew_lut,
    int32_t fixed_bsu_index,
    int32_t size_count,
    int32_t slew_count,
    int32_t load_count,
    const int64_t* sink_node_id,
    const int64_t* sink_net_index,
    int64_t* candidate_index_by_node,
    scalar_t* effective_node_cap,
    scalar_t* lout,
    scalar_t* lin,
    scalar_t* probe_arrival_in,
    scalar_t* probe_arrival_out,
    scalar_t* probe_slew_in,
    scalar_t* probe_slew_out,
    scalar_t* candidate_input_slew,
    scalar_t* candidate_output_load,
    scalar_t* buffer_delay,
    scalar_t* buffer_output_slew,
    scalar_t* buffer_delay_grad_slew,
    scalar_t* buffer_delay_grad_load,
    scalar_t* buffer_output_slew_grad_slew,
    scalar_t* buffer_output_slew_grad_load,
    scalar_t* arrival_in,
    scalar_t* arrival_out,
    scalar_t* slew_in,
    scalar_t* slew_out,
    scalar_t* sink_arrival,
    scalar_t* sink_slew,
    scalar_t* sink_load,
    scalar_t* sink_cap,
    scalar_t* sink_net_delay,
    scalar_t* sink_net_impulse,
    int32_t net_count,
    int32_t node_count,
    int32_t candidate_count,
    int32_t sink_count,
    cudaStream_t stream) {
  const int32_t threads = 128;
  const int32_t candidate_blocks = (candidate_count + threads - 1) / threads;
  if (candidate_blocks > 0) {
    candidateIndexByNodeKernel<scalar_t><<<candidate_blocks, threads, 0, stream>>>(
        candidate_node_id,
        candidate_index_by_node,
        candidate_count,
        node_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const int32_t net_blocks = (net_count + threads - 1) / threads;
  if (net_blocks > 0) {
    candidateNetSubgraphFixedBsuForwardKernel<scalar_t>
        <<<net_blocks, threads, 0, stream>>>(
            net_flat_topo_sort,
            net_flat_topo_sort_start,
            pin_fa,
            flat_pin_to_start,
            flat_pin_to,
            edge_resistance,
            node_capacitance,
            edge_capacitance,
            driver_arrival,
            driver_slew,
            candidate_index_by_node,
            candidate_bu,
            buffer_input_cap,
            probe_buffer_delay,
            probe_buffer_output_slew,
            effective_node_cap,
            lout,
            lin,
            probe_arrival_in,
            probe_arrival_out,
            probe_slew_in,
            probe_slew_out,
            net_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (candidate_blocks > 0) {
    candidateFixedBsuLookupKernel<scalar_t>
        <<<candidate_blocks, threads, 0, stream>>>(
            candidate_node_id,
            lout,
            probe_slew_in,
            upstream_retained_cap,
            buffer_slew_axis,
            buffer_load_axis,
            buffer_delay_lut,
            buffer_output_slew_lut,
            fixed_bsu_index,
            size_count,
            slew_count,
            load_count,
            candidate_input_slew,
            candidate_output_load,
            buffer_delay,
            buffer_output_slew,
            buffer_delay_grad_slew,
            buffer_delay_grad_load,
            buffer_output_slew_grad_slew,
            buffer_output_slew_grad_load,
            candidate_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  if (net_blocks > 0) {
    candidateNetSubgraphTimingForwardKernel<scalar_t>
        <<<net_blocks, threads, 0, stream>>>(
            net_flat_topo_sort,
            net_flat_topo_sort_start,
            pin_fa,
            edge_resistance,
            driver_arrival,
            driver_slew,
            candidate_index_by_node,
            candidate_bu,
            buffer_delay,
            buffer_output_slew,
            lin,
            arrival_in,
            arrival_out,
            slew_in,
            slew_out,
            net_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const int32_t sink_blocks = (sink_count + threads - 1) / threads;
  if (sink_blocks > 0) {
    candidateNetSubgraphGatherSinkKernel<scalar_t><<<sink_blocks, threads, 0, stream>>>(
        net_flat_topo_sort,
        net_flat_topo_sort_start,
        sink_node_id,
        sink_net_index,
        effective_node_cap,
        lin,
        arrival_out,
        slew_out,
        sink_arrival,
        sink_slew,
        sink_load,
        sink_cap,
        sink_net_delay,
        sink_net_impulse,
        sink_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

template <typename scalar_t>
void candidateNetSubgraphBackwardCudaLauncher(
    const int64_t* net_flat_topo_sort,
    const int64_t* net_flat_topo_sort_start,
    const int64_t* pin_fa,
    const int64_t* flat_pin_to_start,
    const int64_t* flat_pin_to,
    const scalar_t* edge_resistance,
    const int64_t* candidate_index_by_node,
    const scalar_t* candidate_bu,
    const scalar_t* buffer_input_cap,
    const scalar_t* buffer_delay,
    const scalar_t* buffer_output_slew,
    const int64_t* sink_node_id,
    const int64_t* sink_net_index,
    const scalar_t* lin,
    const scalar_t* lout,
    const scalar_t* slew_in,
    const scalar_t* slew_out,
    const scalar_t* grad_sink_arrival,
    const scalar_t* grad_sink_slew,
    const scalar_t* grad_sink_load,
    const scalar_t* grad_sink_net_delay,
    const scalar_t* grad_sink_net_impulse,
    scalar_t* grad_lout,
    scalar_t* grad_lin,
    scalar_t* grad_arrival_in,
    scalar_t* grad_arrival_out,
    scalar_t* grad_slew_in,
    scalar_t* grad_slew_out,
    scalar_t* grad_driver_arrival,
    scalar_t* grad_driver_slew,
    scalar_t* grad_candidate_bu,
    scalar_t* grad_buffer_input_cap,
    scalar_t* grad_buffer_delay,
    scalar_t* grad_buffer_output_slew,
    int32_t net_count,
    int32_t sink_count,
    cudaStream_t stream) {
  const int32_t threads = 128;
  const int32_t sink_blocks = (sink_count + threads - 1) / threads;
  if (sink_blocks > 0) {
    candidateNetSubgraphSeedSinkGradKernel<scalar_t><<<sink_blocks, threads, 0, stream>>>(
        net_flat_topo_sort,
        net_flat_topo_sort_start,
        sink_node_id,
        sink_net_index,
        slew_out,
        grad_sink_arrival,
        grad_sink_slew,
        grad_sink_load,
        grad_sink_net_delay,
        grad_sink_net_impulse,
        grad_lin,
        grad_arrival_out,
        grad_slew_out,
        sink_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
  const int32_t net_blocks = (net_count + threads - 1) / threads;
  if (net_blocks > 0) {
    candidateNetSubgraphBackwardKernel<scalar_t><<<net_blocks, threads, 0, stream>>>(
        net_flat_topo_sort,
        net_flat_topo_sort_start,
        pin_fa,
        flat_pin_to_start,
        flat_pin_to,
        edge_resistance,
        candidate_index_by_node,
        candidate_bu,
        buffer_input_cap,
        buffer_delay,
        buffer_output_slew,
        lin,
        lout,
        slew_in,
        slew_out,
        grad_lout,
        grad_lin,
        grad_arrival_in,
        grad_arrival_out,
        grad_slew_in,
        grad_slew_out,
        grad_driver_arrival,
        grad_driver_slew,
        grad_candidate_bu,
        grad_buffer_input_cap,
        grad_buffer_delay,
        grad_buffer_output_slew,
        net_count);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
  }
}

#define INSTANTIATE_CANDIDATE_NET_SUBGRAPH(scalar_t)                                  \
  template void candidateNetSubgraphForwardCudaLauncher<scalar_t>(                   \
      const int64_t*, const int64_t*, const int64_t*, const int64_t*, const int64_t*,\
      const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*,             \
      const scalar_t*, const int64_t*, const scalar_t*, const scalar_t*,              \
      const scalar_t*, const scalar_t*, const int64_t*, const int64_t*, int64_t*,     \
      scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*,    \
      scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, int32_t,      \
      int32_t, int32_t, int32_t, cudaStream_t);                                       \
  template void candidateNetSubgraphBackwardCudaLauncher<scalar_t>(                  \
      const int64_t*, const int64_t*, const int64_t*, const int64_t*, const int64_t*,\
      const scalar_t*, const int64_t*, const scalar_t*, const scalar_t*,              \
      const scalar_t*, const scalar_t*, const int64_t*, const int64_t*,               \
      const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*,             \
      const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*,             \
      const scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*,         \
      scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*,    \
      int32_t, int32_t, cudaStream_t)

#define INSTANTIATE_CANDIDATE_FIXED_BSU_FORWARD(scalar_t)                             \
  template void candidateNetSubgraphFixedBsuForwardCudaLauncher<scalar_t>(            \
      const int64_t*, const int64_t*, const int64_t*, const int64_t*, const int64_t*,\
      const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*,             \
      const scalar_t*, const int64_t*, const scalar_t*, const scalar_t*,              \
      const scalar_t*, const scalar_t*, const scalar_t*, const scalar_t*,             \
      const scalar_t*, const scalar_t*, const scalar_t*, int32_t, int32_t, int32_t,   \
      int32_t, const int64_t*, const int64_t*, int64_t*, scalar_t*, scalar_t*,        \
      scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*,    \
      scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*,    \
      scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*, scalar_t*,    \
      scalar_t*, scalar_t*, int32_t, int32_t, int32_t, int32_t,                       \
      cudaStream_t)

INSTANTIATE_CANDIDATE_NET_SUBGRAPH(float);
INSTANTIATE_CANDIDATE_NET_SUBGRAPH(double);
INSTANTIATE_CANDIDATE_FIXED_BSU_FORWARD(float);
INSTANTIATE_CANDIDATE_FIXED_BSU_FORWARD(double);

#undef INSTANTIATE_CANDIDATE_NET_SUBGRAPH
#undef INSTANTIATE_CANDIDATE_FIXED_BSU_FORWARD
