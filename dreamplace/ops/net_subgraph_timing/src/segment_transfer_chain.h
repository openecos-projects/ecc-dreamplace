#pragma once

#include <limits>

// Included after the Liberty interpolation primitives in net_subgraph_timing.cpp.

template <typename scalar_t>
struct TransferStateGrad {
  scalar_t delay;
  scalar_t output_slew;
  scalar_t grad_input_slew;
  scalar_t grad_downstream_load;
  scalar_t grad_bsu;
  scalar_t grad_resistance;
  scalar_t grad_capacitance;
  std::vector<scalar_t> buffer_slew_violation;
  std::vector<scalar_t> buffer_cap_violation;
};

template <typename scalar_t>
TransferStateGrad<scalar_t> transfer_state_value_and_grad(
    const scalar_t parent_slew,
    const scalar_t downstream_load,
    const scalar_t resistance,
    const scalar_t capacitance,
    const int64_t count,
    const scalar_t* fractions,
    const scalar_t bsu,
    const scalar_t* buffer_input_cap_by_size,
    const int64_t size_count,
    const scalar_t* slew_axis,
    const int64_t slew_count,
    const scalar_t* load_axis,
    const int64_t load_count,
    const scalar_t* delay_lut,
    const scalar_t* output_slew_lut,
    const scalar_t grad_delay,
    const scalar_t grad_output_slew,
    const scalar_t slew_limit = std::numeric_limits<scalar_t>::infinity(),
    const scalar_t cap_limit = std::numeric_limits<scalar_t>::infinity(),
    const scalar_t* grad_slew_violation = nullptr,
    const scalar_t* grad_cap_violation = nullptr,
    const scalar_t violation_weight = static_cast<scalar_t>(1)) {
  const scalar_t log10 = static_cast<scalar_t>(std::log(10.0));
  const scalar_t epsilon = static_cast<scalar_t>(1e-30);
  auto wire_slew = [&](const scalar_t slew, const scalar_t delay) {
    const scalar_t delta = log10 * delay;
    return std::sqrt(slew * slew + delta * delta);
  };
  auto backprop_wire = [&](const scalar_t in_slew,
                           const scalar_t wire_delay,
                           const scalar_t out_slew,
                           const scalar_t grad_out_slew,
                           scalar_t& grad_in_slew,
                           scalar_t& grad_wire_delay) {
    const scalar_t denom = std::max(out_slew, epsilon);
    grad_in_slew += grad_out_slew * in_slew / denom;
    grad_wire_delay +=
        grad_out_slew * (log10 * log10) * wire_delay / denom;
  };

  if (count == 0) {
    const scalar_t delay =
        resistance * (downstream_load + static_cast<scalar_t>(0.5) * capacitance);
    const scalar_t out_slew = wire_slew(parent_slew, delay);
    scalar_t grad_parent_slew = static_cast<scalar_t>(0);
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
        static_cast<scalar_t>(0),
        grad_wire_delay *
            (downstream_load + static_cast<scalar_t>(0.5) * capacitance),
        grad_wire_delay * static_cast<scalar_t>(0.5) * resistance,
    };
  }

  const scalar_t buffer_input_cap =
      interp_size_1d(buffer_input_cap_by_size, size_count, bsu);
  const scalar_t buffer_input_cap_grad =
      interp_size_1d_slope(buffer_input_cap_by_size, size_count, bsu);
  std::vector<scalar_t> wire_delays(static_cast<size_t>(count + 1));
  std::vector<scalar_t> wire_input_slews(static_cast<size_t>(count + 1));
  std::vector<scalar_t> wire_output_slews(static_cast<size_t>(count + 1));
  std::vector<scalar_t> buffer_input_slews(static_cast<size_t>(count));
  std::vector<scalar_t> buffer_output_loads(static_cast<size_t>(count));
  std::vector<scalar_t> buffer_delays(static_cast<size_t>(count));
  std::vector<scalar_t> buffer_output_slews(static_cast<size_t>(count));

  scalar_t total_delay = static_cast<scalar_t>(0);
  std::vector<scalar_t> slew_violation(static_cast<size_t>(count));
  std::vector<scalar_t> cap_violation(static_cast<size_t>(count));
  scalar_t slew = parent_slew;
  for (int64_t step = 0; step < count; ++step) {
    const scalar_t fraction = fractions[step];
    const scalar_t step_resistance = resistance * fraction;
    const scalar_t step_capacitance = capacitance * fraction;
    const scalar_t wire_delay =
        step_resistance *
        (buffer_input_cap + static_cast<scalar_t>(0.5) * step_capacitance);
    wire_delays[static_cast<size_t>(step)] = wire_delay;
    wire_input_slews[static_cast<size_t>(step)] = slew;
    slew = wire_slew(slew, wire_delay);
    wire_output_slews[static_cast<size_t>(step)] = slew;
    buffer_input_slews[static_cast<size_t>(step)] = slew;

    const bool has_next_buffer = count > (step + 1);
    const scalar_t next_fraction = fractions[step + 1];
    const scalar_t next_capacitance = capacitance * next_fraction;
    const scalar_t output_load =
        (has_next_buffer ? buffer_input_cap : downstream_load) + next_capacitance;
    buffer_output_loads[static_cast<size_t>(step)] = output_load;
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
    buffer_delays[static_cast<size_t>(step)] = buffer_delay;
    buffer_output_slews[static_cast<size_t>(step)] = buffer_slew;
    slew_violation[static_cast<size_t>(step)] =
        std::max(static_cast<scalar_t>(0), buffer_slew - slew_limit);
    cap_violation[static_cast<size_t>(step)] =
        std::max(static_cast<scalar_t>(0), output_load - cap_limit);
    total_delay += wire_delay + buffer_delay;
    slew = buffer_slew;
  }
  const scalar_t final_fraction = fractions[count];
  const scalar_t final_resistance = resistance * final_fraction;
  const scalar_t final_capacitance = capacitance * final_fraction;
  const scalar_t downstream_wire_delay =
      final_resistance *
      (downstream_load + static_cast<scalar_t>(0.5) * final_capacitance);
  wire_delays[static_cast<size_t>(count)] = downstream_wire_delay;
  wire_input_slews[static_cast<size_t>(count)] = slew;
  const scalar_t output_slew = wire_slew(slew, downstream_wire_delay);
  wire_output_slews[static_cast<size_t>(count)] = output_slew;
  const scalar_t delay = total_delay + downstream_wire_delay;
  if (grad_delay == static_cast<scalar_t>(0) &&
      grad_output_slew == static_cast<scalar_t>(0) &&
      grad_slew_violation == nullptr && grad_cap_violation == nullptr) {
    return {delay, output_slew, 0, 0, 0, 0, 0, slew_violation, cap_violation};
  }

  scalar_t grad_slew_state = static_cast<scalar_t>(0);
  scalar_t grad_downstream_load = static_cast<scalar_t>(0);
  scalar_t grad_buffer_input_cap = static_cast<scalar_t>(0);
  scalar_t grad_bsu = static_cast<scalar_t>(0);
  scalar_t grad_resistance = static_cast<scalar_t>(0);
  scalar_t grad_capacitance = static_cast<scalar_t>(0);
  scalar_t grad_final_wire_delay = grad_delay;
  backprop_wire(
      wire_input_slews[static_cast<size_t>(count)],
      downstream_wire_delay,
      output_slew,
      grad_output_slew,
      grad_slew_state,
      grad_final_wire_delay);
  grad_downstream_load += grad_final_wire_delay * final_resistance;
  grad_resistance += grad_final_wire_delay * final_fraction *
      (downstream_load + static_cast<scalar_t>(0.5) * capacitance * final_fraction);
  grad_capacitance += grad_final_wire_delay * static_cast<scalar_t>(0.5) *
      resistance * final_fraction * final_fraction;

  for (int64_t step = count - 1; step >= 0; --step) {
    const bool has_next_buffer = count > (step + 1);
    const auto delay_lookup = lookup_lut_3d_with_grad(
        delay_lut,
        size_count,
        slew_count,
        load_count,
        slew_axis,
        load_axis,
        bsu,
        buffer_input_slews[static_cast<size_t>(step)],
        buffer_output_loads[static_cast<size_t>(step)]);
    const auto slew_lookup = lookup_lut_3d_with_grad(
        output_slew_lut,
        size_count,
        slew_count,
        load_count,
        slew_axis,
        load_axis,
        bsu,
        buffer_input_slews[static_cast<size_t>(step)],
        buffer_output_loads[static_cast<size_t>(step)]);

    const scalar_t grad_buffer_delay = grad_delay;
    const scalar_t grad_buffer_slew = grad_slew_state +
        (grad_slew_violation != nullptr &&
         buffer_output_slews[static_cast<size_t>(step)] > slew_limit ?
            grad_slew_violation[step] * violation_weight : static_cast<scalar_t>(0));
    scalar_t grad_buffer_input_slew =
        grad_buffer_delay * delay_lookup.grad_slew +
        grad_buffer_slew * slew_lookup.grad_slew;
    const scalar_t grad_output_load =
        grad_buffer_delay * delay_lookup.grad_load +
        grad_buffer_slew * slew_lookup.grad_load +
        (grad_cap_violation != nullptr &&
         buffer_output_loads[static_cast<size_t>(step)] > cap_limit ?
            grad_cap_violation[step] * violation_weight : static_cast<scalar_t>(0));
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
    scalar_t grad_previous_slew = static_cast<scalar_t>(0);
    backprop_wire(
        wire_input_slews[static_cast<size_t>(step)],
        wire_delays[static_cast<size_t>(step)],
        wire_output_slews[static_cast<size_t>(step)],
        grad_buffer_input_slew,
        grad_previous_slew,
        grad_wire_delay);
    grad_slew_state = grad_previous_slew;
    const scalar_t fraction = fractions[step];
    const scalar_t step_resistance = resistance * fraction;
    grad_buffer_input_cap += grad_wire_delay * step_resistance;
    grad_resistance += grad_wire_delay * fraction *
        (buffer_input_cap + static_cast<scalar_t>(0.5) * capacitance * fraction);
    grad_capacitance += grad_wire_delay * static_cast<scalar_t>(0.5) *
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
      slew_violation,
      cap_violation,
  };
}
