#pragma once

#include "timing_propagation/src/timing_math.h"

template <typename scalar_t>
TIMING_INLINE void candidateBackwardNet(int32_t net,
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
