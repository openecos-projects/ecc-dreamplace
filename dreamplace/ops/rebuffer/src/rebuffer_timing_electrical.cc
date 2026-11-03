// SPDX-License-Identifier: BSD-3-Clause
// Candidate acceptance reads the graph directly and shares the physical ERC.
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_mutation.hh"
#include "rebuffer_timing_buffer_state.hh"
#include "rebuffer_timing_events.hh"
#include "rebuffer_electrical_observation.hh"

#include <algorithm>

#include "pybind11/stl.h"

namespace py = pybind11;
namespace rebuffer {

py::dict PersistentTimingGraph::evaluationObservation(
    int max_fanout, double cap_margin, double slew_margin) const
{
  if (!mutation_state_ || !propagation_program_ || !timing_state_initialized_) {
    throw py::value_error("evaluation observation requires initialized native timing");
  }
  const auto seeds = timingEventSeeds();
  if (seeds.require_full || !seeds.forward.empty() || !seeds.reverse.empty()
      || !seeds.retired_arcs.empty()) {
    throw py::value_error("evaluation observation requires current native propagation");
  }
  const auto& m = *mutation_state_;
  for (const auto& [_, topology] : net_states_) {
    requireSynchronizedTimingTopology(m, *topology);
  }
  const auto& names = timing_index_->pins;
  const auto& rise = state_.at("pin_rtran");
  const auto& fall = state_.at("pin_ftran");
  const auto& rise_load = state_.at("pin_net_cap_rise");
  const auto& fall_load = state_.at("pin_net_cap_fall");
  std::vector<unsigned char> clock_net(m.net_names.size(), 0), clock_pin(names.size(), 0);
  std::copy(m.base_net_is_clock.begin(), m.base_net_is_clock.end(), clock_net.begin());
  for (const auto& buffer : m.buffers) {
    clock_net.at(buffer.output_net) = clock_net.at(buffer.source_net);
  }
  for (const auto pin : timing_index_->clocks) clock_pin.at(pin) = 1;
  std::vector<ElectricalPinObservation> pins;
  pins.reserve(names.size());
  for (std::size_t pin = 0; pin < names.size(); ++pin) {
    if (!m.pin_active[pin]) continue;
    const int net = m.pin_net[pin];
    const bool driver = net >= 0 && m.net_drivers[net] == static_cast<int>(pin);
    const int fanout = driver ? m.net_pins[net].size() - 1 : 0;
    pins.push_back({names[pin], rise[pin], fall[pin], rise_load[pin], fall_load[pin],
                    m.pin_values[3][pin], m.pin_values[4][pin], fanout, driver,
                    clock_pin[pin] != 0 || (net >= 0 && clock_net[net] != 0)});
  }
  py::dict endpoints;
  const auto slacks = endpointSlacks();
  const auto& p = *propagation_program_;
  for (std::size_t row = 0; row < p.metric_pins.size(); ++row) {
    endpoints[py::str(names.at(p.metric_pins[row]))] = slacks[row];
  }
  py::dict result;
  result["electrical"] = observeElectricalPins(pins, max_fanout, cap_margin, slew_margin);
  result["endpoint_slacks_by_name"] = std::move(endpoints);
  return result;
}

}  // namespace rebuffer
