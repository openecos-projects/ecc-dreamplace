// SPDX-License-Identifier: BSD-3-Clause
// Original slots remain fixed; live buffer records map appended/compacted slots.
#include "rebuffer_timing_planning.hh"
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_mutation.hh"
#include "rebuffer_timing_buffer_state.hh"

#include <algorithm>
#include <numeric>
#include <unordered_map>

namespace py = pybind11;
namespace rebuffer {

void NativeTimingPlanningSnapshot::buildIdentity()
{
  const auto& m = *graph_->mutation_state_;
  const auto& d = projection_.design;
  const auto& base = inputs_->carrier;
  const int original_pins = inputs_->design.num_pins;
  const int original_nets = inputs_->design.num_nets;
  const int movable = m.original_node_count - base.num_terminals - base.num_terminal_nis;
  if (movable < 0 || d.num_nodes != m.original_node_count + projection_.buffers.size()
      || d.num_pins != original_pins + 2 * projection_.buffers.size()) {
    throw py::value_error("native planning original and inserted slot counts differ");
  }
  graph_pin_by_pin_.resize(d.num_pins);
  std::iota(graph_pin_by_pin_.begin(), graph_pin_by_pin_.begin() + original_pins, 0);
  graph_node_by_node_.assign(d.num_nodes, -1);
  pin_by_graph_.assign(graph_->timing_index_->pins.size(), -1);
  net_by_graph_.assign(m.net_names.size(), -1);
  for (int node = 0; node < m.original_node_count; ++node) {
    const int target = node < movable ? node : node + projection_.buffers.size();
    graph_node_by_node_.at(target) = node;
  }
  std::iota(net_by_graph_.begin(), net_by_graph_.begin() + original_nets, 0);
  std::unordered_map<int, const TimingBufferRecord*> buffers;
  for (const auto& buffer : m.buffers) {
    if (buffer.active && !buffers.emplace(buffer.action_id, &buffer).second) {
      throw py::value_error("native planning has duplicate active buffer identities");
    }
  }
  for (const auto& projected : projection_.buffers) {
    const auto found = buffers.find(projected.action_id);
    if (found == buffers.end()) {
      throw py::value_error("native planning buffer has no active graph slots");
    }
    const auto& buffer = *found->second;
    graph_pin_by_pin_.at(projected.input_pin) = buffer.input_pin;
    graph_pin_by_pin_.at(projected.output_pin) = buffer.output_pin;
    graph_node_by_node_.at(projected.node) = buffer.node;
    net_by_graph_.at(buffer.output_net) = projected.output_net;
  }
  for (int pin = 0; pin < d.num_pins; ++pin) {
    const int source = graph_pin_by_pin_[pin];
    if (!m.pin_active.at(source) || pin_by_graph_.at(source) >= 0
        || graph_->timing_index_->pins.at(source) != d.pin_names[pin]) {
      throw py::value_error("native planning pin slots do not match physical identities");
    }
    pin_by_graph_[source] = pin;
  }
  if (d.num_pins != std::count(m.pin_active.begin(), m.pin_active.end(), 1)) {
    throw py::value_error("native planning is missing active graph pins");
  }
  for (int node = 0; node < d.num_nodes; ++node) {
    if (m.node_names.at(graph_node_by_node_[node]) != projection_.node_names[node]) {
      throw py::value_error("native planning node slots do not match physical identities");
    }
  }
  for (std::size_t net = 0; net < m.net_names.size(); ++net) {
    const int target = net_by_graph_[net];
    if (target < 0) {
      if (net < m.base_net_pins.size() || m.net_drivers[net] >= 0) {
        throw py::value_error("native planning is missing an active graph net");
      }
    } else if (m.net_names[net] != projection_.net_names.at(target)) {
      throw py::value_error("native planning net slots do not match physical identities");
    }
  }
}

}  // namespace rebuffer
