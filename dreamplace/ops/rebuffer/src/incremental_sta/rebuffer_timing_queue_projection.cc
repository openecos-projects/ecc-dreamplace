// SPDX-License-Identifier: BSD-3-Clause
// Project graph-owned queue identities onto one current physical carrier.
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_mutation.hh"
#include "rebuffer_queue_snapshot.hh"

#include <algorithm>
#include <chrono>
#include <unordered_map>

#include "pybind11/stl.h"

namespace py = pybind11;
namespace rebuffer {
namespace {
std::unordered_map<std::string, int> nameMap(
    const std::vector<std::string>& names, const char* kind)
{
  std::unordered_map<std::string, int> result;
  result.reserve(names.size());
  for (std::size_t id = 0; id < names.size(); ++id) {
    if (!result.emplace(names[id], id).second) {
      throw py::value_error(std::string("queue carrier has duplicate ") + kind);
    }
  }
  return result;
}
}  // namespace

py::dict PersistentTimingGraph::timingQueueStateForCarrier(
    const std::vector<int>& original_net_by_net, const py::object& carrier,
    bool use_path_context) const
{
  if (!mutation_state_) {
    throw py::value_error("queue projection requires native mutation inputs");
  }
  const auto started = std::chrono::steady_clock::now();
  const auto& m = *mutation_state_;
  const auto pins = carrier.attr("pin_names").cast<std::vector<std::string>>();
  const auto nets = carrier.attr("net_names").cast<std::vector<std::string>>();
  const auto pin_net = carrier.attr("pin2net_map").cast<std::vector<int>>();
  const auto drivers = carrier.attr("net2driver_pin_map").cast<std::vector<int>>();
  const auto nodes = carrier.attr("node_names").cast<std::vector<std::string>>();
  const auto masters = carrier.attr("node_master_names").cast<std::vector<std::string>>();
  const auto pin_owner = carrier.attr("pin2node_map").cast<std::vector<int>>();
  if (pin_net.size() != pins.size() || pin_owner.size() != pins.size()
      || drivers.size() != nets.size() || masters.size() != nodes.size()) {
    throw py::value_error("queue carrier identity arrays do not align");
  }
  const auto pin_lookup = nameMap(pins, "pin names");
  const auto net_lookup = nameMap(nets, "net names");
  std::vector<int> pin_map(timing_index_->pins.size(), -1);
  std::vector<int> net_map(m.net_names.size(), -1);
  std::unordered_map<int, int> node_aliases;
  if (py::hasattr(carrier, "session_buffer_pin_ids")) {
    const auto aliases = carrier.attr("session_buffer_pin_ids")
        .cast<std::map<int, std::array<int, 2>>>();
    for (const auto& [action, target] : aliases) {
      const auto found = std::find_if(m.buffers.begin(), m.buffers.end(),
          [&](const auto& buffer) { return buffer.active && buffer.action_id == action; });
      if (found == m.buffers.end()
          || std::count_if(m.buffers.begin(), m.buffers.end(),
               [&](const auto& buffer) { return buffer.active && buffer.action_id == action; }) != 1
          || target[0] < 0 || target[1] < 0 || target[0] == target[1]
          || target[0] >= static_cast<int>(pins.size()) || target[1] >= static_cast<int>(pins.size())
          || pin_owner[target[0]] != pin_owner[target[1]]) {
        throw py::value_error("queue carrier has invalid session pin identities");
      }
      pin_map[found->input_pin] = target[0];
      pin_map[found->output_pin] = target[1];
      net_map[found->output_net] = pin_net[target[1]];
      node_aliases.emplace(found->node, pin_owner[target[0]]);
    }
  }
  std::vector<unsigned char> used_pins(pins.size(), 0), used_nets(nets.size(), 0);
  std::size_t live_pins = 0, live_nets = 0;
  std::size_t remapped_pins = 0, remapped_nets = 0;
  for (std::size_t id = 0; id < pin_map.size(); ++id) {
    if (!m.pin_active[id]) continue;
    ++live_pins;
    if (pin_map[id] < 0) {
      const auto found = pin_lookup.find(timing_index_->pins[id]);
      if (found == pin_lookup.end()) {
        throw py::value_error("queue carrier is missing an active timing pin");
      }
      pin_map[id] = found->second;
    }
    if (used_pins.at(pin_map[id])++) {
      throw py::value_error("queue carrier session pin identities overlap");
    }
    remapped_pins += pin_map[id] != static_cast<int>(id);
  }
  for (std::size_t id = 0; id < net_map.size(); ++id) {
    // Original dangling nets still exist in the physical carrier; retired
    // synthetic output nets have neither a driver nor any active pins.
    if (id >= m.base_net_pins.size() && m.net_drivers[id] < 0) continue;
    ++live_nets;
    if (net_map[id] < 0) {
      const auto found = net_lookup.find(m.net_names[id]);
      if (found == net_lookup.end()) {
        throw py::value_error("queue carrier is missing an active timing net");
      }
      net_map[id] = found->second;
    }
    if (net_map[id] >= static_cast<int>(nets.size()) || used_nets.at(net_map[id])++) {
      throw py::value_error("queue carrier session net identities overlap");
    }
    remapped_nets += net_map[id] != static_cast<int>(id);
  }
  if (live_pins != pins.size() || live_nets != nets.size()) {
    throw py::value_error("queue carrier has different active timing identities");
  }
  for (std::size_t pin = 0; pin < pin_map.size(); ++pin) {
    if (pin_map[pin] < 0) continue;
    const int target = pin_map[pin], source_net = m.pin_net[pin];
    if ((source_net < 0 ? -1 : net_map.at(source_net)) != pin_net[target]) {
      throw py::value_error("queue carrier has different pin net partitions");
    }
    const int source_node = m.pin_owner[pin], target_node = pin_owner[target];
    const auto alias = node_aliases.find(source_node);
    if (source_node < 0 || target_node < 0 || target_node >= static_cast<int>(nodes.size())
        || (alias != node_aliases.end() ? alias->second != target_node
                                      : nodes[target_node] != m.node_names.at(source_node))) {
      throw py::value_error("queue carrier has different pin owners");
    }
    const int cell = m.node_cells[source_node];
    if (cell >= 0 && masters[target_node] != m.cell_names.at(cell)) {
      throw py::value_error("queue carrier has different instance masters");
    }
  }
  for (std::size_t net = 0; net < net_map.size(); ++net) {
    if (net_map[net] < 0) continue;
    const int driver = m.net_drivers[net];
    if ((driver < 0 ? -1 : pin_map.at(driver)) != drivers[net_map[net]]) {
      throw py::value_error("queue carrier has different net drivers");
    }
  }
  const double projection_ms = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  auto result = timingQueueState(original_net_by_net, use_path_context);
  auto snapshot = result["_native_queue_state"]
      .cast<std::shared_ptr<NativeTimingQueueSnapshot>>();
  projectTimingQueueItems(snapshot->items, pin_map, net_map);
  auto statistics = result["graph_queue_statistics"].cast<py::dict>();
  statistics["projection_ms"] = projection_ms;
  statistics["remapped_pin_count"] = remapped_pins;
  statistics["remapped_net_count"] = remapped_nets;
  result["timing_source"] = "persistent_graph_in_carrier_ids";
  return result;
}
}  // namespace rebuffer
