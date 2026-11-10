// SPDX-License-Identifier: BSD-3-Clause
#include "rebuffer_topology_partitions.hh"

#include <unordered_set>

namespace rebuffer {

ElectricalPartitions topologyPartitions(const IncrementalTopologyState& state)
{
  ElectricalPartitions result;
  result.loads.resize(state.buffers.size() + 1);
  std::unordered_map<int, std::vector<int>> children;
  std::unordered_map<int, int> output_slot;
  std::unordered_set<int> expected{state.driver_pin_id};
  for (const auto& [node, _] : state.coordinates) expected.insert(node);
  for (const auto& edge : state.edges) {
    children[edge.parent].push_back(edge.child);
    expected.insert(edge.parent);
    expected.insert(edge.child);
  }
  for (int slot = 0; slot < static_cast<int>(state.buffers.size()); ++slot) {
    if (!output_slot.emplace(state.buffers[slot].output_node, slot).second) {
      throw pybind11::value_error("topology has duplicate buffer output nodes");
    }
  }
  std::vector<std::pair<int, int>> pending{{state.driver_pin_id, 0}};
  while (!pending.empty()) {
    const auto [node, incoming] = pending.back();
    pending.pop_back();
    const auto output = output_slot.find(node);
    const int partition = output == output_slot.end() ? incoming : output->second + 1;
    const auto [found, inserted] = result.node_partition.emplace(node, partition);
    if (!inserted) {
      if (found->second != partition) {
        throw pybind11::value_error("topology node has multiple electrical partitions");
      }
      continue;
    }
    for (int child : children[node]) pending.emplace_back(child, partition);
  }
  for (const auto& [node, loads] : state.node_loads) {
    expected.insert(node);
    const auto found = result.node_partition.find(node);
    if (found == result.node_partition.end()) {
      throw pybind11::value_error("topology load node is disconnected");
    }
    auto& target = result.loads.at(found->second);
    target.insert(target.end(), loads.begin(), loads.end());
  }
  if (result.node_partition.size() != expected.size()) {
    throw pybind11::value_error("topology electrical partitions are disconnected");
  }
  for (const auto& buffer : state.buffers) {
    const auto found = result.node_partition.find(buffer.input_node);
    if (found == result.node_partition.end()) {
      throw pybind11::value_error("topology buffer input is disconnected");
    }
    result.input_partition.push_back(found->second);
  }
  return result;
}

}  // namespace rebuffer
