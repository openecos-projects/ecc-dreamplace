// SPDX-License-Identifier: BSD-3-Clause
// Mutable per-net topology state for native rebuffer transactions.

#include "rebuffer_incremental_topology.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <sstream>
#include <unordered_set>

#include "pybind11/stl.h"

namespace py = pybind11;

namespace rebuffer {
namespace {

template <typename T>
T required(const py::dict& values, const char* field)
{
  if (!values.contains(field)) {
    throw py::value_error(std::string("missing ") + field);
  }
  return py::cast<T>(values[field]);
}

std::vector<IncrementalEdge> edgesFromDict(const py::dict& values)
{
  std::vector<IncrementalEdge> edges;
  if (!values.contains("edges")) {
    return edges;
  }
  for (const auto& item : values["edges"].cast<py::sequence>()) {
    const auto edge = item.cast<py::dict>();
    edges.push_back({required<int>(edge, "parent"),
                     required<int>(edge, "child"),
                     required<double>(edge, "r"),
                     required<double>(edge, "c")});
  }
  return edges;
}

std::unordered_map<int, std::vector<int>> loadsFromDict(const py::dict& values)
{
  std::unordered_map<int, std::vector<int>> loads;
  if (!values.contains("node_loads")) {
    return loads;
  }
  for (const auto& item : values["node_loads"].cast<py::dict>()) {
    const int node = py::cast<int>(item.first);
    loads[node] = item.second.cast<std::vector<int>>();
  }
  return loads;
}

int actionId(const py::dict& action)
{
  if (action.contains("action_id")) {
    return py::cast<int>(action["action_id"]);
  }
  if (action.contains("session_buffer_id")) {
    return py::cast<int>(action["session_buffer_id"]);
  }
  throw py::value_error("buffer split requires action_id or session_buffer_id");
}

std::vector<int> downstreamLoads(
    const IncrementalTopologyState& state,
    int child)
{
  std::unordered_map<int, std::vector<int>> children;
  for (const auto& edge : state.edges) {
    children[edge.parent].push_back(edge.child);
  }
  std::vector<int> nodes;
  std::vector<int> stack{child};
  std::unordered_set<int> seen;
  while (!stack.empty()) {
    const int node = stack.back();
    stack.pop_back();
    if (!seen.insert(node).second) {
      continue;
    }
    nodes.push_back(node);
    for (const int next : children[node]) {
      stack.push_back(next);
    }
  }
  return nodes;
}

bool containsAll(const std::vector<int>& values, const std::vector<int>& subset)
{
  const std::unordered_set<int> available(values.begin(), values.end());
  return std::all_of(
      subset.begin(), subset.end(), [&](const int value) {
        return available.find(value) != available.end();
      });
}

py::list edgeList(const std::vector<IncrementalEdge>& edges)
{
  py::list result;
  for (const auto& edge : edges) {
    py::dict value;
    value["parent"] = edge.parent;
    value["child"] = edge.child;
    value["r"] = edge.resistance;
    value["c"] = edge.capacitance;
    result.append(value);
  }
  return result;
}

}  // namespace

std::shared_ptr<IncrementalTopologyState> createIncrementalTopology(
    const int net_id,
    const int driver_pin_id,
    const int next_node_id,
    const py::dict& topology_values,
    std::string& reason)
{
  auto state = std::make_shared<IncrementalTopologyState>();
  state->net_id = net_id;
  state->driver_pin_id = driver_pin_id;
  state->next_node_id = next_node_id;
  state->edges = edgesFromDict(topology_values);
  state->node_loads = loadsFromDict(topology_values);
  if (topology_values.contains("coordinates")) {
    for (const auto& item : topology_values["coordinates"].cast<py::dict>()) {
      state->coordinates[py::cast<int>(item.first)]
          = py::cast<std::array<double, 2>>(item.second);
    }
  }
  if (!validateIncrementalTopology(*state, reason)) {
    return nullptr;
  }
  return state;
}

bool validateIncrementalTopology(
    const IncrementalTopologyState& state,
    std::string& reason)
{
  if (state.driver_pin_id < 0) {
    reason = "driver_pin_id_must_be_nonnegative";
    return false;
  }
  std::unordered_set<int> nodes{state.driver_pin_id};
  std::unordered_set<int> children;
  std::unordered_set<std::string> edge_keys;
  std::unordered_map<int, std::vector<int>> adjacency;
  for (const auto& edge : state.edges) {
    if (edge.parent < 0 || edge.child < 0 || edge.parent == edge.child) {
      reason = "invalid_edge_endpoint";
      return false;
    }
    if (!std::isfinite(edge.resistance) || !std::isfinite(edge.capacitance)
        || edge.resistance < 0.0 || edge.capacitance < 0.0) {
      reason = "invalid_edge_rc";
      return false;
    }
    std::ostringstream key;
    key << edge.parent << ':' << edge.child;
    if (!edge_keys.insert(key.str()).second) {
      reason = "duplicate_edge";
      return false;
    }
    nodes.insert(edge.parent);
    nodes.insert(edge.child);
    if (!children.insert(edge.child).second) {
      reason = "node_has_multiple_parents";
      return false;
    }
    adjacency[edge.parent].push_back(edge.child);
  }
  for (const auto& [node, loads] : state.node_loads) {
    if (nodes.find(node) == nodes.end()) {
      reason = "load_node_not_in_topology";
      return false;
    }
    std::unordered_set<int> local(loads.begin(), loads.end());
    if (local.size() != loads.size()) {
      reason = "duplicate_load_pin_in_node";
      return false;
    }
    if (std::any_of(loads.begin(), loads.end(), [](int pin) { return pin < 0; })) {
      reason = "load_pin_must_be_nonnegative";
      return false;
    }
  }
  if (children.count(state.driver_pin_id) != 0) {
    reason = "driver_has_parent";
    return false;
  }
  if (state.next_node_id <= *std::max_element(nodes.begin(), nodes.end())) {
    reason = "next_node_id_collides_with_topology";
    return false;
  }
  if (!state.coordinates.empty()) {
    for (const int node : nodes) {
      const auto point = state.coordinates.find(node);
      if (point == state.coordinates.end()
          || !std::isfinite(point->second[0])
          || !std::isfinite(point->second[1])) {
        reason = "missing_or_invalid_node_coordinate";
        return false;
      }
    }
  }
  std::unordered_set<int> visited;
  std::unordered_set<int> active;
  std::function<bool(int)> visit = [&](const int node) {
    if (active.find(node) != active.end()) {
      return false;
    }
    if (!visited.insert(node).second) {
      return true;
    }
    active.insert(node);
    for (const int child : adjacency[node]) {
      if (!visit(child)) {
        return false;
      }
    }
    active.erase(node);
    return true;
  };
  if (!visit(state.driver_pin_id)) {
    reason = "topology_contains_cycle";
    return false;
  }
  if (visited.size() != nodes.size()) {
    reason = "topology_has_disconnected_node";
    return false;
  }
  std::unordered_set<int> all_loads;
  for (const auto& [node, loads] : state.node_loads) {
    for (const int pin : loads) {
      if (!all_loads.insert(pin).second) {
        reason = "load_pin_in_multiple_partitions";
        return false;
      }
    }
  }
  return true;
}

bool applyIncrementalBufferSplit(
    IncrementalTopologyState& state,
    const py::dict& action,
    std::string& reason,
    py::dict& result)
{
  const int parent = required<int>(action, "parent_node_id");
  const int child = required<int>(action, "child_node_id");
  const double ratio = required<double>(action, "split_ratio");
  const int id = actionId(action);
  if (id < 0 || state.next_node_id > std::numeric_limits<int>::max() - 2) {
    reason = "invalid_action_or_node_id";
    return false;
  }
  if (!(ratio > 0.0 && ratio < 1.0) || !std::isfinite(ratio)) {
    reason = "split_ratio_must_be_inside_edge";
    return false;
  }
  auto edge_it = std::find_if(
      state.edges.begin(),
      state.edges.end(),
      [&](const IncrementalEdge& edge) {
        return edge.parent == parent && edge.child == child;
      });
  if (edge_it == state.edges.end()) {
    reason = "active_edge_not_found";
    return false;
  }
  if (std::any_of(state.buffers.begin(), state.buffers.end(),
                  [&](const auto& buffer) {
                    return buffer.input_node == parent && buffer.output_node == child;
                  })) {
    reason = "cannot_split_buffer_cell_edge";
    return false;
  }
  if (std::any_of(
          state.buffers.begin(),
          state.buffers.end(),
          [&](const IncrementalBuffer& buffer) {
            return buffer.action_id == id;
          })) {
    reason = "duplicate_action_id";
    return false;
  }
  if (!action.contains("downstream_pin_ids")) {
    reason = "downstream_pin_ids_required";
    return false;
  }
  const auto selected = action["downstream_pin_ids"].cast<std::vector<int>>();
  const std::unordered_set<int> selected_set(selected.begin(), selected.end());
  if (selected.empty() || selected_set.size() != selected.size()) {
    reason = "downstream_partition_must_be_nonempty_and_unique";
    return false;
  }
  const auto subtree = downstreamLoads(state, child);
  std::vector<int> subtree_loads;
  for (const int node : subtree) {
    const auto found = state.node_loads.find(node);
    if (found != state.node_loads.end()) {
      subtree_loads.insert(
          subtree_loads.end(), found->second.begin(), found->second.end());
    }
  }
  if (!containsAll(subtree_loads, selected)) {
    reason = "downstream_partition_is_outside_edge_subtree";
    return false;
  }
  // Splitting a wire moves its complete subtree. A strict subset requires a
  // different branch edge; changing only a load ledger would miswire siblings.
  if (selected.size() != subtree_loads.size()) {
    reason = "downstream_partition_does_not_match_edge_subtree";
    return false;
  }
  std::array<double, 2> location{};
  if (!state.coordinates.empty()) {
    const auto& from = state.coordinates.at(parent);
    const auto& to = state.coordinates.at(child);
    for (int axis = 0; axis < 2; ++axis) {
      location[axis] = from[axis] + ratio * (to[axis] - from[axis]);
    }
    if (action.contains("x_dbu") && action.contains("y_dbu")) {
      const double x = required<double>(action, "x_dbu");
      const double y = required<double>(action, "y_dbu");
      if (!std::isfinite(x) || !std::isfinite(y)
          || std::abs(x - location[0]) > 1.0
          || std::abs(y - location[1]) > 1.0) {
        reason = "split_location_does_not_match_ratio";
        return false;
      }
    }
  }

  IncrementalBuffer buffer;
  buffer.action_id = id;
  if (action.contains("buffer_master_name")) {
    buffer.master_name = py::cast<std::string>(action["buffer_master_name"]);
  }
  buffer.original_parent = parent;
  buffer.original_child = child;
  buffer.split_ratio = ratio;
  buffer.next_node_before = state.next_node_id;
  buffer.epoch_before = state.epoch;
  buffer.edges_before = state.edges;
  buffer.loads_before = state.node_loads;
  buffer.input_node = state.next_node_id++;
  buffer.output_node = state.next_node_id++;
  buffer.upstream_node = buffer.input_node;
  buffer.downstream_node = buffer.output_node;

  const IncrementalEdge original = *edge_it;
  state.edges.erase(edge_it);
  state.edges.push_back(
      {parent,
       buffer.input_node,
       original.resistance * ratio,
       original.capacitance * ratio});
  state.edges.push_back({buffer.input_node, buffer.output_node, 0.0, 0.0});
  state.edges.push_back(
      {buffer.output_node,
       child,
       original.resistance * (1.0 - ratio),
       original.capacitance * (1.0 - ratio)});
  // Loads stay at their physical sink nodes. Electrical partitions are
  // separated by the buffer cell edge when exported, never by relocating caps.
  if (!state.coordinates.empty()) {
    state.coordinates[buffer.input_node] = location;
    state.coordinates[buffer.output_node] = location;
  }
  ++state.epoch;

  if (!validateIncrementalTopology(state, reason)) {
    state.edges = buffer.edges_before;
    state.node_loads = buffer.loads_before;
    state.next_node_id = buffer.next_node_before;
    state.epoch = buffer.epoch_before;
    state.coordinates.erase(buffer.input_node);
    state.coordinates.erase(buffer.output_node);
    return false;
  }
  state.buffers.push_back(buffer);
  result["status"] = "ok";
  result["action_id"] = id;
  result["input_node"] = buffer.input_node;
  result["output_node"] = buffer.output_node;
  result["epoch"] = state.epoch;
  result["edge_count"] = state.edges.size();
  result["partition_count"] = state.buffers.size() + 1;
  result["downstream_pin_ids"] = selected;
  return true;
}

bool removeIncrementalBuffer(
    IncrementalTopologyState& state,
    const int action_id,
    std::string& reason,
    py::dict& result)
{
  if (state.buffers.empty() || state.buffers.back().action_id != action_id) {
    reason = "only_last_buffer_can_be_removed_without_rebuild";
    return false;
  }
  const auto buffer = state.buffers.back();
  state.edges = buffer.edges_before;
  state.node_loads = buffer.loads_before;
  state.next_node_id = buffer.next_node_before;
  state.epoch = buffer.epoch_before + 1;
  state.coordinates.erase(buffer.input_node);
  state.coordinates.erase(buffer.output_node);
  state.buffers.pop_back();
  if (!validateIncrementalTopology(state, reason)) {
    return false;
  }
  result["status"] = "ok";
  result["action_id"] = action_id;
  result["epoch"] = state.epoch;
  result["edge_count"] = state.edges.size();
  result["partition_count"] = state.buffers.size() + 1;
  return true;
}

std::shared_ptr<IncrementalTopologyCheckpoint> checkpointIncrementalTopology(
    const IncrementalTopologyState& state)
{
  auto checkpoint = std::make_shared<IncrementalTopologyCheckpoint>();
  checkpoint->state = state;
  return checkpoint;
}

bool restoreIncrementalTopology(
    IncrementalTopologyState& state,
    const IncrementalTopologyCheckpoint& checkpoint,
    std::string& reason)
{
  state = checkpoint.state;
  return validateIncrementalTopology(state, reason);
}

py::dict incrementalTopologyView(const IncrementalTopologyState& state)
{
  py::dict result;
  result["status"] = "ok";
  result["net_id"] = state.net_id;
  result["driver_pin_id"] = state.driver_pin_id;
  result["next_node_id"] = state.next_node_id;
  result["epoch"] = state.epoch;
  result["edges"] = edgeList(state.edges);
  py::dict loads;
  for (const auto& [node, pins] : std::map<int, std::vector<int>>(
           state.node_loads.begin(), state.node_loads.end())) {
    loads[py::int_(node)] = pins;
  }
  result["node_loads"] = loads;
  py::dict coordinates;
  for (const auto& [node, point] : std::map<int, std::array<double, 2>>(
           state.coordinates.begin(), state.coordinates.end())) {
    coordinates[py::int_(node)] = point;
  }
  result["coordinates"] = coordinates;
  // A zero-RC cell edge is a logical connection, not a wire. Each buffer
  // output starts a new electrical net; its input loads the upstream net.
  std::unordered_map<int, int> buffer_inputs;
  std::unordered_map<int, int> buffer_outputs;
  for (const auto& buffer : state.buffers) {
    buffer_inputs[buffer.input_node] = buffer.action_id;
    buffer_outputs[buffer.output_node] = buffer.action_id;
  }
  std::unordered_map<int, std::vector<int>> children;
  for (const auto& edge : state.edges) children[edge.parent].push_back(edge.child);
  std::map<int, int> node_partition;
  std::map<int, int> drivers{{-1, state.driver_pin_id}};
  std::map<int, std::vector<int>> partition_nodes;
  std::map<int, std::vector<int>> partition_loads;
  std::vector<std::pair<int, int>> pending{{state.driver_pin_id, -1}};
  while (!pending.empty()) {
    auto [node, partition] = pending.back();
    pending.pop_back();
    const auto output = buffer_outputs.find(node);
    if (output != buffer_outputs.end()) {
      partition = output->second;
      drivers[partition] = node;
    }
    node_partition[node] = partition;
    partition_nodes[partition].push_back(node);
    const auto loads_at_node = state.node_loads.find(node);
    if (loads_at_node != state.node_loads.end()) {
      auto& loads = partition_loads[partition];
      loads.insert(loads.end(), loads_at_node->second.begin(), loads_at_node->second.end());
    }
    if (buffer_inputs.count(node) != 0) partition_loads[partition].push_back(node);
    for (const int child : children[node]) pending.emplace_back(child, partition);
  }
  py::dict partitions;
  for (auto& [partition, nodes] : partition_nodes) {
    std::sort(nodes.begin(), nodes.end());
    auto& pins = partition_loads[partition];
    std::sort(pins.begin(), pins.end());
    std::vector<IncrementalEdge> wires;
    for (const auto& edge : state.edges) {
      if (node_partition.at(edge.parent) == partition
          && node_partition.at(edge.child) == partition) wires.push_back(edge);
    }
    py::dict value;
    value["driver_node_id"] = drivers.at(partition);
    value["node_ids"] = nodes;
    value["load_pin_ids"] = pins;
    value["edges"] = edgeList(wires);
    partitions[py::int_(partition)] = value;
  }
  result["partitions"] = partitions;
  result["node_partition"] = node_partition;
  py::list buffers;
  for (const auto& buffer : state.buffers) {
    py::dict value;
    value["action_id"] = buffer.action_id;
    value["buffer_master_name"] = buffer.master_name;
    value["input_node"] = buffer.input_node;
    value["output_node"] = buffer.output_node;
    value["original_parent"] = buffer.original_parent;
    value["original_child"] = buffer.original_child;
    value["split_ratio"] = buffer.split_ratio;
    buffers.append(value);
  }
  result["buffers"] = buffers;
  return result;
}

}  // namespace rebuffer
