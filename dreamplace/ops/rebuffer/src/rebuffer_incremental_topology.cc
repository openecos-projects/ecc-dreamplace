// SPDX-License-Identifier: BSD-3-Clause
// Mutable per-net topology state for native rebuffer transactions.

#include "rebuffer_incremental_topology.hh"
#include "rebuffer_topology_history.hh"

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

void captureBaseTopology(IncrementalTopologyState& state)
{
  state.base_edges = state.edges;
  state.base_node_loads = state.node_loads;
  state.base_coordinates = state.coordinates;
  state.base_source_node_by_node = state.source_node_by_node;
  state.base_next_node_id = state.next_node_id;
  state.base_epoch = state.epoch;
  state.base_captured = true;
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
  state->source_net_id = net_id;
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
  for (const auto& edge : state->edges) {
    state->source_node_by_node.emplace(edge.parent, edge.parent);
    state->source_node_by_node.emplace(edge.child, edge.child);
  }
  if (!validateIncrementalTopology(*state, reason)) {
    return nullptr;
  }
  captureBaseTopology(*state);
  return state;
}

std::shared_ptr<IncrementalTopologyState> remapIncrementalTopology(
    const std::shared_ptr<IncrementalTopologyState>& source,
    const int original_pin_count,
    const int node_shift,
    const int target_net_id,
    const int synthetic_node_base,
    const py::dict& source_node_mapping)
{
  if (!source) throw std::invalid_argument("cannot_remap_null_topology");
  if (original_pin_count < 0 || node_shift < 0) {
    throw std::invalid_argument("invalid_topology_node_namespace");
  }
  auto mapped = std::make_shared<IncrementalTopologyState>(*source);
  if (target_net_id >= 0) mapped->net_id = target_net_id;
  std::unordered_map<int, int> explicit_nodes;
  explicit_nodes.reserve(py::len(source_node_mapping));
  for (const auto& item : source_node_mapping) {
    explicit_nodes.emplace(
        py::cast<int>(item.first), py::cast<int>(item.second));
  }
  std::unordered_map<int, int> synthetic_nodes;
  if (synthetic_node_base >= 0) {
    for (std::size_t slot = 0; slot < source->buffers.size(); ++slot) {
      const auto& buffer = source->buffers[slot];
      synthetic_nodes[buffer.input_node]
          = synthetic_node_base + static_cast<int>(2 * slot);
      synthetic_nodes[buffer.output_node]
          = synthetic_node_base + static_cast<int>(2 * slot + 1);
    }
  }
  const auto remap_node = [&](const int node) {
    if (node < 0 || node < original_pin_count) return node;
    const auto synthetic = synthetic_nodes.find(node);
    if (synthetic != synthetic_nodes.end()) return synthetic->second;
    const auto source_node = source->source_node_by_node.find(node);
    if (source_node != source->source_node_by_node.end()) {
      const auto mapped_source = explicit_nodes.find(source_node->second);
      if (mapped_source != explicit_nodes.end()) return mapped_source->second;
    }
    const auto mapped_direct = explicit_nodes.find(node);
    if (mapped_direct != explicit_nodes.end()) return mapped_direct->second;
    return node + node_shift;
  };
  const auto remap_edge = [&](IncrementalEdge& edge) {
    edge.parent = remap_node(edge.parent);
    edge.child = remap_node(edge.child);
  };
  auto remap_edges = [&](std::vector<IncrementalEdge>& edges) {
    for (auto& edge : edges) remap_edge(edge);
  };
  auto remap_load_map = [&](std::unordered_map<int, std::vector<int>>& loads) {
    std::unordered_map<int, std::vector<int>> remapped;
    remapped.reserve(loads.size());
    for (auto& [node, pins] : loads) {
      remapped.emplace(remap_node(node), std::move(pins));
    }
    loads = std::move(remapped);
  };
  auto remap_coordinates =
      [&](std::unordered_map<int, std::array<double, 2>>& coordinates) {
        std::unordered_map<int, std::array<double, 2>> remapped;
        remapped.reserve(coordinates.size());
        for (const auto& [node, point] : coordinates) {
          remapped.emplace(remap_node(node), point);
        }
        coordinates = std::move(remapped);
      };
  auto remap_source_nodes = [&](const auto& source_nodes) {
    std::unordered_map<int, int> remapped;
    remapped.reserve(source_nodes.size());
    for (const auto& [node, source_node] : source_nodes) {
      remapped[remap_node(node)] = source_node;
    }
    return remapped;
  };

  mapped->next_node_id = remap_node(mapped->next_node_id);
  mapped->base_next_node_id = remap_node(mapped->base_next_node_id);
  if (synthetic_node_base >= 0) {
    const int minimum_next = synthetic_node_base
        + static_cast<int>(2 * source->buffers.size());
    mapped->next_node_id = std::max(mapped->next_node_id, minimum_next);
    mapped->base_next_node_id = std::max(mapped->base_next_node_id, minimum_next);
  }
  remap_edges(mapped->edges);
  remap_edges(mapped->base_edges);
  remap_load_map(mapped->node_loads);
  remap_load_map(mapped->base_node_loads);
  remap_coordinates(mapped->coordinates);
  remap_coordinates(mapped->base_coordinates);
  mapped->source_node_by_node = remap_source_nodes(source->source_node_by_node);
  mapped->base_source_node_by_node = remap_source_nodes(source->base_source_node_by_node);
  for (auto& buffer : mapped->buffers) {
    buffer.input_node = remap_node(buffer.input_node);
    buffer.output_node = remap_node(buffer.output_node);
    buffer.original_parent = remap_node(buffer.original_parent);
    buffer.original_child = remap_node(buffer.original_child);
    buffer.upstream_node = remap_node(buffer.upstream_node);
    buffer.downstream_node = remap_node(buffer.downstream_node);
    remap_edge(buffer.original_edge);
  }

  int max_mapped_node = mapped->driver_pin_id;
  for (const auto& edge : mapped->edges) {
    max_mapped_node = std::max(max_mapped_node, edge.parent);
    max_mapped_node = std::max(max_mapped_node, edge.child);
  }
  for (const auto& [node, _point] : mapped->coordinates) {
    max_mapped_node = std::max(max_mapped_node, node);
  }
  mapped->next_node_id = std::max(mapped->next_node_id, max_mapped_node + 1);
  mapped->base_next_node_id = std::max(
      mapped->base_next_node_id, max_mapped_node + 1);

  // The structural cache belongs to the old node namespace.  A remapped
  // state must rebuild it under the caller's compact graph identity.
  invalidateIncrementalTopologyCaches(*mapped);

  std::string reason;
  if (!validateIncrementalTopology(*mapped, reason)) {
    throw std::invalid_argument("remapped_topology_invalid:" + reason);
  }
  return mapped;
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
  invalidateIncrementalTopologyCaches(state);
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
  buffer.original_edge = *edge_it;
  buffer.original_edge_index = static_cast<std::size_t>(
      std::distance(state.edges.begin(), edge_it));
  if (!state.coordinates.empty()) {
    buffer.has_coordinates = true;
    buffer.split_location = location;
  }
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
  state.source_node_by_node[buffer.input_node] = buffer.input_node;
  state.source_node_by_node[buffer.output_node] = buffer.output_node;
  ++state.epoch;

  if (!validateIncrementalTopology(state, reason)) {
    if (!undoLastTopologySplit(state, buffer, reason)) {
      return false;
    }
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
  invalidateIncrementalTopologyCaches(state);
  if (state.general_removal) {
    return removeIncrementalBufferAnywhere(state, action_id, reason, result);
  }
  const auto buffer = state.buffers.back();
  if (!undoLastTopologySplit(state, buffer, reason)) {
    return false;
  }
  // Preserve the historical epoch contract: removing a buffer is itself a
  // topology mutation, even though the journal restores the pre-split graph.
  state.epoch = buffer.epoch_before + 1;
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

bool resizeIncrementalBuffer(
    IncrementalTopologyState& state,
    const py::dict& action,
    std::string& reason,
    py::dict& result)
{
  const int id = actionId(action);
  if (id < 0 || !action.contains("buffer_master_name")) {
    reason = "resize_action_requires_buffer_master";
    return false;
  }
  const std::string master
      = py::cast<std::string>(action["buffer_master_name"]);
  if (master.empty()) {
    reason = "resize_buffer_master_is_empty";
    return false;
  }
  auto found = std::find_if(
      state.buffers.begin(), state.buffers.end(),
      [&](const IncrementalBuffer& buffer) { return buffer.action_id == id; });
  if (found == state.buffers.end()) {
    reason = "resize_buffer_action_not_found";
    return false;
  }
  if (found->master_name == master) {
    result["status"] = "noop";
    result["action_id"] = id;
    result["buffer_master_name"] = master;
    result["epoch"] = state.epoch;
    return true;
  }
  invalidateIncrementalTopologyCaches(state);
  const std::string previous = found->master_name;
  found->master_name = master;
  ++state.epoch;
  result["status"] = "ok";
  result["action_id"] = id;
  result["previous_buffer_master_name"] = previous;
  result["buffer_master_name"] = master;
  result["epoch"] = state.epoch;
  result["partition_count"] = state.buffers.size() + 1;
  return true;
}

py::dict incrementalTopologyView(const IncrementalTopologyState& state)
{
  py::dict result;
  result["status"] = "ok";
  result["net_id"] = state.net_id;
  result["source_net_id"] = state.source_net_id >= 0 ? state.source_net_id : state.net_id;
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
  py::dict source_nodes;
  for (const auto& [node, source_node] : std::map<int, int>(
           state.source_node_by_node.begin(), state.source_node_by_node.end())) {
    source_nodes[py::int_(node)] = source_node;
  }
  result["source_node_by_node"] = source_nodes;
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
