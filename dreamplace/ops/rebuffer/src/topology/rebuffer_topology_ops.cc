// SPDX-License-Identifier: BSD-3-Clause

#include "rebuffer_types.hh"


#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <vector>

namespace rebuffer {

namespace {

bool valid_topology(const DesignView& design,
                    const TopologyView& topology,
                    std::string& reason)
{
  if (topology.steiner_node_base < 0) {
    reason = "invalid_steiner_node_base";
    return false;
  }
  if (topology.net_flat_topo_sort_start.size() < 2
      || topology.net_flat_topo_sort_start.front() != 0
      || topology.net_flat_topo_sort_start.back()
             != static_cast<int>(topology.net_flat_topo_sort.size())) {
    reason = "invalid_topology_csr";
    return false;
  }
  if (topology.net_flat_topo_sort_start.size() - 1 != topology.net_ids.size()) {
    reason = "topology_net_count_mismatch";
    return false;
  }
  if (topology.pin_fa.empty()) {
    reason = "empty_pin_fa";
    return false;
  }
  for (std::size_t index = 1; index < topology.net_flat_topo_sort_start.size();
       ++index) {
    if (topology.net_flat_topo_sort_start[index]
        < topology.net_flat_topo_sort_start[index - 1]) {
      reason = "topology_csr_not_monotonic";
      return false;
    }
  }
  std::vector<bool> seen(design.num_nets, false);
  for (const int net_id : topology.net_ids) {
    if (net_id < 0 || net_id >= design.num_nets || seen[net_id]) {
      reason = "invalid_or_duplicate_topology_net";
      return false;
    }
    seen[net_id] = true;
  }
  for (const int node : topology.net_flat_topo_sort) {
    if (node < 0 || node >= static_cast<int>(topology.pin_fa.size())) {
      reason = "topology_node_out_of_range";
      return false;
    }
  }
  return true;
}

}  // namespace

bool build_forest(const DesignView& design,
                  const TopologyView& topology,
                  ForestResult& forest,
                  std::string& reason)
{
  if (!validate_design(design, reason)) {
    return false;
  }
  if (!valid_topology(design, topology, reason)) {
    return false;
  }

  forest = ForestResult{};
  forest.num_nets = static_cast<int>(topology.net_ids.size());
  forest.net_node_start.reserve(forest.num_nets + 1);
  forest.net_node_start.push_back(0);
  forest.net_summaries.reserve(forest.num_nets);

  for (int net_index = 0; net_index < forest.num_nets; ++net_index) {
    const int net_id = topology.net_ids[net_index];
    const int begin = topology.net_flat_topo_sort_start[net_index];
    const int end = topology.net_flat_topo_sort_start[net_index + 1];
    if (begin == end) {
      reason = "empty_topology_net";
      return false;
    }

    const int node_count = end - begin;
    std::unordered_map<int, int> local_index;
    local_index.reserve(node_count);
    for (int local = 0; local < node_count; ++local) {
      const int global = topology.net_flat_topo_sort[begin + local];
      if (!local_index.emplace(global, local).second) {
        reason = "duplicate_topology_node_in_net";
        return false;
      }
    }

    NetForestSummary summary;
    summary.net_id = net_id;
    summary.driver_pin = design.net2driver_pin_map[net_id];
    summary.node_count = node_count;

    std::vector<int> depth(node_count, -1);
    std::vector<int> fanout(node_count, 0);
    int root_index = -1;
    for (int local = 0; local < node_count; ++local) {
      const int global = topology.net_flat_topo_sort[begin + local];
      const int parent = topology.pin_fa[global];
      if (parent < 0) {
        if (root_index >= 0) {
          reason = "multiple_topology_roots";
          return false;
        }
        root_index = local;
        depth[local] = 0;
        summary.root_node = global;
        continue;
      }
      const auto parent_it = local_index.find(parent);
      if (parent_it == local_index.end() || parent_it->second >= local) {
        reason = "parent_not_before_child_in_topology";
        return false;
      }
      depth[local] = depth[parent_it->second] + 1;
      ++fanout[parent_it->second];
    }
    if (root_index < 0) {
      reason = "missing_topology_root";
      return false;
    }

    for (int local = 0; local < node_count; ++local) {
      const int global = topology.net_flat_topo_sort[begin + local];
      const bool is_pin = global < topology.steiner_node_base;
      summary.pin_node_count += is_pin ? 1 : 0;
      summary.steiner_node_count += is_pin ? 0 : 1;
      summary.max_depth = std::max(summary.max_depth, depth[local]);
      summary.max_fanout = std::max(summary.max_fanout, fanout[local]);
      if (global == summary.driver_pin) {
        summary.driver_in_topology = true;
      }
    }
    summary.edge_count = node_count - 1;

    for (int local = 0; local < node_count; ++local) {
      const int global = topology.net_flat_topo_sort[begin + local];
      const int parent = topology.pin_fa[global];
      forest.net_node_ids.push_back(global);
      forest.node_parent.push_back(parent);
      forest.node_depth.push_back(depth[local]);
      forest.node_fanout.push_back(fanout[local]);
    }
    forest.net_node_start.push_back(
        static_cast<int>(forest.net_node_ids.size()));

    forest.num_nodes += summary.node_count;
    forest.num_pin_nodes += summary.pin_node_count;
    forest.num_steiner_nodes += summary.steiner_node_count;
    forest.num_edges += summary.edge_count;
    forest.max_depth = std::max(forest.max_depth, summary.max_depth);
    forest.max_fanout = std::max(forest.max_fanout, summary.max_fanout);
    forest.net_summaries.push_back(summary);
  }

  return true;
}

}  // namespace rebuffer

// --- rebuffer_topology_partitions.cc ---
#include "rebuffer_incremental_topology.hh"


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

// --- rebuffer_topology_remove.cc ---
// Bypass any active buffer without discarding surviving descendants.

#include <algorithm>
#include <limits>

namespace py = pybind11;
namespace rebuffer {

bool removeIncrementalBufferAnywhere(IncrementalTopologyState& state, int action_id,
                                     std::string& reason, py::dict& result)
{
  const auto found = std::find_if(state.buffers.begin(), state.buffers.end(),
      [&](const auto& buffer) { return buffer.action_id == action_id; });
  if (found == state.buffers.end()) {
    reason = "buffer_remove_action_not_found";
    return false;
  }
  if (state.epoch == std::numeric_limits<int>::max()) {
    reason = "topology_epoch_overflow";
    return false;
  }
  const auto buffer = *found;
  int incoming = -1, cell = -1, outgoing = -1, input_children = 0;
  for (int id = 0; id < static_cast<int>(state.edges.size()); ++id) {
    const auto& edge = state.edges[id];
    if (edge.child == buffer.input_node) {
      if (incoming >= 0) { reason = "buffer_input_has_multiple_parents"; return false; }
      incoming = id;
    }
    if (edge.parent == buffer.input_node) {
      ++input_children;
      if (edge.child == buffer.output_node) cell = id;
    }
    if (edge.parent == buffer.output_node) {
      if (outgoing >= 0) { reason = "buffer_output_has_multiple_wire_children"; return false; }
      outgoing = id;
    }
  }
  if (incoming < 0 || outgoing < 0 || cell < 0 || input_children != 1
      || state.edges[cell].resistance != 0 || state.edges[cell].capacitance != 0
      || state.node_loads.count(buffer.input_node) || state.node_loads.count(buffer.output_node)) {
    reason = "buffer_remove_has_invalid_wire_boundary";
    return false;
  }
  const auto& before = state.edges[incoming];
  const auto& after = state.edges[outgoing];
  IncrementalTopologyState candidate = state;
  candidate.edges.clear();
  const int joined_at = std::min(incoming, outgoing);
  for (int id = 0; id < static_cast<int>(state.edges.size()); ++id) {
    if (id == joined_at) {
      candidate.edges.push_back({before.parent, after.child,
          before.resistance + after.resistance, before.capacitance + after.capacitance});
    }
    if (id != incoming && id != outgoing && id != cell) candidate.edges.push_back(state.edges[id]);
  }
  candidate.buffers.erase(candidate.buffers.begin() + (found - state.buffers.begin()));
  candidate.coordinates.erase(buffer.input_node); candidate.coordinates.erase(buffer.output_node);
  candidate.source_node_by_node.erase(buffer.input_node);
  candidate.source_node_by_node.erase(buffer.output_node);
  candidate.general_removal = true;
  ++candidate.epoch;
  invalidateIncrementalTopologyCaches(candidate);
  if (!validateIncrementalTopology(candidate, reason)) return false;
  state = std::move(candidate);
  result["status"] = "ok";
  result["action_id"] = action_id;
  result["epoch"] = state.epoch;
  result["edge_count"] = state.edges.size();
  result["partition_count"] = state.buffers.size() + 1;
  return true;
}

}  // namespace rebuffer

// --- rebuffer_topology_history.cc ---
// Source-local topology checkpoint history and exact restore.

#include <algorithm>

namespace py = pybind11;
namespace rebuffer {

void invalidateIncrementalTopologyCaches(IncrementalTopologyState& state)
{
  state.checkpoint_image.reset();
  state.cached_checkpoint.reset();
  state.validated_epoch = -1;
  state.schedule_cache = py::dict();
  state.schedule_cache_changed_pins.clear();
  state.schedule_cache_epoch = -1;
  state.schedule_cache_slot = -1;
  state.schedule_cache_pin_count = -1;
  state.schedule_cache_row_count = -1;
  state.schedule_cache_net_count = -1;
  state.schedule_cache_event = -1;
}

bool undoLastTopologySplit(
    IncrementalTopologyState& state,
    const IncrementalBuffer& buffer,
    std::string& reason)
{
  // A split erases one edge and appends exactly three edges.  Since only the
  // most recent buffer may be removed, no later mutation can have changed
  // the suffix.  Check the suffix before editing so a stale or foreign
  // journal entry fails loudly instead of silently corrupting the tree.
  if (state.edges.size() < 3) {
    reason = "topology_journal_has_no_split_suffix";
    return false;
  }
  const std::size_t first = state.edges.size() - 3;
  const auto& input_edge = state.edges[first];
  const auto& cell_edge = state.edges[first + 1];
  const auto& output_edge = state.edges[first + 2];
  if (input_edge.parent != buffer.original_edge.parent
      || input_edge.child != buffer.input_node
      || cell_edge.parent != buffer.input_node
      || cell_edge.child != buffer.output_node
      || output_edge.parent != buffer.output_node
      || output_edge.child != buffer.original_edge.child) {
    reason = "topology_journal_split_suffix_mismatch";
    return false;
  }
  if (buffer.original_edge_index > first) {
    reason = "topology_journal_original_edge_index_invalid";
    return false;
  }
  state.edges.resize(first);
  state.edges.insert(
      state.edges.begin() + static_cast<std::ptrdiff_t>(buffer.original_edge_index),
      buffer.original_edge);
  state.next_node_id = buffer.next_node_before;
  state.epoch = buffer.epoch_before;
  state.coordinates.erase(buffer.input_node);
  state.coordinates.erase(buffer.output_node);
  state.source_node_by_node.erase(buffer.input_node);
  state.source_node_by_node.erase(buffer.output_node);
  invalidateIncrementalTopologyCaches(state);
  return true;
}

bool redoTopologySplit(
    IncrementalTopologyState& state,
    const IncrementalBuffer& buffer,
    std::string& reason)
{
  if (std::any_of(
          state.buffers.begin(),
          state.buffers.end(),
          [&](const IncrementalBuffer& active) {
            return active.action_id == buffer.action_id;
          })) {
    reason = "topology_checkpoint_replays_duplicate_action";
    return false;
  }
  auto edge_it = std::find_if(
      state.edges.begin(),
      state.edges.end(),
      [&](const IncrementalEdge& edge) {
        return edge.parent == buffer.original_edge.parent
            && edge.child == buffer.original_edge.child;
      });
  if (edge_it == state.edges.end()) {
    reason = "topology_checkpoint_original_edge_missing";
    return false;
  }
  if (buffer.input_node < 0 || buffer.output_node < 0
      || buffer.input_node == buffer.output_node
      || state.coordinates.count(buffer.input_node) != 0
      || state.coordinates.count(buffer.output_node) != 0) {
    reason = "topology_checkpoint_buffer_node_conflict";
    return false;
  }
  const IncrementalEdge original = *edge_it;
  state.edges.erase(edge_it);
  state.edges.push_back(
      {original.parent,
       buffer.input_node,
       original.resistance * buffer.split_ratio,
       original.capacitance * buffer.split_ratio});
  state.edges.push_back(
      {buffer.input_node, buffer.output_node, 0.0, 0.0});
  state.edges.push_back(
      {buffer.output_node,
       original.child,
       original.resistance * (1.0 - buffer.split_ratio),
       original.capacitance * (1.0 - buffer.split_ratio)});
  if (buffer.has_coordinates) {
    state.coordinates[buffer.input_node] = buffer.split_location;
    state.coordinates[buffer.output_node] = buffer.split_location;
  }
  state.source_node_by_node[buffer.input_node] = buffer.input_node;
  state.source_node_by_node[buffer.output_node] = buffer.output_node;
  state.next_node_id = std::max(state.next_node_id, buffer.output_node + 1);
  state.buffers.push_back(buffer);
  invalidateIncrementalTopologyCaches(state);
  return true;
}

std::shared_ptr<IncrementalTopologyCheckpoint> checkpointIncrementalTopology(
    const IncrementalTopologyState& state)
{
  if (state.cached_checkpoint != nullptr
      && state.cached_checkpoint->net_id == state.net_id
      && state.cached_checkpoint->driver_pin_id == state.driver_pin_id
      && state.cached_checkpoint->buffer_count == state.buffers.size()
      && state.cached_checkpoint->next_node_id == state.next_node_id
      && state.cached_checkpoint->epoch == state.epoch
      && state.cached_checkpoint->schedule_cache_epoch == state.schedule_cache_epoch
      && state.cached_checkpoint->schedule_cache_slot == state.schedule_cache_slot
      && state.cached_checkpoint->schedule_cache_pin_count == state.schedule_cache_pin_count
      && state.cached_checkpoint->schedule_cache_row_count == state.schedule_cache_row_count
      && state.cached_checkpoint->schedule_cache_net_count == state.schedule_cache_net_count
      && state.cached_checkpoint->schedule_cache_event == state.schedule_cache_event
      && (!state.general_removal || state.cached_checkpoint->image != nullptr)) {
    return state.cached_checkpoint;
  }
  auto checkpoint = std::make_shared<IncrementalTopologyCheckpoint>();
  checkpoint->net_id = state.net_id;
  checkpoint->driver_pin_id = state.driver_pin_id;
  checkpoint->buffer_count = state.buffers.size();
  checkpoint->next_node_id = state.next_node_id;
  checkpoint->epoch = state.epoch;
  checkpoint->action_ids.reserve(state.buffers.size());
  for (const auto& buffer : state.buffers) {
    checkpoint->action_ids.push_back(buffer.action_id);
  }
  checkpoint->buffers = state.buffers;
  checkpoint->schedule_cache = state.schedule_cache;
  checkpoint->schedule_cache_changed_pins = state.schedule_cache_changed_pins;
  checkpoint->schedule_cache_epoch = state.schedule_cache_epoch;
  checkpoint->schedule_cache_slot = state.schedule_cache_slot;
  checkpoint->schedule_cache_pin_count = state.schedule_cache_pin_count;
  checkpoint->schedule_cache_row_count = state.schedule_cache_row_count;
  checkpoint->schedule_cache_net_count = state.schedule_cache_net_count;
  checkpoint->schedule_cache_event = state.schedule_cache_event;
  if (state.general_removal) {
    if (!state.checkpoint_image) {
      state.checkpoint_image = std::make_shared<IncrementalTopologyImage>(state);
    }
    checkpoint->image = state.checkpoint_image;
  }
  state.cached_checkpoint = checkpoint;
  return checkpoint;
}

bool restoreIncrementalTopology(
    IncrementalTopologyState& state,
    const IncrementalTopologyCheckpoint& checkpoint,
    std::string& reason)
{
  if (state.cached_checkpoint != nullptr
      && state.cached_checkpoint.get() == &checkpoint
      && state.validated_epoch == state.epoch
      && state.epoch == checkpoint.epoch) {
    return true;
  }
  if (checkpoint.net_id != state.net_id
      || checkpoint.driver_pin_id != state.driver_pin_id) {
    reason = "topology_checkpoint_identity_mismatch";
    return false;
  }
  if (checkpoint.buffer_count != checkpoint.buffers.size()
      || checkpoint.action_ids.size() != checkpoint.buffer_count) {
    reason = "topology_checkpoint_journal_is_incomplete";
    return false;
  }
  for (std::size_t index = 0; index < checkpoint.buffers.size(); ++index) {
    if (checkpoint.buffers[index].action_id != checkpoint.action_ids[index]) {
      reason = "topology_checkpoint_journal_action_mismatch";
      return false;
    }
  }
  if (checkpoint.image) {
    state = checkpoint.image->state;
    state.checkpoint_image = checkpoint.image;
  } else {
    const std::size_t common = std::min(
        state.buffers.size(), checkpoint.buffers.size());
    bool history_matches = !state.general_removal;
    for (std::size_t index = 0; index < common; ++index) {
      const auto& current = state.buffers[index];
      const auto& expected = checkpoint.buffers[index];
      if (current.action_id != expected.action_id
          || current.input_node != expected.input_node
          || current.output_node != expected.output_node
          || current.original_parent != expected.original_parent
          || current.original_child != expected.original_child
          || current.split_ratio != expected.split_ratio
          || current.master_name != expected.master_name
          || current.has_coordinates != expected.has_coordinates
          || current.split_location != expected.split_location) {
        history_matches = false;
        break;
      }
    }
    if (!history_matches) {
      if (!state.base_captured) {
        reason = "topology_checkpoint_history_diverged";
        return false;
      }
      state.general_removal = false;
      state.edges = state.base_edges;
      state.node_loads = state.base_node_loads;
      state.coordinates = state.base_coordinates;
      state.source_node_by_node = state.base_source_node_by_node;
      state.buffers.clear();
      state.next_node_id = state.base_next_node_id;
      state.epoch = state.base_epoch;
      invalidateIncrementalTopologyCaches(state);
    }
    while (state.buffers.size() > checkpoint.buffer_count) {
      const auto buffer = state.buffers.back();
      if (!undoLastTopologySplit(state, buffer, reason)) {
        return false;
      }
      state.buffers.pop_back();
    }
    while (state.buffers.size() < checkpoint.buffer_count) {
      const auto& buffer = checkpoint.buffers[state.buffers.size()];
      if (!redoTopologySplit(state, buffer, reason)) {
        return false;
      }
    }
    state.general_removal = false;
    state.checkpoint_image.reset();
  }
  state.next_node_id = checkpoint.next_node_id;
  state.epoch = checkpoint.epoch;
  state.schedule_cache = checkpoint.schedule_cache;
  state.schedule_cache_changed_pins = checkpoint.schedule_cache_changed_pins;
  state.schedule_cache_epoch = checkpoint.schedule_cache_epoch;
  state.schedule_cache_slot = checkpoint.schedule_cache_slot;
  state.schedule_cache_pin_count = checkpoint.schedule_cache_pin_count;
  state.schedule_cache_row_count = checkpoint.schedule_cache_row_count;
  state.schedule_cache_net_count = checkpoint.schedule_cache_net_count;
  state.schedule_cache_event = checkpoint.schedule_cache_event;
  if (!validateIncrementalTopology(state, reason)) {
    return false;
  }
  state.cached_checkpoint = std::make_shared<IncrementalTopologyCheckpoint>(checkpoint);
  return true;
}


}  // namespace rebuffer
