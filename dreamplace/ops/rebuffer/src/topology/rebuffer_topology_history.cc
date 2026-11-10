// SPDX-License-Identifier: BSD-3-Clause
// Source-local topology checkpoint history and exact restore.
#include "rebuffer_topology_history.hh"

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
