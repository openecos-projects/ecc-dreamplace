// SPDX-License-Identifier: BSD-3-Clause
// Bypass any active buffer without discarding surviving descendants.
#include "rebuffer_topology_history.hh"

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
