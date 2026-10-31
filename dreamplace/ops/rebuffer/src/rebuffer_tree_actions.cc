// SPDX-License-Identifier: BSD-3-Clause
// Stable source-edge actions and their native/physical application order.
#include "rebuffer_tree_actions.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include "pybind11/stl.h"

namespace py = pybind11;
namespace rebuffer {

std::vector<py::dict> orderTreeBufferActions(const py::list& actions)
{
  struct OrderedAction
  {
    py::dict value;
    int index = -1;
    int depth = -1;
    int source_parent = -1;
    int source_child = -1;
    double source_fraction = 0.0;
  };
  std::vector<OrderedAction> ordered;
  ordered.reserve(actions.size());
  bool has_depth = false;
  for (const auto& item : actions) {
    const py::dict value = item.cast<py::dict>();
    const int depth = value.contains("segment_tree_depth")
                          ? py::cast<int>(value["segment_tree_depth"])
                          : -1;
    has_depth = has_depth || depth >= 0;
    const int index = static_cast<int>(ordered.size());
    const bool chain = value.contains("source_edge_fraction");
    ordered.push_back({value, index, depth,
                       chain ? py::cast<int>(value["parent_node_id"]) : index,
                       chain ? py::cast<int>(value["child_node_id"]) : -1,
                       chain ? py::cast<double>(value["source_edge_fraction"]) : 0.0});
  }
  if (has_depth) {
    std::stable_sort(ordered.begin(), ordered.end(),
                     [](const OrderedAction& lhs, const OrderedAction& rhs) {
                       const int lhs_depth = lhs.depth < 0 ? std::numeric_limits<int>::max()
                                                           : lhs.depth;
                       const int rhs_depth = rhs.depth < 0 ? std::numeric_limits<int>::max()
                                                           : rhs.depth;
                       if (lhs_depth != rhs_depth) return lhs_depth < rhs_depth;
                       // Native materialization walks a source-edge chain
                       // load-first. The physical overlay walks driver-first
                       // so its downstream terminal partition stays intact.
                       if (lhs.source_parent != rhs.source_parent) return lhs.source_parent < rhs.source_parent;
                       if (lhs.source_child != rhs.source_child) return lhs.source_child < rhs.source_child;
                       if (lhs.source_fraction != rhs.source_fraction) return lhs.source_fraction < rhs.source_fraction;
                       return lhs.index < rhs.index;
                     });
  }
  std::vector<py::dict> values;
  for (const auto& action : ordered) values.push_back(action.value);
  return values;
}

bool applyCurrentDriverTreeBufferActions(
    const CurrentDriverTree& source,
    const LibertyView& liberty,
    const std::vector<BufferAction>& actions,
    CurrentDriverTree& target,
    std::string& reason)
{
  if (!validateCurrentDriverTree(source, reason)) {
    return false;
  }
  target = source;
  if (actions.empty()) {
    target.tree_digest = currentDriverTreeDigest(target);
    return true;
  }
  target.topology_epoch = source.topology_epoch + 1;
  // Chain actions retain the original edge identity and arrive from load to
  // driver. Splitting again uses the remaining upstream segment, whose child
  // is the buffer introduced by the preceding action on that source edge.
  std::map<std::pair<int, int>, std::pair<int, double>> chain_tips;
  for (const auto& action : actions) {
    if (action.parent_node_id < 0
        || action.parent_node_id >= static_cast<int>(source.nodes.size())
        || action.child_node_id < 0
        || action.child_node_id >= static_cast<int>(source.nodes.size())) {
      reason = "buffer_action_tree_edge_out_of_range";
      return false;
    }
    if (action.buffer_cell_index < 0
        || action.buffer_cell_index >= static_cast<int>(liberty.cells.size())
        || !liberty.cells[action.buffer_cell_index].is_buffer) {
      reason = "buffer_action_cell_is_not_buffer";
      return false;
    }
    if (action.source_parent_node_id >= 0
        || action.source_child_node_id >= 0) {
      if (action.source_parent_node_id < 0
          || action.source_child_node_id < 0) {
        reason = "buffer_action_physical_source_identity_incomplete";
        return false;
      }
      if (source.nodes[action.parent_node_id].source_node_id
              != action.source_parent_node_id
          || source.nodes[action.child_node_id].source_node_id
                 != action.source_child_node_id) {
        reason = "buffer_action_physical_source_identity_mismatch";
        return false;
      }
    }
    const auto key = std::make_pair(action.parent_node_id, action.child_node_id);
    const auto& from = source.nodes[key.first];
    const auto& to = source.nodes[key.second];
    const double dx = to.x - from.x, dy = to.y - from.y;
    const double length2 = dx * dx + dy * dy;
    const double split = length2 > 0.0
        ? ((action.x_dbu - from.x) * dx + (action.y_dbu - from.y) * dy) / length2 : 0.0;
    const auto previous = chain_tips.find(key);
    if (previous != chain_tips.end() && split >= previous->second.second - 1.0e-12) {
      reason = "buffer_action_chain_order_invalid";
      return false;
    }
    const int active_child = previous == chain_tips.end()
                                 ? key.second : previous->second.first;
    const auto found = std::find_if(target.edges.begin(), target.edges.end(), [&](const DriverTreeEdge& edge) {
      return edge.parent_node_id == key.first && edge.child_node_id == active_child;
    });
    const int edge_index = found == target.edges.end() ? -1
        : static_cast<int>(found - target.edges.begin());
    if (edge_index < 0) {
      reason = "buffer_action_tree_edge_missing";
      return false;
    }
    const DriverTreeEdge original = target.edges[edge_index];
    target.edges.erase(target.edges.begin() + edge_index);

    DriverTreeNode buffer;
    buffer.source_node_id = -1;
    buffer.source_pin_id = -1;
    buffer.source_instance_id = -1;
    buffer.master_id = liberty.cells[action.buffer_cell_index].main_id;
    buffer.cell_id = action.buffer_cell_index;
    buffer.net_id = original.net_id;
    buffer.partition_id = original.partition_id;
    buffer.kind = static_cast<int>(DriverTreeNodeKind::buffer);
    buffer.existing_buffer = false;
    buffer.x = action.x_dbu;
    buffer.y = action.y_dbu;
    buffer.id = static_cast<int>(target.nodes.size());
    const int buffer_node = buffer.id;
    target.nodes.push_back(std::move(buffer));
    chain_tips[key] = {buffer_node, split};

    DriverTreeEdge upstream = original;
    upstream.id = static_cast<int>(target.edges.size());
    upstream.child_node_id = buffer_node;
    upstream.kind = static_cast<int>(DriverTreeEdgeKind::wire);
    target.edges.push_back(upstream);
    DriverTreeEdge downstream = original;
    downstream.id = static_cast<int>(target.edges.size());
    downstream.parent_node_id = buffer_node;
    downstream.kind = static_cast<int>(DriverTreeEdgeKind::wire);
    target.edges.push_back(downstream);

    for (int index = 0; index < static_cast<int>(target.edges.size()); ++index) {
      target.edges[index].id = index;
    }
    for (auto& node : target.nodes) {
      node.child_node_ids.clear();
      node.parent_node_id = -1;
    }
    for (const auto& edge : target.edges) {
      target.nodes[edge.parent_node_id].child_node_ids.push_back(
          edge.child_node_id);
      target.nodes[edge.child_node_id].parent_node_id = edge.parent_node_id;
    }
  }
  if (!validateCurrentDriverTree(target, reason)) {
    return false;
  }
  target.tree_digest = currentDriverTreeDigest(target);
  return true;
}

}  // namespace rebuffer
