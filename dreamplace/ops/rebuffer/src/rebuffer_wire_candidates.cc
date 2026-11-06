// SPDX-License-Identifier: BSD-3-Clause
#include "rebuffer_wire_candidates.hh"

#include <algorithm>
#include <cmath>
#include "rebuffer_tree_electrical.hh"

namespace rebuffer {
namespace {

TreeOption addWire(TreeOption option, double length, const RepairConfig& config)
{
  const double cap = length * config.c_unit;
  const double res = length * config.r_unit;
  option.slack -= res * (option.cap + cap / 2.0);
  option.slew_rise += res * option.cap;
  option.slew_fall += res * option.cap;
  option.cap += cap;
  option.max_load_wire += length;
  return option;
}

void insertAt(std::vector<TreeOption>& output, const TreeOption& downstream,
              double upstream_length, double split,
              const DesignView& design, const LibertyView& liberty,
              const CharacterizationResult& characterization,
              const RepairConfig& config, const CurrentDriverTree& tree,
              const DriverTreeEdge& edge, int depth)
{
  if (!config.insert_buffers || downstream.buffer_count >= config.tree_max_buffers) return;
  const auto& parent = tree.nodes[edge.parent_node_id];
  const auto& child = tree.nodes[edge.child_node_id];
  for (const auto& buffer : characterization.fast_buffers) {
    if (buffer.cell_index < 0 || buffer.cell_index >= static_cast<int>(liberty.cells.size())) continue;
    const auto& cell = liberty.cells[buffer.cell_index];
    if (!cell.is_buffer || !treeCanDrive(downstream, buffer.drive_resistance,
                                        cell.output_cap, buffer.max_cap, config)) continue;
    TreeOption option = downstream;
    option.slack -= cell.intrinsic_delay
                    + buffer.drive_resistance * (downstream.cap + cell.output_cap);
    option.slew_rise = option.slew_fall
        = buffer.drive_resistance * (downstream.cap + cell.output_cap);
    option.cap = buffer.input_cap;
    option.area += buffer.area;
    ++option.buffer_count;
    option.max_load_wire = 0.0;
    option.max_load_slew = buffer.max_slew > 0.0
                               ? buffer.max_slew : std::numeric_limits<double>::infinity();
    option.fanout = 1;
    BufferAction action;
    action.net_id = parent.net_id;
    action.driver_pin = tree.driver_pin_id;
    action.downstream_pins = downstream.loads;
    action.buffer_cell_index = buffer.cell_index;
    action.parent_node_id = edge.parent_node_id;
    action.child_node_id = edge.child_node_id;
    action.tree_depth = depth;
    // Original edge identity is shared by every member of a wire chain.
    if (parent.source_node_id >= 0 && child.source_node_id >= 0) {
      action.source_parent_node_id = parent.source_node_id;
      action.source_child_node_id = child.source_node_id;
    }
    action.x_dbu = static_cast<int>(std::llround(parent.x + split * (child.x - parent.x)));
    action.y_dbu = static_cast<int>(std::llround(parent.y + split * (child.y - parent.y)));
    action.split_ratio = split;
    action.downstream_load = downstream.cap;
    action.target_load = buffer.target_load;
    action.reason = "tree_vgdp_insert";
    option.actions.push_back(std::move(action));
    output.push_back(addWire(std::move(option), upstream_length, config));
  }
}

}  // namespace

std::vector<TreeOption> treeWireCandidates(
    const DesignView& design, const LibertyView& liberty,
    const CharacterizationResult& characterization, const RepairConfig& config,
    const CurrentDriverTree& tree, const DriverTreeEdge& edge, int depth,
    const std::vector<TreeOption>& children, TreeVgdpResult& result,
    int& trace_id, std::string& reason)
{
  if (edge.kind != static_cast<int>(DriverTreeEdgeKind::wire)) return children;
  const auto& parent = tree.nodes[edge.parent_node_id];
  const auto& child = tree.nodes[edge.child_node_id];
  const double length = (std::abs(parent.x - child.x) + std::abs(parent.y - child.y))
                        / std::max(1, design.dbu);
  const double step = treeWireStep(liberty, characterization, config);
  std::vector<TreeOption> options;
  if (!config.insert_buffers || characterization.fast_buffers.empty()
      || length <= step || !std::isfinite(step)) {
    for (const auto& downstream : children) {
      options.push_back(addWire(downstream, length, config));
      for (const double split : {0.0, 0.5, 1.0}) {
        insertAt(options, addWire(downstream, length * (1.0 - split), config),
                 length * split, split, design, liberty, characterization,
                 config, tree, edge, depth);
      }
    }
  } else {
    if (!(step > 0.0) || length / step > 1024.0) {
      reason = "tree_wire_step_budget_exhausted";
      return {};
    }
    ++result.long_wire_edge_count;
    options = children;
    // Walk from load towards driver. Each site sees the frontier produced
    // at the preceding site, so several buffers can occupy one source edge.
    const int segments = static_cast<int>(std::ceil(length / step));
    for (int site = segments; site >= 0; --site) {
      const double split = static_cast<double>(site) / segments;
      if (site != segments) {
        for (auto& option : options) option = addWire(std::move(option), length / segments, config);
      }
      auto next = options;
      for (const auto& option : options) {
        insertAt(next, option, 0.0, split, design, liberty, characterization,
                 config, tree, edge, depth);
      }
      options = std::move(next);
      ++result.wire_site_count;
      pruneTreeOptions(options, edge.parent_node_id, result, trace_id);
    }
  }
  for (auto& option : options) option.node_id = edge.parent_node_id;
  return options;
}

}  // namespace rebuffer
