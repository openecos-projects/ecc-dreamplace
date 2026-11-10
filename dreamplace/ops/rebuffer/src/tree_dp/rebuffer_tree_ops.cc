// SPDX-License-Identifier: BSD-3-Clause
#include "rebuffer_tree_dp.hh"


#include <algorithm>

namespace rebuffer {

void appendTreeLoads(std::vector<int>& target, const std::vector<int>& source)
{
  target.insert(target.end(), source.begin(), source.end());
  std::sort(target.begin(), target.end());
  target.erase(std::unique(target.begin(), target.end()), target.end());
}

void annotateTreeLoads(const DriverTreeNode& node, const TimingView& timing,
                       int driver, TreeOption& option)
{
  appendTreeLoads(option.loads, node.load_pin_ids);
  for (const int pin : node.load_pin_ids) {
    if (pin >= 0 && pin < static_cast<int>(timing.pin_cap.size())) {
      option.cap += timing.pin_cap[pin];
    }
    if (!timing.pin_arrival_rise.empty()) {
      const double rise = timing.pin_required_rise[pin] - timing.pin_arrival_rise[pin];
      const double fall = timing.pin_required_fall[pin] - timing.pin_arrival_fall[pin];
      const double budget = rise <= fall
          ? timing.pin_required_rise[pin] - timing.pin_arrival_rise[driver]
          : timing.pin_required_fall[pin] - timing.pin_arrival_fall[driver];
      option.slack = std::min(option.slack, budget);
    } else if (pin >= 0 && pin < static_cast<int>(timing.pin_slack.size())) {
      option.slack = std::min(option.slack, timing.pin_slack[pin]);
    } else {
      option.slack = std::min(option.slack, 0.0);
    }
    if (pin >= 0 && pin < static_cast<int>(timing.pin_slew_rise.size())) {
      option.slew_rise = std::max(option.slew_rise, timing.pin_slew_rise[pin]);
    }
    if (pin >= 0 && pin < static_cast<int>(timing.pin_slew_fall.size())) {
      option.slew_fall = std::max(option.slew_fall, timing.pin_slew_fall[pin]);
    }
    if (pin >= 0 && pin < static_cast<int>(timing.pin_max_slew.size())
        && timing.pin_max_slew[pin] > 0.0) {
      option.max_load_slew = std::min(option.max_load_slew, timing.pin_max_slew[pin]);
    }
    ++option.fanout;
  }
}

TreeRootTiming annotateTreeRoot(const DesignView& design,
                                const TimingView& timing,
                                const LibertyView& liberty, int driver)
{
  TreeRootTiming root;
  const int pin = rebufferLibPinIndex(design, driver);
  if (pin >= 0 && pin < static_cast<int>(design.pin_drive_resistance.size())) {
    root.resistance = design.pin_drive_resistance[pin];
  }
  if (root.resistance <= 0.0 && driver >= 0 && driver < design.num_pins) {
    const int cell = rebufferCellIndexForNode(design, design.pin2node_map[driver]);
    if (cell >= 0 && cell < static_cast<int>(liberty.cells.size())) {
      root.resistance = liberty.cells[cell].drive_resistance;
    }
  }
  if (driver >= 0 && driver < static_cast<int>(timing.pin_load.size())) {
    root.current_load = timing.pin_load[driver];
  }
  if (driver >= 0 && driver < static_cast<int>(timing.pin_cap.size())) {
    root.output_cap = timing.pin_cap[driver];
  }
  return root;
}

double treeRootCorrection(const TreeRootTiming& root, double cap)
{
  // Anchor the fixed driver model at the TP snapshot; d0 cancels.
  return root.resistance * (root.current_load - cap - root.output_cap);
}

}  // namespace rebuffer

// --- rebuffer_tree_electrical.cc ---

#include <algorithm>
#include <cmath>
#include <limits>

namespace rebuffer {

double treeMarginedLimit(double limit, double margin)
{
  return limit > 0.0 ? limit * (1.0 - margin / 100.0)
                     : std::numeric_limits<double>::infinity();
}

bool treeCanDrive(const TreeOption& option, double resistance,
                  double output_cap, double max_cap, const RepairConfig& config)
{
  if (!config.tree_enforce_electrical) return true;
  const double cap = option.cap + output_cap;
  const double slew = (resistance + config.r_unit * option.max_load_wire)
                      * cap * config.slew_shape_factor;
  return cap <= treeMarginedLimit(max_cap, config.cap_margin)
         && slew <= treeMarginedLimit(option.max_load_slew, config.slew_margin)
         && (config.max_fanout <= 0 || option.fanout <= config.max_fanout);
}

bool treeRootFeasible(const TreeOption& option, const TreeRootTiming& root,
                      const TimingView& timing, int driver,
                      const RepairConfig& config)
{
  const double max_cap = driver < static_cast<int>(timing.pin_max_cap.size())
                             ? timing.pin_max_cap[driver] : 0.0;
  if (!treeCanDrive(option, root.resistance, root.output_cap, max_cap, config)) {
    return false;
  }
  if (!config.tree_enforce_electrical) return true;
  const double max_slew = driver < static_cast<int>(timing.pin_max_slew.size())
                              ? timing.pin_max_slew[driver] : 0.0;
  const double observed = std::max(
      driver < static_cast<int>(timing.pin_slew_rise.size()) ? timing.pin_slew_rise[driver] : 0.0,
      driver < static_cast<int>(timing.pin_slew_fall.size()) ? timing.pin_slew_fall[driver] : 0.0);
  const double slew = std::max(0.0, observed
      - config.slew_shape_factor * treeRootCorrection(root, option.cap));
  return slew <= treeMarginedLimit(max_slew, config.slew_margin);
}

double treeWireStep(const LibertyView& liberty,
                    const CharacterizationResult& characterization,
                    const RepairConfig& config)
{
  double step = config.tree_max_wire_length_um > 0.0
                    ? config.tree_max_wire_length_um
                    : std::numeric_limits<double>::infinity();
  if (!config.tree_enforce_electrical) return step;
  // OpenROAD derives the shared step from the smallest fast buffer's
  // ability to drive another copy of itself under cap and load-slew limits.
  if (characterization.fast_buffers.empty()) return step;
  const auto& buffer = characterization.fast_buffers.front();
  if (buffer.cell_index < 0 || buffer.cell_index >= static_cast<int>(liberty.cells.size())) {
    return step;
  }
  const auto& cell = liberty.cells[buffer.cell_index];
  const double cap = buffer.input_cap + cell.output_cap;
  const double cap_limit = treeMarginedLimit(buffer.max_cap, config.cap_margin);
  if (config.c_unit > 0.0) step = std::min(step, std::max(0.0, (cap_limit - cap) / config.c_unit));
  const double limit = treeMarginedLimit(buffer.max_slew, config.slew_margin)
                       / config.slew_shape_factor;
  if (std::isfinite(limit)) {
    const double a = config.r_unit * config.c_unit;
    const double b = config.r_unit * cap + buffer.drive_resistance * config.c_unit;
    const double c = buffer.drive_resistance * cap - limit;
    // Stable positive root of a L^2 + b L + c = 0, also covering zero RC.
    const double length = c >= 0.0 ? 0.0
        : (a > 0.0 ? -2.0 * c / (b + std::sqrt(b * b - 4.0 * a * c))
                    : (b > 0.0 ? -c / b : std::numeric_limits<double>::infinity()));
    step = std::min(step, length);
  }
  return step;
}

}  // namespace rebuffer

// --- rebuffer_wire_candidates.cc ---

#include <algorithm>
#include <cmath>

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

// --- rebuffer_tree_actions.cc ---
// Stable source-edge actions and their native/physical application order.
#include "rebuffer_driver_tree.hh"


#include <algorithm>
#include <cmath>
#include <list>
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
    std::string& reason,
    const bool compute_digest)
{
  if (!validateCurrentDriverTree(source, reason)) {
    return false;
  }
  target = source;
  if (actions.empty()) {
    target.tree_digest = compute_digest ? currentDriverTreeDigest(target) : "";
    return true;
  }
  target.topology_epoch = source.topology_epoch + 1;
  // Chain actions retain the original edge identity and arrive from load to
  // driver. Splitting again uses the remaining upstream segment, whose child
  // is the buffer introduced by the preceding action on that source edge.
  std::map<std::pair<int, int>, std::pair<int, double>> chain_tips;
  // Edge ids are stable while materializing a candidate: each action rewrites
  // one existing edge in place and appends exactly one downstream edge.  The
  // old implementation erased the edge and rebuilt every node's adjacency
  // list after every action, making a long candidate tree quadratic in the
  // number of inserted buffers.
  std::map<std::pair<int, int>, int> edge_indices;
  std::list<int> edge_order;
  std::map<int, std::list<int>::iterator> edge_order_positions;
  for (int index = 0; index < static_cast<int>(target.edges.size()); ++index) {
    const auto& edge = target.edges[index];
    const auto inserted = edge_indices.emplace(
        std::make_pair(edge.parent_node_id, edge.child_node_id), index);
    if (!inserted.second) {
      reason = "current_driver_tree_duplicate_edge";
      return false;
    }
    edge_order.push_back(index);
    edge_order_positions.emplace(index, std::prev(edge_order.end()));
  }
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
    const auto edge_key = std::make_pair(key.first, active_child);
    const auto edge_position = edge_indices.find(edge_key);
    if (edge_position == edge_indices.end()) {
      reason = "buffer_action_tree_edge_missing";
      return false;
    }
    const int edge_index = edge_position->second;
    const DriverTreeEdge original = target.edges[edge_index];

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

    // Keep the existing edge slot for the upstream half.  This preserves the
    // tree's canonical topology while avoiding vector erase, edge-id rewrite,
    // and a full adjacency rebuild for every action.
    target.edges[edge_index].child_node_id = buffer_node;
    target.edges[edge_index].kind = static_cast<int>(DriverTreeEdgeKind::wire);
    edge_indices.erase(edge_position);
    edge_indices.emplace(std::make_pair(key.first, buffer_node), edge_index);

    const auto order_position = edge_order_positions.at(edge_index);
    edge_order.erase(order_position);
    edge_order.push_back(edge_index);
    edge_order_positions[edge_index] = std::prev(edge_order.end());

    DriverTreeEdge downstream = original;
    downstream.id = static_cast<int>(target.edges.size());
    downstream.parent_node_id = buffer_node;
    downstream.kind = static_cast<int>(DriverTreeEdgeKind::wire);
    const int downstream_index = static_cast<int>(target.edges.size());
    target.edges.push_back(downstream);
    edge_indices.emplace(std::make_pair(buffer_node, active_child), downstream_index);
    edge_order.push_back(downstream_index);
    edge_order_positions.emplace(downstream_index, std::prev(edge_order.end()));

    auto& parent_children = target.nodes[key.first].child_node_ids;
    const auto child_position = std::find(parent_children.begin(), parent_children.end(), active_child);
    if (child_position == parent_children.end()) {
      reason = "buffer_action_parent_child_adjacency_missing";
      return false;
    }
    parent_children.erase(child_position);
    parent_children.push_back(buffer_node);
    target.nodes[buffer_node].parent_node_id = key.first;
    target.nodes[buffer_node].child_node_ids.push_back(active_child);
    target.nodes[active_child].parent_node_id = buffer_node;
  }
  if (!actions.empty()) {
    std::vector<DriverTreeEdge> ordered_edges;
    ordered_edges.reserve(edge_order.size());
    for (const int stable_id : edge_order) {
      DriverTreeEdge edge = target.edges[stable_id];
      edge.id = static_cast<int>(ordered_edges.size());
      ordered_edges.push_back(std::move(edge));
    }
    target.edges = std::move(ordered_edges);
  }
  if (!validateCurrentDriverTree(target, reason)) {
    return false;
  }
  target.tree_digest = compute_digest ? currentDriverTreeDigest(target) : "";
  return true;
}

}  // namespace rebuffer

// --- rebuffer_tree_bindings.cc ---
#include "rebuffer_repair.hh"


namespace py = pybind11;
using namespace rebuffer_python;

namespace rebuffer {
void registerTreeSearchBindings(py::module_& module)
{
  auto plan = [](bool optimize, const py::dict& design_values,
         const py::dict& timing_values,
         const py::dict& liberty_values,
         const py::dict& characterization_values,
         const py::dict& config_values,
         const std::shared_ptr<rebuffer::CurrentDriverTree>& tree) {
        if (tree == nullptr) {
          throw py::value_error("current driver tree is null");
        }
        rebuffer::TreeVgdpResult result;
        std::string reason;
        try {
          const auto design = designFromDict(design_values);
          const auto timing = timingFromDict(timing_values);
          const auto liberty = libertyFromDict(liberty_values);
          const auto characterization
              = characterizationFromDict(characterization_values);
          const auto config = configFromDict(config_values);
          const bool planned = optimize
              ? rebuffer::optimizeCurrentDriverTree(design, timing, liberty,
                    characterization, config, *tree, result, reason)
              : rebuffer::planCurrentDriverTree(design, timing, liberty,
                    characterization, config, *tree, result, reason);
          if (!planned) {
            py::dict output;
            output["status"] = "unsupported";
            py::list reasons;
            reasons.append(reason);
            output["unsupported_reasons"] = std::move(reasons);
            return output;
          }
        } catch (const std::exception& error) {
          py::dict output;
          output["status"] = "unsupported";
          py::list reasons;
          reasons.append(error.what());
          output["unsupported_reasons"] = std::move(reasons);
          return output;
        }
        py::dict output;
        output["status"] = "ok";
        output["generated_option_count"] = result.generated_option_count;
        output["surviving_option_count"] = result.surviving_option_count;
        output["pruned_option_count"] = result.pruned_option_count;
        output["tree_node_count"] = result.tree_node_count;
        output["tree_edge_count"] = result.tree_edge_count;
        output["tree_branch_count"] = result.tree_branch_count;
        output["tree_max_depth"] = result.tree_max_depth;
        output["driver_resistance"] = result.driver_resistance;
        output["current_driver_load"] = result.current_driver_load;
        output["long_wire_edge_count"] = result.long_wire_edge_count;
        output["wire_site_count"] = result.wire_site_count;
        output["electrical_rejected_count"] = result.electrical_rejected_count;
        output["timing_round_count"] = result.timing_round_count;
        output["recovery_attempt_count"] = result.recovery_attempt_count;
        output["recovered_buffer_count"] = result.recovered_buffer_count;
        output["round_source_tree_digests"] = result.round_source_tree_digests;
        output["round_source_buffer_counts"] = result.round_source_buffer_counts;
        py::list candidates;
        for (const auto& candidate : result.candidates) {
          py::dict row;
          row["cap"] = candidate.cap;
          row["slack"] = candidate.slack;
          row["slew_rise"] = candidate.slew_rise;
          row["slew_fall"] = candidate.slew_fall;
          row["area"] = candidate.area;
          row["buffer_count"] = candidate.buffer_count;
          py::list actions;
          for (const auto& action : candidate.actions) {
            actions.append(bufferActionDict(action));
          }
          row["actions"] = std::move(actions);
          candidates.append(std::move(row));
        }
        output["candidates"] = std::move(candidates);
        py::list trace;
        for (const auto& item : result.trace) {
          py::dict row;
          row["trace_id"] = item.trace_id;
          row["node_id"] = item.node_id;
          row["operation"] = item.operation;
          row["cap"] = item.cap;
          row["slack"] = item.slack;
          row["slew_rise"] = item.slew_rise;
          row["slew_fall"] = item.slew_fall;
          row["area"] = item.area;
          row["buffer_count"] = item.buffer_count;
          row["survivor"] = item.survivor;
          row["reason"] = item.reason;
          trace.append(std::move(row));
        }
        output["trace"] = std::move(trace);
        return output;
      };
  for (const bool optimize : {false, true}) {
    module.def(optimize ? "optimize_current_driver_tree" : "plan_current_driver_tree",
        [plan, optimize](const py::dict& design, const py::dict& timing,
                         const py::dict& liberty, const py::dict& characterization,
                         const py::dict& config,
                         const std::shared_ptr<CurrentDriverTree>& tree) {
          return plan(optimize, design, timing, liberty, characterization, config, tree);
        }, py::arg("design"), py::arg("timing"), py::arg("liberty"),
        py::arg("characterization"), py::arg("config"), py::arg("tree"));
  }
}
}  // namespace rebuffer
