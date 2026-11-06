// SPDX-License-Identifier: BSD-3-Clause
// Linear (capacitance, required-time budget) current-tree VGDP.

#include "rebuffer_tree_dp.hh"
#include "rebuffer_tree_annotation.hh"
#include "rebuffer_tree_electrical.hh"
#include "rebuffer_wire_candidates.hh"

#include <algorithm>
#include <cmath>
#include <functional>
#include <limits>
#include <map>

namespace rebuffer {
using Option = TreeOption;
constexpr double kInf = std::numeric_limits<double>::infinity();

void pruneTreeOptions(std::vector<Option>& options,
                  int node_id,
                  TreeVgdpResult& result,
                  int& next_trace_id)
{
  result.generated_option_count += static_cast<int>(options.size());
  // Timing-mode VGDP keeps strictly increasing Q with increasing C.
  // Area/count break exact ties; they do not introduce extra Pareto axes.
  std::sort(options.begin(), options.end(), [](const Option& lhs, const Option& rhs) {
    if (lhs.cap != rhs.cap) return lhs.cap < rhs.cap;
    if (lhs.slack != rhs.slack) return lhs.slack > rhs.slack;
    if (lhs.buffer_count != rhs.buffer_count) {
      return lhs.buffer_count < rhs.buffer_count;
    }
    return lhs.area < rhs.area;
  });
  std::vector<Option> survivors;
  survivors.reserve(options.size());
  double best_slack = -kInf;
  for (int index = 0; index < static_cast<int>(options.size()); ++index) {
    const bool dominated = !survivors.empty() && options[index].slack <= best_slack;
    TreeVgdpTrace trace;
    trace.trace_id = next_trace_id++;
    trace.node_id = node_id;
    trace.operation = "pareto_prune";
    trace.cap = options[index].cap;
    trace.slack = options[index].slack;
    trace.slew_rise = options[index].slew_rise;
    trace.slew_fall = options[index].slew_fall;
    trace.area = options[index].area;
    trace.buffer_count = options[index].buffer_count;
    trace.survivor = !dominated;
    trace.reason = dominated ? "dominated" : "frontier_survivor";
    result.trace.push_back(std::move(trace));
    if (!dominated) {
      best_slack = options[index].slack;
      survivors.push_back(std::move(options[index]));
    }
  }
  result.pruned_option_count += static_cast<int>(options.size() - survivors.size());
  options = std::move(survivors);
}

namespace {

int findEdge(const CurrentDriverTree& tree, int parent, int child)
{
  for (int index = 0; index < static_cast<int>(tree.edges.size()); ++index) {
    if (tree.edges[index].parent_node_id == parent
        && tree.edges[index].child_node_id == child) {
      return index;
    }
  }
  return -1;
}

}  // namespace

bool planCurrentDriverTree(const DesignView& design,
                           const TimingView& timing,
                           const LibertyView& liberty,
                           const CharacterizationResult& characterization,
                           const RepairConfig& config,
                           const CurrentDriverTree& tree,
                           TreeVgdpResult& result,
                           std::string& reason)
{
  result = TreeVgdpResult{};
  if (!validateCurrentDriverTree(tree, reason)) {
    return false;
  }
  if (tree.nodes.size() > 8192) {
    reason = "tree_node_budget_exhausted";
    return false;
  }
  const auto pin_count = static_cast<std::size_t>(design.num_pins);
  if ((!timing.pin_arrival_rise.empty() || !timing.pin_arrival_fall.empty()
       || !timing.pin_required_rise.empty() || !timing.pin_required_fall.empty())
      && (timing.pin_arrival_rise.size() != pin_count
          || timing.pin_arrival_fall.size() != pin_count
          || timing.pin_required_rise.size() != pin_count
          || timing.pin_required_fall.size() != pin_count)) {
    reason = "tree_timing_snapshot_size_mismatch";
    return false;
  }
  result.tree_node_count = static_cast<int>(tree.nodes.size());
  result.tree_edge_count = static_cast<int>(tree.edges.size());
  std::vector<int> depth(tree.nodes.size(), 0);
  std::vector<int> traversal{tree.root_node_id};
  for (std::size_t i = 0; i < traversal.size(); ++i) {
    const int parent = traversal[i];
    if (tree.nodes[parent].child_node_ids.size() > 1) ++result.tree_branch_count;
    for (const int child : tree.nodes[parent].child_node_ids) {
      depth[child] = depth[parent] + 1;
      result.tree_max_depth = std::max(result.tree_max_depth, depth[child]);
      traversal.push_back(child);
    }
  }
  std::vector<int> trace_id{0};
  std::map<int, std::vector<Option>> memo;

  std::function<std::vector<Option>(int)> visit = [&](const int node_id) {
    const auto found = memo.find(node_id);
    if (found != memo.end()) {
      return found->second;
    }
    const auto& node = tree.nodes[node_id];
    std::vector<Option> options;
    if (node.child_node_ids.empty()) {
      Option leaf;
      leaf.node_id = node_id;
      leaf.slack = kInf;
      annotateTreeLoads(node, timing, tree.driver_pin_id, leaf);
      if (node.load_pin_ids.empty()) leaf.slack = 0.0;
      options.push_back(std::move(leaf));
    } else {
      Option sentinel;
      sentinel.slack = kInf;
      sentinel.node_id = node_id;
      options.push_back(std::move(sentinel));
      for (const int child_id : node.child_node_ids) {
        std::vector<Option> edge_options;
        const auto child_options = visit(child_id);
        const int edge_id = findEdge(tree, node_id, child_id);
        if (edge_id < 0) {
          reason = "tree_edge_missing_from_child_ancestry";
          return std::vector<Option>{};
        }
        const auto& edge = tree.edges[edge_id];
        edge_options = treeWireCandidates(design, liberty, characterization,
                                          config, tree, edge, depth[node_id],
                                          child_options, result, trace_id[0], reason);
        if (!reason.empty()) return std::vector<Option>{};
        std::vector<Option> merged;
        if (options.size() == 1 && options.front().slack == kInf
            && options.front().cap == 0.0) {
          merged = std::move(edge_options);
        } else {
          for (const auto& lhs : options) {
            for (const auto& rhs : edge_options) {
              Option combined = lhs;
              combined.cap += rhs.cap;
              combined.slack = std::min(combined.slack, rhs.slack);
              combined.slew_rise = std::max(combined.slew_rise,
                                            rhs.slew_rise);
              combined.slew_fall = std::max(combined.slew_fall,
                                            rhs.slew_fall);
              combined.max_load_slew = std::min(combined.max_load_slew, rhs.max_load_slew);
              combined.max_load_wire = std::max(combined.max_load_wire, rhs.max_load_wire);
              combined.fanout += rhs.fanout;
              combined.area += rhs.area;
              combined.buffer_count += rhs.buffer_count;
              appendTreeLoads(combined.loads, rhs.loads);
              combined.actions.insert(combined.actions.end(),
                                      rhs.actions.begin(),
                                      rhs.actions.end());
              if (combined.buffer_count <= config.tree_max_buffers) {
                merged.push_back(std::move(combined));
              }
            }
          }
        }
        options = std::move(merged);
        pruneTreeOptions(options, node_id, result, trace_id[0]);
        if (options.empty()) {
          reason = "tree_vgdp_frontier_empty";
          return options;
        }
      }
    }

    // A logic input can be an electrical load at an internal Steiner vertex
    // while another sink continues below it.  Its load belongs to every
    // option reaching the parent, but it must not be included in a buffer
    // action placed on the child edge.  Add it only after all child edges
    // have been combined so the action partition remains physical.
    if (!node.child_node_ids.empty()) {
      for (auto& option : options) {
        annotateTreeLoads(node, timing, tree.driver_pin_id, option);
      }
    }

    if (node.kind == static_cast<int>(DriverTreeNodeKind::buffer)
        && !options.empty()) {
      const int cell = node.cell_id >= 0 ? node.cell_id
          : rebufferCellIndexForNode(design, node.source_instance_id);
      if (cell >= 0 && cell < static_cast<int>(liberty.cells.size())) {
        const auto& buffer = liberty.cells[cell];
        options.erase(std::remove_if(options.begin(), options.end(), [&](const Option& option) {
          return !treeCanDrive(option, buffer.drive_resistance, buffer.output_cap,
                               buffer.max_cap, config);
        }), options.end());
        for (auto& option : options) {
          option.slack -= buffer.intrinsic_delay
                          + buffer.drive_resistance * (option.cap + buffer.output_cap);
          option.cap = buffer.input_cap;
          option.max_load_wire = 0.0;
          option.max_load_slew = buffer.max_slew > 0.0 ? buffer.max_slew : kInf;
          option.fanout = 1;
          option.area += buffer.area;
        }
      }
    }
    pruneTreeOptions(options, node_id, result, trace_id[0]);
    memo.emplace(node_id, options);
    return options;
  };

  const auto root_options = visit(tree.root_node_id);
  if (root_options.empty()) {
    if (reason.empty()) reason = "tree_vgdp_no_root_option";
    return false;
  }
  std::vector<Option> ranked = root_options;
  const int driver = tree.driver_pin_id;
  const auto root = annotateTreeRoot(design, timing, liberty, driver);
  result.driver_resistance = root.resistance;
  result.current_driver_load = root.current_load;
  ranked.erase(std::remove_if(ranked.begin(), ranked.end(), [&](const Option& option) {
    const bool feasible = treeRootFeasible(option, root, timing, driver, config);
    if (!feasible) ++result.electrical_rejected_count;
    return !feasible;
  }), ranked.end());
  if (ranked.empty()) {
    reason = "tree_no_electrically_feasible_root";
    return false;
  }
  for (auto& option : ranked) {
    option.slack += treeRootCorrection(root, option.cap);
    result.trace.push_back({trace_id[0]++, tree.root_node_id, "root_score",
                            option.cap, option.slack, option.slew_rise,
                            option.slew_fall, option.area, option.buffer_count,
                            true, "linear_driver_load_correction"});
  }
  std::sort(ranked.begin(), ranked.end(), [](const Option& lhs, const Option& rhs) {
    if (lhs.slack != rhs.slack) return lhs.slack > rhs.slack;
    if (lhs.buffer_count != rhs.buffer_count) {
      return lhs.buffer_count < rhs.buffer_count;
    }
    return lhs.area < rhs.area;
  });
  const int max_candidates = 8;
  for (int index = 0; index < std::min(max_candidates, static_cast<int>(ranked.size()));
       ++index) {
    TreeVgdpCandidate candidate;
    candidate.actions = ranked[index].actions;
    candidate.cap = ranked[index].cap;
    candidate.slack = ranked[index].slack;
    candidate.slew_rise = ranked[index].slew_rise;
    candidate.slew_fall = ranked[index].slew_fall;
    candidate.area = ranked[index].area;
    candidate.buffer_count = ranked[index].buffer_count;
    result.candidates.push_back(std::move(candidate));
  }
  result.surviving_option_count = static_cast<int>(result.candidates.size());
  return true;
}

}  // namespace rebuffer
