// SPDX-License-Identifier: BSD-3-Clause
// Linear (capacitance, required-time budget) current-tree VGDP.

#include "rebuffer_tree_dp.hh"


#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>

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
    if (next_trace_id >= 0) {
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
    }
    if (!dominated) {
      best_slack = options[index].slack;
      survivors.push_back(std::move(options[index]));
    }
  }
  result.pruned_option_count += static_cast<int>(options.size() - survivors.size());
  options = std::move(survivors);
}

bool planCurrentDriverTree(const DesignView& design,
                           const TimingView& timing,
                           const LibertyView& liberty,
                           const CharacterizationResult& characterization,
                           const RepairConfig& config,
                           const CurrentDriverTree& tree,
                           TreeVgdpResult& result,
                           std::string& reason,
                           bool record_trace)
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
  std::vector<int> edge_by_child(tree.nodes.size(), -1);
  for (int index = 0; index < static_cast<int>(tree.edges.size()); ++index) {
    edge_by_child[tree.edges[index].child_node_id] = index;
  }
  std::vector<int> depth(tree.nodes.size(), 0);
  std::vector<int> traversal{tree.root_node_id};
  traversal.reserve(tree.nodes.size());
  for (std::size_t i = 0; i < traversal.size(); ++i) {
    const int parent = traversal[i];
    if (tree.nodes[parent].child_node_ids.size() > 1) ++result.tree_branch_count;
    for (const int child : tree.nodes[parent].child_node_ids) {
      depth[child] = depth[parent] + 1;
      result.tree_max_depth = std::max(result.tree_max_depth, depth[child]);
      traversal.push_back(child);
    }
  }
  int trace_id = record_trace ? 0 : -1;
  std::vector<std::vector<Option>> memo(tree.nodes.size());
  std::vector<std::uint8_t> memo_ready(tree.nodes.size(), 0);

  std::function<const std::vector<Option>&(int)> visit = [&](const int node_id) -> const std::vector<Option>& {
    if (memo_ready[node_id]) {
      return memo[node_id];
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
        const auto& child_options = visit(child_id);
        const int edge_id = edge_by_child[child_id];
        if (edge_id < 0 || tree.edges[edge_id].parent_node_id != node_id) {
          reason = "tree_edge_missing_from_child_ancestry";
          memo_ready[node_id] = 1;
          return memo[node_id];
        }
        const auto& edge = tree.edges[edge_id];
        edge_options = treeWireCandidates(design, liberty, characterization,
                                          config, tree, edge, depth[node_id],
                                          child_options, result, trace_id, reason);
        if (!reason.empty()) {
          memo_ready[node_id] = 1;
          return memo[node_id];
        }
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
        pruneTreeOptions(options, node_id, result, trace_id);
        if (options.empty()) {
          reason = "tree_vgdp_frontier_empty";
          memo_ready[node_id] = 1;
          return memo[node_id];
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
    pruneTreeOptions(options, node_id, result, trace_id);
    memo[node_id] = std::move(options);
    memo_ready[node_id] = 1;
    return memo[node_id];
  };

  const auto& root_options = visit(tree.root_node_id);
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
    if (record_trace) {
      result.trace.push_back({trace_id++, tree.root_node_id, "root_score",
                              option.cap, option.slack, option.slew_rise,
                              option.slew_fall, option.area, option.buffer_count,
                              true, "linear_driver_load_correction"});
    }
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

// --- rebuffer_tree_search.cc ---

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>

namespace rebuffer {
namespace {

std::vector<int> terminalLoads(const CurrentDriverTree& tree, int node)
{
  auto loads = tree.nodes[node].load_pin_ids;
  for (const int child : tree.nodes[node].child_node_ids) {
    appendTreeLoads(loads, terminalLoads(tree, child));
  }
  return loads;
}

// Added buffers remain on a source edge. Flatten every chain back to that
// stable edge so the physical bridge never receives synthetic graph IDs.
std::vector<BufferAction> flattenTree(const CurrentDriverTree& source,
                                      const CurrentDriverTree& target)
{
  std::vector<BufferAction> actions;
  for (const auto& edge : source.edges) {
    const auto& parent = source.nodes[edge.parent_node_id];
    const auto& child = source.nodes[edge.child_node_id];
    int depth = 0;
    for (int p = parent.id; source.nodes[p].parent_node_id >= 0;
         p = source.nodes[p].parent_node_id) ++depth;
    for (int node = target.nodes[child.id].parent_node_id; node != parent.id;
         node = target.nodes[node].parent_node_id) {
      const auto& buffer = target.nodes[node];
      if (buffer.source_instance_id >= 0 || buffer.existing_buffer) {
        continue;
      }
      BufferAction action;
      action.net_id = parent.net_id;
      action.driver_pin = source.driver_pin_id;
      action.downstream_pins = terminalLoads(source, child.id);
      action.buffer_cell_index = buffer.cell_id;
      action.parent_node_id = parent.id;
      action.child_node_id = child.id;
      if (parent.source_node_id >= 0 && child.source_node_id >= 0) {
        action.source_parent_node_id = parent.source_node_id;
        action.source_child_node_id = child.source_node_id;
      }
      action.tree_depth = depth;
      action.x_dbu = static_cast<int>(std::llround(buffer.x));
      action.y_dbu = static_cast<int>(std::llround(buffer.y));
      const double dx = child.x - parent.x, dy = child.y - parent.y;
      const double length2 = dx * dx + dy * dy;
      action.split_ratio = length2 > 0.0
          ? ((buffer.x - parent.x) * dx + (buffer.y - parent.y) * dy) / length2 : 0.0;
      action.reason = "tree_vgdp_insert";
      actions.push_back(std::move(action));
    }
  }
  return actions;
}

bool better(const TreeVgdpCandidate& lhs, const TreeVgdpCandidate& rhs)
{
  if (lhs.slack != rhs.slack) return lhs.slack > rhs.slack;
  if (lhs.buffer_count != rhs.buffer_count) return lhs.buffer_count < rhs.buffer_count;
  return lhs.area < rhs.area;
}

}  // namespace

bool optimizeCurrentDriverTree(
    const DesignView& design, const TimingView& timing, const LibertyView& liberty,
    const CharacterizationResult& characterization, const RepairConfig& config,
    const CurrentDriverTree& source, TreeVgdpResult& result, std::string& reason,
    std::map<std::string, double>* phase_timings,
    bool record_trace,
    bool prune_worse_than_unbuffered)
{
  auto record = [&](const char* phase, std::chrono::steady_clock::time_point started) {
    if (phase_timings) (*phase_timings)[phase] += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - started).count();
  };
  auto plan = [&](const CurrentDriverTree& tree, const RepairConfig& settings,
                  TreeVgdpResult& planned, std::string& rejected, bool trace = false) {
    const auto started = std::chrono::steady_clock::now();
    const bool ok = planCurrentDriverTree(design, timing, liberty, characterization,
                                         settings, tree, planned, rejected, trace);
    // Both paths use the same linear model. Only search enumerates insertions.
    record(settings.insert_buffers ? "vgdp_dp" : "vgdp_rescore", started);
    return ok;
  };
  auto materialize = [&](const CurrentDriverTree& tree, const std::vector<BufferAction>& actions,
                         CurrentDriverTree& target, std::string& rejected) {
    const auto started = std::chrono::steady_clock::now();
    // Candidate trees are private inputs to the native linear re-score.  Their
    // identity is not published or used for acceptance; computing the full
    // canonical digest here repeats a deep subtree-string construction for
    // every candidate.  Keep validation and topology checks, and defer the
    // digest to the round identity/output paths that actually consume it.
    const bool ok = applyCurrentDriverTreeBufferActions(
        tree, liberty, actions, target, rejected, false);
    record("vgdp_materialize", started);
    return ok;
  };
  if (!plan(source, config, result, reason, record_trace)) return false;
  result.timing_round_count = 1;
  result.round_source_tree_digests.push_back(source.tree_digest);
  result.round_source_buffer_counts.push_back(0);
  auto scoring_config = config;
  scoring_config.insert_buffers = false;
  auto score_tree = [&](const CurrentDriverTree& tree,
                        const std::vector<BufferAction>& actions,
                        TreeVgdpCandidate& candidate) {
    std::string rejected;
    TreeVgdpResult scored;
    if (!plan(tree, scoring_config, scored, rejected, false)) return false;
    candidate = scored.candidates.front();
    candidate.actions = actions;
    candidate.buffer_count = static_cast<int>(actions.size());
    return true;
  };
  auto score = [&](const std::vector<BufferAction>& actions, TreeVgdpCandidate& candidate) {
    if (actions.size() > static_cast<std::size_t>(config.tree_max_buffers)) return false;
    CurrentDriverTree tree;
    std::string rejected;
    if (!materialize(source, actions, tree, rejected)) return false;
    return score_tree(tree, actions, candidate);
  };

  // Re-score the materialized tree as well as DP options. Integer DBU sites
  // and concrete buffer cell IDs must obey the same model used by recovery.
  auto raw = std::move(result.candidates);
  double unbuffered_slack = -std::numeric_limits<double>::infinity();
  for (const auto& candidate : raw) {
    TreeVgdpCandidate scored;
    if (score(candidate.actions, scored)) {
      if (scored.actions.empty()) {
        unbuffered_slack = std::max(unbuffered_slack, scored.slack);
      }
      result.candidates.push_back(std::move(scored));
    }
  }
  if (prune_worse_than_unbuffered && std::isfinite(unbuffered_slack)) {
    result.candidates.erase(
        std::remove_if(result.candidates.begin(), result.candidates.end(),
                       [&](const TreeVgdpCandidate& candidate) {
                         return !candidate.actions.empty()
                                && candidate.slack <= unbuffered_slack + 1.0e-9;
                       }),
        result.candidates.end());
  }
  if (result.candidates.empty()) {
    reason = "tree_no_materializable_candidate";
    return false;
  }
  std::stable_sort(result.candidates.begin(), result.candidates.end(), better);
  auto incumbent = result.candidates.front();
  for (int round = 1; round < config.tree_timing_rounds && config.insert_buffers; ++round) {
    CurrentDriverTree tree;
    if (!materialize(source, incumbent.actions, tree, reason)) return false;
    TreeVgdpResult next;
    std::string no_improvement;
    ++result.timing_round_count;
    tree.tree_digest = currentDriverTreeDigest(tree);
    result.round_source_tree_digests.push_back(tree.tree_digest);
    result.round_source_buffer_counts.push_back(incumbent.buffer_count);
    if (!plan(tree, config, next, no_improvement, false)) break;
    result.generated_option_count += next.generated_option_count;
    result.pruned_option_count += next.pruned_option_count;
    result.long_wire_edge_count += next.long_wire_edge_count;
    result.wire_site_count += next.wire_site_count;
    result.electrical_rejected_count += next.electrical_rejected_count;
    auto best = incumbent;
    for (const auto& addition : next.candidates) {
      if (addition.actions.empty()) continue;
      CurrentDriverTree proposed;
      if (!materialize(tree, addition.actions, proposed, no_improvement)) continue;
      TreeVgdpCandidate candidate;
      if (!score(flattenTree(source, proposed), candidate)) continue;
      if (better(candidate, best)) best = candidate;
      result.candidates.push_back(std::move(candidate));
    }
    if (best.slack <= incumbent.slack + 1.0e-9) break;
    incumbent = std::move(best);
  }

  // Area recovery is local to the proposed insertion bundle. Existing-buffer
  // removal/rewrite remains owned by rebuffer_tree_recovery/placeio_ecc.
  // Preserve the best predicted timing; no unqualified relaxation allowance.
  for (int round = 0; round < config.tree_recovery_rounds; ++round) {
    if (incumbent.actions.empty()) break;
    CurrentDriverTree incumbent_tree;
    std::string rejected;
    if (!materialize(source, incumbent.actions, incumbent_tree, rejected)) break;
    auto recovered = incumbent;
    for (int index = 0; index < static_cast<int>(incumbent.actions.size()); ++index) {
      const int old_cell = incumbent.actions[index].buffer_cell_index;
      auto probe = [&](std::vector<BufferAction> actions) {
        ++result.recovery_attempt_count;
        TreeVgdpCandidate candidate;
        if (score(actions, candidate) && candidate.slack >= incumbent.slack - 1.0e-9
            && candidate.area < recovered.area) recovered = std::move(candidate);
      };
      auto removed = incumbent.actions;
      removed.erase(removed.begin() + index);
      probe(std::move(removed));
      const int node_id = static_cast<int>(source.nodes.size()) + index;
      for (const auto& buffer : characterization.filtered_buffers) {
        if (buffer.cell_index < 0 || buffer.cell_index >= static_cast<int>(liberty.cells.size())) continue;
        const auto& cell = liberty.cells[buffer.cell_index];
        if (!cell.is_buffer || cell.main_id != liberty.cells[old_cell].main_id
            || cell.area >= liberty.cells[old_cell].area) continue;
        ++result.recovery_attempt_count;
        if (incumbent.area - liberty.cells[old_cell].area + cell.area >= recovered.area) {
          continue;
        }
        auto rewritten = incumbent.actions;
        rewritten[index].buffer_cell_index = buffer.cell_index;
        incumbent_tree.nodes[node_id].cell_id = buffer.cell_index;
        TreeVgdpCandidate candidate;
        const bool ok = score_tree(incumbent_tree, rewritten, candidate);
        incumbent_tree.nodes[node_id].cell_id = old_cell;
        if (ok && candidate.slack >= incumbent.slack - 1.0e-9
            && candidate.area < recovered.area) {
          recovered = std::move(candidate);
        }
      }
    }
    if (recovered.area >= incumbent.area) break;
    result.recovered_buffer_count += incumbent.buffer_count - recovered.buffer_count;
    incumbent = std::move(recovered);
    result.candidates.push_back(incumbent);
  }
  std::stable_sort(result.candidates.begin(), result.candidates.end(), better);
  // Different rounds can rediscover one action bundle; expose each once.
  std::vector<TreeVgdpCandidate> unique;
  for (auto& candidate : result.candidates) {
    const auto same = [&](const TreeVgdpCandidate& other) {
      if (candidate.actions.size() != other.actions.size()) return false;
      for (std::size_t i = 0; i < candidate.actions.size(); ++i) {
        const auto& a = candidate.actions[i];
        const auto& b = other.actions[i];
        if (a.parent_node_id != b.parent_node_id || a.child_node_id != b.child_node_id
            || a.buffer_cell_index != b.buffer_cell_index || a.x_dbu != b.x_dbu
            || a.y_dbu != b.y_dbu) return false;
      }
      return true;
    };
    if (std::none_of(unique.begin(), unique.end(), same)) unique.push_back(std::move(candidate));
  }
  if (unique.size() > 8) unique.resize(8);
  result.candidates = std::move(unique);
  result.surviving_option_count = static_cast<int>(result.candidates.size());
  return true;
}

}  // namespace rebuffer
