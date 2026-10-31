// SPDX-License-Identifier: BSD-3-Clause
#include "rebuffer_tree_search.hh"

#include <algorithm>
#include <cmath>
#include "rebuffer_tree_annotation.hh"

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
    const CurrentDriverTree& source, TreeVgdpResult& result, std::string& reason)
{
  if (!planCurrentDriverTree(design, timing, liberty, characterization, config,
                             source, result, reason)) return false;
  result.timing_round_count = 1;
  result.round_source_tree_digests.push_back(source.tree_digest);
  result.round_source_buffer_counts.push_back(0);
  auto score = [&](const std::vector<BufferAction>& actions, TreeVgdpCandidate& candidate) {
    if (actions.size() > static_cast<std::size_t>(config.max_actions)) return false;
    CurrentDriverTree tree;
    std::string rejected;
    if (!applyCurrentDriverTreeBufferActions(source, liberty, actions, tree, rejected)) return false;
    auto scoring_config = config;
    scoring_config.insert_buffers = false;
    TreeVgdpResult scored;
    if (!planCurrentDriverTree(design, timing, liberty, characterization, scoring_config,
                               tree, scored, rejected)) return false;
    candidate = scored.candidates.front();
    candidate.actions = actions;
    candidate.buffer_count = static_cast<int>(actions.size());
    return true;
  };

  // Re-score the materialized tree as well as DP options. Integer DBU sites
  // and concrete buffer cell IDs must obey the same model used by recovery.
  auto raw = std::move(result.candidates);
  for (const auto& candidate : raw) {
    TreeVgdpCandidate scored;
    if (score(candidate.actions, scored)) result.candidates.push_back(std::move(scored));
  }
  if (result.candidates.empty()) {
    reason = "tree_no_materializable_candidate";
    return false;
  }
  std::stable_sort(result.candidates.begin(), result.candidates.end(), better);
  auto incumbent = result.candidates.front();
  for (int round = 1; round < config.tree_timing_rounds && config.insert_buffers; ++round) {
    CurrentDriverTree tree;
    if (!applyCurrentDriverTreeBufferActions(source, liberty, incumbent.actions, tree, reason)) return false;
    TreeVgdpResult next;
    std::string no_improvement;
    ++result.timing_round_count;
    result.round_source_tree_digests.push_back(tree.tree_digest);
    result.round_source_buffer_counts.push_back(incumbent.buffer_count);
    if (!planCurrentDriverTree(design, timing, liberty, characterization, config,
                               tree, next, no_improvement)) break;
    result.generated_option_count += next.generated_option_count;
    result.pruned_option_count += next.pruned_option_count;
    result.long_wire_edge_count += next.long_wire_edge_count;
    result.wire_site_count += next.wire_site_count;
    result.electrical_rejected_count += next.electrical_rejected_count;
    auto best = incumbent;
    for (const auto& addition : next.candidates) {
      if (addition.actions.empty()) continue;
      CurrentDriverTree proposed;
      if (!applyCurrentDriverTreeBufferActions(tree, liberty, addition.actions, proposed, no_improvement)) continue;
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
      for (const auto& buffer : characterization.filtered_buffers) {
        if (buffer.cell_index < 0 || buffer.cell_index >= static_cast<int>(liberty.cells.size())) continue;
        const auto& cell = liberty.cells[buffer.cell_index];
        if (!cell.is_buffer || cell.main_id != liberty.cells[old_cell].main_id
            || cell.area >= liberty.cells[old_cell].area) continue;
        auto rewritten = incumbent.actions;
        rewritten[index].buffer_cell_index = buffer.cell_index;
        probe(std::move(rewritten));
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
