#include "selector.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <unordered_set>

namespace dreamplace {
namespace diff_guided_batch {

namespace {

void addIds(std::unordered_set<int32_t>* seen, const std::vector<int32_t>& ids) {
  for (int32_t id : ids) {
    if (id >= 0) {
      seen->insert(id);
    }
  }
}

bool intersects(const std::unordered_set<int32_t>& seen,
                const std::vector<int32_t>& ids) {
  for (int32_t id : ids) {
    if (id >= 0 && seen.find(id) != seen.end()) {
      return true;
    }
  }
  return false;
}

double firstPositiveBudgetForComponents(const std::vector<int32_t>& component_ids,
                                        const std::vector<double>& budgets) {
  for (int32_t component_id : component_ids) {
    if (component_id >= 0 && component_id < static_cast<int32_t>(budgets.size()) &&
        budgets[component_id] > 0.0) {
      return budgets[component_id];
    }
  }
  return 0.0;
}

bool hasPositiveResidualBudget(const ActionConflictFootprint& footprint,
                               const ConflictPrecompute& conflict_precompute,
                               std::string* source) {
  if (firstPositiveBudgetForComponents(
          footprint.top_path_component_ids,
          conflict_precompute.local_residual_budget_by_top_path_component) > 0.0) {
    if (source != nullptr) {
      *source = "top_path_component_budget";
    }
    return true;
  }
  if (firstPositiveBudgetForComponents(
          footprint.endpoint_component_ids,
          conflict_precompute.local_residual_budget_by_endpoint_component) > 0.0) {
    if (source != nullptr) {
      *source = "endpoint_component_budget";
    }
    return true;
  }
  return false;
}

double residualBudgetForComponents(const std::vector<int32_t>& component_ids,
                                   const std::vector<double>& budgets) {
  double result = 0.0;
  for (int32_t component_id : component_ids) {
    if (component_id >= 0 && component_id < static_cast<int32_t>(budgets.size())) {
      result = std::max(result, budgets[component_id]);
    }
  }
  return result;
}

double residualBudgetForFootprint(const ActionConflictFootprint& footprint,
                                  const ConflictPrecompute& conflict_precompute,
                                  std::string* source) {
  const double top_path_budget = residualBudgetForComponents(
      footprint.top_path_component_ids,
      conflict_precompute.local_residual_budget_by_top_path_component);
  const double endpoint_budget = residualBudgetForComponents(
      footprint.endpoint_component_ids,
      conflict_precompute.local_residual_budget_by_endpoint_component);
  if (top_path_budget > 0.0 && top_path_budget >= endpoint_budget) {
    if (source != nullptr) {
      *source = "top_path_component_budget";
    }
    return top_path_budget;
  }
  if (endpoint_budget > 0.0) {
    if (source != nullptr) {
      *source = "endpoint_component_budget";
    }
    return endpoint_budget;
  }
  return 0.0;
}

double predictedGain(const ActionProposal& proposal) {
  if (proposal.sensitivity_score > 0.0) {
    return proposal.sensitivity_score;
  }
  if (proposal.predicted_delta_obj < 0.0) {
    return -proposal.predicted_delta_obj;
  }
  return 0.0;
}

bool conflictsWithSelected(
    const ActionConflictFootprint& footprint,
    const std::unordered_set<int32_t>& selected_instances,
    const std::unordered_set<int32_t>& selected_nets,
    const std::unordered_set<int32_t>& selected_pins,
    const std::unordered_set<int32_t>& selected_endpoint_components,
    const std::unordered_set<int32_t>& selected_top_path_components,
    const std::unordered_set<int32_t>& selected_static_components,
    const std::unordered_set<int32_t>& selected_physical_bins,
    const SelectorConfig& config) {
  if (config.use_same_instance_conflict && footprint.primary_inst_id >= 0 &&
      selected_instances.find(footprint.primary_inst_id) != selected_instances.end()) {
    return true;
  }
  return (config.use_same_instance_conflict &&
          intersects(selected_instances, footprint.affected_instance_ids)) ||
         (config.use_static_component_conflict &&
          intersects(selected_nets, footprint.affected_net_ids)) ||
         (config.use_static_component_conflict &&
          intersects(selected_pins, footprint.affected_pin_ids)) ||
         (config.use_endpoint_conflict &&
          intersects(selected_endpoint_components, footprint.endpoint_component_ids)) ||
         (config.use_top_path_conflict &&
          intersects(selected_top_path_components, footprint.top_path_component_ids)) ||
         (config.use_static_component_conflict &&
          intersects(selected_static_components, footprint.static_component_ids)) ||
         (config.use_static_component_conflict &&
          intersects(selected_physical_bins, footprint.physical_bin_ids));
}

void markSelected(const ActionConflictFootprint& footprint,
                  std::unordered_set<int32_t>* selected_instances,
                  std::unordered_set<int32_t>* selected_nets,
                  std::unordered_set<int32_t>* selected_pins,
                  std::unordered_set<int32_t>* selected_endpoint_components,
                  std::unordered_set<int32_t>* selected_top_path_components,
                  std::unordered_set<int32_t>* selected_static_components,
                  std::unordered_set<int32_t>* selected_physical_bins) {
  if (footprint.primary_inst_id >= 0) {
    selected_instances->insert(footprint.primary_inst_id);
  }
  addIds(selected_instances, footprint.affected_instance_ids);
  addIds(selected_nets, footprint.affected_net_ids);
  addIds(selected_pins, footprint.affected_pin_ids);
  addIds(selected_endpoint_components, footprint.endpoint_component_ids);
  addIds(selected_top_path_components, footprint.top_path_component_ids);
  addIds(selected_static_components, footprint.static_component_ids);
  addIds(selected_physical_bins, footprint.physical_bin_ids);
}

}  // namespace

SelectorSummary selectConflictFreeBatch(
    const std::vector<ActionProposal>& proposals,
    const std::vector<ActionConflictFootprint>& footprints,
    const ConflictPrecompute& conflict_precompute,
    const SelectorConfig& config) {
  const auto begin = std::chrono::steady_clock::now();
  SelectorSummary summary;
  summary.proposal_seed_count = static_cast<int32_t>(proposals.size());
  summary.parallel_worker_count = std::max(1, config.parallel_worker_count);
  summary.parallel_strategy =
      config.parallel_strategy.empty() ? "serial_greedy" : config.parallel_strategy;
  summary.component_count = conflict_precompute.component_count;
  summary.max_component_size = conflict_precompute.max_component_size;
  summary.csr_edge_count = conflict_precompute.csr_edge_count;
  summary.bitset_block_count = conflict_precompute.bitset_block_count;
  summary.precompute_build_ms = conflict_precompute.build_ms;
  summary.conflict_precompute_source = conflict_precompute.source;
  summary.conflict_precompute_version = conflict_precompute.version;
  summary.conflict_precompute_input_hash = conflict_precompute.input_fingerprint;
  summary.selector_residual_budget_filter_mode =
      config.residual_budget_filter_mode.empty() ? "off" : config.residual_budget_filter_mode;
  summary.selector_residual_budget_score_mode =
      config.residual_budget_score_mode.empty() ? "off" : config.residual_budget_score_mode;
  summary.residual_snapshot_source = conflict_precompute.local_residual_snapshot_source;
  summary.enabled_same_instance_conflict = config.use_same_instance_conflict;
  summary.enabled_endpoint_conflict = config.use_endpoint_conflict;
  summary.enabled_top_path_conflict = config.use_top_path_conflict;
  summary.enabled_static_component_conflict = config.use_static_component_conflict;

  std::unordered_set<int32_t> selected_instances;
  std::unordered_set<int32_t> selected_nets;
  std::unordered_set<int32_t> selected_pins;
  std::unordered_set<int32_t> selected_endpoint_components;
  std::unordered_set<int32_t> selected_top_path_components;
  std::unordered_set<int32_t> selected_static_components;
  std::unordered_set<int32_t> selected_physical_bins;

  const int32_t max_batch_size = std::max(1, config.max_batch_size);
  const size_t limit = std::min(proposals.size(), footprints.size());
  std::vector<size_t> ordered_indices;
  ordered_indices.reserve(limit);
  for (size_t idx = 0; idx < limit; ++idx) {
    ordered_indices.push_back(idx);
  }
  if (summary.selector_residual_budget_score_mode == "clip_predicted_gain") {
    std::vector<double> effective_gains(limit, 0.0);
    for (size_t idx = 0; idx < limit; ++idx) {
      std::string score_source;
      const double budget =
          residualBudgetForFootprint(footprints[idx], conflict_precompute, &score_source);
      const double gain = predictedGain(proposals[idx]);
      effective_gains[idx] = budget > 0.0 ? std::min(gain, budget) : 0.0;
      if (effective_gains[idx] > 0.0 && summary.residual_budget_score_source.empty()) {
        summary.residual_budget_score_source = score_source;
      }
    }
    std::stable_sort(
        ordered_indices.begin(),
        ordered_indices.end(),
        [&effective_gains](size_t left, size_t right) {
          return effective_gains[left] > effective_gains[right];
        });
    for (size_t order_idx = 0; order_idx < ordered_indices.size(); ++order_idx) {
      if (ordered_indices[order_idx] != order_idx) {
        summary.residual_budget_score_reorder_count += 1;
      }
    }
  }
  for (size_t ordered_idx = 0; ordered_idx < ordered_indices.size(); ++ordered_idx) {
    const size_t idx = ordered_indices[ordered_idx];
    const auto& proposal = proposals[idx];
    const auto& footprint = footprints[idx];
    if (summary.selected_batch_size >= max_batch_size) {
      summary.remaining_action_ids.push_back(proposal.action_id);
      continue;
    }
    if (summary.selector_residual_budget_filter_mode == "positive_violation" &&
        !hasPositiveResidualBudget(footprint, conflict_precompute, &summary.residual_budget_source)) {
      summary.residual_budget_reject_count += 1;
      summary.remaining_action_ids.push_back(proposal.action_id);
      continue;
    }

    const bool conflict = conflictsWithSelected(
        footprint,
        selected_instances,
        selected_nets,
        selected_pins,
        selected_endpoint_components,
        selected_top_path_components,
        selected_static_components,
        selected_physical_bins,
        config);
    if (conflict) {
      summary.conflict_reject_count += 1;
      summary.component_conflict_count += 1;
      summary.remaining_action_ids.push_back(proposal.action_id);
      continue;
    }

    markSelected(footprint,
                 &selected_instances,
                 &selected_nets,
                 &selected_pins,
                 &selected_endpoint_components,
                 &selected_top_path_components,
                 &selected_static_components,
                 &selected_physical_bins);
    summary.selected_action_ids.push_back(proposal.action_id);
    summary.selected_batch_size += 1;
    summary.predicted_batch_delta_obj += proposal.predicted_delta_obj;
  }
  for (size_t idx = limit; idx < proposals.size(); ++idx) {
    summary.remaining_action_ids.push_back(proposals[idx].action_id);
  }
  summary.remaining_seed_count =
      static_cast<int32_t>(summary.remaining_action_ids.size());
  const auto end = std::chrono::steady_clock::now();
  summary.cpp_selector_ms =
      std::chrono::duration<double, std::milli>(end - begin).count();
  return summary;
}

}  // namespace diff_guided_batch
}  // namespace dreamplace
