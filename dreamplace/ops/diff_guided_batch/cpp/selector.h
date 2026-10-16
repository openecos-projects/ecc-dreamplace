#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "action.h"
#include "conflict_precompute.h"

namespace dreamplace {
namespace diff_guided_batch {

struct SelectorConfig {
  int32_t max_batch_size = 64;
  int32_t parallel_worker_count = 1;
  std::string parallel_strategy = "serial_greedy";
  bool use_same_instance_conflict = true;
  bool use_endpoint_conflict = true;
  bool use_top_path_conflict = true;
  bool use_static_component_conflict = true;
  std::string residual_budget_filter_mode = "off";
  std::string residual_budget_score_mode = "off";
};

struct SelectorSummary {
  int32_t proposal_seed_count = 0;
  int32_t selected_batch_size = 0;
  int32_t remaining_seed_count = 0;
  int32_t conflict_reject_count = 0;
  int32_t component_conflict_count = 0;
  int32_t parallel_worker_count = 1;
  double predicted_batch_delta_obj = 0.0;
  double predicted_batch_delta_tns = 0.0;
  double cpp_selector_ms = 0.0;
  std::string parallel_strategy = "serial_greedy";
  int32_t component_count = 0;
  int32_t max_component_size = 0;
  int64_t csr_edge_count = 0;
  int32_t bitset_block_count = 0;
  double precompute_build_ms = 0.0;
  int32_t residual_budget_reject_count = 0;
  std::string selector_residual_budget_filter_mode = "off";
  std::string selector_residual_budget_score_mode = "off";
  int32_t residual_budget_score_reorder_count = 0;
  std::string residual_budget_source;
  std::string residual_budget_score_source;
  std::string residual_snapshot_source;
  std::string conflict_precompute_source;
  int32_t conflict_precompute_version = 1;
  std::string conflict_precompute_input_hash;
  bool enabled_same_instance_conflict = true;
  bool enabled_endpoint_conflict = true;
  bool enabled_top_path_conflict = true;
  bool enabled_static_component_conflict = true;
  std::vector<int64_t> selected_action_ids;
  std::vector<int64_t> remaining_action_ids;
};

SelectorSummary selectConflictFreeBatch(
    const std::vector<ActionProposal>& proposals,
    const std::vector<ActionConflictFootprint>& footprints,
    const ConflictPrecompute& conflict_precompute,
    const SelectorConfig& config);

}  // namespace diff_guided_batch
}  // namespace dreamplace
