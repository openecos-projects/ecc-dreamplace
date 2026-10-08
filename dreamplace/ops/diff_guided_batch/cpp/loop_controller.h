#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "action.h"
#include "conflict_precompute.h"
#include "opensta_trial.h"
#include "selector.h"
#include "transaction.h"

namespace dreamplace {
namespace diff_guided_batch {

struct LoopControllerConfig {
  int32_t max_batch_size = 64;
  int32_t max_loop_batches = 16;
  double min_recent_tns_gain = 0.0;
  int32_t parallel_worker_count = 1;
  bool dry_run_only = true;
};

struct LoopControllerSummary {
  std::string status = "dry_run_only";
  std::string opensta_status = "not_integrated";
  std::string cxx_batch_loop_status = "dry_run_only";
  int32_t proposal_seed_count = 0;
  int32_t loop_count = 0;
  int32_t selected_batch_count = 0;
  int32_t accepted_action_count = 0;
  int32_t rejected_action_count = 0;
  int32_t confirmed_batch_count = 0;
  int32_t rolled_back_batch_count = 0;
  double total_actual_delta_tns = 0.0;
  std::vector<SelectorSummary> selector_summaries;
  std::vector<TransactionDecision> transaction_decisions;
};

LoopControllerSummary runDiffGuidedBatchLoopDryRun(
    const std::vector<ActionProposal>& proposals,
    const std::vector<ActionConflictFootprint>& footprints,
    const ConflictPrecompute& conflict_precompute,
    const LoopControllerConfig& config);

}  // namespace diff_guided_batch
}  // namespace dreamplace
