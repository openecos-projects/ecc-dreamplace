#include "loop_controller.h"

#include <algorithm>
#include <unordered_set>

namespace dreamplace {
namespace diff_guided_batch {

LoopControllerSummary runDiffGuidedBatchLoopDryRun(
    const std::vector<ActionProposal>& proposals,
    const std::vector<ActionConflictFootprint>& footprints,
    const ConflictPrecompute& conflict_precompute,
    const LoopControllerConfig& config) {
  LoopControllerSummary summary;
  summary.proposal_seed_count = static_cast<int32_t>(proposals.size());

  std::vector<ActionProposal> remaining_proposals = proposals;
  std::vector<ActionConflictFootprint> remaining_footprints = footprints;
  const int32_t max_loop_batches = std::max(1, config.max_loop_batches);

  for (int32_t loop_id = 0;
       loop_id < max_loop_batches && !remaining_proposals.empty();
       ++loop_id) {
    SelectorConfig selector_config;
    selector_config.max_batch_size = std::max(1, config.max_batch_size);
    selector_config.parallel_worker_count = std::max(1, config.parallel_worker_count);
    selector_config.parallel_strategy = "serial_greedy";

    SelectorSummary selector_summary = selectConflictFreeBatch(
        remaining_proposals,
        remaining_footprints,
        conflict_precompute,
        selector_config);
    summary.selector_summaries.push_back(selector_summary);
    if (selector_summary.selected_action_ids.empty()) {
      break;
    }

    std::unordered_set<int64_t> selected_ids(
        selector_summary.selected_action_ids.begin(),
        selector_summary.selected_action_ids.end());
    std::vector<ActionProposal> selected_proposals;
    std::vector<ActionConflictFootprint> next_footprints;
    std::vector<ActionProposal> next_proposals;
    for (size_t idx = 0; idx < remaining_proposals.size(); ++idx) {
      const auto& proposal = remaining_proposals[idx];
      if (selected_ids.find(proposal.action_id) != selected_ids.end()) {
        selected_proposals.push_back(proposal);
      } else {
        next_proposals.push_back(proposal);
        if (idx < remaining_footprints.size()) {
          next_footprints.push_back(remaining_footprints[idx]);
        }
      }
    }

    TrialConfig trial_config;
    trial_config.parallel_worker_count = std::max(1, config.parallel_worker_count);
    std::vector<ActionResult> action_results =
        evaluateActionsDryRun(selected_proposals, trial_config);
    TransactionDecision decision =
        verifyAndCommitDryRun(action_results, config.min_recent_tns_gain);

    summary.loop_count += 1;
    summary.selected_batch_count += 1;
    summary.transaction_decisions.push_back(decision);
    summary.accepted_action_count += decision.accepted_action_count;
    if (decision.confirmed) {
      summary.confirmed_batch_count += 1;
    }
    if (decision.rolled_back) {
      summary.rolled_back_batch_count += 1;
    }
    for (const auto& result : action_results) {
      if (result.accepted) {
        summary.total_actual_delta_tns += result.actual_delta_tns;
      } else {
        summary.rejected_action_count += 1;
      }
    }

    if (decision.confirmed &&
        config.min_recent_tns_gain > 0.0 &&
        summary.total_actual_delta_tns < config.min_recent_tns_gain) {
      break;
    }
    remaining_proposals = std::move(next_proposals);
    remaining_footprints = std::move(next_footprints);
  }

  return summary;
}

}  // namespace diff_guided_batch
}  // namespace dreamplace
