#include "opensta_trial.h"

#include <algorithm>

namespace dreamplace {
namespace diff_guided_batch {

std::vector<ActionResult> evaluateActionsDryRun(
    const std::vector<ActionProposal>& proposals,
    const TrialConfig& config) {
  std::vector<ActionResult> results;
  results.reserve(proposals.size());
  (void)config;
  for (const auto& proposal : proposals) {
    if (!isSupportedV1(proposal.action_kind)) {
      results.push_back(unsupportedResult(proposal));
      continue;
    }
    ActionResult result;
    result.action_id = proposal.action_id;
    result.action_kind = proposal.action_kind;
    result.supported = true;
    result.accepted = proposal.predicted_delta_obj < 0.0;
    result.status = result.accepted ? "dry_run_accepted" : "dry_run_rejected";
    result.reject_reason = result.accepted ? "" : "non_improving_predicted_delta";
    result.actual_delta_tns = -proposal.predicted_delta_obj;
    result.actual_delta_wns = 0.0;
    results.push_back(result);
  }
  return results;
}

}  // namespace diff_guided_batch
}  // namespace dreamplace
