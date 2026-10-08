#include "transaction.h"

namespace dreamplace {
namespace diff_guided_batch {

TransactionDecision verifyAndCommitDryRun(
    const std::vector<ActionResult>& action_results,
    double min_total_tns_gain) {
  TransactionDecision decision;
  double total_tns_gain = 0.0;
  for (const auto& result : action_results) {
    if (!result.supported) {
      decision.unsupported_action_count += 1;
      continue;
    }
    if (result.accepted) {
      decision.accepted_action_count += 1;
      total_tns_gain += result.actual_delta_tns;
    }
  }

  if (decision.unsupported_action_count > 0) {
    decision.confirmed = false;
    decision.rolled_back = true;
    decision.status = "rolled_back";
    decision.reject_reason = "unsupported_action_kind";
    return decision;
  }
  if (decision.accepted_action_count == 0) {
    decision.confirmed = false;
    decision.rolled_back = false;
    decision.status = "empty";
    decision.reject_reason = "no_accepted_actions";
    return decision;
  }
  if (total_tns_gain < min_total_tns_gain) {
    decision.confirmed = false;
    decision.rolled_back = true;
    decision.status = "rolled_back";
    decision.reject_reason = "min_total_tns_gain_not_met";
    return decision;
  }
  decision.confirmed = true;
  decision.rolled_back = false;
  decision.status = "confirmed";
  decision.reject_reason = "";
  return decision;
}

}  // namespace diff_guided_batch
}  // namespace dreamplace
