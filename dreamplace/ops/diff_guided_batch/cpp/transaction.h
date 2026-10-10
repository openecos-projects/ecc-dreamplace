#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "action.h"

namespace dreamplace {
namespace diff_guided_batch {

struct TransactionDecision {
  bool confirmed = false;
  bool rolled_back = false;
  int32_t accepted_action_count = 0;
  int32_t unsupported_action_count = 0;
  std::string status;
  std::string reject_reason;
};

TransactionDecision verifyAndCommitDryRun(
    const std::vector<ActionResult>& action_results,
    double min_total_tns_gain = 0.0);

}  // namespace diff_guided_batch
}  // namespace dreamplace
