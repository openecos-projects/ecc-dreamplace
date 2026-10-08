#pragma once

#include <vector>

#include "action.h"

namespace dreamplace {
namespace diff_guided_batch {

struct TrialConfig {
  int32_t max_up_step = 3;
  int32_t max_down_step = 0;
  int32_t parallel_worker_count = 1;
};

std::vector<ActionResult> evaluateActionsDryRun(
    const std::vector<ActionProposal>& proposals,
    const TrialConfig& config);

}  // namespace diff_guided_batch
}  // namespace dreamplace
