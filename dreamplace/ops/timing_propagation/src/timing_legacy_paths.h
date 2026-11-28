// SPDX-License-Identifier: BSD-3-Clause
// Existing slack-based fallback, shared by Tensor and persistent-graph callers.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <unordered_set>
#include <vector>

namespace dreamplace {
namespace timing {

struct LegacyPathTrace {
  std::vector<int64_t> pins, arc_ids;
};

template <typename Predecessors, typename IsStart, typename Slack,
          typename Arrival, typename ArcForPair>
LegacyPathTrace traceLegacySlackPath(int64_t sink, int64_t num_pins,
    int64_t max_depth, double slack_epsilon, const Predecessors& predecessors,
    const IsStart& is_start, const Slack& slack, const Arrival& arrival,
    const ArcForPair& arc_for_pair) {
  LegacyPathTrace result;
  if (sink < 0 || sink >= num_pins) return result;
  const bool use_rise = slack(sink, true) <= slack(sink, false);
  double current_slack = slack(sink, use_rise);
  int64_t current_pin = sink;
  result.pins.push_back(current_pin);
  std::unordered_set<int64_t> visited{current_pin};
  for (int64_t steps = 0; steps < max_depth; ++steps) {
    if (current_pin < 0 || current_pin >= num_pins || is_start(current_pin)) break;
    int64_t best_pred = -1, best_arc = -1;
    double best_slack = std::numeric_limits<double>::infinity();
    double best_gap = std::numeric_limits<double>::infinity();
    double best_arrival = -std::numeric_limits<double>::infinity();
    predecessors(current_pin, [&](int64_t pred) {
      if (pred < 0 || pred >= num_pins) return;
      const double pred_slack = slack(pred, use_rise);
      const double slack_gap = std::abs(pred_slack - current_slack);
      const double pred_arrival = arrival(pred, use_rise);
      bool improved = false;
      if (pred_slack < best_slack - slack_epsilon) {
        improved = true;
      } else if (std::abs(pred_slack - best_slack) <= slack_epsilon) {
        if (slack_gap < best_gap - slack_epsilon) improved = true;
        else if (std::abs(slack_gap - best_gap) <= slack_epsilon &&
                 pred_arrival > best_arrival + slack_epsilon) improved = true;
      }
      if (improved) {
        best_pred = pred; best_slack = pred_slack; best_gap = slack_gap;
        best_arrival = pred_arrival; best_arc = arc_for_pair(pred, current_pin);
      }
    });
    if (best_pred < 0 || visited.count(best_pred)) break;
    result.pins.push_back(best_pred); result.arc_ids.push_back(best_arc);
    visited.insert(best_pred); current_pin = best_pred; current_slack = best_slack;
  }
  std::reverse(result.pins.begin(), result.pins.end());
  std::reverse(result.arc_ids.begin(), result.arc_ids.end());
  return result;
}

}  // namespace timing
}  // namespace dreamplace
