// SPDX-License-Identifier: BSD-3-Clause
// Shared hard-TP path selection/backtrace. Inputs borrow their owner's storage.
#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace dreamplace {
namespace timing {

struct SetupPathState {
  int64_t pin, test_id, transition;
  double slack;
};

struct SetupPathTrace {
  std::vector<int64_t> pins, transitions, arc_ids;
  uint8_t valid = 0;
  int64_t invalid_reason = 0;
  double max_residual_ps = 0.0;
};

struct SetupPathBatch {
  std::vector<SetupPathState> states;
  std::vector<SetupPathTrace> traces;
  int64_t failing_state_count = 0;
};

struct SetupPathTimingView {
  const double *rise_aat, *fall_aat, *net_delay_rise, *net_delay_fall;
  const double *delay_rr, *delay_fr, *delay_rf, *delay_ff;
};

struct SetupPathPredecessor {
  int64_t pin, arc_id, sense;
  // Tensor callers use (0, flat arc id). Mutable graphs use current cell level
  // and stable slot; this preserves flat-arc tie order after relevelization.
  // Net arcs precede cell arcs in a tie and use (-1, -1).
  std::pair<int64_t, int64_t> order;
};

inline void appendSetupPathStates(std::vector<SetupPathState>& states,
                                 int64_t pin, int64_t test_id,
                                 double rise_slack, double fall_slack) {
  if (std::isfinite(rise_slack) && rise_slack < 0.0) {
    states.push_back({pin, test_id, 0, rise_slack});
  }
  if (std::isfinite(fall_slack) && fall_slack < 0.0) {
    states.push_back({pin, test_id, 1, fall_slack});
  }
}

// visit_predecessors(pin, visitor) must visit in canonical cell-then-net order.
// The epsilon comparison is deliberately identical to the tensor extractor.
template <typename VisitPredecessors, typename IsStart>
SetupPathBatch extractSetupPaths(std::vector<SetupPathState> states,
                                const SetupPathTimingView& timing,
                                int64_t num_pins, int64_t global_k,
                                int64_t max_depth, double residual_tolerance_ps,
                                const VisitPredecessors& visit_predecessors,
                                const IsStart& is_start) {
  std::sort(states.begin(), states.end(), [](const auto& lhs, const auto& rhs) {
    return std::tie(lhs.slack, lhs.transition, lhs.pin, lhs.test_id) <
           std::tie(rhs.slack, rhs.transition, rhs.pin, rhs.test_id);
  });
  SetupPathBatch batch;
  batch.failing_state_count = static_cast<int64_t>(states.size());
  if (global_k > 0 && static_cast<int64_t>(states.size()) > global_k) {
    states.resize(static_cast<size_t>(global_k));
  }
  batch.states = std::move(states);
  batch.traces.resize(batch.states.size());
  if (max_depth <= 0) max_depth = std::max<int64_t>(1, num_pins);

#ifdef _OPENMP
#pragma omp parallel for schedule(static)
#endif
  for (int64_t state_idx = 0;
       state_idx < static_cast<int64_t>(batch.states.size()); ++state_idx) {
    const auto& state = batch.states[static_cast<size_t>(state_idx)];
    auto& result = batch.traces[static_cast<size_t>(state_idx)];
    int64_t current_pin = state.pin, current_transition = state.transition;
    result.pins.push_back(current_pin);
    result.transitions.push_back(current_transition);
    std::unordered_set<int64_t> visited{current_pin};
    bool reached_start = false;
    for (int64_t depth = 0; depth <= max_depth; ++depth) {
      const bool current_is_start = is_start(current_pin);
      if (depth == max_depth) {
        if (current_is_start) reached_start = true;
        else result.invalid_reason = 3;
        break;
      }
      struct Candidate {
        bool defined = false;
        int64_t pred_pin = -1, pred_transition = -1, arc_id = -2;
        std::pair<int64_t, int64_t> order;
        double aat = -std::numeric_limits<double>::infinity();
        double residual = std::numeric_limits<double>::infinity();
      } best;
      const double current_aat = current_transition == 0
          ? timing.rise_aat[current_pin] : timing.fall_aat[current_pin];
      auto consider = [&](const SetupPathPredecessor& edge,
                          int64_t pred_transition, double candidate_aat) {
        if (!std::isfinite(candidate_aat)) return;
        const double residual = std::abs(current_aat - candidate_aat);
        const double epsilon = 1.0e-12;
        const bool better = !best.defined ||
            candidate_aat > best.aat + epsilon ||
            (std::abs(candidate_aat - best.aat) <= epsilon &&
             std::tie(residual, edge.order, edge.pin, pred_transition) <
             std::tie(best.residual, best.order, best.pred_pin,
                      best.pred_transition));
        if (better) {
          best = {true, edge.pin, pred_transition, edge.arc_id, edge.order,
                  candidate_aat, residual};
        }
      };
      visit_predecessors(current_pin, [&](const SetupPathPredecessor& edge) {
        const auto pred = edge.pin, arc = edge.arc_id;
        if (arc == -1) {
          const double pred_aat = current_transition == 0
              ? timing.rise_aat[pred] : timing.fall_aat[pred];
          const double delay = current_transition == 0
              ? timing.net_delay_rise[current_pin] : timing.net_delay_fall[current_pin];
          consider(edge, current_transition, pred_aat + delay);
        } else if (current_transition == 0) {
          if (edge.sense >= 0) consider(edge, 0, timing.rise_aat[pred] + timing.delay_rr[arc]);
          if (edge.sense <= 0) consider(edge, 1, timing.fall_aat[pred] + timing.delay_fr[arc]);
        } else {
          if (edge.sense >= 0) consider(edge, 1, timing.fall_aat[pred] + timing.delay_ff[arc]);
          if (edge.sense <= 0) consider(edge, 0, timing.rise_aat[pred] + timing.delay_rf[arc]);
        }
      });
      // A sequential output can have an in-graph async launch predecessor.
      if (current_is_start && (!best.defined || best.residual > residual_tolerance_ps)) {
        reached_start = true;
        break;
      }
      if (!best.defined) {
        result.invalid_reason = 1;
        break;
      }
      if (visited.find(best.pred_pin) != visited.end()) {
        result.invalid_reason = 2;
        break;
      }
      result.max_residual_ps = std::max(result.max_residual_ps, best.residual);
      if (best.residual > residual_tolerance_ps && result.invalid_reason == 0) {
        result.invalid_reason = 4;
      }
      result.pins.push_back(best.pred_pin);
      result.transitions.push_back(best.pred_transition);
      result.arc_ids.push_back(best.arc_id);
      visited.insert(best.pred_pin);
      current_pin = best.pred_pin;
      current_transition = best.pred_transition;
    }
    result.valid = reached_start && result.invalid_reason == 0;
    std::reverse(result.pins.begin(), result.pins.end());
    std::reverse(result.transitions.begin(), result.transitions.end());
    std::reverse(result.arc_ids.begin(), result.arc_ids.end());
  }
  return batch;
}

}  // namespace timing
}  // namespace dreamplace
