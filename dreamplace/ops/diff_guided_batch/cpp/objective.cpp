#include "diff_guided_batch/cpp/objective.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

namespace dreamplace {
namespace diff_guided_batch {

double actualDeltaObj(double actual_delta_tns,
                      double leakage_delta,
                      double leakage_weight)
{
  return actual_delta_tns - leakage_weight * std::max(0.0, leakage_delta);
}

double metricDeltaObj(double tns_delta,
                      double slew_vio_delta,
                      double cap_vio_delta,
                      double leakage_delta,
                      double slew_weight,
                      double cap_weight,
                      double leakage_weight)
{
  return tns_delta
         - slew_weight * slew_vio_delta
         - cap_weight * cap_vio_delta
         - leakage_weight * std::max(0.0, leakage_delta);
}

double localResidualActualDeltaObj(double actual_delta_tns,
                                   double leakage_delta,
                                   double leakage_weight,
                                   double local_delta_slew_penalty,
                                   double local_delta_cap_penalty,
                                   double local_delta_slew_violation_improvement,
                                   double local_delta_cap_violation_improvement)
{
  return actual_delta_tns + local_delta_slew_violation_improvement + local_delta_cap_violation_improvement
         - local_delta_slew_penalty - local_delta_cap_penalty
         - leakage_weight * std::max(0.0, leakage_delta);
}

double localResidualBudget(const CompactBridgeActionProposal& action)
{
  if (action.local_delta_delay_ps > 0.0) {
    return action.local_delta_delay_ps;
  }
  if (action.predicted_improvement > 0.0) {
    return action.predicted_improvement;
  }
  if (action.predicted_delta_obj < 0.0) {
    return -action.predicted_delta_obj;
  }
  return 0.0;
}

double trialStepScale(const CompactBridgeActionProposal& action, int trial_step)
{
  const int seed_step = action.seed_step != 0
                            ? action.seed_step
                            : action.target_size_idx - action.current_size_idx;
  const double seed_abs_step = static_cast<double>(std::max(1, std::abs(seed_step)));
  const double trial_abs_step = static_cast<double>(std::max(1, std::abs(trial_step)));
  return trial_abs_step / seed_abs_step;
}

double localDeltaSlewPenalty(const CompactBridgeActionProposal& action,
                             int trial_step,
                             double penalty_weight)
{
  return penalty_weight
         * std::max(0.0, action.local_delta_slew_ps)
         * trialStepScale(action, trial_step);
}

double localDeltaCapPenalty(const CompactBridgeActionProposal& action,
                            int trial_step,
                            double penalty_weight)
{
  return penalty_weight
         * std::max(0.0, action.local_delta_cap)
         * trialStepScale(action, trial_step);
}

double localResidualTrialDelta(const CompactBridgeActionProposal& action,
                               int trial_step,
                               double residual_budget)
{
  const double base_gain = localResidualBudget(action);
  const double estimated_slack_delta =
      std::max(0.0, base_gain * trialStepScale(action, trial_step));
  const double old_violation = std::max(0.0, residual_budget);
  const double new_violation = std::max(0.0, residual_budget - estimated_slack_delta);
  return old_violation - new_violation;
}

double candidateDeltaForTrial(const CompactBridgeActionProposal& trial_action,
                              double residual_budget)
{
  const double candidate_timing_gain = std::max(0.0, trial_action.local_delta_delay_ps);
  return std::min(candidate_timing_gain, std::max(0.0, residual_budget));
}

double candidateLocalDeltaSlewPenalty(const CompactBridgeActionProposal& trial_action,
                                      double penalty_weight)
{
  return penalty_weight * std::max(0.0, trial_action.local_delta_slew_ps);
}

double candidateLocalDeltaCapPenalty(double local_delta_cap,
                                     double penalty_weight)
{
  return penalty_weight * std::max(0.0, local_delta_cap);
}

double localSlewCapViolationImprovement(double old_actual,
                                        double new_actual,
                                        double old_limit,
                                        double new_limit,
                                        double objective_weight)
{
  if (objective_weight <= 0.0 || old_limit <= 0.0 || new_limit <= 0.0
      || !std::isfinite(old_actual) || !std::isfinite(new_actual)) {
    return 0.0;
  }
  const double old_violation = std::max(0.0, old_actual - old_limit);
  const double new_violation = std::max(0.0, new_actual - new_limit);
  return objective_weight * std::max(0.0, old_violation - new_violation);
}

double penaltyFromSlackDegradation(double slack_ps, double degradation_ps)
{
  if (!std::isfinite(slack_ps) || !std::isfinite(degradation_ps) || degradation_ps <= 0.0) {
    return 0.0;
  }
  const double old_violation = std::max(0.0, -slack_ps);
  const double new_violation = std::max(0.0, -(slack_ps - degradation_ps));
  return std::max(0.0, new_violation - old_violation);
}

double slackViolationDeltaFromDelayDelta(double old_slack_ps, double delta_delay_ps)
{
  if (!std::isfinite(old_slack_ps) || !std::isfinite(delta_delay_ps)) {
    return 0.0;
  }
  const double new_slack_ps = old_slack_ps - delta_delay_ps;
  const double old_violation = std::max(0.0, -old_slack_ps);
  const double new_violation = std::max(0.0, -new_slack_ps);
  return old_violation - new_violation;
}

}  // namespace diff_guided_batch
}  // namespace dreamplace
