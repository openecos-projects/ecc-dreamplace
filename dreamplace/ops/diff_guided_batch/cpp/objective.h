#pragma once

#include "diff_guided_batch/cpp/action_types.h"

namespace dreamplace {
namespace diff_guided_batch {

double actualDeltaObj(double actual_delta_tns,
                      double leakage_delta,
                      double leakage_weight);
double metricDeltaObj(double tns_delta,
                      double slew_vio_delta,
                      double cap_vio_delta,
                      double leakage_delta,
                      double slew_weight,
                      double cap_weight,
                      double leakage_weight);
double localResidualActualDeltaObj(double actual_delta_tns,
                                   double leakage_delta,
                                   double leakage_weight,
                                   double local_delta_slew_penalty,
                                   double local_delta_cap_penalty,
                                   double local_delta_slew_violation_improvement = 0.0,
                                   double local_delta_cap_violation_improvement = 0.0);
double localResidualBudget(const CompactBridgeActionProposal& action);
double trialStepScale(const CompactBridgeActionProposal& action, int trial_step);
double localDeltaSlewPenalty(const CompactBridgeActionProposal& action,
                             int trial_step,
                             double penalty_weight);
double localDeltaCapPenalty(const CompactBridgeActionProposal& action,
                            int trial_step,
                            double penalty_weight);
double localResidualTrialDelta(const CompactBridgeActionProposal& action,
                               int trial_step,
                               double residual_budget);
double candidateDeltaForTrial(const CompactBridgeActionProposal& trial_action,
                              double residual_budget);
double candidateLocalDeltaSlewPenalty(const CompactBridgeActionProposal& trial_action,
                                      double penalty_weight);
double candidateLocalDeltaCapPenalty(double local_delta_cap,
                                     double penalty_weight);
double localSlewCapViolationImprovement(double old_actual,
                                        double new_actual,
                                        double old_limit,
                                        double new_limit,
                                        double objective_weight);
double penaltyFromSlackDegradation(double slack_ps, double degradation_ps);
double slackViolationDeltaFromDelayDelta(double old_slack_ps, double delta_delay_ps);

}  // namespace diff_guided_batch
}  // namespace dreamplace
