#include "diff_guided_batch/cpp/config.h"

#include <algorithm>

namespace dreamplace {
namespace diff_guided_batch {
namespace {

int pyIntOrDefault(const py::dict& value, const char* key, int fallback = -1)
{
  if (!value.contains(key) || value[key].is_none()) {
    return fallback;
  }
  return value[key].cast<int>();
}

double pyDoubleOrDefault(const py::dict& value, const char* key, double fallback = 0.0)
{
  if (!value.contains(key) || value[key].is_none()) {
    return fallback;
  }
  return value[key].cast<double>();
}

bool pyBoolOrDefault(const py::dict& value, const char* key, bool fallback = false)
{
  if (!value.contains(key) || value[key].is_none()) {
    return fallback;
  }
  return value[key].cast<bool>();
}

std::string pyStringOrDefault(const py::dict& value,
                              const char* key,
                              const std::string& fallback = "")
{
  if (!value.contains(key) || value[key].is_none()) {
    return fallback;
  }
  return value[key].cast<std::string>();
}

}  // namespace

int selectorWorkerCount(const py::dict& config)
{
  return std::max(
      1,
      pyIntOrDefault(
          config,
          "cpp_parallel_workers",
          pyIntOrDefault(config, "diff_guided_batch_cpp_parallel_workers", 1)));
}

int trialWorkerCount(const py::dict& config)
{
  return std::max(
      1,
      pyIntOrDefault(
          config,
          "trial_worker_count",
          pyIntOrDefault(config, "precise_improvement_parallel_workers", 1)));
}

std::string acceptMode(const py::dict& config)
{
  return pyStringOrDefault(
      config,
      "accept_mode",
      pyStringOrDefault(config, "diff_guided_batch_accept_mode", "batch"));
}

double tnsPowerLeakageWeight(const py::dict& config)
{
  return std::max(
      0.0,
      pyDoubleOrDefault(
          config,
          "tns_power_leakage_weight",
          pyDoubleOrDefault(config, "diff_guided_batch_tns_power_leakage_weight", 0.0)));
}

bool localObjectiveAlignGlobal(const py::dict& config)
{
  return pyBoolOrDefault(
      config,
      "local_objective_align_global",
      pyBoolOrDefault(config, "diff_guided_batch_local_objective_align_global", true));
}

bool noopBaseline(const py::dict& config)
{
  return pyBoolOrDefault(
      config,
      "noop_baseline",
      pyBoolOrDefault(config, "diff_guided_batch_noop_baseline", false));
}

bool rejectNonpositiveTrial(const py::dict& config)
{
  return pyBoolOrDefault(
      config,
      "reject_nonpositive_trial",
      pyBoolOrDefault(config, "diff_guided_batch_reject_nonpositive_trial", false));
}

bool forceAcceptSelectedActions(const py::dict& config)
{
  return pyBoolOrDefault(
      config,
      "force_accept_selected_actions",
      pyBoolOrDefault(config, "diff_guided_batch_force_accept_selected_actions", false));
}

bool singleActionAudit(const py::dict& config)
{
  return pyBoolOrDefault(
      config,
      "single_action_audit",
      pyBoolOrDefault(config, "diff_guided_batch_single_action_audit", false));
}

int singleActionAuditLimit(const py::dict& config)
{
  return std::max(
      0,
      pyIntOrDefault(
          config,
          "single_action_audit_limit",
          pyIntOrDefault(config, "diff_guided_batch_single_action_audit_limit", 64)));
}

double nonpositiveTrialEps(const py::dict& config)
{
  return std::max(
      0.0,
      pyDoubleOrDefault(
          config,
          "nonpositive_trial_eps",
          pyDoubleOrDefault(config, "diff_guided_batch_nonpositive_trial_eps", 0.0)));
}

double localDeltaSlewPenaltyWeight(const py::dict& config)
{
  return std::max(
      0.0,
      pyDoubleOrDefault(
          config,
          "local_delta_slew_penalty_weight",
          pyDoubleOrDefault(config, "diff_guided_batch_local_delta_slew_penalty_weight", 0.0)));
}

double localDeltaCapPenaltyWeight(const py::dict& config)
{
  return std::max(
      0.0,
      pyDoubleOrDefault(
          config,
          "local_delta_cap_penalty_weight",
          pyDoubleOrDefault(config, "diff_guided_batch_local_delta_cap_penalty_weight", 0.0)));
}

double faninPenaltySensitivityPsPerPf(const py::dict& config)
{
  return std::max(
      0.0,
      pyDoubleOrDefault(
          config,
          "fanin_penalty_ps_per_pf",
          pyDoubleOrDefault(config, "diff_guided_batch_fanin_penalty_ps_per_pf", 1000.0)));
}

std::string slackDeltaEvaluatorMode(const py::dict& config)
{
  return pyStringOrDefault(
      config,
      "slack_delta_evaluator_mode",
      pyStringOrDefault(config, "diff_guided_batch_slack_delta_evaluator_mode", "heuristic"));
}

std::string slackDeltaWeightMode(const py::dict& config)
{
  const std::string mode = pyStringOrDefault(
      config,
      "slack_delta_weight_mode",
      pyStringOrDefault(config, "diff_guided_batch_slack_delta_weight_mode", "raw_slack_delta"));
  if (mode == "npath_weighted_tns") {
    return mode;
  }
  return "raw_slack_delta";
}

double npathWeightCap(const py::dict& config)
{
  return std::max(
      1.0,
      pyDoubleOrDefault(
          config,
          "npath_weight_cap",
          pyDoubleOrDefault(config, "diff_guided_batch_npath_weight_cap", 128.0)));
}

bool laneContains(const std::string& lane, const std::string& term)
{
  if (term == "timing") {
    return lane == "timing_only" || lane == "timing_slew"
           || lane == "timing_cap" || lane == "timing_slew_cap"
           || lane == "timing_slew_cap_leakage";
  }
  if (term == "slew") {
    return lane == "timing_slew" || lane == "timing_slew_cap"
           || lane == "timing_slew_cap_leakage";
  }
  if (term == "cap") {
    return lane == "timing_cap" || lane == "timing_slew_cap"
           || lane == "timing_slew_cap_leakage";
  }
  if (term == "leakage") {
    return lane == "timing_slew_cap_leakage";
  }
  return false;
}

EffectiveLocalWeight effectiveLocalDeltaSlewPenaltyWeight(const py::dict& config)
{
  const double explicit_weight = localDeltaSlewPenaltyWeight(config);
  const bool align_global = localObjectiveAlignGlobal(config);
  const std::string lane = pyStringOrDefault(config, "timing_objective_lane", "timing_only");
  const double global_weight = std::max(0.0, pyDoubleOrDefault(config, "timing_slew_weight", 1.0));
  if (align_global && explicit_weight <= 0.0 && laneContains(lane, "slew")) {
    return {global_weight, "timing_objective_lane_default"};
  }
  return {explicit_weight, align_global ? "explicit_or_lane_disabled" : "explicit_local_weight"};
}

EffectiveLocalWeight effectiveLocalDeltaCapPenaltyWeight(const py::dict& config)
{
  const double explicit_weight = localDeltaCapPenaltyWeight(config);
  const bool align_global = localObjectiveAlignGlobal(config);
  const std::string lane = pyStringOrDefault(config, "timing_objective_lane", "timing_only");
  const double global_weight = std::max(0.0, pyDoubleOrDefault(config, "timing_cap_weight", 1.0));
  if (align_global && explicit_weight <= 0.0 && laneContains(lane, "cap")) {
    return {global_weight, "timing_objective_lane_default"};
  }
  return {explicit_weight, align_global ? "explicit_or_lane_disabled" : "explicit_local_weight"};
}

double localResidualFeedbackScale(const py::dict& config)
{
  const double scale = pyDoubleOrDefault(
      config,
      "local_residual_feedback_scale",
      pyDoubleOrDefault(config, "diff_guided_batch_local_residual_feedback_scale", 1.0));
  return std::min(1.0, std::max(0.0, scale));
}

std::string localResidualFeedbackPolicy(const py::dict& config)
{
  return pyStringOrDefault(
      config,
      "local_residual_feedback_policy",
      pyStringOrDefault(
          config,
          "diff_guided_batch_local_residual_feedback_policy",
          "soft_decay"));
}

double localResidualFeedbackDecay(const py::dict& config)
{
  const double decay = pyDoubleOrDefault(
      config,
      "local_residual_feedback_decay",
      pyDoubleOrDefault(config, "diff_guided_batch_local_residual_feedback_decay", 0.5));
  return std::min(1.0, std::max(0.0, decay));
}

double localResidualMaxGainPerAction(const py::dict& config)
{
  return std::max(
      0.0,
      pyDoubleOrDefault(
          config,
          "local_residual_max_gain_per_action",
          pyDoubleOrDefault(config, "diff_guided_batch_local_residual_max_gain_per_action", 0.0)));
}

double localResidualBudgetDiscount(const py::dict& config)
{
  const double discount = pyDoubleOrDefault(
      config,
      "local_residual_budget_discount",
      pyDoubleOrDefault(config, "diff_guided_batch_local_residual_budget_discount", 1.0));
  return std::min(1.0, std::max(0.0, discount));
}

std::string trialMode(const py::dict& config)
{
  return pyStringOrDefault(
      config,
      "trial_mode",
      pyStringOrDefault(config, "diff_guided_batch_trial_mode", "local_candidate_delta"));
}

int trialMiniBatchSize(const py::dict& config)
{
  return std::max(
      1,
      pyIntOrDefault(
          config,
          "trial_mini_batch_size",
          pyIntOrDefault(config, "diff_guided_batch_trial_mini_batch_size", 4)));
}

}  // namespace diff_guided_batch
}  // namespace dreamplace
