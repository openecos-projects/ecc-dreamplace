#pragma once

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <omp.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "est/EstimateParasitics.h"
#include "odb/db.h"
#include "odb/dbTypes.h"
#include "odb/defin.h"
#include "db_sta/dbSta.hh"
#include "db_sta/dbNetwork.hh"
#include "ord/Design.h"
#include "ord/OpenRoad.hh"
#include "ord/Tech.h"
#include "ord/Timing.h"
#include "ord/ordMain.hh"
#include "sta/Clock.hh"
#include "sta/Corner.hh"
#include "sta/DcalcAnalysisPt.hh"
#include "sta/LeakagePower.hh"
#include "sta/Liberty.hh"
#include "sta/MinMax.hh"
#include "sta/Path.hh"
#include "sta/PathExpanded.hh"
#include "sta/PathEnd.hh"
#include "sta/PortDirection.hh"
#include "sta/PortDelay.hh"
#include "sta/RiseFallMinMax.hh"
#include "sta/Sdc.hh"
#include "sta/Search.hh"
#include "sta/TableModel.hh"
#include "sta/TimingArc.hh"
#include "sta/TimingRole.hh"
#include "sta/Units.hh"
#include "utl/Logger.h"

#include "diff_guided_batch/cpp/action_types.h"
#include "diff_guided_batch/cpp/bridge_action_io.h"
#include "diff_guided_batch/cpp/config.h"
#include "diff_guided_batch/cpp/objective.h"
#include "openroad_place_io_bridge.h"
#include "openroad_place_io_bridge_internal.h"
#include "openroad_place_io_types.h"
#include "openroad_pyplacedb_export.h"
#include "openroad_runtime.h"

namespace py = pybind11;
namespace dgb = dreamplace::diff_guided_batch;

namespace dreamplace {
namespace placeio_openroad {
namespace impl {

using dgb::ActionKind;
using dgb::CompactBridgeAction;
using dgb::CompactBridgeConflictPrecompute;
using dgb::CompactBridgeActionProposal;
using dgb::CompactBridgeActionProposalBuffers;
using dgb::CompactBridgeActionResult;
using dgb::CompactBridgeSelection;
using dgb::DiffGuidedBatchConflictRuleMask;
using DiffGuidedBatchEffectiveLocalWeight = dgb::EffectiveLocalWeight;
using dreamplace::placeio_openroad::NetRecord;
using dreamplace::placeio_openroad::OpenRoadPlaceIOBridge;
using dreamplace::placeio_openroad::PendingPinRecord;
using dreamplace::placeio_openroad::PinRecord;
using dreamplace::placeio_openroad::PyPlaceDB;
using dreamplace::placeio_openroad::OpenRoadRuntime;

inline bool isSignalNet(odb::dbNet* net)
{
  if (net == nullptr || net->isSpecial()) {
    return false;
  }
  const std::string sig_type = net->getSigType().getString();
  return sig_type != "POWER" && sig_type != "GROUND" && sig_type != "CLOCK";
}

inline std::string ioTypeString(odb::dbIoType io_type)
{
  return io_type.getString();
}

inline bool isTimingDriverPin(const PinRecord& pin)
{
  return (pin.is_io && pin.direction == "INPUT")
         || (!pin.is_io && (pin.direction == "OUTPUT" || pin.direction == "INOUT"));
}

inline int findTimingDriverPinId(const std::vector<PendingPinRecord>& pins)
{
  for (int pin_id = 0; pin_id < static_cast<int>(pins.size()); ++pin_id) {
    if (isTimingDriverPin(pins.at(pin_id).pin)) {
      return pin_id;
    }
  }
  return -1;
}

inline void moveTimingDriverPinToFront(std::vector<PendingPinRecord>& pins, int driver_pin_id)
{
  if (driver_pin_id <= 0 || driver_pin_id >= static_cast<int>(pins.size())) {
    return;
  }
  std::rotate(pins.begin(), pins.begin() + driver_pin_id, pins.begin() + driver_pin_id + 1);
}

inline std::string tclQuote(const std::string& value)
{
  return "{" + value + "}";
}

inline std::string clockPinKey(const std::string& inst_name, const std::string& pin_name)
{
  return inst_name + ":" + pin_name;
}

inline ActionKind diffGuidedBatchActionKind(const py::dict& action)
{
  if (!action.contains("action_kind") || action["action_kind"].is_none()) {
    return ActionKind::kSizing;
  }
  return dgb::actionKindFromString(action["action_kind"].cast<std::string>());
}

inline std::string diffGuidedBatchActionKindName(ActionKind kind)
{
  return dgb::actionKindName(kind);
}

inline int pyIntOrDefault(const py::dict& value, const char* key, int fallback = -1)
{
  if (!value.contains(key) || value[key].is_none()) {
    return fallback;
  }
  return value[key].cast<int>();
}

inline int64_t pyInt64OrDefault(const py::dict& value, const char* key, int64_t fallback = -1)
{
  if (!value.contains(key) || value[key].is_none()) {
    return fallback;
  }
  if (py::isinstance<py::str>(value[key])) {
    return fallback;
  }
  return value[key].cast<int64_t>();
}

inline double pyDoubleOrDefault(const py::dict& value, const char* key, double fallback = 0.0)
{
  if (!value.contains(key) || value[key].is_none()) {
    return fallback;
  }
  return value[key].cast<double>();
}

inline bool pyBoolOrDefault(const py::dict& value, const char* key, bool fallback = false)
{
  if (!value.contains(key) || value[key].is_none()) {
    return fallback;
  }
  return value[key].cast<bool>();
}

inline std::string pyStringOrDefault(const py::dict& value,
                              const char* key,
                              const std::string& fallback = "")
{
  if (!value.contains(key) || value[key].is_none()) {
    return fallback;
  }
  return value[key].cast<std::string>();
}

inline std::vector<std::string> pyStringListOrEmpty(const py::dict& value, const char* key)
{
  std::vector<std::string> result;
  if (!value.contains(key) || value[key].is_none()) {
    return result;
  }
  try {
    result = value[key].cast<std::vector<std::string>>();
  } catch (const py::cast_error&) {
    return {};
  }
  return result;
}

inline std::vector<int> pyIntListOrEmpty(const py::dict& value, const char* key)
{
  std::vector<int> result;
  if (!value.contains(key) || value[key].is_none()) {
    return result;
  }
  try {
    result = value[key].cast<std::vector<int>>();
  } catch (const py::cast_error&) {
    return {};
  }
  return result;
}

template <typename T>
inline std::vector<T> pyArrayBufferToVector(const py::array_t<T, py::array::c_style | py::array::forcecast>& array)
{
  const py::buffer_info buffer = array.request();
  if (buffer.ndim != 1) {
    throw py::cast_error("expected a one-dimensional compact proposal buffer");
  }
  const auto* data = static_cast<const T*>(buffer.ptr);
  return std::vector<T>(data, data + buffer.shape[0]);
}

inline std::vector<int> pyIntArrayToVector(const py::handle& object)
{
  const py::array_t<int, py::array::c_style | py::array::forcecast> array(
      py::reinterpret_borrow<py::object>(object));
  return pyArrayBufferToVector<int>(array);
}

inline std::vector<int> pyIntArrayOrListOrEmpty(const py::dict& value, const char* key)
{
  std::vector<int> result;
  if (!value.contains(key) || value[key].is_none()) {
    return result;
  }
  try {
    result = pyIntArrayToVector(value[key]);
  } catch (const py::error_already_set&) {
    return pyIntListOrEmpty(value, key);
  } catch (const py::cast_error&) {
    return pyIntListOrEmpty(value, key);
  }
  return result;
}

inline std::vector<int64_t> pyInt64ListOrEmpty(const py::dict& value, const char* key)
{
  std::vector<int64_t> result;
  if (!value.contains(key) || value[key].is_none()) {
    return result;
  }
  try {
    result = value[key].cast<std::vector<int64_t>>();
  } catch (const py::cast_error&) {
    return {};
  }
  return result;
}

inline std::vector<int64_t> pyInt64ArrayToVector(const py::handle& object)
{
  const py::array_t<int64_t, py::array::c_style | py::array::forcecast> array(
      py::reinterpret_borrow<py::object>(object));
  return pyArrayBufferToVector<int64_t>(array);
}

inline std::vector<int64_t> pyInt64ArrayOrListOrEmpty(const py::dict& value, const char* key)
{
  std::vector<int64_t> result;
  if (!value.contains(key) || value[key].is_none()) {
    return result;
  }
  try {
    result = pyInt64ArrayToVector(value[key]);
  } catch (const py::error_already_set&) {
    return pyInt64ListOrEmpty(value, key);
  } catch (const py::cast_error&) {
    return pyInt64ListOrEmpty(value, key);
  }
  return result;
}

inline std::vector<double> pyDoubleListOrEmpty(const py::dict& value, const char* key)
{
  std::vector<double> result;
  if (!value.contains(key) || value[key].is_none()) {
    return result;
  }
  try {
    result = value[key].cast<std::vector<double>>();
  } catch (const py::cast_error&) {
    return {};
  }
  return result;
}

inline std::vector<double> pyDoubleArrayToVector(const py::handle& object)
{
  const py::array_t<double, py::array::c_style | py::array::forcecast> array(
      py::reinterpret_borrow<py::object>(object));
  return pyArrayBufferToVector<double>(array);
}

inline std::vector<double> pyDoubleArrayOrListOrEmpty(const py::dict& value, const char* key)
{
  std::vector<double> result;
  if (!value.contains(key) || value[key].is_none()) {
    return result;
  }
  try {
    result = pyDoubleArrayToVector(value[key]);
  } catch (const py::error_already_set&) {
    return pyDoubleListOrEmpty(value, key);
  } catch (const py::cast_error&) {
    return pyDoubleListOrEmpty(value, key);
  }
  return result;
}

template <typename T>
inline T vectorValueOrDefault(const std::vector<T>& values, std::size_t index, const T& fallback)
{
  return index < values.size() ? values[index] : fallback;
}

struct CompactBridgeObjectiveDeltaPrescreenResult
{
  py::dict summary;
  std::vector<double> predicted_delta_objs;
  std::vector<double> predicted_improvements;
  std::vector<double> prescreen_delta_objs;
  std::vector<int> trial_steps;
  std::vector<int> pass_flags;
  int filtered_count{0};
};

inline int pyIndexedIntOrDefault(const py::dict& value,
                          const char* key,
                          int index,
                          int fallback = -1)
{
  if (index < 0 || !value.contains(key) || value[key].is_none()) {
    return fallback;
  }
  try {
    const auto values = value[key].cast<std::vector<int>>();
    if (index >= static_cast<int>(values.size())) {
      return fallback;
    }
    return values[index];
  } catch (const py::cast_error&) {
    return fallback;
  }
}

inline int vectorIndexedIntOrDefault(const std::vector<int>& values, int index, int fallback = -1)
{
  if (index < 0 || index >= static_cast<int>(values.size())) {
    return fallback;
  }
  return values[index];
}

inline int diffGuidedBatchSelectorWorkerCount(const py::dict& config)
{
  return dgb::selectorWorkerCount(config);
}

inline int diffGuidedBatchTrialWorkerCount(const py::dict& config)
{
  return dgb::trialWorkerCount(config);
}

inline std::string diffGuidedBatchAcceptMode(const py::dict& config)
{
  return dgb::acceptMode(config);
}

inline double diffGuidedBatchTnsPowerLeakageWeight(const py::dict& config)
{
  return dgb::tnsPowerLeakageWeight(config);
}

inline bool diffGuidedBatchLocalObjectiveAlignGlobal(const py::dict& config)
{
  return dgb::localObjectiveAlignGlobal(config);
}

inline bool diffGuidedBatchNoopBaseline(const py::dict& config)
{
  return dgb::noopBaseline(config);
}

inline bool diffGuidedBatchRejectNonpositiveTrial(const py::dict& config)
{
  return dgb::rejectNonpositiveTrial(config);
}

inline bool diffGuidedBatchForceAcceptSelectedActions(const py::dict& config)
{
  return dgb::forceAcceptSelectedActions(config);
}

inline bool diffGuidedBatchSingleActionAudit(const py::dict& config)
{
  return dgb::singleActionAudit(config);
}

inline int diffGuidedBatchSingleActionAuditLimit(const py::dict& config)
{
  return dgb::singleActionAuditLimit(config);
}

inline double diffGuidedBatchNonpositiveTrialEps(const py::dict& config)
{
  return dgb::nonpositiveTrialEps(config);
}

inline double diffGuidedBatchLocalDeltaSlewPenaltyWeight(const py::dict& config)
{
  return dgb::localDeltaSlewPenaltyWeight(config);
}

inline double diffGuidedBatchLocalDeltaCapPenaltyWeight(const py::dict& config)
{
  return dgb::localDeltaCapPenaltyWeight(config);
}

inline double diffGuidedBatchFaninPenaltySensitivityPsPerPf(const py::dict& config)
{
  return dgb::faninPenaltySensitivityPsPerPf(config);
}

inline std::string diffGuidedBatchSlackDeltaEvaluatorMode(const py::dict& config)
{
  return dgb::slackDeltaEvaluatorMode(config);
}

inline std::string diffGuidedBatchSlackDeltaWeightMode(const py::dict& config)
{
  return dgb::slackDeltaWeightMode(config);
}

inline double diffGuidedBatchNPathWeightCap(const py::dict& config)
{
  return dgb::npathWeightCap(config);
}

inline bool diffGuidedBatchLaneContains(const std::string& lane, const std::string& term)
{
  return dgb::laneContains(lane, term);
}

inline DiffGuidedBatchEffectiveLocalWeight
diffGuidedBatchEffectiveLocalDeltaSlewPenaltyWeight(const py::dict& config)
{
  return dgb::effectiveLocalDeltaSlewPenaltyWeight(config);
}

inline DiffGuidedBatchEffectiveLocalWeight
diffGuidedBatchEffectiveLocalDeltaCapPenaltyWeight(const py::dict& config)
{
  return dgb::effectiveLocalDeltaCapPenaltyWeight(config);
}

inline double diffGuidedBatchLocalResidualFeedbackScale(const py::dict& config)
{
  return dgb::localResidualFeedbackScale(config);
}

inline std::string diffGuidedBatchLocalResidualFeedbackPolicy(const py::dict& config)
{
  return dgb::localResidualFeedbackPolicy(config);
}

inline double diffGuidedBatchLocalResidualFeedbackDecay(const py::dict& config)
{
  return dgb::localResidualFeedbackDecay(config);
}

inline double diffGuidedBatchLocalResidualMaxGainPerAction(const py::dict& config)
{
  return dgb::localResidualMaxGainPerAction(config);
}

inline double diffGuidedBatchLocalResidualBudgetDiscount(const py::dict& config)
{
  return dgb::localResidualBudgetDiscount(config);
}

inline double diffGuidedBatchActualDeltaObj(double actual_delta_tns,
                                     double leakage_delta,
                                     double leakage_weight)
{
  return dgb::actualDeltaObj(actual_delta_tns, leakage_delta, leakage_weight);
}

inline double diffGuidedBatchMetricDeltaObj(double tns_delta,
                                     double slew_vio_delta,
                                     double cap_vio_delta,
                                     double leakage_delta,
                                     double slew_weight,
                                     double cap_weight,
                                     double leakage_weight)
{
  return dgb::metricDeltaObj(
      tns_delta,
      slew_vio_delta,
      cap_vio_delta,
      leakage_delta,
      slew_weight,
      cap_weight,
      leakage_weight);
}

inline double diffGuidedBatchLocalResidualActualDeltaObj(double actual_delta_tns,
                                                  double leakage_delta,
                                                  double leakage_weight,
                                                  double local_delta_slew_penalty,
                                                  double local_delta_cap_penalty,
                                                  double local_delta_slew_violation_improvement = 0.0,
                                                  double local_delta_cap_violation_improvement = 0.0)
{
  return dgb::localResidualActualDeltaObj(
      actual_delta_tns,
      leakage_delta,
      leakage_weight,
      local_delta_slew_penalty,
      local_delta_cap_penalty,
      local_delta_slew_violation_improvement,
      local_delta_cap_violation_improvement);
}

inline double diffGuidedBatchLocalResidualBudget(const CompactBridgeActionProposal& action)
{
  return dgb::localResidualBudget(action);
}

inline std::string diffGuidedBatchLocalResidualTrialDeltaSource(
    const CompactBridgeActionProposal& action)
{
  if (action.local_delta_delay_ps > 0.0) {
    return action.local_delta_source.empty() ? "local_delta_delay_ps" : action.local_delta_source;
  }
  return "predicted_improvement_or_delta_obj_fallback";
}

inline double diffGuidedBatchTrialStepScale(const CompactBridgeActionProposal& action, int trial_step)
{
  return dgb::trialStepScale(action, trial_step);
}

inline double diffGuidedBatchLocalDeltaSlewPenalty(const CompactBridgeActionProposal& action,
                                            int trial_step,
                                            double penalty_weight)
{
  return dgb::localDeltaSlewPenalty(action, trial_step, penalty_weight);
}

inline double diffGuidedBatchLocalDeltaCapPenalty(const CompactBridgeActionProposal& action,
                                           int trial_step,
                                           double penalty_weight)
{
  return dgb::localDeltaCapPenalty(action, trial_step, penalty_weight);
}

inline double diffGuidedBatchLocalResidualTrialDelta(const CompactBridgeActionProposal& action,
                                              int trial_step,
                                              double residual_budget)
{
  return dgb::localResidualTrialDelta(action, trial_step, residual_budget);
}

inline double diffGuidedBatchCandidateDeltaForTrial(const CompactBridgeActionProposal& trial_action,
                                             double residual_budget)
{
  return dgb::candidateDeltaForTrial(trial_action, residual_budget);
}

inline double diffGuidedBatchCandidateLocalDeltaSlewPenalty(
    const CompactBridgeActionProposal& trial_action,
    double penalty_weight)
{
  return dgb::candidateLocalDeltaSlewPenalty(trial_action, penalty_weight);
}

inline double diffGuidedBatchCandidateLocalDeltaCapPenalty(double local_delta_cap,
                                                    double penalty_weight)
{
  return dgb::candidateLocalDeltaCapPenalty(local_delta_cap, penalty_weight);
}

inline double diffGuidedBatchLocalSlewCapViolationImprovement(double old_actual,
                                                       double new_actual,
                                                       double old_limit,
                                                       double new_limit,
                                                       double objective_weight)
{
  return dgb::localSlewCapViolationImprovement(
      old_actual,
      new_actual,
      old_limit,
      new_limit,
      objective_weight);
}

inline py::dict makeDiffGuidedBatchLocalResidualPlaceholderMetrics(
    const std::string& evaluator_label = "local_residual")
{
  py::dict metrics;
  metrics["artifact"] = "diff_guided_batch_opensta_timing_metrics";
  metrics["metric_source"] = evaluator_label + "_no_opensta_snapshot";
  metrics["status"] = "skipped_" + evaluator_label + "_snapshot";
  metrics["wns"] = 0.0;
  metrics["tns"] = 0.0;
  metrics["setup_violation_count"] = 0;
  metrics["slew_vio"] = -1.0;
  metrics["cap_vio"] = -1.0;
  metrics["slew_vio_count"] = -1.0;
  metrics["cap_vio_count"] = -1.0;
  metrics["slew_vio_semantics"] = "not_queried";
  metrics["cap_vio_semantics"] = "not_queried";
  metrics["leakage"] = -1.0;
  metrics["query_ms"] = 0.0;
  return metrics;
}

struct DiffGuidedBatchLocalResidualKey
{
  std::string key;
  int component_id{-1};
  std::string source;
};

struct DiffGuidedBatchLocalResidualSnapshotBudget
{
  double budget{0.0};
  std::string source{"predicted_improvement_or_delta_obj_fallback"};
  bool from_snapshot{false};
};

inline DiffGuidedBatchLocalResidualKey diffGuidedBatchLocalResidualKey(
    const CompactBridgeActionProposal& action)
{
  if (action.top_path_component >= 0) {
    return {
        "top_path:" + std::to_string(action.top_path_component),
        action.top_path_component,
        "top_path_component"};
  }
  if (action.endpoint_component >= 0) {
    return {
        "endpoint:" + std::to_string(action.endpoint_component),
        action.endpoint_component,
        "endpoint_component"};
  }
  return {
      "inst:" + std::to_string(action.inst_id),
      action.inst_id,
      "inst_id_fallback"};
}

inline DiffGuidedBatchLocalResidualSnapshotBudget diffGuidedBatchLocalResidualSnapshotBudget(
    const CompactBridgeActionProposal& action,
    const std::vector<double>& top_path_residual_budgets,
    const std::vector<double>& endpoint_residual_budgets)
{
  if (action.top_path_component >= 0
      && action.top_path_component < static_cast<int>(top_path_residual_budgets.size())) {
    return {
        std::max(0.0, top_path_residual_budgets[action.top_path_component]),
        "top_path_component_snapshot",
        true};
  }
  if (action.endpoint_component >= 0
      && action.endpoint_component < static_cast<int>(endpoint_residual_budgets.size())) {
    return {
        std::max(0.0, endpoint_residual_budgets[action.endpoint_component]),
        "endpoint_component_snapshot",
        true};
  }
  return {
      diffGuidedBatchLocalResidualBudget(action),
      diffGuidedBatchLocalResidualTrialDeltaSource(action),
      false};
}

inline py::dict copyDiffGuidedBatchConfig(const py::dict& config)
{
  py::dict copied;
  for (const auto& item : config) {
    copied[item.first] = item.second;
  }
  return copied;
}

inline std::string diffGuidedBatchTrialMode(const py::dict& config)
{
  return dgb::trialMode(config);
}

inline int diffGuidedBatchTrialMiniBatchSize(const py::dict& config)
{
  return dgb::trialMiniBatchSize(config);
}

struct CompactBridgeEvaluationSummary
{
  py::dict summary;
  std::vector<CompactBridgeActionResult> accepted_action_results;
  int evaluator_result_count{0};
  double accepted_delta_tns_sum{0.0};
};

inline double elapsedMs(std::chrono::steady_clock::time_point begin)
{
  const auto end = std::chrono::steady_clock::now();
  return std::chrono::duration<double, std::milli>(end - begin).count();
}

struct CompactBridgeSizingTrial
{
  std::string master_name;
  int step{0};
  int target_size_idx{-1};
  int target_cell_id{-1};
  double target_timing_coordinate{0.0};
};

inline CompactBridgeSizingTrial makeDiffGuidedBatchSizingTrial(
    const CompactBridgeActionProposal& action,
    int target_size_idx,
    int step,
    const std::string& fallback_master_name)
{
  CompactBridgeSizingTrial trial;
  trial.step = step;
  trial.target_size_idx = target_size_idx;
  trial.master_name = fallback_master_name;
  if (target_size_idx >= 0) {
    trial.target_cell_id = vectorIndexedIntOrDefault(
        action.legal_cell_id_candidates, target_size_idx, -1);
    trial.target_timing_coordinate = vectorValueOrDefault<double>(
        action.legal_timing_coordinate_candidates,
        static_cast<std::size_t>(target_size_idx),
        action.new_timing_coordinate);
    trial.master_name = vectorValueOrDefault<std::string>(
        action.legal_master_candidates,
        static_cast<std::size_t>(target_size_idx),
        fallback_master_name);
  } else {
    trial.target_timing_coordinate = action.new_timing_coordinate;
  }
  return trial;
}

inline void writeDiffGuidedBatchFinalSizingMetadata(py::dict& result,
                                             const CompactBridgeActionProposal& action,
                                             const CompactBridgeSizingTrial& trial)
{
  const int best_target_size_idx = trial.target_size_idx;
  const int best_target_cell_id = trial.target_cell_id;
  const double best_target_timing_coordinate = trial.target_timing_coordinate;
  const int best_current_cell_id = vectorIndexedIntOrDefault(
      action.legal_cell_id_candidates, action.current_size_idx, -1);
  const double best_current_timing_coordinate = vectorValueOrDefault<double>(
      action.legal_timing_coordinate_candidates,
      static_cast<std::size_t>(std::max(0, action.current_size_idx)),
      action.old_timing_coordinate);
  result["target_size_idx"] = best_target_size_idx;
  result["evaluated_size_idx"] = best_target_size_idx;
  result["best_trial_step"] = trial.step;
  result["old_cell_id"] = best_current_cell_id;
  result["old_master_id"] = best_current_cell_id;
  result["new_cell_id"] = best_target_cell_id;
  result["new_master_id"] = best_target_cell_id;
  result["target_master_id"] = best_target_cell_id;
  result["old_timing_coordinate"] = best_current_timing_coordinate;
  result["new_timing_coordinate"] = best_target_timing_coordinate;
}

inline std::vector<CompactBridgeSizingTrial> buildDiffGuidedBatchSizingTrials(
    const CompactBridgeActionProposal& action,
    odb::dbMaster* old_master,
    const py::dict& config)
{
  std::vector<CompactBridgeSizingTrial> trials;
  const auto& legal_master_candidates = action.legal_master_candidates;
  const int current_size_idx = action.current_size_idx;
  const int seed_size_idx = action.target_size_idx >= 0 ? action.target_size_idx : action.seed_size_idx;
  const int max_up_step = pyIntOrDefault(config, "precise_improvement_max_up_step", 1);
  const int max_down_step = pyIntOrDefault(config, "precise_improvement_max_down_step", 0);

  if (!legal_master_candidates.empty() && current_size_idx >= 0
      && current_size_idx < static_cast<int>(legal_master_candidates.size())) {
    int direction = 0;
    if (seed_size_idx > current_size_idx) {
      direction = 1;
    } else if (seed_size_idx < current_size_idx) {
      direction = -1;
    } else {
      if (action.seed_direction == "up") {
        direction = 1;
      } else if (action.seed_direction == "down") {
        direction = -1;
      }
    }

    if (direction > 0) {
      const int max_index = std::min(
          static_cast<int>(legal_master_candidates.size()) - 1,
          current_size_idx + std::max(max_up_step, 0));
      for (int index = current_size_idx + 1; index <= max_index; ++index) {
        if (!legal_master_candidates[index].empty()) {
          trials.push_back(makeDiffGuidedBatchSizingTrial(
              action, index, index - current_size_idx, legal_master_candidates[index]));
        }
      }
    } else if (direction < 0) {
      const int min_index = std::max(0, current_size_idx - std::max(max_down_step, 0));
      for (int index = current_size_idx - 1; index >= min_index; --index) {
        if (!legal_master_candidates[index].empty()) {
          trials.push_back(makeDiffGuidedBatchSizingTrial(
              action, index, index - current_size_idx, legal_master_candidates[index]));
        }
      }
    }
  }

  if (trials.empty()) {
    std::string target_master_name = action.target_master.empty() ? action.seed_master : action.target_master;
    if (!target_master_name.empty()
        && (old_master == nullptr || target_master_name != old_master->getName())) {
      int step = action.seed_step != 0
                     ? action.seed_step
                     : (seed_size_idx >= 0 && current_size_idx >= 0 ? seed_size_idx - current_size_idx : 0);
      trials.push_back(makeDiffGuidedBatchSizingTrial(
          action, seed_size_idx, step, target_master_name));
    }
  }
  return trials;
}

inline CompactBridgeObjectiveDeltaPrescreenResult evaluateDiffGuidedBatchObjectiveDeltaPrescreen(
    const std::vector<CompactBridgeActionProposal>& compact_actions,
    const py::dict& config = py::dict())
{
  CompactBridgeObjectiveDeltaPrescreenResult result;
  const auto begin = std::chrono::steady_clock::now();
  const int worker_count = diffGuidedBatchTrialWorkerCount(config);
  const double max_predicted_delta_obj = pyDoubleOrDefault(
      config,
      "parallel_prescreen_max_predicted_delta_obj",
      pyDoubleOrDefault(config, "diff_guided_batch_parallel_prescreen_max_predicted_delta_obj", 0.0));
  result.predicted_delta_objs.resize(compact_actions.size(), 0.0);
  result.predicted_improvements.resize(compact_actions.size(), 0.0);
  result.prescreen_delta_objs.resize(compact_actions.size(), 0.0);
  result.trial_steps.resize(compact_actions.size(), 0);
  result.pass_flags.resize(compact_actions.size(), 0);

#pragma omp parallel for num_threads(worker_count) schedule(static)
  for (int index = 0; index < static_cast<int>(compact_actions.size()); ++index) {
    const auto& action = compact_actions[index];
    const double prescreen_predicted_delta_obj = action.predicted_delta_obj;
    const double predicted_improvement =
        action.predicted_improvement != 0.0 ? action.predicted_improvement
                                            : -prescreen_predicted_delta_obj;
    const int step = action.target_size_idx >= 0 && action.current_size_idx >= 0
                         ? action.target_size_idx - action.current_size_idx
                         : action.seed_step;
    result.predicted_delta_objs[index] = prescreen_predicted_delta_obj;
    result.predicted_improvements[index] = predicted_improvement;
    result.prescreen_delta_objs[index] = prescreen_predicted_delta_obj;
    result.trial_steps[index] = step;
    result.pass_flags[index] = action.kind == ActionKind::kSizing
                                   && prescreen_predicted_delta_obj <= max_predicted_delta_obj;
  }

  for (const int pass_flag : result.pass_flags) {
    if (pass_flag == 0) {
      result.filtered_count += 1;
    }
  }

  py::list prescreen_delta_objs;
  py::list predicted_improvements;
  py::list trial_steps;
  for (std::size_t index = 0; index < compact_actions.size(); ++index) {
    prescreen_delta_objs.append(result.prescreen_delta_objs[index]);
    predicted_improvements.append(result.predicted_improvements[index]);
    trial_steps.append(result.trial_steps[index]);
  }

  result.summary["parallel_prescreen_mode"] = "record_only";
  result.summary["parallel_prescreen_worker_count"] = worker_count;
  result.summary["parallel_prescreen_candidate_count"] = static_cast<int>(compact_actions.size());
  result.summary["parallel_prescreen_filtered_count"] = result.filtered_count;
  result.summary["parallel_prescreen_filter_enabled"] = false;
  result.summary["parallel_prescreen_max_predicted_delta_obj"] = max_predicted_delta_obj;
  result.summary["parallel_prescreen_ms"] = elapsedMs(begin);
  result.summary["prescreen_predicted_delta_obj"] = prescreen_delta_objs;
  result.summary["prescreen_predicted_improvement"] = predicted_improvements;
  result.summary["prescreen_trial_step"] = trial_steps;
  return result;
}

inline py::dict checkDiffGuidedBatchGuardrails(const py::dict& baseline_metrics,
                                        const py::dict& verified_metrics,
                                        const py::dict& config)
{
  py::dict guardrail;
  py::list failed_reasons;
  const double baseline_tns = pyDoubleOrDefault(baseline_metrics, "tns", 0.0);
  const double verified_tns = pyDoubleOrDefault(verified_metrics, "tns", baseline_tns);
  const double tns_delta = verified_tns - baseline_tns;

  const double baseline_wns = pyDoubleOrDefault(baseline_metrics, "wns", 0.0);
  const double verified_wns = pyDoubleOrDefault(verified_metrics, "wns", baseline_wns);
  const double max_batch_wns_degradation = pyDoubleOrDefault(
      config, "max_batch_wns_degradation", std::numeric_limits<double>::infinity());
  const double wns_delta = verified_wns - baseline_wns;
  if (-wns_delta > max_batch_wns_degradation) {
    failed_reasons.append("wns_guardrail_failed");
  }

  const double baseline_slew_vio = pyDoubleOrDefault(baseline_metrics, "slew_vio", 0.0);
  const double verified_slew_vio = pyDoubleOrDefault(verified_metrics, "slew_vio", baseline_slew_vio);
  const double max_batch_slew_vio_increase = pyDoubleOrDefault(
      config, "max_batch_slew_vio_increase", 0.0);
  const double slew_vio_delta = verified_slew_vio - baseline_slew_vio;
  if (slew_vio_delta > max_batch_slew_vio_increase) {
    failed_reasons.append("slew_violation_guardrail_failed");
  }

  const double baseline_cap_vio = pyDoubleOrDefault(baseline_metrics, "cap_vio", 0.0);
  const double verified_cap_vio = pyDoubleOrDefault(verified_metrics, "cap_vio", baseline_cap_vio);
  const double max_batch_cap_vio_increase = pyDoubleOrDefault(
      config, "max_batch_cap_vio_increase", 0.0);
  const double cap_vio_delta = verified_cap_vio - baseline_cap_vio;
  if (cap_vio_delta > max_batch_cap_vio_increase) {
    failed_reasons.append("cap_violation_guardrail_failed");
  }

  const double baseline_leakage = pyDoubleOrDefault(baseline_metrics, "leakage", 0.0);
  const double verified_leakage = pyDoubleOrDefault(verified_metrics, "leakage", baseline_leakage);
  const double max_batch_leakage_increase = pyDoubleOrDefault(
      config, "max_batch_leakage_increase",
      pyDoubleOrDefault(config, "diff_guided_batch_max_batch_leakage_increase",
                        std::numeric_limits<double>::infinity()));
  const double leakage_delta = verified_leakage - baseline_leakage;
  if (leakage_delta > max_batch_leakage_increase) {
    failed_reasons.append("leakage_guardrail_failed");
  }

  const double slew_weight = diffGuidedBatchEffectiveLocalDeltaSlewPenaltyWeight(config).value;
  const double cap_weight = diffGuidedBatchEffectiveLocalDeltaCapPenaltyWeight(config).value;
  const double leakage_weight = diffGuidedBatchTnsPowerLeakageWeight(config);
  const double batch_objective_delta =
      diffGuidedBatchMetricDeltaObj(tns_delta,
                                    slew_vio_delta,
                                    cap_vio_delta,
                                    leakage_delta,
                                    slew_weight,
                                    cap_weight,
                                    leakage_weight);

  guardrail["tns_delta"] = tns_delta;
  guardrail["wns_delta"] = wns_delta;
  guardrail["slew_vio_delta"] = slew_vio_delta;
  guardrail["cap_vio_delta"] = cap_vio_delta;
  guardrail["leakage_delta"] = leakage_delta;
  guardrail["batch_objective_delta"] = batch_objective_delta;
  guardrail["batch_objective_slew_weight"] = slew_weight;
  guardrail["batch_objective_cap_weight"] = cap_weight;
  guardrail["batch_objective_leakage_weight"] = leakage_weight;
  guardrail["max_batch_wns_degradation"] = max_batch_wns_degradation;
  guardrail["max_batch_slew_vio_increase"] = max_batch_slew_vio_increase;
  guardrail["max_batch_cap_vio_increase"] = max_batch_cap_vio_increase;
  guardrail["max_batch_leakage_increase"] = max_batch_leakage_increase;
  guardrail["guardrail_failed_reasons"] = failed_reasons;
  guardrail["passed"] = py::len(failed_reasons) == 0;
  return guardrail;
}

double staTimeToPsOrDefault(const sta::Unit* unit, double value, double fallback);
double psToStaTimeOrDefault(const sta::Unit* unit, double value_ps, double fallback);
double staCapToPfOrDefault(const sta::Unit* unit, double value, double fallback);
double pfToStaCapOrDefault(const sta::Unit* unit, double value_pf, double fallback);
double exportLibcellLeakageForPython(sta::LibertyCell* liberty_cell);
bool staScalarIsUsable(double value);
double resolveLibPinSlewLimitForPythonPs(const sta::Unit* unit, sta::LibertyPort* liberty_port);

struct DiffGuidedBatchCppLocalCapEstimatorResult
{
  double local_delta_cap{0.0};
  int compared_input_port_count{0};
  int positive_input_port_count{0};
  bool used{false};
  std::string estimator_source{"cpp_hot_path_input_cap_estimator_unavailable"};
};

struct DiffGuidedBatchCppLocalDelaySlewEstimatorResult
{
  double local_delta_delay_ps{0.0};
  double local_delta_slew_ps{0.0};
  double signed_delay_delta{0.0};
  double signed_slew_delta{0.0};
  double selected_current_delay_ps{0.0};
  double selected_target_delay_ps{0.0};
  double selected_current_slew_ps{0.0};
  double selected_target_slew_ps{0.0};
  int compared_arc_count{0};
  int positive_delay_arc_count{0};
  int positive_slew_arc_count{0};
  bool used{false};
  std::string estimator_source{"cpp_hot_path_liberty_table_delay_slew_estimator_unavailable"};
  std::string fallback_reason{"missing_liberty_table_arc_match"};
};

struct DiffGuidedBatchFaninNeighborhoodPenaltyResult
{
  double local_predicted_fanin_neighborhood_penalty{0.0};
  int fanin_net_count{0};
  int affected_fo_cell_count{0};
  int positive_input_cap_delta_count{0};
  double fanin_penalty_ps_per_pf{1000.0};
  bool used{false};
  std::string local_predicted_fanin_penalty_source{
      "cpp_hot_path_fanin_neighborhood_penalty_unavailable"};
  std::string fallback_reason{"missing_fanin_context"};
};

struct DiffGuidedBatchNPathSnapshot
{
  bool valid{false};
  double weight_cap{128.0};
  double build_ms{0.0};
  int node_count{0};
  int edge_count{0};
  int source_boundary_count{0};
  int endpoint_boundary_count{0};
  int sequential_or_macro_boundary_count{0};
  int cycle_or_unresolved_count{0};
  int fallback_weight_count{0};
  std::string source{"unavailable"};
  std::vector<double> inst_npath_weight_by_id;
  std::vector<double> n_from_by_id;
  std::vector<double> n_to_by_id;
  std::unordered_map<odb::dbInst*, int> inst_id_by_db_inst;
};

struct DiffGuidedBatchSlackDeltaPoint
{
  bool used{false};
  int point_inst_id{-1};
  std::string point_kind{"fallback"};
  std::string pin_name;
  std::string old_master_name;
  std::string new_master_name;
  double old_slack_ps{0.0};
  double old_arrival_ps{0.0};
  double old_required_ps{0.0};
  double old_delay_ps{0.0};
  double new_delay_ps{0.0};
  double delta_delay_ps{0.0};
  double new_slack_ps{0.0};
  double slack_violation_delta_ps{0.0};
  double old_slew_ps{0.0};
  double new_slew_ps{0.0};
  double delta_slew_ps{0.0};
  double npath_weight{1.0};
  double weighted_slack_violation_delta_ps{0.0};
  std::string dcalc_source{"unavailable"};
  std::string fallback_reason;
};

struct DiffGuidedBatchCellReplaceSlackDeltaResult
{
  bool used{false};
  double downstream_slack_improvement_ps{0.0};
  double downstream_old_slack_ps{0.0};
  double downstream_new_slack_ps{0.0};
  double downstream_delta_delay_ps{0.0};
  double downstream_npath_weight{1.0};
  double downstream_weighted_slack_improvement_ps{0.0};
  std::string downstream_dcalc_source{"unavailable"};
  double fanin_slack_penalty_ps{0.0};
  double fanin_slack_delta_ps{0.0};
  double fanin_weighted_slack_delta_ps{0.0};
  double net_slack_obj_delta_ps{0.0};
  double weighted_net_tns_delta_ps{0.0};
  double slew_violation_delta_ps{0.0};
  double cap_violation_delta_pf{0.0};
  int downstream_point_count{0};
  int fanin_point_count{0};
  int dcalc_query_count{0};
  int fallback_point_count{0};
  int fanin_net_count{0};
  int affected_fo_cell_count{0};
  int positive_input_cap_delta_count{0};
  int negative_input_cap_delta_count{0};
  double heuristic_fanin_penalty_ps{0.0};
  std::string slack_delta_weight_mode{"raw_slack_delta"};
  std::string estimator_source{"unavailable"};
  std::string fanin_source{"unavailable"};
  std::string fallback_reason;
  std::vector<DiffGuidedBatchSlackDeltaPoint> points;
};

struct DiffGuidedBatchCurrentLocalTimingContext
{
  bool used{false};
  double input_slew_ps{0.0};
  double load_cap_pf{0.0};
  double output_slew_ps{0.0};
  double output_slew_for_violation_ps{0.0};
  double output_cap_for_violation_pf{0.0};
  double output_slew_limit_ps{0.0};
  double output_cap_limit_pf{0.0};
  double output_slew_violation_ps{0.0};
  double output_cap_violation_pf{0.0};
  bool output_slack_used{false};
  double output_slack_ps{0.0};
  double output_arrival_ps{0.0};
  double output_required_ps{0.0};
  double output_slack_violation_ps{0.0};
  int input_slew_pin_count{0};
  int load_cap_pin_count{0};
  int output_slew_pin_count{0};
  int output_slew_limit_pin_count{0};
  int output_cap_limit_pin_count{0};
  int output_slack_pin_count{0};
  std::string local_context_source{"unavailable"};
  std::string input_slew_source{"proposal_fallback"};
  std::string load_cap_source{"proposal_fallback"};
  std::string output_slew_source{"proposal_fallback"};
  std::string output_slew_limit_source{"unavailable"};
  std::string output_cap_limit_source{"unavailable"};
  std::string output_slew_limit_port_name;
  std::string output_cap_limit_port_name;
  std::string slack_source{"proposal_fallback"};
  std::string fallback_reason{"missing_opensta_current_context"};
};

struct DiffGuidedBatchEffectiveCandidateBudget
{
  double budget{0.0};
  double component_budget{0.0};
  double pin_slack_budget{0.0};
  bool pin_slack_used{false};
  std::string source{"component_residual_budget"};
};

inline DiffGuidedBatchEffectiveCandidateBudget diffGuidedBatchEffectiveLocalCandidateBudget(
    double component_budget,
    const DiffGuidedBatchCurrentLocalTimingContext& current_context)
{
  DiffGuidedBatchEffectiveCandidateBudget result;
  component_budget = std::max(0.0, component_budget);
  result.budget = component_budget;
  result.component_budget = component_budget;
  if (!current_context.output_slack_used) {
    return result;
  }

  const double pin_slack_budget = std::max(0.0, current_context.output_slack_violation_ps);
  result.pin_slack_used = true;
  result.pin_slack_budget = pin_slack_budget;
  if (component_budget > 0.0) {
    result.budget = std::min(component_budget, pin_slack_budget);
    result.source = "component_residual_and_pin_slack_min";
  } else {
    result.budget = pin_slack_budget;
    result.source = "pin_slack_violation";
  }
  return result;
}

inline bool diffGuidedBatchCandidateBetterThanBest(bool has_valid_trial,
                                            double actual_delta_obj,
                                            int trial_step,
                                            double best_actual_delta_obj,
                                            int best_trial_step)
{
  if (!has_valid_trial || actual_delta_obj > best_actual_delta_obj) {
    return true;
  }
  if (actual_delta_obj == best_actual_delta_obj) {
    return std::abs(trial_step) < std::abs(best_trial_step);
  }
  return false;
}

struct DiffGuidedBatchCppLeakageDeltaEstimatorResult
{
  double leakage_delta{0.0};
  double current_leakage{0.0};
  double target_leakage{0.0};
  bool used{false};
  std::string estimator_source{"cpp_hot_path_liberty_leakage_delta_estimator_unavailable"};
  std::string fallback_reason{"missing_liberty_cell_or_leakage"};
};

struct DiffGuidedBatchStaAwareReplaceCellResult
{
  bool replaced{false};
  std::string transaction_apply_primitive{"opensta_replace_cell"};
  std::string fallback_reason;
};

inline DiffGuidedBatchStaAwareReplaceCellResult diffGuidedBatchStaAwareReplaceCell(
    sta::dbSta* sta,
    sta::dbNetwork* network,
    odb::dbInst* inst,
    odb::dbMaster* target_master)
{
  DiffGuidedBatchStaAwareReplaceCellResult result;
  if (sta == nullptr || network == nullptr || inst == nullptr || target_master == nullptr) {
    result.fallback_reason = "missing_sta_network_inst_or_master";
    return result;
  }
  sta::Instance* sta_inst = network->dbToSta(inst);
  sta::Cell* target_cell = network->dbToSta(target_master);
  if (sta_inst == nullptr || target_cell == nullptr) {
    result.fallback_reason = "missing_sta_instance_or_cell_mapping";
    return result;
  }

  sta->replaceCell(sta_inst, target_cell);
  result.replaced = inst->getMaster() == target_master;
  if (!result.replaced) {
    result.fallback_reason = "opensta_replace_cell_did_not_update_db_master";
  }
  return result;
}

inline DiffGuidedBatchCppLeakageDeltaEstimatorResult diffGuidedBatchCppLeakageDeltaEstimator(
    sta::dbNetwork* network,
    odb::dbMaster* current_master,
    odb::dbMaster* target_master)
{
  DiffGuidedBatchCppLeakageDeltaEstimatorResult result;
  if (network == nullptr || current_master == nullptr || target_master == nullptr) {
    result.fallback_reason = "missing_network_or_master";
    return result;
  }
  auto* current_cell = network->dbToSta(current_master);
  auto* target_cell = network->dbToSta(target_master);
  auto* current_liberty_cell = current_cell == nullptr ? nullptr : network->libertyCell(current_cell);
  auto* target_liberty_cell = target_cell == nullptr ? nullptr : network->libertyCell(target_cell);
  if (current_liberty_cell == nullptr || target_liberty_cell == nullptr) {
    result.fallback_reason = "missing_liberty_cell";
    return result;
  }

  result.current_leakage = exportLibcellLeakageForPython(current_liberty_cell);
  result.target_leakage = exportLibcellLeakageForPython(target_liberty_cell);
  result.leakage_delta = result.target_leakage - result.current_leakage;
  if (!std::isfinite(result.current_leakage) || !std::isfinite(result.target_leakage)) {
    result.fallback_reason = "non_finite_liberty_leakage";
    result.leakage_delta = 0.0;
    return result;
  }

  result.used = true;
  result.estimator_source = "cpp_hot_path_liberty_leakage_delta_estimator";
  result.fallback_reason.clear();
  return result;
}

inline DiffGuidedBatchCppLocalCapEstimatorResult diffGuidedBatchCppLocalInputCapEstimator(
    sta::dbNetwork* network,
    odb::dbMaster* current_master,
    odb::dbMaster* target_master)
{
  DiffGuidedBatchCppLocalCapEstimatorResult result;
  if (network == nullptr || current_master == nullptr || target_master == nullptr) {
    return result;
  }
  auto* current_cell = network->dbToSta(current_master);
  auto* target_cell = network->dbToSta(target_master);
  auto* current_liberty_cell = current_cell == nullptr ? nullptr : network->libertyCell(current_cell);
  auto* target_liberty_cell = target_cell == nullptr ? nullptr : network->libertyCell(target_cell);
  if (current_liberty_cell == nullptr || target_liberty_cell == nullptr) {
    return result;
  }

  sta::LibertyCellPortIterator target_port_iter(target_liberty_cell);
  while (target_port_iter.hasNext()) {
    sta::LibertyPort* target_liberty_port = target_port_iter.next();
    if (target_liberty_port == nullptr || target_liberty_port->isPwrGnd()) {
      continue;
    }
    const auto* target_direction = target_liberty_port->direction();
    if (target_direction == nullptr || !target_direction->isAnyInput()) {
      continue;
    }
    sta::LibertyPort* current_liberty_port =
        current_liberty_cell->findLibertyPort(target_liberty_port->name());
    if (current_liberty_port == nullptr || current_liberty_port->isPwrGnd()) {
      continue;
    }
    const auto* current_direction = current_liberty_port->direction();
    if (current_direction == nullptr || !current_direction->isAnyInput()) {
      continue;
    }

    const sta::Units* target_units = target_liberty_port->libertyLibrary() == nullptr
                                         ? nullptr
                                         : target_liberty_port->libertyLibrary()->units();
    const sta::Units* current_units = current_liberty_port->libertyLibrary() == nullptr
                                          ? nullptr
                                          : current_liberty_port->libertyLibrary()->units();
    const sta::Unit* target_cap_unit = target_units == nullptr ? nullptr : target_units->capacitanceUnit();
    const sta::Unit* current_cap_unit = current_units == nullptr ? nullptr : current_units->capacitanceUnit();
    const double target_cap = staCapToPfOrDefault(
        target_cap_unit, target_liberty_port->capacitance(), 0.0);
    const double current_cap = staCapToPfOrDefault(
        current_cap_unit, current_liberty_port->capacitance(), 0.0);
    const double delta_cap = std::max(0.0, target_cap - current_cap);
    result.local_delta_cap = std::max(result.local_delta_cap, delta_cap);
    result.compared_input_port_count += 1;
    if (delta_cap > 0.0) {
      result.positive_input_port_count += 1;
    }
  }
  if (result.compared_input_port_count > 0) {
    result.used = true;
    result.estimator_source = "cpp_hot_path_input_cap_estimator";
  }
  return result;
}

inline double diffGuidedBatchPenaltyFromSlackDegradation(double slack_ps, double degradation_ps)
{
  return dgb::penaltyFromSlackDegradation(slack_ps, degradation_ps);
}

inline double diffGuidedBatchSlackViolationDeltaFromDelayDelta(double old_slack_ps,
                                                        double delta_delay_ps)
{
  return dgb::slackViolationDeltaFromDelayDelta(old_slack_ps, delta_delay_ps);
}

inline DiffGuidedBatchSlackDeltaPoint diffGuidedBatchMakeSlackDeltaPoint(
    const std::string& point_kind,
    int point_inst_id,
    const std::string& pin_name,
    const std::string& old_master_name,
    const std::string& new_master_name,
    double old_slack_ps,
    double old_arrival_ps,
    double old_required_ps,
    double old_delay_ps,
    double new_delay_ps,
    double old_slew_ps,
    double new_slew_ps,
    const std::string& dcalc_source,
    double npath_weight = 1.0)
{
  DiffGuidedBatchSlackDeltaPoint point;
  point.used = std::isfinite(old_slack_ps) && std::isfinite(old_delay_ps)
               && std::isfinite(new_delay_ps);
  point.point_inst_id = point_inst_id;
  point.point_kind = point_kind;
  point.pin_name = pin_name;
  point.old_master_name = old_master_name;
  point.new_master_name = new_master_name;
  point.old_slack_ps = old_slack_ps;
  point.old_arrival_ps = old_arrival_ps;
  point.old_required_ps = old_required_ps;
  point.old_delay_ps = old_delay_ps;
  point.new_delay_ps = new_delay_ps;
  point.delta_delay_ps = new_delay_ps - old_delay_ps;
  point.new_slack_ps = old_slack_ps - point.delta_delay_ps;
  point.slack_violation_delta_ps =
      diffGuidedBatchSlackViolationDeltaFromDelayDelta(old_slack_ps, point.delta_delay_ps);
  point.old_slew_ps = old_slew_ps;
  point.new_slew_ps = new_slew_ps;
  point.delta_slew_ps = new_slew_ps - old_slew_ps;
  point.npath_weight = std::max(1.0, npath_weight);
  point.weighted_slack_violation_delta_ps =
      point.slack_violation_delta_ps * point.npath_weight;
  point.dcalc_source = dcalc_source;
  point.fallback_reason = point.used ? "" : "non_finite_slack_or_delay";
  return point;
}

inline py::dict diffGuidedBatchSlackDeltaPointToPyDict(const DiffGuidedBatchSlackDeltaPoint& point)
{
  py::dict row;
  row["used"] = point.used;
  row["point_inst_id"] = point.point_inst_id;
  row["point_kind"] = point.point_kind;
  row["pin_name"] = point.pin_name;
  row["old_master_name"] = point.old_master_name;
  row["new_master_name"] = point.new_master_name;
  row["old_slack_ps"] = point.old_slack_ps;
  row["old_arrival_ps"] = point.old_arrival_ps;
  row["old_required_ps"] = point.old_required_ps;
  row["old_delay_ps"] = point.old_delay_ps;
  row["new_delay_ps"] = point.new_delay_ps;
  row["delta_delay_ps"] = point.delta_delay_ps;
  row["new_slack_ps"] = point.new_slack_ps;
  row["slack_violation_delta_ps"] = point.slack_violation_delta_ps;
  row["old_slew_ps"] = point.old_slew_ps;
  row["new_slew_ps"] = point.new_slew_ps;
  row["delta_slew_ps"] = point.delta_slew_ps;
  row["point_npath_weight"] = point.npath_weight;
  row["point_weighted_slack_delta"] = point.weighted_slack_violation_delta_ps;
  row["dcalc_source"] = point.dcalc_source;
  row["fallback_reason"] = point.fallback_reason;
  return row;
}

inline py::list diffGuidedBatchSlackDeltaPointsToPyList(
    const std::vector<DiffGuidedBatchSlackDeltaPoint>& points)
{
  py::list rows;
  for (const auto& point : points) {
    rows.append(diffGuidedBatchSlackDeltaPointToPyDict(point));
  }
  return rows;
}

inline bool diffGuidedBatchWorstSlackAtITerm(ord::Timing& timing,
                                      const sta::Unit* time_unit,
                                      odb::dbITerm* iterm,
                                      double& slack_ps,
                                      double& arrival_ps,
                                      double& required_ps)
{
  if (iterm == nullptr) {
    return false;
  }
  const double rise_slack = timing.getPinSlack(iterm, ord::Timing::Rise, ord::Timing::Max);
  const double fall_slack = timing.getPinSlack(iterm, ord::Timing::Fall, ord::Timing::Max);
  const double rise_arrival = timing.getPinArrival(iterm, ord::Timing::Rise, ord::Timing::Max);
  const double fall_arrival = timing.getPinArrival(iterm, ord::Timing::Fall, ord::Timing::Max);
  const bool rise_usable = staScalarIsUsable(rise_slack) && staScalarIsUsable(rise_arrival);
  const bool fall_usable = staScalarIsUsable(fall_slack) && staScalarIsUsable(fall_arrival);
  if (!rise_usable && !fall_usable) {
    return false;
  }
  const bool use_fall = !rise_usable || (fall_usable && fall_slack < rise_slack);
  const double slack = use_fall ? fall_slack : rise_slack;
  const double arrival = use_fall ? fall_arrival : rise_arrival;
  slack_ps = staTimeToPsOrDefault(time_unit, slack, 0.0);
  arrival_ps = staTimeToPsOrDefault(time_unit, arrival, 0.0);
  required_ps = staTimeToPsOrDefault(time_unit, arrival + slack, arrival_ps + slack_ps);
  return true;
}

inline double diffGuidedBatchMaxInputSlewPs(ord::Timing& timing,
                                     const sta::Unit* time_unit,
                                     odb::dbInst* inst,
                                     double fallback_ps)
{
  double max_slew_ps = fallback_ps;
  bool used = false;
  if (inst == nullptr) {
    return std::max(0.0, fallback_ps);
  }
  for (auto* iterm : inst->getITerms()) {
    if (iterm == nullptr) {
      continue;
    }
    const auto io_type = iterm->getIoType();
    if (io_type != odb::dbIoType::INPUT && io_type != odb::dbIoType::INOUT) {
      continue;
    }
    const double rise_slew = timing.getPinSlew(iterm, ord::Timing::Rise, ord::Timing::Max);
    const double fall_slew = timing.getPinSlew(iterm, ord::Timing::Fall, ord::Timing::Max);
    if (!staScalarIsUsable(rise_slew) && !staScalarIsUsable(fall_slew)) {
      continue;
    }
    const double slew_ps = staTimeToPsOrDefault(
        time_unit,
        std::max(staScalarIsUsable(rise_slew) ? rise_slew : 0.0,
                 staScalarIsUsable(fall_slew) ? fall_slew : 0.0),
        0.0);
    max_slew_ps = used ? std::max(max_slew_ps, slew_ps) : slew_ps;
    used = true;
  }
  return std::max(0.0, max_slew_ps);
}

inline odb::dbITerm* diffGuidedBatchFindNetDriverITerm(odb::dbNet* net)
{
  if (net == nullptr) {
    return nullptr;
  }
  for (auto* iterm : net->getITerms()) {
    if (iterm == nullptr) {
      continue;
    }
    const auto io_type = iterm->getIoType();
    if (io_type == odb::dbIoType::OUTPUT || io_type == odb::dbIoType::INOUT) {
      return iterm;
    }
  }
  return nullptr;
}

inline double diffGuidedBatchNPathWeightForInst(const DiffGuidedBatchNPathSnapshot* snapshot,
                                         int inst_id)
{
  if (snapshot == nullptr || !snapshot->valid || inst_id < 0
      || inst_id >= static_cast<int>(snapshot->inst_npath_weight_by_id.size())) {
    return 1.0;
  }
  const double weight = snapshot->inst_npath_weight_by_id[inst_id];
  return std::isfinite(weight) ? std::max(1.0, weight) : 1.0;
}

inline int diffGuidedBatchInstIdForDbInst(const DiffGuidedBatchNPathSnapshot* snapshot,
                                   odb::dbInst* inst)
{
  if (snapshot == nullptr || inst == nullptr) {
    return -1;
  }
  const auto it = snapshot->inst_id_by_db_inst.find(inst);
  return it == snapshot->inst_id_by_db_inst.end() ? -1 : it->second;
}

inline DiffGuidedBatchCellReplaceSlackDeltaResult diffGuidedBatchCellReplaceSlackDeltaEvaluator(
    sta::dbSta* sta,
    ord::Timing& timing,
    const sta::Unit* time_unit,
    const sta::Unit* cap_unit,
    sta::dbNetwork* network,
    sta::Corner* corner,
    odb::dbInst* inst,
    odb::dbMaster* current_master,
    odb::dbMaster* target_master,
    const DiffGuidedBatchCurrentLocalTimingContext& current_context,
    const DiffGuidedBatchCppLocalDelaySlewEstimatorResult& downstream_dcalc,
    double heuristic_fanin_penalty_ps_per_pf,
    const std::string& mode,
    const std::string& slack_delta_weight_mode,
    const DiffGuidedBatchNPathSnapshot* npath_snapshot,
    std::vector<double>& fanin_raw_delta_by_inst_id,
    std::vector<double>& fanin_signed_delta_by_inst_id,
    std::vector<int>& fanin_seen_epoch_by_inst_id,
    std::vector<int>& fanin_touched_inst_ids,
    int fanin_scratch_epoch)
{
  DiffGuidedBatchCellReplaceSlackDeltaResult result;
  result.heuristic_fanin_penalty_ps = 0.0;
  result.slack_delta_weight_mode = slack_delta_weight_mode;
  if (sta == nullptr || network == nullptr || inst == nullptr || current_master == nullptr
      || target_master == nullptr) {
    result.fallback_reason = "missing_sta_network_instance_or_master";
    return result;
  }
  if (mode == "off") {
    result.estimator_source = "off";
    result.fanin_source = "off";
    result.fallback_reason = "slack_delta_evaluator_mode_off";
    return result;
  }

  auto* current_cell = network->dbToSta(current_master);
  auto* target_cell = network->dbToSta(target_master);
  auto* current_liberty_cell = current_cell == nullptr ? nullptr : network->libertyCell(current_cell);
  auto* target_liberty_cell = target_cell == nullptr ? nullptr : network->libertyCell(target_cell);
  if (current_liberty_cell == nullptr || target_liberty_cell == nullptr) {
    result.fallback_reason = "missing_current_or_target_liberty_cell";
    return result;
  }

  if (current_context.output_slack_used && downstream_dcalc.used) {
    const int downstream_inst_id = diffGuidedBatchInstIdForDbInst(npath_snapshot, inst);
    const double downstream_npath_weight =
        slack_delta_weight_mode == "npath_weighted_tns"
            ? diffGuidedBatchNPathWeightForInst(npath_snapshot, downstream_inst_id)
            : 1.0;
    DiffGuidedBatchSlackDeltaPoint downstream_point = diffGuidedBatchMakeSlackDeltaPoint(
        "downstream_output",
        downstream_inst_id,
        current_context.output_slew_limit_port_name.empty()
            ? current_context.output_cap_limit_port_name
            : current_context.output_slew_limit_port_name,
        current_master->getName(),
        target_master->getName(),
        current_context.output_slack_ps,
        current_context.output_arrival_ps,
        current_context.output_required_ps,
        downstream_dcalc.selected_current_delay_ps,
        downstream_dcalc.selected_target_delay_ps,
        downstream_dcalc.selected_current_slew_ps,
        downstream_dcalc.selected_target_slew_ps,
        "downstream_arc_lut_dcalc",
        downstream_npath_weight);
    if (downstream_point.used) {
      result.downstream_slack_improvement_ps += downstream_point.slack_violation_delta_ps;
      result.downstream_weighted_slack_improvement_ps +=
          downstream_point.weighted_slack_violation_delta_ps;
      result.downstream_old_slack_ps = downstream_point.old_slack_ps;
      result.downstream_new_slack_ps = downstream_point.new_slack_ps;
      result.downstream_delta_delay_ps = downstream_point.delta_delay_ps;
      result.downstream_npath_weight = downstream_npath_weight;
      result.downstream_dcalc_source = downstream_point.dcalc_source;
      result.downstream_point_count += 1;
      result.dcalc_query_count += std::max(1, downstream_dcalc.compared_arc_count);
    } else {
      result.fallback_point_count += 1;
    }
    result.points.push_back(downstream_point);
  }

  const bool use_lut_dcalc = mode == "lut_dcalc";
  const bool use_heuristic = mode == "heuristic" || mode == "lut_dcalc";
  auto* dcalc_ap = corner == nullptr ? nullptr : corner->findDcalcAnalysisPt(sta::MinMax::max());
  const sta::Pvt* pvt = dcalc_ap == nullptr ? nullptr : dcalc_ap->operatingConditions();
  const float fallback_input_slew_sta = static_cast<float>(
      current_context.input_slew_ps > 0.0 && time_unit != nullptr
          ? psToStaTimeOrDefault(time_unit, current_context.input_slew_ps, 0.0)
          : 0.0);
  fanin_touched_inst_ids.clear();
  std::unordered_set<odb::dbNet*> visited_fanin_nets;

  for (auto* target_iterm : inst->getITerms()) {
    if (target_iterm == nullptr) {
      continue;
    }
    const auto target_io_type = target_iterm->getIoType();
    if (target_io_type != odb::dbIoType::INPUT && target_io_type != odb::dbIoType::INOUT) {
      continue;
    }
    auto* mterm = target_iterm->getMTerm();
    if (mterm == nullptr) {
      continue;
    }
    sta::LibertyPort* current_input_port =
        current_liberty_cell->findLibertyPort(mterm->getName().c_str());
    sta::LibertyPort* target_input_port =
        target_liberty_cell->findLibertyPort(mterm->getName().c_str());
    if (current_input_port == nullptr || target_input_port == nullptr
        || current_input_port->isPwrGnd() || target_input_port->isPwrGnd()) {
      continue;
    }
    const sta::Units* current_units = current_input_port->libertyLibrary() == nullptr
                                          ? nullptr
                                          : current_input_port->libertyLibrary()->units();
    const sta::Units* target_units = target_input_port->libertyLibrary() == nullptr
                                         ? nullptr
                                         : target_input_port->libertyLibrary()->units();
    const double current_input_cap_pf = staCapToPfOrDefault(
        current_units == nullptr ? cap_unit : current_units->capacitanceUnit(),
        current_input_port->capacitance(),
        0.0);
    const double target_input_cap_pf = staCapToPfOrDefault(
        target_units == nullptr ? cap_unit : target_units->capacitanceUnit(),
        target_input_port->capacitance(),
        0.0);
    const double input_cap_delta_pf = target_input_cap_pf - current_input_cap_pf;
    if (input_cap_delta_pf > 0.0) {
      result.positive_input_cap_delta_count += 1;
    } else if (input_cap_delta_pf < 0.0) {
      result.negative_input_cap_delta_count += 1;
    }

    odb::dbNet* net = target_iterm->getNet();
    if (net == nullptr || !isSignalNet(net)) {
      continue;
    }
    if (visited_fanin_nets.insert(net).second) {
      result.fanin_net_count += 1;
    }

    bool net_delta_used = false;
    std::string net_source = "unavailable";
    if (use_lut_dcalc && dcalc_ap != nullptr) {
      odb::dbITerm* driver_iterm = diffGuidedBatchFindNetDriverITerm(net);
      odb::dbInst* driver_inst = driver_iterm == nullptr ? nullptr : driver_iterm->getInst();
      odb::dbMaster* driver_master = driver_inst == nullptr ? nullptr : driver_inst->getMaster();
      auto* driver_cell = (driver_master == nullptr) ? nullptr : network->dbToSta(driver_master);
      auto* driver_liberty_cell = driver_cell == nullptr ? nullptr : network->libertyCell(driver_cell);
      auto* driver_mterm = driver_iterm == nullptr ? nullptr : driver_iterm->getMTerm();
      sta::LibertyPort* driver_output_port =
          (driver_liberty_cell == nullptr || driver_mterm == nullptr)
              ? nullptr
              : driver_liberty_cell->findLibertyPort(driver_mterm->getName().c_str());
      if (driver_inst != nullptr && driver_liberty_cell != nullptr && driver_output_port != nullptr) {
        const double old_load_pf = staCapToPfOrDefault(
            cap_unit,
            corner == nullptr ? 0.0 : timing.getNetCap(net, corner, ord::Timing::Max),
            0.0);
        const double new_load_pf = std::max(0.0, old_load_pf + input_cap_delta_pf);
        const float old_load_sta = static_cast<float>(
            pfToStaCapOrDefault(cap_unit, old_load_pf, 0.0));
        const float new_load_sta = static_cast<float>(
            pfToStaCapOrDefault(cap_unit, new_load_pf, 0.0));
        const double driver_input_slew_ps =
            diffGuidedBatchMaxInputSlewPs(timing, time_unit, driver_inst, current_context.input_slew_ps);
        const float in_slew_sta = driver_input_slew_ps > 0.0 && time_unit != nullptr
                                      ? static_cast<float>(
                                            psToStaTimeOrDefault(
                                                time_unit, driver_input_slew_ps, 0.0))
                                      : fallback_input_slew_sta;
        double selected_delta_delay_ps = 0.0;
        double selected_old_delay_ps = 0.0;
        double selected_new_delay_ps = 0.0;
        double selected_old_slew_ps = 0.0;
        double selected_new_slew_ps = 0.0;
        bool selected_arc = false;
        for (sta::TimingArcSet* arc_set : driver_liberty_cell->timingArcSets()) {
          if (arc_set == nullptr || arc_set->role() == nullptr
              || arc_set->role()->isTimingCheck()) {
            continue;
          }
          for (sta::TimingArc* arc : arc_set->arcs()) {
            if (arc == nullptr || arc->to() != driver_output_port) {
              continue;
            }
            sta::GateTableModel* model = arc->gateTableModel(dcalc_ap);
            if (model == nullptr) {
              continue;
            }
            sta::ArcDelay old_delay = 0.0;
            sta::Slew old_slew = 0.0;
            sta::ArcDelay new_delay = 0.0;
            sta::Slew new_slew = 0.0;
            model->gateDelay(pvt, in_slew_sta, old_load_sta, false, old_delay, old_slew);
            model->gateDelay(pvt, in_slew_sta, new_load_sta, false, new_delay, new_slew);
            const double old_delay_ps = staTimeToPsOrDefault(time_unit, old_delay, 0.0);
            const double new_delay_ps = staTimeToPsOrDefault(time_unit, new_delay, 0.0);
            const double delta_delay_ps = new_delay_ps - old_delay_ps;
            result.dcalc_query_count += 1;
            if (!selected_arc || std::abs(delta_delay_ps) > std::abs(selected_delta_delay_ps)) {
              selected_arc = true;
              selected_delta_delay_ps = delta_delay_ps;
              selected_old_delay_ps = old_delay_ps;
              selected_new_delay_ps = new_delay_ps;
              selected_old_slew_ps = staTimeToPsOrDefault(time_unit, old_slew, 0.0);
              selected_new_slew_ps = staTimeToPsOrDefault(time_unit, new_slew, 0.0);
            }
          }
        }
        if (selected_arc) {
          net_delta_used = true;
          net_source = "fi_arc_lut_dcalc";
          for (auto* sink_iterm : net->getITerms()) {
            if (sink_iterm == nullptr || sink_iterm == driver_iterm) {
              continue;
            }
            const auto sink_io_type = sink_iterm->getIoType();
            if (sink_io_type != odb::dbIoType::INPUT && sink_io_type != odb::dbIoType::INOUT) {
              continue;
            }
            auto* sink_inst = sink_iterm->getInst();
            if (sink_inst == nullptr) {
              continue;
            }
            double slack_ps = 0.0;
            double arrival_ps = 0.0;
            double required_ps = 0.0;
            if (!diffGuidedBatchWorstSlackAtITerm(
                    timing, time_unit, sink_iterm, slack_ps, arrival_ps, required_ps)) {
              result.fallback_point_count += 1;
              continue;
            }
            const int sink_inst_id =
                diffGuidedBatchInstIdForDbInst(npath_snapshot, sink_inst);
            const double sink_npath_weight =
                slack_delta_weight_mode == "npath_weighted_tns"
                    ? diffGuidedBatchNPathWeightForInst(npath_snapshot, sink_inst_id)
                    : 1.0;
            const std::string point_pin_name =
                std::string(sink_inst->getName()) + "/" +
                (sink_iterm->getMTerm() == nullptr ? "" : sink_iterm->getMTerm()->getName());
            DiffGuidedBatchSlackDeltaPoint point = diffGuidedBatchMakeSlackDeltaPoint(
                "fanin_sink",
                sink_inst_id,
                point_pin_name,
                driver_master == nullptr ? "" : driver_master->getName(),
                driver_master == nullptr ? "" : driver_master->getName(),
                slack_ps,
                arrival_ps,
                required_ps,
                selected_old_delay_ps,
                selected_new_delay_ps,
                selected_old_slew_ps,
                selected_new_slew_ps,
                net_source,
                sink_npath_weight);
            if (!point.used) {
              result.fallback_point_count += 1;
              continue;
            }
            result.points.push_back(point);
            result.fanin_point_count += 1;
            const double weighted_signed_delta = point.weighted_slack_violation_delta_ps;
            if (sink_inst_id >= 0
                && sink_inst_id < static_cast<int>(fanin_signed_delta_by_inst_id.size())) {
              if (fanin_seen_epoch_by_inst_id[sink_inst_id] != fanin_scratch_epoch) {
                fanin_seen_epoch_by_inst_id[sink_inst_id] = fanin_scratch_epoch;
                fanin_raw_delta_by_inst_id[sink_inst_id] = point.slack_violation_delta_ps;
                fanin_signed_delta_by_inst_id[sink_inst_id] = weighted_signed_delta;
                fanin_touched_inst_ids.push_back(sink_inst_id);
              } else {
                fanin_raw_delta_by_inst_id[sink_inst_id] =
                    std::min(fanin_raw_delta_by_inst_id[sink_inst_id],
                             point.slack_violation_delta_ps);
                fanin_signed_delta_by_inst_id[sink_inst_id] =
                    std::min(fanin_signed_delta_by_inst_id[sink_inst_id],
                             weighted_signed_delta);
              }
            } else {
              result.fallback_point_count += 1;
            }
          }
        }
      }
    }

    if (!net_delta_used && use_heuristic) {
      const double heuristic_penalty =
          std::max(0.0, input_cap_delta_pf) * heuristic_fanin_penalty_ps_per_pf;
      result.heuristic_fanin_penalty_ps += heuristic_penalty;
      net_source = "heuristic_input_cap_ps_per_pf";
      if (heuristic_penalty <= 0.0) {
        continue;
      }
      for (auto* sink_iterm : net->getITerms()) {
        auto* sink_inst = sink_iterm == nullptr ? nullptr : sink_iterm->getInst();
        if (sink_iterm == nullptr || sink_inst == nullptr) {
          continue;
        }
        const auto sink_io_type = sink_iterm->getIoType();
        if (sink_io_type != odb::dbIoType::INPUT && sink_io_type != odb::dbIoType::INOUT) {
          continue;
        }
        double slack_ps = 0.0;
        double arrival_ps = 0.0;
        double required_ps = 0.0;
        if (!diffGuidedBatchWorstSlackAtITerm(
                timing, time_unit, sink_iterm, slack_ps, arrival_ps, required_ps)) {
          continue;
        }
        const double signed_delta =
            -diffGuidedBatchPenaltyFromSlackDegradation(slack_ps, heuristic_penalty);
        const int sink_inst_id = diffGuidedBatchInstIdForDbInst(npath_snapshot, sink_inst);
        const double sink_npath_weight =
            slack_delta_weight_mode == "npath_weighted_tns"
                ? diffGuidedBatchNPathWeightForInst(npath_snapshot, sink_inst_id)
                : 1.0;
        const double weighted_signed_delta = signed_delta * sink_npath_weight;
        if (sink_inst_id >= 0
            && sink_inst_id < static_cast<int>(fanin_signed_delta_by_inst_id.size())) {
          if (fanin_seen_epoch_by_inst_id[sink_inst_id] != fanin_scratch_epoch) {
            fanin_seen_epoch_by_inst_id[sink_inst_id] = fanin_scratch_epoch;
            fanin_raw_delta_by_inst_id[sink_inst_id] = signed_delta;
            fanin_signed_delta_by_inst_id[sink_inst_id] = weighted_signed_delta;
            fanin_touched_inst_ids.push_back(sink_inst_id);
          } else {
            fanin_raw_delta_by_inst_id[sink_inst_id] =
                std::min(fanin_raw_delta_by_inst_id[sink_inst_id], signed_delta);
            fanin_signed_delta_by_inst_id[sink_inst_id] =
                std::min(fanin_signed_delta_by_inst_id[sink_inst_id], weighted_signed_delta);
          }
        }
      }
    }
  }

  for (const int inst_id : fanin_touched_inst_ids) {
    if (inst_id < 0 || inst_id >= static_cast<int>(fanin_signed_delta_by_inst_id.size())) {
      continue;
    }
    const double raw_signed_delta = fanin_raw_delta_by_inst_id[inst_id];
    const double weighted_signed_delta = fanin_signed_delta_by_inst_id[inst_id];
    result.fanin_slack_delta_ps += raw_signed_delta;
    result.fanin_weighted_slack_delta_ps += weighted_signed_delta;
    result.fanin_slack_penalty_ps += std::max(
        0.0,
        -(slack_delta_weight_mode == "npath_weighted_tns" ? weighted_signed_delta
                                                          : raw_signed_delta));
  }
  result.affected_fo_cell_count = static_cast<int>(fanin_touched_inst_ids.size());
  result.net_slack_obj_delta_ps =
      result.downstream_slack_improvement_ps + result.fanin_slack_delta_ps;
  result.weighted_net_tns_delta_ps =
      result.downstream_weighted_slack_improvement_ps + result.fanin_weighted_slack_delta_ps;
  if (slack_delta_weight_mode != "npath_weighted_tns") {
    result.fanin_weighted_slack_delta_ps = result.fanin_slack_delta_ps;
    result.downstream_weighted_slack_improvement_ps = result.downstream_slack_improvement_ps;
    result.weighted_net_tns_delta_ps = result.net_slack_obj_delta_ps;
  }
  result.used = result.downstream_point_count > 0 || result.fanin_point_count > 0
                || result.affected_fo_cell_count > 0;
  result.estimator_source = result.used ? "cell_replace_slack_delta_evaluator"
                                        : "cell_replace_slack_delta_evaluator_unavailable";
  if (result.fanin_point_count > 0) {
    result.fanin_source = "fi_arc_lut_dcalc";
  } else if (result.heuristic_fanin_penalty_ps > 0.0 || mode == "heuristic") {
    result.fanin_source = "heuristic_input_cap_ps_per_pf";
  } else {
    result.fanin_source = "unavailable";
  }
  result.fallback_reason = result.used ? "" : "missing_downstream_or_fanin_slack_delta_points";
  return result;
}

inline DiffGuidedBatchFaninNeighborhoodPenaltyResult diffGuidedBatchFaninNeighborhoodPenalty(
    ord::Timing& timing,
    const sta::Unit* time_unit,
    const sta::Unit* cap_unit,
    sta::dbNetwork* network,
    odb::dbInst* inst,
    odb::dbMaster* current_master,
    odb::dbMaster* target_master,
    double fanin_penalty_ps_per_pf)
{
  DiffGuidedBatchFaninNeighborhoodPenaltyResult result;
  result.fanin_penalty_ps_per_pf = fanin_penalty_ps_per_pf;
  if (network == nullptr || inst == nullptr || current_master == nullptr || target_master == nullptr) {
    result.fallback_reason = "missing_network_instance_or_master";
    return result;
  }
  auto* current_cell = network->dbToSta(current_master);
  auto* target_cell = network->dbToSta(target_master);
  auto* current_liberty_cell = current_cell == nullptr ? nullptr : network->libertyCell(current_cell);
  auto* target_liberty_cell = target_cell == nullptr ? nullptr : network->libertyCell(target_cell);
  if (current_liberty_cell == nullptr || target_liberty_cell == nullptr) {
    result.fallback_reason = "missing_liberty_cell";
    return result;
  }

  std::unordered_map<odb::dbInst*, double> penalty_by_affected_cell;
  std::unordered_set<odb::dbNet*> visited_fanin_nets;
  for (auto* iterm : inst->getITerms()) {
    auto* mterm = iterm == nullptr ? nullptr : iterm->getMTerm();
    if (iterm == nullptr || mterm == nullptr) {
      continue;
    }
    if (iterm->getIoType() != odb::dbIoType::INPUT
        && iterm->getIoType() != odb::dbIoType::INOUT) {
      continue;
    }

    sta::LibertyPort* current_liberty_port =
        current_liberty_cell->findLibertyPort(mterm->getName().c_str());
    sta::LibertyPort* target_liberty_port =
        target_liberty_cell->findLibertyPort(mterm->getName().c_str());
    if (current_liberty_port == nullptr || target_liberty_port == nullptr
        || current_liberty_port->isPwrGnd() || target_liberty_port->isPwrGnd()) {
      continue;
    }
    const sta::Units* current_units = current_liberty_port->libertyLibrary() == nullptr
                                          ? nullptr
                                          : current_liberty_port->libertyLibrary()->units();
    const sta::Units* target_units = target_liberty_port->libertyLibrary() == nullptr
                                         ? nullptr
                                         : target_liberty_port->libertyLibrary()->units();
    const double current_cap_pf = staCapToPfOrDefault(
        current_units == nullptr ? cap_unit : current_units->capacitanceUnit(),
        current_liberty_port->capacitance(),
        0.0);
    const double target_cap_pf = staCapToPfOrDefault(
        target_units == nullptr ? cap_unit : target_units->capacitanceUnit(),
        target_liberty_port->capacitance(),
        0.0);
    const double positive_input_cap_delta_pf = std::max(0.0, target_cap_pf - current_cap_pf);
    if (positive_input_cap_delta_pf <= 0.0) {
      continue;
    }
    result.positive_input_cap_delta_count += 1;

    odb::dbNet* net = iterm->getNet();
    if (net == nullptr || !isSignalNet(net)) {
      continue;
    }
    if (visited_fanin_nets.insert(net).second) {
      result.fanin_net_count += 1;
    }

    const double fi_delay_degradation_ps =
        positive_input_cap_delta_pf * fanin_penalty_ps_per_pf;

    for (auto* sink_iterm : net->getITerms()) {
      auto* sink_inst = sink_iterm == nullptr ? nullptr : sink_iterm->getInst();
      if (sink_iterm == nullptr || sink_inst == nullptr || sink_inst->isFixed()) {
        continue;
      }
      const auto sink_io_type = sink_iterm->getIoType();
      if (sink_io_type != odb::dbIoType::INPUT && sink_io_type != odb::dbIoType::INOUT) {
        continue;
      }
      const double rise_slack = timing.getPinSlack(sink_iterm, ord::Timing::Rise, ord::Timing::Max);
      const double fall_slack = timing.getPinSlack(sink_iterm, ord::Timing::Fall, ord::Timing::Max);
      const bool rise_usable = staScalarIsUsable(rise_slack);
      const bool fall_usable = staScalarIsUsable(fall_slack);
      if (!rise_usable && !fall_usable) {
        continue;
      }
      const double slack = !rise_usable ? fall_slack
                                        : (!fall_usable ? rise_slack
                                                        : std::min(rise_slack, fall_slack));
      const double slack_ps = staTimeToPsOrDefault(time_unit, slack, 0.0);
      const double cell_penalty =
          diffGuidedBatchPenaltyFromSlackDegradation(slack_ps, fi_delay_degradation_ps);
      const double existing_cell_penalty =
          penalty_by_affected_cell.count(sink_inst) == 0 ? 0.0 : penalty_by_affected_cell[sink_inst];
      penalty_by_affected_cell[sink_inst] = std::max(existing_cell_penalty, cell_penalty);
    }
  }

  for (const auto& [_, penalty] : penalty_by_affected_cell) {
    result.local_predicted_fanin_neighborhood_penalty += std::max(0.0, penalty);
  }
  result.affected_fo_cell_count = static_cast<int>(penalty_by_affected_cell.size());
  if (result.positive_input_cap_delta_count > 0 || result.fanin_net_count > 0) {
    result.used = true;
    result.local_predicted_fanin_penalty_source = "cpp_hot_path_fanin_neighborhood_penalty";
    result.fallback_reason.clear();
  }
  return result;
}

inline sta::LibertyPort* diffGuidedBatchFindLibertyPort(sta::dbNetwork* network,
                                                 odb::dbMaster* master,
                                                 const std::string& port_name)
{
  if (network == nullptr || master == nullptr || port_name.empty()) {
    return nullptr;
  }
  auto* cell = network->dbToSta(master);
  auto* liberty_cell = cell == nullptr ? nullptr : network->libertyCell(cell);
  return liberty_cell == nullptr ? nullptr : liberty_cell->findLibertyPort(port_name.c_str());
}

inline double diffGuidedBatchResolveLibPinCapLimitPf(const sta::Unit* cap_unit,
                                              sta::LibertyPort* liberty_port)
{
  if (liberty_port == nullptr) {
    return 0.0;
  }
  float limit = 0.0f;
  bool exists = false;
  liberty_port->capacitanceLimit(sta::MinMax::max(), limit, exists);
  if (!exists && liberty_port->libertyLibrary() != nullptr) {
    liberty_port->libertyLibrary()->defaultMaxCapacitance(limit, exists);
  }
  return exists ? staCapToPfOrDefault(cap_unit, limit, 0.0) : 0.0;
}

inline sta::TimingArc* diffGuidedBatchFindMatchingLibertyArc(sta::LibertyCell* liberty_cell,
                                                      const sta::TimingArc* target_arc)
{
  if (liberty_cell == nullptr || target_arc == nullptr || target_arc->from() == nullptr
      || target_arc->to() == nullptr || target_arc->fromEdge() == nullptr
      || target_arc->toEdge() == nullptr) {
    return nullptr;
  }
  sta::LibertyPort* current_from = liberty_cell->findLibertyPort(target_arc->from()->name());
  sta::LibertyPort* current_to = liberty_cell->findLibertyPort(target_arc->to()->name());
  if (current_from == nullptr || current_to == nullptr) {
    return nullptr;
  }
  for (sta::TimingArcSet* current_set : liberty_cell->timingArcSets(current_from, current_to)) {
    if (current_set == nullptr || current_set->role() == nullptr
        || current_set->role()->isTimingCheck()) {
      continue;
    }
    for (sta::TimingArc* current_arc : current_set->arcs()) {
      if (current_arc == nullptr || current_arc->fromEdge() == nullptr
          || current_arc->toEdge() == nullptr) {
        continue;
      }
      if (current_arc->fromEdge() == target_arc->fromEdge()
          && current_arc->toEdge() == target_arc->toEdge()) {
        return current_arc;
      }
    }
  }
  return nullptr;
}

inline DiffGuidedBatchCppLocalDelaySlewEstimatorResult
diffGuidedBatchCppLocalDelaySlewEstimator(sta::dbSta* sta,
                                          sta::dbNetwork* network,
                                          sta::Corner* corner,
                                          odb::dbMaster* current_master,
                                          odb::dbMaster* target_master,
                                          const DiffGuidedBatchCurrentLocalTimingContext& current_context,
                                          const CompactBridgeActionProposal& action,
                                          double local_delta_cap_for_trial)
{
  DiffGuidedBatchCppLocalDelaySlewEstimatorResult result;
  if (sta == nullptr || network == nullptr || current_master == nullptr || target_master == nullptr) {
    result.fallback_reason = "missing_sta_network_or_master";
    return result;
  }
  auto* current_cell = network->dbToSta(current_master);
  auto* target_cell = network->dbToSta(target_master);
  auto* current_liberty_cell = current_cell == nullptr ? nullptr : network->libertyCell(current_cell);
  auto* target_liberty_cell = target_cell == nullptr ? nullptr : network->libertyCell(target_cell);
  if (current_liberty_cell == nullptr || target_liberty_cell == nullptr) {
    result.fallback_reason = "missing_liberty_cell";
    return result;
  }

  auto* dcalc_ap = corner == nullptr ? nullptr : corner->findDcalcAnalysisPt(sta::MinMax::max());
  const sta::Pvt* pvt = dcalc_ap == nullptr ? nullptr : dcalc_ap->operatingConditions();
  const sta::Units* units = sta->units();
  const sta::Unit* time_unit = units == nullptr ? nullptr : units->timeUnit();
  const sta::Unit* cap_unit = units == nullptr ? nullptr : units->capacitanceUnit();
  const double input_slew_ps = current_context.used && current_context.input_slew_ps > 0.0
                                   ? current_context.input_slew_ps
                                   : std::max(0.0, action.local_delta_slew_ps);
  const float in_slew = static_cast<float>(
      input_slew_ps > 0.0 && time_unit != nullptr
          ? psToStaTimeOrDefault(time_unit, input_slew_ps, 0.0)
          : 0.0);
  const double fallback_cap_pf = current_context.used && current_context.load_cap_pf > 0.0
                                     ? current_context.load_cap_pf
                                     : (local_delta_cap_for_trial > 0.0
                                            ? local_delta_cap_for_trial
                                            : std::max(0.0, action.local_delta_cap));
  const float load_cap = static_cast<float>(pfToStaCapOrDefault(cap_unit, fallback_cap_pf, 0.0));

  for (sta::TimingArcSet* target_set : target_liberty_cell->timingArcSets()) {
    if (target_set == nullptr || target_set->role() == nullptr
        || target_set->role()->isTimingCheck()) {
      continue;
    }
    for (sta::TimingArc* target_arc : target_set->arcs()) {
      if (target_arc == nullptr) {
        continue;
      }
      auto* current_arc = diffGuidedBatchFindMatchingLibertyArc(current_liberty_cell, target_arc);
      if (current_arc == nullptr) {
        continue;
      }
      sta::GateTableModel* current_model = current_arc->gateTableModel(dcalc_ap);
      sta::GateTableModel* target_model = target_arc->gateTableModel(dcalc_ap);
      if (current_model == nullptr || target_model == nullptr) {
        continue;
      }
      sta::ArcDelay current_delay = 0.0;
      sta::Slew current_slew = 0.0;
      sta::ArcDelay target_delay = 0.0;
      sta::Slew target_slew = 0.0;
      current_model->gateDelay(pvt, in_slew, load_cap, false, current_delay, current_slew);
      target_model->gateDelay(pvt, in_slew, load_cap, false, target_delay, target_slew);
      const double current_delay_ps = staTimeToPsOrDefault(time_unit, current_delay, 0.0);
      const double target_delay_ps = staTimeToPsOrDefault(time_unit, target_delay, 0.0);
      const double current_slew_ps = staTimeToPsOrDefault(time_unit, current_slew, 0.0);
      const double target_slew_ps = staTimeToPsOrDefault(time_unit, target_slew, 0.0);
      const double signed_delay_delta = current_delay_ps - target_delay_ps;
      const double signed_slew_delta = current_slew_ps - target_slew_ps;
      const double delay_delta = std::max(0.0, signed_delay_delta);
      const double slew_delta = std::max(0.0, -signed_slew_delta);
      result.local_delta_delay_ps = std::max(result.local_delta_delay_ps, delay_delta);
      result.local_delta_slew_ps = std::max(result.local_delta_slew_ps, slew_delta);
      if (std::abs(signed_delay_delta) > std::abs(result.signed_delay_delta)
          || result.compared_arc_count == 0) {
        result.signed_delay_delta = signed_delay_delta;
        result.selected_current_delay_ps = current_delay_ps;
        result.selected_target_delay_ps = target_delay_ps;
      }
      if (std::abs(signed_slew_delta) > std::abs(result.signed_slew_delta)
          || result.compared_arc_count == 0) {
        result.signed_slew_delta = signed_slew_delta;
        result.selected_current_slew_ps = current_slew_ps;
        result.selected_target_slew_ps = target_slew_ps;
      }
      result.compared_arc_count += 1;
      if (delay_delta > 0.0) {
        result.positive_delay_arc_count += 1;
      }
      if (slew_delta > 0.0) {
        result.positive_slew_arc_count += 1;
      }
    }
  }

  if (result.compared_arc_count > 0) {
    result.used = true;
    result.estimator_source = "cpp_hot_path_liberty_table_delay_slew_estimator";
    result.fallback_reason.clear();
  }
  return result;
}

}  // namespace impl
}  // namespace placeio_openroad
}  // namespace dreamplace
