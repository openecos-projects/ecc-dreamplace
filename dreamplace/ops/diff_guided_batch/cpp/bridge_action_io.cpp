#include "diff_guided_batch/cpp/bridge_action_io.h"

#include <algorithm>
#include <unordered_set>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

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

int64_t pyInt64OrDefault(const py::dict& value, const char* key, int64_t fallback = -1)
{
  if (!value.contains(key) || value[key].is_none()) {
    return fallback;
  }
  if (py::isinstance<py::str>(value[key])) {
    return fallback;
  }
  return value[key].cast<int64_t>();
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

std::vector<std::string> pyStringListOrEmpty(const py::dict& value, const char* key)
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

std::vector<int> pyIntListOrEmpty(const py::dict& value, const char* key)
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
std::vector<T> pyArrayBufferToVector(const py::array_t<T, py::array::c_style | py::array::forcecast>& array)
{
  const py::buffer_info buffer = array.request();
  if (buffer.ndim != 1) {
    throw py::cast_error("expected a one-dimensional compact proposal buffer");
  }
  const auto* data = static_cast<const T*>(buffer.ptr);
  return std::vector<T>(data, data + buffer.shape[0]);
}

std::vector<int> pyIntArrayToVector(const py::handle& object)
{
  const py::array_t<int, py::array::c_style | py::array::forcecast> array(
      py::reinterpret_borrow<py::object>(object));
  return pyArrayBufferToVector<int>(array);
}

std::vector<int> pyIntArrayOrListOrEmpty(const py::dict& value, const char* key)
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

std::vector<int64_t> pyInt64ListOrEmpty(const py::dict& value, const char* key)
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

std::vector<int64_t> pyInt64ArrayToVector(const py::handle& object)
{
  const py::array_t<int64_t, py::array::c_style | py::array::forcecast> array(
      py::reinterpret_borrow<py::object>(object));
  return pyArrayBufferToVector<int64_t>(array);
}

std::vector<int64_t> pyInt64ArrayOrListOrEmpty(const py::dict& value, const char* key)
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

std::vector<double> pyDoubleListOrEmpty(const py::dict& value, const char* key)
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

std::vector<double> pyDoubleArrayToVector(const py::handle& object)
{
  const py::array_t<double, py::array::c_style | py::array::forcecast> array(
      py::reinterpret_borrow<py::object>(object));
  return pyArrayBufferToVector<double>(array);
}

std::vector<double> pyDoubleArrayOrListOrEmpty(const py::dict& value, const char* key)
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

std::string compactStringTableValue(const CompactBridgeActionProposalBuffers& buffers,
                                    int table_id,
                                    const std::string& fallback = "")
{
  if (table_id < 0 || table_id >= static_cast<int>(buffers.string_table.size())) {
    return fallback;
  }
  return buffers.string_table[table_id];
}

std::vector<std::string> compactLegalMasterCandidates(
    const CompactBridgeActionProposalBuffers& buffers,
    std::size_t index)
{
  std::vector<std::string> candidates;
  if (index + 1 >= buffers.legal_master_candidate_indptr.size()) {
    return candidates;
  }
  const int begin = std::max(0, buffers.legal_master_candidate_indptr[index]);
  const int end = std::max(begin, buffers.legal_master_candidate_indptr[index + 1]);
  for (int candidate_index = begin; candidate_index < end; ++candidate_index) {
    const std::string fallback = vectorValueOrDefault<std::string>(
        buffers.legal_master_candidate_names, candidate_index, "");
    if (candidate_index < static_cast<int>(buffers.legal_master_candidate_name_ids.size())) {
      candidates.push_back(compactStringTableValue(
          buffers, buffers.legal_master_candidate_name_ids[candidate_index], fallback));
      continue;
    }
    if (candidate_index < static_cast<int>(buffers.legal_master_candidate_names.size())) {
      candidates.push_back(fallback);
    }
  }
  return candidates;
}

template <typename T>
std::vector<T> compactLegalCandidateValues(
    const CompactBridgeActionProposalBuffers& buffers,
    const std::vector<T>& flat_values,
    std::size_t index,
    const T& fallback)
{
  std::vector<T> candidates;
  if (index + 1 >= buffers.legal_master_candidate_indptr.size()) {
    return candidates;
  }
  const int begin = std::max(0, buffers.legal_master_candidate_indptr[index]);
  const int end = std::max(begin, buffers.legal_master_candidate_indptr[index + 1]);
  candidates.reserve(static_cast<std::size_t>(std::max(0, end - begin)));
  for (int candidate_index = begin; candidate_index < end; ++candidate_index) {
    candidates.push_back(vectorValueOrDefault<T>(
        flat_values, static_cast<std::size_t>(candidate_index), fallback));
  }
  return candidates;
}

}  // namespace

int vectorIndexedIntOrDefault(const std::vector<int>& values, int index, int fallback)
{
  if (index < 0 || index >= static_cast<int>(values.size())) {
    return fallback;
  }
  return values[index];
}

CompactBridgeActionProposal parseActionProposal(const py::dict& action)
{
  CompactBridgeActionProposal proposal;
  proposal.kind = actionKindFromString(pyStringOrDefault(action, "action_kind", "sizing"));
  proposal.action_index = pyInt64OrDefault(
      action, "action_index", pyInt64OrDefault(action, "action_id", -1));
  proposal.inst_id = pyIntOrDefault(action, "inst_id", pyIntOrDefault(action, "instance_id", -1));
  proposal.endpoint_component = pyIntOrDefault(
      action, "endpoint_component", pyIntOrDefault(action, "endpoint_component_id", -1));
  proposal.top_path_component = pyIntOrDefault(
      action, "top_path_component", pyIntOrDefault(action, "top_path_component_id", -1));
  proposal.affected_net_id = pyIntOrDefault(
      action, "affected_net_id", pyIntOrDefault(action, "net_id", -1));
  proposal.driver_pin_id = pyIntOrDefault(action, "driver_pin_id", -1);
  proposal.load_pin_id = pyIntOrDefault(action, "load_pin_id", -1);
  proposal.buffer_master_id = pyIntOrDefault(action, "buffer_master_id", -1);
  proposal.current_size_idx = pyIntOrDefault(action, "current_size_idx", -1);
  proposal.seed_size_idx = pyIntOrDefault(action, "seed_size_idx", -1);
  proposal.target_size_idx = pyIntOrDefault(action, "target_size_idx", proposal.seed_size_idx);
  proposal.seed_step = pyIntOrDefault(
      action,
      "seed_step",
      proposal.target_size_idx >= 0 && proposal.current_size_idx >= 0
          ? proposal.target_size_idx - proposal.current_size_idx
          : 0);
  proposal.seed_direction = pyStringOrDefault(action, "seed_direction", "");
  proposal.target_master = pyStringOrDefault(action, "target_master", "");
  proposal.seed_master = pyStringOrDefault(action, "seed_master", "");
  proposal.buffer_master_name = pyStringOrDefault(action, "buffer_master_name", "");
  proposal.legal_master_candidates = pyStringListOrEmpty(action, "legal_master_candidates");
  proposal.legal_cell_id_candidates = pyIntListOrEmpty(action, "legal_cell_id_candidates");
  proposal.legal_timing_coordinate_candidates = pyDoubleListOrEmpty(
      action, "legal_timing_coordinate_candidates");
  if (proposal.legal_timing_coordinate_candidates.empty()) {
    proposal.legal_timing_coordinate_candidates = pyDoubleListOrEmpty(action, "legal_size_candidates");
  }
  proposal.predicted_delta_obj = pyDoubleOrDefault(action, "predicted_delta_obj", 0.0);
  proposal.predicted_improvement = pyDoubleOrDefault(action, "predicted_improvement", 0.0);
  proposal.sensitivity_score = pyDoubleOrDefault(action, "sensitivity_score", 0.0);
  proposal.old_timing_coordinate = pyDoubleOrDefault(action, "old_timing_coordinate", 0.0);
  proposal.new_timing_coordinate = pyDoubleOrDefault(action, "new_timing_coordinate", 0.0);
  proposal.local_delta_delay_ps = pyDoubleOrDefault(
      action, "local_delta_delay_ps", pyDoubleOrDefault(action, "delta_delay_ps", 0.0));
  proposal.local_delta_slew_ps = pyDoubleOrDefault(
      action, "local_delta_slew_ps", pyDoubleOrDefault(action, "delta_slew_ps", 0.0));
  proposal.local_delta_cap = pyDoubleOrDefault(
      action, "local_delta_cap", pyDoubleOrDefault(action, "delta_output_cap", 0.0));
  proposal.local_delta_source = pyStringOrDefault(action, "local_delta_source", "");
  proposal.estimator_source = pyStringOrDefault(
      action,
      "estimator_source",
      pyStringOrDefault(action, "local_delta_estimator_source", proposal.local_delta_source));
  proposal.candidate_location_x = pyDoubleOrDefault(action, "candidate_location_x", 0.0);
  proposal.candidate_location_y = pyDoubleOrDefault(action, "candidate_location_y", 0.0);
  return proposal;
}

CompactBridgeActionResult parseActionResult(const py::dict& action_result)
{
  CompactBridgeActionResult result;
  result.kind = actionKindFromString(pyStringOrDefault(action_result, "action_kind", "sizing"));
  result.action_index = pyInt64OrDefault(
      action_result, "action_index", pyInt64OrDefault(action_result, "action_id", -1));
  result.inst_id = pyIntOrDefault(
      action_result, "inst_id", pyIntOrDefault(action_result, "instance_id", -1));
  result.accepted = pyBoolOrDefault(action_result, "accepted", false);
  result.supported = pyBoolOrDefault(action_result, "supported", false);
  result.status = pyStringOrDefault(action_result, "status", "");
  result.reject_reason = pyStringOrDefault(action_result, "reject_reason", "");
  result.old_master = pyStringOrDefault(action_result, "old_master", "");
  result.new_master = pyStringOrDefault(action_result, "new_master", "");
  result.target_master = pyStringOrDefault(action_result, "target_master", "");
  result.old_master_id = pyIntOrDefault(
      action_result,
      "old_master_id",
      pyIntOrDefault(action_result, "current_master_id", pyIntOrDefault(action_result, "old_cell_id", -1)));
  result.new_master_id = pyIntOrDefault(
      action_result,
      "new_master_id",
      pyIntOrDefault(
          action_result,
          "target_master_id",
          pyIntOrDefault(action_result, "new_cell_id", pyIntOrDefault(action_result, "target_cell_id", -1))));
  result.current_size_idx = pyIntOrDefault(action_result, "current_size_idx", -1);
  result.target_size_idx = pyIntOrDefault(
      action_result, "target_size_idx", pyIntOrDefault(action_result, "seed_size_idx", -1));
  result.best_trial_step = pyIntOrDefault(action_result, "best_trial_step", 0);
  result.old_timing_coordinate = pyDoubleOrDefault(action_result, "old_timing_coordinate", 0.0);
  result.new_timing_coordinate = pyDoubleOrDefault(action_result, "new_timing_coordinate", 0.0);
  result.actual_delta_tns = pyDoubleOrDefault(action_result, "actual_delta_tns", 0.0);
  result.actual_delta_wns = pyDoubleOrDefault(action_result, "actual_delta_wns", 0.0);
  result.actual_delta_obj = pyDoubleOrDefault(action_result, "actual_delta_obj", 0.0);
  result.local_predicted_net_slack_delta =
      pyDoubleOrDefault(action_result, "local_predicted_net_slack_delta", 0.0);
  result.local_predicted_weighted_net_tns_delta =
      pyDoubleOrDefault(action_result, "local_predicted_weighted_net_tns_delta", 0.0);
  result.local_predicted_slack_delta_source =
      pyStringOrDefault(action_result, "local_predicted_slack_delta_source", "");
  result.local_predicted_slack_delta_weight_mode =
      pyStringOrDefault(action_result, "local_predicted_slack_delta_weight_mode", "raw_slack_delta");
  return result;
}

std::vector<CompactBridgeActionProposal> parseActionProposals(
    const std::vector<py::dict>& actions)
{
  std::vector<CompactBridgeActionProposal> proposals;
  proposals.reserve(actions.size());
  for (const auto& action : actions) {
    proposals.push_back(parseActionProposal(action));
  }
  return proposals;
}

CompactBridgeActionProposalBuffers parseActionProposalBuffers(const py::dict& buffers)
{
  CompactBridgeActionProposalBuffers parsed;
  parsed.action_ids = pyInt64ArrayOrListOrEmpty(buffers, "action_ids");
  parsed.action_kind_ids = pyIntArrayOrListOrEmpty(buffers, "action_kind_ids");
  parsed.action_kinds = pyStringListOrEmpty(buffers, "action_kinds");
  parsed.string_table = pyStringListOrEmpty(buffers, "string_table");
  parsed.primary_inst_ids = pyIntArrayOrListOrEmpty(buffers, "primary_inst_ids");
  parsed.current_size_idxs = pyIntArrayOrListOrEmpty(buffers, "current_size_idxs");
  parsed.target_size_idxs = pyIntArrayOrListOrEmpty(buffers, "target_size_idxs");
  parsed.seed_size_idxs = pyIntArrayOrListOrEmpty(buffers, "seed_size_idxs");
  parsed.seed_steps = pyIntArrayOrListOrEmpty(buffers, "seed_steps");
  parsed.seed_directions = pyStringListOrEmpty(buffers, "seed_directions");
  parsed.seed_direction_ids = pyIntArrayOrListOrEmpty(buffers, "seed_direction_ids");
  parsed.target_masters = pyStringListOrEmpty(buffers, "target_masters");
  parsed.target_master_name_ids = pyIntArrayOrListOrEmpty(buffers, "target_master_name_ids");
  parsed.seed_masters = pyStringListOrEmpty(buffers, "seed_masters");
  parsed.seed_master_name_ids = pyIntArrayOrListOrEmpty(buffers, "seed_master_name_ids");
  parsed.legal_master_candidate_indptr = pyIntArrayOrListOrEmpty(buffers, "legal_master_candidate_indptr");
  parsed.legal_master_candidate_names = pyStringListOrEmpty(buffers, "legal_master_candidate_names");
  parsed.legal_master_candidate_name_ids = pyIntArrayOrListOrEmpty(buffers, "legal_master_candidate_name_ids");
  parsed.legal_cell_id_candidate_ids = pyIntArrayOrListOrEmpty(buffers, "legal_cell_id_candidate_ids");
  parsed.legal_timing_coordinate_candidates =
      pyDoubleArrayOrListOrEmpty(buffers, "legal_timing_coordinate_candidates");
  parsed.predicted_delta_objs = pyDoubleArrayOrListOrEmpty(buffers, "predicted_delta_objs");
  parsed.predicted_improvements = pyDoubleArrayOrListOrEmpty(buffers, "predicted_improvements");
  parsed.sensitivity_scores = pyDoubleArrayOrListOrEmpty(buffers, "sensitivity_scores");
  parsed.old_timing_coordinates = pyDoubleArrayOrListOrEmpty(buffers, "old_timing_coordinates");
  parsed.new_timing_coordinates = pyDoubleArrayOrListOrEmpty(buffers, "new_timing_coordinates");
  parsed.local_delta_delay_ps = pyDoubleArrayOrListOrEmpty(buffers, "local_delta_delay_ps");
  parsed.local_delta_slew_ps = pyDoubleArrayOrListOrEmpty(buffers, "local_delta_slew_ps");
  parsed.local_delta_caps = pyDoubleArrayOrListOrEmpty(buffers, "local_delta_caps");
  parsed.local_delta_sources = pyStringListOrEmpty(buffers, "local_delta_sources");
  parsed.estimator_sources = pyStringListOrEmpty(buffers, "estimator_sources");
  parsed.affected_net_ids = pyIntArrayOrListOrEmpty(buffers, "affected_net_ids");
  parsed.driver_pin_ids = pyIntArrayOrListOrEmpty(buffers, "driver_pin_ids");
  parsed.load_pin_ids = pyIntArrayOrListOrEmpty(buffers, "load_pin_ids");
  parsed.buffer_master_ids = pyIntArrayOrListOrEmpty(buffers, "buffer_master_ids");
  parsed.buffer_master_names = pyStringListOrEmpty(buffers, "buffer_master_names");
  parsed.buffer_master_name_ids = pyIntArrayOrListOrEmpty(buffers, "buffer_master_name_ids");
  parsed.candidate_location_xs = pyDoubleArrayOrListOrEmpty(buffers, "candidate_location_xs");
  parsed.candidate_location_ys = pyDoubleArrayOrListOrEmpty(buffers, "candidate_location_ys");
  return parsed;
}

std::vector<CompactBridgeActionProposal> buildActionProposalsFromBuffers(
    const CompactBridgeActionProposalBuffers& buffers)
{
  const std::size_t action_count = buffers.action_ids.size();
  std::vector<CompactBridgeActionProposal> proposals;
  proposals.reserve(action_count);
  for (std::size_t index = 0; index < action_count; ++index) {
    CompactBridgeActionProposal proposal;
    proposal.action_index = vectorValueOrDefault<int64_t>(buffers.action_ids, index, static_cast<int64_t>(index));
    const int kind_id = vectorValueOrDefault<int>(buffers.action_kind_ids, index, 0);
    const std::string kind_name = vectorValueOrDefault<std::string>(
        buffers.action_kinds, index, actionKindName(static_cast<ActionKind>(kind_id)));
    proposal.kind = !kind_name.empty()
                        ? actionKindFromString(kind_name)
                        : static_cast<ActionKind>(kind_id);
    if (proposal.kind < ActionKind::kSizing || proposal.kind > ActionKind::kUnknown) {
      proposal.kind = ActionKind::kUnknown;
    }
    proposal.inst_id = vectorValueOrDefault<int>(buffers.primary_inst_ids, index, -1);
    proposal.affected_net_id = vectorValueOrDefault<int>(buffers.affected_net_ids, index, -1);
    proposal.driver_pin_id = vectorValueOrDefault<int>(buffers.driver_pin_ids, index, -1);
    proposal.load_pin_id = vectorValueOrDefault<int>(buffers.load_pin_ids, index, -1);
    proposal.buffer_master_id = vectorValueOrDefault<int>(buffers.buffer_master_ids, index, -1);
    proposal.current_size_idx = vectorValueOrDefault<int>(buffers.current_size_idxs, index, -1);
    proposal.seed_size_idx = vectorValueOrDefault<int>(buffers.seed_size_idxs, index, -1);
    proposal.target_size_idx = vectorValueOrDefault<int>(
        buffers.target_size_idxs, index, proposal.seed_size_idx);
    proposal.seed_step = vectorValueOrDefault<int>(
        buffers.seed_steps,
        index,
        proposal.target_size_idx >= 0 && proposal.current_size_idx >= 0
            ? proposal.target_size_idx - proposal.current_size_idx
            : 0);
    proposal.seed_direction = compactStringTableValue(
        buffers,
        vectorValueOrDefault<int>(buffers.seed_direction_ids, index, -1),
        vectorValueOrDefault<std::string>(buffers.seed_directions, index, ""));
    proposal.target_master = compactStringTableValue(
        buffers,
        vectorValueOrDefault<int>(buffers.target_master_name_ids, index, -1),
        vectorValueOrDefault<std::string>(buffers.target_masters, index, ""));
    proposal.seed_master = compactStringTableValue(
        buffers,
        vectorValueOrDefault<int>(buffers.seed_master_name_ids, index, -1),
        vectorValueOrDefault<std::string>(buffers.seed_masters, index, ""));
    proposal.buffer_master_name = compactStringTableValue(
        buffers,
        vectorValueOrDefault<int>(buffers.buffer_master_name_ids, index, -1),
        vectorValueOrDefault<std::string>(buffers.buffer_master_names, index, ""));
    proposal.legal_master_candidates = compactLegalMasterCandidates(buffers, index);
    proposal.legal_cell_id_candidates = compactLegalCandidateValues<int>(
        buffers, buffers.legal_cell_id_candidate_ids, index, -1);
    proposal.legal_timing_coordinate_candidates = compactLegalCandidateValues<double>(
        buffers, buffers.legal_timing_coordinate_candidates, index, 0.0);
    proposal.predicted_delta_obj = vectorValueOrDefault<double>(buffers.predicted_delta_objs, index, 0.0);
    proposal.predicted_improvement = vectorValueOrDefault<double>(buffers.predicted_improvements, index, 0.0);
    proposal.sensitivity_score = vectorValueOrDefault<double>(buffers.sensitivity_scores, index, 0.0);
    proposal.old_timing_coordinate = vectorValueOrDefault<double>(buffers.old_timing_coordinates, index, 0.0);
    proposal.new_timing_coordinate = vectorValueOrDefault<double>(buffers.new_timing_coordinates, index, 0.0);
    proposal.local_delta_delay_ps = vectorValueOrDefault<double>(buffers.local_delta_delay_ps, index, 0.0);
    proposal.local_delta_slew_ps = vectorValueOrDefault<double>(buffers.local_delta_slew_ps, index, 0.0);
    proposal.local_delta_cap = vectorValueOrDefault<double>(buffers.local_delta_caps, index, 0.0);
    proposal.local_delta_source = vectorValueOrDefault<std::string>(buffers.local_delta_sources, index, "");
    proposal.estimator_source = vectorValueOrDefault<std::string>(
        buffers.estimator_sources, index, proposal.local_delta_source);
    proposal.candidate_location_x = vectorValueOrDefault<double>(buffers.candidate_location_xs, index, 0.0);
    proposal.candidate_location_y = vectorValueOrDefault<double>(buffers.candidate_location_ys, index, 0.0);
    proposals.push_back(std::move(proposal));
  }
  return proposals;
}

std::vector<CompactBridgeActionProposal> parseActionProposalsFromBuffers(
    const py::dict& buffers)
{
  return buildActionProposalsFromBuffers(parseActionProposalBuffers(buffers));
}

CompactBridgeConflictPrecompute parseConflictPrecompute(
    const py::dict& conflict_precompute,
    const py::dict& dynamic_conflict_signature)
{
  CompactBridgeConflictPrecompute typed_conflict_precompute;
  typed_conflict_precompute.component_id_by_endpoint =
      pyIntArrayOrListOrEmpty(conflict_precompute, "component_id_by_endpoint");
  typed_conflict_precompute.component_id_by_top_path =
      pyIntArrayOrListOrEmpty(conflict_precompute, "component_id_by_top_path");
  typed_conflict_precompute.component_id_by_net_neighborhood =
      pyIntArrayOrListOrEmpty(conflict_precompute, "component_id_by_net_neighborhood");
  typed_conflict_precompute.component_id_by_fanin_fanout_neighborhood =
      pyIntArrayOrListOrEmpty(conflict_precompute, "component_id_by_fanin_fanout_neighborhood");
  typed_conflict_precompute.component_id_by_endpoint_dynamic =
      pyIntArrayOrListOrEmpty(dynamic_conflict_signature, "component_id_by_endpoint_dynamic");
  typed_conflict_precompute.component_id_by_top_path_dynamic =
      pyIntArrayOrListOrEmpty(dynamic_conflict_signature, "component_id_by_top_path_dynamic");
  typed_conflict_precompute.local_residual_budget_by_top_path_component =
      pyDoubleArrayOrListOrEmpty(dynamic_conflict_signature, "local_residual_budget_by_top_path_component");
  typed_conflict_precompute.local_residual_budget_by_endpoint_component =
      pyDoubleArrayOrListOrEmpty(dynamic_conflict_signature, "local_residual_budget_by_endpoint_component");
  typed_conflict_precompute.local_residual_snapshot_source =
      pyStringOrDefault(dynamic_conflict_signature, "local_residual_snapshot_source", "");
  return typed_conflict_precompute;
}

std::vector<CompactBridgeActionResult> parseActionResults(
    const std::vector<py::dict>& action_results)
{
  std::vector<CompactBridgeActionResult> results;
  results.reserve(action_results.size());
  for (const auto& action_result : action_results) {
    results.push_back(parseActionResult(action_result));
  }
  return results;
}

std::vector<CompactBridgeAction> buildCompactActions(
    const std::vector<CompactBridgeActionProposal>& typed_proposal_pool,
    const std::vector<int>& action_indices,
    const CompactBridgeConflictPrecompute& typed_conflict_precompute)
{
  std::vector<CompactBridgeAction> compact_actions;
  compact_actions.reserve(action_indices.size());
  for (const int action_pool_index : action_indices) {
    if (action_pool_index < 0 || action_pool_index >= static_cast<int>(typed_proposal_pool.size())) {
      continue;
    }
    const auto& action = typed_proposal_pool[action_pool_index];
    CompactBridgeAction compact;
    compact.queue_index = action_pool_index;
    compact.action_index = action.action_index;
    compact.inst_id = action.inst_id;
    compact.endpoint_component = vectorIndexedIntOrDefault(
        typed_conflict_precompute.component_id_by_endpoint_dynamic, compact.inst_id);
    if (compact.endpoint_component < 0) {
      compact.endpoint_component = vectorIndexedIntOrDefault(
          typed_conflict_precompute.component_id_by_endpoint, compact.inst_id);
    }
    compact.top_path_component = vectorIndexedIntOrDefault(
        typed_conflict_precompute.component_id_by_top_path_dynamic, compact.inst_id);
    if (compact.top_path_component < 0) {
      compact.top_path_component = vectorIndexedIntOrDefault(
          typed_conflict_precompute.component_id_by_top_path, compact.inst_id);
    }
    compact.net_component = vectorIndexedIntOrDefault(
        typed_conflict_precompute.component_id_by_net_neighborhood, compact.inst_id);
    compact.fanin_fanout_component = vectorIndexedIntOrDefault(
        typed_conflict_precompute.component_id_by_fanin_fanout_neighborhood, compact.inst_id);
    compact.predicted_delta_obj = action.predicted_delta_obj;
    compact.predicted_improvement = action.predicted_improvement;
    compact_actions.push_back(compact);
  }
  return compact_actions;
}

std::vector<int> makeSequentialActionIndices(std::size_t action_count)
{
  std::vector<int> action_indices;
  action_indices.reserve(action_count);
  for (int index = 0; index < static_cast<int>(action_count); ++index) {
    action_indices.push_back(index);
  }
  return action_indices;
}

void assignActionProposalComponents(
    CompactBridgeActionProposal& proposal,
    const CompactBridgeConflictPrecompute& typed_conflict_precompute)
{
  proposal.endpoint_component = vectorIndexedIntOrDefault(
      typed_conflict_precompute.component_id_by_endpoint_dynamic, proposal.inst_id);
  if (proposal.endpoint_component < 0) {
    proposal.endpoint_component = vectorIndexedIntOrDefault(
        typed_conflict_precompute.component_id_by_endpoint, proposal.inst_id);
  }
  proposal.top_path_component = vectorIndexedIntOrDefault(
      typed_conflict_precompute.component_id_by_top_path_dynamic, proposal.inst_id);
  if (proposal.top_path_component < 0) {
    proposal.top_path_component = vectorIndexedIntOrDefault(
        typed_conflict_precompute.component_id_by_top_path, proposal.inst_id);
  }
}

std::vector<CompactBridgeActionProposal> collectActionProposals(
    const std::vector<CompactBridgeActionProposal>& typed_proposal_pool,
    const std::vector<int>& selected_action_indices,
    const CompactBridgeConflictPrecompute& typed_conflict_precompute)
{
  std::vector<CompactBridgeActionProposal> selected_proposals;
  selected_proposals.reserve(selected_action_indices.size());
  for (const int action_pool_index : selected_action_indices) {
    if (action_pool_index >= 0 && action_pool_index < static_cast<int>(typed_proposal_pool.size())) {
      CompactBridgeActionProposal proposal = typed_proposal_pool[action_pool_index];
      assignActionProposalComponents(proposal, typed_conflict_precompute);
      selected_proposals.push_back(proposal);
    }
  }
  return selected_proposals;
}

std::vector<CompactBridgeAction> buildCompactActions(
    const std::vector<py::dict>& seed_queue,
    const py::dict& conflict_precompute,
    const py::dict& dynamic_conflict_signature)
{
  const std::vector<CompactBridgeActionProposal> typed_proposal_pool =
      parseActionProposals(seed_queue);
  const CompactBridgeConflictPrecompute typed_conflict_precompute =
      parseConflictPrecompute(conflict_precompute, dynamic_conflict_signature);
  return buildCompactActions(
      typed_proposal_pool,
      makeSequentialActionIndices(typed_proposal_pool.size()),
      typed_conflict_precompute);
}

std::vector<std::string> splitConflictRules(const std::string& value)
{
  std::vector<std::string> rules;
  std::string current;
  for (char ch : value) {
    if (ch == ',') {
      if (!current.empty()) {
        rules.push_back(current);
        current.clear();
      }
    } else if (ch != ' ' && ch != '\t' && ch != '\n' && ch != '\r') {
      current.push_back(ch);
    }
  }
  if (!current.empty()) {
    rules.push_back(current);
  }
  return rules;
}

bool hasConflictRule(const std::vector<std::string>& rules,
                     const std::string& rule)
{
  return std::find(rules.begin(), rules.end(), rule) != rules.end();
}

DiffGuidedBatchConflictRuleMask conflictRuleMask(const py::dict& config)
{
  DiffGuidedBatchConflictRuleMask mask;
  if (!config.contains("diff_guided_batch_conflict_rules") ||
      config["diff_guided_batch_conflict_rules"].is_none()) {
    return mask;
  }
  const std::vector<std::string> rules = splitConflictRules(
      pyStringOrDefault(config, "diff_guided_batch_conflict_rules", ""));
  mask.same_instance = hasConflictRule(rules, "same_instance");
  mask.same_endpoint = hasConflictRule(rules, "same_endpoint");
  mask.same_top_path = hasConflictRule(rules, "same_top_path");
  mask.static_component =
      hasConflictRule(rules, "static_component") ||
      hasConflictRule(rules, "same_net") ||
      hasConflictRule(rules, "net_neighborhood") ||
      hasConflictRule(rules, "fanin_fanout") ||
      hasConflictRule(rules, "fanin_fanout_neighborhood");
  return mask;
}

CompactBridgeSelection selectCompact(
    const std::vector<CompactBridgeAction>& compact_actions,
    int max_batch_size,
    const DiffGuidedBatchConflictRuleMask& conflict_rules,
    const CompactBridgeConflictPrecompute& conflict_precompute,
    const std::string& residual_budget_filter_mode,
    const std::string& residual_budget_score_mode)
{
  CompactBridgeSelection selection;
  std::unordered_set<int> selected_instances;
  std::unordered_set<int> selected_endpoint_components;
  std::unordered_set<int> selected_top_path_components;
  std::unordered_set<int> selected_static_components;

  std::vector<int> ordered_indices;
  ordered_indices.reserve(compact_actions.size());
  for (int index = 0; index < static_cast<int>(compact_actions.size()); ++index) {
    ordered_indices.push_back(index);
  }
  if (residual_budget_score_mode == "clip_predicted_gain") {
    std::vector<double> effective_gains(compact_actions.size(), 0.0);
    for (int index = 0; index < static_cast<int>(compact_actions.size()); ++index) {
      const auto& action = compact_actions[index];
      double budget = 0.0;
      std::string budget_source;
      if (action.top_path_component >= 0 &&
          action.top_path_component < static_cast<int>(
              conflict_precompute.local_residual_budget_by_top_path_component.size())) {
        budget = std::max(
            budget,
            conflict_precompute.local_residual_budget_by_top_path_component[
                action.top_path_component]);
        if (budget > 0.0) {
          budget_source = "top_path_component_budget";
        }
      }
      if (action.endpoint_component >= 0 &&
          action.endpoint_component < static_cast<int>(
              conflict_precompute.local_residual_budget_by_endpoint_component.size())) {
        const double endpoint_budget =
            conflict_precompute.local_residual_budget_by_endpoint_component[
                action.endpoint_component];
        if (endpoint_budget > budget) {
          budget = endpoint_budget;
          budget_source = "endpoint_component_budget";
        }
      }
      const double predicted_gain =
          action.predicted_improvement > 0.0
              ? action.predicted_improvement
              : std::max(0.0, -action.predicted_delta_obj);
      effective_gains[index] = budget > 0.0 ? std::min(predicted_gain, budget) : 0.0;
      if (effective_gains[index] > 0.0 && selection.residual_budget_score_source.empty()) {
        selection.residual_budget_score_source = budget_source;
      }
    }
    std::stable_sort(
        ordered_indices.begin(),
        ordered_indices.end(),
        [&effective_gains](int left, int right) {
          return effective_gains[left] > effective_gains[right];
        });
    for (int order_index = 0; order_index < static_cast<int>(ordered_indices.size()); ++order_index) {
      if (ordered_indices[order_index] != order_index) {
        selection.residual_budget_score_reorder_count += 1;
      }
    }
  }

  for (const int action_index : ordered_indices) {
    const auto& action = compact_actions[action_index];
    bool conflict = false;
    bool residual_budget_filtered = false;
    if (residual_budget_filter_mode == "positive_violation") {
      const bool has_top_path_budget =
          action.top_path_component >= 0
          && action.top_path_component < static_cast<int>(
                 conflict_precompute.local_residual_budget_by_top_path_component.size())
          && conflict_precompute.local_residual_budget_by_top_path_component[action.top_path_component] > 0.0;
      const bool has_endpoint_budget =
          action.endpoint_component >= 0
          && action.endpoint_component < static_cast<int>(
                 conflict_precompute.local_residual_budget_by_endpoint_component.size())
          && conflict_precompute.local_residual_budget_by_endpoint_component[action.endpoint_component] > 0.0;
      residual_budget_filtered = !has_top_path_budget && !has_endpoint_budget;
      if (!residual_budget_filtered && selection.residual_budget_source.empty()) {
        selection.residual_budget_source =
            has_top_path_budget ? "top_path_component_budget" : "endpoint_component_budget";
      }
    }
    if (conflict_rules.same_instance &&
        action.inst_id >= 0 && selected_instances.find(action.inst_id) != selected_instances.end()) {
      conflict = true;
    }
    if (conflict_rules.same_endpoint && action.endpoint_component >= 0 &&
        selected_endpoint_components.find(action.endpoint_component) != selected_endpoint_components.end()) {
      conflict = true;
    }
    if (conflict_rules.same_top_path && action.top_path_component >= 0 &&
        selected_top_path_components.find(action.top_path_component) != selected_top_path_components.end()) {
      conflict = true;
    }
    if (conflict_rules.static_component && action.net_component >= 0 &&
        selected_static_components.find(action.net_component) != selected_static_components.end()) {
      conflict = true;
    }
    if (conflict_rules.static_component && action.fanin_fanout_component >= 0 &&
        selected_static_components.find(action.fanin_fanout_component) != selected_static_components.end()) {
      conflict = true;
    }

    if (static_cast<int>(selection.selected_queue_indices.size()) >= max_batch_size || conflict ||
        residual_budget_filtered) {
      selection.remaining_queue_indices.push_back(action.queue_index);
      selection.remaining_action_ids.push_back(action.action_index);
      if (conflict) {
        selection.conflict_reject_count += 1;
      }
      if (residual_budget_filtered) {
        selection.residual_budget_reject_count += 1;
      }
      continue;
    }

    selection.selected_queue_indices.push_back(action.queue_index);
    selection.selected_action_ids.push_back(action.action_index);
    if (action.inst_id >= 0) {
      selected_instances.insert(action.inst_id);
    }
    if (action.endpoint_component >= 0) {
      selected_endpoint_components.insert(action.endpoint_component);
    }
    if (action.top_path_component >= 0) {
      selected_top_path_components.insert(action.top_path_component);
    }
    if (action.net_component >= 0) {
      selected_static_components.insert(action.net_component);
    }
    if (action.fanin_fanout_component >= 0) {
      selected_static_components.insert(action.fanin_fanout_component);
    }
  }
  return selection;
}

}  // namespace diff_guided_batch
}  // namespace dreamplace
