#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

#include "action.h"
#include "conflict_precompute.h"
#include "loop_controller.h"
#include "opensta_trial.h"
#include "selector.h"
#include "transaction.h"

namespace py = pybind11;

namespace dreamplace {
namespace diff_guided_batch {
namespace {

std::vector<std::string> splitRuleString(const std::string& value) {
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

bool containsRule(const std::vector<std::string>& rules, const std::string& rule) {
  for (const auto& item : rules) {
    if (item == rule) {
      return true;
    }
  }
  return false;
}

ActionKind actionKindFromString(const std::string& value) {
  if (value == "sizing") {
    return ActionKind::kSizing;
  }
  if (value == "buffer_insert") {
    return ActionKind::kBufferInsert;
  }
  if (value == "buffer_remove") {
    return ActionKind::kBufferRemove;
  }
  if (value == "pin_swap") {
    return ActionKind::kPinSwap;
  }
  throw std::invalid_argument("unknown ActionKind: " + value);
}

std::vector<int32_t> intVectorFromDict(const py::dict& dict, const char* key) {
  std::vector<int32_t> result;
  if (!dict.contains(key) || dict[key].is_none()) {
    return result;
  }
  for (const auto& item : dict[key]) {
    result.push_back(py::cast<int32_t>(item));
  }
  return result;
}

template <typename T>
std::vector<T> pyArrayBufferToVector(
    const py::array_t<T, py::array::c_style | py::array::forcecast>& array) {
  const py::buffer_info buffer = array.request();
  if (buffer.ndim != 1) {
    throw py::cast_error("expected a one-dimensional compact selector buffer");
  }
  const auto* data = static_cast<const T*>(buffer.ptr);
  return std::vector<T>(data, data + buffer.shape[0]);
}

std::vector<int64_t> pyInt64ArrayOrList(const py::object& object) {
  if (object.is_none()) {
    return {};
  }
  try {
    const py::array_t<int64_t, py::array::c_style | py::array::forcecast> array(object);
    return pyArrayBufferToVector<int64_t>(array);
  } catch (const py::error_already_set&) {
    std::vector<int64_t> result;
    for (const auto& item : object) {
      result.push_back(py::cast<int64_t>(item));
    }
    return result;
  } catch (const py::cast_error&) {
    std::vector<int64_t> result;
    for (const auto& item : object) {
      result.push_back(py::cast<int64_t>(item));
    }
    return result;
  }
}

std::vector<int32_t> pyInt32ArrayOrList(const py::object& object) {
  if (object.is_none()) {
    return {};
  }
  try {
    const py::array_t<int32_t, py::array::c_style | py::array::forcecast> array(object);
    return pyArrayBufferToVector<int32_t>(array);
  } catch (const py::error_already_set&) {
    std::vector<int32_t> result;
    for (const auto& item : object) {
      result.push_back(py::cast<int32_t>(item));
    }
    return result;
  } catch (const py::cast_error&) {
    std::vector<int32_t> result;
    for (const auto& item : object) {
      result.push_back(py::cast<int32_t>(item));
    }
    return result;
  }
}

std::vector<double> pyDoubleArrayOrList(const py::object& object) {
  if (object.is_none()) {
    return {};
  }
  try {
    const py::array_t<double, py::array::c_style | py::array::forcecast> array(object);
    return pyArrayBufferToVector<double>(array);
  } catch (const py::error_already_set&) {
    std::vector<double> result;
    for (const auto& item : object) {
      result.push_back(py::cast<double>(item));
    }
    return result;
  } catch (const py::cast_error&) {
    std::vector<double> result;
    for (const auto& item : object) {
      result.push_back(py::cast<double>(item));
    }
    return result;
  }
}

std::vector<int32_t> sliceCompactIds(const std::vector<int32_t>& indptr,
                                     const std::vector<int32_t>& ids,
                                     size_t index) {
  if (indptr.size() < index + 2) {
    return {};
  }
  const int32_t begin = indptr[index];
  const int32_t end = indptr[index + 1];
  if (begin < 0 || end < begin ||
      static_cast<size_t>(end) > ids.size()) {
    return {};
  }
  return std::vector<int32_t>(ids.begin() + begin, ids.begin() + end);
}

std::string stringFromDict(const py::dict& dict, const char* key, std::string fallback = "") {
  if (!dict.contains(key) || dict[key].is_none()) {
    return fallback;
  }
  return py::cast<std::string>(dict[key]);
}

double doubleFromDict(const py::dict& dict, const char* key, double fallback = 0.0) {
  if (!dict.contains(key) || dict[key].is_none()) {
    return fallback;
  }
  return py::cast<double>(dict[key]);
}

int32_t intFromDict(const py::dict& dict, const char* key, int32_t fallback = -1) {
  if (!dict.contains(key) || dict[key].is_none()) {
    return fallback;
  }
  return py::cast<int32_t>(dict[key]);
}

int64_t longFromDict(const py::dict& dict, const char* key, int64_t fallback = -1) {
  if (!dict.contains(key) || dict[key].is_none()) {
    return fallback;
  }
  return py::cast<int64_t>(dict[key]);
}

ActionProposal proposalFromDict(const py::dict& dict) {
  ActionProposal proposal;
  proposal.action_id = longFromDict(dict, "action_index", -1);
  if (proposal.action_id < 0) {
    proposal.action_id = longFromDict(dict, "action_id", -1);
  }
  if (dict.contains("action_kind") && !dict["action_kind"].is_none()) {
    proposal.action_kind = actionKindFromString(py::cast<std::string>(dict["action_kind"]));
  }
  proposal.predicted_delta_obj = doubleFromDict(dict, "predicted_delta_obj", 0.0);
  proposal.sensitivity_score = doubleFromDict(dict, "sensitivity_score", 0.0);
  proposal.sizing.inst_id = intFromDict(dict, "inst_id", -1);
  proposal.sizing.current_master_id = intFromDict(dict, "current_master_id", -1);
  proposal.sizing.target_master_id = intFromDict(dict, "target_master_id", -1);
  proposal.sizing.current_size_idx = intFromDict(dict, "current_size_idx", -1);
  proposal.sizing.target_size_idx = intFromDict(dict, "seed_size_idx", -1);
  proposal.sizing.size_idx_delta = intFromDict(dict, "seed_step", 0);
  return proposal;
}

ActionConflictFootprint footprintFromDict(const py::dict& dict) {
  ActionConflictFootprint footprint;
  footprint.action_id = longFromDict(dict, "action_index", -1);
  if (footprint.action_id < 0) {
    footprint.action_id = longFromDict(dict, "action_id", -1);
  }
  if (dict.contains("action_kind") && !dict["action_kind"].is_none()) {
    footprint.action_kind = actionKindFromString(py::cast<std::string>(dict["action_kind"]));
  }
  footprint.primary_inst_id = intFromDict(dict, "primary_inst_id", -1);
  footprint.affected_instance_ids = intVectorFromDict(dict, "affected_instance_ids");
  footprint.affected_net_ids = intVectorFromDict(dict, "affected_net_ids");
  footprint.affected_pin_ids = intVectorFromDict(dict, "affected_pin_ids");
  footprint.endpoint_component_ids = intVectorFromDict(dict, "endpoint_component_ids");
  footprint.top_path_component_ids = intVectorFromDict(dict, "top_path_component_ids");
  footprint.static_component_ids = intVectorFromDict(dict, "static_component_ids");
  footprint.physical_bin_ids = intVectorFromDict(dict, "physical_bin_ids");
  if (footprint.primary_inst_id < 0 && !footprint.affected_instance_ids.empty()) {
    footprint.primary_inst_id = footprint.affected_instance_ids.front();
  }
  return footprint;
}

py::dict selectorSummaryToDict(const SelectorSummary& summary) {
  py::dict result;
  result["proposal_seed_count"] = summary.proposal_seed_count;
  result["selected_batch_size"] = summary.selected_batch_size;
  result["remaining_seed_count"] = summary.remaining_seed_count;
  result["conflict_reject_count"] = summary.conflict_reject_count;
  result["component_conflict_count"] = summary.component_conflict_count;
  result["parallel_worker_count"] = summary.parallel_worker_count;
  result["predicted_batch_delta_obj"] = summary.predicted_batch_delta_obj;
  result["predicted_batch_delta_tns"] = summary.predicted_batch_delta_tns;
  result["cpp_selector_ms"] = summary.cpp_selector_ms;
  result["parallel_strategy"] = summary.parallel_strategy;
  result["component_count"] = summary.component_count;
  result["max_component_size"] = summary.max_component_size;
  result["csr_edge_count"] = summary.csr_edge_count;
  result["bitset_block_count"] = summary.bitset_block_count;
  result["precompute_build_ms"] = summary.precompute_build_ms;
  result["residual_budget_reject_count"] = summary.residual_budget_reject_count;
  result["selector_residual_budget_filter_mode"] =
      summary.selector_residual_budget_filter_mode;
  result["selector_residual_budget_score_mode"] =
      summary.selector_residual_budget_score_mode;
  result["residual_budget_score_reorder_count"] =
      summary.residual_budget_score_reorder_count;
  result["residual_budget_source"] = summary.residual_budget_source;
  result["residual_budget_score_source"] = summary.residual_budget_score_source;
  result["residual_snapshot_source"] = summary.residual_snapshot_source;
  result["conflict_precompute_source"] = summary.conflict_precompute_source;
  result["conflict_precompute_version"] = summary.conflict_precompute_version;
  result["conflict_precompute_input_hash"] = summary.conflict_precompute_input_hash;
  result["enabled_same_instance_conflict"] = summary.enabled_same_instance_conflict;
  result["enabled_endpoint_conflict"] = summary.enabled_endpoint_conflict;
  result["enabled_top_path_conflict"] = summary.enabled_top_path_conflict;
  result["enabled_static_component_conflict"] = summary.enabled_static_component_conflict;
  result["selected_action_ids"] = summary.selected_action_ids;
  result["remaining_action_ids"] = summary.remaining_action_ids;
  return result;
}

ConflictPrecompute conflictPrecomputeFromDict(const py::dict& conflict_precompute_dict) {
  ConflictPrecompute precompute;
  if (conflict_precompute_dict.contains("component_count")) {
    precompute.component_count =
        py::cast<int32_t>(conflict_precompute_dict["component_count"]);
  }
  precompute.max_component_size =
      intFromDict(conflict_precompute_dict, "max_component_size", 0);
  if (conflict_precompute_dict.contains("csr_edge_count")) {
    precompute.csr_edge_count =
        py::cast<int64_t>(conflict_precompute_dict["csr_edge_count"]);
  }
  precompute.bitset_block_count =
      intFromDict(conflict_precompute_dict, "bitset_block_count", 0);
  precompute.build_ms =
      doubleFromDict(conflict_precompute_dict, "precompute_build_ms", 0.0);
  if (precompute.build_ms == 0.0) {
    precompute.build_ms =
        doubleFromDict(conflict_precompute_dict, "build_ms", 0.0);
  }
  precompute.source = stringFromDict(
      conflict_precompute_dict,
      "conflict_precompute_source",
      stringFromDict(conflict_precompute_dict, "source", ""));
  precompute.version = intFromDict(
      conflict_precompute_dict,
      "conflict_precompute_version",
      intFromDict(conflict_precompute_dict, "version", 1));
  precompute.input_fingerprint = stringFromDict(
      conflict_precompute_dict,
      "conflict_precompute_input_hash",
      stringFromDict(conflict_precompute_dict, "input_fingerprint", ""));
  if (conflict_precompute_dict.contains("local_residual_budget_by_top_path_component")) {
    precompute.local_residual_budget_by_top_path_component = pyDoubleArrayOrList(
        conflict_precompute_dict["local_residual_budget_by_top_path_component"]);
  }
  if (conflict_precompute_dict.contains("local_residual_budget_by_endpoint_component")) {
    precompute.local_residual_budget_by_endpoint_component = pyDoubleArrayOrList(
        conflict_precompute_dict["local_residual_budget_by_endpoint_component"]);
  }
  precompute.local_residual_snapshot_source = stringFromDict(
      conflict_precompute_dict, "local_residual_snapshot_source", "");
  return precompute;
}

SelectorConfig selectorConfigFromDict(const py::dict& config_dict) {
  SelectorConfig config;
  if (config_dict.contains("max_batch_size")) {
    config.max_batch_size = py::cast<int32_t>(config_dict["max_batch_size"]);
  }
  if (config_dict.contains("parallel_worker_count")) {
    config.parallel_worker_count =
        py::cast<int32_t>(config_dict["parallel_worker_count"]);
  }
  if (config_dict.contains("parallel_strategy")) {
    config.parallel_strategy = py::cast<std::string>(config_dict["parallel_strategy"]);
  }
  config.residual_budget_filter_mode = stringFromDict(
      config_dict,
      "selector_residual_budget_filter_mode",
      stringFromDict(
          config_dict,
          "diff_guided_batch_selector_residual_budget_filter_mode",
          "off"));
  config.residual_budget_score_mode = stringFromDict(
      config_dict,
      "selector_residual_budget_score_mode",
      stringFromDict(
          config_dict,
          "diff_guided_batch_selector_residual_budget_score_mode",
          "off"));
  if (config_dict.contains("diff_guided_batch_conflict_rules") &&
      !config_dict["diff_guided_batch_conflict_rules"].is_none()) {
    const std::vector<std::string> rules =
        splitRuleString(py::cast<std::string>(config_dict["diff_guided_batch_conflict_rules"]));
    config.use_same_instance_conflict = containsRule(rules, "same_instance");
    config.use_endpoint_conflict = containsRule(rules, "same_endpoint");
    config.use_top_path_conflict = containsRule(rules, "same_top_path");
    config.use_static_component_conflict =
        containsRule(rules, "static_component") ||
        containsRule(rules, "same_net") ||
        containsRule(rules, "net_neighborhood") ||
        containsRule(rules, "fanin_fanout") ||
        containsRule(rules, "fanin_fanout_neighborhood");
  }
  return config;
}

py::dict actionResultToDict(const ActionResult& action_result) {
  py::dict result;
  result["action_id"] = action_result.action_id;
  result["action_kind"] = actionKindName(action_result.action_kind);
  result["supported"] = action_result.supported;
  result["accepted"] = action_result.accepted;
  result["status"] = action_result.status;
  result["reject_reason"] = action_result.reject_reason;
  result["actual_delta_tns"] = action_result.actual_delta_tns;
  result["actual_delta_wns"] = action_result.actual_delta_wns;
  return result;
}

py::dict transactionDecisionToDict(const TransactionDecision& decision) {
  py::dict result;
  result["confirmed"] = decision.confirmed;
  result["rolled_back"] = decision.rolled_back;
  result["accepted_action_count"] = decision.accepted_action_count;
  result["unsupported_action_count"] = decision.unsupported_action_count;
  result["status"] = decision.status;
  result["reject_reason"] = decision.reject_reason;
  return result;
}

py::dict loopSummaryToDict(const LoopControllerSummary& summary) {
  py::dict result;
  result["status"] = summary.status;
  result["opensta_status"] = summary.opensta_status;
  result["cxx_batch_loop_status"] = summary.cxx_batch_loop_status;
  result["proposal_seed_count"] = summary.proposal_seed_count;
  result["loop_count"] = summary.loop_count;
  result["selected_batch_count"] = summary.selected_batch_count;
  result["accepted_action_count"] = summary.accepted_action_count;
  result["rejected_action_count"] = summary.rejected_action_count;
  result["confirmed_batch_count"] = summary.confirmed_batch_count;
  result["rolled_back_batch_count"] = summary.rolled_back_batch_count;
  result["total_actual_delta_tns"] = summary.total_actual_delta_tns;
  py::list selector_summaries;
  for (const auto& selector_summary : summary.selector_summaries) {
    selector_summaries.append(selectorSummaryToDict(selector_summary));
  }
  result["selector_summaries"] = selector_summaries;
  py::list transaction_decisions;
  for (const auto& decision : summary.transaction_decisions) {
    transaction_decisions.append(transactionDecisionToDict(decision));
  }
  result["transaction_decisions"] = transaction_decisions;
  return result;
}

py::dict conflictPrecomputeToDict(const ConflictPrecompute& precompute) {
  py::dict result;
  result["component_count"] = precompute.component_count;
  result["max_component_size"] = precompute.max_component_size;
  result["csr_edge_count"] = precompute.csr_edge_count;
  result["bitset_block_count"] = precompute.bitset_block_count;
  result["build_ms"] = precompute.build_ms;
  result["input_fingerprint"] = precompute.input_fingerprint;
  result["source"] = precompute.source;
  result["version"] = precompute.version;
  result["component_ids"] = precompute.component_ids;
  result["csr_indptr"] = precompute.csr_indptr;
  result["csr_indices"] = precompute.csr_indices;
  result["local_residual_budget_by_top_path_component"] =
      precompute.local_residual_budget_by_top_path_component;
  result["local_residual_budget_by_endpoint_component"] =
      precompute.local_residual_budget_by_endpoint_component;
  result["local_residual_snapshot_source"] = precompute.local_residual_snapshot_source;
  return result;
}

}  // namespace

}  // namespace diff_guided_batch
}  // namespace dreamplace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  using namespace dreamplace::diff_guided_batch;

  m.def("build_conflict_precompute_from_component_ids",
        [](const std::vector<int32_t>& component_ids,
           const std::string& input_fingerprint) {
          return conflictPrecomputeToDict(
              buildConflictPrecomputeFromComponentIds(component_ids, input_fingerprint));
        },
        py::arg("component_ids"),
        py::arg("input_fingerprint") = "");

  m.def("select_conflict_free_batch",
        [](const std::vector<py::dict>& proposal_dicts,
           const std::vector<py::dict>& footprint_dicts,
           const py::dict& conflict_precompute_dict,
           const py::dict& config_dict) {
          std::vector<ActionProposal> proposals;
          proposals.reserve(proposal_dicts.size());
          for (const auto& proposal_dict : proposal_dicts) {
            proposals.push_back(proposalFromDict(proposal_dict));
          }

          std::vector<ActionConflictFootprint> footprints;
          footprints.reserve(footprint_dicts.size());
          for (const auto& footprint_dict : footprint_dicts) {
            footprints.push_back(footprintFromDict(footprint_dict));
          }

          ConflictPrecompute precompute = conflictPrecomputeFromDict(conflict_precompute_dict);
          SelectorConfig config = selectorConfigFromDict(config_dict);

          return selectorSummaryToDict(
              selectConflictFreeBatch(proposals, footprints, precompute, config));
        },
        py::arg("proposals"),
        py::arg("footprints"),
        py::arg("conflict_precompute"),
        py::arg("config"));

  m.def("select_conflict_free_batch_compact",
        [](const py::object& action_ids_obj,
           const py::object& predicted_delta_objs_obj,
           const py::object& primary_inst_ids_obj,
           const py::object& endpoint_component_indptr_obj,
           const py::object& endpoint_component_ids_obj,
           const py::object& top_path_component_indptr_obj,
           const py::object& top_path_component_ids_obj,
           const py::object& static_component_indptr_obj,
           const py::object& static_component_ids_obj,
           const py::dict& conflict_precompute_dict,
           const py::dict& config_dict) {
          const std::vector<int64_t> action_ids =
              pyInt64ArrayOrList(action_ids_obj);
          const std::vector<double> predicted_delta_objs =
              pyDoubleArrayOrList(predicted_delta_objs_obj);
          const std::vector<int32_t> primary_inst_ids =
              pyInt32ArrayOrList(primary_inst_ids_obj);
          const std::vector<int32_t> endpoint_component_indptr =
              pyInt32ArrayOrList(endpoint_component_indptr_obj);
          const std::vector<int32_t> endpoint_component_ids =
              pyInt32ArrayOrList(endpoint_component_ids_obj);
          const std::vector<int32_t> top_path_component_indptr =
              pyInt32ArrayOrList(top_path_component_indptr_obj);
          const std::vector<int32_t> top_path_component_ids =
              pyInt32ArrayOrList(top_path_component_ids_obj);
          const std::vector<int32_t> static_component_indptr =
              pyInt32ArrayOrList(static_component_indptr_obj);
          const std::vector<int32_t> static_component_ids =
              pyInt32ArrayOrList(static_component_ids_obj);
          const size_t count = action_ids.size();
          if (predicted_delta_objs.size() != count ||
              primary_inst_ids.size() != count ||
              endpoint_component_indptr.size() != count + 1) {
            throw std::invalid_argument(
                "compact selector buffers require action_ids, predicted_delta_objs, "
                "primary_inst_ids, and endpoint_component_indptr with matching sizes");
          }
          if (!top_path_component_indptr.empty() &&
              top_path_component_indptr.size() != count + 1) {
            throw std::invalid_argument(
                "compact selector top_path_component_indptr must be empty or action_count + 1");
          }
          if (!static_component_indptr.empty() &&
              static_component_indptr.size() != count + 1) {
            throw std::invalid_argument(
                "compact selector static_component_indptr must be empty or action_count + 1");
          }

          std::vector<ActionProposal> proposals;
          std::vector<ActionConflictFootprint> footprints;
          proposals.reserve(count);
          footprints.reserve(count);
          for (size_t idx = 0; idx < count; ++idx) {
            ActionProposal proposal;
            proposal.action_id = action_ids[idx];
            proposal.action_kind = ActionKind::kSizing;
            proposal.predicted_delta_obj = predicted_delta_objs[idx];
            proposal.sensitivity_score = -predicted_delta_objs[idx];
            proposal.sizing.inst_id = primary_inst_ids[idx];
            proposals.push_back(proposal);

            ActionConflictFootprint footprint;
            footprint.action_id = action_ids[idx];
            footprint.action_kind = ActionKind::kSizing;
            footprint.primary_inst_id = primary_inst_ids[idx];
            footprint.affected_instance_ids = {primary_inst_ids[idx]};
            footprint.endpoint_component_ids = sliceCompactIds(
                endpoint_component_indptr, endpoint_component_ids, idx);
            footprint.top_path_component_ids = sliceCompactIds(
                top_path_component_indptr, top_path_component_ids, idx);
            footprint.static_component_ids = sliceCompactIds(
                static_component_indptr, static_component_ids, idx);
            footprints.push_back(std::move(footprint));
          }

          ConflictPrecompute precompute = conflictPrecomputeFromDict(conflict_precompute_dict);
          SelectorConfig config = selectorConfigFromDict(config_dict);
          py::dict result = selectorSummaryToDict(
              selectConflictFreeBatch(proposals, footprints, precompute, config));
          result["selector_input_format"] = "compact_buffers";
          return result;
        },
        py::arg("action_ids"),
        py::arg("predicted_delta_objs"),
        py::arg("primary_inst_ids"),
        py::arg("endpoint_component_indptr"),
        py::arg("endpoint_component_ids"),
        py::arg("top_path_component_indptr"),
        py::arg("top_path_component_ids"),
        py::arg("static_component_indptr"),
        py::arg("static_component_ids"),
        py::arg("conflict_precompute"),
        py::arg("config"));

  m.def("evaluate_actions_dry_run",
        [](const std::vector<py::dict>& proposal_dicts, const py::dict& config_dict) {
          std::vector<ActionProposal> proposals;
          proposals.reserve(proposal_dicts.size());
          for (const auto& proposal_dict : proposal_dicts) {
            proposals.push_back(proposalFromDict(proposal_dict));
          }
          TrialConfig config;
          if (config_dict.contains("max_up_step")) {
            config.max_up_step = py::cast<int32_t>(config_dict["max_up_step"]);
          }
          if (config_dict.contains("max_down_step")) {
            config.max_down_step = py::cast<int32_t>(config_dict["max_down_step"]);
          }
          if (config_dict.contains("parallel_worker_count")) {
            config.parallel_worker_count =
                py::cast<int32_t>(config_dict["parallel_worker_count"]);
          }
          py::list results;
          for (const auto& result : evaluateActionsDryRun(proposals, config)) {
            results.append(actionResultToDict(result));
          }
          return results;
        },
        py::arg("proposals"),
        py::arg("config"));

  m.def("verify_and_commit_dry_run",
        [](const std::vector<py::dict>& action_result_dicts, double min_total_tns_gain) {
          std::vector<ActionResult> action_results;
          action_results.reserve(action_result_dicts.size());
          for (const auto& action_result_dict : action_result_dicts) {
            ActionResult result;
            result.action_id = longFromDict(action_result_dict, "action_id", -1);
            if (action_result_dict.contains("action_kind") &&
                !action_result_dict["action_kind"].is_none()) {
              result.action_kind =
                  actionKindFromString(py::cast<std::string>(action_result_dict["action_kind"]));
            }
            result.supported =
                action_result_dict.contains("supported")
                    ? py::cast<bool>(action_result_dict["supported"])
                    : false;
            result.accepted =
                action_result_dict.contains("accepted")
                    ? py::cast<bool>(action_result_dict["accepted"])
                    : false;
            result.actual_delta_tns =
                doubleFromDict(action_result_dict, "actual_delta_tns", 0.0);
            result.actual_delta_wns =
                doubleFromDict(action_result_dict, "actual_delta_wns", 0.0);
            action_results.push_back(result);
          }
          return transactionDecisionToDict(
              verifyAndCommitDryRun(action_results, min_total_tns_gain));
        },
        py::arg("action_results"),
        py::arg("min_total_tns_gain") = 0.0);

  m.def("run_diff_guided_batch_loop_dry_run",
        [](const std::vector<py::dict>& proposal_dicts,
           const std::vector<py::dict>& footprint_dicts,
           const py::dict& conflict_precompute_dict,
           const py::dict& config_dict) {
          std::vector<ActionProposal> proposals;
          proposals.reserve(proposal_dicts.size());
          for (const auto& proposal_dict : proposal_dicts) {
            proposals.push_back(proposalFromDict(proposal_dict));
          }

          std::vector<ActionConflictFootprint> footprints;
          footprints.reserve(footprint_dicts.size());
          for (const auto& footprint_dict : footprint_dicts) {
            footprints.push_back(footprintFromDict(footprint_dict));
          }

          ConflictPrecompute precompute;
          if (conflict_precompute_dict.contains("component_count")) {
            precompute.component_count =
                py::cast<int32_t>(conflict_precompute_dict["component_count"]);
          }

          LoopControllerConfig config;
          if (config_dict.contains("max_batch_size")) {
            config.max_batch_size = py::cast<int32_t>(config_dict["max_batch_size"]);
          }
          if (config_dict.contains("max_loop_batches")) {
            config.max_loop_batches =
                py::cast<int32_t>(config_dict["max_loop_batches"]);
          }
          if (config_dict.contains("min_recent_tns_gain")) {
            config.min_recent_tns_gain =
                py::cast<double>(config_dict["min_recent_tns_gain"]);
          }
          if (config_dict.contains("parallel_worker_count")) {
            config.parallel_worker_count =
                py::cast<int32_t>(config_dict["parallel_worker_count"]);
          }

          return loopSummaryToDict(
              runDiffGuidedBatchLoopDryRun(proposals, footprints, precompute, config));
        },
        py::arg("proposals"),
        py::arg("footprints"),
        py::arg("conflict_precompute"),
        py::arg("config"));
}
