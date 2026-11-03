// SPDX-License-Identifier: BSD-3-Clause

#include "rebuffer_native_planner.hh"
#include "rebuffer_timing_planning.hh"

#include <algorithm>
#include <sstream>
#include <string>
#include <memory>
#include <utility>

#include "pybind11/stl.h"
#include "rebuffer_driver_tree.hh"
#include "rebuffer_native_queue.hh"
#include "rebuffer_python_views.hh"
#include "rebuffer_tree_search.hh"

namespace py = pybind11;

namespace rebuffer {

namespace {

bool callable(const py::object& callback)
{
  return callback && !callback.is_none() && PyCallable_Check(callback.ptr());
}

py::dict callDictNoArg(const py::object& callback, const char* name)
{
  if (!callable(callback)) {
    throw py::type_error(std::string("native planner requires ") + name
                         + " callback");
  }
  py::object value = callback();
  if (!py::isinstance<py::dict>(value)) {
    throw py::type_error(std::string("native planner ") + name
                         + " callback must return a dict");
  }
  return value.cast<py::dict>();
}

py::dict copyDict(const py::dict& values)
{
  return values.attr("copy")().cast<py::dict>();
}

void mergeConfig(py::dict& target, const py::dict& source)
{
  for (const auto& item : source) {
    target[item.first] = item.second;
  }
}

py::dict unsupported(const char* reason)
{
  py::dict result;
  result["status"] = "unsupported";
  result["reason"] = reason;
  result["actions"] = py::list();
  return result;
}

bool timingBufferMoveEligible(const DesignView& design, int net_id,
                              const RepairConfig& config)
{
  if (!config.enforce_timing_buffer_move_fanout) return true;
  if (net_id < 0 || net_id + 1 >= static_cast<int>(design.flat_net2pin_start_map.size())
      || net_id >= static_cast<int>(design.net2driver_pin_map.size())) return false;
  const int driver = design.net2driver_pin_map[net_id];
  const int begin = design.flat_net2pin_start_map[net_id];
  const int end = design.flat_net2pin_start_map[net_id + 1];
  if (begin < 0 || end < begin || end > static_cast<int>(design.flat_net2pin_map.size())) return false;
  int fanout = 0;
  for (int index = begin; index < end; ++index) {
    if (design.flat_net2pin_map[index] != driver) ++fanout;
  }
  return fanout > 0
         && (config.timing_max_fanout <= 0 || fanout < config.timing_max_fanout);
}

std::string actionKey(const py::dict& action)
{
  std::ostringstream key;
  const auto value = [&](const char* field, const char* fallback = "") {
    if (action.contains(field) && !action[field].is_none()) {
      return py::str(action[field]).cast<std::string>();
    }
    return std::string(fallback);
  };
  key << value("net_id") << ':' << value("driver_pin") << ':'
      << value("buffer_cell_index") << ':' << value("parent_node_id") << ':'
      << value("child_node_id") << ':' << value("x_dbu") << ':'
      << value("y_dbu") << ':' << value("segment_tree_depth");
  return key.str();
}

}  // namespace

py::dict planNativeCurrentTree(const py::dict& request,
    const NativeTreePlannerViews& input, const py::dict& outer_config)
{
  py::list actions;
  py::list candidate_frontier;
  py::dict tree_search;
  std::string net_name;
  int plan_count = 0;
  bool replacement_tree = false;
  int focus_net_id = -1;
  std::string source_tree_digest;

  {
    try {
      const auto& tree = input.tree;
      if (tree == nullptr) return unsupported("native_tree_payload_missing_tree");
      source_tree_digest = tree->tree_digest;
      py::dict planner_config = copyDict(outer_config);
      mergeConfig(planner_config, input.config);
      const auto& design = input.design;
      const auto& timing = input.timing;
      const auto& liberty = input.liberty;
      const auto& characterization = input.characterization;
      const auto config = rebuffer_python::configFromDict(planner_config);
      if (!timingBufferMoveEligible(design, tree->root_net_id, config)) {
        return unsupported("timing_buffer_move_fanout");
      }
      TreeVgdpResult result;
      std::string reason;
      if (!optimizeCurrentDriverTree(design,
                                 timing,
                                 liberty,
                                 characterization,
                                 config,
                                 *tree,
                                 result,
                                 reason)) {
        return unsupported(reason.c_str());
      }
      const int requested_variant = input.candidate_variant;
      focus_net_id = input.focus_net_id;
      std::vector<const TreeVgdpCandidate*> focused;
      std::vector<const TreeVgdpCandidate*> unfocused;
      for (const auto& candidate : result.candidates) {
        if (candidate.actions.empty()) continue;
        const bool matches = focus_net_id < 0 || focus_net_id == tree->root_net_id
            || std::any_of(candidate.actions.begin(), candidate.actions.end(),
                           [&](const BufferAction& action) { return action.net_id == focus_net_id; });
        (matches ? focused : unfocused).push_back(&candidate);
      }
      const auto& frontier = focused.empty() ? unfocused : focused;
      const int top_k = planner_config.contains("python_timing_candidate_top_k")
          ? std::max(1, py::cast<int>(planner_config["python_timing_candidate_top_k"])) : 1;
      for (int index = std::max(0, requested_variant);
           index < std::min(static_cast<int>(frontier.size()), requested_variant + top_k);
           ++index) {
        py::list bundle;
        // A branch focus selects complete trees containing that partition.
        // Dropping other branches would invalidate the DP feasibility/score.
        for (const auto& action : frontier[index]->actions) {
          bundle.append(rebuffer_python::bufferActionDict(action));
        }
        candidate_frontier.append(bundle);
      }
      if (candidate_frontier.size() != 0) {
        actions = candidate_frontier[0].cast<py::list>();
        replacement_tree = true;
        ++plan_count;
      }
      tree_search["generated_option_count"] = result.generated_option_count;
      tree_search["surviving_option_count"] = result.surviving_option_count;
      tree_search["pruned_option_count"] = result.pruned_option_count;
      tree_search["long_wire_edge_count"] = result.long_wire_edge_count;
      tree_search["wire_site_count"] = result.wire_site_count;
      tree_search["electrical_rejected_count"] = result.electrical_rejected_count;
      tree_search["timing_round_count"] = result.timing_round_count;
      tree_search["recovery_attempt_count"] = result.recovery_attempt_count;
      tree_search["recovered_buffer_count"] = result.recovered_buffer_count;
      net_name = input.net_name;
    } catch (const std::exception& error) {
      return unsupported(error.what());
    }
  }

  py::dict candidate;
  candidate["status"] = py::len(actions) == 0 ? "no_actions" : "ok";
  candidate["actions"] = std::move(actions);
  if (candidate_frontier.size() != 0) {
    candidate["candidate_frontier"] = candidate_frontier;
    candidate["tree_search"] = tree_search;
  }
  candidate["net_id"] = request.contains("net_id") ? request["net_id"]
                                                    : py::int_(input.tree->root_net_id);
  if (!net_name.empty()) {
    candidate["net_name"] = net_name;
  }
  if (request.contains("endpoint_pin")) {
    candidate["endpoint_pin"] = request["endpoint_pin"];
  }
  if (request.contains("root_net_id")) {
    candidate["root_net_id"] = request["root_net_id"];
  }
  if (request.contains("driver_pin_id")) {
    candidate["driver_pin_id"] = request["driver_pin_id"];
  }
  if (request.contains("previous_endpoint_slack_ps")) {
    candidate["previous_endpoint_slack_ps"]
        = request["previous_endpoint_slack_ps"];
  }
  candidate["partition_count"] = plan_count;
  candidate["candidate_variant"] = request.contains("candidate_variant")
                                        ? request["candidate_variant"]
                                        : py::int_(0);
  if (focus_net_id >= 0) {
    candidate["focus_net_id"] = focus_net_id;
  }
  if (!source_tree_digest.empty()) {
    candidate["source_tree_digest"] = source_tree_digest;
  }
  candidate["replacement_tree"] = replacement_tree;
  if (replacement_tree) {
    candidate["replacement_source"] = "native_current_driver_tree_vgdp";
    candidate["replacement_mode"] = "current_tree_bundle";
  }

  std::ostringstream digest;
  if (!source_tree_digest.empty()) {
    digest << "source_tree=" << source_tree_digest << ';';
  }
  if (focus_net_id >= 0) {
    digest << "focus_net=" << focus_net_id << ';';
  }
  for (const auto& value : candidate["actions"].cast<py::list>()) {
    digest << actionKey(value.cast<py::dict>()) << ';';
  }
  candidate["candidate_key"] = digest.str();
  return candidate;
}


namespace {
py::dict planFromPayload(const py::dict& request, const py::dict& payload,
                         const py::dict& outer_config)
{
  if (payload.contains("status") && payload["status"].cast<std::string>() != "ok") {
    return copyDict(payload);
  }
  if (!payload.contains("current_tree")) return unsupported("native_tree_payload_missing_tree");
  if (!payload.contains("design") || !payload.contains("timing")
      || !payload.contains("liberty") || !payload.contains("characterization")) {
    return unsupported("native_tree_payload_missing_view");
  }
  try {
    const auto tree = payload["current_tree"].cast<std::shared_ptr<CurrentDriverTree>>();
    const auto design = rebuffer_python::designFromDict(payload["design"].cast<py::dict>());
    const auto timing = rebuffer_python::timingFromDict(payload["timing"].cast<py::dict>());
    const auto liberty = rebuffer_python::libertyFromDict(payload["liberty"].cast<py::dict>());
    const auto characterization = rebuffer_python::characterizationFromDict(
        payload["characterization"].cast<py::dict>());
    const auto config = payload.contains("config") && py::isinstance<py::dict>(payload["config"])
        ? payload["config"].cast<py::dict>() : py::dict();
    NativeTreePlannerViews views{design, timing, liberty, characterization, config, tree,
        payload.contains("net_name") ? payload["net_name"].cast<std::string>() : "",
        payload.contains("focus_net_id") ? payload["focus_net_id"].cast<int>()
            : (request.contains("focus_net_id") ? request["focus_net_id"].cast<int>() : -1)};
    views.candidate_variant = payload.contains("candidate_variant")
        ? payload["candidate_variant"].cast<int>()
        : (request.contains("candidate_variant") ? request["candidate_variant"].cast<int>() : 0);
    return planNativeCurrentTree(request, views, outer_config);
  } catch (const std::exception& error) {
    return unsupported(error.what());
  }
}

}  // namespace

py::dict runRepairTimingNativeSnapshotOperator(
    const py::object& timing_snapshot,
    const py::object& checkpoint,
    const py::object& apply,
    const py::object& evaluate,
    const py::object& restore,
    const py::dict& config)
{
  const bool snapshot_probe_only
      = config.contains("native_snapshot_probe_only")
        && py::cast<bool>(config["native_snapshot_probe_only"]);
  if (snapshot_probe_only) {
    const py::dict snapshot
        = callDictNoArg(timing_snapshot, "timing_snapshot");
    py::dict result;
    result["status"] = "ok";
    result["operator"] = "rebuffer_native_repair_timing_native_snapshot_probe";
    result["native_candidate_planner"] = true;
    result["native_payload_snapshot"] = true;
    result["native_snapshot_probe_only"] = true;
    result["core"] = "cpp";
    if (snapshot.contains("snapshot_statistics")
        && py::isinstance<py::dict>(snapshot["snapshot_statistics"])) {
      result["snapshot_statistics"] = snapshot["snapshot_statistics"];
    } else {
      py::dict statistics;
      statistics["item_count"] = snapshot.contains("items")
                                        ? py::len(snapshot["items"])
                                        : 0;
      result["snapshot_statistics"] = std::move(statistics);
    }
    int planner_payload_rows = 0;
    if (snapshot.contains("_native_planning_state")) {
      planner_payload_rows = snapshot["planner_payload_rows"].cast<int>();
    } else if (snapshot.contains("planner_payloads")
        && py::isinstance<py::list>(snapshot["planner_payloads"])) {
      planner_payload_rows = py::len(snapshot["planner_payloads"]);
    }
    result["planner_payload_rows"] = planner_payload_rows;
    result["planner_payload_hits"] = 0;
    result["planner_payload_misses"] = 0;
    result["planner_payload_requests"] = 0;
    result["accepted_actions"] = py::list();
    result["accepted_count"] = 0;
    result["passes"] = py::list();
    return result;
  }
  auto latest_snapshot = std::make_shared<py::dict>();
  auto payload_hits = std::make_shared<int>(0);
  auto payload_misses = std::make_shared<int>(0);
  py::cpp_function snapshot_adapter = py::cpp_function(
      [timing_snapshot, latest_snapshot]() {
        *latest_snapshot = callDictNoArg(timing_snapshot, "timing_snapshot");
        return *latest_snapshot;
      });
  py::cpp_function native_provider = py::cpp_function(
      [latest_snapshot, config, payload_hits, payload_misses](
          const py::object& request_value) {
        const py::dict request = request_value.cast<py::dict>();
        if (latest_snapshot->contains("_native_planning_state")) {
          ++(*payload_hits);
          try {
            return (*latest_snapshot)["_native_planning_state"]
                .cast<std::shared_ptr<NativeTimingPlanningSnapshot>>()->plan(request, config);
          } catch (const std::exception& error) {
            return unsupported(error.what());
          }
        }
        if (!latest_snapshot->contains("planner_payloads")
            || !py::isinstance<py::list>(
                (*latest_snapshot)["planner_payloads"])) {
          ++(*payload_misses);
          return unsupported("native_snapshot_requires_planner_payloads");
        }
        const int endpoint = request.contains("endpoint_pin")
                                 ? py::cast<int>(request["endpoint_pin"])
                                 : -1;
        const int root_net = request.contains("root_net_id")
                                 ? py::cast<int>(request["root_net_id"])
                                 : -1;
        const int net_id = request.contains("net_id")
                               ? py::cast<int>(request["net_id"])
                               : -1;
        for (const auto& value : (*latest_snapshot)["planner_payloads"]
                                     .cast<py::list>()) {
          const py::dict row = value.cast<py::dict>();
          if (!row.contains("payload") || !row.contains("endpoint_pin")
              || !row.contains("root_net_id") || !row.contains("net_id")) {
            continue;
          }
          if (py::cast<int>(row["endpoint_pin"]) != endpoint
              || py::cast<int>(row["root_net_id"]) != root_net
              || py::cast<int>(row["net_id"]) != net_id) {
            continue;
          }
          ++(*payload_hits);
          // The bridge may defer export until the queue actually requests a
          // row; most window rows are never visited before the next epoch.
          if (PyCallable_Check(row["payload"].ptr())) {
            row["payload"] = row["payload"]();
          }
          return planFromPayload(
              request, row["payload"].cast<py::dict>(), config);
        }
        ++(*payload_misses);
        return unsupported("native_snapshot_planner_payload_not_found");
      });
  py::dict result = runRepairTimingQueueOperator(snapshot_adapter,
                                                  native_provider,
                                                  checkpoint,
                                                  apply,
                                                  evaluate,
                                                  restore,
                                                  config);
  result["operator"] = "rebuffer_native_repair_timing_native_snapshot";
  result["native_candidate_planner"] = true;
  result["native_payload_snapshot"] = true;
  result["core"] = "cpp";
  result["planner_payload_hits"] = *payload_hits;
  result["planner_payload_misses"] = *payload_misses;
  result["planner_payload_requests"] = *payload_hits + *payload_misses;
  return result;
}

}  // namespace rebuffer
