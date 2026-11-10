// SPDX-License-Identifier: BSD-3-Clause
// LEGACY: first follow global WNS, then walk the initially sorted endpoints.
#include "rebuffer_legacy_queue.hh"
#include "rebuffer_candidate_frontier.hh"
#include "rebuffer_electrical_observation.hh"
#include "rebuffer_timing_policy.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include "pybind11/stl.h"

namespace py = pybind11;
namespace rebuffer {
namespace {

double number(const py::dict& values, const char* key, double fallback = 0.0)
{
  return values.contains(key) ? values[key].cast<double>() : fallback;
}

py::dict copy(const py::dict& value)
{
  return value.attr("copy")().cast<py::dict>();
}

py::object token(const py::object& checkpoint, const std::string& label)
{
  py::object result = checkpoint(label);
  if (py::isinstance<py::dict>(result)) {
    auto record = result.cast<py::dict>();
    if (record.contains("checkpoint_id")) return record["checkpoint_id"];
    if (record.contains("token")) return record["token"];
  }
  return result;
}

void releaseCheckpoint(const py::object& callback, const py::object& value)
{
  if (callback.is_none()) return;
  if (!PyCallable_Check(callback.ptr())) {
    throw py::type_error("native LEGACY requires release_checkpoint callback");
  }
  callback(value);
}

std::vector<TimingEndpoint> endpoints(const py::object& callback, double margin)
{
  auto state = callback().cast<std::shared_ptr<TimingEndpointState>>();
  std::vector<TimingEndpoint> result;
  for (const auto& endpoint : state->endpoints) {
    if (std::isfinite(endpoint.slack) && endpoint.slack < margin) result.push_back(endpoint);
  }
  std::stable_sort(result.begin(), result.end(), [](const auto& a, const auto& b) {
    return a.slack < b.slack;
  });
  return result;
}

py::dict observe(const py::object& evaluate)
{
  auto value = evaluate().cast<py::dict>();
  if ((value.contains("status") && value["status"].cast<std::string>() != "ok")
      || !std::isfinite(number(value, "wns_ps", NAN))
      || !std::isfinite(number(value, "tns_ps", NAN))) {
    throw py::value_error("LEGACY requires a valid timing observation");
  }
  return value;
}

}  // namespace

py::dict runRepairTimingLegacyQueueOperator(
    const py::object& endpoint_state, const py::object& timing_snapshot,
    const py::object& candidate_provider, const py::object& checkpoint,
    const py::object& apply, const py::object& evaluate,
    const py::object& restore, const py::dict& config,
    const py::object& release_checkpoint)
{
  if (endpoint_state.is_none()) throw py::value_error("LEGACY requires endpoint_state callback");
  const int max_attempts = static_cast<int>(number(config, "max_passes", 50));
  const int max_passes = static_cast<int>(number(config, "legacy_max_passes_per_endpoint", 50));
  const int max_evaluations = static_cast<int>(number(config, "max_tp_evaluations", 0));
  const int decreasing_limit = static_cast<int>(number(config, "decreasing_slack_max_passes", 5));
  const double fraction = number(config, "repair_tns_end_percent", 1.0);
  const double margin = number(config, "setup_slack_margin_ps", 0.0);
  if (max_attempts < 0 || max_passes < 1 || decreasing_limit < 0 || max_evaluations < 0
      || !std::isfinite(fraction) || fraction < 0 || fraction > 1
      || !std::isfinite(margin) || margin > 0) {
    throw py::value_error("invalid LEGACY repair budget");
  }
  if (number(config, "max_repairs_per_pass", 1) != 1) {
    throw py::value_error("native LEGACY currently requires max_repairs_per_pass=1");
  }
  py::dict policy = copy(config);
  policy["acceptance"] = "openroad_legacy";
  const py::dict initial = observe(evaluate);
  py::dict current = initial;
  const auto ordered = endpoints(endpoint_state, margin);
  const int end_count = ordered.empty() ? 0
      : std::max(1, static_cast<int>(ordered.size() * fraction));
  py::list history;
  std::vector<py::dict> actions;
  int attempts = 0, evaluations = 1, snapshots = 0, pass_count = 0;
  int opto_iteration = 0;
  const double initial_tns = number(initial, "tns_ps");
  double prev_tns = initial_tns;
  double fix_rate_threshold = 0.0001;
  bool prev_termination = false, two_cons_terminations = false;
  auto advance_opto_iteration = [&](const py::dict& best_state) {
    ++opto_iteration;
    if (opto_iteration % 1000 == 0) {
      fix_rate_threshold *= 2.0;
    }
    if (opto_iteration % 100 == 0) {
      const double curr_tns = number(best_state, "tns_ps");
      const double inc_fix_rate = initial_tns != 0.0
          ? (prev_tns - curr_tns) / initial_tns : 0.0;
      prev_tns = curr_tns;
      if (opto_iteration > 1000 && inc_fix_rate < fix_rate_threshold) {
        if (prev_termination) two_cons_terminations = true;
        else prev_termination = true;
        return true;
      }
      prev_termination = false;
    }
    return false;
  };
  std::string stop = "endpoints_exhausted";
  const auto exhausted = [&]() {
    return attempts >= max_attempts
        || (max_evaluations > 0 && evaluations >= max_evaluations);
  };
  const py::object root = token(checkpoint, "legacy_root");
  // The bridge and the C++ scheduler share this configuration object. Only
  // the selected endpoint is expanded by the next native planning snapshot.
  py::dict published = config;
  const bool had_endpoint = published.contains("native_endpoint_pin");
  const py::object previous_endpoint = had_endpoint
      ? py::reinterpret_borrow<py::object>(published["native_endpoint_pin"]) : py::none();
  auto clear_endpoint = [&]() {
    if (had_endpoint) published["native_endpoint_pin"] = previous_endpoint;
    else if (published.contains("native_endpoint_pin")) published.attr("pop")("native_endpoint_pin");
  };
  try {
    for (int slot = 0; slot < end_count && !exhausted(); ++slot) {
      auto active = endpoints(endpoint_state, margin);
      if (active.empty()) { stop = "timing_closed"; break; }
      auto selected = slot == 0 ? active.begin()
          : std::find_if(active.begin(), active.end(), [&](const auto& endpoint) {
              return endpoint.name == ordered[slot].name;
            });
      if (selected == active.end()) continue;
      TimingEndpoint endpoint = *selected;
      const std::string phase = slot == 0 ? "wns" : "endpoint";
      py::object best_token = token(checkpoint, "legacy_endpoint_" + std::to_string(slot));
      py::dict best = copy(current);
      double best_endpoint_slack = endpoint.slack;
      std::size_t best_action_count = actions.size();
      int decreasing = 0;
      bool provisional = false;
      for (int pass = 1; pass <= max_passes && !exhausted(); ++pass) {
        if (advance_opto_iteration(best)) break;
        // Preserve the LEGACY pass boundary: a newly clean current endpoint
        // ends this search before considering another global worst endpoint.
        if (endpoint.slack >= margin) break;
        published["native_endpoint_pin"] = endpoint.pin;
        const py::dict snapshot = timing_snapshot().cast<py::dict>();
        ++snapshots;
        const auto items = parseTimingQueueSnapshot(snapshot);
        py::dict row;
        row["phase"] = phase;
        row["endpoint_slot"] = slot;
        row["pass"] = pass;
        row["endpoint_pin"] = endpoint.pin;
        row["endpoint_pin_name"] = endpoint.name;
        row["endpoint_slack_ps"] = endpoint.slack;
        row["before_wns_ps"] = current["wns_ps"];
        row["before_tns_ps"] = current["tns_ps"];
        if (snapshot.contains("graph_queue_statistics")) {
          row["path_statistics"] = snapshot["graph_queue_statistics"];
        }
        py::list drivers;
        bool changed = false;
        for (const auto& item : items) {
          if (exhausted()) break;
          if (item.endpoint_pin != endpoint.pin || item.endpoint_pin_name != endpoint.name) {
            throw py::value_error("LEGACY snapshot contains an unrequested endpoint");
          }
          py::dict request;
          request["legacy_phase"] = phase;
          request["endpoint_pin"] = item.endpoint_pin;
          request["endpoint_pin_name"] = item.endpoint_pin_name;
          request["endpoint_slack_ps"] = endpoint.slack;
          request["previous_endpoint_slack_ps"] = best_endpoint_slack;
          request["root_net_id"] = item.root_net_id;
          request["driver_pin_id"] = item.driver_pin_id;
          request["tree_key"] = item.tree_key;
          request["path_pins"] = item.path_pins;
          request["net_ids"] = item.net_ids;
          request["net_id"] = item.root_net_id;
          request["focus_net_id"] = item.root_net_id;
          request["candidate_variant"] = 0;
          request["attempt_index"] = attempts++;
          py::dict driver;
          driver["attempt"] = attempts - 1;
          driver["driver_pin_id"] = item.driver_pin_id;
          driver["net_id"] = item.root_net_id;
          driver["path_pins"] = item.path_pins;
          if (item.has_driver_score) driver["score_ps"] = item.driver_score_ps;
          const auto candidate = candidate_provider(request).cast<py::dict>();
          const auto status = candidate.contains("status")
              ? candidate["status"].cast<std::string>() : "ok";
          driver["status"] = status;
          if (status != "ok" || !candidate.contains("actions") || py::len(candidate["actions"]) == 0) {
            drivers.append(std::move(driver));
            continue;
          }
          const auto trial = token(checkpoint, "legacy_attempt_" + std::to_string(attempts - 1));
          bool trial_checkpoint_live = true;
          const auto release_trial = [&]() {
            if (trial_checkpoint_live) {
              releaseCheckpoint(release_checkpoint, trial);
              trial_checkpoint_live = false;
            }
          };
          TimingFrontierEvaluation frontier;
          try {
            evaluateTimingFrontier(candidate, best, request, item, trial, apply, evaluate,
                restore, policy, max_evaluations > 0 ? max_evaluations - evaluations
                                                    : std::numeric_limits<int>::max(), frontier);
          } catch (const std::exception& error) {
            evaluations += frontier.evaluation_calls;
            restore(trial);
            release_trial();
            driver["status"] = "evaluate_failed";
            driver["reason"] = error.what();
            drivers.append(std::move(driver));
            continue;
          }
          evaluations += frontier.evaluation_calls;
          if (!electricalNonRegressing(best, frontier.observation)) {
            restore(trial);
            release_trial();
            driver["status"] = "electrical_rejected";
            drivers.append(std::move(driver));
            continue;
          }
          current = frontier.observation;
          release_trial();
          endpoint.slack = current["endpoint_slack_ps"].cast<double>();
          for (const auto& action : frontier.candidate["actions"].cast<py::list>()) {
            actions.push_back(copy(action.cast<py::dict>()));
          }
          changed = true;
          provisional = true;
          driver["status"] = "applied";
          driver["action_count"] = py::len(frontier.candidate["actions"]);
          drivers.append(std::move(driver));
          break;  // One successful driver repair per configured pass.
        }
        ++pass_count;
        row["drivers"] = std::move(drivers);
        row["after_wns_ps"] = current["wns_ps"];
        row["after_tns_ps"] = current["tns_ps"];
        row["endpoint_slack_after_ps"] = endpoint.slack;
        py::dict decision;
        decision["legacy_phase"] = phase;
        decision["previous_endpoint_slack_ps"] = best_endpoint_slack;
        if (!changed) {
          row["status"] = "no_change";
          history.append(std::move(row));
          break;
        }
        if (acceptsTimingCandidate(best, current, decision, policy)) {
          best = copy(current);
          best_endpoint_slack = endpoint.slack;
          best_action_count = actions.size();
          const auto previous_best_token = best_token;
          best_token = token(checkpoint, "legacy_best_" + std::to_string(pass_count));
          releaseCheckpoint(release_checkpoint, previous_best_token);
          decreasing = 0;
          provisional = false;
          row["status"] = "accepted";
          row["decreasing_slack_passes"] = decreasing;
          history.append(std::move(row));
        } else {
          ++decreasing;
          row["status"] = "provisional";
          row["decreasing_slack_passes"] = decreasing;
          history.append(std::move(row));
          if (decreasing > decreasing_limit) break;
        }
        if (slot == 0 && endpoint.slack < margin) {
          const auto now = endpoints(endpoint_state, margin);
          if (!now.empty()) endpoint = now.front();
        }
      }
      if (provisional) {
        restore(best_token);
        current = best;
        actions.resize(best_action_count);
        py::dict row;
        row["phase"] = phase;
        row["endpoint_slot"] = slot;
        row["status"] = "restored_best";
        row["restored_action_count"] = best_action_count;
        history.append(std::move(row));
      }
      releaseCheckpoint(release_checkpoint, best_token);
      if (two_cons_terminations) {
        stop = "no_tns_progress";
        break;
      }
    }
  } catch (...) {
    clear_endpoint();
    restore(root);
    releaseCheckpoint(release_checkpoint, root);
    throw;
  }
  clear_endpoint();
  releaseCheckpoint(release_checkpoint, root);
  if (attempts >= max_attempts) stop = "max_attempts";
  else if (max_evaluations > 0 && evaluations >= max_evaluations) stop = "max_tp_evaluations";
  py::list accepted;
  for (const auto& action : actions) accepted.append(action);
  py::dict result;
  result["status"] = "ok";
  result["operator"] = "rebuffer_native_repair_timing_legacy";
  result["scheduler"] = "openroad_legacy";
  result["core"] = "cpp";
  result["initial"] = initial;
  result["baseline"] = initial;
  result["final"] = current;
  result["passes"] = std::move(history);
  result["accepted_actions"] = std::move(accepted);
  result["accepted_count"] = actions.size();
  result["attempt_count"] = attempts;
  result["provider_calls"] = attempts;
  result["pass_count"] = pass_count;
  result["evaluation_calls"] = evaluations;
  result["stop_reason"] = stop;
  py::dict statistics;
  statistics["snapshot_count"] = snapshots;
  statistics["initial_violating_endpoint_count"] = ordered.size();
  statistics["endpoint_budget"] = end_count;
  result["snapshot_statistics"] = statistics;
  return result;
}

}  // namespace rebuffer
