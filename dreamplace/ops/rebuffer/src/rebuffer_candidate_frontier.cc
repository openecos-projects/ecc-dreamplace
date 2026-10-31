// SPDX-License-Identifier: BSD-3-Clause
#include "rebuffer_candidate_frontier.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <tuple>
#include "rebuffer_timing_policy.hh"
#include "rebuffer_electrical_observation.hh"

namespace py = pybind11;
namespace rebuffer {
namespace {

double number(const py::dict& values, const char* key)
{
  return values.contains(key) && !values[key].is_none()
             ? py::cast<double>(values[key])
             : std::numeric_limits<double>::quiet_NaN();
}

bool valid(const py::dict& observation)
{
  return (!observation.contains("status")
          || py::cast<std::string>(observation["status"]) == "ok")
         && std::isfinite(number(observation, "wns_ps"))
         && std::isfinite(number(observation, "tns_ps"));
}

bool better(const py::dict& lhs, int lhs_count,
            const py::dict& rhs, int rhs_count)
{
  // Stable tie order: WNS, TNS, then fewer physical insertions.
  return std::make_tuple(number(lhs, "wns_ps"), number(lhs, "tns_ps"), -lhs_count)
         > std::make_tuple(number(rhs, "wns_ps"), number(rhs, "tns_ps"), -rhs_count);
}

py::list bundleActions(const py::list& source, const py::dict& candidate,
                       const py::dict& request, int rank)
{
  if (!candidate.contains("source_tree_digest")) return source;
  const std::string id = py::cast<std::string>(candidate["source_tree_digest"])
      + ":" + (request.contains("attempt_index") ? py::str(request["attempt_index"]).cast<std::string>() : "0")
      + ":" + std::to_string(rank);
  py::list actions;
  for (const auto& value : source) {
    py::dict action = value.cast<py::dict>().attr("copy")().cast<py::dict>();
    action["tree_bundle_id"] = id;
    actions.append(std::move(action));
  }
  return actions;
}

}  // namespace

void evaluateTimingFrontier(
    const py::dict& candidate, const py::dict& current,
    const py::dict& request, const TimingQueueItem& item,
    const py::object& anchor, const py::object& apply,
    const py::object& evaluate, const py::object& restore,
    const py::dict& config, int evaluation_budget,
    TimingFrontierEvaluation& result)
{
  const int limit = config.contains("python_timing_candidate_top_k")
      ? std::max(1, py::cast<int>(config["python_timing_candidate_top_k"])) : 1;
  py::list frontier;
  if (candidate.contains("candidate_frontier")) {
    frontier = candidate["candidate_frontier"].cast<py::list>();
  }
  if (frontier.size() == 0) frontier.append(candidate["actions"]);
  int active_rank = -1;
  bool selected_acceptable = false;
  std::string last_failure;
  for (int rank = 0; rank < std::min(limit, static_cast<int>(frontier.size())); ++rank) {
    // Once a valid incumbent exists, reserve one TP call to reapply it if
    // the next alternative is worse or fails after partially mutating state.
    const int required = result.selected_rank < 0 ? 1 : 2;
    if (evaluation_budget - result.evaluation_calls < required) break;
    if (rank != 0) restore(anchor);
    active_rank = -1;
    const auto actions = bundleActions(frontier[rank].cast<py::list>(), candidate, request, rank);
    py::dict row;
    row["frontier_rank"] = rank;
    row["action_count"] = py::len(actions);
    py::dict observation;
    try {
      apply(actions);
      ++result.evaluation_calls;
      observation = bindTimingQueueObservation(evaluate().cast<py::dict>(), item);
      if (!valid(observation)) throw py::value_error("candidate_tp_observation_invalid");
      active_rank = rank;
    } catch (const std::exception& error) {
      // Restore failure must propagate to the queue; never evaluate the next
      // alternative after an unsuccessful rollback.
      row["status"] = "evaluate_failed";
      row["reason"] = error.what();
      last_failure = error.what();
      result.trials.append(std::move(row));
      continue;
    }
    const bool acceptable = acceptsTimingCandidate(current, observation, request, config);
    row["status"] = "evaluated";
    row["acceptable"] = acceptable;
    row["electrical_check"] = compareElectricalObservations(current, observation);
    for (const char* field : {"wns_ps", "tns_ps", "endpoint_slack_ps"}) {
      if (observation.contains(field)) row[field] = observation[field];
    }
    result.trials.append(std::move(row));
    if (result.selected_rank < 0 || (acceptable && !selected_acceptable)
        || (acceptable == selected_acceptable
            && better(observation, py::len(actions), result.observation,
                      py::len(result.candidate["actions"])))) {
      result.selected_rank = rank;
      selected_acceptable = acceptable;
      result.observation = observation;
      result.candidate = candidate.attr("copy")().cast<py::dict>();
      result.candidate["actions"] = actions;
    }
  }
  if (result.selected_rank < 0) {
    throw py::value_error("tree_frontier_has_no_valid_tp_candidate: " + last_failure);
  }
  if (active_rank != result.selected_rank) {
    restore(anchor);
    apply(result.candidate["actions"]);
    ++result.evaluation_calls;
    result.observation = bindTimingQueueObservation(evaluate().cast<py::dict>(), item);
    if (!valid(result.observation)) {
      throw py::value_error("selected_tree_tp_observation_invalid");
    }
  }
}

}  // namespace rebuffer
