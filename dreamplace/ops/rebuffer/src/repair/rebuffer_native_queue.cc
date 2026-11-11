// SPDX-License-Identifier: BSD-3-Clause

#include "rebuffer_queue_snapshot.hh"


#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "pybind11/stl.h"

#include "rebuffer_timing_planning.hh"


namespace py = pybind11;

namespace rebuffer {

namespace {

struct QueueWorkItem
{
  std::size_t item_index = 0;
  int net_id = -1;
  int candidate_variant = 0;
};

bool callable(const py::object& callback)
{
  return callback && !callback.is_none() && PyCallable_Check(callback.ptr());
}

py::dict callDictNoArg(const py::object& callback, const char* name)
{
  if (!callable(callback)) {
    throw py::type_error(std::string("native queue requires ") + name
                         + " callback");
  }
  py::object value = callback();
  if (!py::isinstance<py::dict>(value)) {
    throw py::type_error(std::string("native queue ") + name
                         + " callback must return a dict");
  }
  return value.cast<py::dict>();
}

py::dict callDict(const py::object& callback,
                 const py::object& argument,
                 const char* name)
{
  if (!callable(callback)) {
    throw py::type_error(std::string("native queue requires ") + name
                         + " callback");
  }
  py::object value = callback(argument);
  if (!py::isinstance<py::dict>(value)) {
    throw py::type_error(std::string("native queue ") + name
                         + " callback must return a dict");
  }
  return value.cast<py::dict>();
}

py::dict copyDict(const py::dict& values)
{
  return values.attr("copy")().cast<py::dict>();
}

double valueOr(const py::dict& values, const char* key, double fallback)
{
  if (!values.contains(key) || values[key].is_none()) {
    return fallback;
  }
  return py::cast<double>(values[key]);
}

int configInt(const py::dict& config, const char* key, int fallback)
{
  if (!config.contains(key) || config[key].is_none()) {
    return fallback;
  }
  return std::max(0, py::cast<int>(config[key]));
}

bool configBool(const py::dict& config, const char* key, bool fallback)
{
  if (!config.contains(key) || config[key].is_none()) {
    return fallback;
  }
  return py::cast<bool>(config[key]);
}

std::string stringOr(const py::dict& values,
                     const char* key,
                     const char* fallback = "")
{
  if (!values.contains(key) || values[key].is_none()) {
    return fallback;
  }
  return py::cast<std::string>(values[key]);
}

std::vector<int> candidateNetIds(const TimingQueueItem& item, const py::dict& config)
{
  std::vector<int> net_ids = item.net_ids;
  if (configBool(config, "native_tree_queue", false)
      && item.root_net_id >= 0) {
    net_ids = {item.root_net_id};
  }
  if (configBool(config, "native_tree_branch_queue", false)
      && !item.focus_net_ids.empty()) {
    net_ids = item.focus_net_ids;
    const auto root = std::find(net_ids.begin(), net_ids.end(), item.root_net_id);
    if (root != net_ids.end() && root != net_ids.begin()) {
      const int root_net_id = *root;
      net_ids.erase(root);
      net_ids.insert(net_ids.begin(), root_net_id);
    }
  }
  return net_ids;
}

int candidateVariantCount(const TimingQueueItem& item,
                          int net_id,
                          const py::dict& config)
{
  if (!configBool(config, "native_tree_queue", false)) {
    return 1;
  }
  const bool branch_focus = configBool(config, "native_tree_branch_queue", false)
                            && net_id != item.root_net_id;
  const int variants = std::max(1, configInt(config,
      branch_focus ? "native_tree_branch_candidate_variants"
                   : "native_tree_candidate_variants", branch_focus ? 2 : 1));
  const int top_k = std::max(1, configInt(config, "python_timing_candidate_top_k", 1));
  return (variants + top_k - 1) / top_k;
}

py::dict checkpointValue(const py::object& callback, const std::string& label)
{
  if (!callable(callback)) {
    throw py::type_error("native queue requires checkpoint callback");
  }
  py::object value = callback(label);
  if (py::isinstance<py::dict>(value)) {
    return value.cast<py::dict>();
  }
  py::dict result;
  result["token"] = std::move(value);
  return result;
}

py::object checkpointToken(const py::dict& checkpoint)
{
  py::object token = py::reinterpret_borrow<py::object>(checkpoint);
  if (checkpoint.contains("checkpoint_id")) {
    token = checkpoint["checkpoint_id"];
    return token;
  }
  if (checkpoint.contains("token")) {
    token = checkpoint["token"];
    return token;
  }
  return token;
}

void restore(const py::object& callback, const py::object& token)
{
  if (!callable(callback)) {
    throw py::type_error("native queue requires restore callback");
  }
  callback(token);
}

void releaseCheckpoint(const py::object& callback, const py::object& token)
{
  if (callback.is_none()) return;
  if (!callable(callback)) {
    throw py::type_error("native queue requires release_checkpoint callback");
  }
  callback(token);
}

std::string candidateKey(const py::dict& candidate, const py::dict& request)
{
  if (candidate.contains("candidate_key")) {
    const auto requestValue = [&](const char* field) {
      if (!request.contains(field) || request[field].is_none()) {
        return std::string();
      }
      return py::str(request[field]).cast<std::string>();
    };
    std::ostringstream key;
    key << stringOr(candidate, "candidate_key")
        << "|endpoint=" << requestValue("endpoint_pin")
        << "|root=" << requestValue("root_net_id")
        << "|focus=" << requestValue("focus_net_id")
        << "|variant=" << requestValue("candidate_variant");
    if (request.contains("path_pins")
        && py::isinstance<py::list>(request["path_pins"])) {
      key << "|path=";
      for (const auto& value : request["path_pins"].cast<py::list>()) {
        key << py::cast<int>(value) << ',';
      }
    }
    return key.str();
  }
  return {};
}

bool isInsertedBufferAction(const py::dict& action)
{
  return !action.contains("action_kind")
         || stringOr(action, "action_kind") != "resize_cell";
}

int countInsertedBuffers(const py::list& actions)
{
  int count = 0;
  for (const auto& item : actions) {
    if (isInsertedBufferAction(item.cast<py::dict>())) {
      ++count;
    }
  }
  return count;
}

int countInsertedBuffers(const std::vector<py::dict>& actions,
                         std::size_t limit)
{
  int count = 0;
  const std::size_t end = std::min(limit, actions.size());
  for (std::size_t i = 0; i < end; ++i) {
    if (isInsertedBufferAction(actions[i])) {
      ++count;
    }
  }
  return count;
}

}  // namespace

py::dict runRepairTimingQueueOperator(const py::object& timing_snapshot,
                                      const py::object& candidate_provider,
                                      const py::object& checkpoint,
                                      const py::object& apply,
                                      const py::object& evaluate,
                                      const py::object& restore_callback,
                                      const py::dict& config,
                                      const py::object& release_checkpoint)
{
  const int max_attempts = configInt(config, "max_passes", 50);
  const int max_tp_evaluations = configInt(config, "max_tp_evaluations", 0);
  const int max_buffers = configInt(config, "max_buffers", 0);
  py::dict current = callDictNoArg(evaluate, "evaluate");
  if (current.contains("status")
      && stringOr(current, "status") != "ok") {
    py::dict result;
    result["status"] = "unsupported";
    result["operator"] = "rebuffer_native_repair_timing_queue";
    result["core"] = "cpp";
    result["unsupported_reasons"] = py::make_tuple("initial_evaluate_failed");
    return result;
  }

  const py::dict initial = copyDict(current);
  py::list history;
  py::list accepted_actions;
  std::set<std::tuple<int, int, int, int>> attempted;
  std::set<std::string> candidate_keys;
  // Endpoint-independent identities of candidates rejected in the current
  // timing state.  Acceptance is a function of global WNS/TNS/ERC only (when
  // the endpoint journal tie-break is off), so re-evaluating the same tree
  // bundle for another endpoint cannot change the outcome.  Cleared whenever
  // the design changes.
  std::set<std::string> rejected_in_state;
  const bool endpoint_tiebreak
      = configBool(config, "allow_decreasing_slack", false)
        && config.contains("acceptance")
        && py::cast<std::string>(config["acceptance"]) == "endpoint_journal";
  std::set<int> visited_endpoints;
  std::set<int> visited_nets;
  std::set<int> visited_trees;
  std::set<int> visited_drivers;
  std::set<int> visited_branches;
  std::set<int> visited_focus_nets;
  std::set<int> accepted_nets;
  // Attempts per endpoint, keyed by stable pin name: integer pin ids move
  // after each insertion.  Round-robin keeps the budget across acceptances;
  // resetting it on improvement concentrated passes on the WNS endpoint and
  // lost TNS on BM64 (1000 passes: -365047 -> -369522 ps).
  std::map<std::string, int> endpoint_attempts;
  const auto endpointKey = [](const TimingQueueItem& item) {
    return item.endpoint_pin_name.empty() ? std::to_string(item.endpoint_pin)
                                          : item.endpoint_pin_name;
  };
  int provider_calls = 0;
  int skipped_rejected = 0;
  int evaluation_calls = 1;
  const auto evaluationBudgetExhausted = [&]() {
    return max_tp_evaluations > 0 && evaluation_calls >= max_tp_evaluations;
  };
  int accepted_count = 0;
  int inserted_buffer_count = 0;
  const auto bufferBudgetExhausted = [&]() {
    return max_buffers > 0 && inserted_buffer_count >= max_buffers;
  };
  int attempt_count = 0;
  int snapshot_count = 0;
  int snapshot_item_count = 0;
  int snapshot_buffer_output_root_count = 0;
  std::set<int> snapshot_endpoints;
  std::set<int> snapshot_roots;
  std::set<int> snapshot_drivers;
  const bool gradient_ordering = configBool(config, "gradient_ordering", false);
  const int gradient_reject_limit = gradient_ordering
                                        ? configInt(config, "gradient_reject_limit", 0)
                                        : 0;
  int gradient_refresh_count = 0;
  std::string stop_reason = "max_passes";
  const bool round_robin = configBool(config, "native_tree_round_robin", false);
  int round_robin_endpoint = -1;
  int round_robin_root_net = -1;
  const int decreasing_slack_limit = configInt(
      config,
      "decreasing_slack_max_passes",
      0);
  const bool native_decreasing_slack_journal = configBool(
      config,
      "native_openroad_decreasing_slack_journal",
      false);
  bool journal_active = false;
  int journal_passes = 0;
  int journal_endpoint = -1;
  int journal_root_net = -1;
  py::object journal_anchor_token = py::none();
  py::dict journal_anchor_current;
  py::list journal_actions;

  while (attempt_count < max_attempts) {
    if (bufferBudgetExhausted()) {
      stop_reason = "max_buffers";
      break;
    }
    if (evaluationBudgetExhausted()) {
      stop_reason = "max_tp_evaluations";
      break;
    }
    const int endpoint_attempt_budget = configInt(
        config,
        "native_tree_max_attempts_per_endpoint",
        0);
    // ``published`` aliases the caller's config dict, so the bridge's
    // select_timing_queue_items(snapshot, config) trims the same endpoints.
    std::vector<std::string> retired_names;
    if (endpoint_attempt_budget > 0) {
      for (const auto& [name, attempts] : endpoint_attempts) {
        if (attempts >= endpoint_attempt_budget) retired_names.push_back(name);
      }
    }
    py::dict published = config;
    published["retired_endpoints"] = py::cast(retired_names);
    const py::dict snapshot
        = callDictNoArg(timing_snapshot, "timing_snapshot");
    ++snapshot_count;
    if (snapshot.contains("snapshot_statistics")
        && py::isinstance<py::dict>(snapshot["snapshot_statistics"])) {
      const py::dict statistics = snapshot["snapshot_statistics"].cast<py::dict>();
      snapshot_item_count += configInt(statistics, "item_count", 0);
      snapshot_buffer_output_root_count += configInt(
          statistics,
          "buffer_output_root_count",
          0);
    }
    const std::vector<TimingQueueItem> items = parseTimingQueueSnapshot(snapshot);
    std::set<int> retired_endpoints;
    if (endpoint_attempt_budget > 0) {
      for (const auto& item : items) {
        const auto found = endpoint_attempts.find(endpointKey(item));
        if (found != endpoint_attempts.end() && found->second >= endpoint_attempt_budget) {
          retired_endpoints.insert(item.endpoint_pin);
        }
      }
    }
    if (items.empty()) {
      stop_reason = "no_violating_queue_items";
      break;
    }
    for (const TimingQueueItem& item : items) {
      snapshot_endpoints.insert(item.endpoint_pin);
      snapshot_roots.insert(item.root_net_id);
      if (item.driver_pin_id >= 0) {
        snapshot_drivers.insert(item.driver_pin_id);
      }
    }

    bool accepted_this_epoch = false;
    bool gradient_refresh_requested = false;
    int gradient_rejections = 0;
    bool provider_done = false;
    std::vector<int> item_order = timingQueueWindow(items, config, retired_endpoints);
    if (round_robin && !items.empty()
        && (round_robin_endpoint >= 0 || journal_active)) {
      int start = -1;
      for (int order_index = 0;
           order_index < static_cast<int>(item_order.size());
           ++order_index) {
        const TimingQueueItem& item = items[item_order[order_index]];
        const int target_endpoint
            = journal_active ? journal_endpoint : round_robin_endpoint;
        const int target_root
            = journal_active ? journal_root_net : round_robin_root_net;
        if (item.endpoint_pin == target_endpoint
            && item.root_net_id == target_root) {
          start = journal_active
                     ? order_index
                     : (order_index + 1)
                           % static_cast<int>(item_order.size());
          break;
        }
      }
      if (start >= 0) {
        std::rotate(item_order.begin(),
                    item_order.begin() + start,
                    item_order.end());
      }
    }

    std::vector<QueueWorkItem> work;
    if (!round_robin) {
      for (const int item_index : item_order) {
        const TimingQueueItem& item = items[item_index];
        const std::vector<int> net_ids = candidateNetIds(item, config);
        for (const int net_id : net_ids) {
          const int variants = candidateVariantCount(item, net_id, config);
          for (int candidate_variant = 0; candidate_variant < variants;
               ++candidate_variant) {
            work.push_back(
                {static_cast<std::size_t>(item_index),
                 net_id,
                 candidate_variant});
          }
        }
      }
    } else {
      // OpenROAD's repair loop exposes more than one driver tree before it
      // exhausts all variants of the first tree.  The native queue used to do
      // the opposite: one endpoint, then one tree, then every branch and
      // variant.  Interleave variant/focus/item dimensions so a finite
      // attempt budget measures distinct repair opportunities instead of
      // repeatedly probing one endpoint's current tree.
      std::vector<std::vector<QueueWorkItem>> work_by_item;
      work_by_item.reserve(item_order.size());
      std::size_t max_work_count = 0;
      for (const int item_index : item_order) {
        const std::vector<int> net_ids = candidateNetIds(items[item_index], config);
        work_by_item.emplace_back();
        for (const int net_id : net_ids) {
          const int variants = candidateVariantCount(items[item_index], net_id, config);
          for (int candidate_variant = 0;
               candidate_variant < variants;
               ++candidate_variant) {
            work_by_item.back().push_back(
                {static_cast<std::size_t>(item_index),
                 net_id,
                 candidate_variant});
          }
        }
        max_work_count = std::max(max_work_count, work_by_item.back().size());
      }
      const int endpoint_quantum = configInt(
          config,
          "native_tree_round_robin_endpoint_quantum",
          0);
      if (endpoint_quantum <= 0) {
        // Compatibility mode: expose one candidate slot across the window,
        // then the next slot.  Production parity profiles set a positive
        // quantum so one tree's frontier is not split by the window.
        for (std::size_t slot = 0; slot < max_work_count; ++slot) {
          for (const auto& item_work : work_by_item) {
            if (slot < item_work.size()) {
              work.push_back(item_work[slot]);
            }
          }
        }
      } else {
        for (std::size_t chunk_start = 0;
             chunk_start < max_work_count;
             chunk_start += static_cast<std::size_t>(endpoint_quantum)) {
          const std::size_t chunk_end = std::min(
              max_work_count,
              chunk_start + static_cast<std::size_t>(endpoint_quantum));
          for (const auto& item_work : work_by_item) {
            const std::size_t end = std::min(chunk_end, item_work.size());
            for (std::size_t slot = chunk_start; slot < end; ++slot) {
              work.push_back(item_work[slot]);
            }
          }
        }
      }
    }

    for (const QueueWorkItem& work_item : work) {
      if (attempt_count >= max_attempts || evaluationBudgetExhausted()
          || bufferBudgetExhausted()) {
        break;
      }
      const TimingQueueItem& item = items[work_item.item_index];
      const int net_id = work_item.net_id;
      const int top_k = configBool(config, "native_tree_queue", false)
          ? std::max(1, configInt(config, "python_timing_candidate_top_k", 1)) : 1;
      const int candidate_variant = work_item.candidate_variant * top_k;
      if (endpoint_attempt_budget > 0
          && endpoint_attempts[endpointKey(item)] >= endpoint_attempt_budget) {
        continue;
      }
      visited_endpoints.insert(item.endpoint_pin);
      visited_trees.insert(item.root_net_id);
      if (item.driver_pin_id >= 0) {
        visited_drivers.insert(item.driver_pin_id);
      }
      for (const int path_net_id : item.net_ids) {
        visited_nets.insert(path_net_id);
      }
      const std::vector<int>& branch_ids = item.focus_net_ids.empty()
                                                ? item.net_ids
                                                : item.focus_net_ids;
      for (const int branch_net_id : branch_ids) {
        visited_branches.insert(branch_net_id);
      }
      visited_focus_nets.insert(net_id);
      const auto pair = std::make_tuple(item.endpoint_pin,
                                        item.root_net_id,
                                        net_id,
                                        candidate_variant);
      if (attempted.count(pair) != 0) {
        continue;
      }
      attempted.insert(pair);

      py::dict request = copyDict(current);
      request["endpoint_pin"] = item.endpoint_pin;
      if (!item.endpoint_pin_name.empty()) {
        request["endpoint_pin_name"] = item.endpoint_pin_name;
      }
      request["endpoint_slack_ps"] = item.endpoint_slack_ps;
      request["previous_endpoint_slack_ps"] = item.endpoint_slack_ps;
      if (journal_active && item.endpoint_pin == journal_endpoint
          && item.root_net_id == journal_root_net
          && journal_anchor_current.contains("endpoint_slack_ps")) {
        request["previous_endpoint_slack_ps"]
            = journal_anchor_current["endpoint_slack_ps"];
      }
      request["root_net_id"] = item.root_net_id;
      request["driver_pin_id"] = item.driver_pin_id;
      request["tree_key"] = item.tree_key;
      request["path_pins"] = item.path_pins;
      request["net_ids"] = item.net_ids;
      request["focus_net_id"] = net_id;
      request["net_id"] = net_id;
      request["candidate_variant"] = candidate_variant;
      request["attempt_index"] = attempt_count;
      request["visited_endpoint_count"]
          = static_cast<int>(visited_endpoints.size());
      request["visited_net_count"] = static_cast<int>(visited_nets.size());
      ++attempt_count;
      ++endpoint_attempts[endpointKey(item)];
      py::dict candidate
          = callDict(candidate_provider, request, "candidate_provider");
      ++provider_calls;

      const std::string status = stringOr(candidate, "status", "ok");
      py::dict row;
      row["attempt"] = attempt_count - 1;
      row["candidate_variant"] = candidate_variant;
      row["endpoint_pin"] = item.endpoint_pin;
      if (!item.endpoint_pin_name.empty()) {
        row["endpoint_pin_name"] = item.endpoint_pin_name;
      }
      row["endpoint_slack_ps"] = item.endpoint_slack_ps;
      row["root_net_id"] = item.root_net_id;
      row["driver_pin_id"] = item.driver_pin_id;
      row["tree_key"] = item.tree_key;
      row["focus_net_id"] = net_id;
      row["net_id"] = net_id;
      row["path_pin_count"] = item.path_pins.size();
      row["status"] = status;
      if (status == "done" || configBool(candidate, "stop", false)) {
        provider_done = true;
        stop_reason = candidate.contains("stop_reason")
                          ? stringOr(candidate, "stop_reason")
                          : "candidate_provider_done";
        history.append(std::move(row));
        break;
      }
      if (status != "ok" || !candidate.contains("actions")
          || py::len(candidate["actions"]) == 0) {
        row["status"] = status == "ok" ? "no_actions" : status;
        if (candidate.contains("reason")) {
          row["reason"] = candidate["reason"];
        }
        history.append(std::move(row));
        continue;
      }

      const std::string key = candidateKey(candidate, request);
      if (!key.empty() && candidate_keys.count(key) != 0) {
        row["status"] = "duplicate_candidate";
        history.append(std::move(row));
        continue;
      }
      if (!key.empty()) {
        candidate_keys.insert(key);
      }
      const std::string state_key
          = endpoint_tiebreak || !candidate.contains("candidate_key") ? std::string()
          : stringOr(candidate, "candidate_key") + "|variant="
              + std::to_string(candidate_variant);
      if (!state_key.empty() && rejected_in_state.count(state_key) != 0) {
        // Does not consume the attempt budget: no TP evaluation is made.
        --attempt_count;
        --endpoint_attempts[endpointKey(item)];
        ++skipped_rejected;
        continue;
      }

      const py::dict checkpoint_values
          = checkpointValue(checkpoint, "native_queue_attempt_"
                                           + std::to_string(attempt_count - 1));
      const py::object token = checkpointToken(checkpoint_values);
      const py::dict target_current = bindTimingQueueObservation(current, item);
      row["action_count"] = py::len(candidate["actions"]);
      if (candidate.contains("source_tree_digest")) {
        row["source_tree_digest"] = candidate["source_tree_digest"];
      }
      if (candidate.contains("net_name")) {
        row["net_name"] = candidate["net_name"];
      }
      TimingFrontierEvaluation frontier;
      bool trial_checkpoint_live = true;
      const auto release_trial = [&]() {
        if (trial_checkpoint_live) {
          releaseCheckpoint(release_checkpoint, token);
          trial_checkpoint_live = false;
        }
      };
      try {
        py::dict acceptance_current = current;
        py::dict acceptance_request = request;
        if (journal_active) {
          acceptance_current = journal_anchor_current;
          acceptance_request = copyDict(request);
          if (journal_anchor_current.contains("endpoint_slack_ps")
              && item.endpoint_pin == journal_endpoint
              && item.root_net_id == journal_root_net) {
            acceptance_request["previous_endpoint_slack_ps"]
                = journal_anchor_current["endpoint_slack_ps"];
          }
        }
        evaluateTimingFrontier(
            candidate, acceptance_current, acceptance_request, item, token,
            apply, evaluate, restore_callback, config,
            max_tp_evaluations > 0 ? max_tp_evaluations - evaluation_calls
                                   : std::numeric_limits<int>::max(), frontier);
        candidate = frontier.candidate;
        const py::dict observation = frontier.observation;
        row["action_count"] = py::len(candidate["actions"]);
        if (candidate.contains("tree_search")) row["tree_search"] = candidate["tree_search"];
        row["before_wns_ps"] = valueOr(current, "wns_ps", 0.0);
        row["before_tns_ps"] = valueOr(current, "tns_ps", 0.0);
        row["after_wns_ps"] = valueOr(observation, "wns_ps", 0.0);
        row["after_tns_ps"] = valueOr(observation, "tns_ps", 0.0);
        row["endpoint_slack_after_ps"] = valueOr(
            observation,
            "endpoint_slack_ps",
            valueOr(observation, "worst_slack_ps", 0.0));
        const bool better
            = acceptsTimingCandidate(acceptance_current, observation, acceptance_request, config);
        const bool branch_focus =
            configBool(config, "native_tree_branch_queue", false)
            && net_id != item.root_net_id;
        const bool keep_decreasing_slack =
            native_decreasing_slack_journal
            && !branch_focus
            && configBool(config, "openroad_journal_keep_nonimproving", false)
            && configBool(config, "allow_decreasing_slack", false)
            && decreasing_slack_limit > 0;
        if (better) {
          row["status"] = "accepted";
          const py::list actions = candidate["actions"].cast<py::list>();
          if (journal_active) {
            for (const auto& action : journal_actions) {
              accepted_actions.append(copyDict(action.cast<py::dict>()));
            }
            accepted_count += py::len(journal_actions);
            inserted_buffer_count += countInsertedBuffers(journal_actions);
            row["journal_committed_action_count"] = py::len(journal_actions);
          }
          for (const auto& action : actions) {
            accepted_actions.append(
                copyDict(action.cast<py::dict>()));
          }
          accepted_count += py::len(actions);
          inserted_buffer_count += countInsertedBuffers(actions);
          current = observation;
          accepted_nets.insert(net_id);
          if (round_robin) {
            round_robin_endpoint = item.endpoint_pin;
            round_robin_root_net = item.root_net_id;
          }
          attempted.clear();
          if (!round_robin) {
            endpoint_attempts.clear();
          }
          // The accepted mutation changes the timing state.  A candidate
          // rejected before this mutation must be eligible for the new
          // state; retaining a design-wide key would incorrectly suppress
          // the next endpoint/tree epoch.
          candidate_keys.clear();
          rejected_in_state.clear();
          journal_active = false;
          journal_passes = 0;
          journal_endpoint = -1;
          journal_root_net = -1;
          journal_anchor_token = py::none();
          journal_anchor_current = py::dict();
          journal_actions = py::list();
          accepted_this_epoch = true;
        } else if (keep_decreasing_slack) {
          if (!journal_active) {
            journal_active = true;
            journal_passes = 0;
            journal_endpoint = item.endpoint_pin;
            journal_root_net = item.root_net_id;
            journal_anchor_token = token;
            journal_anchor_current = copyDict(target_current);
            journal_actions = py::list();
          }
          ++journal_passes;
          if (journal_passes <= decreasing_slack_limit) {
            const py::list actions = candidate["actions"].cast<py::list>();
            for (const auto& action : actions) {
              journal_actions.append(copyDict(action.cast<py::dict>()));
            }
            current = observation;
            row["status"] = "journal_kept";
            row["journal_pass"] = journal_passes;
            row["journal_limit"] = decreasing_slack_limit;
            round_robin_endpoint = item.endpoint_pin;
            round_robin_root_net = item.root_net_id;
            // The physical tree changed.  The next epoch must rebuild its
            // payload and may expose a new candidate frontier.
            attempted.clear();
            candidate_keys.clear();
            rejected_in_state.clear();
            accepted_this_epoch = true;
          } else {
            restore(restore_callback, journal_anchor_token);
            current = journal_anchor_current;
            row["status"] = "journal_reverted";
            row["journal_pass"] = journal_passes;
            row["journal_limit"] = decreasing_slack_limit;
            row["journal_reverted_action_count"]
                = py::len(journal_actions) + py::len(candidate["actions"]);
            journal_active = false;
            journal_passes = 0;
            journal_endpoint = -1;
            journal_root_net = -1;
            journal_anchor_token = py::none();
            journal_anchor_current = py::dict();
            journal_actions = py::list();
            // Keep attempted/candidate ledgers from the failed journal so
            // the next snapshot advances to a different frontier.
            accepted_this_epoch = true;
          }
        } else {
          restore(restore_callback, token);
          row["status"] = "rejected";
          if (!state_key.empty()) rejected_in_state.insert(state_key);
          ++gradient_rejections;
          if (gradient_reject_limit > 0
              && gradient_rejections >= gradient_reject_limit) {
            gradient_refresh_requested = true;
          }
        }
      } catch (const std::exception& error) {
        restore(restore_callback, token);
        row["status"] = "evaluate_failed";
        row["reason"] = error.what();
      }
      release_trial();
      // Failed TP calls are part of the same budget and diagnostic ledger.
      evaluation_calls += frontier.evaluation_calls;
      row["selected_frontier_rank"] = frontier.selected_rank;
      row["frontier_trials"] = frontier.trials;
      history.append(std::move(row));
      if (accepted_this_epoch || gradient_refresh_requested) {
        break;
      }
    }
    if (provider_done) {
      break;
    }
    if (accepted_this_epoch) {
      continue;
    }
    if (gradient_refresh_requested) {
      ++gradient_refresh_count;
      attempted.clear();
      candidate_keys.clear();
      rejected_in_state.clear();
      stop_reason = "gradient_reject_limit";
      continue;
    }
    stop_reason = bufferBudgetExhausted() ? "max_buffers"
        : (evaluationBudgetExhausted() ? "max_tp_evaluations"
            : (attempt_count >= max_attempts ? "max_passes" : "queue_exhausted"));
    break;
  }

  if (journal_active) {
    restore(restore_callback, journal_anchor_token);
    current = journal_anchor_current;
    journal_active = false;
  }

  py::dict coverage;
  coverage["visited_endpoint_count"]
      = static_cast<int>(visited_endpoints.size());
  coverage["visited_net_count"] = static_cast<int>(visited_nets.size());
  coverage["visited_tree_count"] = static_cast<int>(visited_trees.size());
  coverage["visited_driver_count"] = static_cast<int>(visited_drivers.size());
  coverage["visited_branch_count"] = static_cast<int>(visited_branches.size());
  coverage["visited_focus_net_count"]
      = static_cast<int>(visited_focus_nets.size());
  coverage["accepted_net_count"] = static_cast<int>(accepted_nets.size());
  coverage["visited_endpoint_pins"] = visited_endpoints;
  coverage["visited_net_ids"] = visited_nets;
  coverage["visited_tree_root_net_ids"] = visited_trees;
  coverage["visited_driver_pin_ids"] = visited_drivers;
  coverage["visited_branch_net_ids"] = visited_branches;
  coverage["visited_focus_net_ids"] = visited_focus_nets;
  coverage["accepted_net_ids"] = accepted_nets;

  py::dict result;
  result["status"] = "ok";
  result["operator"] = "rebuffer_native_repair_timing_queue";
  result["core"] = "cpp";
  result["initial"] = initial;
  result["baseline"] = initial;
  result["final"] = current;
  result["passes"] = std::move(history);
  result["accepted_actions"] = std::move(accepted_actions);
  result["accepted_count"] = accepted_count;
  result["inserted_buffer_count"] = inserted_buffer_count;
  result["max_buffers"] = max_buffers;
  result["provider_calls"] = provider_calls;
  result["skipped_rejected_in_state"] = skipped_rejected;
  result["evaluation_calls"] = evaluation_calls;
  result["attempt_count"] = attempt_count;
  result["gradient_reject_limit"] = gradient_reject_limit;
  result["gradient_refresh_count"] = gradient_refresh_count;
  result["stop_reason"] = stop_reason;
  result["queue_coverage"] = std::move(coverage);
  py::dict snapshot_statistics;
  snapshot_statistics["snapshot_count"] = snapshot_count;
  snapshot_statistics["item_count"] = snapshot_item_count;
  snapshot_statistics["endpoint_count"]
      = static_cast<int>(snapshot_endpoints.size());
  snapshot_statistics["root_count"] = static_cast<int>(snapshot_roots.size());
  snapshot_statistics["driver_count"]
      = static_cast<int>(snapshot_drivers.size());
  snapshot_statistics["buffer_output_root_count"]
      = snapshot_buffer_output_root_count;
  result["snapshot_statistics"] = std::move(snapshot_statistics);
  return result;
}

}  // namespace rebuffer

// --- rebuffer_legacy_queue.cc ---
// LEGACY: first follow global WNS, then walk the initially sorted endpoints.
#include "rebuffer_timing_graph.hh"


#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

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
  const int max_buffers = static_cast<int>(number(config, "max_buffers", 0));
  const int decreasing_limit = static_cast<int>(number(config, "decreasing_slack_max_passes", 5));
  const double fraction = number(config, "repair_tns_end_percent", 1.0);
  const double margin = number(config, "setup_slack_margin_ps", 0.0);
  if (max_attempts < 0 || max_passes < 1 || decreasing_limit < 0 || max_evaluations < 0
      || max_buffers < 0
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
  int committed_buffers = 0;
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
        || (max_evaluations > 0 && evaluations >= max_evaluations)
        || (max_buffers > 0 && committed_buffers >= max_buffers);
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
          committed_buffers = countInsertedBuffers(actions, best_action_count);
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
  if (max_buffers > 0 && committed_buffers >= max_buffers) stop = "max_buffers";
  else if (attempts >= max_attempts) stop = "max_attempts";
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
  result["inserted_buffer_count"] = committed_buffers;
  result["max_buffers"] = max_buffers;
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
