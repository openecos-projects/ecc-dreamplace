// SPDX-License-Identifier: BSD-3-Clause

#include "rebuffer_timing_session.hh"

#include <algorithm>
#include <string>

#include "pybind11/stl.h"
#include "rebuffer_overlay_actions.hh"

namespace py = pybind11;

namespace rebuffer {

RepairTimingSession::RepairTimingSession(
    std::shared_ptr<IncrementalTopologyState> topology)
    : topology_(std::move(topology))
{
  if (topology_ == nullptr) {
    throw py::value_error("repair timing session requires topology state");
  }
}

py::dict RepairTimingSession::applyCandidate(const py::dict& action)
{
  // Lightweight fixtures and legacy callers may provide an explicit active
  // edge without coordinates.  In that contract there is nothing to resolve;
  // production placement overlays carry coordinates and use the active-edge
  // projection below.
  py::dict resolved;
  if (topology_->coordinates.empty() && action.contains("parent_node_id")
      && action.contains("child_node_id") && action.contains("split_ratio")) {
    resolved = action.attr("copy")().cast<py::dict>();
  } else {
    resolved = resolveOverlaySplit(*topology_, action);
  }
  std::string reason;
  py::dict result;
  if (!applyIncrementalBufferSplit(*topology_, resolved, reason, result)) {
    throw py::value_error(reason);
  }
  for (const char* field : {"parent_node_id",
                            "child_node_id",
                            "split_ratio",
                            "x_dbu",
                            "y_dbu",
                            "resolution_source",
                            "projection_distance_dbu"}) {
    if (resolved.contains(field)) {
      result[field] = resolved[field];
    }
  }
  return result;
}

py::dict RepairTimingSession::removeCandidate(const int action_id)
{
  std::string reason;
  py::dict result;
  if (!removeIncrementalBuffer(*topology_, action_id, reason, result)) {
    throw py::value_error(reason);
  }
  return result;
}

std::shared_ptr<RepairTimingSessionCheckpoint>
RepairTimingSession::checkpoint() const
{
  auto checkpoint = std::make_shared<RepairTimingSessionCheckpoint>();
  checkpoint->topology = checkpointIncrementalTopology(*topology_);
  return checkpoint;
}

py::dict RepairTimingSession::restore(
    const std::shared_ptr<RepairTimingSessionCheckpoint>& checkpoint)
{
  if (checkpoint == nullptr || checkpoint->topology == nullptr) {
    throw py::value_error("repair timing session checkpoint is required");
  }
  std::string reason;
  if (!restoreIncrementalTopology(*topology_, *checkpoint->topology, reason)) {
    throw py::value_error(reason);
  }
  return incrementalTopologyView(*topology_);
}

py::dict RepairTimingSession::summary() const
{
  return incrementalTopologyView(*topology_);
}

std::shared_ptr<RepairTimingSession> createRepairTimingSession(
    const std::shared_ptr<IncrementalTopologyState>& topology)
{
  return std::make_shared<RepairTimingSession>(topology);
}

void RepairTimingCampaignSession::attachTopology(
    const int net_id,
    const std::shared_ptr<IncrementalTopologyState>& topology)
{
  if (topology == nullptr) {
    throw py::value_error("repair timing campaign requires topology state");
  }
  if (topology->net_id != net_id) {
    throw py::value_error("campaign net id does not match topology state");
  }
  topologies_[net_id] = topology;
}

void RepairTimingCampaignSession::appendLedgerEntry(
    RepairTimingLedgerEntry entry)
{
  entry.sequence = next_ledger_sequence_++;
  ledger_history_.push_back(std::move(entry));
}

py::dict RepairTimingCampaignSession::recordCandidate(
    const int net_id,
    const py::dict& observation,
    const std::string& state)
{
  RepairTimingLedgerEntry entry;
  entry.kind = "candidate";
  entry.state = state;
  entry.net_id = net_id;
  if (observation.contains("variant_index")) {
    entry.variant_index = py::cast<int>(observation["variant_index"]);
  }
  if (observation.contains("status")) {
    entry.status = py::cast<std::string>(observation["status"]);
  }
  if (observation.contains("wns_ps")) {
    entry.wns_ps = py::cast<double>(observation["wns_ps"]);
  }
  if (observation.contains("tns_ps")) {
    entry.tns_ps = py::cast<double>(observation["tns_ps"]);
  }
  if (observation.contains("endpoint_slack_ps")) {
    entry.endpoint_slack_ps
        = py::cast<double>(observation["endpoint_slack_ps"]);
  }
  if (observation.contains("action_ids")) {
    entry.action_ids
        = observation["action_ids"].cast<std::vector<int>>();
  }
  appendLedgerEntry(std::move(entry));

  py::dict result;
  result["status"] = "ok";
  result["state"] = state;
  result["net_id"] = net_id;
  result["variant_index"] = entry.variant_index;
  result["sequence"] = next_ledger_sequence_ - 1;
  return result;
}

py::dict RepairTimingCampaignSession::recordActionState(
    const int net_id,
    const int action_id,
    const std::string& state)
{
  if (action_id < 0) {
    throw py::value_error("campaign action id must be nonnegative");
  }
  action_ledger_[action_id] = state;
  RepairTimingLedgerEntry entry;
  entry.kind = "action";
  entry.state = state;
  entry.net_id = net_id;
  entry.action_id = action_id;
  appendLedgerEntry(std::move(entry));

  py::dict result;
  result["status"] = "ok";
  result["state"] = state;
  result["net_id"] = net_id;
  result["action_id"] = action_id;
  result["sequence"] = next_ledger_sequence_ - 1;
  return result;
}

py::dict RepairTimingCampaignSession::applyCandidate(
    const int net_id,
    const py::dict& action)
{
  const auto found = topologies_.find(net_id);
  if (found == topologies_.end()) {
    throw py::value_error("campaign topology is not attached");
  }
  py::dict result = RepairTimingSession(found->second).applyCandidate(action);
  if (result.contains("action_id")) {
    recordActionState(
        net_id, py::cast<int>(result["action_id"]), "active");
  }
  return result;
}

py::dict RepairTimingCampaignSession::removeCandidate(
    const int net_id,
    const int action_id)
{
  const auto found = topologies_.find(net_id);
  if (found == topologies_.end()) {
    throw py::value_error("campaign topology is not attached");
  }
  py::dict result = RepairTimingSession(found->second).removeCandidate(action_id);
  recordActionState(net_id, action_id, "retired");
  return result;
}

py::dict RepairTimingCampaignSession::replaceCandidateTree(
    const int net_id,
    const py::list& actions)
{
  const auto found = topologies_.find(net_id);
  if (found == topologies_.end()) {
    throw py::value_error("campaign topology is not attached");
  }

  const IncrementalTopologyState topology_before = *found->second;
  const auto ledger_before = action_ledger_;
  std::vector<int> removed_action_ids;
  py::list applied;
  try {
    // IncrementalTopologyState intentionally removes the most recently
    // inserted buffer only. Reverse order keeps replacement atomic without
    // weakening the ordinary remove invariant.
    while (!found->second->buffers.empty()) {
      const int action_id = found->second->buffers.back().action_id;
      std::string reason;
      py::dict remove_result;
      if (!removeIncrementalBuffer(
              *found->second, action_id, reason, remove_result)) {
        throw py::value_error(reason);
      }
      removed_action_ids.push_back(action_id);
    }

    for (const auto& item : actions) {
      const py::dict action = item.cast<py::dict>();
      py::dict apply_result
          = RepairTimingSession(found->second).applyCandidate(action);
      applied.append(apply_result);
    }
  } catch (...) {
    *found->second = topology_before;
    action_ledger_ = ledger_before;
    throw;
  }

  for (const int action_id : removed_action_ids) {
    recordActionState(net_id, action_id, "replaced");
  }
  for (const auto& item : applied) {
    const py::dict apply_result = item.cast<py::dict>();
    if (apply_result.contains("action_id")) {
      recordActionState(
          net_id, py::cast<int>(apply_result["action_id"]), "active");
    }
  }

  py::dict result;
  result["status"] = "ok";
  result["net_id"] = net_id;
  result["removed_action_ids"] = removed_action_ids;
  result["applied"] = std::move(applied);
  result["buffer_count"] = found->second->buffers.size();
  result["epoch"] = found->second->epoch;
  return result;
}

std::shared_ptr<RepairTimingCampaignCheckpoint>
RepairTimingCampaignSession::checkpoint() const
{
  auto result = std::make_shared<RepairTimingCampaignCheckpoint>();
  result->topologies = topologies_;
  result->action_ledger = action_ledger_;
  for (const auto& [net_id, topology] : topologies_) {
    result->checkpoints[net_id] = checkpointIncrementalTopology(*topology);
  }
  return result;
}

py::dict RepairTimingCampaignSession::restore(
    const std::shared_ptr<RepairTimingCampaignCheckpoint>& checkpoint)
{
  if (checkpoint == nullptr) {
    throw py::value_error("repair timing campaign checkpoint is required");
  }
  std::map<int, std::shared_ptr<IncrementalTopologyState>> restored;
  for (const auto& [net_id, topology_checkpoint] : checkpoint->checkpoints) {
    const auto topology = checkpoint->topologies.at(net_id);
    std::string reason;
    if (!restoreIncrementalTopology(*topology, *topology_checkpoint, reason)) {
      throw py::value_error(reason);
    }
    restored[net_id] = topology;
  }
  topologies_ = std::move(restored);
  action_ledger_ = checkpoint->action_ledger;
  return summary();
}

py::dict RepairTimingCampaignSession::summary() const
{
  py::dict result;
  for (const auto& [net_id, topology] : topologies_) {
    result[py::int_(net_id)] = incrementalTopologyView(*topology);
  }
  py::dict ledger;
  for (const auto& [action_id, state] : action_ledger_) {
    ledger[py::int_(action_id)] = state;
  }
  result["action_ledger"] = std::move(ledger);
  py::list history;
  for (const auto& entry : ledger_history_) {
    py::dict value;
    value["sequence"] = entry.sequence;
    value["kind"] = entry.kind;
    value["state"] = entry.state;
    value["net_id"] = entry.net_id;
    value["action_id"] = entry.action_id;
    value["variant_index"] = entry.variant_index;
    value["status"] = entry.status;
    value["wns_ps"] = entry.wns_ps;
    value["tns_ps"] = entry.tns_ps;
    value["endpoint_slack_ps"] = entry.endpoint_slack_ps;
    value["action_ids"] = entry.action_ids;
    history.append(std::move(value));
  }
  result["ledger_history"] = std::move(history);
  return result;
}

std::shared_ptr<RepairTimingCampaignSession>
createRepairTimingCampaignSession()
{
  return std::make_shared<RepairTimingCampaignSession>();
}

}  // namespace rebuffer
