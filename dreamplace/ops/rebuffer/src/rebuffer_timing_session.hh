// SPDX-License-Identifier: BSD-3-Clause
// Native transaction boundary for placement-RC timing repair.

#pragma once

#include <memory>
#include <map>
#include <string>
#include <vector>

#include "pybind11/pybind11.h"

#include "rebuffer_incremental_topology.hh"

namespace rebuffer {

struct RepairTimingSessionCheckpoint
{
  std::shared_ptr<IncrementalTopologyCheckpoint> topology;
};

class RepairTimingSession
{
 public:
  explicit RepairTimingSession(
      std::shared_ptr<IncrementalTopologyState> topology);

  pybind11::dict applyCandidate(const pybind11::dict& action);
  pybind11::dict removeCandidate(int action_id);
  std::shared_ptr<RepairTimingSessionCheckpoint> checkpoint() const;
  pybind11::dict restore(
      const std::shared_ptr<RepairTimingSessionCheckpoint>& checkpoint);
  pybind11::dict summary() const;

  const std::shared_ptr<IncrementalTopologyState>& topology() const
  {
    return topology_;
  }

 private:
  std::shared_ptr<IncrementalTopologyState> topology_;
};

std::shared_ptr<RepairTimingSession> createRepairTimingSession(
    const std::shared_ptr<IncrementalTopologyState>& topology);

struct RepairTimingCampaignCheckpoint
{
  std::map<int, std::shared_ptr<IncrementalTopologyState>> topologies;
  std::map<int, std::shared_ptr<IncrementalTopologyCheckpoint>> checkpoints;
  std::map<int, std::string> action_ledger;
};

// Candidate and action lifecycle records are kept in the native campaign
// object so the Python TimingPropagation callback and the physical topology
// projection can be audited against one state machine.  The history is
// append-only for diagnostics; checkpoint/restore controls the active
// action_ledger above.
struct RepairTimingLedgerEntry
{
  int sequence = -1;
  std::string kind;
  std::string state;
  int net_id = -1;
  int action_id = -1;
  int variant_index = -1;
  std::string status;
  double wns_ps = 0.0;
  double tns_ps = 0.0;
  double endpoint_slack_ps = 0.0;
  std::vector<int> action_ids;
};

class RepairTimingCampaignSession
{
 public:
  void attachTopology(
      int net_id, const std::shared_ptr<IncrementalTopologyState>& topology);
  pybind11::dict applyCandidate(int net_id, const pybind11::dict& action);
  pybind11::dict removeCandidate(int net_id, int action_id);
  pybind11::dict replaceCandidateTree(int net_id,
                                      const pybind11::list& actions);
  pybind11::dict recordCandidate(int net_id,
                                 const pybind11::dict& observation,
                                 const std::string& state);
  pybind11::dict recordActionState(int net_id,
                                   int action_id,
                                   const std::string& state);
  std::shared_ptr<RepairTimingCampaignCheckpoint> checkpoint() const;
  pybind11::dict restore(
      const std::shared_ptr<RepairTimingCampaignCheckpoint>& checkpoint);
  pybind11::dict summary() const;

 private:
  void appendLedgerEntry(RepairTimingLedgerEntry entry);

  std::map<int, std::shared_ptr<IncrementalTopologyState>> topologies_;
  std::map<int, std::string> action_ledger_;
  std::vector<RepairTimingLedgerEntry> ledger_history_;
  int next_ledger_sequence_ = 0;
};

std::shared_ptr<RepairTimingCampaignSession>
createRepairTimingCampaignSession();

}  // namespace rebuffer
