// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <string>
#include <vector>

#include "pybind11/pybind11.h"


namespace rebuffer {

struct TimingLibraryArc
{
  std::string source, sink;
  int sense, trigger;
};

inline bool sameTimingSignature(const TimingLibraryArc& a, const TimingLibraryArc& b)
{
  return a.source == b.source && a.sink == b.sink
         && a.sense == b.sense && a.trigger == b.trigger;
}

// Repeated state-dependent arcs retain their occurrence rank across masters.
// Lookup may read an imported library or the carrier's existing lazy cache.
template <typename Lookup>
int equivalentTimingArc(int begin, int end, int old_lut, int old_offset, Lookup lookup)
{
  if (old_lut >= begin && old_lut < end) return old_lut;
  const auto previous = lookup(old_lut);
  int rank = 0;
  if (old_offset >= 0) {
    for (int lut = old_lut - old_offset; lut < old_lut; ++lut) {
      if (sameTimingSignature(lookup(lut), previous)) ++rank;
    }
  }
  std::vector<int> matches;
  for (int lut = begin; lut < end; ++lut) {
    if (sameTimingSignature(lookup(lut), previous)) matches.push_back(lut);
  }
  if (matches.empty()) {
    throw pybind11::value_error("current master does not contain the timing arc");
  }
  if (matches.size() > 1 && old_offset < 0) {
    throw pybind11::value_error("ambiguous equivalent master timing arc");
  }
  if (rank >= static_cast<int>(matches.size())) {
    throw pybind11::value_error("current master has fewer repeated timing arcs");
  }
  return matches[rank];
}

}  // namespace rebuffer

// --- rebuffer_incremental_timing.hh ---
#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "pybind11/numpy.h"


namespace rebuffer {

using TimingCellArc = std::array<int, 8>;
using TimingNetArc = std::array<int, 2>;
using TimingArcIdentity = std::array<int, 6>;

struct TimingArcHash
{
  std::size_t operator()(const TimingArcIdentity& value) const;
};

// The index belongs to one immutable carrier snapshot, never to a mutable IDB.
struct TimingGraphIndex
{
  std::vector<std::string> pins;
  std::vector<TimingCellArc> cells;
  // Empty means all slots are active (the immutable carrier representation).
  std::vector<unsigned char> cell_active;
  std::vector<TimingNetArc> nets;
  // Empty means all edges are active; mutable owners retain retired net slots.
  std::vector<unsigned char> net_active;
  std::vector<int> clocks;
  int clock_end = 0;
  std::unordered_map<std::string, int> pin_ids;
  std::unordered_map<TimingArcIdentity, int, TimingArcHash> arc_ids;

  // Immutable topology structure shared by every affected propagation on
  // this carrier.  Seed masks, endpoint filtering, and worklists remain
  // per-call state in affectedTimingSchedule().
  std::vector<int> rank;
  // Carrier flat-arc order counts cell edges, while net edges have weight 0.
  std::vector<int> path_level;
  std::vector<std::vector<int>> successors;
  std::vector<std::vector<int>> cell_out;
  std::vector<std::vector<int>> cell_in;
  std::vector<std::vector<int>> net_out;
  std::vector<std::vector<int>> net_in;

  int source(int id) const;
  TimingArcIdentity key(int id) const;
  void build();
};

std::shared_ptr<TimingGraphIndex> timingGraphIndex(
    const pybind11::object& db, bool* cache_hit = nullptr);
std::shared_ptr<TimingGraphIndex> makeTimingGraphIndex(
    const std::vector<std::string>& pins, const std::vector<TimingCellArc>& cells,
    const std::vector<TimingNetArc>& nets, const std::vector<int>& clocks, int clock_end);

pybind11::dict affectedTimingSchedule(
    const pybind11::object& previous, const pybind11::object& current,
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>& previous_rc,
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>& current_rc,
    const pybind11::object& changed_pins = pybind11::none());
}  // namespace rebuffer

// --- rebuffer_timing_payload.hh ---

namespace rebuffer {

pybind11::tuple pinPairArcMapping(const pybind11::object& db);
pybind11::dict timingPinCapacitance(const pybind11::object& db);

using TimingPayloadArray = pybind11::array_t<
    double, pybind11::array::c_style | pybind11::array::forcecast>;
pybind11::dict repairDesignElectricalPayload(
    const TimingPayloadArray& rise_load, const TimingPayloadArray& fall_load,
    const TimingPayloadArray& rise_cap, const TimingPayloadArray& fall_cap);

}  // namespace rebuffer

// --- rebuffer_timing_session.hh ---
// Native transaction boundary for placement-RC timing repair.


#include <memory>
#include <map>
#include <string>
#include <vector>


#include "rebuffer_incremental_topology.hh"


namespace rebuffer {

class PersistentTimingGraph;
class RepairTimingCampaignSession;
class PersistentTimingGraphCheckpoint;

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
  friend class RepairTimingCampaignSession;
  pybind11::dict applyCandidateImpl(const pybind11::dict& action,
      PersistentTimingGraph* timing_graph);
  std::shared_ptr<IncrementalTopologyState> topology_;
};

std::shared_ptr<RepairTimingSession> createRepairTimingSession(
    const std::shared_ptr<IncrementalTopologyState>& topology);

struct RepairTimingCampaignCheckpoint
{
  std::map<int, std::shared_ptr<IncrementalTopologyState>> topologies;
  std::map<int, std::shared_ptr<IncrementalTopologyCheckpoint>> checkpoints;
  std::map<int, std::string> action_ledger;
  std::size_t undo_size = 0;
  std::shared_ptr<PersistentTimingGraphCheckpoint> timing_graph;
  std::weak_ptr<const unsigned char> campaign_identity;
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
  explicit RepairTimingCampaignSession(
      std::shared_ptr<PersistentTimingGraph> timing_graph = nullptr);
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
  void restoreState(
      const std::shared_ptr<RepairTimingCampaignCheckpoint>& checkpoint);
  pybind11::dict restore(
      const std::shared_ptr<RepairTimingCampaignCheckpoint>& checkpoint);
  pybind11::dict summary() const;
  const std::shared_ptr<PersistentTimingGraph>& timingGraph() const
  {
    return timing_graph_;
  }

 private:
  struct CampaignUndoEntry
  {
    bool is_topology = false;
    int key = -1;
    bool existed = false;
    std::string previous_state;
  };
  void appendLedgerEntry(RepairTimingLedgerEntry entry);

  std::map<int, std::shared_ptr<IncrementalTopologyState>> topologies_;
  std::map<int, std::string> action_ledger_;
  std::vector<CampaignUndoEntry> undo_journal_;
  std::vector<RepairTimingLedgerEntry> ledger_history_;
  int next_ledger_sequence_ = 0;
  std::shared_ptr<PersistentTimingGraph> timing_graph_;
  std::shared_ptr<const unsigned char> checkpoint_identity_
      = std::make_shared<const unsigned char>(0);
};

std::shared_ptr<RepairTimingCampaignSession>
createRepairTimingCampaignSession(
    const std::shared_ptr<PersistentTimingGraph>& timing_graph = nullptr);

}  // namespace rebuffer
