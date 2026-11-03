// SPDX-License-Identifier: BSD-3-Clause
// Persistent native timing-graph handle, stage 3a of the incremental TP plan.

#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

#include "rebuffer_incremental_patch.hh"
#include "rebuffer_incremental_topology.hh"
#include "rebuffer_incremental_timing.hh"

namespace rebuffer {

struct TimingPropagationProgram;
struct TimingMutationState;
struct TimingResizeUndo;
struct TimingTopologyUndo;
struct TimingEventSeeds;
struct TimingStateUndo;
struct TimingNetRcValues;
struct TimingStateViewStorage;
struct TimingSetupPaths;
struct LibertyView;
struct TimingPlanningInputs;
class NativeTimingPlanningSnapshot;
enum class TimingTopologyOperation;

struct TimingPropagationMark
{
  bool ready = false;
  std::uint64_t sequence = 0;
};

class PersistentTimingGraphCheckpoint
{
 public:
  std::size_t journal_size = 0;
  std::uint64_t epoch = 0;
  bool timing_state_initialized = false;
  std::weak_ptr<const unsigned char> graph_identity;
  std::uint64_t journal_generation = 0;
  std::uint64_t journal_head_sequence = 0;
  std::map<int, std::shared_ptr<IncrementalTopologyCheckpoint>>
      topology_checkpoints;
};

// Owns the carrier index, attached net topologies and imported Python TP state.
// Canonical pin/arc state has a strict schema; per-net RC scratch fields retain
// the generic journal API. AAT/RAT propagation is added in a separate module.
class PersistentTimingGraph : public std::enable_shared_from_this<PersistentTimingGraph>
{
 public:
  explicit PersistentTimingGraph(const pybind11::object& carrier);
  PersistentTimingGraph(const PersistentTimingGraph&) = delete;
  PersistentTimingGraph& operator=(const PersistentTimingGraph&) = delete;

  pybind11::dict summary() const;
  std::uint64_t epoch() const { return epoch_; }
  pybind11::dict stateView(bool copy = true) const;
  pybind11::dict stateViewStatistics() const;
  pybind11::dict setState(const pybind11::dict& values);
  pybind11::dict timingStateSchema() const;
  pybind11::dict initializeTimingState(const pybind11::dict& values);
  pybind11::dict initializePropagation(const pybind11::dict& values);
  pybind11::dict propagate();
  pybind11::dict propagateIncremental();
  pybind11::dict setupPathDetails(const pybind11::object& endpoint_pins,
      int64_t global_k, int64_t max_depth, double residual_tolerance_ps) const;
  pybind11::dict initializeQueue(const pybind11::dict& liberty_values,
      const std::vector<std::uint8_t>& original_node_is_buffer);
  pybind11::dict timingQueueState(const std::vector<int>& original_net_by_net,
      bool use_path_context = false) const;
  void applyQueueGradientScores(pybind11::dict& snapshot, const pybind11::dict& config) const;
  pybind11::dict timingQueueStateForCarrier(
      const std::vector<int>& original_net_by_net,
      const pybind11::object& carrier, bool use_path_context = false) const;
  pybind11::dict evaluationObservation(int max_fanout, double cap_margin, double slew_margin) const;
  pybind11::dict initializePlanning(const pybind11::dict& design,
      const pybind11::dict& characterization, const pybind11::dict& steiner,
      const pybind11::dict& config, double r_unit, double c_unit);
  std::shared_ptr<NativeTimingPlanningSnapshot> planningSnapshot(
      const pybind11::list& actions, const pybind11::dict& config);
  pybind11::dict initializeMutations();
  void requireTimingMutationOwner() const;
  pybind11::dict mutationView() const;
  pybind11::dict mutationStatistics() const;
  pybind11::dict mutationIndexOracle() const;
  pybind11::dict compact();
  pybind11::dict resizeCell(const std::string& instance_name,
                           const std::string& master_name);

  pybind11::dict attachNetTopology(
      int net_id,
      int driver_pin_id,
      int next_node_id,
      const pybind11::dict& topology_values);
  pybind11::dict attachNetTopologyState(
      int net_id,
      const std::shared_ptr<IncrementalTopologyState>& state);
  pybind11::dict clearNetTopologies();
  pybind11::dict applyBufferSplit(int net_id, const pybind11::dict& action);
  pybind11::dict removeBuffer(int net_id, int action_id);
  pybind11::dict resizeBuffer(int net_id, const pybind11::dict& action);
  pybind11::dict topologyView(int net_id) const;
  pybind11::dict relevelizeNet(int net_id) const;
  pybind11::dict compactNetTopology(int net_id);
  pybind11::dict criticalPathSubset(
      const pybind11::dict& path_details,
      const pybind11::object& endpoint_pins) const;
  pybind11::list finalizeActions(const pybind11::list& actions) const;
  pybind11::dict buildNetRc(int net_id) const;
  pybind11::dict buildTimingPatch(
      int net_id,
      const pybind11::dict& action,
      int base_vertex_count,
      int slot) const;
  pybind11::dict exportCarrier(
      const pybind11::dict& steiner,
      const pybind11::list& actions,
      double r_unit,
      double c_unit) const;
  pybind11::dict evaluateNetRc(
      int net_id,
      const pybind11::dict& rc,
      const pybind11::array_t<double,
                              pybind11::array::c_style
                                  | pybind11::array::forcecast>& generic_caps,
      const pybind11::array_t<double,
                              pybind11::array::c_style
                                  | pybind11::array::forcecast>& rise_caps,
      const pybind11::array_t<double,
                              pybind11::array::c_style
                                  | pybind11::array::forcecast>& fall_caps,
      const pybind11::dict& buffer_caps,
      int pin_count);

  std::shared_ptr<PersistentTimingGraphCheckpoint> checkpoint() const;
  pybind11::dict restore(
      const std::shared_ptr<PersistentTimingGraphCheckpoint>& checkpoint);
  pybind11::dict commit(
      const std::shared_ptr<PersistentTimingGraphCheckpoint>& checkpoint);

  const std::shared_ptr<TimingGraphIndex>& timingIndex() const
  {
    return timing_index_;
  }

 private:
  friend class NativeTimingPlanningSnapshot;
  struct JournalEntry
  {
    std::string name;
    bool existed = false;
    std::uint64_t previous_epoch = 0;
    std::uint64_t sequence = 0;
    std::vector<double> previous;
    std::shared_ptr<TimingResizeUndo> resize_undo;
    std::shared_ptr<TimingTopologyUndo> topology_undo;
    std::shared_ptr<TimingStateUndo> state_undo;
    std::optional<TimingPropagationMark> propagation_undo;
  };

  static std::vector<double> readState(const pybind11::handle& value,
                                       const char* name);
  static pybind11::array_t<double> writeState(
      const std::vector<double>& value);
  pybind11::dict applyStateValues(
      std::map<std::string, std::vector<double>> values);
  pybind11::dict finishPropagation(
      std::map<std::string, std::vector<double>> values,
      pybind11::dict coverage, const char* mode);
  TimingEventSeeds timingEventSeeds() const;
  TimingSetupPaths nativeSetupPaths(
      const std::optional<std::vector<int64_t>>& endpoint_pins,
      int64_t global_k, int64_t max_depth, double residual_tolerance_ps) const;
  TimingNetRcValues recomputeTimingNetRc(const IncrementalTopologyState& topology,
      const std::array<std::vector<double>, 5>& pin_values);
  std::shared_ptr<TimingStateUndo> beginStateUpdate();
  void detachStateViews();
  void applyTimingRc(const std::vector<TimingNetRcValues>& changes,
                     const std::vector<int>& inactive_pins = {});
  pybind11::dict finishSparsePropagation(const std::shared_ptr<TimingStateUndo>& undo,
                                         pybind11::dict coverage,
                                         const std::vector<double>& previous_slacks,
                                         std::vector<int64_t> changed_pins);
  std::vector<double> endpointSlacks() const;
  std::map<std::string, std::vector<double>> remappedTimingState(
      const std::vector<int>& pins, const std::vector<int>& arcs) const;
  pybind11::dict mutateTimingTopology(int net_id, const pybind11::dict& action,
                                      TimingTopologyOperation operation);
  void validateCheckpoint(
      const std::shared_ptr<PersistentTimingGraphCheckpoint>& checkpoint) const;
  void restoreTo(std::size_t journal_size);
  std::shared_ptr<IncrementalTopologyState> topologyState(int net_id) const;

  pybind11::object carrier_;
  std::shared_ptr<TimingGraphIndex> timing_index_;
  std::map<std::string, std::vector<double>> state_;
  mutable std::map<std::string, std::weak_ptr<TimingStateViewStorage>> state_views_;
  struct StateViewCounters {
    std::uint64_t exports = 0, detached_fields = 0, detached_values = 0;
  };
  mutable StateViewCounters state_view_counters_;
  std::vector<JournalEntry> journal_;
  std::map<int, std::shared_ptr<IncrementalTopologyState>> net_states_;
  std::uint64_t epoch_ = 0;
  bool timing_state_initialized_ = false;
  std::shared_ptr<const TimingPropagationProgram> propagation_program_;
  std::shared_ptr<TimingMutationState> mutation_state_;
  std::shared_ptr<const LibertyView> queue_liberty_;
  std::shared_ptr<const TimingPlanningInputs> planning_inputs_;
  std::vector<std::uint8_t> queue_original_buffer_nodes_;
  TimingPropagationMark propagation_mark_;
  std::shared_ptr<const unsigned char> checkpoint_identity_
      = std::make_shared<const unsigned char>(0);
  std::uint64_t journal_generation_ = 0;
  std::uint64_t next_journal_sequence_ = 1;
};

pybind11::dict persistentTimingStateSchema(
    std::size_t pin_count, std::size_t cell_arc_count);

std::shared_ptr<PersistentTimingGraph> createPersistentTimingGraph(
    const pybind11::object& carrier);

}  // namespace rebuffer
