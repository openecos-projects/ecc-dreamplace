// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Sparse undo for graph-owned canonical pin and cell-arc arrays.

#include <array>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace rebuffer {

struct TimingStateField
{
  const char* name;
  const char* source;
  bool pin_indexed;
  const char* unit;
  double initial = 0;
};

const std::array<TimingStateField, 16>& timingStateFields();

struct TimingIndexedValue
{
  std::size_t index;
  double value;
};

struct TimingFieldUndo
{
  std::size_t previous_size;
  std::vector<TimingIndexedValue> previous;
};

struct TimingStateUndo
{
  std::map<std::string, TimingFieldUndo> fields;
  std::size_t value_count = 0;
};

void captureTimingValue(TimingStateUndo& undo, const char* field,
                         const std::vector<double>& values, std::size_t index);
void patchTimingValue(TimingStateUndo& undo, const char* field,
                       std::vector<double>& values, std::size_t index, double value);
void compactTimingUndo(TimingStateUndo& undo,
    const std::map<std::string, std::vector<double>>& state,
    bool indices_are_unique = false);
void restoreTimingState(std::map<std::string, std::vector<double>>& state,
                         const TimingStateUndo& undo);

}  // namespace rebuffer

// --- rebuffer_timing_paths.hh ---
#include <cstddef>
#include "rebuffer_incremental_timing.hh"

#include "timing_setup_paths.h"


namespace rebuffer {

struct TimingSetupPaths
{
  dreamplace::timing::SetupPathBatch batch;
  std::size_t endpoint_record_count = 0;
};

struct TimingPathPredecessors
{
  const TimingGraphIndex& index;
  bool reordered;

  template <typename Visit>
  void operator()(int64_t pin, const Visit& visit) const
  {
    auto order = [&](int id) {
      return std::pair<int64_t, int64_t>{reordered ? index.path_level[index.source(id)] : 0, id};
    };
    auto cell = [&](int id) {
      if (id >= index.clock_end) {
        visit(dreamplace::timing::SetupPathPredecessor{
            index.source(id), id, index.cells[id][4], order(id)});
      }
    };
    const auto& incoming = index.cell_in[pin];
    if (reordered && incoming.size() > 1) {
      auto sorted = incoming;
      std::sort(sorted.begin(), sorted.end(), [&](int a, int b) { return order(a) < order(b); });
      for (int id : sorted) cell(id);
    } else {
      for (int id : incoming) cell(id);
    }
    for (int id : index.net_in[pin]) {
      visit(dreamplace::timing::SetupPathPredecessor{index.nets[id][0], -1, 0, {-1, -1}});
    }
  }
};

}  // namespace rebuffer

// --- rebuffer_electrical_observation.hh ---
#include <string_view>
#include <vector>

#include "pybind11/pybind11.h"


namespace rebuffer {
struct ElectricalPinObservation
{
  std::string_view name;
  double slew_rise, slew_fall, load_rise, load_fall, max_slew, max_cap;
  int fanout;
  bool driver, clock;
};

pybind11::dict observeElectricalPins(const std::vector<ElectricalPinObservation>& pins,
                                    int max_fanout, double cap_margin, double slew_margin);
bool electricalNonRegressing(const pybind11::dict& before, const pybind11::dict& after);
pybind11::dict compareElectricalObservations(const pybind11::dict& before,
                                            const pybind11::dict& after);
void registerElectricalObservationBindings(pybind11::module_& module);
}

// --- rebuffer_timing_graph.hh ---
// Persistent native timing-graph handle, stage 3a of the incremental TP plan.


#include <cstddef>
#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "pybind11/numpy.h"


#include "rebuffer_incremental_topology.hh"


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
struct TimingEndpointState;
struct ElectricalObservationCache;
class NativeTimingPlanningSnapshot;
class NativeTimingLocalPlanningSnapshot;
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
  std::uint64_t topology_revision = 0;
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
  std::shared_ptr<TimingEndpointState> timingEndpointState() const;
  pybind11::dict timingQueueState(const std::vector<int>& original_net_by_net,
      bool use_path_context = false, std::optional<int> endpoint_pin = std::nullopt) const;
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
  std::shared_ptr<NativeTimingLocalPlanningSnapshot> localPlanningSnapshot(
      const pybind11::list& actions, const pybind11::dict& config);
  std::shared_ptr<IncrementalTopologyState> createNetTopology(int net_id) const;
  pybind11::dict initializeMutations();
  void requireTimingMutationOwner() const;
  pybind11::dict mutationView() const;
  pybind11::dict mutationStatistics() const;
  pybind11::dict mutationIndexOracle() const;
  pybind11::dict compact();
  pybind11::dict resizeCell(const std::string& instance_name,
                           const std::string& master_name);
  std::pair<int, int> findDriverResizeCandidate(
      int native_driver_pin,
      const std::vector<int>& native_path_pins,
      const pybind11::dict& config) const;

  pybind11::dict attachNetTopology(
      int net_id,
      int driver_pin_id,
      int next_node_id,
      const pybind11::dict& topology_values);
  pybind11::dict attachNetTopologyState(
      int net_id,
      const std::shared_ptr<IncrementalTopologyState>& state);
  void attachNetTopologyStateFast(
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
  void restoreState(
      const std::shared_ptr<PersistentTimingGraphCheckpoint>& checkpoint);
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
  friend class NativeTimingLocalPlanningSnapshot;
  struct JournalEntry
  {
    std::string name;
    bool existed = false;
    std::uint64_t previous_epoch = 0;
    std::uint64_t previous_topology_revision = 0;
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
      int64_t global_k, int64_t max_depth, double residual_tolerance_ps,
      bool worst_per_endpoint = false) const;
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
  void markElectricalDirtyPin(int pin) const;
  void markElectricalFullDirty() const;

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
  std::uint64_t topology_revision_ = 1;
  std::uint64_t next_topology_revision_ = 1;
  std::uint64_t net_states_revision_ = 1;
  bool timing_state_initialized_ = false;
  std::shared_ptr<const TimingPropagationProgram> propagation_program_;
  std::shared_ptr<TimingMutationState> mutation_state_;
  std::shared_ptr<const LibertyView> queue_liberty_;
  std::shared_ptr<const TimingPlanningInputs> planning_inputs_;
  std::vector<std::uint8_t> queue_original_buffer_nodes_;
  TimingPropagationMark propagation_mark_;
  mutable std::vector<double> cached_endpoint_slacks_;
  mutable bool cached_endpoint_slacks_valid_ = false;
  mutable pybind11::dict cached_endpoint_slacks_by_name_;
  mutable std::vector<double> cached_endpoint_slacks_by_name_values_;
  mutable std::vector<pybind11::str> cached_endpoint_pin_names_;
  mutable std::shared_ptr<ElectricalObservationCache> electrical_cache_;
  mutable std::size_t cached_active_net_arcs_ = 0;
  mutable std::size_t cached_active_pins_ = 0;
  mutable std::size_t cached_active_cell_arcs_ = 0;
  mutable bool cached_active_counts_valid_ = false;
  mutable std::size_t cached_journal_full_values_ = 0;
  mutable std::size_t cached_journal_sparse_values_ = 0;
  mutable std::size_t cached_journal_stats_size_ = 0;
  mutable std::uint64_t cached_journal_stats_head_seq_ = 0;
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
