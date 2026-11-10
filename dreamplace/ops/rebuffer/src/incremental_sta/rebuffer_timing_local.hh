// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Local adjacency, DAG ranks and per-pin event rows share one reversible delta.

#include <map>
#include <memory>
#include <vector>

#include "rebuffer_timing_mutation.hh"


namespace rebuffer {
struct TimingMutationState;
struct TimingTopologyUndo;

struct TimingIndexPinUndo
{
  int rank, path_level, net_arc;
  std::vector<int> successors, cell_out, cell_in, net_out, net_in;
  std::vector<TimingEventOperation> forward_inputs, reverse_outputs;
  std::vector<int> forward_successors, reverse_predecessors;
};

struct TimingIndexCellUndo
{
  int id;
  TimingCellArc row;
  unsigned char active;
};

struct TimingIndexNetUndo
{
  int id;
  TimingNetArc row;
  unsigned char active;
};

struct TimingIndexUndo
{
  std::size_t pins, cells, nets;
  bool canonical_schedule;
  std::map<int, TimingIndexPinUndo> rows;
  std::vector<TimingIndexCellUndo> cell_rows;
  std::vector<TimingIndexNetUndo> net_rows;
};

std::shared_ptr<TimingIndexUndo> captureTimingIndex(
    const TimingMutationState& state, const TimingGraphIndex& index);
void captureTimingIndexPin(TimingIndexUndo& undo, const TimingMutationState& state,
                            const TimingGraphIndex& index, int pin);
void initializeTimingLocalIndex(TimingMutationState& state, TimingGraphIndex& index,
                                 const TimingPropagationProgram& program);
void updateTimingLocalIndex(TimingMutationState& state, TimingGraphIndex& index,
    const TimingPropagationProgram& program, const TimingTopologyUndo& topology_undo);
void restoreTimingLocalIndex(TimingMutationState& state, TimingGraphIndex& index,
                              TimingIndexUndo& undo);
void updateTimingEventRows(TimingMutationState& state, const TimingGraphIndex& index,
    const TimingPropagationProgram& program, const std::vector<int>& pins,
    TimingIndexUndo& undo);
}  // namespace rebuffer

// --- rebuffer_timing_local.hh ---
// A selected-tree snapshot retains canonical identities and fills only its domain.

#include "rebuffer_timing_planning.hh"

#include "rebuffer_driver_tree.hh"

#include <set>
#include <unordered_map>

namespace rebuffer {

inline void requireLocal(bool valid, const std::string& message)
{
  if (!valid) throw pybind11::value_error(message);
}

struct CachedLocalPlanningIdentity
{
  std::uint64_t topology_revision = 0;
  std::uint64_t net_states_revision = 0;
  std::vector<std::shared_ptr<IncrementalTopologyState>> net_state_list;
  mutable std::vector<int> net_state_epochs;
  std::vector<TimingBufferRecord> buffers;
  std::vector<std::array<double, 2>> buffer_coordinates;
  std::vector<int> buffer_index_by_node;
  std::vector<int> pin_to_canonical, canonical_to_pin;
  std::vector<int> node_to_canonical;
  std::vector<int> net_to_canonical, canonical_to_net, original_by_net;
  std::vector<std::pair<std::pair<int, int>, int>> virtual_pins;
  std::vector<std::pair<int, int>> source_by_pin;
  std::vector<CurrentDriverBufferLink> links;
  std::vector<int> link_by_output;
  int pin_count = 0, node_count = 0, net_count = 0, vertex_count = 0;
};

struct LocalPlanningView
{
  OverlayCarrierProjection projection;
  TimingView timing;
  std::shared_ptr<CurrentDriverTree> tree;
  std::set<int> pins;
  std::set<int> nets;
  std::vector<int> touched_vertices;
  std::vector<int> touched_pins;
  std::vector<int> touched_nodes;
  std::uint64_t cached_topology_revision = 0;
  bool static_design_initialized = false;
};

class NativeTimingLocalPlanningSnapshot
    : public std::enable_shared_from_this<NativeTimingLocalPlanningSnapshot>
{
 public:
  ~NativeTimingLocalPlanningSnapshot();
  pybind11::dict queueState();
  pybind11::dict plan(const pybind11::dict& request, const pybind11::dict& config);
  std::vector<int> sourceNets(const pybind11::list& actions) const;
  pybind11::list projectActions(const pybind11::list& actions) const;
  std::shared_ptr<CurrentDriverTree> currentTree(int net);
  pybind11::dict summary() const;

 private:
  friend class PersistentTimingGraph;
  NativeTimingLocalPlanningSnapshot(std::shared_ptr<PersistentTimingGraph> graph,
      std::shared_ptr<const TimingPlanningInputs> inputs,
      const pybind11::list& actions, const pybind11::dict& config);
  void requireCurrent() const;
  int pinId(int native) const;
  int nodeId(int native) const;
  int netId(int native) const;
  int graphNet(int canonical) const;
  int originalNet(int canonical) const;
  int vertex(int original_net, int source_node) const;
  std::vector<int> reachable(int root) const;
  std::shared_ptr<LocalPlanningView> view(int root);
  void buildTopology(LocalPlanningView& out);
  void buildViews(LocalPlanningView& out);

  std::shared_ptr<PersistentTimingGraph> graph_;
  std::shared_ptr<PersistentTimingGraphCheckpoint> checkpoint_;
  std::shared_ptr<const TimingPlanningInputs> inputs_;
  std::shared_ptr<CachedLocalPlanningIdentity> identity_;
  pybind11::dict config_, queue_metadata_;
  std::shared_ptr<NativeTimingQueueSnapshot> queue_;
  std::unordered_map<int, std::shared_ptr<LocalPlanningView>> views_;
  std::unordered_map<int, std::shared_ptr<CurrentDriverTree>> trees_;
  std::shared_ptr<LocalPlanningView> last_view_;
  int last_view_root_ = -1;
  std::vector<double> endpoint_slacks_;
  bool shared_view_data_ready_ = false;
  std::map<std::string, double> timings_;
  int pin_count_ = 0, node_count_ = 0, net_count_ = 0, vertex_count_ = 0;
  int planned_ = 0, filled_pins_ = 0, filled_vertices_ = 0;
};

void resetLocalPlanningView(LocalPlanningView& v);
void registerTimingLocalBindings(pybind11::module_& module);

}  // namespace rebuffer
