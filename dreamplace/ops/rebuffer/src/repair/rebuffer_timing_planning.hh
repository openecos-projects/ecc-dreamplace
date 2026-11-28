// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Immutable C++ DP inputs for one qualified timing/placement snapshot.

#include <cstdint>
#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "rebuffer_carrier.hh"

#include "rebuffer_driver_tree.hh"

#include "rebuffer_queue_snapshot.hh"


namespace rebuffer {

class PersistentTimingGraph;
struct IncrementalTopologyState;
class PersistentTimingGraphCheckpoint;
struct CachedLocalPlanningIdentity;
struct LocalPlanningView;

struct TimingPlanningInputs
{
  DesignView design;
  SteinerView steiner;
  CharacterizationResult characterization;
  OverlayCarrierInputs carrier;
  pybind11::dict config;
  double r_unit, c_unit;
  mutable std::shared_ptr<const std::vector<CurrentDriverBufferLink>> base_links;
  // Active session buffers are owned by the persistent mutation graph.  Cache
  // their native links per graph revision so a local planning snapshot does
  // not rescan the complete projected design to rediscover them.
  mutable std::shared_ptr<const std::vector<CurrentDriverBufferLink>> active_links;
  mutable std::uint64_t active_links_epoch = 0;
  mutable std::size_t active_links_journal_size = 0;
  mutable std::uint64_t active_links_journal_head = 0;
  mutable bool active_links_valid = false;
  mutable std::shared_ptr<CachedLocalPlanningIdentity> cached_local_identity;
  mutable std::vector<int> metric_row_by_pin;
  mutable std::vector<int> clock_order_by_pin;
  mutable bool clock_pins_unique = true;
  mutable bool static_pin_lookup_ready = false;
  mutable std::vector<std::shared_ptr<LocalPlanningView>> local_view_pool;
};

// References are borrowed only for the duration of the shared native planner.
struct NativeTreePlannerViews
{
  const DesignView& design;
  const TimingView& timing;
  const LibertyView& liberty;
  const CharacterizationResult& characterization;
  const pybind11::dict& config;
  std::shared_ptr<CurrentDriverTree> tree;
  std::string net_name;
  int focus_net_id = -1;
  int candidate_variant = 0;
  std::map<std::string, double>* phase_timings = nullptr;
  int driver_resize_node_id = -1;
  int driver_resize_cell_index = -1;
};

class NativeTimingPlanningSnapshot
    : public std::enable_shared_from_this<NativeTimingPlanningSnapshot>
{
 public:
  pybind11::dict queueState();
  pybind11::dict plan(const pybind11::dict& request, const pybind11::dict& config);
  std::vector<int> sourceNets(const pybind11::list& actions) const;
  pybind11::list projectActions(const pybind11::list& actions) const;
  std::shared_ptr<CurrentDriverTree> currentTree(int net);
  pybind11::dict summary() const;
  pybind11::dict timingView() const;
  void requireCurrent() const;

 private:
  friend class PersistentTimingGraph;
  NativeTimingPlanningSnapshot(std::shared_ptr<PersistentTimingGraph> graph,
      std::shared_ptr<const TimingPlanningInputs> inputs,
      const pybind11::list& actions, const pybind11::dict& config);
  void buildViews();
  void buildIdentity();
  void buildQueue();
  void buildBufferLinks();
  int rootNet(int net, bool use_driver_scope = true) const;
  const std::vector<int>& reachableNets(int root);

  std::shared_ptr<PersistentTimingGraph> graph_;
  std::shared_ptr<PersistentTimingGraphCheckpoint> checkpoint_;
  std::shared_ptr<const TimingPlanningInputs> inputs_;
  std::shared_ptr<const LibertyView> liberty_;
  OverlayCarrierProjection projection_;
  TimingView timing_;
  pybind11::dict config_, queue_metadata_;
  std::shared_ptr<NativeTimingQueueSnapshot> queue_;
  std::vector<int> pin_by_graph_, net_by_graph_;
  std::vector<int> graph_pin_by_pin_, graph_node_by_node_;
  std::map<int, int> input_net_by_output_;
  std::map<int, std::vector<int>> outputs_by_input_, reachable_;
  std::vector<CurrentDriverBufferLink> buffer_links_;
  std::map<std::pair<int, int>, int> selected_item_;
  std::map<int, std::shared_ptr<CurrentDriverTree>> trees_;
  int payload_rows_ = 0, planned_requests_ = 0;
  std::map<std::string, double> phase_timings_ms_;
};

void registerTimingPlanningBindings(pybind11::module_& module);
pybind11::list projectPlanningActions(const OverlayActionInputs& base,
    const OverlayCarrierProjection& projection, const LibertyView& liberty,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const pybind11::list& actions);
pybind11::dict planNativeCurrentTree(const pybind11::dict& request,
    const NativeTreePlannerViews& input, const pybind11::dict& config);

}  // namespace rebuffer

// --- rebuffer_timing_policy.hh ---
#include "pybind11/pybind11.h"


namespace rebuffer {

bool acceptsTimingCandidate(const pybind11::dict& current,
                            const pybind11::dict& candidate,
                            const pybind11::dict& request,
                            const pybind11::dict& config);

}  // namespace rebuffer

// --- rebuffer_native_planner.hh ---
// Native repair_timing candidate dispatch over Python data payloads.



namespace rebuffer {

// ``timing_snapshot`` returns both queue items and a
// per-epoch ``planner_payloads`` list.  C++ looks up the payload for each
// request and invokes the native planner directly; there is no Python
// candidate/payload callback in this path.
pybind11::dict runRepairTimingNativeSnapshotOperator(
    const pybind11::object& timing_snapshot,
    const pybind11::object& checkpoint,
    const pybind11::object& apply,
    const pybind11::object& evaluate,
    const pybind11::object& restore,
    const pybind11::dict& config,
    const pybind11::object& endpoint_state = pybind11::none(),
    const pybind11::object& release_checkpoint = pybind11::none());

}  // namespace rebuffer
