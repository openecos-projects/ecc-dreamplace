// SPDX-License-Identifier: BSD-3-Clause
// Immutable C++ DP inputs for one qualified timing/placement snapshot.
#pragma once

#include <map>
#include <memory>
#include <utility>
#include <vector>

#include "rebuffer_carrier_projection.hh"
#include "rebuffer_driver_tree.hh"
#include "rebuffer_queue_snapshot.hh"

namespace rebuffer {

class PersistentTimingGraph;
class PersistentTimingGraphCheckpoint;

struct TimingPlanningInputs
{
  DesignView design;
  SteinerView steiner;
  CharacterizationResult characterization;
  pybind11::dict config;
  double r_unit, c_unit;
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
  std::map<int, int> input_net_by_output_;
  std::map<int, std::vector<int>> outputs_by_input_, reachable_;
  std::map<std::pair<int, int>, int> selected_item_;
  std::map<int, std::shared_ptr<CurrentDriverTree>> trees_;
  int payload_rows_ = 0, planned_requests_ = 0;
};

void registerTimingPlanningBindings(pybind11::module_& module);
pybind11::dict planNativeCurrentTree(const pybind11::dict& request,
    const NativeTreePlannerViews& input, const pybind11::dict& config);

}  // namespace rebuffer
