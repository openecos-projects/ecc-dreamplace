// SPDX-License-Identifier: BSD-3-Clause
// A selected-tree snapshot retains canonical identities and fills only its domain.
#pragma once

#include "rebuffer_timing_planning.hh"
#include "rebuffer_driver_tree_links.hh"
#include "rebuffer_timing_mutation.hh"
#include <set>

namespace rebuffer {

inline void requireLocal(bool valid, const std::string& message)
{
  if (!valid) throw pybind11::value_error(message);
}

struct LocalPlanningView
{
  OverlayCarrierProjection projection;
  TimingView timing;
  std::shared_ptr<CurrentDriverTree> tree;
  std::set<int> pins;
  std::set<int> nets;
};

class NativeTimingLocalPlanningSnapshot
    : public std::enable_shared_from_this<NativeTimingLocalPlanningSnapshot>
{
 public:
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
  pybind11::dict config_, queue_metadata_;
  std::shared_ptr<NativeTimingQueueSnapshot> queue_;
  std::vector<TimingBufferRecord> buffers_;
  std::map<int, int> pin_to_canonical_, canonical_to_pin_;
  std::map<int, int> node_to_canonical_, canonical_to_node_;
  std::map<int, int> net_to_canonical_, canonical_to_net_, original_by_net_;
  std::map<std::pair<int, int>, int> virtual_pins_;
  std::map<int, std::pair<int, int>> source_by_pin_;
  std::vector<CurrentDriverBufferLink> links_;
  std::map<int, int> input_by_output_;
  std::map<int, std::vector<int>> outputs_by_input_;
  std::map<int, std::shared_ptr<LocalPlanningView>> views_;
  std::shared_ptr<LocalPlanningView> last_view_;
  std::map<std::string, double> timings_;
  int pin_count_ = 0, node_count_ = 0, net_count_ = 0, vertex_count_ = 0;
  int planned_ = 0, filled_pins_ = 0, filled_vertices_ = 0;
};

void registerTimingLocalBindings(pybind11::module_& module);

}  // namespace rebuffer
