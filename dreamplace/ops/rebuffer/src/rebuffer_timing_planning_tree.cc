// SPDX-License-Identifier: BSD-3-Clause
// Current-tree and endpoint annotation reuse the existing native VGDP inputs.
#include "rebuffer_timing_planning.hh"
#include "rebuffer_driver_tree_links.hh"

#include <algorithm>
#include <set>

#include "pybind11/stl.h"

namespace py = pybind11;
namespace rebuffer {

void NativeTimingPlanningSnapshot::buildBufferLinks()
{
  std::unordered_map<int, CurrentDriverBufferLink> by_input;
  std::string reason;
  if (!collectCurrentDriverBufferLinks(projection_.design, buffer_links_, by_input, reason)) {
    throw py::value_error(reason);
  }
  for (const auto& link : buffer_links_) {
    input_net_by_output_.emplace(link.output_net_id, link.input_net_id);
    outputs_by_input_[link.input_net_id].push_back(link.output_net_id);
  }
}

int NativeTimingPlanningSnapshot::rootNet(int net, bool use_driver_scope) const
{
  if (net < 0 || net >= projection_.design.num_nets) {
    throw py::value_error("native planning root net is out of range");
  }
  if (use_driver_scope && inputs_->config.contains("native_tree_per_driver")
      && inputs_->config["native_tree_per_driver"].cast<bool>()) return net;
  std::set<int> seen;
  auto found = input_net_by_output_.find(net);
  while (found != input_net_by_output_.end()) {
    if (!seen.insert(net).second) throw py::value_error("native buffer net graph contains a cycle");
    net = found->second;
    found = input_net_by_output_.find(net);
  }
  return net;
}

const std::vector<int>& NativeTimingPlanningSnapshot::reachableNets(int root)
{
  const auto cached = reachable_.find(root);
  if (cached != reachable_.end()) return cached->second;
  std::set<int> found{root};
  std::vector<int> pending{root};
  while (!pending.empty()) {
    const int net = pending.back();
    pending.pop_back();
    const auto children = outputs_by_input_.find(net);
    if (children == outputs_by_input_.end()) continue;
    for (const int child : children->second) {
      if (found.insert(child).second) pending.push_back(child);
    }
  }
  return reachable_.emplace(root, std::vector<int>(found.begin(), found.end())).first->second;
}

std::shared_ptr<CurrentDriverTree> NativeTimingPlanningSnapshot::currentTree(int net)
{
  const int root = rootNet(net);
  const auto cached = trees_.find(root);
  if (cached != trees_.end()) return cached->second;
  TopologyView topology;
  const auto& source = projection_.steiner;
  topology.steiner_node_base = source.steiner_node_base;
  topology.pin_fa = source.pin_fa;
  topology.net_ids = reachableNets(root);
  topology.net_flat_topo_sort_start.push_back(0);
  for (const int current : topology.net_ids) {
    for (const int node : projection_.topologies.at(current)) {
      topology.net_flat_topo_sort.push_back(node);
      topology.node_x.emplace(node, source.node_x.at(node));
      topology.node_y.emplace(node, source.node_y.at(node));
    }
    topology.net_flat_topo_sort_start.push_back(topology.net_flat_topo_sort.size());
  }
  auto tree = std::make_shared<CurrentDriverTree>();
  std::string reason;
  const bool stop_at_existing_buffers = config_.contains("native_tree_stop_at_existing_buffers")
      && config_["native_tree_stop_at_existing_buffers"].cast<bool>();
  if (!importCurrentDriverTree(projection_.design, topology, root, buffer_links_,
                               *tree, reason, stop_at_existing_buffers)) {
    throw py::value_error(reason);
  }
  return trees_.emplace(root, std::move(tree)).first->second;
}

py::dict NativeTimingPlanningSnapshot::plan(const py::dict& request, const py::dict& config)
{
  requireCurrent();
  const int endpoint = request["endpoint_pin"].cast<int>();
  const int request_root = request["root_net_id"].cast<int>();
  const int focus = request["net_id"].cast<int>();
  const auto found = selected_item_.find({endpoint, request_root});
  if (found == selected_item_.end()) {
    throw py::value_error("native planning request is outside the selected snapshot");
  }
  const auto& reachable = reachableNets(rootNet(request_root));
  if (!std::binary_search(reachable.begin(), reachable.end(), focus)) {
    throw py::value_error("native planning request has an invalid focus net");
  }
  const auto& item = queue_->items.at(found->second);
  auto timing = timing_;
  // Match the current annotateLoadSlacks transport exactly. The native tree
  // scorer may use arrival/required anchoring, but the fallback field remains
  // correct for every physical load on the selected path.
  for (const int pin : item.path_pins) {
    const int net = projection_.design.pin2net_map.at(pin);
    if (net >= 0 && projection_.design.net2driver_pin_map.at(net) != pin) {
      timing.pin_slack.at(pin) = item.endpoint_slack_ps;
    }
  }
  NativeTreePlannerViews views{projection_.design, timing, *liberty_,
      inputs_->characterization, inputs_->config, currentTree(request_root),
      projection_.net_names.at(rootNet(request_root)), focus == request_root ? -1 : focus};
  views.candidate_variant = request.contains("candidate_variant")
      ? request["candidate_variant"].cast<int>() : 0;
  views.phase_timings = &phase_timings_ms_;
  ++planned_requests_;
  return planNativeCurrentTree(request, views, config);
}

}  // namespace rebuffer
