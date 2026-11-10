// SPDX-License-Identifier: BSD-3-Clause
// Native queue dispatch and action identities share one owned projection.
#include "rebuffer_timing_planning.hh"
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_mutation.hh"
#include "rebuffer_overlay_actions.hh"

#include <algorithm>
#include <set>

#include "pybind11/stl.h"

namespace py = pybind11;
namespace rebuffer {

void NativeTimingPlanningSnapshot::buildQueue()
{
  queue_ = queue_metadata_["_native_queue_state"].cast<std::shared_ptr<NativeTimingQueueSnapshot>>();
  projectTimingQueueItems(queue_->items, pin_by_graph_, net_by_graph_);
  const auto selected = selectTimingQueueIndices(queue_->items, config_);
  std::map<int, std::vector<int>> focus_by_root;
  for (const int index : selected) {
    const auto& item = queue_->items[index];
    const auto& focus = reachableNets(rootNet(item.root_net_id));
    selected_item_.emplace(std::make_pair(item.endpoint_pin, item.root_net_id), index);
    focus_by_root.emplace(item.root_net_id, focus);
    payload_rows_ += focus.size();
  }
  // The existing focus updater also updates every endpoint sharing the same
  // tree key. Grouping by root preserves that policy without repeated scans.
  for (auto& item : queue_->items) {
    const auto found = focus_by_root.find(item.root_net_id);
    if (found != focus_by_root.end()) item.focus_net_ids = found->second;
  }
  queue_metadata_["planner_payload_item_count"] = selected.size();
  queue_metadata_["planner_payload_rows"] = payload_rows_;
  queue_metadata_["timing_source"] = "native_graph_planning";
  queue_metadata_["local_snapshot_enabled"] = false;
}

py::dict NativeTimingPlanningSnapshot::queueState()
{
  auto result = queue_metadata_.attr("copy")().cast<py::dict>();
  result["_native_planning_state"] = shared_from_this();
  return result;
}

std::vector<int> NativeTimingPlanningSnapshot::sourceNets(const py::list& actions) const
{
  requireCurrent();
  const auto& m = *graph_->mutation_state_;
  std::set<int> nets;
  for (const auto& value : actions) {
    const auto action = value.cast<py::dict>();
    if (action.contains("action_kind")
        && action["action_kind"].cast<std::string>() == "resize_cell") {
      const int owner = action["node_id"].cast<int>();
      if (owner < 0 || owner >= m.original_node_count) {
        throw py::value_error("resize node is outside original nodes");
      }
      for (const int pin : m.node_pins[owner]) {
        const int net = m.pin_net[pin];
        if (net < 0) continue;
        const int source = sourceNetForMutationNet(m, net);
        if (source < 0) throw py::value_error("resize input has no source net provenance");
        nets.insert(source);
      }
      continue;
    }
    const int partition = action["net_id"].cast<int>();
    // Every active partition of the imported complete tree participates in
    // source projection, including original nets behind existing buffers.
    const int root = rootNet(partition, false);
    std::vector<int> pending{root};
    std::set<int> visited{root};
    while (!pending.empty()) {
      const int net = pending.back(); pending.pop_back();
      nets.insert(projection_.actions.original_net_by_net.at(net));
      const auto found = outputs_by_input_.find(net);
      if (found == outputs_by_input_.end()) continue;
      for (const int child : found->second) {
        if (visited.insert(child).second) pending.push_back(child);
      }
    }
  }
  return {nets.begin(), nets.end()};
}

py::list NativeTimingPlanningSnapshot::projectActions(const py::list& actions) const
{
  requireCurrent();
  const auto& m = *graph_->mutation_state_;
  py::list resize_projected;
  py::list buffer_actions;
  for (const auto& value : actions) {
    const auto row = value.cast<py::dict>();
    if (row.contains("action_kind")
        && row["action_kind"].cast<std::string>() == "resize_cell") {
      const int net = row["net_id"].cast<int>();
      const int owner = row["node_id"].cast<int>();
      const int target_cell = row["target_cell_index"].cast<int>();
      if (owner < 0 || owner >= m.original_node_count) {
        throw py::value_error("resize node is outside original nodes");
      }
      const int prev_cell = m.node_cells.at(owner);
      const int source_net = projection_.actions.original_net_by_net.at(net);
      py::dict projected = row.attr("copy")().cast<py::dict>();
      projected["action_kind"] = "resize_cell";
      projected["net_id"] = source_net;
      projected["net_name"] = m.net_names.at(source_net);
      projected["node_id"] = owner;
      projected["instance_name"] = m.node_names.at(owner);
      projected["previous_cell_index"] = prev_cell;
      projected["previous_master_name"] = m.cell_names.at(prev_cell);
      projected["target_cell_index"] = target_cell;
      projected["target_master_name"] = m.cell_names.at(target_cell);
      resize_projected.append(std::move(projected));
    } else {
      buffer_actions.append(row);
    }
  }
  if (py::len(buffer_actions) == 0) return resize_projected;
  std::vector<std::shared_ptr<IncrementalTopologyState>> states;
  for (const auto& [_, state] : graph_->net_states_) states.push_back(state);
  py::list projected_buffers = projectPlanningActions(
      inputs_->carrier.identity, projection_, *liberty_, states, buffer_actions);
  if (py::len(resize_projected) == 0) return projected_buffers;
  for (const auto& item : projected_buffers) resize_projected.append(item);
  return resize_projected;
}

py::list projectPlanningActions(const OverlayActionInputs& base,
    const OverlayCarrierProjection& projection, const LibertyView& liberty,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const py::list& actions)
{
  py::list mapped;
  int order = 0;
  for (const auto& value : actions) {
    const auto raw = value.cast<py::dict>();
    auto row = raw.attr("copy")().cast<py::dict>();
    const int cell = raw["buffer_cell_index"].cast<int>();
    row["action_kind"] = "buffer_insert";
    row["net_name"] = projection.net_names.at(raw["net_id"].cast<int>());
    row["buffer_master_name"] = liberty.cells.at(cell).name;
    row["candidate_location_x_dbu"] = py::int_(raw["x_dbu"]);
    row["candidate_location_y_dbu"] = py::int_(raw["y_dbu"]);
    row["commit_order"] = order++;
    row["parent_node_id"] = raw["source_parent_node_id"];
    row["child_node_id"] = raw["source_child_node_id"];
    row["tree_candidate_parent_graph_id"] = raw["parent_node_id"];
    row["tree_candidate_child_graph_id"] = raw["child_node_id"];
    // The shared projector supplies final physical names from the immutable
    // base PyPlaceDB and terminal closures, so no Python carrier is needed.
    mapped.append(std::move(row));
  }
  return projectOverlayActionsFromProjection(base, projection.actions, states, mapped);
}

py::dict NativeTimingPlanningSnapshot::summary() const
{
  py::dict result;
  result["status"] = "ok";
  result["epoch"] = checkpoint_->epoch;
  result["pin_count"] = projection_.design.num_pins;
  result["net_count"] = projection_.design.num_nets;
  result["queue_item_count"] = queue_->items.size();
  result["planner_payload_rows"] = payload_rows_;
  result["planned_requests"] = planned_requests_;
  result["imported_tree_count"] = trees_.size();
  result["phase_timings_ms"] = phase_timings_ms_;
  return result;
}

void registerTimingPlanningBindings(py::module_& module)
{
  py::class_<NativeTimingPlanningSnapshot, std::shared_ptr<NativeTimingPlanningSnapshot>>(
      module, "NativeTimingPlanningSnapshot")
      .def("queue_state", &NativeTimingPlanningSnapshot::queueState)
      .def("plan", &NativeTimingPlanningSnapshot::plan)
      .def("source_nets", &NativeTimingPlanningSnapshot::sourceNets)
      .def("project_actions", &NativeTimingPlanningSnapshot::projectActions)
      .def("current_tree", &NativeTimingPlanningSnapshot::currentTree)
      .def("timing_view", &NativeTimingPlanningSnapshot::timingView)
      .def("summary", &NativeTimingPlanningSnapshot::summary);
}

}  // namespace rebuffer
