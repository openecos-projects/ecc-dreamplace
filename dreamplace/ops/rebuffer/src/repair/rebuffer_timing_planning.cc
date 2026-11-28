// SPDX-License-Identifier: BSD-3-Clause
// Build owned native placement/timing views without a Python TP carrier.
#include "rebuffer_timing_planning.hh"

#include "rebuffer_timing_graph.hh"

#include "rebuffer_timing_mutation.hh"

#include "rebuffer_carrier.hh"

#include "rebuffer_repair.hh"


#include <algorithm>
#include <cmath>
#include <chrono>
#include <tuple>

#include "timing_hard_math.h"

#include "pybind11/stl.h"


namespace py = pybind11;
namespace rebuffer {
namespace {
void require(bool valid, const char* message)
{
  if (!valid) throw py::value_error(message);
}

}  // namespace

py::dict PersistentTimingGraph::initializePlanning(const py::dict& design,
    const py::dict& characterization, const py::dict& steiner,
    const py::dict& config, double r_unit, double c_unit)
{
  require(!planning_inputs_, "native planning inputs are immutable");
  require(mutation_state_ && queue_liberty_, "native planning requires mutation and queue inputs");
  require(mutation_state_->buffers.empty(), "native planning initializes before buffer mutations");
  auto inputs = std::make_shared<TimingPlanningInputs>();
  inputs->design = rebuffer_python::designFromDict(design);
  inputs->characterization = rebuffer_python::characterizationFromDict(characterization);
  inputs->steiner = rebuffer_python::steinerFromDict(steiner);
  inputs->config = config.attr("copy")().cast<py::dict>();
  inputs->r_unit = r_unit;
  inputs->c_unit = c_unit;
  require(std::isfinite(r_unit) && r_unit > 0 && std::isfinite(c_unit) && c_unit >= 0,
          "native planning requires valid wire RC units");
  const auto& base = inputs->design;
  const auto& m = *mutation_state_;
  require(base.num_pins == timing_index_->pins.size()
              && base.pin_names == timing_index_->pins
              && base.pin2node_map == m.pin_owner && base.pin2net_map == m.pin_net
              && base.net2driver_pin_map == m.net_drivers
              && base.num_nodes == m.original_node_count
              && base.num_nets == m.base_net_pins.size()
              && base.node_is_buffer == queue_original_buffer_nodes_
              && base.dbu == queue_liberty_->dbu,
          "native planning design does not match timing owner");
  require(inputs->steiner.steiner_node_base == base.num_pins
              && inputs->steiner.net_flat_topo_sort_start.size() == base.num_nets + 1,
          "native planning Steiner does not match base physical graph");
  require(inputs->steiner.net_ids.size() == base.num_nets,
          "native planning requires complete base Steiner net identities");
  for (int net = 0; net < base.num_nets; ++net) {
    require(inputs->steiner.net_ids[net] == net,
            "native planning requires ordered base Steiner net identities");
  }
  require(base.node_is_fixed.empty() || base.node_is_fixed.size() == base.num_nodes,
          "native planning physical fixed flags do not align");
  inputs->carrier = overlayCarrierInputs(carrier_);
  // Import immutable base buffer boundaries once.  Subsequent local
  // snapshots derive newly inserted boundaries from the native mutation
  // records instead of scanning every projected pin and instance.
  {
    auto links = std::make_shared<std::vector<CurrentDriverBufferLink>>();
    std::unordered_map<int, CurrentDriverBufferLink> by_input;
    std::string reason;
    if (!collectCurrentDriverBufferLinks(inputs->design, *links, by_input, reason)) {
      throw py::value_error(reason);
    }
    inputs->base_links = std::move(links);
  }
  planning_inputs_ = std::move(inputs);
  py::dict result;
  result["status"] = "ok";
  result["pin_count"] = base.num_pins;
  return result;
}

std::shared_ptr<NativeTimingPlanningSnapshot> PersistentTimingGraph::planningSnapshot(
    const py::list& actions, const py::dict& config)
{
  require(planning_inputs_ != nullptr, "native planning inputs are not initialized");
  return std::shared_ptr<NativeTimingPlanningSnapshot>(
      new NativeTimingPlanningSnapshot(shared_from_this(), planning_inputs_, actions, config));
}

std::shared_ptr<IncrementalTopologyState> PersistentTimingGraph::createNetTopology(
    int net_id) const
{
  require(planning_inputs_ != nullptr, "native planning inputs are not initialized");
  require(net_id >= 0 && net_id < planning_inputs_->design.num_nets,
          "net_id is out of range for native planning");
  return incrementalTopologyFromSteiner(
      planning_inputs_->steiner,
      net_id,
      planning_inputs_->design.net2driver_pin_map.at(net_id),
      planning_inputs_->r_unit,
      planning_inputs_->c_unit);
}

NativeTimingPlanningSnapshot::NativeTimingPlanningSnapshot(
    std::shared_ptr<PersistentTimingGraph> graph,
    std::shared_ptr<const TimingPlanningInputs> inputs,
    const py::list& actions, const py::dict& config)
    : graph_(std::move(graph)), checkpoint_(graph_->checkpoint()), inputs_(std::move(inputs)),
      liberty_(graph_->queue_liberty_), config_(config.attr("copy")().cast<py::dict>())
{
  // This validates current propagation/topology before copying any views.
  auto previous = std::chrono::steady_clock::now();
  auto finish_phase = [&](const char* name) {
    const auto now = std::chrono::steady_clock::now();
    phase_timings_ms_[name] = std::chrono::duration<double, std::milli>(now - previous).count();
    previous = now;
  };
  const bool path_context = config_.contains("use_native_timing_path_context")
      && config_["use_native_timing_path_context"].cast<bool>();
  const auto endpoint = config_.contains("native_endpoint_pin")
      ? std::make_optional(config_["native_endpoint_pin"].cast<int>()) : std::nullopt;
  queue_metadata_ = graph_->timingQueueState(inputs_->steiner.net_ids, path_context, endpoint);
  graph_->applyQueueGradientScores(queue_metadata_, config_);
  finish_phase("queue_state");
  std::vector<std::shared_ptr<IncrementalTopologyState>> states;
  for (const auto& [_, state] : graph_->net_states_) states.push_back(state);
  projection_ = buildOverlayCarrierProjection(inputs_->carrier, inputs_->steiner,
      states, actions, inputs_->r_unit, inputs_->c_unit);
  finish_phase("carrier_projection");
  buildIdentity();
  finish_phase("identity");
  buildViews();
  finish_phase("timing_views");
  buildBufferLinks();
  finish_phase("buffer_links");
  buildQueue();
  finish_phase("queue_projection");
}

void NativeTimingPlanningSnapshot::requireCurrent() const
{
  graph_->validateCheckpoint(checkpoint_);
  require(graph_->epoch_ == checkpoint_->epoch
              && graph_->journal_.size() == checkpoint_->journal_size,
          "native planning snapshot is stale");
  for (const auto& [net, _] : checkpoint_->topology_checkpoints) {
    require(graph_->net_states_.count(net) != 0, "native planning topology was detached");
  }
  for (const auto& [_, state] : graph_->net_states_) {
    requireSynchronizedTimingTopology(*graph_->mutation_state_, *state);
  }
}

void NativeTimingPlanningSnapshot::buildViews()
{
  const auto& m = *graph_->mutation_state_;
  const auto& b = inputs_->design;
  auto& d = projection_.design;
  d.pin_cap_limit = b.pin_cap_limit;
  d.pin_cap = b.pin_cap;
  d.pin_slew_limit = b.pin_slew_limit;
  d.pin_drive_resistance = b.pin_drive_resistance;
  d.main_id_2_cell_id_start = b.main_id_2_cell_id_start;
  d.cell_id_2_libpin_id_start = b.cell_id_2_libpin_id_start;
  for (int node = 0; node < d.num_nodes; ++node) {
    const int original = graph_node_by_node_.at(node);
    const int cell = m.node_cells.at(original);
    const int main = cell < 0 ? -1 : m.cell_mains.at(cell);
    d.inst_main_id.push_back(main);
    d.inst_libcell_offset.push_back(main < 0 ? 0 : cell - b.main_id_2_cell_id_start.at(main));
    d.node_is_buffer.push_back(original < m.original_node_count
        ? b.node_is_buffer.at(original) : liberty_->cells.at(cell).is_buffer);
    if (!b.node_is_fixed.empty()) {
      d.node_is_fixed.push_back(original < m.original_node_count ? b.node_is_fixed.at(original) : 0);
    }
    if (cell >= 0) projection_.node_master_names[node] = m.cell_names.at(cell);
  }
  for (int pin = 0; pin < d.num_pins; ++pin) {
    const int source = graph_pin_by_pin_[pin], owner = m.pin_owner[source];
    const int cell = m.node_cells[owner], net = m.pin_net[source];
    require((net < 0 ? -1 : net_by_graph_.at(net)) == d.pin2net_map[pin],
            "native planning pin partition differs from timing owner");
    int offset = -1;
    if (cell >= 0) {
      const auto begin = m.library_pin_names.begin() + m.cell_pin_start.at(cell);
      const auto end = m.library_pin_names.begin() + m.cell_pin_start.at(cell + 1);
      const auto found = std::find(begin, end, m.pin_ports[source]);
      require(found != end, "native planning pin is absent from current cell");
      offset = std::distance(begin, found);
    }
    d.pin_2_libpin_offset.push_back(offset);
    timing_.pin_cap.push_back(m.pin_values[0][source]);
    timing_.pin_max_slew.push_back(m.pin_values[3][source]);
    timing_.pin_max_cap.push_back(m.pin_values[4][source]);
  }
  for (std::size_t net = 0; net < m.net_drivers.size(); ++net) {
    if (net_by_graph_[net] < 0) continue;
    const int driver = m.net_drivers[net];
    require((driver < 0 ? -1 : pin_by_graph_.at(driver))
                == d.net2driver_pin_map.at(net_by_graph_[net]),
            "native planning driver differs from timing owner");
  }
  auto values = [&](const char* field) {
    const auto& source = graph_->state_.at(field);
    std::vector<double> result;
    result.reserve(d.num_pins);
    for (const int pin : graph_pin_by_pin_) result.push_back(source.at(pin));
    return result;
  };
  timing_.pin_slew_rise = values("pin_rtran");
  timing_.pin_slew_fall = values("pin_ftran");
  timing_.pin_load = values("pin_net_cap_rise");
  timing_.pin_arrival_rise = values("pin_rAAT");
  timing_.pin_arrival_fall = values("pin_fAAT");
  timing_.pin_required_rise = values("pin_rRAT");
  timing_.pin_required_fall = values("pin_fRAT");
  for (int pin = 0; pin < d.num_pins; ++pin) {
    timing_.pin_slack.push_back(dreamplace::timing::hardMinimum(
        timing_.pin_required_rise[pin] - timing_.pin_arrival_rise[pin],
        timing_.pin_required_fall[pin] - timing_.pin_arrival_fall[pin]));
  }
  const auto slacks = graph_->endpointSlacks();
  for (std::size_t row = 0; row < slacks.size(); ++row) {
    timing_.pin_slack.at(pin_by_graph_.at(graph_->propagation_program_->metric_pins[row])) = slacks[row];
  }
  for (const int pin : graph_->timing_index_->clocks) {
    timing_.clock_pins.push_back(pin_by_graph_.at(pin));
  }
}

py::dict NativeTimingPlanningSnapshot::timingView() const
{
  py::dict result;
  result["pin_slew_rise"] = timing_.pin_slew_rise;
  result["pin_slew_fall"] = timing_.pin_slew_fall;
  result["pin_load"] = timing_.pin_load;
  result["pin_cap"] = timing_.pin_cap;
  result["pin_max_slew"] = timing_.pin_max_slew;
  result["pin_max_cap"] = timing_.pin_max_cap;
  result["pin_slack"] = timing_.pin_slack;
  result["pin_arrival_rise"] = timing_.pin_arrival_rise;
  result["pin_arrival_fall"] = timing_.pin_arrival_fall;
  result["pin_required_rise"] = timing_.pin_required_rise;
  result["pin_required_fall"] = timing_.pin_required_fall;
  result["clock_pins"] = timing_.clock_pins;
  return result;
}

}  // namespace rebuffer

// --- rebuffer_timing_planning_identity.cc ---
// Original slots remain fixed; live buffer records map appended/compacted slots.

#include <algorithm>
#include <numeric>
#include <unordered_map>

namespace py = pybind11;
namespace rebuffer {

void NativeTimingPlanningSnapshot::buildIdentity()
{
  const auto& m = *graph_->mutation_state_;
  const auto& d = projection_.design;
  const auto& base = inputs_->carrier;
  const int original_pins = inputs_->design.num_pins;
  const int original_nets = inputs_->design.num_nets;
  const int movable = m.original_node_count - base.num_terminals - base.num_terminal_nis;
  if (movable < 0 || d.num_nodes != m.original_node_count + projection_.buffers.size()
      || d.num_pins != original_pins + 2 * projection_.buffers.size()) {
    throw py::value_error("native planning original and inserted slot counts differ");
  }
  graph_pin_by_pin_.resize(d.num_pins);
  std::iota(graph_pin_by_pin_.begin(), graph_pin_by_pin_.begin() + original_pins, 0);
  graph_node_by_node_.assign(d.num_nodes, -1);
  pin_by_graph_.assign(graph_->timing_index_->pins.size(), -1);
  net_by_graph_.assign(m.net_names.size(), -1);
  for (int node = 0; node < m.original_node_count; ++node) {
    const int target = node < movable ? node : node + projection_.buffers.size();
    graph_node_by_node_.at(target) = node;
  }
  std::iota(net_by_graph_.begin(), net_by_graph_.begin() + original_nets, 0);
  std::unordered_map<int, const TimingBufferRecord*> buffers;
  for (const auto& buffer : m.buffers) {
    if (buffer.active && !buffers.emplace(buffer.action_id, &buffer).second) {
      throw py::value_error("native planning has duplicate active buffer identities");
    }
  }
  for (const auto& projected : projection_.buffers) {
    const auto found = buffers.find(projected.action_id);
    if (found == buffers.end()) {
      throw py::value_error("native planning buffer has no active graph slots");
    }
    const auto& buffer = *found->second;
    graph_pin_by_pin_.at(projected.input_pin) = buffer.input_pin;
    graph_pin_by_pin_.at(projected.output_pin) = buffer.output_pin;
    graph_node_by_node_.at(projected.node) = buffer.node;
    net_by_graph_.at(buffer.output_net) = projected.output_net;
  }
  for (int pin = 0; pin < d.num_pins; ++pin) {
    const int source = graph_pin_by_pin_[pin];
    if (!m.pin_active.at(source) || pin_by_graph_.at(source) >= 0
        || graph_->timing_index_->pins.at(source) != d.pin_names[pin]) {
      throw py::value_error("native planning pin slots do not match physical identities");
    }
    pin_by_graph_[source] = pin;
  }
  if (d.num_pins != std::count(m.pin_active.begin(), m.pin_active.end(), 1)) {
    throw py::value_error("native planning is missing active graph pins");
  }
  for (int node = 0; node < d.num_nodes; ++node) {
    if (m.node_names.at(graph_node_by_node_[node]) != projection_.node_names[node]) {
      throw py::value_error("native planning node slots do not match physical identities");
    }
  }
  for (std::size_t net = 0; net < m.net_names.size(); ++net) {
    const int target = net_by_graph_[net];
    if (target < 0) {
      if (net < m.base_net_pins.size() || m.net_drivers[net] >= 0) {
        throw py::value_error("native planning is missing an active graph net");
      }
    } else if (m.net_names[net] != projection_.net_names.at(target)) {
      throw py::value_error("native planning net slots do not match physical identities");
    }
  }
}

}  // namespace rebuffer

// --- rebuffer_timing_planning_tree.cc ---
// Current-tree and endpoint annotation reuse the existing native VGDP inputs.
#include "rebuffer_driver_tree.hh"


#include <algorithm>
#include <set>


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
  const int driver_pin = views.tree->driver_pin_id;
  if (driver_pin >= 0 && driver_pin < static_cast<int>(graph_pin_by_pin_.size())) {
    std::vector<int> native_path;
    native_path.reserve(item.path_pins.size());
    for (const int pin : item.path_pins) {
      if (pin >= 0 && pin < static_cast<int>(graph_pin_by_pin_.size())) {
        native_path.push_back(graph_pin_by_pin_[pin]);
      }
    }
    const auto [resize_node, resize_cell] =
        graph_->findDriverResizeCandidate(graph_pin_by_pin_[driver_pin], native_path, config);
    views.driver_resize_node_id = resize_node;
    views.driver_resize_cell_index = resize_cell;
  }
  ++planned_requests_;
  return planNativeCurrentTree(request, views, config);
}

}  // namespace rebuffer

// --- rebuffer_timing_planning_queue.cc ---
// Native queue dispatch and action identities share one owned projection.

#include <algorithm>
#include <set>


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

// --- rebuffer_timing_policy.cc ---

#include <algorithm>
#include <cmath>
#include <limits>
#include <string>

namespace py = pybind11;
namespace rebuffer {
namespace {

double number(const py::dict& values, const char* key, double fallback)
{
  return values.contains(key) && !values[key].is_none()
             ? py::cast<double>(values[key]) : fallback;
}

bool boolean(const py::dict& values, const char* key, bool fallback)
{
  return values.contains(key) && !values[key].is_none()
             ? py::cast<bool>(values[key]) : fallback;
}

}  // namespace

bool acceptsTimingCandidate(const py::dict& current, const py::dict& candidate,
                            const py::dict& request, const py::dict& config)
{
  if (candidate.contains("status") && py::cast<std::string>(candidate["status"]) != "ok") return false;
  const double missing = std::numeric_limits<double>::quiet_NaN();
  const double wns = number(candidate, "wns_ps", missing);
  const double tns = number(candidate, "tns_ps", missing);
  const double old_wns = number(current, "wns_ps", missing);
  const double old_tns = number(current, "tns_ps", missing);
  if (!std::isfinite(wns) || !std::isfinite(tns)
      || !std::isfinite(old_wns) || !std::isfinite(old_tns)) return false;
  if (!electricalNonRegressing(current, candidate)) return false;
  constexpr double epsilon = 1.0e-9;
  if (config.contains("acceptance")
      && config["acceptance"].cast<std::string>() == "openroad_legacy") {
    const bool endpoint_phase = request.contains("legacy_phase")
        && request["legacy_phase"].cast<std::string>() == "endpoint";
    const double previous = number(request, "previous_endpoint_slack_ps", missing);
    const double endpoint = number(candidate, "endpoint_slack_ps", missing);
    return wns > old_wns + epsilon
        || (endpoint_phase && std::abs(wns - old_wns) <= epsilon
            && std::isfinite(previous) && std::isfinite(endpoint)
            && endpoint > previous + epsilon);
  }
  const double margin = std::max(0.0, number(config, "tns_regression_margin_ps", 0.0));
  // Every accepted epoch protects global WNS and the configured TNS envelope,
  // including endpoint-journal fallback and branch-focused candidates.
  if (wns < old_wns - epsilon || tns < old_tns - margin) return false;
  if (wns > old_wns + epsilon || tns > old_tns + margin) return true;
  const bool branch = boolean(config, "native_tree_branch_queue", false)
      && boolean(config, "native_tree_branch_global_acceptance", true)
      && request.contains("focus_net_id") && request.contains("root_net_id")
      && py::cast<int>(request["focus_net_id"]) != py::cast<int>(request["root_net_id"]);
  if (branch || !config.contains("acceptance")
      || py::cast<std::string>(config["acceptance"]) != "endpoint_journal"
      || !boolean(config, "allow_decreasing_slack", false)) return false;
  const double previous = number(candidate, "previous_endpoint_slack_ps",
      number(request, "previous_endpoint_slack_ps", number(current, "endpoint_slack_ps", missing)));
  const double endpoint = number(candidate, "endpoint_slack_ps", missing);
  return std::isfinite(previous) && std::isfinite(endpoint) && endpoint > previous + epsilon;
}

}  // namespace rebuffer
