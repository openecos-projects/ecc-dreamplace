// SPDX-License-Identifier: BSD-3-Clause

#include "rebuffer_timing_graph.hh"

#include "rebuffer_timing_mutation.hh"

#include "rebuffer_timing_local.hh"


#include <algorithm>
#include <array>
#include <queue>
#include <string>
#include <unordered_set>
#include <utility>

#include "pybind11/stl.h"

#include "rebuffer_carrier.hh"

#include "rebuffer_repair.hh"

#include "rebuffer_incremental_topology.hh"


namespace py = pybind11;

namespace rebuffer {
namespace {

std::string netStateField(const int net_id, const char* field)
{
  return "net_" + std::to_string(net_id) + "_" + field;
}

}  // namespace

PersistentTimingGraph::PersistentTimingGraph(const py::object& carrier)
    : carrier_(carrier), timing_index_(timingGraphIndex(carrier_))
{
  if (timing_index_ == nullptr) {
    throw py::value_error("persistent timing graph requires a carrier graph");
  }
}

py::dict PersistentTimingGraph::summary() const
{
  if (!cached_active_counts_valid_) {
    cached_active_net_arcs_ = timing_index_->net_active.empty()
        ? timing_index_->nets.size()
        : std::count(timing_index_->net_active.begin(), timing_index_->net_active.end(), 1);
    cached_active_pins_ = mutation_state_ == nullptr
        ? timing_index_->pins.size()
        : std::count(mutation_state_->pin_active.begin(), mutation_state_->pin_active.end(), 1);
    cached_active_cell_arcs_ = mutation_state_ == nullptr
        ? timing_index_->cells.size()
        : std::count(mutation_state_->cell_active.begin(), mutation_state_->cell_active.end(), 1);
    cached_active_counts_valid_ = true;
  }
  py::dict result;
  result["status"] = "ok";
  result["pin_count"] = static_cast<int>(timing_index_->pins.size());
  result["cell_arc_count"] = static_cast<int>(timing_index_->cells.size());
  result["net_arc_count"] = static_cast<int>(timing_index_->nets.size());
  result["active_net_arc_count"] = cached_active_net_arcs_;
  result["clock_pin_count"] = static_cast<int>(timing_index_->clocks.size());
  result["clock_end"] = timing_index_->clock_end;
  result["net_count"] = static_cast<int>(net_states_.size());
  result["journal_size"] = static_cast<std::int64_t>(journal_.size());
  if (journal_.empty()) {
    cached_journal_full_values_ = 0;
    cached_journal_sparse_values_ = 0;
    cached_journal_stats_size_ = 0;
    cached_journal_stats_head_seq_ = 0;
  } else if (journal_.size() >= cached_journal_stats_size_
             && (cached_journal_stats_size_ == 0
                 || journal_[cached_journal_stats_size_ - 1].sequence
                        == cached_journal_stats_head_seq_)) {
    for (std::size_t i = cached_journal_stats_size_; i < journal_.size(); ++i) {
      const auto& entry = journal_[i];
      cached_journal_full_values_ += entry.previous.size();
      if (entry.state_undo) cached_journal_sparse_values_ += entry.state_undo->value_count;
    }
    cached_journal_stats_size_ = journal_.size();
    cached_journal_stats_head_seq_ = journal_.back().sequence;
  } else {
    std::size_t full_values = 0, sparse_values = 0;
    for (const auto& entry : journal_) {
      full_values += entry.previous.size();
      if (entry.state_undo) sparse_values += entry.state_undo->value_count;
    }
    cached_journal_full_values_ = full_values;
    cached_journal_sparse_values_ = sparse_values;
    cached_journal_stats_size_ = journal_.size();
    cached_journal_stats_head_seq_ = journal_.back().sequence;
  }
  const std::size_t full_values = cached_journal_full_values_;
  const std::size_t sparse_values = cached_journal_sparse_values_;
  result["journal_full_state_values"] = full_values;
  result["journal_sparse_state_values"] = sparse_values;
  result["journal_state_value_bytes"]
      = full_values * sizeof(double) + sparse_values * sizeof(TimingIndexedValue);
  result["epoch"] = static_cast<std::uint64_t>(epoch_);
  result["timing_state_initialized"] = timing_state_initialized_;
  result["propagation_initialized"] = propagation_program_ != nullptr;
  result["mutations_initialized"] = mutation_state_ != nullptr;
  result["active_pin_count"] = cached_active_pins_;
  result["active_cell_arc_count"] = cached_active_cell_arcs_;
  py::list fields;
  for (const auto& [name, _] : state_) {
    fields.append(name);
  }
  result["state_fields"] = std::move(fields);
  return result;
}

std::shared_ptr<IncrementalTopologyState>
PersistentTimingGraph::topologyState(const int net_id) const
{
  const auto found = net_states_.find(net_id);
  if (found == net_states_.end()) {
    throw py::value_error("persistent timing graph net topology is not attached");
  }
  return found->second;
}

py::dict PersistentTimingGraph::attachNetTopology(
    const int net_id,
    const int driver_pin_id,
    const int next_node_id,
    const py::dict& topology_values)
{
  if (net_states_.find(net_id) != net_states_.end()) {
    throw py::value_error("persistent timing graph net topology is already attached");
  }
  std::string reason;
  const auto state = createIncrementalTopology(
      net_id, driver_pin_id, next_node_id, topology_values, reason);
  if (state == nullptr) {
    throw py::value_error(reason);
  }
  net_states_.emplace(net_id, state);
  topology_revision_ = ++next_topology_revision_;
  ++net_states_revision_;
  return incrementalTopologyView(*state);
}

void PersistentTimingGraph::attachNetTopologyStateFast(
    const int net_id,
    const std::shared_ptr<IncrementalTopologyState>& state)
{
  if (state == nullptr) {
    throw py::value_error("persistent timing graph net topology state is null");
  }
  if (state->net_id != net_id) {
    throw py::value_error("persistent timing graph net topology identity mismatch");
  }
  const auto found = net_states_.find(net_id);
  if (found != net_states_.end() && found->second != state) {
    throw py::value_error("persistent timing graph net topology is already attached");
  }
  if (found == net_states_.end()) {
    topology_revision_ = ++next_topology_revision_;
    ++net_states_revision_;
    net_states_.emplace(net_id, state);
  }
}

py::dict PersistentTimingGraph::attachNetTopologyState(
    const int net_id,
    const std::shared_ptr<IncrementalTopologyState>& state)
{
  attachNetTopologyStateFast(net_id, state);
  return incrementalTopologyView(*state);
}

py::dict PersistentTimingGraph::clearNetTopologies()
{
  const int removed_count = static_cast<int>(net_states_.size());
  net_states_.clear();
  topology_revision_ = ++next_topology_revision_;
  ++net_states_revision_;
  py::dict result;
  result["status"] = "ok";
  result["removed_count"] = removed_count;
  result["net_count"] = static_cast<int>(net_states_.size());
  return result;
}

py::dict PersistentTimingGraph::applyBufferSplit(
    const int net_id, const py::dict& action)
{
  if (mutation_state_ != nullptr) {
    return mutateTimingTopology(net_id, action, TimingTopologyOperation::insert);
  }
  const auto state = topologyState(net_id);
  std::string reason;
  py::dict result;
  if (!applyIncrementalBufferSplit(*state, action, reason, result)) {
    throw py::value_error(reason);
  }
  return result;
}

py::dict PersistentTimingGraph::removeBuffer(
    const int net_id, const int action_id)
{
  if (mutation_state_ != nullptr) {
    py::dict action;
    action["action_id"] = action_id;
    return mutateTimingTopology(net_id, action, TimingTopologyOperation::remove);
  }
  const auto state = topologyState(net_id);
  std::string reason;
  py::dict result;
  if (!removeIncrementalBuffer(*state, action_id, reason, result)) {
    throw py::value_error(reason);
  }
  return result;
}

py::dict PersistentTimingGraph::resizeBuffer(
    const int net_id, const py::dict& action)
{
  if (mutation_state_ != nullptr) {
    return mutateTimingTopology(net_id, action, TimingTopologyOperation::resize);
  }
  const auto state = topologyState(net_id);
  std::string reason;
  py::dict result;
  if (!resizeIncrementalBuffer(*state, action, reason, result)) {
    throw py::value_error(reason);
  }
  return result;
}

py::dict PersistentTimingGraph::topologyView(const int net_id) const
{
  return incrementalTopologyView(*topologyState(net_id));
}

py::dict PersistentTimingGraph::relevelizeNet(const int net_id) const
{
  const auto state = topologyState(net_id);
  int max_node = std::max(0, state->driver_pin_id);
  for (const auto& edge : state->edges) {
    max_node = std::max(max_node, std::max(edge.parent, edge.child));
  }
  std::vector<unsigned char> present(max_node + 1, 0);
  auto markPresent = [&](const int node) {
    if (node < 0) {
      throw py::value_error("persistent timing graph level node is negative");
    }
    present.at(node) = 1;
  };
  markPresent(state->driver_pin_id);
  for (const auto& edge : state->edges) {
    markPresent(edge.parent);
    markPresent(edge.child);
  }

  std::vector<std::vector<std::pair<int, int>>> successors(max_node + 1);
  std::vector<int> indegree(max_node + 1, 0);
  for (const auto& edge : state->edges) {
    bool cell_edge = false;
    for (const auto& buffer : state->buffers) {
      if (edge.parent == buffer.input_node
          && edge.child == buffer.output_node) {
        cell_edge = true;
        break;
      }
    }
    successors.at(edge.parent).emplace_back(edge.child, cell_edge ? 1 : 0);
    ++indegree.at(edge.child);
  }

  std::vector<int> level(max_node + 1, -1);
  std::queue<int> ready;
  for (int node = 0; node <= max_node; ++node) {
    if (present.at(node) && indegree.at(node) == 0) {
      level.at(node) = 0;
      ready.push(node);
    }
  }
  std::vector<int> order;
  while (!ready.empty()) {
    const int node = ready.front();
    ready.pop();
    order.push_back(node);
    for (const auto& [next, weight] : successors.at(node)) {
      level.at(next) = std::max(level.at(next), level.at(node) + weight);
      if (--indegree.at(next) == 0) ready.push(next);
    }
  }
  if (static_cast<int>(order.size())
      != static_cast<int>(std::count(present.begin(), present.end(), 1))) {
    throw py::value_error("persistent timing graph level graph is cyclic or disconnected");
  }
  int max_level = 0;
  for (const int node : order) max_level = std::max(max_level, level.at(node));

  py::dict result;
  result["status"] = "ok";
  result["levels"] = level;
  result["topological_order"] = order;
  result["max_level"] = max_level;
  return result;
}

py::dict PersistentTimingGraph::compactNetTopology(const int net_id)
{
  const auto state = topologyState(net_id);
  const int pin_count = static_cast<int>(timing_index_->pins.size());
  std::vector<int> nodes;
  auto collect = [&](const int node) {
    if (node >= pin_count) nodes.push_back(node);
  };
  for (const auto& edge : state->edges) {
    collect(edge.parent);
    collect(edge.child);
  }
  for (const auto& [node, _] : state->node_loads) collect(node);
  for (const auto& [node, _] : state->coordinates) collect(node);
  for (const auto& buffer : state->buffers) {
    collect(buffer.input_node);
    collect(buffer.output_node);
  }
  std::sort(nodes.begin(), nodes.end());
  nodes.erase(std::unique(nodes.begin(), nodes.end()), nodes.end());

  py::dict mapping;
  bool changed = false;
  for (std::size_t index = 0; index < nodes.size(); ++index) {
    const int target = pin_count + static_cast<int>(index);
    mapping[py::int_(nodes[index])] = target;
    changed = changed || nodes[index] != target;
  }
  if (changed) {
    const auto compacted = remapIncrementalTopology(
        state, pin_count, 0, state->net_id, -1, mapping);
    const int compact_next_node
        = pin_count + static_cast<int>(nodes.size());
    compacted->next_node_id = compact_next_node;
    compacted->base_next_node_id = compact_next_node;
    invalidateIncrementalTopologyCaches(*compacted);
    *state = *compacted;
    topology_revision_ = ++next_topology_revision_;
  }

  py::dict result;
  result["status"] = "ok";
  result["changed"] = changed;
  result["mapping"] = mapping;
  result["next_node_id"] = state->next_node_id;
  result["epoch"] = state->epoch;
  return result;
}

py::dict PersistentTimingGraph::criticalPathSubset(
    const py::dict& path_details,
    const py::object& endpoint_pins) const
{
  std::unordered_set<int> requested;
  if (!endpoint_pins.is_none()) {
    for (const int endpoint : endpoint_pins.cast<std::vector<int>>()) {
      requested.insert(endpoint);
    }
  }
  auto connected = [&](const int source, const int sink) {
    if (source < 0 || sink < 0
        || source >= static_cast<int>(timing_index_->pins.size())
        || sink >= static_cast<int>(timing_index_->pins.size())) {
      return false;
    }
    for (const int arc_id : timing_index_->cell_out[source]) {
      if (timing_index_->cells.at(arc_id)[1] == sink) return true;
    }
    for (const int arc_id : timing_index_->net_out[source]) {
      if (timing_index_->nets.at(arc_id)[1] == sink) return true;
    }
    return false;
  };

  py::dict paths;
  py::list invalid;
  for (const auto& item : path_details) {
    const int endpoint = py::cast<int>(item.first);
    if (!requested.empty() && requested.find(endpoint) == requested.end()) {
      continue;
    }
    const py::dict details = item.second.cast<py::dict>();
    bool valid = details.contains("path");
    std::vector<int> path;
    if (valid) path = details["path"].cast<std::vector<int>>();
    if (valid && (path.empty() || path.back() != endpoint)) valid = false;
    if (valid) {
      for (std::size_t index = 1; index < path.size(); ++index) {
        if (!connected(path[index - 1], path[index])) {
          valid = false;
          break;
        }
      }
    }
    if (valid) {
      paths[py::int_(endpoint)] = details;
    } else {
      py::dict failure;
      failure["endpoint_pin"] = endpoint;
      failure["reason"] = "path_is_not_connected_to_carrier_graph";
      invalid.append(failure);
    }
  }
  py::dict result;
  result["status"] = "ok";
  result["paths"] = paths;
  result["invalid"] = invalid;
  result["endpoint_count"] = static_cast<int>(py::len(paths));
  return result;
}

py::list PersistentTimingGraph::finalizeActions(const py::list& actions) const
{
  std::vector<std::shared_ptr<IncrementalTopologyState>> states;
  states.reserve(net_states_.size());
  for (const auto& [_, state] : net_states_) states.push_back(state);
  return finalizeOverlayActions(carrier_, states, actions);
}

py::dict PersistentTimingGraph::buildNetRc(const int net_id) const
{
  const auto state = topologyState(net_id);
  int max_node = std::max(0, state->driver_pin_id);
  for (const auto& edge : state->edges) {
    max_node = std::max(max_node, std::max(edge.parent, edge.child));
  }
  for (const auto& [node, _] : state->node_loads) {
    max_node = std::max(max_node, node);
  }
  for (const auto& buffer : state->buffers) {
    max_node = std::max(max_node, std::max(buffer.input_node, buffer.output_node));
  }

  std::vector<unsigned char> present(max_node + 1, 0);
  auto markPresent = [&](const int node) {
    if (node < 0) {
      throw py::value_error("persistent timing graph RC node is negative");
    }
    present.at(node) = 1;
  };
  markPresent(state->driver_pin_id);
  for (const auto& edge : state->edges) {
    markPresent(edge.parent);
    markPresent(edge.child);
  }
  for (const auto& [node, _] : state->node_loads) markPresent(node);
  for (const auto& buffer : state->buffers) {
    markPresent(buffer.input_node);
    markPresent(buffer.output_node);
  }

  std::vector<int> parent(max_node + 1, -1);
  std::vector<std::vector<int>> children(max_node + 1);
  std::vector<double> resistance(max_node + 1, 0.0);
  std::vector<double> capacitance(max_node + 1, 0.0);
  for (const auto& edge : state->edges) {
    bool logical_buffer_edge = false;
    for (const auto& buffer : state->buffers) {
      if (edge.parent == buffer.input_node
          && edge.child == buffer.output_node) {
        logical_buffer_edge = true;
        break;
      }
    }
    if (logical_buffer_edge) continue;
    const int child = edge.child;
    if (parent.at(child) >= 0 && parent.at(child) != edge.parent) {
      throw py::value_error("persistent timing graph RC node has multiple parents");
    }
    parent.at(child) = edge.parent;
    children.at(edge.parent).push_back(child);
    resistance.at(child) = edge.resistance;
    capacitance.at(child) = edge.capacitance;
  }

  std::vector<int> roots{state->driver_pin_id};
  for (const auto& buffer : state->buffers) roots.push_back(buffer.output_node);
  std::vector<unsigned char> visit(max_node + 1, 0);
  std::vector<int> topology_starts{0};
  std::vector<int> topology;
  for (const int root : roots) {
    if (root < 0 || root > max_node || !present.at(root)) {
      throw py::value_error("persistent timing graph RC root is missing");
    }
    if (parent.at(root) >= 0 && parent.at(root) != root) {
      throw py::value_error("persistent timing graph RC root has a parent");
    }
    parent.at(root) = root;
    if (visit.at(root) == 2) continue;
    std::vector<std::pair<int, bool>> pending{{root, false}};
    while (!pending.empty()) {
      const auto [node, leaving] = pending.back();
      pending.pop_back();
      if (leaving) {
        visit.at(node) = 2;
        continue;
      }
      if (visit.at(node) == 2) continue;
      if (visit.at(node) == 1) {
        throw py::value_error("persistent timing graph RC forest has a cycle");
      }
      visit.at(node) = 1;
      topology.push_back(node);
      pending.emplace_back(node, true);
      for (auto it = children.at(node).rbegin();
           it != children.at(node).rend(); ++it) {
        if (visit.at(*it) == 1) {
          throw py::value_error("persistent timing graph RC forest has a cycle");
        }
        pending.emplace_back(*it, false);
      }
    }
    topology_starts.push_back(static_cast<int>(topology.size()));
  }

  for (int node = 0; node <= max_node; ++node) {
    if (present.at(node) && visit.at(node) != 2) {
      throw py::value_error("persistent timing graph RC forest is disconnected");
    }
  }

  std::vector<int> child_starts{0};
  std::vector<int> flat_children;
  for (const auto& row : children) {
    flat_children.insert(flat_children.end(), row.begin(), row.end());
    child_starts.push_back(static_cast<int>(flat_children.size()));
  }
  std::vector<int> affected;
  for (int node = 0; node <= max_node; ++node) {
    if (present.at(node)) affected.push_back(node);
  }

  py::dict result;
  result["parent"] = parent;
  result["child_start"] = child_starts;
  result["child"] = flat_children;
  result["topo_start"] = topology_starts;
  result["topo"] = topology;
  result["edge_resistance"] = resistance;
  result["edge_capacitance"] = capacitance;
  result["affected_nodes"] = affected;
  return result;
}

py::dict PersistentTimingGraph::buildTimingPatch(
    const int net_id,
    const py::dict& action,
    const int base_vertex_count,
    const int slot) const
{
  return buildIncrementalTimingPatch(
      carrier_, topologyState(net_id), action, base_vertex_count, slot);
}

py::dict PersistentTimingGraph::exportCarrier(
    const py::dict& steiner,
    const py::list& actions,
    const double r_unit,
    const double c_unit) const
{
  std::vector<std::shared_ptr<IncrementalTopologyState>> states;
  states.reserve(net_states_.size());
  for (const auto& [_, state] : net_states_) states.push_back(state);
  return exportOverlayCarrier(
      carrier_, rebuffer_python::steinerFromDict(steiner), states, actions,
      r_unit, c_unit);
}

py::dict PersistentTimingGraph::evaluateNetRc(
    const int net_id,
    const py::dict& rc,
    const py::array_t<double,
                     py::array::c_style | py::array::forcecast>& generic_caps,
    const py::array_t<double,
                     py::array::c_style | py::array::forcecast>& rise_caps,
    const py::array_t<double,
                     py::array::c_style | py::array::forcecast>& fall_caps,
    const py::dict& buffer_caps,
    const int pin_count)
{
  // The 3b slice receives the compact RC forest from the existing native
  // patch builder. Later slices will derive it directly from this topology.
  topologyState(net_id);
  py::dict output = evaluateIncrementalRc(
      rc, generic_caps, rise_caps, fall_caps, buffer_caps, pin_count);
  py::dict state_values;
  constexpr std::array<const char*, 9> fields{
      "loads_generic", "delays_generic", "impulses_generic",
      "loads_rise", "delays_rise", "impulses_rise",
      "loads_fall", "delays_fall", "impulses_fall"};
  for (const char* field : fields) {
    const auto values = output[field].cast<std::vector<double>>();
    state_values[py::str(netStateField(net_id, field))] = writeState(values);
  }
  output["net_id"] = net_id;
  output["state"] = setState(state_values);
  return output;
}

std::shared_ptr<PersistentTimingGraphCheckpoint>
PersistentTimingGraph::checkpoint() const
{
  auto result = std::make_shared<PersistentTimingGraphCheckpoint>();
  result->journal_size = journal_.size();
  result->epoch = epoch_;
  result->topology_revision = topology_revision_;
  result->timing_state_initialized = timing_state_initialized_;
  result->graph_identity = checkpoint_identity_;
  result->journal_generation = journal_generation_;
  result->journal_head_sequence = journal_.empty() ? 0 : journal_.back().sequence;
  for (const auto& [net_id, state] : net_states_) {
    result->topology_checkpoints.emplace(
        net_id, checkpointIncrementalTopology(*state));
  }
  return result;
}

void PersistentTimingGraph::restoreTo(const std::size_t journal_size)
{
  if (journal_size > journal_.size()) {
    throw py::value_error("persistent timing graph checkpoint is from the future");
  }
  if (journal_size < journal_.size()) detachStateViews();
  while (journal_.size() > journal_size) {
    const JournalEntry& entry = journal_.back();
    if (cached_journal_stats_size_ == journal_.size()
        && cached_journal_stats_head_seq_ == entry.sequence) {
      cached_journal_full_values_ -= entry.previous.size();
      if (entry.state_undo) cached_journal_sparse_values_ -= entry.state_undo->value_count;
      --cached_journal_stats_size_;
      cached_journal_stats_head_seq_ =
          cached_journal_stats_size_ > 0 ? journal_[cached_journal_stats_size_ - 1].sequence : 0;
    }
    if (entry.propagation_undo) propagation_mark_ = *entry.propagation_undo;
    if (entry.topology_undo != nullptr) {
      for (const auto& pin : entry.topology_undo->previous_pins) {
        markElectricalDirtyPin(pin.id);
      }
      for (const auto& net : entry.topology_undo->previous_nets) {
        for (const int pin : net.pins) markElectricalDirtyPin(pin);
      }
      if (cached_active_counts_valid_) {
        const auto& undo = *entry.topology_undo;
        for (const auto& pin : undo.previous_pins) {
          cached_active_pins_ += static_cast<int>(pin.active)
                               - static_cast<int>(mutation_state_->pin_active[pin.id]);
        }
        for (std::size_t p = undo.pins; p < mutation_state_->pin_active.size(); ++p) {
          cached_active_pins_ -= static_cast<std::size_t>(mutation_state_->pin_active[p]);
        }
        for (const auto& arc : undo.previous_arcs) {
          cached_active_cell_arcs_ += static_cast<int>(arc.active)
                                    - static_cast<int>(mutation_state_->cell_active[arc.id]);
        }
        for (std::size_t a = undo.arcs; a < mutation_state_->cell_active.size(); ++a) {
          cached_active_cell_arcs_ -= static_cast<std::size_t>(mutation_state_->cell_active[a]);
        }
        for (const auto& row : undo.index_undo->net_rows) {
          cached_active_net_arcs_ += static_cast<int>(row.active)
                                   - static_cast<int>(timing_index_->net_active[row.id]);
        }
        for (std::size_t n = undo.index_undo->nets; n < timing_index_->net_active.size(); ++n) {
          cached_active_net_arcs_ -= static_cast<std::size_t>(timing_index_->net_active[n]);
        }
      }
      restoreTimingTopology(*mutation_state_, *timing_index_, *entry.topology_undo);
      topology_revision_ = entry.previous_topology_revision;
    } else if (entry.resize_undo != nullptr) {
      for (const auto& [pin, _] : entry.resize_undo->pins) {
        markElectricalDirtyPin(pin);
      }
      restoreTimingResize(*mutation_state_, *timing_index_, *entry.resize_undo);
      topology_revision_ = entry.previous_topology_revision;
    } else if (entry.state_undo != nullptr) {
      for (const char* field :
           {"pin_rtran", "pin_ftran", "pin_net_cap_rise", "pin_net_cap_fall"}) {
        const auto found = entry.state_undo->fields.find(field);
        if (found != entry.state_undo->fields.end()) {
          for (const auto& old : found->second.previous) {
            markElectricalDirtyPin(static_cast<int>(old.index));
          }
        }
      }
      restoreTimingState(state_, *entry.state_undo);
    } else if (entry.existed) {
      state_[entry.name] = entry.previous;
      markElectricalFullDirty();
    } else {
      state_.erase(entry.name);
      markElectricalFullDirty();
    }
    epoch_ = entry.previous_epoch;
    journal_.pop_back();
  }
}

void PersistentTimingGraph::validateCheckpoint(
    const std::shared_ptr<PersistentTimingGraphCheckpoint>& checkpoint) const
{
  if (checkpoint == nullptr
      || checkpoint->graph_identity.lock() != checkpoint_identity_) {
    throw py::value_error("persistent timing graph checkpoint belongs to another graph");
  }
  if (checkpoint->journal_generation != journal_generation_
      || checkpoint->journal_size > journal_.size()
      || (checkpoint->journal_size > 0
          && checkpoint->journal_head_sequence
                 != journal_[checkpoint->journal_size - 1].sequence)) {
    throw py::value_error("persistent timing graph checkpoint has stale journal history");
  }
}

py::dict PersistentTimingGraph::restore(
    const std::shared_ptr<PersistentTimingGraphCheckpoint>& checkpoint)
{
  restoreState(checkpoint);
  return summary();
}

void PersistentTimingGraph::restoreState(
    const std::shared_ptr<PersistentTimingGraphCheckpoint>& checkpoint)
{
  validateCheckpoint(checkpoint);
  for (const auto& [net_id, topology_checkpoint]
       : checkpoint->topology_checkpoints) {
    const auto state = topologyState(net_id);
    if (topology_checkpoint == nullptr) {
      throw py::value_error("persistent timing graph topology checkpoint is null");
    }
    std::string reason;
    if (!restoreIncrementalTopology(*state, *topology_checkpoint, reason)) {
      throw py::value_error(reason);
    }
  }
  bool erased_net_state = false;
  for (auto it = net_states_.begin(); it != net_states_.end();) {
    if (checkpoint->topology_checkpoints.find(it->first)
        == checkpoint->topology_checkpoints.end()) {
      it = net_states_.erase(it);
      erased_net_state = true;
    } else {
      ++it;
    }
  }
  if (erased_net_state) ++net_states_revision_;
  restoreTo(checkpoint->journal_size);
  topology_revision_ = checkpoint->topology_revision;
  timing_state_initialized_ = checkpoint->timing_state_initialized;
}

py::dict PersistentTimingGraph::commit(
    const std::shared_ptr<PersistentTimingGraphCheckpoint>& checkpoint)
{
  validateCheckpoint(checkpoint);
  for (std::size_t id = 0; id < checkpoint->journal_size; ++id) {
    if (journal_[id].sequence > propagation_mark_.sequence) propagation_mark_.ready = false;
  }
  journal_.erase(journal_.begin(), journal_.begin() + checkpoint->journal_size);
  ++journal_generation_;
  return summary();
}

std::shared_ptr<PersistentTimingGraph> createPersistentTimingGraph(
    const py::object& carrier)
{
  return std::make_shared<PersistentTimingGraph>(carrier);
}

}  // namespace rebuffer
