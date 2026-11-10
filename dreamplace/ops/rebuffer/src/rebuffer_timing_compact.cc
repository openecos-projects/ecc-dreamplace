// SPDX-License-Identifier: BSD-3-Clause
// Compact native timing slots at a committed boundary; preserve original ids.
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_buffer_state.hh"

#include <algorithm>
#include <map>

#include "pybind11/stl.h"

namespace py = pybind11;
namespace rebuffer {
namespace {

std::vector<int> liveMap(const std::vector<unsigned char>& live, std::size_t prefix)
{
  if (prefix > live.size()
      || !std::all_of(live.begin(), live.begin() + prefix, [](auto value) { return value != 0; })) {
    throw py::value_error("compact cannot remove original timing records");
  }
  std::vector<int> result(live.size(), -1);
  int next = 0;
  for (std::size_t id = 0; id < live.size(); ++id) if (live[id]) result[id] = next++;
  return result;
}

template <typename T>
std::vector<T> gather(const std::vector<T>& old, const std::vector<int>& mapping)
{
  if (old.size() != mapping.size()) throw py::value_error("compact metadata lengths do not align");
  std::vector<T> result(std::count_if(mapping.begin(), mapping.end(), [](int id) { return id >= 0; }));
  for (std::size_t id = 0; id < old.size(); ++id) {
    if (mapping[id] >= 0) result.at(mapping[id]) = old[id];
  }
  return result;
}

int mapped(int id, const std::vector<int>& mapping)
{
  if (id < 0 || id >= static_cast<int>(mapping.size()) || mapping[id] < 0) {
    throw py::value_error("compact active record references a removed slot");
  }
  return mapping[id];
}

py::array_t<int64_t> array(const std::vector<int>& values)
{
  py::array_t<int64_t> result({static_cast<py::ssize_t>(values.size())},
                              {static_cast<py::ssize_t>(sizeof(int64_t))});
  std::copy(values.begin(), values.end(), result.mutable_data());
  return result;
}

}  // namespace

py::dict PersistentTimingGraph::compact()
{
  if (!mutation_state_ || !propagation_program_ || !timing_state_initialized_) {
    throw py::value_error("compact requires initialized native mutation and timing state");
  }
  if (!journal_.empty()) {
    throw py::value_error("compact requires a committed journal boundary");
  }
  requireSynchronizedTimingTopologies(*mutation_state_, net_states_);
  const auto& old = *mutation_state_;
  const auto pins = liveMap(old.pin_active, propagation_program_->pin_count);
  const auto arcs = liveMap(old.cell_active, propagation_program_->schedule.cells.size());
  std::vector<unsigned char> node_live(old.node_names.size(), 0), net_live(old.net_names.size(), 0);
  for (std::size_t id = 0; id < node_live.size(); ++id) {
    node_live[id] = id < static_cast<std::size_t>(old.original_node_count) || old.node_cells[id] >= 0;
  }
  for (std::size_t id = 0; id < net_live.size(); ++id) {
    net_live[id] = id < old.base_net_pins.size() || old.net_drivers[id] >= 0;
  }
  const auto nodes = liveMap(node_live, old.original_node_count);
  const auto nets = liveMap(net_live, old.base_net_pins.size());
  const bool changed = std::find(pins.begin(), pins.end(), -1) != pins.end()
      || std::find(arcs.begin(), arcs.end(), -1) != arcs.end()
      || std::find(nodes.begin(), nodes.end(), -1) != nodes.end()
      || std::find(nets.begin(), nets.end(), -1) != nets.end()
      || std::find(timing_index_->net_active.begin(), timing_index_->net_active.end(), 0)
             != timing_index_->net_active.end();
  py::dict result;
  result["status"] = "ok";
  result["changed"] = changed;
  result["pin_old_to_new"] = array(pins);
  result["cell_arc_old_to_new"] = array(arcs);
  result["node_old_to_new"] = array(nodes);
  result["net_old_to_new"] = array(nets);
  if (!changed) {
    std::vector<int> net_arcs(timing_index_->nets.size());
    for (std::size_t id = 0; id < net_arcs.size(); ++id) net_arcs[id] = id;
    result["net_arc_old_to_new"] = array(net_arcs);
    result["state"] = summary();
    return result;
  }
  // Prepare and validate a complete candidate before changing the owner.
  // Compaction is infrequent; per-candidate mutation retains sparse metadata undo.
  auto next = std::make_shared<TimingMutationState>(old);
  auto& m = *next;
  auto index = std::make_shared<TimingGraphIndex>();
  index->pins = gather(timing_index_->pins, pins);
  index->clocks = timing_index_->clocks;
  index->clock_end = timing_index_->clock_end;
  m.cells = gather(old.cells, arcs);
  for (std::size_t slot = 0; slot < m.cells.size(); ++slot) {
    auto& row = m.cells[slot];
    if (slot >= static_cast<std::size_t>(index->clock_end)) row[0] = mapped(row[0], pins);
    row[1] = mapped(row[1], pins);
    row[7] = mapped(row[7], nodes);
  }
  m.cell_active.assign(m.cells.size(), 1);
  m.pin_active.assign(index->pins.size(), 1);
  m.pin_owner = gather(old.pin_owner, pins);
  m.pin_net = gather(old.pin_net, pins);
  m.pin_ports = gather(old.pin_ports, pins);
  for (int& node : m.pin_owner) node = mapped(node, nodes);
  for (int& net : m.pin_net) if (net >= 0) net = mapped(net, nets);
  for (int lane = 0; lane < 5; ++lane) m.pin_values[lane] = gather(old.pin_values[lane], pins);
  m.node_names = gather(old.node_names, nodes);
  m.node_cells = gather(old.node_cells, nodes);
  m.node_pins = gather(old.node_pins, nodes);
  m.node_arcs = gather(old.node_arcs, nodes);
  for (auto& row : m.node_pins) for (int& pin : row) pin = mapped(pin, pins);
  for (auto& row : m.node_arcs) for (int& arc : row) arc = mapped(arc, arcs);
  m.net_names = gather(old.net_names, nets);
  m.net_drivers = gather(old.net_drivers, nets);
  m.net_pins = gather(old.net_pins, nets);
  for (int& driver : m.net_drivers) if (driver >= 0) driver = mapped(driver, pins);
  for (auto& row : m.net_pins) for (int& pin : row) pin = mapped(pin, pins);
  m.buffers.clear();
  for (auto& vec : m.buffers_by_source_net) vec.clear();
  m.buffer_id_by_action_id.clear();
  for (const auto& previous : old.buffers) {
    if (!previous.active) continue;
    auto row = previous;
    row.node = mapped(row.node, nodes);
    row.input_pin = mapped(row.input_pin, pins); row.output_pin = mapped(row.output_pin, pins);
    row.output_net = mapped(row.output_net, nets);
    for (int& arc : row.arcs) arc = mapped(arc, arcs);
    const int new_id = static_cast<int>(m.buffers.size());
    m.buffers_by_source_net.at(row.source_net).push_back(new_id);
    m.buffer_id_by_action_id.emplace(row.action_id, new_id);
    m.buffers.push_back(std::move(row));
  }
  auto values = remappedTimingState(pins, arcs);
  rebuildTimingMutationIndex(m, *index, *propagation_program_);
  std::map<TimingNetArc, int> net_arc_ids;
  for (int id = 0; id < static_cast<int>(index->nets.size()); ++id) net_arc_ids.emplace(index->nets[id], id);
  std::vector<int> net_arc_map;
  for (std::size_t id = 0; id < timing_index_->nets.size(); ++id) {
    if (!timing_index_->net_active.empty() && !timing_index_->net_active[id]) {
      net_arc_map.push_back(-1);
      continue;
    }
    const auto& edge = timing_index_->nets[id];
    const TimingNetArc mapped_edge{mapped(edge[0], pins), mapped(edge[1], pins)};
    net_arc_map.push_back(net_arc_ids.at(mapped_edge));
  }
  result["net_arc_old_to_new"] = array(net_arc_map);
  detachStateViews();
  mutation_state_ = std::move(next);
  timing_index_ = std::move(index);
  state_ = std::move(values);
  cached_active_counts_valid_ = false;
  ++topology_revision_;
  markElectricalFullDirty();
  cached_endpoint_pin_names_.clear();
  cached_endpoint_slacks_by_name_values_.clear();
  ++epoch_;
  ++journal_generation_;
  result["state"] = summary();
  return result;
}

}  // namespace rebuffer
