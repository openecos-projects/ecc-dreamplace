// SPDX-License-Identifier: BSD-3-Clause
// Session-owned cumulative timing graph and RC patch.

#include "rebuffer_incremental_patch.hh"

#include "rebuffer_carrier.hh"
#include "rebuffer_incremental_timing.hh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <map>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pybind11/stl.h"
#include "pybind11/numpy.h"

namespace py = pybind11;

namespace rebuffer {
namespace {

using CellArc = std::array<int, 8>;
using NetArc = std::array<int, 2>;

struct BufferSlotLib
{
  int cell = -1;
  int arc_begin = -1;
  int sense = -1;
  int trigger = -1;
  std::string input_port;
  std::string output_port;
  std::vector<double> input_caps;
  std::vector<double> output_caps;
  std::vector<double> input_limits;
  std::vector<double> output_limits;
};

template <typename T>
py::array_t<int64_t> intArray(const std::vector<T>& values)
{
  py::array_t<int64_t> result(
      {static_cast<py::ssize_t>(values.size())},
      {static_cast<py::ssize_t>(sizeof(int64_t))});
  auto* data = result.mutable_data();
  for (std::size_t index = 0; index < values.size(); ++index) {
    data[index] = static_cast<int64_t>(values[index]);
  }
  return result;
}

template <std::size_t Width>
py::array_t<int64_t> intArray2(const std::vector<std::array<int, Width>>& values)
{
  py::array_t<int64_t> result(
      {static_cast<py::ssize_t>(values.size()), static_cast<py::ssize_t>(Width)},
      {static_cast<py::ssize_t>(Width * sizeof(int64_t)), static_cast<py::ssize_t>(sizeof(int64_t))});
  auto* data = result.mutable_data();
  for (std::size_t row = 0; row < values.size(); ++row) {
    for (std::size_t column = 0; column < Width; ++column) {
      data[row * Width + column] = static_cast<int64_t>(values[row][column]);
    }
  }
  return result;
}

struct PatchBaseCache
{
  std::vector<std::string> pins;
  std::vector<std::string> nets;
  std::vector<int> net_starts;
  std::vector<NetArc> net_arcs;
  std::vector<int> net_drivers;
  std::vector<int> pin_net;
  std::vector<std::string> cell_names;
  std::vector<int> cell_arc_starts;
  std::vector<CellArc> timing_rows;
  std::vector<int> timing_level_starts;
  std::vector<int> clock_pins;
  std::shared_ptr<TimingGraphIndex> timing_index;
};

std::shared_ptr<PatchBaseCache> patchBaseCache(const py::object& base)
{
  static std::unordered_map<PyObject*, std::shared_ptr<PatchBaseCache>> cache;
  const auto key = base.ptr();
  const auto found = cache.find(key);
  if (found != cache.end()) return found->second;
  auto value = std::make_shared<PatchBaseCache>();
  value->pins = carrierArray<std::string>(base, "pin_names");
  value->nets = carrierArray<std::string>(base, "net_names");
  value->net_starts = carrierArray<int>(base, "net_flat_arcs_start");
  value->net_arcs = carrierArray<NetArc>(base, "net_flat_arcs");
  value->net_drivers = carrierArray<int>(base, "net2driver_pin_map");
  value->pin_net = carrierArray<int>(base, "pin2net_map");
  value->cell_names = carrierArray<std::string>(base, "flat_libcell_names");
  value->cell_arc_starts = carrierArray<int>(base, "cell_id_2_arc_id_start");
  value->timing_rows = carrierArray<CellArc>(base, "flat_inst_arcs_by_level");
  value->timing_level_starts = carrierArray<int>(base, "flat_inst_arcs_by_level_start");
  value->clock_pins = carrierArray<int>(base, "clock_pins");
  if (value->timing_level_starts.size() < 2) {
    throw py::value_error("base timing arc schedule is missing its clock level");
  }
  value->timing_index = makeTimingGraphIndex(
      value->pins,
      value->timing_rows,
      value->net_arcs,
      value->clock_pins,
      value->timing_level_starts[1]);
  cache.emplace(key, value);
  return value;
}

void closePatchCone(
    std::vector<unsigned char>& marked,
    const std::vector<std::vector<int>>& edges)
{
  std::queue<int> pending;
  for (int pin = 0; pin < static_cast<int>(marked.size()); ++pin) {
    if (marked[pin]) pending.push(pin);
  }
  while (!pending.empty()) {
    const int pin = pending.front();
    pending.pop();
    for (const int next : edges[pin]) {
      if (!marked[next]) {
        marked[next] = 1;
        pending.push(next);
      }
    }
  }
}

py::dict buildPatchSchedule(
    const PatchBaseCache& base,
    const int current_pin_count,
    const std::vector<CellArc>& current_rows,
    const std::vector<int>& current_level_starts,
    const std::vector<NetArc>& current_nets,
    const std::vector<int>& changed_pins,
    const py::object& endpoints,
    const py::object& starts,
    const py::object& setup_arcs,
    const py::object& check_arcs)
{
  const auto& old = *base.timing_index;
  const int count = current_pin_count;
  if (count < static_cast<int>(old.pins.size())) {
    throw py::value_error("incremental patch timing graph cannot remove base pins");
  }
  if (current_level_starts.size() < 2 || current_level_starts.front() != 0
      || current_level_starts.back() != static_cast<int>(current_rows.size())) {
    throw py::value_error("incremental patch timing arc schedule is inconsistent");
  }
  const int clock_end = current_level_starts[1];
  if (clock_end != base.timing_level_starts.at(1)) {
    throw py::value_error("incremental patch cannot change the clock arc level");
  }
  std::vector<int> pin_map(count, -1), arc_map(current_rows.size(), -1);
  for (int pin = 0; pin < static_cast<int>(old.pins.size()); ++pin) {
    pin_map[pin] = pin;
  }
  std::vector<unsigned char> forward(count, 0), backward(count, 0);
  for (const int pin : changed_pins) {
    if (pin < 0 || pin >= count) throw py::value_error("changed timing pin is out of range");
    forward[pin] = 1;
  }
  std::vector<std::vector<int>> successors(count), predecessors(count);
  std::vector<int> indegree(count, 0), rank(count, 0);
  std::vector<unsigned char> required_net_sink(count, 0);
  for (const int pin : endpoints.cast<std::vector<int>>()) required_net_sink.at(pin) = 1;
  auto source = [&](const int id) {
    return id < clock_end ? base.clock_pins.at(current_rows.at(id)[0])
                          : current_rows.at(id)[0];
  };
  for (int id = 0; id < static_cast<int>(current_rows.size()); ++id) {
    const auto& row = current_rows[id];
    const std::array<int, 6> key{
        source(id), row[1], row[3], row[4], row[5], id < clock_end};
    const auto found = old.arc_ids.find(key);
    if (found != old.arc_ids.end()) arc_map[id] = found->second;
    else if (row[1] >= 0) forward.at(row[1]) = 1;
    if (id >= clock_end) {
      const int cell_source = row[0];
      const int sink = row[1];
      required_net_sink.at(cell_source) = 1;
      successors.at(cell_source).push_back(sink);
      predecessors.at(sink).push_back(cell_source);
      ++indegree[sink];
    }
  }
  for (const auto& edge : current_nets) {
    successors.at(edge[0]).push_back(edge[1]);
    if (required_net_sink[edge[1]]) predecessors.at(edge[1]).push_back(edge[0]);
    ++indegree[edge[1]];
  }
  std::queue<int> ready;
  for (int pin = 0; pin < count; ++pin) if (indegree[pin] == 0) ready.push(pin);
  int visited = 0;
  while (!ready.empty()) {
    const int pin = ready.front();
    ready.pop();
    ++visited;
    for (const int next : successors[pin]) {
      rank[next] = std::max(rank[next], rank[pin] + 1);
      if (--indegree[next] == 0) ready.push(next);
    }
  }
  if (visited != count) throw py::value_error("incremental patch timing graph is not a DAG");
  closePatchCone(forward, successors);
  backward = forward;
  closePatchCone(backward, predecessors);

  const int levels = rank.empty() ? 1 : *std::max_element(rank.begin(), rank.end()) + 1;
  std::vector<std::vector<int>> forward_cells(levels), forward_nets(levels);
  std::vector<std::vector<int>> backward_cells(levels), backward_nets(levels);
  std::vector<int> clock_rows;
  for (int id = 0; id < static_cast<int>(current_rows.size()); ++id) {
    const int cell_source = source(id);
    const int sink = current_rows[id][1];
    if (id < clock_end) {
      if (forward[sink]) clock_rows.push_back(id);
    } else {
      if (forward[sink]) forward_cells[rank[sink]].push_back(id);
      if (backward[cell_source]) backward_cells[rank[cell_source]].push_back(id);
    }
  }
  for (int id = 0; id < static_cast<int>(current_nets.size()); ++id) {
    const auto& edge = current_nets[id];
    if (forward[edge[1]]) forward_nets[rank[edge[1]]].push_back(id);
    if (backward[edge[0]] && required_net_sink[edge[1]]) {
      backward_nets[rank[edge[0]]].push_back(id);
    }
  }
  auto exportWorklist = [](const std::vector<std::vector<int>>& cells,
                           const std::vector<std::vector<int>>& nets,
                           bool reverse,
                           py::dict& output,
                           const char* prefix) {
    std::vector<int> flat_cells, flat_nets, cell_start{0}, net_start{0};
    for (int index = 0; index < static_cast<int>(cells.size()); ++index) {
      const int level = reverse ? static_cast<int>(cells.size()) - 1 - index : index;
      if (cells[level].empty() && nets[level].empty()) continue;
      flat_cells.insert(flat_cells.end(), cells[level].begin(), cells[level].end());
      flat_nets.insert(flat_nets.end(), nets[level].begin(), nets[level].end());
      cell_start.push_back(static_cast<int>(flat_cells.size()));
      net_start.push_back(static_cast<int>(flat_nets.size()));
    }
    output[(std::string(prefix) + "_cells").c_str()] = intArray(flat_cells);
    output[(std::string(prefix) + "_nets").c_str()] = intArray(flat_nets);
    output[(std::string(prefix) + "_cell_start").c_str()] = intArray(cell_start);
    output[(std::string(prefix) + "_net_start").c_str()] = intArray(net_start);
  };
  auto selectedPins = [](const py::object& value, const std::vector<unsigned char>& mask) {
    const auto pins = value.cast<std::vector<int>>();
    std::vector<int> selected;
    for (int row = 0; row < static_cast<int>(pins.size()); ++row) {
      if (mask.at(pins[row])) selected.push_back(row);
    }
    return selected;
  };
  auto selectedRows = [](const py::object& value, const std::vector<unsigned char>& mask) {
    const auto rows = value.cast<std::vector<CellArc>>();
    std::vector<int> selected;
    for (int id = 0; id < static_cast<int>(rows.size()); ++id) {
      if (mask.at(rows[id][1])) selected.push_back(id);
    }
    return selected;
  };
  py::dict output;
  output["pin_from_previous"] = intArray(pin_map);
  output["arc_from_previous"] = intArray(arc_map);
  std::vector<int> forward_selected, backward_selected;
  for (int pin = 0; pin < count; ++pin) {
    if (forward[pin]) forward_selected.push_back(pin);
    if (backward[pin]) backward_selected.push_back(pin);
  }
  output["forward_pins"] = intArray(forward_selected);
  output["backward_pins"] = intArray(backward_selected);
  output["clock_rows"] = intArray(clock_rows);
  output["forward_start_rows"] = intArray(selectedPins(starts, forward));
  output["backward_endpoint_rows"] = intArray(selectedPins(endpoints, backward));
  output["setup_rows"] = intArray(selectedRows(setup_arcs, backward));
  output["check_rows"] = intArray(selectedRows(check_arcs, backward));
  exportWorklist(forward_cells, forward_nets, false, output, "forward");
  exportWorklist(backward_cells, backward_nets, true, output, "backward");
  return output;
}

template <typename T>
void requireSize(const std::vector<T>& values, std::size_t expected, const char* field)
{
  if (values.size() != expected) {
    throw py::value_error(std::string("invalid ") + field + " length");
  }
}

std::vector<int> actionIds(const py::dict& action)
{
  if (!action.contains("downstream_pin_ids")) {
    throw py::value_error("incremental timing patch requires downstream_pin_ids");
  }
  return action["downstream_pin_ids"].cast<std::vector<int>>();
}

std::vector<int> mapNodes(
    const std::vector<int>& nodes,
    int pin_count,
    int base_vertex_count,
    const IncrementalBuffer& buffer)
{
  std::vector<int> result;
  result.reserve(nodes.size());
  for (const int node : nodes) {
    if (node == buffer.input_node) {
      result.push_back(pin_count);
    } else if (node == buffer.output_node) {
      result.push_back(pin_count + 1);
    } else if (node >= 0 && node < pin_count) {
      result.push_back(node);
    } else if (node >= pin_count && node < base_vertex_count) {
      result.push_back(node + 2);
    } else {
      throw py::value_error("incremental timing patch contains an unknown RC vertex");
    }
  }
  return result;
}

std::vector<int> mapNodeList(
    const std::vector<int>& nodes,
    int pin_count,
    int base_vertex_count,
    const IncrementalBuffer& buffer)
{
  return mapNodes(nodes, pin_count, base_vertex_count, buffer);
}

}  // namespace

py::dict buildIncrementalTimingPatch(
    const py::object& base,
    const std::shared_ptr<IncrementalTopologyState>& state,
    const py::dict& action,
    const int base_vertex_count,
    const int slot)
{
  if (state == nullptr) {
    throw py::value_error("incremental timing patch requires a topology state");
  }
  if (base_vertex_count <= 0) {
    throw py::value_error("incremental timing patch requires a positive RC vertex count");
  }
  if (state->buffers.empty()) {
    throw py::value_error(
        "incremental timing patch requires at least one active buffer slot");
  }
  const int buffer_count = static_cast<int>(state->buffers.size());
  const int action_id = action.contains("session_buffer_id")
      ? py::cast<int>(action["session_buffer_id"])
      : state->buffers.back().action_id;
  const auto action_buffer = std::find_if(
      state->buffers.begin(),
      state->buffers.end(),
      [&](const IncrementalBuffer& buffer) {
        return buffer.action_id == action_id;
      });
  if (action_buffer == state->buffers.end()) {
    throw py::value_error("incremental timing patch action/state identity mismatch");
  }

  const auto base_cache = patchBaseCache(base);
  const auto& base_pins = base_cache->pins;
  const auto& base_nets = base_cache->nets;
  const int pin_count = static_cast<int>(base_pins.size());
  const int net_count = static_cast<int>(base_nets.size());
  const int current_pin_count = pin_count + 2 * buffer_count;
  const int current_net_count = net_count + buffer_count;
  const auto inputPin = [&](const int slot_index) {
    return pin_count + 2 * slot_index;
  };
  const auto outputPin = [&](const int slot_index) {
    return pin_count + 2 * slot_index + 1;
  };
  const auto childNet = [&](const int slot_index) {
    return net_count + slot_index;
  };
  if (state->net_id < 0 || state->net_id >= net_count) {
    throw py::value_error("incremental timing patch net identity is out of range");
  }

  const auto selected = actionIds(action);
  std::unordered_set<int> selected_set;
  for (const int pin : selected) {
    if (pin < 0 || pin >= pin_count || !selected_set.insert(pin).second) {
      throw py::value_error("incremental timing patch has an invalid load pin");
    }
  }
  if (selected.empty()) {
    throw py::value_error("incremental timing patch requires a nonempty load set");
  }

  const auto& base_starts = base_cache->net_starts;
  const auto& base_arcs = base_cache->net_arcs;
  const auto& base_drivers = base_cache->net_drivers;
  requireSize(base_starts, static_cast<std::size_t>(net_count + 1), "net_flat_arcs_start");
  requireSize(base_drivers, static_cast<std::size_t>(net_count), "net2driver_pin_map");
  if (base_starts.back() != static_cast<int>(base_arcs.size())) {
    throw py::value_error("base timing net CSR is inconsistent");
  }
  const int driver = base_drivers[state->net_id];
  if (driver != state->driver_pin_id) {
    throw py::value_error("incremental timing patch driver identity mismatch");
  }

  std::vector<NetArc> net_arcs;
  std::vector<int> net_starts{0};
  std::vector<int> pin_net = base_cache->pin_net;
  requireSize(pin_net, static_cast<std::size_t>(pin_count), "pin2net_map");

  std::unordered_map<int, std::vector<int>> children;
  for (const auto& edge : state->edges) {
    children[edge.parent].push_back(edge.child);
  }
  std::unordered_map<int, int> output_slot;
  std::unordered_map<int, int> input_slot;
  for (int slot_index = 0; slot_index < buffer_count; ++slot_index) {
    const auto& buffer = state->buffers[slot_index];
    if (!output_slot.emplace(buffer.output_node, slot_index).second
        || !input_slot.emplace(buffer.input_node, slot_index).second) {
      throw py::value_error("incremental timing patch has duplicate buffer nodes");
    }
  }

  // Assign every physical RC node to its electrical partition.  The edge
  // input->output of a buffer is a logical cell edge; the output node starts
  // the next child net.  This makes a same-net chain a forest of B+1 timing
  // nets instead of pretending that all buffers share one RC slot.
  std::unordered_map<int, int> node_partition;
  std::vector<std::pair<int, int>> pending_nodes{{state->driver_pin_id, 0}};
  while (!pending_nodes.empty()) {
    const auto [node, incoming_partition] = pending_nodes.back();
    pending_nodes.pop_back();
    int partition = incoming_partition;
    const auto output = output_slot.find(node);
    if (output != output_slot.end()) {
      partition = output->second + 1;
    }
    const auto found = node_partition.find(node);
    if (found != node_partition.end() && found->second != partition) {
      throw py::value_error("incremental timing patch node has multiple partitions");
    }
    node_partition[node] = partition;
    for (const int child : children[node]) {
      pending_nodes.emplace_back(child, partition);
    }
  }
  if (node_partition.size() != state->coordinates.size()) {
    throw py::value_error("incremental timing patch topology is disconnected");
  }

  std::vector<std::vector<int>> partition_loads(buffer_count + 1);
  for (const auto& [node, loads] : state->node_loads) {
    const auto partition = node_partition.find(node);
    if (partition == node_partition.end()) {
      throw py::value_error("incremental timing patch load node is disconnected");
    }
    auto& target = partition_loads.at(partition->second);
    target.insert(target.end(), loads.begin(), loads.end());
  }
  for (int slot_index = 0; slot_index < buffer_count; ++slot_index) {
    const auto& buffer = state->buffers[slot_index];
    const auto partition = node_partition.find(buffer.input_node);
    if (partition == node_partition.end()) {
      throw py::value_error("incremental timing patch buffer input is disconnected");
    }
    partition_loads.at(partition->second).push_back(inputPin(slot_index));
  }
  for (auto& loads : partition_loads) {
    std::sort(loads.begin(), loads.end());
    loads.erase(std::unique(loads.begin(), loads.end()), loads.end());
  }

  for (int net = 0; net < net_count; ++net) {
    const int begin = base_starts[net];
    const int end = base_starts[net + 1];
    if (net != state->net_id) {
      net_arcs.insert(net_arcs.end(), base_arcs.begin() + begin, base_arcs.begin() + end);
    } else {
      for (const int load : partition_loads.front()) {
        if (load < pin_count) {
          net_arcs.push_back({driver, load});
        } else {
          net_arcs.push_back({driver, load});
        }
      }
    }
    net_starts.push_back(static_cast<int>(net_arcs.size()));
  }
  for (int slot_index = 0; slot_index < buffer_count; ++slot_index) {
    for (const int load : partition_loads[slot_index + 1]) {
      net_arcs.push_back({outputPin(slot_index), load});
    }
    net_starts.push_back(static_cast<int>(net_arcs.size()));
  }

  for (int partition = 0; partition <= buffer_count; ++partition) {
    const int net = partition == 0 ? state->net_id : childNet(partition - 1);
    for (const int load : partition_loads[partition]) {
      if (load < pin_count) {
        pin_net[load] = net;
      }
    }
  }
  for (int slot_index = 0; slot_index < buffer_count; ++slot_index) {
    pin_net.push_back(
        node_partition.at(state->buffers[slot_index].input_node) == 0
            ? state->net_id
            : childNet(node_partition.at(state->buffers[slot_index].input_node) - 1));
    pin_net.push_back(childNet(slot_index));
  }
  std::vector<int> drivers = base_drivers;
  for (int slot_index = 0; slot_index < buffer_count; ++slot_index) {
    drivers.push_back(outputPin(slot_index));
  }
  std::set<int> changed_pin_set{driver};
  for (int index = base_starts[state->net_id]; index < base_starts[state->net_id + 1]; ++index) {
    changed_pin_set.insert(base_arcs[index][0]);
    changed_pin_set.insert(base_arcs[index][1]);
  }
  for (int slot_index = 0; slot_index < buffer_count; ++slot_index) {
    changed_pin_set.insert(inputPin(slot_index));
    changed_pin_set.insert(outputPin(slot_index));
  }
  std::vector<int> changed_pins(changed_pin_set.begin(), changed_pin_set.end());

  const auto& cell_names = base_cache->cell_names;
  const auto& cell_arc_starts = base_cache->cell_arc_starts;
  const auto lib_arcs = base.attr("flat_libarc_info").cast<py::sequence>();
  const auto lib_pin_starts = carrierArray<int>(base, "cell_id_2_libpin_id_start");
  const auto lib_pin_names = carrierArray<std::string>(base, "flat_lib_pin_names");
  const auto lib_caps = carrierArray<double>(base, "flat_lib_pin_cap");
  const auto lib_rcaps = carrierArray<double>(base, "flat_lib_pin_rcap");
  const auto lib_fcaps = carrierArray<double>(base, "flat_lib_pin_fcap");
  const auto lib_cap_limits = carrierArray<double>(base, "flat_lib_pin_cap_limit");
  const auto lib_slew_limits = carrierArray<double>(base, "flat_lib_pin_slew_limit");
  auto pinCaps = [&](const int cell, const std::string& port) {
    int pin = -1;
    for (int index = lib_pin_starts.at(cell); index < lib_pin_starts.at(cell + 1); ++index) {
      if (lib_pin_names.at(index) == port) {
        pin = index;
        break;
      }
    }
    return std::vector<double>{
        pin >= 0 && pin < static_cast<int>(lib_caps.size()) ? lib_caps[pin] : 0.0,
        pin >= 0 && pin < static_cast<int>(lib_rcaps.size()) ? lib_rcaps[pin] : 0.0,
        pin >= 0 && pin < static_cast<int>(lib_fcaps.size()) ? lib_fcaps[pin] : 0.0};
  };
  auto pinLimits = [&](const int cell, const std::string& port) {
    int pin = -1;
    for (int index = lib_pin_starts.at(cell); index < lib_pin_starts.at(cell + 1); ++index) {
      if (lib_pin_names.at(index) == port) {
        pin = index;
        break;
      }
    }
    return std::vector<double>{
        pin >= 0 && pin < static_cast<int>(lib_slew_limits.size())
            ? lib_slew_limits[pin]
            : 0.0,
        pin >= 0 && pin < static_cast<int>(lib_cap_limits.size())
            ? lib_cap_limits[pin]
            : 0.0};
  };
  std::vector<BufferSlotLib> slot_libs;
  slot_libs.reserve(buffer_count);
  for (int slot_index = 0; slot_index < buffer_count; ++slot_index) {
    const auto& buffer = state->buffers[slot_index];
    std::string master = buffer.master_name;
    if (master.empty() && buffer.action_id == action_id
        && action.contains("buffer_master_name")) {
      master = py::cast<std::string>(action["buffer_master_name"]);
    }
    const auto cell_it = std::find(cell_names.begin(), cell_names.end(), master);
    if (cell_it == cell_names.end()) {
      throw py::value_error("incremental timing patch buffer master is absent from Liberty");
    }
    const int cell = static_cast<int>(cell_it - cell_names.begin());
    if (cell + 1 >= static_cast<int>(cell_arc_starts.size())) {
      throw py::value_error("incremental timing patch cell arc CSR is invalid");
    }
    const int arc_begin = cell_arc_starts[cell];
    const int arc_end = cell_arc_starts[cell + 1];
    if (arc_begin >= arc_end || arc_end > static_cast<int>(lib_arcs.size())) {
      throw py::value_error("incremental timing patch buffer has no timing arc");
    }
    const auto arc_info = lib_arcs[arc_begin].cast<py::sequence>();
    BufferSlotLib info;
    info.cell = cell;
    info.arc_begin = arc_begin;
    info.input_port = py::cast<std::string>(arc_info[0]);
    info.output_port = py::cast<std::string>(arc_info[1]);
    info.sense = py::cast<int>(arc_info[4]);
    info.trigger = py::cast<int>(arc_info[5]);
    if (info.sense != 1) {
      throw py::value_error("incremental timing patch requires a positive-unate buffer");
    }
    info.input_caps = pinCaps(cell, info.input_port);
    info.output_caps = pinCaps(cell, info.output_port);
    info.input_limits = pinLimits(cell, info.input_port);
    info.output_limits = pinLimits(cell, info.output_port);
    slot_libs.push_back(std::move(info));
  }

  const auto& base_rows = base_cache->timing_rows;
  const auto& base_level_starts = base_cache->timing_level_starts;
  if (base_level_starts.size() < 2 || base_level_starts.front() != 0
      || base_level_starts.back() != static_cast<int>(base_rows.size())) {
    throw py::value_error("base timing arc schedule is inconsistent");
  }
  const int clock_end = base_level_starts[1];
  // Reconstruct the weighted ranks of the current graph. Net arcs contribute
  // zero levels and cell arcs contribute one level. This schedule is shared
  // by every synthetic buffer arc in the multi-slot graph.
  std::vector<std::vector<std::pair<int, int>>> weighted(current_pin_count);
  std::vector<int> indegree(current_pin_count, 0);
  for (const auto& arc : net_arcs) {
    weighted[arc[0]].push_back({arc[1], 0});
    ++indegree[arc[1]];
  }
  std::vector<CellArc> unscheduled = base_rows;
  for (int slot_index = 0; slot_index < buffer_count; ++slot_index) {
    const auto& info = slot_libs.at(slot_index);
    unscheduled.push_back({inputPin(slot_index),
                           outputPin(slot_index),
                           info.cell,
                           info.arc_begin,
                           info.sense,
                           info.trigger,
                           0,
                           base_vertex_count + slot_index});
  }
  for (int id = clock_end; id < static_cast<int>(unscheduled.size()); ++id) {
    weighted[unscheduled[id][0]].push_back({unscheduled[id][1], 1});
    ++indegree[unscheduled[id][1]];
  }
  std::vector<int> current_rank(current_pin_count, 0);
  std::queue<int> ready;
  for (int pin = 0; pin < current_pin_count; ++pin) {
    if (indegree[pin] == 0) ready.push(pin);
  }
  int visited = 0;
  while (!ready.empty()) {
    const int pin = ready.front();
    ready.pop();
    ++visited;
    for (const auto& [next, weight] : weighted[pin]) {
      current_rank[next] = std::max(current_rank[next], current_rank[pin] + weight);
      if (--indegree[next] == 0) ready.push(next);
    }
  }
  if (visited != current_pin_count) {
    throw py::value_error("incremental timing patch graph is not a DAG");
  }
  int level_count = 1;
  for (int id = 0; id < static_cast<int>(unscheduled.size()); ++id) {
    const int level = id < clock_end ? 0 : current_rank[unscheduled[id][0]] + 1;
    level_count = std::max(level_count, level + 1);
  }
  std::vector<std::vector<CellArc>> rows(level_count);
  for (int id = 0; id < static_cast<int>(unscheduled.size()); ++id) {
    const int level = id < clock_end ? 0 : current_rank[unscheduled[id][0]] + 1;
    rows[level].push_back(unscheduled[id]);
  }
  std::vector<CellArc> scheduled;
  std::vector<int> scheduled_starts{0};
  for (const auto& row : rows) {
    scheduled.insert(scheduled.end(), row.begin(), row.end());
    scheduled_starts.push_back(static_cast<int>(scheduled.size()));
  }

  // The persistent timing slot replaces the graph topology in place.  Keep
  // the compact predecessor CSR in lockstep with the replacement graph so
  // the native critical-path extractor can traverse newly inserted buffer
  // pins after the first committed action.  The base CSR cannot be reused:
  // it has neither the synthetic pins nor the new buffer cell arcs.
  std::vector<std::vector<std::pair<int, int>>> predecessors(current_pin_count);
  for (const auto& net_arc : net_arcs) {
    predecessors.at(net_arc[1]).emplace_back(net_arc[0], -1);
  }
  for (int id = 0; id < static_cast<int>(scheduled.size()); ++id) {
    const auto& row = scheduled[id];
    const int source_pin = id < clock_end
                               ? base_cache->clock_pins.at(row[0])
                               : row[0];
    predecessors.at(row[1]).emplace_back(source_pin, id);
  }
  std::vector<int> predecessor_start{0};
  std::vector<int> predecessor_pins;
  std::vector<int> predecessor_arc_ids;
  for (const auto& entries : predecessors) {
    for (const auto& [pin, arc_id] : entries) {
      predecessor_pins.push_back(pin);
      predecessor_arc_ids.push_back(arc_id);
    }
    predecessor_start.push_back(static_cast<int>(predecessor_pins.size()));
  }

  // Build the native RC forest for the affected net.  The zero-RC cell edge
  // is omitted: its input/output relation is represented by the timing cell
  // arcs above. All original Steiner ids are shifted by the synthetic pins.
  const int slot_vertex_count = base_vertex_count + 2 * buffer_count;
  const auto mapNode = [&](const int node) {
    const auto input = input_slot.find(node);
    if (input != input_slot.end()) return inputPin(input->second);
    const auto output = output_slot.find(node);
    if (output != output_slot.end()) return outputPin(output->second);
    if (node >= 0 && node < pin_count) return node;
    if (node >= pin_count && node < base_vertex_count) {
      return node + 2 * buffer_count;
    }
    throw py::value_error("incremental timing patch RC node is out of range");
  };
  std::map<int, std::vector<int>> rc_children;
  std::vector<int> edge_parent, edge_child;
  std::vector<double> edge_r, edge_c;
  for (const auto& edge : state->edges) {
    const auto input = input_slot.find(edge.parent);
    const auto output = output_slot.find(edge.child);
    if (input != input_slot.end() && output != output_slot.end()
        && input->second == output->second) {
      continue;
    }
    const int parent = mapNode(edge.parent);
    const int child = mapNode(edge.child);
    rc_children[parent].push_back(child);
    edge_parent.push_back(parent);
    edge_child.push_back(child);
    edge_r.push_back(edge.resistance);
    edge_c.push_back(edge.capacitance);
  }
  std::vector<int> parent(slot_vertex_count, -1);
  std::vector<std::vector<int>> topologies;
  std::unordered_set<int> visited_nodes;
  std::vector<int> roots{mapNode(state->driver_pin_id)};
  for (int slot_index = 0; slot_index < buffer_count; ++slot_index) {
    roots.push_back(outputPin(slot_index));
  }
  for (const int root : roots) {
    if (visited_nodes.find(root) != visited_nodes.end()) continue;
    parent[root] = root;
    std::vector<int> topology;
    std::vector<int> stack{root};
    while (!stack.empty()) {
      const int node = stack.back();
      stack.pop_back();
      if (!visited_nodes.insert(node).second) continue;
      topology.push_back(node);
    const auto found = rc_children.find(node);
    if (found == rc_children.end()) continue;
      for (auto it = found->second.rbegin(); it != found->second.rend(); ++it) {
        parent[*it] = node;
        stack.push_back(*it);
      }
    }
    topologies.push_back(std::move(topology));
  }
  if (visited_nodes.size() != state->coordinates.size()) {
    // The state contains one logical zero-RC edge whose endpoints are both
    // visited as the two partition roots.  All other nodes must be in a wire
    // forest; this check catches accidental disconnected exports.
    throw py::value_error("incremental timing patch RC forest is disconnected");
  }

  std::vector<int> child_start{0}, flat_child;
  for (int node = 0; node < slot_vertex_count; ++node) {
    const auto found = rc_children.find(node);
    if (found != rc_children.end()) flat_child.insert(flat_child.end(), found->second.begin(), found->second.end());
    child_start.push_back(static_cast<int>(flat_child.size()));
  }
  std::vector<int> topo_start{0}, flat_topo;
  for (const auto& topology : topologies) {
    flat_topo.insert(flat_topo.end(), topology.begin(), topology.end());
    topo_start.push_back(static_cast<int>(flat_topo.size()));
  }
  std::vector<double> edge_resistance(slot_vertex_count, 0.0), edge_capacitance(slot_vertex_count, 0.0);
  for (std::size_t index = 0; index < edge_child.size(); ++index) {
    edge_resistance[edge_child[index]] = edge_r[index];
    edge_capacitance[edge_child[index]] = edge_c[index];
  }
  std::vector<int> affected_nodes(visited_nodes.begin(), visited_nodes.end());
  std::sort(affected_nodes.begin(), affected_nodes.end());

  // The persistent Python slot only needs numeric graph arrays.  Returning
  // the old nested Python lists here caused every candidate to allocate and
  // destroy tens of thousands of Python objects.  The native affected
  // schedule below also means this compact graph never needs pin-name lookup
  // on the Python side.
  py::dict graph;
  graph["num_pins"] = current_pin_count;
  graph["pin2net_map"] = intArray(pin_net);
  graph["net_flat_arcs_start"] = intArray(net_starts);
  graph["net_flat_arcs"] = intArray2(net_arcs);
  graph["net2driver_pin_map"] = intArray(drivers);
  graph["flat_inst_arcs_by_level"] = intArray2(scheduled);
  graph["flat_inst_arcs_by_level_start"] = intArray(scheduled_starts);
  graph["pin_pred_start"] = intArray(predecessor_start);
  graph["pin_pred_pin"] = intArray(predecessor_pins);
  graph["pin_pred_arc_id"] = intArray(predecessor_arc_ids);

  py::dict rc;
  rc["parent"] = parent;
  rc["child_start"] = child_start;
  rc["child"] = flat_child;
  rc["topo_start"] = topo_start;
  rc["topo"] = flat_topo;
  rc["edge_resistance"] = edge_resistance;
  rc["edge_capacitance"] = edge_capacitance;
  rc["affected_nodes"] = affected_nodes;

  py::dict output;
  output["status"] = "ok";
  output["graph"] = graph;
  output["rc"] = rc;
  std::vector<int> slot_pin_ids;
  slot_pin_ids.reserve(2 * buffer_count);
  for (int slot_index = 0; slot_index < buffer_count; ++slot_index) {
    slot_pin_ids.push_back(inputPin(slot_index));
    slot_pin_ids.push_back(outputPin(slot_index));
  }
  output["slot_pin_ids"] = slot_pin_ids;
  output["new_net_id"] = childNet(0);
  output["new_net_ids"] = [&]() {
    std::vector<int> values;
    for (int slot_index = 0; slot_index < buffer_count; ++slot_index) {
      values.push_back(childNet(slot_index));
    }
    return values;
  }();
  output["driver_pin_id"] = driver;
  output["buffer_count"] = buffer_count;
  output["slot_vertex_count"] = slot_vertex_count;
  output["changed_pin_ids"] = changed_pins;
  output["schedule"] = buildPatchSchedule(
      *base_cache,
      current_pin_count,
      scheduled,
      scheduled_starts,
      net_arcs,
      changed_pins,
      base.attr("end_points"),
      base.attr("start_points"),
      base.attr("endpoints_constraint_arcs"),
      base.attr("endpoints_timing_check_arcs"));
  py::dict buffer_caps;
  py::list slots;
  for (const auto& info : slot_libs) {
    py::dict slot_caps;
    slot_caps["input"] = info.input_caps;
    slot_caps["output"] = info.output_caps;
    slot_caps["input_limits"] = info.input_limits;
    slot_caps["output_limits"] = info.output_limits;
    slots.append(slot_caps);
  }
  buffer_caps["slots"] = slots;
  // Keep the old shape for one-slot callers and fixtures.
  if (buffer_count == 1) {
    buffer_caps["input"] = slot_libs.front().input_caps;
    buffer_caps["output"] = slot_libs.front().output_caps;
  }
  output["buffer_caps"] = buffer_caps;
  return output;
}

py::dict evaluateIncrementalRc(
    const py::dict& rc,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& generic_caps,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& rise_caps,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& fall_caps,
    const py::dict& buffer_caps,
    const int pin_count)
{
  const auto global_parent = rc["parent"].cast<std::vector<int>>();
  const auto global_child_start = rc["child_start"].cast<std::vector<int>>();
  const auto global_children = rc["child"].cast<std::vector<int>>();
  const auto global_topo_start = rc["topo_start"].cast<std::vector<int>>();
  const auto global_topo = rc["topo"].cast<std::vector<int>>();
  const auto global_resistance = rc["edge_resistance"].cast<std::vector<double>>();
  const auto global_edge_cap = rc["edge_capacitance"].cast<std::vector<double>>();
  const auto affected = rc["affected_nodes"].cast<std::vector<int>>();
  std::unordered_map<int, int> local;
  local.reserve(affected.size());
  for (int index = 0; index < static_cast<int>(affected.size()); ++index) {
    local.emplace(affected[index], index);
  }
  std::vector<int> parent(affected.size(), -1);
  std::vector<std::vector<int>> children(affected.size());
  for (int index = 0; index < static_cast<int>(affected.size()); ++index) {
    parent[index] = local.at(global_parent[affected[index]]);
    const int begin = global_child_start[affected[index]];
    const int end = global_child_start[affected[index] + 1];
    for (int offset = begin; offset < end; ++offset) {
      children[index].push_back(local.at(global_children[offset]));
    }
  }
  std::vector<std::vector<int>> topologies;
  for (int index = 0; index + 1 < static_cast<int>(global_topo_start.size()); ++index) {
    std::vector<int> topology;
    for (int offset = global_topo_start[index]; offset < global_topo_start[index + 1]; ++offset) {
      topology.push_back(local.at(global_topo[offset]));
    }
    topologies.push_back(std::move(topology));
  }
  std::vector<double> wire_cap(affected.size(), 0.0);
  for (int index = 0; index < static_cast<int>(affected.size()); ++index) {
    const int global_node = affected[index];
    if (global_parent[global_node] == global_node) continue;
    const double half = global_edge_cap[global_node] * 0.5;
    wire_cap[index] += half;
    wire_cap[parent[index]] += half;
  }
  auto modeCaps = [&](const py::array_t<double, py::array::c_style | py::array::forcecast>& base,
                      const char* mode) {
    std::vector<double> caps(affected.size(), 0.0);
    const auto values = base.unchecked<1>();
    std::vector<std::vector<double>> slot_inputs;
    std::vector<std::vector<double>> slot_outputs;
    if (buffer_caps.contains("slots")) {
      for (const auto& item : buffer_caps["slots"].cast<py::sequence>()) {
        const auto slot = item.cast<py::dict>();
        slot_inputs.push_back(slot["input"].cast<std::vector<double>>());
        slot_outputs.push_back(slot["output"].cast<std::vector<double>>());
      }
    } else {
      slot_inputs.push_back(buffer_caps["input"].cast<std::vector<double>>());
      slot_outputs.push_back(buffer_caps["output"].cast<std::vector<double>>());
    }
    const int mode_index = std::string(mode) == "generic" ? 0 : std::string(mode) == "rise" ? 1 : 2;
    for (int index = 0; index < static_cast<int>(affected.size()); ++index) {
      const int node = affected[index];
      if (node < pin_count && node < values.shape(0)) caps[index] = values(node);
      else if (node >= pin_count) {
        const int offset = node - pin_count;
        const int slot_index = offset / 2;
        if (slot_index >= 0 && slot_index < static_cast<int>(slot_inputs.size())) {
          const auto& lane = (offset % 2 == 0)
                                 ? slot_inputs.at(slot_index)
                                 : slot_outputs.at(slot_index);
          caps[index] = lane.at(mode_index);
        }
      }
      caps[index] += wire_cap[index];
    }
    return caps;
  };
  auto evaluateMode = [&](const std::vector<double>& caps) {
    const int count = affected.size();
    std::vector<double> load(count), delay(count), ldelay(count), beta(count), impulse(count);
    for (const auto& topology : topologies) {
      for (auto it = topology.rbegin(); it != topology.rend(); ++it) {
        const int node = *it;
        double value = caps[node];
        for (const int child : children[node]) value += load[child];
        load[node] = value;
      }
      for (const int node : topology) {
        if (parent[node] == node) {
          delay[node] = 0.0;
          continue;
        }
        delay[node] = delay[parent[node]] + global_resistance[affected[node]] * load[node];
      }
      for (auto it = topology.rbegin(); it != topology.rend(); ++it) {
        const int node = *it;
        double value = caps[node] * delay[node];
        for (const int child : children[node]) value += ldelay[child];
        ldelay[node] = value;
      }
      for (const int node : topology) {
        if (parent[node] == node) {
          beta[node] = 0.0;
        } else {
          beta[node] = beta[parent[node]] + global_resistance[affected[node]] * ldelay[node];
        }
        impulse[node] = 2.0 * beta[node] - delay[node] * delay[node];
      }
    }
    return std::array<std::vector<double>, 3>{load, delay, impulse};
  };
  const auto generic = evaluateMode(modeCaps(generic_caps, "generic"));
  const auto rise = evaluateMode(modeCaps(rise_caps, "rise"));
  const auto fall = evaluateMode(modeCaps(fall_caps, "fall"));
  py::dict output;
  output["affected_nodes"] = intArray(affected);
  output["loads_generic"] = generic[0];
  output["delays_generic"] = generic[1];
  output["impulses_generic"] = generic[2];
  output["loads_rise"] = rise[0];
  output["delays_rise"] = rise[1];
  output["impulses_rise"] = rise[2];
  output["loads_fall"] = fall[0];
  output["delays_fall"] = fall[1];
  output["impulses_fall"] = fall[2];
  return output;
}

}  // namespace rebuffer
