// SPDX-License-Identifier: BSD-3-Clause
// Incremental graph carrier rebuild. Timing evaluation remains Python
// TimingPropagation; this file only owns graph/index/CSR transport.
#include "rebuffer_carrier.hh"

#include "rebuffer_incremental_timing.hh"


#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "pybind11/stl.h"


namespace py = pybind11;
namespace rebuffer {
namespace {

using Arc = TimingCellArc;
using NetArc = TimingNetArc;

template <typename T>
void exportRows(py::dict& out, const char* rows_name, const char* flat_name,
                const char* start_name, const std::vector<std::vector<T>>& rows)
{
  std::vector<T> flat;
  std::vector<int> starts{0};
  for (const auto& row : rows) {
    flat.insert(flat.end(), row.begin(), row.end());
    starts.push_back(flat.size());
  }
  out[rows_name] = rows;
  out[flat_name] = flat;
  out[start_name] = starts;
}

template <typename T>
void exportAdjacency(py::dict& out, const char* start_name, const char* flat_name,
                     const std::vector<std::vector<T>>& rows)
{
  std::vector<T> flat;
  std::vector<int> starts{0};
  for (const auto& row : rows) {
    flat.insert(flat.end(), row.begin(), row.end());
    starts.push_back(flat.size());
  }
  out[start_name] = starts;
  out[flat_name] = flat;
}

template <typename T>
std::unordered_map<T, int> indexValues(const std::vector<T>& values, const char* what)
{
  std::unordered_map<T, int> result;
  result.reserve(values.size());
  for (int index = 0; index < static_cast<int>(values.size()); ++index) {
    if (!result.emplace(values[index], index).second) {
      throw py::value_error(std::string("duplicate ") + what);
    }
  }
  return result;
}

int remapPin(int pin, const std::vector<int>& old_to_current, const char* field)
{
  if (pin < 0 || pin >= static_cast<int>(old_to_current.size()) || old_to_current[pin] < 0) {
    throw py::value_error(std::string("removed timing pin in ") + field);
  }
  return old_to_current[pin];
}

std::vector<int> remapPins(const py::object& previous, const char* field,
                           const std::vector<int>& old_to_current)
{
  std::vector<int> values = carrierArray<int>(previous, field);
  for (int& value : values) value = remapPin(value, old_to_current, field);
  return values;
}

void copyPhysical(py::dict& out, const py::object& physical)
{
  for (const char* field : {
           "dbu", "num_nodes", "num_terminals", "num_terminal_NIs", "node_names",
           "node_name2id_map", "node_master_names", "node_x", "node_y", "node_size_x",
           "node_size_y", "node_orient", "node2orig_node_map", "node_is_hard_macro",
           "macro_writeback_candidate", "clock_net_names", "xl", "yl", "xh", "yh",
           "row_height", "site_width", "rows", "regions", "flat_region_boxes",
           "flat_region_boxes_start", "node2fence_region_map", "m2_pg_rail_blockage_rects",
           "m2_pg_rail_boxes", "m2_pg_rail_density_boxes", "total_fixed_node_area",
           "total_space_area", "routing_grid_xl", "routing_grid_yl", "routing_grid_xh",
           "routing_grid_yh", "num_routing_grids_x", "num_routing_grids_y",
           "unit_horizontal_capacities", "unit_vertical_capacities", "min_wire_widths",
           "min_wire_spacings", "initial_horizontal_demand_map", "initial_vertical_demand_map",
           "getCongestionMap", "write_macro_placement_back", "pin_names", "pin2node_map",
           "pin_offset_x", "pin_offset_y", "pin_direct", "net_names", "net2pin_map",
           "flat_net2pin_map", "flat_net2pin_start_map"}) {
    if (py::hasattr(physical, field)) out[field] = physical.attr(field);
  }
}

}  // namespace

py::dict rebuildTimingCarrierIncremental(
    const py::object& base, const py::object& previous, const py::object& physical)
{
  py::dict out;
  copyPhysical(out, physical);

  const auto old_pins = carrierArray<std::string>(previous, "pin_names");
  const auto current_pins = carrierArray<std::string>(physical, "pin_names");
  const auto old_nodes = carrierArray<std::string>(previous, "node_names");
  const auto current_nodes = carrierArray<std::string>(physical, "node_names");
  const auto current_nets = carrierArray<std::string>(physical, "net_names");
  const auto old_pin_ids = indexValues(old_pins, "previous timing pin");
  const auto current_pin_ids = indexValues(current_pins, "current timing pin");
  const auto old_node_ids = indexValues(old_nodes, "previous physical node");
  const auto current_node_ids = indexValues(current_nodes, "current physical node");
  std::vector<int> old_to_current(old_pins.size(), -1);
  for (int id = 0; id < static_cast<int>(old_pins.size()); ++id) {
    const auto found = current_pin_ids.find(old_pins[id]);
    if (found != current_pin_ids.end()) old_to_current[id] = found->second;
  }
  std::vector<int> old_node_to_current(old_nodes.size(), -1);
  for (int id = 0; id < static_cast<int>(old_nodes.size()); ++id) {
    const auto found = current_node_ids.find(old_nodes[id]);
    if (found != current_node_ids.end()) old_node_to_current[id] = found->second;
  }

  const auto previous_arcs = carrierArray<Arc>(previous, "flat_inst_arcs_by_level");
  const auto previous_levels = carrierArray<int>(previous, "flat_inst_arcs_by_level_start");
  if (previous_levels.size() < 2 || previous_levels[1] > static_cast<int>(previous_arcs.size())) {
    throw py::value_error("previous timing carrier has invalid level offsets");
  }
  const int clock_end = previous_levels[1];
  std::vector<Arc> arcs;
  arcs.reserve(previous_arcs.size() + current_nodes.size() - old_nodes.size() + 4);
  for (int id = 0; id < static_cast<int>(previous_arcs.size()); ++id) {
    Arc row = previous_arcs[id];
    if (row[7] < 0 || row[7] >= static_cast<int>(old_node_to_current.size())) continue;
    if (old_node_to_current[row[7]] < 0) continue;
    row[7] = old_node_to_current[row[7]];
    row[1] = remapPin(row[1], old_to_current, "previous cell arc sink");
    if (id >= clock_end) row[0] = remapPin(row[0], old_to_current, "previous cell arc source");
    arcs.push_back(row);
  }

  const auto lib_names = carrierArray<std::string>(base, "flat_libcell_names");
  const auto lib_ids = indexValues(lib_names, "Liberty cell");
  const auto main_start = carrierArray<int>(base, "main_id_2_cell_id_start");
  const auto cell_arc_start = carrierArray<int>(base, "cell_id_2_arc_id_start");
  const auto lib_arcs = base.attr("flat_libarc_info").cast<py::sequence>();
  const auto masters = carrierArray<std::string>(physical, "node_master_names");
  const auto pin_ids = current_pin_ids;
  for (int node = 0; node < static_cast<int>(current_nodes.size()); ++node) {
    if (old_node_ids.count(current_nodes[node])) continue;
    const auto cell_found = lib_ids.find(masters.at(node));
    if (cell_found == lib_ids.end()) throw py::value_error("new timing instance is not a Liberty cell");
    const int cell = cell_found->second;
    const std::string prefix = current_nodes[node] + ":";
    int added = 0;
    for (int lut = cell_arc_start.at(cell); lut < cell_arc_start.at(cell + 1); ++lut) {
      const auto info = lib_arcs[lut].cast<py::sequence>();
      const auto source = pin_ids.find(prefix + py::cast<std::string>(info[0]));
      const auto sink = pin_ids.find(prefix + py::cast<std::string>(info[1]));
      if (source == pin_ids.end() || sink == pin_ids.end()) {
        throw py::value_error("new timing instance pin is absent from physical carrier");
      }
      arcs.push_back({source->second, sink->second, cell, lut,
                      py::cast<int>(info[4]), py::cast<int>(info[5]),
                      lut - cell_arc_start[cell], node});
      ++added;
    }
    if (!added) throw py::value_error("new timing instance has no Liberty arc");
  }

  std::vector<int> clock_pins = remapPins(previous, "clock_pins", old_to_current);
  std::vector<int> starts = remapPins(previous, "start_points", old_to_current);
  std::vector<int> endpoints = remapPins(previous, "end_points", old_to_current);
  out["clock_pins"] = clock_pins;
  out["FF_ids"] = remapPins(previous, "FF_ids", old_to_current);
  out["start_points"] = starts; out["start_pin_ids"] = starts;
  out["end_points"] = endpoints; out["endpoint_pin_ids"] = endpoints;

  for (const char* field : {"endpoints_constraint_arcs", "endpoints_timing_check_arcs"}) {
    const auto rows = previous.attr(field).cast<std::vector<Arc>>();
    const bool clock_local = std::string(field) == "endpoints_constraint_arcs";
    std::vector<Arc> remapped = rows;
    for (auto& row : remapped) {
      if (!clock_local) row[0] = remapPin(row[0], old_to_current, field);
      row[1] = remapPin(row[1], old_to_current, field);
    }
    out[field] = remapped;
  }

  const auto physical_start = carrierArray<int>(physical, "flat_net2pin_start_map");
  const auto physical_flat = carrierArray<int>(physical, "flat_net2pin_map");
  if (physical_start.size() != current_nets.size() + 1 || physical_start.front() != 0
      || physical_start.back() != static_cast<int>(physical_flat.size())) {
    throw py::value_error("current physical net CSR is invalid");
  }
  std::vector<int> drivers, net_starts{0};
  std::vector<NetArc> net_arcs;
  for (int net = 0; net < static_cast<int>(current_nets.size()); ++net) {
    const int begin = physical_start[net], end = physical_start[net + 1];
    const int driver = begin < end ? physical_flat[begin] : -1;
    drivers.push_back(driver);
    for (int index = begin + 1; index < end; ++index) {
      net_arcs.push_back({driver, physical_flat[index]});
    }
    net_starts.push_back(net_arcs.size());
  }
  for (auto& row : net_arcs) {
    row[0] = row[0] < static_cast<int>(current_pins.size()) ? row[0] : -1;
    row[1] = row[1] < static_cast<int>(current_pins.size()) ? row[1] : -1;
    if (row[0] < 0 || row[1] < 0) throw py::value_error("physical net pin is invalid");
  }
  out["net2driver_pin_map"] = drivers;
  out["net_flat_arcs_start"] = net_starts;
  out["net_flat_arcs"] = net_arcs;

  const int pin_count = current_pins.size();
  std::vector<std::vector<std::pair<int, int>>> successors(pin_count);
  std::vector<int> indegree(pin_count, 0), levels(pin_count, 0);
  std::vector<unsigned char> is_clock(arcs.size(), 0);
  auto edge = [&](int source, int sink, int weight) {
    successors.at(source).push_back({sink, weight});
    ++indegree.at(sink);
  };
  for (int id = 0; id < static_cast<int>(arcs.size()); ++id) {
    is_clock[id] = id < clock_end;
    if (!is_clock[id]) edge(arcs[id][0], arcs[id][1], 1);
  }
  for (const auto& row : net_arcs) edge(row[0], row[1], 0);
  std::priority_queue<int, std::vector<int>, std::greater<int>> ready;
  for (int pin = 0; pin < pin_count; ++pin) if (!indegree[pin]) ready.push(pin);
  int visited = 0;
  while (!ready.empty()) {
    const int pin = ready.top(); ready.pop(); ++visited;
    for (const auto& [next, weight] : successors[pin]) {
      levels[next] = std::max(levels[next], levels[pin] + weight);
      if (!--indegree[next]) ready.push(next);
    }
  }
  if (visited != pin_count) throw py::value_error("incremental timing graph contains a cycle");

  std::vector<std::vector<int>> by_level(1);
  for (int id = 0; id < static_cast<int>(arcs.size()); ++id) {
    const int level = is_clock[id] ? 0 : levels[arcs[id][0]] + 1;
    if (level >= static_cast<int>(by_level.size())) by_level.resize(level + 1);
    by_level[level].push_back(id);
  }
  std::vector<Arc> scheduled;
  std::vector<int> level_start{0}, topo_start{0}, topo_ids;
  std::vector<std::vector<int>> inst_arcs(current_nodes.size()), pred(pin_count), succ(pin_count);
  std::vector<std::vector<int>> pred_arcs(pin_count), succ_arcs(pin_count);
  std::vector<int> graph_src, graph_dst, graph_inst, graph_cell, graph_lut, graph_sense, graph_type, graph_offset;
  auto graphArc = [&](int source, int sink, int id, int owner, int cell, int lut, int sense, int type, int offset) {
    graph_src.push_back(source); graph_dst.push_back(sink); graph_inst.push_back(owner);
    graph_cell.push_back(cell); graph_lut.push_back(lut); graph_sense.push_back(sense);
    graph_type.push_back(type); graph_offset.push_back(offset);
    if (id != -2) {
      succ[source].push_back(sink); succ_arcs[source].push_back(id);
      pred[sink].push_back(source); pred_arcs[sink].push_back(id);
    }
  };
  for (const auto& ids : by_level) {
    std::unordered_set<int> seen;
    for (const int id : ids) {
      const auto& row = arcs[id];
      const int index = scheduled.size();
      scheduled.push_back(row);
      inst_arcs[row[7]].push_back(index);
      if (seen.insert(row[7]).second) topo_ids.push_back(row[7]);
      graphArc(is_clock[id] ? clock_pins.at(row[0]) : row[0], row[1],
               is_clock[id] ? -2 : index, row[7], row[2], row[3], row[4], row[5], row[6]);
    }
    level_start.push_back(scheduled.size());
    topo_start.push_back(topo_ids.size());
  }
  for (const auto& row : net_arcs) graphArc(row[0], row[1], -1, -1, -1, -1, 0, 0, -1);

  out["flat_inst_arcs_by_level"] = scheduled;
  out["flat_inst_arcs_by_level_start"] = level_start;
  out["timing_graph_index"] = makeTimingGraphIndex(current_pins, scheduled, net_arcs, clock_pins, level_start[1]);
  out["arc_level_start"] = level_start;
  out["inst_topo_start"] = topo_start;
  out["inst_topo_ids"] = topo_ids;
  exportAdjacency(out, "inst_flat_arcs_start", "inst_flat_arcs", inst_arcs);
  exportAdjacency(out, "pin_pred_start", "pin_pred_pin", pred);
  exportAdjacency(out, "pin_pred_start", "pin_pred_arc_id", pred_arcs);
  exportAdjacency(out, "pin_succ_start", "pin_succ_pin", succ);
  exportAdjacency(out, "pin_succ_start", "pin_succ_arc_id", succ_arcs);
  exportAdjacency(out, "flat_pin_to_graph_start", "flat_pin_to_graph", succ);
  exportAdjacency(out, "flat_pin_to_graph_start_reverse", "flat_pin_to_graph_reverse", pred);
  out["arc_src_pin"] = graph_src; out["arc_dst_pin"] = graph_dst;
  out["arc_inst_id"] = graph_inst; out["arc_libcell_id"] = graph_cell;
  out["arc_libarc_id"] = graph_lut; out["arc_sense"] = graph_sense;
  out["arc_type"] = graph_type; out["arc_offset"] = graph_offset;

  std::vector<int> pin_net(pin_count, -1);
  for (int net = 0; net < static_cast<int>(current_nets.size()); ++net) {
    for (int index = physical_start[net]; index < physical_start[net + 1]; ++index) {
      const int pin = physical_flat[index];
      if (pin < 0 || pin >= pin_count || pin_net[pin] >= 0) throw py::value_error("invalid physical pin net");
      pin_net[pin] = net;
    }
  }
  if (std::find(pin_net.begin(), pin_net.end(), -1) != pin_net.end()) {
    throw py::value_error("physical carrier contains an unconnected pin");
  }
  out["pin2net_map"] = pin_net;
  std::vector<double> net_weights(current_nets.size(), 1.0);
  const auto clock_names = carrierArray<std::string>(base, "clock_net_names");
  const auto clock_set = std::unordered_set<std::string>(clock_names.begin(), clock_names.end());
  for (int net = 0; net < static_cast<int>(current_nets.size()); ++net) {
    if (clock_set.count(current_nets[net])) net_weights[net] = 0.0;
  }
  out["net_weights"] = net_weights;

  const auto main_starts = main_start;
  const auto lib_pin_starts = carrierArray<int>(base, "cell_id_2_libpin_id_start");
  const auto lib_pin_names = carrierArray<std::string>(base, "flat_lib_pin_names");
  std::vector<int> inst_main(current_nodes.size(), -1), inst_offset(current_nodes.size(), 0);
  for (int node = 0; node < static_cast<int>(current_nodes.size()); ++node) {
    const auto found = lib_ids.find(masters.at(node));
    if (found == lib_ids.end()) continue;
    const int cell = found->second;
    const int main = std::upper_bound(main_starts.begin(), main_starts.end(), cell) - main_starts.begin() - 1;
    inst_main[node] = main; inst_offset[node] = cell - main_starts.at(main);
  }
  std::vector<int> pin_lib_offset(pin_count, -1);
  const auto pin2node = carrierArray<int>(physical, "pin2node_map");
  for (int pin = 0; pin < pin_count; ++pin) {
    const int node = pin2node.at(pin), main = inst_main.at(node);
    if (main < 0) continue;
    const int cell = main_starts.at(main) + inst_offset.at(node);
    const std::string prefix = current_nodes.at(node) + ":";
    const std::string name = current_pins.at(pin);
    if (name.rfind(prefix, 0) != 0) throw py::value_error("physical timing pin owner mismatch");
    const std::string port = name.substr(prefix.size());
    for (int id = lib_pin_starts.at(cell); id < lib_pin_starts.at(cell + 1); ++id) {
      if (lib_pin_names.at(id) == port) pin_lib_offset[pin] = id - lib_pin_starts[cell];
    }
    if (pin_lib_offset[pin] < 0) throw py::value_error("physical timing pin is absent from Liberty");
  }
  out["inst_main_id"] = inst_main;
  out["inst_libcell_offset"] = inst_offset;
  out["pin_2_libpin_offset"] = pin_lib_offset;
  out["pin_to_inst_id"] = pin2node;
  out["pin_to_node_id"] = pin2node;

  std::vector<std::vector<int>> node_pins(current_nodes.size());
  for (int pin = 0; pin < pin_count; ++pin) node_pins.at(pin2node[pin]).push_back(pin);
  exportRows(out, "node2pin_map", "flat_node2pin_map", "flat_node2pin_start_map", node_pins);
  out["num_pins"] = pin_count;
  out["num_nets"] = current_nets.size();
  out["pin_name2id_map"] = current_pin_ids;
  out["net_name2id_map"] = indexValues(current_nets, "current timing net");
  out["num_movable_pins"] = pin_count;
  out["timing_backend_snapshot_valid"] = false;
  out["timing_carrier_source"] = "incremental_native_graph_patch";
  out["ecc_sta_run"] = false;
  return out;
}

}  // namespace rebuffer

// --- rebuffer_incremental_rc.cc ---
#include "rebuffer_incremental_topology.hh"


#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace rebuffer {

CompactRcForest compactRcForest(const IncrementalTopologyState& state,
                                int original_pin_count)
{
  CompactRcForest result;
  auto& ids = result.node_ids;
  ids.push_back(state.driver_pin_id);
  for (const auto& edge : state.edges) {
    ids.push_back(edge.parent); ids.push_back(edge.child);
  }
  for (const auto& [node, _] : state.node_loads) ids.push_back(node);
  std::unordered_map<int, int> buffer_slots, buffer_outputs;
  for (int slot = 0; slot < static_cast<int>(state.buffers.size()); ++slot) {
    const auto& buffer = state.buffers[slot];
    ids.push_back(buffer.input_node); ids.push_back(buffer.output_node);
    if (!buffer_slots.emplace(buffer.input_node, 2 * slot).second
        || !buffer_slots.emplace(buffer.output_node, 2 * slot + 1).second) {
      throw std::invalid_argument("native timing RC node namespace collides");
    }
    buffer_outputs.emplace(buffer.input_node, buffer.output_node);
  }
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  if (ids.front() < 0) throw std::invalid_argument("persistent timing graph RC node is negative");
  // Preserve the former dense adapter's wire-cap accumulation order: original
  // physical pins, current buffer slots, then shifted Steiner identities.
  auto key = [&](int node) -> std::int64_t {
    const auto slot = buffer_slots.find(node);
    if (slot != buffer_slots.end()) return static_cast<std::int64_t>(original_pin_count) + slot->second;
    return node < original_pin_count ? node
        : static_cast<std::int64_t>(node) + 2 * state.buffers.size();
  };
  std::sort(ids.begin(), ids.end(), [&](int a, int b) { return key(a) < key(b); });
  for (std::size_t i = 1; i < ids.size(); ++i) {
    if (key(ids[i - 1]) == key(ids[i])) {
      throw std::invalid_argument("native timing RC node namespace collides");
    }
  }
  std::unordered_map<int, int> local;
  for (int i = 0; i < static_cast<int>(ids.size()); ++i) local.emplace(ids[i], i);
  result.parent.assign(ids.size(), -1);
  result.resistance.assign(ids.size(), 0);
  result.edge_capacitance.assign(ids.size(), 0);
  std::vector<std::vector<int>> children(ids.size());
  for (const auto& edge : state.edges) {
    const auto output = buffer_outputs.find(edge.parent);
    if (output != buffer_outputs.end() && output->second == edge.child) continue;
    const int source = local.at(edge.parent), sink = local.at(edge.child);
    if (result.parent[sink] >= 0 && result.parent[sink] != source) {
      throw std::invalid_argument("persistent timing graph RC node has multiple parents");
    }
    result.parent[sink] = source;
    children[source].push_back(sink);
    result.resistance[sink] = edge.resistance;
    result.edge_capacitance[sink] = edge.capacitance;
  }
  std::vector<int> roots{state.driver_pin_id};
  for (const auto& buffer : state.buffers) roots.push_back(buffer.output_node);
  std::vector<unsigned char> visited(ids.size(), 0);
  result.topo_start.push_back(0);
  for (int root_id : roots) {
    const int root = local.at(root_id);
    if (result.parent[root] >= 0 && result.parent[root] != root) {
      throw std::invalid_argument("persistent timing graph RC root has a parent");
    }
    result.parent[root] = root;
    if (visited[root] == 2) continue;
    std::vector<std::pair<int, bool>> pending{{root, false}};
    while (!pending.empty()) {
      const auto [node, leaving] = pending.back(); pending.pop_back();
      if (leaving) { visited[node] = 2; continue; }
      if (visited[node] == 2) continue;
      if (visited[node] == 1) throw std::invalid_argument("persistent timing graph RC forest has a cycle");
      visited[node] = 1;
      result.topo.push_back(node);
      pending.emplace_back(node, true);
      for (auto it = children[node].rbegin(); it != children[node].rend(); ++it) {
        if (visited[*it] == 1) throw std::invalid_argument("persistent timing graph RC forest has a cycle");
        pending.emplace_back(*it, false);
      }
    }
    result.topo_start.push_back(result.topo.size());
  }
  if (std::find(visited.begin(), visited.end(), 0) != visited.end()) {
    throw std::invalid_argument("persistent timing graph RC forest is disconnected");
  }
  result.child_start.push_back(0);
  for (const auto& row : children) {
    result.children.insert(result.children.end(), row.begin(), row.end());
    result.child_start.push_back(result.children.size());
  }
  return result;
}

std::vector<double> compactRcWireCaps(const CompactRcForest& forest)
{
  std::vector<double> caps(forest.parent.size(), 0);
  for (int node = 0; node < static_cast<int>(caps.size()); ++node) {
    if (forest.parent[node] == node) continue;
    const double half = forest.edge_capacitance[node] * 0.5;
    caps[node] += half;
    caps[forest.parent[node]] += half;
  }
  return caps;
}

std::array<std::vector<double>, 3> evaluateCompactRc(
    const CompactRcForest& forest, const std::vector<double>& caps)
{
  const int count = forest.parent.size();
  std::vector<double> load(count), delay(count), ldelay(count), beta(count), impulse(count);
  for (std::size_t tree = 0; tree + 1 < forest.topo_start.size(); ++tree) {
    const int first = forest.topo_start[tree], last = forest.topo_start[tree + 1];
    for (int row = last; row-- > first;) {
      const int node = forest.topo[row];
      double value = caps[node];
      for (int edge = forest.child_start[node]; edge < forest.child_start[node + 1]; ++edge) {
        value += load[forest.children[edge]];
      }
      load[node] = value;
    }
    for (int row = first; row < last; ++row) {
      const int node = forest.topo[row], parent = forest.parent[node];
      delay[node] = parent == node ? 0.0 : delay[parent] + forest.resistance[node] * load[node];
    }
    for (int row = last; row-- > first;) {
      const int node = forest.topo[row];
      double value = caps[node] * delay[node];
      for (int edge = forest.child_start[node]; edge < forest.child_start[node + 1]; ++edge) {
        value += ldelay[forest.children[edge]];
      }
      ldelay[node] = value;
    }
    for (int row = first; row < last; ++row) {
      const int node = forest.topo[row], parent = forest.parent[node];
      beta[node] = parent == node ? 0.0 : beta[parent] + forest.resistance[node] * ldelay[node];
      impulse[node] = 2.0 * beta[node] - delay[node] * delay[node];
    }
  }
  return {std::move(load), std::move(delay), std::move(impulse)};
}

}  // namespace rebuffer

// --- rebuffer_incremental_rc_python.cc ---
// Legacy dense dictionary/NumPy boundary for the shared compact native RC math.

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>


namespace py = pybind11;
namespace rebuffer {

py::dict evaluateIncrementalRc(
    const py::dict& rc,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& generic_caps,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& rise_caps,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& fall_caps,
    const py::dict& buffer_caps, const int pin_count)
{
  const auto global_parent = rc["parent"].cast<std::vector<int>>();
  const auto global_child_start = rc["child_start"].cast<std::vector<int>>();
  const auto global_children = rc["child"].cast<std::vector<int>>();
  const auto global_topo_start = rc["topo_start"].cast<std::vector<int>>();
  const auto global_topo = rc["topo"].cast<std::vector<int>>();
  const auto global_resistance = rc["edge_resistance"].cast<std::vector<double>>();
  const auto global_edge_cap = rc["edge_capacitance"].cast<std::vector<double>>();
  CompactRcForest forest;
  auto& affected = forest.node_ids;
  affected = rc["affected_nodes"].cast<std::vector<int>>();
  std::unordered_map<int, int> local;
  local.reserve(affected.size());
  for (int i = 0; i < static_cast<int>(affected.size()); ++i) local.emplace(affected[i], i);
  forest.child_start.push_back(0);
  for (int node : affected) {
    forest.parent.push_back(local.at(global_parent[node]));
    forest.resistance.push_back(global_resistance[node]);
    forest.edge_capacitance.push_back(global_edge_cap[node]);
    for (int offset = global_child_start[node]; offset < global_child_start[node + 1]; ++offset) {
      forest.children.push_back(local.at(global_children[offset]));
    }
    forest.child_start.push_back(forest.children.size());
  }
  forest.topo_start = global_topo_start;
  for (int node : global_topo) forest.topo.push_back(local.at(node));
  const auto wire_cap = compactRcWireCaps(forest);
  auto modeCaps = [&](const auto& base, int mode) {
    std::vector<double> caps(affected.size(), 0);
    const auto values = base.template unchecked<1>();
    std::vector<std::vector<double>> slot_inputs, slot_outputs;
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
    for (int i = 0; i < static_cast<int>(affected.size()); ++i) {
      const int node = affected[i];
      if (node < pin_count && node < values.shape(0)) caps[i] = values(node);
      else if (node >= pin_count) {
        const int offset = node - pin_count, slot = offset / 2;
        if (slot >= 0 && slot < static_cast<int>(slot_inputs.size())) {
          caps[i] = (offset % 2 == 0 ? slot_inputs[slot] : slot_outputs[slot]).at(mode);
        }
      }
      caps[i] += wire_cap[i];
    }
    return caps;
  };
  py::dict result;
  py::array_t<int64_t> nodes({static_cast<py::ssize_t>(affected.size())},
                            {static_cast<py::ssize_t>(sizeof(int64_t))});
  std::copy(affected.begin(), affected.end(), nodes.mutable_data());
  result["affected_nodes"] = std::move(nodes);
  const std::array<const char*, 3> modes{"generic", "rise", "fall"};
  const std::array<const char*, 3> fields{"loads_", "delays_", "impulses_"};
  const std::array<const py::array_t<double, py::array::c_style | py::array::forcecast>*, 3>
      caps{&generic_caps, &rise_caps, &fall_caps};
  for (int mode = 0; mode < 3; ++mode) {
    const auto values = evaluateCompactRc(forest, modeCaps(*caps[mode], mode));
    for (int field = 0; field < 3; ++field) {
      result[(std::string(fields[field]) + modes[mode]).c_str()] = values[field];
    }
  }
  return result;
}

}  // namespace rebuffer
