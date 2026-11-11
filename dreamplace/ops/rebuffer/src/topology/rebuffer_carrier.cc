// SPDX-License-Identifier: BSD-3-Clause
#include "rebuffer_carrier.hh"


#include <algorithm>
#include <unordered_set>

namespace py = pybind11;
namespace rebuffer {
namespace {

std::unordered_map<std::string, int> nameIndex(const std::vector<std::string>& names)
{
  std::unordered_map<std::string, int> result;
  for (int i = 0; i < static_cast<int>(names.size()); ++i) {
    if (!result.emplace(names[i], i).second) throw py::value_error("duplicate physical identity");
  }
  return result;
}

void exportRows(py::dict& output, const char* rows_name, const char* flat_name,
                const char* start_name, const std::vector<std::vector<int>>& rows)
{
  std::vector<int> flat;
  std::vector<int> start{0};
  for (const auto& row : rows) {
    flat.insert(flat.end(), row.begin(), row.end());
    start.push_back(flat.size());
  }
  output[rows_name] = rows;
  output[flat_name] = flat;
  output[start_name] = start;
}

}  // namespace

py::dict rebuildTimingCarrier(const py::object& base, const py::object& physical)
{
  if (!py::hasattr(base, "flat_lib_pin_names") || !py::hasattr(physical, "node_master_names")) {
    throw py::value_error("timing carrier requires Liberty pin names and physical node masters");
  }
  TimingCarrierView state;
  state.node_names = carrierArray<std::string>(physical, "node_names");
  state.pin_names = carrierArray<std::string>(physical, "pin_names");
  state.net_names = carrierArray<std::string>(physical, "net_names");
  state.pin2node = carrierArray<int>(physical, "pin2node_map");
  const auto masters = carrierArray<std::string>(physical, "node_master_names");
  const auto node_ids = nameIndex(state.node_names);
  state.pin_ids = nameIndex(state.pin_names);
  auto net_ids = nameIndex(state.net_names);
  const auto base_nodes = carrierArray<std::string>(base, "node_names");
  const auto base_pins = carrierArray<std::string>(base, "pin_names");
  const auto base_node_ids = nameIndex(base_nodes);
  const auto clock_names = carrierArray<std::string>(base, "clock_net_names");
  const std::unordered_set<std::string> clocks(clock_names.begin(), clock_names.end());
  auto x = carrierArray<double>(physical, "pin_offset_x");
  auto y = carrierArray<double>(physical, "pin_offset_y");
  auto orientation = carrierArray<std::string>(physical, "pin_direct");
  if (masters.size() != state.node_names.size() || state.pin2node.size() != state.pin_names.size()
      || x.size() != state.pin_names.size() || y.size() != state.pin_names.size()
      || orientation.size() != state.pin_names.size()) {
    throw py::value_error("physical carrier arrays do not align");
  }
  state.old_node_ids.assign(base_nodes.size(), -1);
  for (int node = 0; node < static_cast<int>(base_nodes.size()); ++node) {
    const auto found = node_ids.find(base_nodes[node]);
    if (found != node_ids.end()) state.old_node_ids[node] = found->second;
  }
  // Physical exports must include clocks. Missing timing pins are rejected
  // by the graph compiler rather than resurrected from stale connectivity.
  state.old_pin_ids.assign(base_pins.size(), -1);
  for (int pin = 0; pin < static_cast<int>(base_pins.size()); ++pin) {
    const auto found = state.pin_ids.find(base_pins[pin]);
    if (found != state.pin_ids.end()) state.old_pin_ids[pin] = found->second;
  }
  const auto physical_flat = carrierArray<int>(physical, "flat_net2pin_map");
  const auto physical_start = carrierArray<int>(physical, "flat_net2pin_start_map");
  if (physical_start.size() != state.net_names.size() + 1 || physical_start.front() != 0
      || physical_start.back() != static_cast<int>(physical_flat.size())) {
    throw py::value_error("physical net CSR does not align");
  }
  for (int net = 0; net < static_cast<int>(state.net_names.size()); ++net) {
    const int begin = physical_start[net], end = physical_start[net + 1];
    if (begin < 0 || end < begin || end > static_cast<int>(physical_flat.size())) {
      throw py::value_error("invalid physical net CSR range");
    }
    state.net_pins.emplace_back(physical_flat.begin() + begin, physical_flat.begin() + end);
  }
  const auto lib_names = carrierArray<std::string>(base, "flat_libcell_names");
  const auto lib_ids = nameIndex(lib_names);
  const auto main_start = carrierArray<int>(base, "main_id_2_cell_id_start");
  const auto lib_pin_start = carrierArray<int>(base, "cell_id_2_libpin_id_start");
  const auto lib_pin_names = carrierArray<std::string>(base, "flat_lib_pin_names");
  state.cell_arc_start = carrierArray<int>(base, "cell_id_2_arc_id_start");
  std::vector<int> main_id(state.node_names.size(), -1), cell_offset(main_id.size(), 0);
  state.cell_by_node.assign(state.node_names.size(), -1);
  const auto old_main = carrierArray<int>(base, "inst_main_id");
  for (int node = 0; node < static_cast<int>(masters.size()); ++node) {
    if (masters[node].empty()) continue;  // IO and synthetic physical obstacles.
    const auto found = lib_ids.find(masters[node]);
    if (found == lib_ids.end()) {
      const auto old = base_node_ids.find(state.node_names[node]);
      if (old == base_node_ids.end() || old_main.at(old->second) >= 0) {
        throw py::value_error("physical master is absent from Liberty: " + masters[node]);
      }
      continue;  // Existing physical-only cells stay outside the timing graph.
    }
    const int cell = found->second;
    const int main = std::upper_bound(main_start.begin(), main_start.end(), cell) - main_start.begin() - 1;
    state.cell_by_node[node] = cell;
    main_id[node] = main;
    cell_offset[node] = cell - main_start.at(main);
  }
  std::vector<int> lib_offset(state.pin_names.size(), -1);
  for (int pin = 0; pin < static_cast<int>(state.pin_names.size()); ++pin) {
    const int owner = state.pin2node.at(pin);
    const int cell = state.cell_by_node.at(owner);
    if (cell < 0) continue;
    const auto prefix = state.node_names[owner] + ":";
    if (state.pin_names[pin].rfind(prefix, 0) != 0) throw py::value_error("pin owner identity mismatch");
    const std::string port = state.pin_names[pin].substr(prefix.size());
    for (int i = lib_pin_start.at(cell); i < lib_pin_start.at(cell + 1); ++i) {
      if (lib_pin_names.at(i) == port) lib_offset[pin] = i - lib_pin_start[cell];
    }
    if (lib_offset[pin] < 0) throw py::value_error("physical signal pin is absent from Liberty");
  }
  std::vector<int> pin_net(state.pin_names.size(), -1);
  std::vector<double> weights;
  for (int net = 0; net < static_cast<int>(state.net_pins.size()); ++net) {
    weights.push_back(clocks.count(state.net_names[net]) ? 0.0 : 1.0);
    for (const int pin : state.net_pins[net]) {
      if (pin < 0 || pin >= static_cast<int>(pin_net.size()) || pin_net[pin] != -1) {
        throw py::value_error("physical pin belongs to multiple nets or is out of range");
      }
      pin_net[pin] = net;
    }
  }
  if (std::find(pin_net.begin(), pin_net.end(), -1) != pin_net.end()) {
    throw py::value_error("physical carrier contains an unconnected pin");
  }
  std::vector<std::vector<int>> node_pins(state.node_names.size());
  int movable_pins = 0;
  const int movable_nodes = py::cast<int>(physical.attr("num_nodes"))
      - py::cast<int>(physical.attr("num_terminals")) - py::cast<int>(physical.attr("num_terminal_NIs"));
  for (int pin = 0; pin < static_cast<int>(state.pin_names.size()); ++pin) {
    node_pins.at(state.pin2node[pin]).push_back(pin);
    movable_pins += state.pin2node[pin] < movable_nodes;
  }
  auto& out = state.values;
  out["pin_names"] = state.pin_names;
  out["pin_name2id_map"] = state.pin_ids;
  out["num_pins"] = state.pin_names.size();
  out["num_nets"] = state.net_names.size();
  out["net_names"] = state.net_names;
  out["net_name2id_map"] = net_ids;
  out["net_weights"] = weights;
  out["pin2node_map"] = state.pin2node;
  out["pin_to_node_id"] = state.pin2node;
  out["pin_to_inst_id"] = state.pin2node;
  out["pin2net_map"] = pin_net;
  out["pin_offset_x"] = x;
  out["pin_offset_y"] = y;
  out["pin_direct"] = orientation;
  out["num_movable_pins"] = movable_pins;
  out["inst_main_id"] = main_id;
  out["inst_libcell_offset"] = cell_offset;
  out["pin_2_libpin_offset"] = lib_offset;
  exportRows(out, "net2pin_map", "flat_net2pin_map", "flat_net2pin_start_map", state.net_pins);
  exportRows(out, "node2pin_map", "flat_node2pin_map", "flat_node2pin_start_map", node_pins);
  compileCarrierGraph(base, state);
  return out;
}

}  // namespace rebuffer

// --- rebuffer_carrier_graph.cc ---
// Build level schedules and adjacency from the committed physical netlist.
#include "rebuffer_incremental_timing.hh"


#include <algorithm>
#include <array>
#include <functional>
#include <queue>
#include <unordered_set>

namespace py = pybind11;
namespace rebuffer {
namespace {

using Arc = std::array<int, 8>;

void adjacency(py::dict& out, const char* starts, const char* flat,
               const std::vector<std::vector<int>>& rows)
{
  std::vector<int> values, offsets{0};
  for (const auto& row : rows) {
    values.insert(values.end(), row.begin(), row.end());
    offsets.push_back(values.size());
  }
  out[starts] = offsets;
  out[flat] = values;
}

std::vector<int> remapField(const py::object& base, const char* name,
                           const std::vector<int>& mapping)
{
  auto values = carrierArray<int>(base, name);
  for (int& value : values) value = carrierIndex(value, mapping, name);
  return values;
}

}  // namespace

void pruneBidirectionalPadNetArcs(
    int pin_count,
    const std::vector<char>& has_cell_arc,
    const std::vector<int>& net_arc_count,
    std::vector<std::array<int, 2>>& net_arcs,
    std::vector<int>& net_starts)
{
  std::unordered_set<long long> present;
  present.reserve(net_arcs.size() * 2);
  for (const auto& row : net_arcs) {
    present.insert(static_cast<long long>(row[0]) * pin_count + row[1]);
  }
  std::vector<std::array<int, 2>> kept;
  std::vector<int> rebuilt_start{0};
  int offset = 0;
  for (const int count : net_arc_count) {
    for (int index = 0; index < count; ++index) {
      const auto row = net_arcs[offset + index];
      const bool reverse = present.count(static_cast<long long>(row[1]) * pin_count + row[0]) != 0;
      if (reverse && !has_cell_arc[row[1]]) continue;
      kept.push_back(row);
    }
    offset += count;
    rebuilt_start.push_back(kept.size());
  }
  if (kept.size() != net_arcs.size()) {
    net_arcs.swap(kept);
    net_starts.swap(rebuilt_start);
  }
}

void compileCarrierGraph(const py::object& base, TimingCarrierView& state)
{
  auto& out = state.values;
  const int pin_count = state.pin_names.size(), node_count = state.node_names.size();
  const auto lib_arcs = base.attr("flat_libarc_info").cast<py::sequence>();
  const auto lib_cells = carrierArray<std::string>(base, "flat_libcell_names");
  const auto main_start = carrierArray<int>(base, "main_id_2_cell_id_start");
  const auto buffer_mains = carrierArray<int>(base, "buffer_main_type_candidate_indices");
  const std::unordered_set<int> buffers(buffer_mains.begin(), buffer_mains.end());
  const auto base_graph = timingGraphIndex(base);
  const auto& base_arcs = base_graph->cells;
  const auto base_levels = carrierArray<int>(base, "flat_inst_arcs_by_level_start");
  if (base_levels.size() < 2) throw py::value_error("timing carrier requires a clock arc level");
  // Import each used Liberty row once, rather than query Python per cell arc.
  std::unordered_map<int, TimingLibraryArc> library;
  auto libraryArc = [&](int lut) -> const TimingLibraryArc& {
    const auto found = library.find(lut);
    if (found != library.end()) return found->second;
    const auto info = lib_arcs[lut].cast<py::sequence>();
    return library.emplace(lut, TimingLibraryArc{py::cast<std::string>(info[0]),
        py::cast<std::string>(info[1]), py::cast<int>(info[4]), py::cast<int>(info[5])}).first->second;
  };
  auto currentArc = [&](int cell, int old_lut, int old_offset) {
    return equivalentTimingArc(state.cell_arc_start.at(cell),
        state.cell_arc_start.at(cell + 1), old_lut, old_offset, libraryArc);
  };
  std::vector<Arc> arcs;
  std::vector<bool> is_clock;
  for (int i = 0; i < static_cast<int>(base_arcs.size()); ++i) {
    Arc row = base_arcs[i];
    const int old_owner = row[7];
    if (old_owner < 0 || old_owner >= static_cast<int>(state.old_node_ids.size())) {
      throw py::value_error("invalid timing arc owner");
    }
    const int owner = state.old_node_ids[old_owner];
    if (owner < 0) continue;  // Removed physical instances have no cell arcs.
    // Arcs synthesized by the native graph builder carry no Liberty row, so
    // there is nothing to re-resolve against the current master.  The sparse
    // constraint compiler skips the same arcs.
    if (row[3] < 0) continue;
    row[7] = owner;
    row[1] = carrierIndex(row[1], state.old_pin_ids, "cell sink");
    const bool clock = i < base_levels[1];
    if (!clock) row[0] = carrierIndex(row[0], state.old_pin_ids, "cell source");
    const int cell = state.cell_by_node[owner];
    if (cell < 0 || row[3] < 0 || row[3] >= static_cast<int>(lib_arcs.size())) {
      throw py::value_error(
          "current master does not contain the timing arc: instance=" + state.node_names[owner]
          + " cell=" + std::to_string(cell) + " lib_arc=" + std::to_string(row[3])
          + " lib_arc_count=" + std::to_string(lib_arcs.size())
          + " source=" + state.pin_names[row[0]] + " sink=" + state.pin_names[row[1]]
          + " clock=" + (clock ? "1" : "0"));
    }
    const int lut = currentArc(cell, row[3], row[6]);
    const auto& info = libraryArc(lut);
    const auto& node = state.node_names[owner];
    const auto target = node + ":" + info.sink;
    if (target != state.pin_names[row[1]]
        || (!clock && node + ":" + info.source != state.pin_names[row[0]])) {
      throw py::value_error("current master timing arc port schema differs");
    }
    row[2] = cell;
    row[3] = lut;
    row[4] = info.sense;
    row[5] = info.trigger;
    row[6] = lut - state.cell_arc_start[cell];
    arcs.push_back(row);
    is_clock.push_back(clock);
  }
  std::unordered_set<int> original_nodes;
  for (const int node : state.old_node_ids) if (node >= 0) original_nodes.insert(node);
  for (int node = 0; node < node_count; ++node) {
    if (original_nodes.count(node) || state.cell_by_node[node] < 0) continue;
    const int cell = state.cell_by_node[node];
    const int main = std::upper_bound(main_start.begin(), main_start.end(), cell) - main_start.begin() - 1;
    if (!buffers.count(main)) throw py::value_error("new timing instances must be buffers");
    int added = 0;
    for (int lut = state.cell_arc_start.at(cell); lut < state.cell_arc_start.at(cell + 1); ++lut) {
      const auto& info = libraryArc(lut);
      const auto prefix = state.node_names[node] + ":";
      const auto source = state.pin_ids.find(prefix + info.source);
      const auto sink = state.pin_ids.find(prefix + info.sink);
      if (source == state.pin_ids.end() || sink == state.pin_ids.end()) {
        throw py::value_error("new buffer timing pins are absent from physical export");
      }
      arcs.push_back({source->second, sink->second, cell, lut,
                      info.sense, info.trigger,
                      lut - state.cell_arc_start[cell], node});
      is_clock.push_back(false);
      ++added;
    }
    if (!added) throw py::value_error("new buffer has no timing arcs");
  }
  auto clock_pins = remapField(base, "clock_pins", state.old_pin_ids);
  out["clock_pins"] = clock_pins;
  out["FF_ids"] = remapField(base, "FF_ids", state.old_pin_ids);
  const auto starts = remapField(base, "start_points", state.old_pin_ids);
  const auto endpoints = remapField(base, "end_points", state.old_pin_ids);
  out["start_points"] = starts;
  out["start_pin_ids"] = starts;
  out["end_points"] = endpoints;
  out["endpoint_pin_ids"] = endpoints;
  // The SDC deadlines and check qualification retain their row ordering.
  // Only pin/cell identities change; no STA engine is queried here.
  for (const char* field : {"endpoints_constraint_arcs", "endpoints_timing_check_arcs"}) {
    auto checks = base.attr(field).cast<std::vector<Arc>>();
    const bool clock_local = std::string(field) == "endpoints_constraint_arcs";
    for (auto& row : checks) {
      if (!clock_local) row[0] = carrierIndex(row[0], state.old_pin_ids, field);
      row[1] = carrierIndex(row[1], state.old_pin_ids, field);
      const int cell = state.cell_by_node.at(state.pin2node.at(row[1]));
      if (cell >= 0) {
        row[2] = cell;
        if (row[3] < 0 || row[3] >= static_cast<int>(lib_arcs.size())) {
          throw py::value_error("current master check arc is absent");
        }
        row[3] = currentArc(cell, row[3], row[7]);
        row[7] = row[3] - state.cell_arc_start[cell];
      }
    }
    out[field] = checks;
  }
  std::unordered_set<int> driver_pins(starts.begin(), starts.end());
  for (const auto& row : arcs) driver_pins.insert(row[1]);
  for (const int pin : carrierArray<int>(base, "net2driver_pin_map")) {
    if (pin >= 0 && state.old_pin_ids.at(pin) >= 0) driver_pins.insert(state.old_pin_ids[pin]);
  }
  std::vector<int> drivers, net_start{0};
  std::vector<std::array<int, 2>> net_arcs;
  std::vector<int> net_arc_count;
  for (const auto& pins : state.net_pins) {
    const int driver = !pins.empty() && driver_pins.count(pins.front()) ? pins.front() : -1;
    drivers.push_back(driver);
    int added = 0;
    if (driver >= 0) {
      for (const int sink : pins) {
        if (sink == driver) continue;
        net_arcs.push_back({driver, sink});
        ++added;
      }
    }
    net_arc_count.push_back(added);
    net_start.push_back(net_arcs.size());
  }
  // A bidirectional pad and its top-level port share one physical net and the
  // export can expose both directions, which is a two-cycle rather than a
  // forward path.  Both ends are start points, so keep the arc that enters a
  // cell pin and drop the arc that only enters the bare port; the pad-output
  // direction stays an endpoint check instead of a forward edge.
  {
    std::vector<char> has_cell_arc(pin_count, 0);
    for (int index = 0; index < static_cast<int>(arcs.size()); ++index) {
      if (is_clock[index]) continue;
      has_cell_arc[arcs[index][0]] = 1;
      has_cell_arc[arcs[index][1]] = 1;
    }
    pruneBidirectionalPadNetArcs(pin_count, has_cell_arc, net_arc_count, net_arcs, net_start);
  }
  std::vector<std::vector<std::pair<int, int>>> successors(pin_count);
  std::vector<int> indegree(pin_count, 0), levels(pin_count, 0);
  auto edge = [&](int source, int sink, int weight) {
    successors.at(source).emplace_back(sink, weight);
    ++indegree.at(sink);
  };
  for (int i = 0; i < static_cast<int>(arcs.size()); ++i) {
    if (!is_clock[i]) edge(arcs[i][0], arcs[i][1], 1);
  }
  for (const auto& row : net_arcs) edge(row[0], row[1], 0);
  std::priority_queue<int, std::vector<int>, std::greater<int>> ready;
  for (int pin = 0; pin < pin_count; ++pin) if (!indegree[pin]) ready.push(pin);
  int visited = 0;
  while (!ready.empty()) {
    const int pin = ready.top();
    ready.pop();
    ++visited;
    for (const auto& [next, weight] : successors[pin]) {
      levels[next] = std::max(levels[next], levels[pin] + weight);
      if (!--indegree[next]) ready.push(next);
    }
  }
  if (visited != pin_count) throw py::value_error("committed timing graph contains a combinational cycle");
  std::vector<std::vector<int>> by_level(1);
  for (int i = 0; i < static_cast<int>(arcs.size()); ++i) {
    const int level = is_clock[i] ? 0 : levels[arcs[i][0]] + 1;
    if (level >= static_cast<int>(by_level.size())) by_level.resize(level + 1);
    by_level[level].push_back(i);
  }
  std::vector<Arc> scheduled;
  std::vector<int> level_start{0}, inst_topo_start{0}, inst_topo_ids;
  std::vector<std::vector<int>> inst_arcs(node_count), pred(pin_count), succ(pin_count);
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
      if (seen.insert(row[7]).second) inst_topo_ids.push_back(row[7]);
      graphArc(is_clock[id] ? clock_pins.at(row[0]) : row[0], row[1],
               is_clock[id] ? -2 : index, row[7], row[2], row[3], row[4], row[5], row[6]);
    }
    level_start.push_back(scheduled.size());
    inst_topo_start.push_back(inst_topo_ids.size());
  }
  for (const auto& row : net_arcs) graphArc(row[0], row[1], -1, -1, -1, -1, 0, 0, -1);
  out["net2driver_pin_map"] = drivers;
  out["net_flat_arcs_start"] = net_start;
  out["net_flat_arcs"] = net_arcs;
  out["flat_inst_arcs_by_level"] = scheduled;
  out["flat_inst_arcs_by_level_start"] = level_start;
  out["timing_graph_index"] = makeTimingGraphIndex(state.pin_names, scheduled, net_arcs, clock_pins, level_start[1]);
  out["arc_level_start"] = level_start;
  out["inst_topo_start"] = inst_topo_start;
  out["inst_topo_ids"] = inst_topo_ids;
  adjacency(out, "inst_flat_arcs_start", "inst_flat_arcs", inst_arcs);
  adjacency(out, "pin_pred_start", "pin_pred_pin", pred);
  adjacency(out, "pin_succ_start", "pin_succ_pin", succ);
  adjacency(out, "pin_pred_start", "pin_pred_arc_id", pred_arcs);
  adjacency(out, "pin_succ_start", "pin_succ_arc_id", succ_arcs);
  adjacency(out, "flat_pin_to_graph_start", "flat_pin_to_graph", succ);
  adjacency(out, "flat_pin_to_graph_start_reverse", "flat_pin_to_graph_reverse", pred);
  out["arc_src_pin"] = graph_src; out["arc_dst_pin"] = graph_dst;
  out["arc_inst_id"] = graph_inst; out["arc_libcell_id"] = graph_cell;
  out["arc_libarc_id"] = graph_lut; out["arc_sense"] = graph_sense;
  out["arc_type"] = graph_type; out["arc_offset"] = graph_offset;
  out["timing_backend_snapshot_valid"] = false;
  out["timing_carrier_source"] = "physical_pydb_native_graph_rebuild";
  out["ecc_sta_run"] = false;
}

}  // namespace rebuffer

// --- rebuffer_carrier_projection.cc ---

#include <algorithm>
#include <stdexcept>

namespace rebuffer {

void normalizePartitionProvenance(PartitionProvenance& rows)
{
  std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) {
    return a.first < b.first;
  });
  for (std::size_t index = 1; index < rows.size(); ++index) {
    if (rows[index - 1].first == rows[index].first
        && rows[index - 1].second != rows[index].second) {
      throw std::invalid_argument("duplicate partition vertex provenance");
    }
  }
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());
}

PartitionProvenance::const_iterator findPartitionProvenance(
    const PartitionProvenance& rows, const std::pair<int, int>& key)
{
  const auto found = std::lower_bound(rows.begin(), rows.end(), key,
      [](const auto& row, const auto& value) { return row.first < value; });
  return found != rows.end() && found->first == key ? found : rows.end();
}

}  // namespace rebuffer
