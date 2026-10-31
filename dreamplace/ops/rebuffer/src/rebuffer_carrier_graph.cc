// SPDX-License-Identifier: BSD-3-Clause
// Build level schedules and adjacency from the committed physical netlist.
#include "rebuffer_carrier.hh"
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
  struct LibraryArc {
    std::string source, sink;
    int sense, trigger;
  };
  // Import each used Liberty row once, rather than query Python per cell arc.
  std::unordered_map<int, LibraryArc> library;
  auto libraryArc = [&](int lut) -> const LibraryArc& {
    const auto found = library.find(lut);
    if (found != library.end()) return found->second;
    const auto info = lib_arcs[lut].cast<py::sequence>();
    return library.emplace(lut, LibraryArc{py::cast<std::string>(info[0]),
        py::cast<std::string>(info[1]), py::cast<int>(info[4]), py::cast<int>(info[5])}).first->second;
  };
  // Equivalent masters need not store their Liberty arcs in the same order.
  auto sameSignature = [&](const LibraryArc& a, const LibraryArc& b) {
    return a.source == b.source && a.sink == b.sink && a.sense == b.sense
           && a.trigger == b.trigger;
  };
  auto currentArc = [&](int cell, int old_lut, int old_offset) {
    if (old_lut >= state.cell_arc_start.at(cell) && old_lut < state.cell_arc_start.at(cell + 1)) {
      return old_lut;
    }
    const auto previous = libraryArc(old_lut);
    // Equivalent masters can repeat a port/sense/trigger signature (state
    // dependent `when` arcs).  Masters of one family list those repeats in
    // the same order, so map the k-th repeat of the old master to the k-th
    // repeat of the current one.
    int rank = 0;
    if (old_offset >= 0) {
      for (int lut = old_lut - old_offset; lut < old_lut; ++lut) {
        if (sameSignature(libraryArc(lut), previous)) ++rank;
      }
    }
    std::vector<int> matches;
    for (int lut = state.cell_arc_start.at(cell); lut < state.cell_arc_start.at(cell + 1); ++lut) {
      if (sameSignature(libraryArc(lut), previous)) matches.push_back(lut);
    }
    if (matches.empty()) throw py::value_error("current master does not contain the timing arc");
    if (matches.size() > 1 && old_offset < 0) {
      throw py::value_error("ambiguous equivalent master timing arc");
    }
    if (rank >= static_cast<int>(matches.size())) {
      throw py::value_error("current master has fewer repeated timing arcs");
    }
    return matches[rank];
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
    row[7] = owner;
    row[1] = carrierIndex(row[1], state.old_pin_ids, "cell sink");
    const bool clock = i < base_levels[1];
    if (!clock) row[0] = carrierIndex(row[0], state.old_pin_ids, "cell source");
    const int cell = state.cell_by_node[owner];
    if (cell < 0 || row[3] < 0 || row[3] >= static_cast<int>(lib_arcs.size())) {
      throw py::value_error("current master does not contain the timing arc");
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
  for (const auto& pins : state.net_pins) {
    const int driver = !pins.empty() && driver_pins.count(pins.front()) ? pins.front() : -1;
    drivers.push_back(driver);
    if (driver >= 0) {
      for (const int sink : pins) if (sink != driver) net_arcs.push_back({driver, sink});
    }
    net_start.push_back(net_arcs.size());
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
