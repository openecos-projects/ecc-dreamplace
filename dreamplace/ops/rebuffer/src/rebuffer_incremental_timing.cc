// SPDX-License-Identifier: BSD-3-Clause
// Exact electrical forward and required-time reverse worklists. No STA queries.
#include "rebuffer_incremental_timing.hh"
#include "rebuffer_carrier.hh"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <queue>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace py = pybind11;
namespace rebuffer {

std::size_t TimingArcHash::operator()(const TimingArcIdentity& value) const
{
  std::size_t result = 0;
  for (const int part : value) result ^= std::hash<int>{}(part) + 0x9e3779b9 + (result << 6) + (result >> 2);
  return result;
}

int TimingGraphIndex::source(int id) const
{
  return id < clock_end ? clocks.at(cells[id][0]) : cells[id][0];
}

TimingArcIdentity TimingGraphIndex::key(int id) const
{
  const auto& row = cells[id];
  return {source(id), row[1], row[3], row[4], row[5], id < clock_end};
}

void TimingGraphIndex::build()
{
  pin_ids.clear();
  arc_ids.clear();
  if (!cell_active.empty() && cell_active.size() != cells.size()) {
    throw py::value_error("timing graph active cell slots do not align");
  }
  if (!net_active.empty() && net_active.size() != nets.size()) {
    throw py::value_error("timing graph active net slots do not align");
  }
  pin_ids.reserve(pins.size()); arc_ids.reserve(cells.size());
  for (int pin = 0; pin < static_cast<int>(pins.size()); ++pin) {
    if (!pin_ids.emplace(pins[pin], pin).second) throw py::value_error("duplicate timing pin");
  }
  for (int id = 0; id < static_cast<int>(cells.size()); ++id) {
    if (!cell_active.empty() && !cell_active[id]) continue;
    if (!arc_ids.emplace(key(id), id).second) throw py::value_error("duplicate timing arc");
  }

  const int pin_count = static_cast<int>(pins.size());
  rank.assign(pin_count, 0);
  path_level.assign(pin_count, 0);
  successors.assign(pin_count, {});
  cell_out.assign(pin_count, {});
  cell_in.assign(pin_count, {});
  net_out.assign(pin_count, {});
  net_in.assign(pin_count, {});
  std::vector<int> indegree(pin_count, 0);
  for (int id = 0; id < static_cast<int>(cells.size()); ++id) {
    if (!cell_active.empty() && !cell_active[id]) continue;
    const int source_pin = source(id);
    const int sink_pin = cells[id][1];
    if (source_pin < 0 || source_pin >= pin_count
        || sink_pin < 0 || sink_pin >= pin_count) {
      throw py::value_error("timing graph arc pin is out of range");
    }
    cell_out[source_pin].push_back(id);
    cell_in[sink_pin].push_back(id);
    if (id >= clock_end) {
      successors[source_pin].push_back(sink_pin);
      ++indegree[sink_pin];
    }
  }
  for (int id = 0; id < static_cast<int>(nets.size()); ++id) {
    if (!net_active.empty() && !net_active[id]) continue;
    const int source_pin = nets[id][0];
    const int sink_pin = nets[id][1];
    if (source_pin < 0 || source_pin >= pin_count
        || sink_pin < 0 || sink_pin >= pin_count) {
      throw py::value_error("timing graph net pin is out of range");
    }
    net_out[source_pin].push_back(id);
    net_in[sink_pin].push_back(id);
    successors[source_pin].push_back(sink_pin);
    ++indegree[sink_pin];
  }
  std::queue<int> ready;
  for (int pin = 0; pin < pin_count; ++pin) {
    if (indegree[pin] == 0) ready.push(pin);
  }
  int visited = 0;
  while (!ready.empty()) {
    const int pin = ready.front();
    ready.pop();
    ++visited;
    for (int id : cell_out[pin]) {
      if (id >= clock_end) {
        const int sink = cells[id][1];
        path_level[sink] = std::max(path_level[sink], path_level[pin] + 1);
      }
    }
    for (int id : net_out[pin]) {
      const int sink = nets[id][1];
      path_level[sink] = std::max(path_level[sink], path_level[pin]);
    }
    for (const int next : successors[pin]) {
      rank[next] = std::max(rank[next], rank[pin] + 1);
      if (--indegree[next] == 0) ready.push(next);
    }
  }
  if (visited != pin_count) {
    throw py::value_error("incremental timing requires a combinational DAG");
  }
}

std::shared_ptr<TimingGraphIndex> makeTimingGraphIndex(
    const std::vector<std::string>& pins, const std::vector<TimingCellArc>& cells,
    const std::vector<TimingNetArc>& nets, const std::vector<int>& clocks, int clock_end)
{
  auto graph = std::make_shared<TimingGraphIndex>();
  graph->pins = pins; graph->cells = cells; graph->nets = nets; graph->clocks = clocks; graph->clock_end = clock_end;
  graph->build();
  return graph;
}

namespace {

void mixHash(std::uint64_t& state, const std::uint64_t value)
{
  state ^= value + UINT64_C(0x9e3779b97f4a7c15) + (state << 6) + (state >> 2);
}

void hashValue(std::uint64_t& state, const int value)
{
  mixHash(state, static_cast<std::uint64_t>(static_cast<std::int64_t>(value)));
}

void hashValue(std::uint64_t& state, const std::string& value)
{
  mixHash(state, static_cast<std::uint64_t>(std::hash<std::string>{}(value)));
}

template <std::size_t Width>
void hashValue(std::uint64_t& state, const std::array<int, Width>& value)
{
  for (const int item : value) hashValue(state, item);
}

template <typename T>
void hashVector(std::uint64_t& state, const std::vector<T>& values)
{
  mixHash(state, static_cast<std::uint64_t>(values.size()));
  for (const auto& value : values) hashValue(state, value);
}

std::uint64_t graphFingerprint(
    const std::vector<std::string>& pins,
    const std::vector<TimingCellArc>& cells,
    const std::vector<TimingNetArc>& nets,
    const std::vector<int>& clocks,
    const int clock_end)
{
  std::uint64_t state = UINT64_C(0x243f6a8885a308d3);
  hashVector(state, pins);
  hashVector(state, cells);
  hashVector(state, nets);
  hashVector(state, clocks);
  hashValue(state, clock_end);
  return state;
}

bool sameGraph(
    const TimingGraphIndex& graph,
    const std::vector<std::string>& pins,
    const std::vector<TimingCellArc>& cells,
    const std::vector<TimingNetArc>& nets,
    const std::vector<int>& clocks,
    const int clock_end)
{
  return graph.pins == pins && graph.cells == cells && graph.nets == nets
      && graph.clocks == clocks && graph.clock_end == clock_end;
}

}  // namespace

std::shared_ptr<TimingGraphIndex> timingGraphIndex(
    const py::object& db, bool* cache_hit)
{
  const auto cells = db.attr("flat_inst_arcs_by_level").cast<std::vector<TimingCellArc>>();
  const auto offsets = carrierArray<int>(db, "flat_inst_arcs_by_level_start");
  if (offsets.size() < 2 || offsets[0] != 0 || offsets[1] < 0 || offsets[1] > static_cast<int>(cells.size())) {
    throw py::value_error("incremental timing requires a valid clock arc level");
  }
  const auto pins = carrierArray<std::string>(db, "pin_names");
  const auto nets = db.attr("net_flat_arcs").cast<std::vector<TimingNetArc>>();
  const auto clocks = carrierArray<int>(db, "clock_pins");
  if (cache_hit != nullptr) *cache_hit = false;
  if (py::hasattr(db, "timing_graph_index")) {
    const auto cached = db.attr("timing_graph_index").cast<std::shared_ptr<TimingGraphIndex>>();
    // Compact persistent views may forward untouched attributes to their base
    // carrier. Never reuse that cached index after the graph arrays changed.
    if (sameGraph(*cached, pins, cells, nets, clocks, offsets[1])) {
      if (cache_hit != nullptr) *cache_hit = true;
      return cached;
    }
  }
  // Compact persistent views cannot safely receive a pybind object as a new
  // Python attribute: ordinary PyDB test doubles and carrier snapshots rely
  // on deepcopy/pickle of their __dict__. Keep a bounded structural cache so
  // separate compact wrappers with the same immutable timing graph share the
  // same native index. The fingerprint is only a lookup accelerator; every
  // candidate is still compared element by element before reuse.
  struct CacheEntry {
    std::uint64_t fingerprint = 0;
    std::shared_ptr<TimingGraphIndex> graph;
  };
  static std::vector<CacheEntry> cache;
  const auto fingerprint = graphFingerprint(pins, cells, nets, clocks, offsets[1]);
  for (auto& entry : cache) {
    if (entry.fingerprint != fingerprint) continue;
    if (sameGraph(*entry.graph, pins, cells, nets, clocks, offsets[1])) {
      if (cache_hit != nullptr) *cache_hit = true;
      return entry.graph;
    }
  }
  if (cache.size() >= 32) cache.erase(cache.begin());
  cache.push_back({
      fingerprint,
      makeTimingGraphIndex(pins, cells, nets, clocks, offsets[1])});
  return cache.back().graph;
}

namespace {

using Arc = TimingCellArc;

template <typename T>
py::array_t<int64_t> indices(const std::vector<T>& values)
{
  // Explicit element stride also works with the repository's pybind11 2.11
  // headers when Python uses NumPy 2 (dtype.itemsize has a different layout).
  py::array_t<int64_t> output({static_cast<py::ssize_t>(values.size())},
                              {static_cast<py::ssize_t>(sizeof(int64_t))});
  std::copy(values.begin(), values.end(), output.mutable_data());
  return output;
}

void exportCsr(py::dict& output,
               const char* prefix,
               const std::vector<std::vector<int>>& rows)
{
  std::vector<int> offsets{0};
  std::vector<int> values;
  for (const auto& row : rows) {
    values.insert(values.end(), row.begin(), row.end());
    offsets.push_back(static_cast<int>(values.size()));
  }
  output[(std::string(prefix) + "_start").c_str()] = indices(offsets);
  output[(std::string(prefix) + "_ids").c_str()] = indices(values);
}

void closeCone(std::vector<unsigned char>& marked, const std::vector<std::vector<int>>& edges)
{
  std::queue<int> pending;
  for (int pin = 0; pin < static_cast<int>(marked.size()); ++pin) if (marked[pin]) pending.push(pin);
  while (!pending.empty()) {
    const int pin = pending.front(); pending.pop();
    for (const int next : edges[pin]) {
      if (!marked[next]) { marked[next] = 1; pending.push(next); }
    }
  }
}

void closeBackwardCone(
    std::vector<unsigned char>& marked,
    const TimingGraphIndex& graph,
    const std::vector<unsigned char>& required_net_sink)
{
  std::queue<int> pending;
  for (int pin = 0; pin < static_cast<int>(marked.size()); ++pin) {
    if (marked[pin]) pending.push(pin);
  }
  while (!pending.empty()) {
    const int pin = pending.front();
    pending.pop();
    for (const int arc_id : graph.cell_in[pin]) {
      if (arc_id < graph.clock_end) continue;
      const int previous = graph.source(arc_id);
      if (!marked[previous]) {
        marked[previous] = 1;
        pending.push(previous);
      }
    }
    if (!required_net_sink[pin]) continue;
    for (const int net_id : graph.net_in[pin]) {
      const int previous = graph.nets[net_id][0];
      if (!marked[previous]) {
        marked[previous] = 1;
        pending.push(previous);
      }
    }
  }
}

struct Worklist
{
  std::vector<std::vector<int>> cells, nets;

  explicit Worklist(int levels) : cells(levels), nets(levels) {}

  void exportTo(py::dict& out, const std::string& prefix, bool reverse) const
  {
    std::vector<int> flat_cells, flat_nets, cell_start{0}, net_start{0};
    for (int index = 0; index < static_cast<int>(cells.size()); ++index) {
      const int level = reverse ? cells.size() - 1 - index : index;
      if (cells[level].empty() && nets[level].empty()) continue;
      flat_cells.insert(flat_cells.end(), cells[level].begin(), cells[level].end());
      flat_nets.insert(flat_nets.end(), nets[level].begin(), nets[level].end());
      cell_start.push_back(flat_cells.size()); net_start.push_back(flat_nets.size());
    }
    out[(prefix + "_cells").c_str()] = indices(flat_cells);
    out[(prefix + "_nets").c_str()] = indices(flat_nets);
    out[(prefix + "_cell_start").c_str()] = indices(cell_start);
    out[(prefix + "_net_start").c_str()] = indices(net_start);
  }
};

std::vector<int> selected(const std::vector<unsigned char>& mask)
{
  std::vector<int> pins;
  for (int pin = 0; pin < static_cast<int>(mask.size()); ++pin) if (mask[pin]) pins.push_back(pin);
  return pins;
}

}  // namespace

py::dict affectedTimingSchedule(
    const py::object& previous, const py::object& current,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& previous_rc,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& current_rc,
    const py::object& changed_pins)
{
  bool previous_cache_hit = false;
  bool current_cache_hit = false;
  const auto old_owner = timingGraphIndex(previous, &previous_cache_hit);
  const auto current_owner = timingGraphIndex(current, &current_cache_hit);
  const auto& old = *old_owner; const auto& graph = *current_owner;
  const int count = graph.pins.size();
  if (previous_rc.ndim() != 2 || current_rc.ndim() != 2 || previous_rc.shape(0) != 6 || current_rc.shape(0) != 6
      || previous_rc.shape(1) != static_cast<int>(old.pins.size()) || current_rc.shape(1) != count) {
    throw py::value_error("incremental RC input requires six lanes per physical pin");
  }
  std::vector<int> pin_map(count, -1), arc_map(graph.cells.size(), -1);
  std::vector<unsigned char> forward(count, 0), backward(count, 0);
  const bool use_changed_pins = !changed_pins.is_none();
  auto before = previous_rc.unchecked<2>(); auto after = current_rc.unchecked<2>();
  for (int pin = 0; pin < count; ++pin) {
    int previous_pin = -1;
    if (pin < static_cast<int>(old.pins.size()) && graph.pins[pin] == old.pins[pin]) {
      previous_pin = pin;
    } else {
      const auto found = old.pin_ids.find(graph.pins[pin]);
      if (found != old.pin_ids.end()) previous_pin = found->second;
    }
    pin_map[pin] = previous_pin;
    if (previous_pin < 0) {
      forward[pin] = 1;
    } else if (!use_changed_pins) {
      for (int lane = 0; lane < 6; ++lane) {
        if (before(lane, previous_pin) != after(lane, pin)) forward[pin] = 1;
      }
    }
  }
  if (use_changed_pins) {
    for (const auto& item : changed_pins.cast<py::sequence>()) {
      const int pin = py::cast<int>(item);
      if (pin < 0 || pin >= count) throw py::value_error("changed timing pin is out of range");
      forward[pin] = 1;
    }
  }
  std::vector<unsigned char> matched(old.cells.size(), 0);
  std::vector<unsigned char> reverse_seed(count, 0);
  const auto& successors = graph.successors;
  const auto& rank = graph.rank;
  // Full TimingPropagation propagates net RAT from endpoints and data-cell
  // inputs, never from an unused/clock-only electrical leaf.
  std::vector<unsigned char> required_net_sink(count, 0);
  for (const int pin : carrierArray<int>(current, "end_points")) required_net_sink.at(pin) = 1;
  for (int id = 0; id < static_cast<int>(graph.cells.size()); ++id) {
    auto key = graph.key(id);
    key[0] = pin_map.at(key[0]); key[1] = pin_map.at(key[1]);
    const auto found = old.arc_ids.find(key);
    if (found == old.arc_ids.end()) forward.at(graph.cells[id][1]) = 1;
    else { arc_map[id] = found->second; matched[found->second] = 1; }
    if (id >= graph.clock_end) {
      required_net_sink.at(graph.source(id)) = 1;
    }
  }
  for (int id = 0; id < static_cast<int>(old.cells.size()); ++id) {
    if (matched[id]) continue;
    const auto found = graph.pin_ids.find(old.pins.at(old.cells[id][1]));
    if (found != graph.pin_ids.end()) forward[found->second] = 1;
    if (id < old.clock_end) continue;
    const int old_source = old.source(id);
    const auto source = graph.pin_ids.find(old.pins.at(old_source));
    if (source != graph.pin_ids.end()) reverse_seed[source->second] = 1;
  }
  // Graph successors, ranks, and CSR adjacency are immutable for this
  // carrier and were built once by TimingGraphIndex::build().
  // A split or a restore can remove an old net edge while leaving its driver
  // pin alive. The forward cone reaches the new sinks, but it cannot reach
  // that driver through the new graph. Export the driver as an explicit
  // reverse seed so the Python event pass can discard the stale RAT value.
  auto edgeKey = [](int source, int sink) {
    return (static_cast<std::uint64_t>(static_cast<std::uint32_t>(source)) << 32)
           | static_cast<std::uint32_t>(sink);
  };
  std::unordered_set<std::uint64_t> current_net_edges;
  current_net_edges.reserve(graph.nets.size());
  for (const auto& edge : graph.nets) current_net_edges.insert(edgeKey(edge[0], edge[1]));
  for (const auto& edge : old.nets) {
    const auto source = graph.pin_ids.find(old.pins.at(edge[0]));
    const auto sink = graph.pin_ids.find(old.pins.at(edge[1]));
    if (source == graph.pin_ids.end() || sink == graph.pin_ids.end()) continue;
    if (!current_net_edges.count(edgeKey(source->second, sink->second))) {
      reverse_seed[source->second] = 1;
    }
  }
  if (!use_changed_pins) {
    std::vector<std::vector<std::string>> old_drivers(old.pins.size());
    for (const auto& edge : old.nets) old_drivers.at(edge[1]).push_back(old.pins.at(edge[0]));
    std::vector<std::vector<std::string>> drivers(count);
    for (const auto& edge : graph.nets) drivers.at(edge[1]).push_back(graph.pins.at(edge[0]));
    for (int pin = 0; pin < count; ++pin) {
      std::sort(drivers[pin].begin(), drivers[pin].end());
      if (pin_map[pin] < 0) continue;
      auto& previous_drivers = old_drivers[pin_map[pin]];
      std::sort(previous_drivers.begin(), previous_drivers.end());
      if (drivers[pin] != previous_drivers) forward[pin] = 1;
    }
  }
  const std::vector<int> forward_seeds = selected(forward);
  closeCone(forward, successors);
  backward = forward;
  closeBackwardCone(backward, graph, required_net_sink);
  const int levels = rank.empty() ? 1 : *std::max_element(rank.begin(), rank.end()) + 1;
  Worklist fwd(levels), bwd(levels);
  std::vector<int> clock_rows;
  for (int id = 0; id < static_cast<int>(graph.cells.size()); ++id) {
    const int source = graph.source(id), sink = graph.cells[id][1];
    if (id < graph.clock_end) {
      if (forward[sink]) clock_rows.push_back(id);
      continue;
    }
    if (forward[sink]) fwd.cells[rank[sink]].push_back(id);
    if (backward[source]) bwd.cells[rank[source]].push_back(id);
  }
  for (int id = 0; id < static_cast<int>(graph.nets.size()); ++id) {
    const auto& edge = graph.nets[id];
    if (forward[edge[1]]) fwd.nets[rank[edge[1]]].push_back(id);
    if (backward[edge[0]] && required_net_sink[edge[1]]) bwd.nets[rank[edge[0]]].push_back(id);
  }
  py::dict out;
  out["pin_from_previous"] = indices(pin_map);
  out["arc_from_previous"] = indices(arc_map);
  out["forward_pins"] = indices(selected(forward));
  out["forward_seed_pins"] = indices(forward_seeds);
  out["reverse_seed_pins"] = indices(selected(reverse_seed));
  out["backward_pins"] = indices(selected(backward));
  out["pin_rank"] = indices(rank);
  exportCsr(out, "cell_out", graph.cell_out);
  exportCsr(out, "cell_in", graph.cell_in);
  exportCsr(out, "net_out", graph.net_out);
  exportCsr(out, "net_in", graph.net_in);
  out["clock_rows"] = indices(clock_rows);
  auto selectedRows = [&](const char* name, int pin_column, const std::vector<unsigned char>& mask) {
    std::vector<int> rows;
    const auto values = current.attr(name).cast<std::vector<Arc>>();
    for (int id = 0; id < static_cast<int>(values.size()); ++id) if (mask.at(values[id][pin_column])) rows.push_back(id);
    return indices(rows);
  };
  auto selectedPins = [&](const char* name, const std::vector<unsigned char>& mask) {
    std::vector<int> rows;
    const auto pins = carrierArray<int>(current, name);
    for (int id = 0; id < static_cast<int>(pins.size()); ++id) if (mask.at(pins[id])) rows.push_back(id);
    return indices(rows);
  };
  out["forward_start_rows"] = selectedPins("start_points", forward);
  out["backward_endpoint_rows"] = selectedPins("end_points", backward);
  out["setup_rows"] = selectedRows("endpoints_constraint_arcs", 1, backward);
  out["check_rows"] = selectedRows("endpoints_timing_check_arcs", 1, backward);
  fwd.exportTo(out, "forward", false); bwd.exportTo(out, "backward", true);
  out["total_pins"] = count;
  out["total_cell_arcs"] = graph.cells.size();
  out["timing_graph_cache_hit_previous"] = previous_cache_hit;
  out["timing_graph_cache_hit_current"] = current_cache_hit;
  return out;
}

}  // namespace rebuffer
