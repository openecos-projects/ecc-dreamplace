// SPDX-License-Identifier: BSD-3-Clause
// Exact electrical forward and required-time reverse worklists. No STA queries.
#include "rebuffer_incremental_timing.hh"
#include "rebuffer_carrier.hh"

#include <algorithm>
#include <array>
#include <queue>
#include <unordered_map>

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
  pin_ids.reserve(pins.size()); arc_ids.reserve(cells.size());
  for (int pin = 0; pin < static_cast<int>(pins.size()); ++pin) {
    if (!pin_ids.emplace(pins[pin], pin).second) throw py::value_error("duplicate timing pin");
  }
  for (int id = 0; id < static_cast<int>(cells.size()); ++id) {
    if (!arc_ids.emplace(key(id), id).second) throw py::value_error("duplicate timing arc");
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

std::shared_ptr<TimingGraphIndex> timingGraphIndex(const py::object& db)
{
  if (py::hasattr(db, "timing_graph_index")) return db.attr("timing_graph_index").cast<std::shared_ptr<TimingGraphIndex>>();
  const auto cells = db.attr("flat_inst_arcs_by_level").cast<std::vector<TimingCellArc>>();
  const auto offsets = carrierArray<int>(db, "flat_inst_arcs_by_level_start");
  if (offsets.size() < 2 || offsets[0] != 0 || offsets[1] < 0 || offsets[1] > static_cast<int>(cells.size())) {
    throw py::value_error("incremental timing requires a valid clock arc level");
  }
  return makeTimingGraphIndex(carrierArray<std::string>(db, "pin_names"), cells,
      db.attr("net_flat_arcs").cast<std::vector<TimingNetArc>>(), carrierArray<int>(db, "clock_pins"), offsets[1]);
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
  const auto old_owner = timingGraphIndex(previous), current_owner = timingGraphIndex(current);
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
  std::vector<std::vector<int>> successors(count), predecessors(count);
  std::vector<int> indegree(count, 0), rank(count, 0);
  auto connect = [&](int source, int sink, bool required = true) {
    successors.at(source).push_back(sink);
    if (required) predecessors.at(sink).push_back(source);
    ++indegree.at(sink);
  };
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
      connect(graph.source(id), graph.cells[id][1]);
    }
  }
  for (int id = 0; id < static_cast<int>(old.cells.size()); ++id) {
    if (matched[id]) continue;
    const auto found = graph.pin_ids.find(old.pins.at(old.cells[id][1]));
    if (found != graph.pin_ids.end()) forward[found->second] = 1;
  }
  for (const auto& edge : graph.nets) {
    connect(edge[0], edge[1], required_net_sink.at(edge[1]));
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
  std::queue<int> ready;
  for (int pin = 0; pin < count; ++pin) if (!indegree[pin]) ready.push(pin);
  int visited = 0;
  while (!ready.empty()) {
    const int pin = ready.front(); ready.pop(); ++visited;
    for (const int next : successors[pin]) {
      rank[next] = std::max(rank[next], rank[pin] + 1);
      if (!--indegree[next]) ready.push(next);
    }
  }
  if (visited != count) throw py::value_error("incremental timing requires a combinational DAG");
  closeCone(forward, successors);
  backward = forward;
  closeCone(backward, predecessors);
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
  out["backward_pins"] = indices(selected(backward));
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
  return out;
}

}  // namespace rebuffer
