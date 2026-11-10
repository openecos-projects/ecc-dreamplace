// SPDX-License-Identifier: BSD-3-Clause
// Update only changed physical edges and ranks; keep net-edge slots stable.
#include "rebuffer_timing_local_index.hh"
#include "rebuffer_timing_buffer_state.hh"

#include <algorithm>
#include <chrono>
#include <functional>
#include <queue>
#include <set>
#include <unordered_set>

namespace rebuffer {
namespace {

void eraseId(std::vector<int>& row, int id)
{
  row.erase(std::remove(row.begin(), row.end(), id), row.end());
}

void eraseArc(TimingGraphIndex& index, int id)
{
  const auto found = index.arc_ids.find(index.key(id));
  if (found != index.arc_ids.end() && found->second == id) index.arc_ids.erase(found);
}

void resizeRows(TimingMutationState& m, TimingGraphIndex& index, std::size_t size)
{
  index.rank.resize(size, 0); index.path_level.resize(size, 0); index.successors.resize(size);
  index.cell_out.resize(size); index.cell_in.resize(size);
  index.net_out.resize(size); index.net_in.resize(size);
  m.net_arc_by_pin.resize(size, -1);
  m.events.forward_inputs.resize(size); m.events.reverse_outputs.resize(size);
  m.events.forward_successors.resize(size); m.events.reverse_predecessors.resize(size);
}

}  // namespace

std::shared_ptr<TimingIndexUndo> captureTimingIndex(
    const TimingMutationState& m, const TimingGraphIndex& index)
{
  auto undo = std::make_shared<TimingIndexUndo>();
  undo->pins = index.pins.size(); undo->cells = index.cells.size(); undo->nets = index.nets.size();
  undo->canonical_schedule = m.canonical_schedule;
  return undo;
}

void captureTimingIndexPin(TimingIndexUndo& undo, const TimingMutationState& m,
                            const TimingGraphIndex& index, int pin)
{
  if (pin >= static_cast<int>(undo.pins) || undo.rows.count(pin)) return;
  undo.rows.emplace(pin, TimingIndexPinUndo{
      index.rank.at(pin), index.path_level.at(pin), m.net_arc_by_pin.at(pin), index.successors.at(pin),
      index.cell_out.at(pin), index.cell_in.at(pin), index.net_out.at(pin), index.net_in.at(pin),
      m.events.forward_inputs.at(pin), m.events.reverse_outputs.at(pin),
      m.events.forward_successors.at(pin), m.events.reverse_predecessors.at(pin)});
}

void initializeTimingLocalIndex(TimingMutationState& m, TimingGraphIndex& index,
                                 const TimingPropagationProgram& program)
{
  index.cell_active = m.cell_active;
  index.net_active.assign(index.nets.size(), 1);
  m.net_arc_by_pin.assign(index.pins.size(), -1);
  for (int id = 0; id < static_cast<int>(index.nets.size()); ++id) {
    const auto& edge = index.nets[id];
    const int net = m.pin_net.at(edge[1]);
    if (m.net_arc_by_pin.at(edge[1]) >= 0 || net < 0
        || m.net_drivers.at(net) != edge[0] || m.pin_net.at(edge[0]) != net) {
      throw pybind11::value_error("native mutation requires one physical net driver per load pin");
    }
    m.net_arc_by_pin[edge[1]] = id;
  }
  const auto started = std::chrono::steady_clock::now();
  auto schedule = makeTimingMutationSchedule(m, index, program);
  m.events = std::move(schedule->events);
  ++m.statistics.schedule_builds;
  m.statistics.schedule_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
}

void updateTimingLocalIndex(TimingMutationState& m, TimingGraphIndex& index,
    const TimingPropagationProgram& program, const TimingTopologyUndo& topology_undo)
{
  const auto started = std::chrono::steady_clock::now();
  auto& undo = *topology_undo.index_undo;
  resizeRows(m, index, index.pins.size());
  for (int pin = undo.pins; pin < static_cast<int>(index.pins.size()); ++pin) {
    if (!index.pin_ids.emplace(index.pins[pin], pin).second) {
      throw pybind11::value_error("duplicate timing pin");
    }
  }
  std::set<int> pins, cells, structural, rank_seeds, events;
  for (const auto& pin : topology_undo.previous_pins) pins.insert(pin.id);
  for (const auto& arc : topology_undo.previous_arcs) cells.insert(arc.id);
  for (int buf_id : m.buffers_by_source_net.at(topology_undo.source_net)) {
    const auto& buffer = m.buffers[buf_id];
    pins.insert(buffer.input_pin); pins.insert(buffer.output_pin);
    cells.insert(buffer.arcs.begin(), buffer.arcs.end());
  }
  events = pins;
  index.cells.resize(m.cells.size()); index.cell_active.resize(m.cells.size(), 0);
  for (int id : cells) {
    if (id < static_cast<int>(undo.cells)) {
      undo.cell_rows.push_back({id, index.cells.at(id), index.cell_active.at(id)});
    }
    if (index.cell_active[id]) eraseArc(index, id);
  }
  auto touch = [&](int pin) {
    captureTimingIndexPin(undo, m, index, pin);
    structural.insert(pin); events.insert(pin);
  };
  for (int id : cells) {
    const bool was_active = index.cell_active[id], active = m.cell_active[id];
    const int old_source = was_active ? index.source(id) : -1;
    const int old_sink = was_active ? index.cells[id][1] : -1;
    const int source = m.cells[id][0], sink = m.cells[id][1];
    const bool connectivity = was_active != active
        || (active && (old_source != source || old_sink != sink));
    if (connectivity && was_active) {
      touch(old_source); touch(old_sink);
      eraseId(index.cell_out[old_source], id); eraseId(index.cell_in[old_sink], id);
      rank_seeds.insert(old_sink);
    }
    std::copy(m.cells[id].begin(), m.cells[id].end(), index.cells[id].begin());
    index.cell_active[id] = active;
    if (connectivity && active) {
      touch(source); touch(sink);
      index.cell_out[source].push_back(id); index.cell_in[sink].push_back(id);
      rank_seeds.insert(sink);
    }
    if (active && !index.arc_ids.emplace(index.key(id), id).second) {
      throw pybind11::value_error("duplicate timing arc");
    }
  }
  for (int pin : pins) {
    const int net = m.pin_net.at(pin);
    const int driver = net < 0 ? -1 : m.net_drivers.at(net);
    const bool active = m.pin_active.at(pin) && driver >= 0 && driver != pin;
    int slot = m.net_arc_by_pin.at(pin);
    const bool was_active = slot >= 0 && index.net_active.at(slot);
    if (was_active == active && (!active || index.nets[slot][0] == driver)) continue;
    touch(pin);
    rank_seeds.insert(pin);
    if (slot >= 0 && slot < static_cast<int>(undo.nets)) {
      undo.net_rows.push_back({slot, index.nets[slot], index.net_active[slot]});
    }
    if (was_active) {
      const int previous = index.nets[slot][0];
      touch(previous); eraseId(index.net_out[previous], slot); eraseId(index.net_in[pin], slot);
    }
    if (active) {
      if (slot < 0) {
        slot = index.nets.size(); m.net_arc_by_pin[pin] = slot;
        index.nets.push_back({driver, pin}); index.net_active.push_back(0);
      }
      index.nets[slot] = {driver, pin}; index.net_active[slot] = 1;
      touch(driver); index.net_out[driver].push_back(slot); index.net_in[pin].push_back(slot);
    } else if (slot >= 0) index.net_active[slot] = 0;
  }
  for (int pin : structural) {
    auto& out = index.cell_out[pin]; auto& in = index.cell_in[pin];
    std::sort(out.begin(), out.end()); std::sort(in.begin(), in.end());
    auto& nets = index.net_out[pin];
    std::sort(nets.begin(), nets.end(), [&](int a, int b) { return index.nets[a][1] < index.nets[b][1]; });
    auto& successors = index.successors[pin]; successors.clear();
    for (int id : out) if (id >= index.clock_end) successors.push_back(index.cells[id][1]);
    for (int id : nets) successors.push_back(index.nets[id][1]);
  }
  // Source-forest partitioning only subdivides/removes edges in a qualified
  // combinational DAG. Recompute all current fanin and stop at unchanged ranks.
  using Event = std::pair<int, int>;
  std::priority_queue<Event, std::vector<Event>, std::greater<Event>> pending;
  std::unordered_set<int> queued;
  auto enqueue = [&](int pin) {
    if (queued.insert(pin).second) pending.emplace(index.rank[pin], pin);
  };
  for (int pin : rank_seeds) enqueue(pin);
  std::vector<int> changed_ranks;
  while (!pending.empty()) {
    const int pin = pending.top().second; pending.pop(); queued.erase(pin);
    ++m.statistics.rank_visits;
    int rank = 0, path_level = 0;
    for (int arc : index.cell_in[pin]) {
      if (arc >= index.clock_end) {
        const int source = index.source(arc);
        rank = std::max(rank, index.rank[source] + 1);
        path_level = std::max(path_level, index.path_level[source] + 1);
      }
    }
    for (int arc : index.net_in[pin]) {
      const int source = index.nets[arc][0];
      rank = std::max(rank, index.rank[source] + 1);
      path_level = std::max(path_level, index.path_level[source]);
    }
    if (rank == index.rank[pin] && path_level == index.path_level[pin]) continue;
    captureTimingIndexPin(undo, m, index, pin);
    if (rank != index.rank[pin]) changed_ranks.push_back(pin);
    index.rank[pin] = rank; index.path_level[pin] = path_level;
    for (int sink : index.successors[pin]) enqueue(sink);
  }
  for (int pin : changed_ranks) {
    events.insert(pin);
    const int net = m.pin_net[pin];
    if (net >= 0 && m.net_drivers[net] >= 0) events.insert(m.net_drivers[net]);
    for (int id : index.cell_out[pin]) {
      const int sink = index.cells[id][1]; events.insert(sink);
      for (int edge : index.net_out[sink]) events.insert(index.nets[edge][1]);
    }
  }
  const auto indexed = std::chrono::steady_clock::now();
  m.statistics.local_index_ms += std::chrono::duration<double, std::milli>(indexed - started).count();
  updateTimingEventRows(m, index, program, std::vector<int>(events.begin(), events.end()), undo);
  m.statistics.local_events_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - indexed).count();
  ++m.statistics.local_updates;
  m.canonical_schedule = true;
}

void restoreTimingLocalIndex(TimingMutationState& m, TimingGraphIndex& index, TimingIndexUndo& undo)
{
  for (const auto& row : undo.cell_rows) if (index.cell_active.at(row.id)) eraseArc(index, row.id);
  for (std::size_t id = undo.cells; id < index.cells.size(); ++id) {
    if (index.cell_active[id]) eraseArc(index, id);
  }
  for (const auto& row : undo.cell_rows) {
    index.cells[row.id] = row.row; index.cell_active[row.id] = row.active;
    if (row.active) index.arc_ids.emplace(index.key(row.id), row.id);
  }
  for (const auto& row : undo.net_rows) {
    index.nets[row.id] = row.row; index.net_active[row.id] = row.active;
  }
  for (auto& [pin, row] : undo.rows) {
    index.rank[pin] = row.rank; index.path_level[pin] = row.path_level;
    m.net_arc_by_pin[pin] = row.net_arc;
    index.successors[pin] = std::move(row.successors);
    index.cell_out[pin] = std::move(row.cell_out); index.cell_in[pin] = std::move(row.cell_in);
    index.net_out[pin] = std::move(row.net_out); index.net_in[pin] = std::move(row.net_in);
    m.events.forward_inputs[pin] = std::move(row.forward_inputs);
    m.events.reverse_outputs[pin] = std::move(row.reverse_outputs);
    m.events.forward_successors[pin] = std::move(row.forward_successors);
    m.events.reverse_predecessors[pin] = std::move(row.reverse_predecessors);
  }
  for (std::size_t pin = undo.pins; pin < index.pins.size(); ++pin) {
    const auto found = index.pin_ids.find(index.pins[pin]);
    if (found != index.pin_ids.end() && found->second == static_cast<int>(pin)) index.pin_ids.erase(found);
  }
  index.pins.resize(undo.pins); index.cells.resize(undo.cells); index.cell_active.resize(undo.cells);
  index.nets.resize(undo.nets); index.net_active.resize(undo.nets);
  resizeRows(m, index, undo.pins);
  m.canonical_schedule = undo.canonical_schedule;
}

}  // namespace rebuffer
