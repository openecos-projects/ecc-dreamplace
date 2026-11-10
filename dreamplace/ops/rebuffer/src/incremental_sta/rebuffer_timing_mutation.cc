// SPDX-License-Identifier: BSD-3-Clause
// Native existing-cell resize, local RC write-back and sparse metadata undo.
#include "rebuffer_timing_mutation.hh"

#include "rebuffer_timing_graph.hh"

#include "rebuffer_carrier.hh"

#include "rebuffer_incremental_timing.hh"

#include "rebuffer_timing_local.hh"


#include <algorithm>
#include <cmath>
#include <set>

#include "pybind11/stl.h"


namespace py = pybind11;
namespace rebuffer {
namespace {

void require(bool condition, const std::string& message)
{
  if (!condition) throw py::value_error(message);
}

void replaceArc(TimingMutationState& state, TimingGraphIndex& index,
                 int id, const ProgramCellArc& row)
{
  index.arc_ids.erase(index.key(id));
  state.cells.at(id) = row;
  std::copy(row.begin(), row.end(), index.cells.at(id).begin());
  index.arc_ids.emplace(index.key(id), id);
}

std::array<double, 5> pinValues(const TimingMutationState& state, int pin)
{
  std::array<double, 5> values;
  for (int lane = 0; lane < 5; ++lane) values[lane] = state.pin_values[lane].at(pin);
  return values;
}

}  // namespace

void restoreTimingResize(TimingMutationState& state, TimingGraphIndex& index,
                         const TimingResizeUndo& undo)
{
  state.node_cells.at(undo.node) = undo.previous_cell;
  for (const auto& [id, row] : undo.arcs) replaceArc(state, index, id, row);
  for (const auto& [pin, values] : undo.pins) {
    for (int lane = 0; lane < 5; ++lane) state.pin_values[lane].at(pin) = values[lane];
  }
}

py::dict PersistentTimingGraph::initializeMutations()
{
  require(mutation_state_ == nullptr, "mutation inputs are already initialized");
  require(propagation_program_ != nullptr && timing_state_initialized_,
          "mutations require initialized propagation inputs and timing state");
  auto value = std::make_shared<TimingMutationState>();
  auto& m = *value;
  m.cells = propagation_program_->schedule.cells;
  m.node_names = carrierArray<std::string>(carrier_, "node_names");
  m.cell_names = carrierArray<std::string>(carrier_, "flat_libcell_names");
  m.pin_owner = carrierArray<int>(carrier_, "pin2node_map");
  m.pin_net = carrierArray<int>(carrier_, "pin2net_map");
  m.net_drivers = carrierArray<int>(carrier_, "net2driver_pin_map");
  m.net_pins.resize(m.net_drivers.size());
  m.cell_arc_start = carrierArray<int>(carrier_, "cell_id_2_arc_id_start");
  m.cell_pin_start = carrierArray<int>(carrier_, "cell_id_2_libpin_id_start");
  m.library_pin_names = carrierArray<std::string>(carrier_, "flat_lib_pin_names");
  const int pins = timing_index_->pins.size(), nodes = m.node_names.size();
  m.original_node_count = nodes;
  m.base_node_ids_by_name.reserve(nodes);
  for (int node = 0; node < nodes; ++node) {
    m.base_node_ids_by_name.emplace(m.node_names[node], node);
    if (m.node_names[node].rfind("__dreamplace_rebuffer_", 0) == 0) {
      m.has_rebuffer_prefix_in_base_nodes = true;
    }
  }
  m.cell_ids_by_name.reserve(m.cell_names.size());
  for (int cell = 0; cell < static_cast<int>(m.cell_names.size()); ++cell) {
    m.cell_ids_by_name.emplace(m.cell_names[cell], cell);
  }
  const auto mains = carrierArray<int>(carrier_, "inst_main_id");
  const auto sizes = carrierArray<int>(carrier_, "inst_libcell_offset");
  const auto main_start = carrierArray<int>(carrier_, "main_id_2_cell_id_start");
  const auto offsets = carrierArray<int>(carrier_, "pin_2_libpin_offset");
  require(m.pin_owner.size() == pins && m.pin_net.size() == pins && offsets.size() == pins
              && mains.size() == nodes && sizes.size() == nodes,
          "mutation pin/instance metadata lengths do not match graph");
  const auto arcs = carrier_.attr("flat_libarc_info").cast<py::sequence>();
  for (const auto& item : arcs) {
    const auto row = item.cast<py::sequence>();
    require(row.size() >= 6, "Liberty timing arc metadata is incomplete");
    m.library_arcs.push_back({py::cast<std::string>(row[0]), py::cast<std::string>(row[1]),
                             py::cast<int>(row[4]), py::cast<int>(row[5])});
  }
  auto valid_offsets = [](const auto& values, std::size_t count, std::size_t last) {
    return values.size() == count + 1 && values.front() == 0 && values.back() == last
           && std::is_sorted(values.begin(), values.end());
  };
  require(valid_offsets(m.cell_arc_start, m.cell_names.size(), m.library_arcs.size())
              && valid_offsets(m.cell_pin_start, m.cell_names.size(), m.library_pin_names.size()),
          "mutation Liberty offsets do not cover their tables");
  const auto info = carrier_.attr("flat_libcell_info").cast<py::sequence>();
  require(info.size() == m.cell_names.size(), "Liberty cell metadata lengths do not match");
  for (const auto& item : info) m.cell_mains.push_back(item.cast<py::sequence>()[1].cast<int>());
  m.node_cells.assign(nodes, -1);
  m.node_pins.resize(nodes);
  m.cell_active.assign(m.cells.size(), 1);
  m.pin_active.assign(pins, 1);
  m.net_names = carrierArray<std::string>(carrier_, "net_names");
  const auto clock_nets = carrierArray<std::string>(carrier_, "clock_net_names");
  for (const auto& name : m.net_names) {
    m.base_net_is_clock.push_back(std::find(clock_nets.begin(), clock_nets.end(), name)
                                   != clock_nets.end());
  }
  m.node_arcs.resize(nodes);
  m.pin_ports.resize(pins);
  for (int node = 0; node < nodes; ++node) {
    if (mains[node] < 0) continue;
    require(mains[node] + 1 < main_start.size(), "instance family id is out of range");
    const int cell = main_start[mains[node]] + sizes[node];
    require(cell >= main_start[mains[node]] && cell < main_start[mains[node] + 1]
                && cell < m.cell_names.size(), "instance cell id is out of range");
    m.node_cells[node] = cell;
  }
  for (int pin = 0; pin < pins; ++pin) {
    const int node = m.pin_owner[pin];
    require(node >= 0 && node < nodes, "pin owner is out of range");
    m.node_pins[node].push_back(pin);
    require(m.pin_net[pin] == -1
                || (m.pin_net[pin] >= 0 && m.pin_net[pin] < m.net_pins.size()),
            "mutation pin net id is out of range");
    if (m.pin_net[pin] >= 0) m.net_pins[m.pin_net[pin]].push_back(pin);
    const int cell = m.node_cells[node];
    if (cell < 0) continue;
    const int lib_pin = m.cell_pin_start[cell] + offsets[pin];
    require(offsets[pin] >= 0 && lib_pin < m.cell_pin_start[cell + 1],
            "instance pin is absent from Liberty");
    m.pin_ports[pin] = m.library_pin_names[lib_pin];
  }
  for (int id = 0; id < static_cast<int>(m.cells.size()); ++id) {
    const int node = m.cells[id][7];
    require(node >= 0 && node < nodes, "cell arc owner is out of range");
    m.node_arcs[node].push_back(id);
  }
  m.base_net_pins = m.net_pins;
  m.buffers_by_source_net.resize(m.base_net_pins.size());
  m.base_node_constraint_flags.assign(nodes, 0);
  for (const auto& arc : propagation_program_->setup_arcs) {
    const int owner = m.pin_owner[arc[1]];
    if (owner >= 0 && owner < nodes) m.base_node_constraint_flags[owner] |= 1;
  }
  for (const auto& arc : propagation_program_->recovery_arcs) {
    const int owner = m.pin_owner[arc[1]];
    if (owner >= 0 && owner < nodes) m.base_node_constraint_flags[owner] |= 2;
  }
  const auto caps = timingPinCapacitance(carrier_);
  constexpr std::array<const char*, 5> names{
      "caps", "rcaps", "fcaps", "max_slew", "max_cap"};
  constexpr std::array<const char*, 5> sources{
      "flat_lib_pin_cap", "flat_lib_pin_rcap", "flat_lib_pin_fcap",
      "flat_lib_pin_slew_limit", "flat_lib_pin_cap_limit"};
  for (int lane = 0; lane < 5; ++lane) {
    m.pin_values[lane] = caps[names[lane]].cast<std::vector<double>>();
    m.library_pin_values[lane] = carrierArray<double>(carrier_, sources[lane]);
    require(m.pin_values[lane].size() == pins
                && m.library_pin_values[lane].size() == m.library_pin_names.size(),
            "mutation Liberty pin value lengths do not match");
  }
  // The carrier cache is immutable and can be shared by other owners. Resize
  // edits only this private index's Liberty identities; connectivity is fixed.
  auto index = makeTimingGraphIndex(timing_index_->pins, timing_index_->cells,
      timing_index_->nets, timing_index_->clocks, timing_index_->clock_end);
  initializeTimingLocalIndex(m, *index, *propagation_program_);
  mutation_state_ = std::move(value);
  timing_index_ = std::move(index);
  cached_active_counts_valid_ = false;
  topology_revision_ = ++next_topology_revision_;
  markElectricalFullDirty();
  return summary();
}

py::dict PersistentTimingGraph::mutationView() const
{
  require(mutation_state_ != nullptr, "mutation inputs are not initialized");
  const auto& m = *mutation_state_;
  py::array_t<int64_t> cells(
      {static_cast<py::ssize_t>(m.cells.size()), py::ssize_t{8}},
      {py::ssize_t{8 * sizeof(int64_t)}, py::ssize_t{sizeof(int64_t)}});
  for (std::size_t id = 0; id < m.cells.size(); ++id) {
    std::copy(m.cells[id].begin(), m.cells[id].end(), cells.mutable_data() + id * 8);
  }
  py::dict result;
  result["cell_arcs"] = cells;
  py::array_t<int64_t> nets(
      {static_cast<py::ssize_t>(timing_index_->nets.size()), py::ssize_t{2}},
      {py::ssize_t{2 * sizeof(int64_t)}, py::ssize_t{sizeof(int64_t)}});
  for (std::size_t id = 0; id < timing_index_->nets.size(); ++id) {
    std::copy(timing_index_->nets[id].begin(), timing_index_->nets[id].end(), nets.mutable_data() + id * 2);
  }
  result["net_arcs"] = nets;
  result["net_arc_active"] = timing_index_->net_active;
  result["pin_rank"] = timing_index_->rank;
  result["canonical_schedule"] = m.canonical_schedule;
  std::vector<std::string> masters;
  for (int cell : m.node_cells) masters.push_back(cell < 0 ? "" : m.cell_names[cell]);
  result["node_master_names"] = masters;
  result["node_names"] = m.node_names;
  result["pin_names"] = timing_index_->pins;
  result["net_names"] = m.net_names;
  result["pin_net"] = m.pin_net;
  result["net_drivers"] = m.net_drivers;
  result["net_pins"] = m.net_pins;
  result["pin_active"] = m.pin_active;
  result["cell_active"] = m.cell_active;
  py::list buffers;
  for (const auto& buffer : m.buffers) {
    py::dict row;
    row["action_id"] = buffer.action_id; row["source_net"] = buffer.source_net;
    row["input_pin"] = buffer.input_pin; row["output_pin"] = buffer.output_pin;
    row["output_net"] = buffer.output_net; row["active"] = buffer.active;
    row["arc_ids"] = buffer.arcs; row["node"] = buffer.node;
    buffers.append(row);
  }
  result["buffers"] = buffers;
  result["pin_caps"] = writeState(m.pin_values[0]);
  result["pin_rcaps"] = writeState(m.pin_values[1]);
  result["pin_fcaps"] = writeState(m.pin_values[2]);
  result["pin_max_slew"] = writeState(m.pin_values[3]);
  result["pin_max_cap"] = writeState(m.pin_values[4]);
  bool index_matches = timing_index_->arc_ids.size()
                         == std::count(m.cell_active.begin(), m.cell_active.end(), 1);
  for (std::size_t id = 0; id < m.cells.size(); ++id) {
    if (!m.cell_active[id]) continue;
    index_matches &= std::equal(m.cells[id].begin(), m.cells[id].end(),
                                timing_index_->cells[id].begin())
                     && timing_index_->arc_ids.at(timing_index_->key(id)) == id;
  }
  result["index_matches"] = index_matches;
  return result;
}

py::dict PersistentTimingGraph::mutationStatistics() const
{
  require(mutation_state_ != nullptr, "mutation inputs are not initialized");
  const auto& s = mutation_state_->statistics;
  py::dict result;
  result["index_builds"] = s.index_builds; result["schedule_builds"] = s.schedule_builds;
  result["rc_builds"] = s.rc_builds;
  result["rc_nodes"] = s.rc_nodes; result["rc_pin_reads"] = s.rc_pin_reads;
  result["index_ms"] = s.index_ms; result["schedule_ms"] = s.schedule_ms;
  result["rc_build_ms"] = s.rc_build_ms; result["rc_evaluate_ms"] = s.rc_evaluate_ms;
  result["local_updates"] = s.local_updates; result["rank_visits"] = s.rank_visits;
  result["event_rows"] = s.event_rows;
  result["local_index_ms"] = s.local_index_ms; result["local_events_ms"] = s.local_events_ms;
  return result;
}

py::dict PersistentTimingGraph::resizeCell(const std::string& instance_name,
                                           const std::string& master_name)
{
  require(mutation_state_ != nullptr && timing_state_initialized_,
          "resize requires initialized mutation inputs and timing state");
  auto& m = *mutation_state_;
  int node = -1;
  const auto base_node_it = m.base_node_ids_by_name.find(instance_name);
  if (base_node_it != m.base_node_ids_by_name.end()) {
    node = base_node_it->second;
  } else {
    for (int id = m.original_node_count; id < static_cast<int>(m.node_names.size()); ++id) {
      if (m.node_names[id] == instance_name) {
        node = id;
        break;
      }
    }
  }
  const auto cell_it = m.cell_ids_by_name.find(master_name);
  require(node >= 0, "resize instance is absent from graph");
  require(cell_it != m.cell_ids_by_name.end(), "resize master is absent from Liberty");
  const int cell = cell_it->second;
  const int previous = m.node_cells[node];
  require(previous >= 0 && m.cell_mains[cell] == m.cell_mains[previous],
          "resize requires an equivalent instance master");
  require(!m.node_arcs[node].empty(), "resize instance has no timing arcs");
  if (node >= m.original_node_count) {
    for (const auto& buffer : m.buffers) {
      if (buffer.active && buffer.node == node) {
        py::dict action;
        action["action_id"] = buffer.action_id;
        action["buffer_master_name"] = master_name;
        return resizeBuffer(buffer.source_net, action);
      }
    }
  }
  for (auto id : m.node_arcs[node]) {
    require(id >= timing_index_->clock_end, "sequential cell resize is not yet supported");
  }
  if (node < m.original_node_count) {
    require(!(m.base_node_constraint_flags[node] & 1),
            "setup-constrained cell resize is not yet supported");
    require(!(m.base_node_constraint_flags[node] & 2),
            "recovery-constrained cell resize is not yet supported");
  } else {
    for (const auto& arc : propagation_program_->setup_arcs) {
      require(m.pin_owner[arc[1]] != node, "setup-constrained cell resize is not yet supported");
    }
    for (const auto& arc : propagation_program_->recovery_arcs) {
      require(m.pin_owner[arc[1]] != node, "recovery-constrained cell resize is not yet supported");
    }
  }
  py::dict result;
  result["status"] = "ok";
  result["instance_name"] = instance_name;
  result["previous_master"] = m.cell_names[previous];
  result["master_name"] = master_name;
  result["changed"] = previous != cell;
  if (previous == cell) return result;

  auto undo = std::make_shared<TimingResizeUndo>();
  undo->node = node;
  undo->previous_cell = previous;
  std::vector<std::pair<int, ProgramCellArc>> rows;
  for (auto id : m.node_arcs[node]) {
    auto row = m.cells[id];
    const int lut = equivalentTimingArc(m.cell_arc_start[cell], m.cell_arc_start[cell + 1],
        row[3], row[6], [&](int index) -> const TimingLibraryArc& {
          return m.library_arcs.at(index);
        });
    for (int table = 0; table < 4; ++table) {
      require(lut >= 0 && lut < propagation_program_->luts[table].rows,
              "resize timing LUT row is out of range");
    }
    undo->arcs.emplace_back(id, row);
    row[2] = cell; row[3] = lut; row[6] = lut - m.cell_arc_start[cell];
    rows.emplace_back(id, row);
  }
  std::vector<std::pair<int, std::array<double, 5>>> proposed_pins;
  std::set<int> changed_nets;
  for (auto pin : m.node_pins[node]) {
    const auto begin = m.library_pin_names.begin() + m.cell_pin_start[cell];
    const auto end = m.library_pin_names.begin() + m.cell_pin_start[cell + 1];
    const auto found = std::find(begin, end, m.pin_ports[pin]);
    require(found != end && std::find(found + 1, end, m.pin_ports[pin]) == end,
            "resize master pin schema differs");
    const auto index = found - m.library_pin_names.begin();
    const auto old_values = pinValues(m, pin);
    undo->pins.emplace_back(pin, old_values);
    std::array<double, 5> proposed;
    for (int lane = 0; lane < 5; ++lane) {
      const double value = m.library_pin_values[lane][index];
      require(std::isfinite(value) && value >= 0, "resize Liberty pin values must be finite and nonnegative");
      proposed[lane] = value;
      if (lane < 3 && value != old_values[lane]) changed_nets.insert(m.pin_net[pin]);
    }
    proposed_pins.emplace_back(pin, proposed);
  }
  std::set<int> source_nets;
  for (int net : changed_nets) {
    require(net >= 0 && net < m.net_drivers.size(), "resize pin has no RC net");
    const int source = sourceNetForMutationNet(m, net);
    require(source >= 0, "resize input has no source net provenance");
    source_nets.insert(source);
  }
  // Validate Liberty before writing. Register the local metadata undo before
  // RC evaluation so a late failure can restore the changed pin caps as well.
  const auto checkpoint_size = journal_.size();
  try {
    JournalEntry entry;
    entry.previous_epoch = epoch_;
    entry.previous_topology_revision = topology_revision_;
    entry.sequence = next_journal_sequence_++;
    entry.resize_undo = std::move(undo);
    journal_.push_back(std::move(entry));
    ++epoch_;
    m.node_cells[node] = cell;
    for (const auto& [id, row] : rows) replaceArc(m, *timing_index_, id, row);
    for (const auto& [pin, values] : proposed_pins) {
      for (int lane = 0; lane < 5; ++lane) m.pin_values[lane][pin] = values[lane];
      markElectricalDirtyPin(pin);
    }
    topology_revision_ = ++next_topology_revision_;
    std::vector<TimingNetRcValues> rc_values;
    for (int source : source_nets) {
      const auto topology = topologyState(source);
      rc_values.push_back(recomputeTimingNetRc(*topology, m.pin_values));
    }
    applyTimingRc(rc_values);
    result["affected_nets"] = std::vector<int>(changed_nets.begin(), changed_nets.end());
    result["updated_arc_count"] = rows.size();
    result["state"] = summary();
    return result;
  } catch (...) {
    restoreTo(checkpoint_size);
    throw;
  }
}

}  // namespace rebuffer

// --- rebuffer_timing_buffer_state.cc ---
// Native buffer records and reversible electrical partition materialization.
#include "rebuffer_incremental_topology.hh"


#include <algorithm>
#include <cmath>
#include <set>
#include <unordered_map>

namespace rebuffer {
namespace {

void require(bool condition, const char* message)
{
  if (!condition) throw pybind11::value_error(message);
}

std::array<double, 5> values(const TimingMutationState& m, int pin)
{
  std::array<double, 5> result;
  for (int lane = 0; lane < 5; ++lane) result[lane] = m.pin_values[lane].at(pin);
  return result;
}

int libraryPin(const TimingMutationState& m, int cell, const std::string& port)
{
  const auto begin = m.library_pin_names.begin() + m.cell_pin_start.at(cell);
  const auto end = m.library_pin_names.begin() + m.cell_pin_start.at(cell + 1);
  const auto found = std::find(begin, end, port);
  require(found != end && std::find(found + 1, end, port) == end,
          "buffer Liberty pin schema is missing or ambiguous");
  return found - m.library_pin_names.begin();
}

}  // namespace

void requireSynchronizedTimingTopology(const TimingMutationState& m,
                                       const IncrementalTopologyState& topology)
{
  const int source = topology.net_id;
  require(source >= 0 && source < static_cast<int>(m.base_net_pins.size())
              && topology.driver_pin_id == m.net_drivers[source],
          "timing RC topology driver does not match source net");
  require(topology.buffers.empty() || m.canonical_schedule,
          "native topology buffer records are not synchronized");
  const auto epoch = m.topology_epochs.find(source);
  require(epoch == m.topology_epochs.end() || epoch->second == topology.epoch,
          "native topology changed outside graph timing mutation");
  const auto& source_buffers = m.buffers_by_source_net.at(source);
  std::size_t active = 0;
  for (int id : source_buffers) {
    if (m.buffers[id].active) ++active;
  }
  require(active == topology.buffers.size(), "native topology buffer records are not synchronized");
  for (const auto& buffer : topology.buffers) {
    int found_id = -1;
    for (int id : source_buffers) {
      const auto& row = m.buffers[id];
      if (row.action_id == buffer.action_id && row.active) {
        found_id = id;
        break;
      }
    }
    require(found_id >= 0 && m.buffers[found_id].logical_input == buffer.input_node
                && m.buffers[found_id].logical_output == buffer.output_node
                && m.buffers[found_id].split_ratio == buffer.split_ratio
                && m.cell_names.at(m.node_cells.at(m.buffers[found_id].node)) == buffer.master_name,
            "native topology buffer identity is not synchronized");
  }
}

void requireSynchronizedTimingTopologies(
    const TimingMutationState& m,
    const std::map<int, std::shared_ptr<IncrementalTopologyState>>& topologies)
{
  for (const auto& [_, state_ptr] : topologies) {
    requireSynchronizedTimingTopology(m, *state_ptr);
  }
}

std::shared_ptr<TimingTopologyUndo> captureTimingTopology(
    const TimingMutationState& m, const TimingGraphIndex& index, int source_net)
{
  auto undo = std::make_shared<TimingTopologyUndo>();
  undo->pins = m.pin_owner.size(); undo->nodes = m.node_names.size();
  undo->nets = m.net_drivers.size(); undo->arcs = m.cells.size();
  undo->buffers = m.buffers.size(); undo->index_undo = captureTimingIndex(m, index);
  undo->source_net = source_net;
  const auto epoch = m.topology_epochs.find(source_net);
  if (epoch != m.topology_epochs.end()) undo->previous_topology_epoch = epoch->second;
  std::set<int> pins(m.base_net_pins.at(source_net).begin(),
                      m.base_net_pins.at(source_net).end());
  std::set<int> nets{source_net};
  for (int id : m.buffers_by_source_net.at(source_net)) {
    const auto& buffer = m.buffers[id];
    undo->previous_buffers.emplace_back(id, buffer);
    undo->previous_cells.emplace_back(buffer.node, m.node_cells[buffer.node]);
    pins.insert(buffer.input_pin); pins.insert(buffer.output_pin);
    nets.insert(buffer.output_net);
    for (int arc : buffer.arcs) {
      undo->previous_arcs.push_back({arc, m.cells[arc], m.cell_active[arc]});
    }
  }
  for (int pin : pins) {
    undo->previous_pins.push_back({pin, m.pin_net[pin], m.pin_active[pin], values(m, pin)});
  }
  for (int net : nets) undo->previous_nets.push_back({net, m.net_drivers[net], m.net_pins[net]});
  return undo;
}

void restoreTimingTopology(TimingMutationState& m, TimingGraphIndex& index,
                           const TimingTopologyUndo& undo)
{
  restoreTimingLocalIndex(m, index, *undo.index_undo);
  index.pins.resize(undo.pins);
  m.pin_owner.resize(undo.pins); m.pin_net.resize(undo.pins);
  m.pin_ports.resize(undo.pins); m.pin_active.resize(undo.pins);
  for (auto& lane : m.pin_values) lane.resize(undo.pins);
  m.node_names.resize(undo.nodes); m.node_cells.resize(undo.nodes);
  m.node_pins.resize(undo.nodes); m.node_arcs.resize(undo.nodes);
  m.net_names.resize(undo.nets); m.net_drivers.resize(undo.nets);
  m.net_pins.resize(undo.nets);
  m.cells.resize(undo.arcs); m.cell_active.resize(undo.arcs);
  for (std::size_t id = undo.buffers; id < m.buffers.size(); ++id) {
    const auto& buf = m.buffers[id];
    m.buffer_id_by_action_id.erase(buf.action_id);
    if (buf.source_net >= 0 && buf.source_net < static_cast<int>(m.buffers_by_source_net.size())) {
      auto& vec = m.buffers_by_source_net[buf.source_net];
      vec.erase(std::remove(vec.begin(), vec.end(), static_cast<int>(id)), vec.end());
    }
  }
  m.buffers.resize(undo.buffers);
  for (const auto& [id, buffer] : undo.previous_buffers) {
    m.buffers[id] = buffer;
    m.node_arcs[buffer.node] = buffer.arcs;
  }
  for (const auto& [node, cell] : undo.previous_cells) m.node_cells[node] = cell;
  for (const auto& pin : undo.previous_pins) {
    m.pin_net[pin.id] = pin.net; m.pin_active[pin.id] = pin.active;
    for (int lane = 0; lane < 5; ++lane) m.pin_values[lane][pin.id] = pin.values[lane];
  }
  for (const auto& arc : undo.previous_arcs) {
    m.cells[arc.id] = arc.row; m.cell_active[arc.id] = arc.active;
  }
  for (const auto& net : undo.previous_nets) {
    m.net_drivers[net.id] = net.driver; m.net_pins[net.id] = net.pins;
  }
  if (undo.previous_topology_epoch < 0) m.topology_epochs.erase(undo.source_net);
  else m.topology_epochs[undo.source_net] = undo.previous_topology_epoch;
}

void synchronizeTimingBuffers(TimingMutationState& m, TimingGraphIndex& index,
                               const TimingPropagationProgram& program,
                               const IncrementalTopologyState& topology)
{
  const int source = topology.net_id;
  require(source >= 0 && source < static_cast<int>(m.base_net_pins.size()),
          "buffer mutation requires an original source net");
  require(topology.driver_pin_id == m.net_drivers[source],
          "buffer source driver does not match timing graph");
  std::vector<int> records;
  std::set<int> active_actions;
  for (const auto& physical : topology.buffers) {
    require(active_actions.insert(physical.action_id).second, "duplicate buffer action");
    const auto master_it = m.cell_ids_by_name.find(physical.master_name);
    require(master_it != m.cell_ids_by_name.end(), "buffer master is absent from Liberty");
    const int cell = master_it->second;
    const int first = m.cell_arc_start[cell], last = m.cell_arc_start[cell + 1];
    require(first < last, "buffer master has no timing arcs");
    const auto& signature = m.library_arcs.at(first);
    for (int lut = first; lut < last; ++lut) {
      const auto& arc = m.library_arcs.at(lut);
      require(arc.source == signature.source && arc.sink == signature.sink
                  && arc.sense == 1 && arc.trigger == 0,
              "buffer mutation requires a noninverting combinational master");
      for (int table = 0; table < 4; ++table) {
        require(lut < program.luts[table].rows, "buffer LUT row is out of range");
      }
    }
    const int input_lib = libraryPin(m, cell, signature.source);
    const int output_lib = libraryPin(m, cell, signature.sink);
    for (const auto& lane : m.library_pin_values) {
      require(std::isfinite(lane[input_lib]) && lane[input_lib] >= 0
                  && std::isfinite(lane[output_lib]) && lane[output_lib] >= 0,
              "buffer Liberty pin values must be finite and nonnegative");
    }
    const auto action_it = m.buffer_id_by_action_id.find(physical.action_id);
    int id;
    if (action_it == m.buffer_id_by_action_id.end()) {
      const std::string name = "__dreamplace_rebuffer_" + std::to_string(physical.action_id);
      require(!m.has_rebuffer_prefix_in_base_nodes || m.base_node_ids_by_name.count(name) == 0,
              "buffer instance identity collides with the base graph");
      id = m.buffers.size();
      TimingBufferRecord buffer;
      buffer.source_net = source; buffer.action_id = physical.action_id;
      buffer.node = m.node_names.size(); buffer.input_pin = m.pin_owner.size();
      buffer.output_pin = buffer.input_pin + 1; buffer.output_net = m.net_drivers.size();
      m.node_names.push_back(name); m.node_cells.push_back(cell);
      m.node_pins.push_back({buffer.input_pin, buffer.output_pin});
      m.node_arcs.emplace_back();
      for (const auto& port : {signature.source, signature.sink}) {
        const std::string pin_name = name + ":" + port;
        require(!index.pin_ids.count(pin_name), "buffer pin identity collides with graph");
        index.pins.push_back(pin_name);
        m.pin_owner.push_back(buffer.node); m.pin_ports.push_back(port);
        m.pin_net.push_back(-1); m.pin_active.push_back(1);
        for (auto& lane : m.pin_values) lane.push_back(0);
      }
      m.net_names.push_back(name + "_out");
      m.net_drivers.push_back(buffer.output_pin); m.net_pins.emplace_back();
      m.buffers.push_back(buffer);
      m.buffers_by_source_net.at(source).push_back(id);
      m.buffer_id_by_action_id.emplace(buffer.action_id, id);
    } else {
      id = action_it->second;
      const auto* found = &m.buffers[id];
      require(found->source_net == source, "buffer action identity belongs to another source net");
      require(m.pin_ports[found->input_pin] == signature.source
                  && m.pin_ports[found->output_pin] == signature.sink,
              "buffer resize changes pin port identities");
      const int previous = m.node_cells[found->node];
      require(previous < 0 || m.cell_mains[previous] == m.cell_mains[cell],
              "buffer resize requires an equivalent master");
    }
    auto& buffer = m.buffers[id];
    buffer.active = true;
    buffer.logical_input = physical.input_node; buffer.logical_output = physical.output_node;
    buffer.split_ratio = physical.split_ratio;
    m.node_cells[buffer.node] = cell;
    m.pin_active[buffer.input_pin] = m.pin_active[buffer.output_pin] = 1;
    for (int lane = 0; lane < 5; ++lane) {
      m.pin_values[lane][buffer.input_pin] = m.library_pin_values[lane].at(input_lib);
      m.pin_values[lane][buffer.output_pin] = m.library_pin_values[lane].at(output_lib);
    }
    if (buffer.arcs.size() != static_cast<std::size_t>(last - first)) {
      for (int arc : buffer.arcs) m.cell_active[arc] = 0;
      buffer.arcs.clear();
      for (int lut = first; lut < last; ++lut) {
        buffer.arcs.push_back(m.cells.size());
        m.cells.push_back({}); m.cell_active.push_back(1);
      }
    }
    for (int offset = 0; offset < last - first; ++offset) {
      const int slot = buffer.arcs[offset];
      m.cells[slot] = {buffer.input_pin, buffer.output_pin, cell, first + offset,
                       1, 0, offset, buffer.node};
      m.cell_active[slot] = 1;
    }
    m.node_arcs[buffer.node] = buffer.arcs;
    records.push_back(id);
  }
  for (int id : m.buffers_by_source_net.at(source)) {
    auto& buffer = m.buffers[id];
    if (active_actions.count(buffer.action_id)) continue;
    buffer.active = false;
    m.node_cells[buffer.node] = -1;
    m.pin_active[buffer.input_pin] = m.pin_active[buffer.output_pin] = 0;
    m.pin_net[buffer.input_pin] = m.pin_net[buffer.output_pin] = -1;
    for (auto& lane : m.pin_values) lane[buffer.input_pin] = lane[buffer.output_pin] = 0;
    for (int arc : buffer.arcs) m.cell_active[arc] = 0;
    m.net_drivers[buffer.output_net] = -1; m.net_pins[buffer.output_net].clear();
  }
  const auto partitions = topologyPartitions(topology);
  std::vector<int> nets{source};
  m.net_pins[source] = {topology.driver_pin_id};
  m.pin_net[topology.driver_pin_id] = source;
  for (int id : records) {
    const auto& buffer = m.buffers[id];
    nets.push_back(buffer.output_net);
    m.net_pins[buffer.output_net] = {buffer.output_pin};
    m.net_drivers[buffer.output_net] = buffer.output_pin;
    m.pin_net[buffer.output_pin] = buffer.output_net;
  }
  std::set<int> physical_pins{topology.driver_pin_id};
  for (std::size_t part = 0; part < partitions.loads.size(); ++part) {
    const int net = nets[part];
    for (int pin : partitions.loads[part]) {
      require(pin >= 0 && pin < program.pin_count && physical_pins.insert(pin).second,
              "buffer topology has invalid or duplicate physical loads");
      m.net_pins[net].push_back(pin); m.pin_net[pin] = net;
    }
  }
  require(std::vector<int>(physical_pins.begin(), physical_pins.end()) == m.base_net_pins[source],
          "buffer topology physical pins do not match source net");
  for (std::size_t slot = 0; slot < records.size(); ++slot) {
    const auto& buffer = m.buffers[records[slot]];
    const int net = nets[partitions.input_partition[slot]];
    m.net_pins[net].push_back(buffer.input_pin); m.pin_net[buffer.input_pin] = net;
  }
  for (int net : nets) std::sort(m.net_pins[net].begin(), m.net_pins[net].end());
  m.topology_epochs[source] = topology.epoch;
}

}  // namespace rebuffer

// --- rebuffer_timing_buffer_rc.cc ---
// Evaluate source-local RC directly, without dense namespaces or Python arrays.

#include <algorithm>
#include <chrono>
#include <unordered_map>

namespace rebuffer {
namespace {

TimingNetRcValues timingNetRcValues(const TimingMutationState& m,
    const IncrementalTopologyState& topology, const CompactRcForest& forest,
    const std::array<std::vector<double>, 5>& pin_values, int original_pin_count)
{
  requireSynchronizedTimingTopology(m, topology);
  std::unordered_map<int, int> buffer_pins;
  for (const auto& buffer : topology.buffers) {
    const auto found = std::find_if(m.buffers.begin(), m.buffers.end(), [&](const auto& row) {
      return row.source_net == topology.net_id && row.action_id == buffer.action_id && row.active;
    });
    if (found == m.buffers.end()) throw pybind11::value_error("RC buffer has no active timing record");
    buffer_pins.emplace(buffer.input_node, found->input_pin);
    buffer_pins.emplace(buffer.output_node, found->output_pin);
  }
  for (const auto& [node, loads] : topology.node_loads) {
    if (loads.size() != 1 || loads.front() != node) {
      throw pybind11::value_error("native timing RC requires explicit physical pin vertices");
    }
  }
  TimingNetRcValues result;
  std::vector<int> original_physical, physical_pins(forest.node_ids.size(), -1), selected;
  for (int i = 0; i < static_cast<int>(forest.node_ids.size()); ++i) {
    const int node = forest.node_ids[i];
    const auto buffer = buffer_pins.find(node);
    if (buffer != buffer_pins.end()) physical_pins[i] = buffer->second;
    else if (node < original_pin_count) {
      original_physical.push_back(node);
      physical_pins[i] = node;
    }
    if (physical_pins[i] >= 0) {
      result.pins.push_back(physical_pins[i]);
      selected.push_back(i);
    }
  }
  if (original_physical != m.base_net_pins.at(topology.net_id)) {
    throw pybind11::value_error("timing RC topology physical pins do not match source net");
  }
  const auto wire_cap = compactRcWireCaps(forest);
  // Canonical timing state consumes rise/fall RC only. The public adapter
  // still computes generic mode for callers that request it.
  for (int mode = 0; mode < 2; ++mode) {
    std::vector<double> caps(forest.node_ids.size(), 0);
    for (int i = 0; i < static_cast<int>(caps.size()); ++i) {
      if (physical_pins[i] >= 0) caps[i] = pin_values[mode + 1].at(physical_pins[i]);
      caps[i] += wire_cap[i];
    }
    const auto values = evaluateCompactRc(forest, caps);
    for (int field = 0; field < 3; ++field) {
      auto& output = result.values[2 * field + mode];
      output.reserve(selected.size());
      for (int i : selected) output.push_back(values[field][i]);
    }
  }
  return result;
}

}  // namespace

TimingNetRcValues PersistentTimingGraph::recomputeTimingNetRc(
    const IncrementalTopologyState& topology,
    const std::array<std::vector<double>, 5>& pin_values)
{
  const auto started = std::chrono::steady_clock::now();
  const auto forest = compactRcForest(topology, propagation_program_->pin_count);
  const auto built = std::chrono::steady_clock::now();
  auto result = timingNetRcValues(*mutation_state_, topology, forest, pin_values,
                                  propagation_program_->pin_count);
  auto& stats = mutation_state_->statistics;
  ++stats.rc_builds;
  stats.rc_nodes += forest.node_ids.size();
  stats.rc_pin_reads += 2 * result.pins.size();
  stats.rc_build_ms += std::chrono::duration<double, std::milli>(built - started).count();
  stats.rc_evaluate_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - built).count();
  return result;
}

}  // namespace rebuffer

// --- rebuffer_timing_graph_buffer.cc ---
// Atomically connect native split/remove/resize to graph-owned timing state.

#include <algorithm>
#include <exception>


namespace py = pybind11;
namespace rebuffer {

void PersistentTimingGraph::requireTimingMutationOwner() const
{
  if (!timing_state_initialized_ || propagation_program_ == nullptr || mutation_state_ == nullptr) {
    throw py::value_error("buffer timing mutation requires initialized graph state and inputs");
  }
}

py::dict PersistentTimingGraph::mutateTimingTopology(int net_id, const py::dict& action,
                                                    TimingTopologyOperation operation)
{
  requireTimingMutationOwner();
  if (net_id < 0 || net_id >= static_cast<int>(mutation_state_->base_net_pins.size())
      || mutation_state_->base_net_is_clock.at(net_id)) {
    throw py::value_error("buffer timing mutation requires an original signal source net");
  }
  const auto topology = topologyState(net_id);
  requireSynchronizedTimingTopology(*mutation_state_, *topology);
  // Only this source topology can be changed by one mutation.  The previous
  // guard called checkpoint(), which copied every attached net topology for
  // every candidate action.  That made apply grow with campaign history even
  // though the journal already contains sparse timing undo for the mutation.
  const auto topology_before = checkpointIncrementalTopology(*topology);
  const std::size_t journal_size_before = journal_.size();
  const std::uint64_t epoch_before = epoch_;
  const std::uint64_t topology_revision_before = topology_revision_;
  const auto propagation_mark_before = propagation_mark_;
  try {
    std::string reason;
    py::dict result;
    bool valid = false;
    if (operation == TimingTopologyOperation::insert) {
      valid = applyIncrementalBufferSplit(*topology, action, reason, result);
    } else if (operation == TimingTopologyOperation::remove) {
      const int id = action["action_id"].cast<int>();
      valid = !topology->general_removal && !topology->buffers.empty()
                      && topology->buffers.back().action_id == id
                  ? removeIncrementalBuffer(*topology, id, reason, result)
                  : removeIncrementalBufferAnywhere(*topology, id, reason, result);
    } else {
      valid = resizeIncrementalBuffer(*topology, action, reason, result);
    }
    if (!valid) throw py::value_error(reason);
    if (result["status"].cast<std::string>() == "noop") return result;
    auto undo = captureTimingTopology(*mutation_state_, *timing_index_, net_id);
    JournalEntry entry;
    entry.previous_epoch = epoch_;
    entry.previous_topology_revision = topology_revision_;
    entry.sequence = next_journal_sequence_++;
    entry.topology_undo = undo;
    journal_.push_back(std::move(entry));
    ++epoch_;
    synchronizeTimingBuffers(*mutation_state_, *timing_index_, *propagation_program_, *topology);
    updateTimingLocalIndex(*mutation_state_, *timing_index_, *propagation_program_, *undo);
    if (cached_active_counts_valid_) {
      for (const auto& pin : undo->previous_pins) {
        cached_active_pins_ += static_cast<int>(mutation_state_->pin_active[pin.id])
                             - static_cast<int>(pin.active);
      }
      for (std::size_t p = undo->pins; p < mutation_state_->pin_active.size(); ++p) {
        cached_active_pins_ += static_cast<std::size_t>(mutation_state_->pin_active[p]);
      }
      for (const auto& arc : undo->previous_arcs) {
        cached_active_cell_arcs_ += static_cast<int>(mutation_state_->cell_active[arc.id])
                                  - static_cast<int>(arc.active);
      }
      for (std::size_t a = undo->arcs; a < mutation_state_->cell_active.size(); ++a) {
        cached_active_cell_arcs_ += static_cast<std::size_t>(mutation_state_->cell_active[a]);
      }
      for (const auto& row : undo->index_undo->net_rows) {
        cached_active_net_arcs_ += static_cast<int>(timing_index_->net_active[row.id])
                                 - static_cast<int>(row.active);
      }
      for (std::size_t n = undo->index_undo->nets; n < timing_index_->net_active.size(); ++n) {
        cached_active_net_arcs_ += static_cast<std::size_t>(timing_index_->net_active[n]);
      }
    }
    topology_revision_ = ++next_topology_revision_;
    for (const auto& pin : undo->previous_pins) markElectricalDirtyPin(pin.id);
    for (const auto& net : undo->previous_nets) {
      for (const int pin : net.pins) markElectricalDirtyPin(pin);
    }
    for (const int pin : mutation_state_->net_pins.at(net_id)) markElectricalDirtyPin(pin);
    for (int buf_id : mutation_state_->buffers_by_source_net.at(net_id)) {
      const auto& row = mutation_state_->buffers[buf_id];
      markElectricalDirtyPin(row.input_pin);
      markElectricalDirtyPin(row.output_pin);
      if (row.output_net >= 0
          && row.output_net < static_cast<int>(mutation_state_->net_pins.size())) {
        for (const int pin : mutation_state_->net_pins[row.output_net]) {
          markElectricalDirtyPin(pin);
        }
      }
    }
    const auto rc = recomputeTimingNetRc(*topology, mutation_state_->pin_values);
    std::vector<int> inactive;
    for (const auto& pin : undo->previous_pins) {
      if (pin.active && !mutation_state_->pin_active.at(pin.id)) inactive.push_back(pin.id);
    }
    applyTimingRc({rc}, inactive);
    const int action_id = result["action_id"].cast<int>();
    for (int buf_id : mutation_state_->buffers_by_source_net.at(net_id)) {
      const auto& record = mutation_state_->buffers[buf_id];
      if (record.action_id == action_id) {
        result["timing_pin_ids"] = std::vector<int>{record.input_pin, record.output_pin};
        result["timing_arc_ids"] = record.arcs;
        result["timing_net_id"] = record.output_net;
        break;
      }
    }
    result["state"] = summary();
    return result;
  } catch (...) {
    const auto failure = std::current_exception();
    std::string restore_reason;
    cached_active_counts_valid_ = false;
    restoreTo(journal_size_before);
    epoch_ = epoch_before;
    topology_revision_ = topology_revision_before;
    propagation_mark_ = propagation_mark_before;
    if (!restoreIncrementalTopology(*topology, *topology_before, restore_reason)) {
      throw py::value_error("native topology mutation rollback failed: " + restore_reason);
    }
    std::rethrow_exception(failure);
  }
}

}  // namespace rebuffer
