// SPDX-License-Identifier: BSD-3-Clause
// Native existing-cell resize, local RC write-back and sparse metadata undo.
#include "rebuffer_timing_mutation.hh"
#include "rebuffer_timing_graph.hh"
#include "rebuffer_carrier.hh"
#include "rebuffer_timing_payload.hh"
#include "rebuffer_timing_buffer_state.hh"
#include "rebuffer_timing_local_index.hh"

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
