// SPDX-License-Identifier: BSD-3-Clause
// Native buffer records and reversible electrical partition materialization.
#include "rebuffer_timing_buffer_state.hh"
#include "rebuffer_topology_partitions.hh"
#include "rebuffer_timing_local_index.hh"

#include <algorithm>
#include <cmath>
#include <set>

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
  std::size_t active = 0;
  for (const auto& row : m.buffers) {
    if (row.source_net == source && row.active) ++active;
  }
  require(active == topology.buffers.size(), "native topology buffer records are not synchronized");
  for (const auto& buffer : topology.buffers) {
    const auto found = std::find_if(m.buffers.begin(), m.buffers.end(), [&](const auto& row) {
      return row.source_net == source && row.action_id == buffer.action_id && row.active;
    });
    require(found != m.buffers.end() && found->logical_input == buffer.input_node
                && found->logical_output == buffer.output_node && found->split_ratio == buffer.split_ratio
                && m.cell_names.at(m.node_cells.at(found->node)) == buffer.master_name,
            "native topology buffer identity is not synchronized");
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
  for (int id = 0; id < static_cast<int>(m.buffers.size()); ++id) {
    const auto& buffer = m.buffers[id];
    if (buffer.source_net != source_net) continue;
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
    const auto master = std::find(m.cell_names.begin(), m.cell_names.end(), physical.master_name);
    require(master != m.cell_names.end(), "buffer master is absent from Liberty");
    const int cell = master - m.cell_names.begin();
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
    auto found = std::find_if(m.buffers.begin(), m.buffers.end(), [&](const auto& row) {
      return row.action_id == physical.action_id;
    });
    int id;
    if (found == m.buffers.end()) {
      const std::string name = "__dreamplace_rebuffer_" + std::to_string(physical.action_id);
      require(std::find(m.node_names.begin(), m.node_names.end(), name) == m.node_names.end(),
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
    } else {
      id = found - m.buffers.begin();
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
  for (auto& buffer : m.buffers) {
    if (buffer.source_net != source || active_actions.count(buffer.action_id)) continue;
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
