// SPDX-License-Identifier: BSD-3-Clause
// Keep canonical index extents; populate timing/physical records only for one tree.
#include "rebuffer_timing_local.hh"
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_program.hh"
#include "rebuffer_timing_buffer_state.hh"
#include "timing_hard_math.h"

#include <algorithm>

namespace rebuffer {

void NativeTimingLocalPlanningSnapshot::buildViews(LocalPlanningView& out)
{
  const auto& m = *graph_->mutation_state_;
  const auto& base = inputs_->design;
  auto& d = out.projection.design;
  auto& t = out.timing;
  d.dbu = base.dbu; d.num_pins = pin_count_; d.num_nodes = node_count_; d.num_nets = net_count_;
  d.num_base_nodes = base.num_base_nodes;
  d.num_terminals = base.num_terminals;
  d.num_terminal_nis = base.num_terminal_nis;
  d.pin2node_map.assign(pin_count_, -1); d.pin2net_map.assign(pin_count_, -1);
  d.pin_offset_x.resize(pin_count_); d.pin_offset_y.resize(pin_count_); d.pin_names.resize(pin_count_);
  d.node_x.resize(node_count_); d.node_y.resize(node_count_);
  d.node_is_buffer.resize(node_count_); d.inst_main_id.assign(node_count_, -1); d.inst_libcell_offset.resize(node_count_);
  if (!base.node_is_fixed.empty()) d.node_is_fixed.resize(node_count_);
  d.pin_2_libpin_offset.assign(pin_count_, -1);
  d.pin_cap_limit = base.pin_cap_limit; d.pin_cap = base.pin_cap;
  d.pin_slew_limit = base.pin_slew_limit; d.pin_drive_resistance = base.pin_drive_resistance;
  d.main_id_2_cell_id_start = base.main_id_2_cell_id_start;
  d.cell_id_2_libpin_id_start = base.cell_id_2_libpin_id_start;
  d.net2driver_pin_map.resize(net_count_);
  for (int net = 0; net < net_count_; ++net) {
    const int driver = m.net_drivers.at(graphNet(net));
    d.net2driver_pin_map[net] = driver < 0 ? -1 : pinId(driver);
  }
  out.projection.net_names.resize(net_count_);
  d.flat_net2pin_start_map.push_back(0);
  for (int net = 0; net < net_count_; ++net) {
    if (out.nets.count(net)) {
      const int native = graphNet(net);
      out.projection.net_names[net] = m.net_names.at(native);
      for (const int pin : out.projection.topologies.at(net)) {
        if (pin < pin_count_) d.flat_net2pin_map.push_back(pin);
      }
    }
    d.flat_net2pin_start_map.push_back(d.flat_net2pin_map.size());
  }
  std::set<int> owners, read_pins;
  for (const int pin : out.pins) {
    const int native = pin < base.num_pins ? pin : canonical_to_pin_.at(pin);
    owners.insert(m.pin_owner.at(native));
  }
  for (const int owner : owners) {
    const int node = nodeId(owner), cell = m.node_cells.at(owner);
    const int main = cell < 0 ? -1 : m.cell_mains.at(cell);
    d.inst_main_id[node] = main;
    d.inst_libcell_offset[node] = main < 0 ? 0 : cell - base.main_id_2_cell_id_start.at(main);
    d.node_is_buffer[node] = owner < m.original_node_count
        ? base.node_is_buffer.at(owner) : graph_->queue_liberty_->cells.at(cell).is_buffer;
    if (!base.node_is_fixed.empty()) d.node_is_fixed[node] = owner < m.original_node_count ? base.node_is_fixed.at(owner) : 0;
    if (owner < m.original_node_count) {
      d.node_x[node] = inputs_->carrier.node_x.at(owner); d.node_y[node] = inputs_->carrier.node_y.at(owner);
    } else {
      const auto buffer = std::find_if(buffers_.begin(), buffers_.end(), [&](const auto& row) { return row.node == owner; });
      requireLocal(buffer != buffers_.end(), "local owner has no active native buffer");
      const auto& state = *graph_->net_states_.at(buffer->source_net);
      const auto physical = std::find_if(state.buffers.begin(), state.buffers.end(), [&](const auto& row) { return row.action_id == buffer->action_id; });
      const auto& point = state.coordinates.at(physical->input_node);
      d.node_x[node] = point[0]; d.node_y[node] = point[1];
    }
    for (const int pin : m.node_pins.at(owner)) if (m.pin_active.at(pin)) read_pins.insert(pin);
  }
  const auto init = [&](std::vector<double>& values) { values.resize(pin_count_); };
  for (auto* values : {&t.pin_cap, &t.pin_max_slew, &t.pin_max_cap, &t.pin_slew_rise, &t.pin_slew_fall,
       &t.pin_load, &t.pin_arrival_rise, &t.pin_arrival_fall, &t.pin_required_rise, &t.pin_required_fall, &t.pin_slack}) init(*values);
  const std::pair<const char*, std::vector<double>*> fields[] = {
    {"pin_rtran", &t.pin_slew_rise}, {"pin_ftran", &t.pin_slew_fall}, {"pin_net_cap_rise", &t.pin_load},
    {"pin_rAAT", &t.pin_arrival_rise}, {"pin_fAAT", &t.pin_arrival_fall},
    {"pin_rRAT", &t.pin_required_rise}, {"pin_fRAT", &t.pin_required_fall},
  };
  for (const int native : read_pins) {
    const int pin = pinId(native), owner = m.pin_owner.at(native), cell = m.node_cells.at(owner);
    d.pin2node_map[pin] = nodeId(owner);
    d.pin2net_map[pin] = m.pin_net.at(native) < 0 ? -1 : netId(m.pin_net[native]);
    d.pin_names[pin] = graph_->timing_index_->pins.at(native);
    if (native < base.num_pins) {
      d.pin_offset_x[pin] = inputs_->carrier.pin_x.at(native); d.pin_offset_y[pin] = inputs_->carrier.pin_y.at(native);
    }
    if (cell >= 0) {
      const auto begin = m.library_pin_names.begin() + m.cell_pin_start.at(cell);
      const auto end = m.library_pin_names.begin() + m.cell_pin_start.at(cell + 1);
      const auto found = std::find(begin, end, m.pin_ports.at(native));
      requireLocal(found != end, "local pin is absent from current Liberty cell");
      d.pin_2_libpin_offset[pin] = std::distance(begin, found);
    }
    t.pin_cap[pin] = m.pin_values[0][native]; t.pin_max_slew[pin] = m.pin_values[3][native]; t.pin_max_cap[pin] = m.pin_values[4][native];
    for (const auto& [field, values] : fields) (*values)[pin] = graph_->state_.at(field).at(native);
    t.pin_slack[pin] = dreamplace::timing::hardMinimum(t.pin_required_rise[pin] - t.pin_arrival_rise[pin], t.pin_required_fall[pin] - t.pin_arrival_fall[pin]);
  }
  const auto slacks = graph_->endpointSlacks();
  for (std::size_t row = 0; row < slacks.size(); ++row) {
    const int native = graph_->propagation_program_->metric_pins[row];
    if (read_pins.count(native)) t.pin_slack[pinId(native)] = slacks[row];
  }
  for (const int native : graph_->timing_index_->clocks) if (read_pins.count(native)) t.clock_pins.push_back(pinId(native));
  for (const auto& item : queue_->items) {
    if (!out.nets.count(item.root_net_id)) continue;
    for (const int pin : item.path_pins) {
      if (pin >= 0 && pin < pin_count_ && out.pins.count(pin)) {
        const int net = d.pin2net_map[pin];
        if (net >= 0 && d.net2driver_pin_map[net] != pin) t.pin_slack[pin] = item.endpoint_slack_ps;
      }
    }
  }
  filled_pins_ += read_pins.size();
}

}  // namespace rebuffer
