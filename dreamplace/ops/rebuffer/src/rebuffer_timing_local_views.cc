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
  d.pin2node_map.resize(pin_count_, -1); d.pin2net_map.resize(pin_count_, -1);
  d.pin_offset_x.resize(pin_count_, 0.0); d.pin_offset_y.resize(pin_count_, 0.0);
  d.node_x.resize(node_count_, 0.0); d.node_y.resize(node_count_, 0.0);
  d.node_is_buffer.resize(node_count_, 0); d.inst_main_id.resize(node_count_, -1); d.inst_libcell_offset.resize(node_count_, 0);
  if (!base.node_is_fixed.empty()) d.node_is_fixed.resize(node_count_, 0);
  d.pin_2_libpin_offset.resize(pin_count_, -1);
  if (!out.static_design_initialized) {
    d.pin_cap_limit = base.pin_cap_limit; d.pin_cap = base.pin_cap;
    d.pin_slew_limit = base.pin_slew_limit; d.pin_drive_resistance = base.pin_drive_resistance;
    d.main_id_2_cell_id_start = base.main_id_2_cell_id_start;
    d.cell_id_2_libpin_id_start = base.cell_id_2_libpin_id_start;
    d.net2driver_pin_map = base.net2driver_pin_map;
  }
  if (!out.static_design_initialized
      || out.cached_topology_revision != identity_->topology_revision
      || static_cast<int>(d.net2driver_pin_map.size()) != net_count_) {
    d.net2driver_pin_map.resize(net_count_);
    for (int net = base.num_nets; net < net_count_; ++net) {
      const int driver = m.net_drivers.at(graphNet(net));
      d.net2driver_pin_map[net] = driver < 0 ? -1 : pinId(driver);
    }
  }
  out.cached_topology_revision = identity_->topology_revision;
  out.static_design_initialized = true;
  if (!shared_view_data_ready_) {
    endpoint_slacks_ = graph_->endpointSlacks();
    shared_view_data_ready_ = true;
  }
  if (!inputs_->static_pin_lookup_ready) {
    const int prog_pins = graph_->propagation_program_->pin_count;
    inputs_->metric_row_by_pin.assign(prog_pins, -1);
    for (std::size_t row = 0; row < graph_->propagation_program_->metric_pins.size(); ++row) {
      const int pin = static_cast<int>(graph_->propagation_program_->metric_pins[row]);
      if (pin >= 0 && pin < prog_pins) {
        inputs_->metric_row_by_pin[pin] = static_cast<int>(row);
      }
    }
    inputs_->clock_order_by_pin.assign(prog_pins, -1);
    inputs_->clock_pins_unique = true;
    for (std::size_t idx = 0; idx < graph_->timing_index_->clocks.size(); ++idx) {
      const int pin = graph_->timing_index_->clocks[idx];
      if (pin >= 0 && pin < prog_pins) {
        if (inputs_->clock_order_by_pin[pin] != -1) inputs_->clock_pins_unique = false;
        inputs_->clock_order_by_pin[pin] = static_cast<int>(idx);
      } else {
        inputs_->clock_pins_unique = false;
      }
    }
    inputs_->static_pin_lookup_ready = true;
  }
  d.flat_net2pin_start_map.resize(net_count_ + 1, 0);
  for (const int net : out.nets) {
    d.flat_net2pin_start_map[net] = static_cast<int>(d.flat_net2pin_map.size());
    for (const int pin : out.projection.topologies.at(net)) {
      if (pin < pin_count_) d.flat_net2pin_map.push_back(pin);
    }
    d.flat_net2pin_start_map[net + 1] = static_cast<int>(d.flat_net2pin_map.size());
  }
  std::vector<int> owners, read_pins;
  owners.reserve(out.pins.size());
  for (const int pin : out.pins) {
    const int native = pin < base.num_pins ? pin : identity_->canonical_to_pin.at(pin - base.num_pins);
    owners.push_back(m.pin_owner.at(native));
  }
  std::sort(owners.begin(), owners.end());
  owners.erase(std::unique(owners.begin(), owners.end()), owners.end());
  read_pins.reserve(owners.size() * 4);
  for (const int owner : owners) {
    const int node = nodeId(owner), cell = m.node_cells.at(owner);
    out.touched_nodes.push_back(node);
    const int main = cell < 0 ? -1 : m.cell_mains.at(cell);
    d.inst_main_id[node] = main;
    d.inst_libcell_offset[node] = main < 0 ? 0 : cell - base.main_id_2_cell_id_start.at(main);
    d.node_is_buffer[node] = owner < m.original_node_count
        ? base.node_is_buffer.at(owner) : graph_->queue_liberty_->cells.at(cell).is_buffer;
    if (!base.node_is_fixed.empty()) d.node_is_fixed[node] = owner < m.original_node_count ? base.node_is_fixed.at(owner) : 0;
    if (owner < m.original_node_count) {
      d.node_x[node] = inputs_->carrier.node_x.at(owner); d.node_y[node] = inputs_->carrier.node_y.at(owner);
    } else {
      const int buf_idx = identity_->buffer_index_by_node.at(owner - m.original_node_count);
      requireLocal(buf_idx >= 0, "local owner has no active native buffer");
      const auto& point = identity_->buffer_coordinates.at(buf_idx);
      d.node_x[node] = point[0]; d.node_y[node] = point[1];
    }
    for (const int pin : m.node_pins.at(owner)) if (m.pin_active.at(pin)) read_pins.push_back(pin);
  }
  std::sort(read_pins.begin(), read_pins.end());
  read_pins.erase(std::unique(read_pins.begin(), read_pins.end()), read_pins.end());
  const auto init = [&](std::vector<double>& values) { values.resize(pin_count_, 0.0); };
  for (auto* values : {&t.pin_cap, &t.pin_max_slew, &t.pin_max_cap, &t.pin_slew_rise, &t.pin_slew_fall,
       &t.pin_load, &t.pin_arrival_rise, &t.pin_arrival_fall, &t.pin_required_rise, &t.pin_required_fall, &t.pin_slack}) init(*values);
  const auto& s_rtran = graph_->state_.at("pin_rtran");
  const auto& s_ftran = graph_->state_.at("pin_ftran");
  const auto& s_load = graph_->state_.at("pin_net_cap_rise");
  const auto& s_raat = graph_->state_.at("pin_rAAT");
  const auto& s_faat = graph_->state_.at("pin_fAAT");
  const auto& s_rrat = graph_->state_.at("pin_rRAT");
  const auto& s_frat = graph_->state_.at("pin_fRAT");
  std::vector<std::pair<int, int>> clock_hits;
  for (const int native : read_pins) {
    const int pin = pinId(native), owner = m.pin_owner.at(native), cell = m.node_cells.at(owner);
    out.touched_pins.push_back(pin);
    d.pin2node_map[pin] = nodeId(owner);
    d.pin2net_map[pin] = m.pin_net.at(native) < 0 ? -1 : netId(m.pin_net[native]);
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
    t.pin_slew_rise[pin] = s_rtran.at(native);
    t.pin_slew_fall[pin] = s_ftran.at(native);
    t.pin_load[pin] = s_load.at(native);
    t.pin_arrival_rise[pin] = s_raat.at(native);
    t.pin_arrival_fall[pin] = s_faat.at(native);
    t.pin_required_rise[pin] = s_rrat.at(native);
    t.pin_required_fall[pin] = s_frat.at(native);
    t.pin_slack[pin] = dreamplace::timing::hardMinimum(t.pin_required_rise[pin] - t.pin_arrival_rise[pin], t.pin_required_fall[pin] - t.pin_arrival_fall[pin]);
    if (native >= 0 && native < static_cast<int>(inputs_->metric_row_by_pin.size())) {
      const int row = inputs_->metric_row_by_pin[native];
      if (row >= 0 && static_cast<std::size_t>(row) < endpoint_slacks_.size()) {
        t.pin_slack[pin] = endpoint_slacks_[row];
      }
    }
    if (inputs_->clock_pins_unique
        && native >= 0 && native < static_cast<int>(inputs_->clock_order_by_pin.size())) {
      const int order = inputs_->clock_order_by_pin[native];
      if (order >= 0) clock_hits.emplace_back(order, pin);
    }
  }
  if (! inputs_->clock_pins_unique) {
    for (const int native : graph_->timing_index_->clocks) {
      if (std::binary_search(read_pins.begin(), read_pins.end(), native)) {
        t.clock_pins.push_back(pinId(native));
      }
    }
  } else if (!clock_hits.empty()) {
    std::sort(clock_hits.begin(), clock_hits.end());
    for (const auto& [_, pin] : clock_hits) t.clock_pins.push_back(pin);
  }
  for (const auto& item : queue_->items) {
    if (!out.nets.count(item.root_net_id)) continue;
    for (const int pin : item.path_pins) {
      if (pin >= 0 && pin < pin_count_) {
        const int net = d.pin2net_map[pin];
        if (net >= 0 && d.net2driver_pin_map[net] != pin && out.pins.count(pin)) {
          t.pin_slack[pin] = item.endpoint_slack_ps;
        }
      }
    }
  }
  filled_pins_ += read_pins.size();
}

}  // namespace rebuffer
