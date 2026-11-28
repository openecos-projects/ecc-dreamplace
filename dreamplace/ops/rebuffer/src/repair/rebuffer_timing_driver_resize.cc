// SPDX-License-Identifier: BSD-3-Clause
// Native inline driver gate sizing candidate selection (upsizeDrvr) for repair_timing.
#include "rebuffer_timing_graph.hh"

#include "rebuffer_timing_mutation.hh"

#include "rebuffer_timing_planning.hh"

#include "rebuffer_liberty.hh"


#include <algorithm>
#include <cmath>
#include <limits>
#include <string>
#include <utility>
#include <vector>

#include "pybind11/stl.h"


namespace py = pybind11;
namespace rebuffer {

std::pair<int, int> PersistentTimingGraph::findDriverResizeCandidate(
    int native_driver_pin,
    const std::vector<int>& native_path_pins,
    const py::dict& config) const
{
  auto configBool = [&](const char* key, bool fallback) {
    if (config.contains(key) && !config[key].is_none()) return config[key].cast<bool>();
    if (planning_inputs_ && planning_inputs_->config.contains(key)
        && !planning_inputs_->config[key].is_none()) {
      return planning_inputs_->config[key].cast<bool>();
    }
    return fallback;
  };
  if (!configBool("resize_drivers", false)) return {-1, -1};
  if (!mutation_state_ || !timing_state_initialized_ || !planning_inputs_
      || !queue_liberty_ || !propagation_program_ || !timing_index_) {
    return {-1, -1};
  }
  const auto& m = *mutation_state_;
  const auto& base = planning_inputs_->design;
  const auto& liberty = *queue_liberty_;
  if (native_driver_pin < 0 || native_driver_pin >= static_cast<int>(m.pin_owner.size())) {
    return {-1, -1};
  }
  const int owner = m.pin_owner[native_driver_pin];
  const int movable =
      m.original_node_count - planning_inputs_->carrier.num_terminals - planning_inputs_->carrier.num_terminal_nis;
  if (owner < 0 || owner >= movable) return {-1, -1};
  if (!base.node_is_fixed.empty() && base.node_is_fixed[owner] != 0) return {-1, -1};
  if (m.base_node_constraint_flags[owner] != 0) return {-1, -1};
  if (m.node_arcs[owner].empty()) return {-1, -1};
  for (int arc_id : m.node_arcs[owner]) {
    if (arc_id < timing_index_->clock_end) return {-1, -1};
  }
  for (int pin : m.node_pins[owner]) {
    if (m.pin_net[pin] < 0 || m.pin_net[pin] >= static_cast<int>(m.net_drivers.size())) {
      return {-1, -1};
    }
  }
  const int current_cell = m.node_cells[owner];
  if (current_cell < 0 || current_cell >= static_cast<int>(liberty.cells.size())) {
    return {-1, -1};
  }
  const int main_id = m.cell_mains[current_cell];
  if (main_id < 0 || main_id + 1 >= static_cast<int>(base.main_id_2_cell_id_start.size())) {
    return {-1, -1};
  }
  const int family_begin = base.main_id_2_cell_id_start[main_id];
  const int family_end = base.main_id_2_cell_id_start[main_id + 1];
  if (family_end - family_begin <= 1) return {-1, -1};

  const auto& s_rtran = state_.at("pin_rtran");
  const auto& s_ftran = state_.at("pin_ftran");
  const auto& s_load = state_.at("pin_net_cap_rise");
  const auto& s_raat = state_.at("pin_rAAT");
  const auto& s_faat = state_.at("pin_fAAT");
  const auto& s_rrat = state_.at("pin_rRAT");
  const auto& s_frat = state_.at("pin_fRAT");
  const double load_cap = s_load.at(native_driver_pin);
  if (!(load_cap > 0.0) || !std::isfinite(load_cap)) return {-1, -1};

  int crit_in_pin = -1;
  for (int pin : m.node_pins[owner]) {
    if (pin == native_driver_pin) continue;
    const int net = m.pin_net[pin];
    if (net < 0 || m.net_drivers[net] == pin) continue;
    if (std::find(native_path_pins.begin(), native_path_pins.end(), pin) != native_path_pins.end()) {
      crit_in_pin = pin;
      break;
    }
  }
  if (crit_in_pin < 0) {
    double worst_slack = std::numeric_limits<double>::infinity();
    for (int pin : m.node_pins[owner]) {
      if (pin == native_driver_pin) continue;
      const int net = m.pin_net[pin];
      if (net < 0 || m.net_drivers[net] == pin) continue;
      const double slack = std::min(s_rrat.at(pin) - s_raat.at(pin),
                                    s_frat.at(pin) - s_faat.at(pin));
      if (crit_in_pin < 0 || slack < worst_slack) {
        worst_slack = slack;
        crit_in_pin = pin;
      }
    }
  }
  if (crit_in_pin < 0) return {-1, -1};

  auto configDouble = [&](const char* key, double fallback) {
    if (config.contains(key) && !config[key].is_none()) return config[key].cast<double>();
    if (planning_inputs_ && planning_inputs_->config.contains(key)
        && !planning_inputs_->config[key].is_none()) {
      return planning_inputs_->config[key].cast<double>();
    }
    return fallback;
  };
  const double cap_margin = configDouble("cap_margin", 0.0);
  const double slew_margin = configDouble("slew_margin", 0.0);
  const double sizing_area_limit = configDouble("sizing_area_limit", 4.0);
  const double sizing_leakage_limit = configDouble("sizing_leakage_limit", 4.0);

  const std::string& in_port = m.pin_ports[crit_in_pin];
  const std::string& out_port = m.pin_ports[native_driver_pin];
  const double in_slew_rise = s_rtran.at(crit_in_pin);
  const double in_slew_fall = s_ftran.at(crit_in_pin);

  auto cellPinOffset = [&](int cell_id, const std::string& port) -> int {
    const auto begin = m.library_pin_names.begin() + m.cell_pin_start[cell_id];
    const auto end = m.library_pin_names.begin() + m.cell_pin_start[cell_id + 1];
    const auto found = std::find(begin, end, port);
    if (found == end || std::find(found + 1, end, port) != end) return -1;
    return static_cast<int>(found - begin);
  };
  auto cellDriveRes = [&](int cell_id, int port_offset) -> double {
    if (port_offset >= 0 && cell_id + 1 < static_cast<int>(base.cell_id_2_libpin_id_start.size())) {
      const int lib_pin = base.cell_id_2_libpin_id_start[cell_id] + port_offset;
      if (lib_pin >= 0 && lib_pin < static_cast<int>(base.pin_drive_resistance.size())
          && base.pin_drive_resistance[lib_pin] > 0.0) {
        return base.pin_drive_resistance[lib_pin];
      }
    }
    return liberty.cells[cell_id].drive_resistance;
  };
  auto upstreamDriveRes = [&](int in_pin) -> double {
    const int net = m.pin_net[in_pin];
    if (net < 0 || net >= static_cast<int>(m.net_drivers.size())) return 0.0;
    const int prev_drv = m.net_drivers[net];
    if (prev_drv < 0 || prev_drv >= static_cast<int>(m.pin_owner.size())) return 0.0;
    const int prev_node = m.pin_owner[prev_drv];
    const int prev_cell = (prev_node >= 0 && prev_node < static_cast<int>(m.node_cells.size()))
                              ? m.node_cells[prev_node]
                              : -1;
    if (prev_cell >= 0 && prev_cell < static_cast<int>(liberty.cells.size())) {
      const int off = cellPinOffset(prev_cell, m.pin_ports[prev_drv]);
      const double r = cellDriveRes(prev_cell, off);
      if (r > 0.0) return r;
    }
    return 0.0;
  };
  auto evalArcDelayAndSlew = [&](int cell_id, double ir, double ifall, double c_load,
                                 double& delay_out, double& slew_out) -> bool {
    const auto& cell = liberty.cells[cell_id];
    double max_delay = -1.0;
    double max_slew = 0.0;
    bool matched = false;
    for (const auto& arc : cell.arcs) {
      if (!arc.source_name.empty() && !arc.sink_name.empty()) {
        if (arc.source_name != in_port || arc.sink_name != out_port) continue;
      }
      const double dr = lutFindValue(arc.rise_delay, ir, c_load);
      const double df = lutFindValue(arc.fall_delay, ifall, c_load);
      const double sr = lutFindValue(arc.rise_transition, ir, c_load);
      const double sf = lutFindValue(arc.fall_transition, ifall, c_load);
      if (!std::isfinite(dr) || !std::isfinite(df) || !std::isfinite(sr) || !std::isfinite(sf)) {
        return false;
      }
      max_delay = std::max(max_delay, 0.5 * (dr + df));
      max_slew = std::max({max_slew, sr, sf});
      matched = true;
    }
    if (!matched) return false;
    delay_out = max_delay;
    slew_out = max_slew;
    return true;
  };

  const int curr_out_off = cellPinOffset(current_cell, out_port);
  const int curr_in_off = cellPinOffset(current_cell, in_port);
  if (curr_out_off < 0 || curr_in_off < 0) return {-1, -1};
  const double r_curr = cellDriveRes(current_cell, curr_out_off);
  double delay_curr = 0.0, slew_curr = 0.0;
  if (!evalArcDelayAndSlew(current_cell, in_slew_rise, in_slew_fall, load_cap, delay_curr, slew_curr)) {
    return {-1, -1};
  }
  const double in_cap_curr = m.library_pin_values[0][m.cell_pin_start[current_cell] + curr_in_off];
  const double r_prev = upstreamDriveRes(crit_in_pin);
  const auto& curr_view = liberty.cells[current_cell];

  int best_cell = -1;
  double best_r = -1.0;
  double best_area = std::numeric_limits<double>::infinity();
  double best_gain = 0.0;

  for (int cand = family_begin; cand < family_end; ++cand) {
    if (cand == current_cell || cand < 0 || cand >= static_cast<int>(liberty.cells.size())) continue;
    const auto& cand_view = liberty.cells[cand];
    if (cand_view.vt != curr_view.vt) continue;
    if (sizing_area_limit > 0.0 && curr_view.area > 0.0
        && cand_view.area / curr_view.area > sizing_area_limit) {
      continue;
    }
    if (liberty.cell_metadata && cand < static_cast<int>(liberty.cell_metadata->cells.size())
        && current_cell < static_cast<int>(liberty.cell_metadata->cells.size())) {
      const auto& curr_meta = liberty.cell_metadata->cells[current_cell];
      const auto& cand_meta = liberty.cell_metadata->cells[cand];
      if (cand_meta.dont_use || !cand_meta.has_master || !cand_meta.master_is_core) continue;
      if (!curr_meta.function_class.empty() && curr_meta.function_class != cand_meta.function_class) {
        continue;
      }
      if (sizing_leakage_limit > 0.0 && curr_meta.has_leakage && cand_meta.has_leakage
          && curr_meta.leakage > 0.0 && cand_meta.leakage / curr_meta.leakage > sizing_leakage_limit) {
        continue;
      }
    }
    const int cand_out_off = cellPinOffset(cand, out_port);
    const int cand_in_off = cellPinOffset(cand, in_port);
    if (cand_out_off < 0 || cand_in_off < 0) continue;
    const double r_cand = cellDriveRes(cand, cand_out_off);
    if (!(r_cand < r_curr - 1e-6)) continue;

    bool arcs_ok = true;
    for (int id : m.node_arcs[owner]) {
      const auto& row = m.cells[id];
      const int lut = equivalentTimingArc(
          m.cell_arc_start[cand], m.cell_arc_start[cand + 1], row[3], row[6],
          [&](int index) -> const TimingLibraryArc& { return m.library_arcs.at(index); });
      for (int table = 0; table < 4; ++table) {
        if (lut < 0 || lut >= propagation_program_->luts[table].rows) {
          arcs_ok = false;
          break;
        }
      }
      if (!arcs_ok) break;
    }
    if (!arcs_ok) continue;

    bool pins_ok = true;
    for (int pin : m.node_pins[owner]) {
      const int off = cellPinOffset(cand, m.pin_ports[pin]);
      if (off < 0) {
        pins_ok = false;
        break;
      }
      const int idx = m.cell_pin_start[cand] + off;
      for (int lane = 0; lane < 5; ++lane) {
        const double val = m.library_pin_values[lane][idx];
        if (!std::isfinite(val) || val < 0.0) {
          pins_ok = false;
          break;
        }
      }
      if (!pins_ok) break;
      if (pin == native_driver_pin) {
        const double max_cap = m.library_pin_values[4][idx];
        if (max_cap > 0.0 && load_cap > max_cap * (1.0 - cap_margin)) {
          pins_ok = false;
          break;
        }
      } else {
        const double old_cap = m.pin_values[0][pin];
        const double new_cap = m.library_pin_values[0][idx];
        const double delta_cap = new_cap - old_cap;
        const double pin_max_slew = m.library_pin_values[3][idx];
        const double pin_slew = std::max(s_rtran.at(pin), s_ftran.at(pin));
        if (pin_max_slew > 0.0 && pin_slew > pin_max_slew * (1.0 - slew_margin)) {
          pins_ok = false;
          break;
        }
        if (delta_cap > 0.0) {
          const int net = m.pin_net[pin];
          const int prev_drv = (net >= 0 && net < static_cast<int>(m.net_drivers.size()))
                                   ? m.net_drivers[net]
                                   : -1;
          if (prev_drv >= 0) {
            const double prev_load = s_load.at(prev_drv);
            const double prev_max_cap = m.pin_values[4][prev_drv];
            if (prev_max_cap > 0.0 && (prev_load + delta_cap) > prev_max_cap * (1.0 - cap_margin)) {
              pins_ok = false;
              break;
            }
            const double prev_max_slew = m.pin_values[3][prev_drv];
            const double prev_slew = std::max(s_rtran.at(prev_drv), s_ftran.at(prev_drv));
            const double rp = upstreamDriveRes(pin);
            if (prev_max_slew > 0.0
                && (prev_slew + rp * delta_cap * 1.5) > prev_max_slew * (1.0 - slew_margin)) {
              pins_ok = false;
              break;
            }
          }
        }
      }
    }
    if (!pins_ok) continue;

    const double in_cap_cand = m.library_pin_values[0][m.cell_pin_start[cand] + cand_in_off];
    const double delta_in_cap = std::max(0.0, in_cap_cand - in_cap_curr);
    const double upstream_penalty = r_prev * delta_in_cap;
    double delay_cand = 0.0, slew_cand = 0.0;
    if (!evalArcDelayAndSlew(cand, in_slew_rise + upstream_penalty, in_slew_fall + upstream_penalty,
                             load_cap, delay_cand, slew_cand)) {
      continue;
    }
    const double out_max_slew = m.library_pin_values[3][m.cell_pin_start[cand] + cand_out_off];
    if (out_max_slew > 0.0 && slew_cand > out_max_slew * (1.0 - slew_margin)) {
      continue;
    }
    const double net_gain = (delay_curr - delay_cand) - upstream_penalty;
    if (!(net_gain > 1e-4)) continue;

    if (best_cell < 0 || r_cand > best_r + 1e-9
        || (std::abs(r_cand - best_r) <= 1e-9
            && (cand_view.area < best_area - 1e-9
                || (std::abs(cand_view.area - best_area) <= 1e-9 && net_gain > best_gain)))) {
      best_cell = cand;
      best_r = r_cand;
      best_area = cand_view.area;
      best_gain = net_gain;
    }
  }
  if (best_cell < 0) return {-1, -1};
  return {owner, best_cell};
}

}  // namespace rebuffer
