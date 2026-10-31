// SPDX-License-Identifier: BSD-3-Clause

#include "rebuffer_repair.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace rebuffer {

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

int libPinIndex(const DesignView& design, int pin)
{
  if (pin < 0 || pin >= design.num_pins
      || pin >= static_cast<int>(design.pin2node_map.size())) {
    return -1;
  }
  const int node = design.pin2node_map[pin];
  if (node < 0 || node >= static_cast<int>(design.inst_main_id.size())) {
    return -1;
  }
  const int main_id = design.inst_main_id[node];
  if (main_id < 0
      || main_id + 1 >= static_cast<int>(design.main_id_2_cell_id_start.size())) {
    return -1;
  }
  const int cell_id
      = design.main_id_2_cell_id_start[main_id]
        + design.inst_libcell_offset[node];
  if (cell_id < 0
      || cell_id + 1
             >= static_cast<int>(design.cell_id_2_libpin_id_start.size())) {
    return -1;
  }
  const int lib_pin
      = design.cell_id_2_libpin_id_start[cell_id]
        + design.pin_2_libpin_offset[pin];
  return lib_pin;
}

int cellIndexForNode(const DesignView& design, int node)
{
  if (node < 0 || node >= static_cast<int>(design.inst_main_id.size())) {
    return -1;
  }
  const int main_id = design.inst_main_id[node];
  if (main_id < 0
      || main_id + 1 >= static_cast<int>(design.main_id_2_cell_id_start.size())) {
    return -1;
  }
  return design.main_id_2_cell_id_start[main_id]
         + design.inst_libcell_offset[node];
}

double pinMaxSlew(const DesignView& design, int pin)
{
  const int lib_pin = libPinIndex(design, pin);
  if (lib_pin < 0
      || lib_pin >= static_cast<int>(design.pin_slew_limit.size())) {
    return 0.0;
  }
  return design.pin_slew_limit[lib_pin];
}

double pinMaxCap(const DesignView& design, int pin)
{
  const int lib_pin = libPinIndex(design, pin);
  if (lib_pin < 0
      || lib_pin >= static_cast<int>(design.pin_cap_limit.size())) {
    return 0.0;
  }
  return design.pin_cap_limit[lib_pin];
}

double edgeCapacitance(const SteinerView& steiner,
                       const RepairConfig& config,
                       int node)
{
  if (config.r_unit <= 0.0 || node < 0
      || node >= static_cast<int>(steiner.edge_resistance.size())) {
    return 0.0;
  }
  return steiner.edge_resistance[node] / config.r_unit * config.c_unit;
}

double cellOutputSlew(const LibertyCellView& cell,
                      double input_slew_rise,
                      double input_slew_fall,
                      double load_cap)
{
  double worst = 0.0;
  for (const auto& arc : cell.arcs) {
    worst = std::max(
        worst,
        lutFindValue(arc.rise_transition, input_slew_rise, load_cap));
    worst = std::max(
        worst,
        lutFindValue(arc.fall_transition, input_slew_fall, load_cap));
  }
  return worst;
}

double cellOutputSlewRise(const LibertyCellView& cell,
                          double input_slew,
                          double load_cap)
{
  double worst = 0.0;
  for (const auto& arc : cell.arcs) {
    worst = std::max(
        worst, lutFindValue(arc.rise_transition, input_slew, load_cap));
  }
  return worst;
}

double cellOutputSlewFall(const LibertyCellView& cell,
                          double input_slew,
                          double load_cap)
{
  double worst = 0.0;
  for (const auto& arc : cell.arcs) {
    worst = std::max(
        worst, lutFindValue(arc.fall_transition, input_slew, load_cap));
  }
  return worst;
}

double slewLoadCap(const LibertyCellView& cell,
                    double target_slew,
                    double input_slew_rise,
                    double input_slew_fall)
{
  if (target_slew <= 0.0 || cell.max_cap <= 0.0) {
    return kInf;
  }
  if (cellOutputSlew(cell, input_slew_rise, input_slew_fall, 0.0)
      >= target_slew) {
    return 0.0;
  }
  double low = 0.0;
  double high = cell.max_cap * 2.0;
  for (int iteration = 0; iteration < 48; ++iteration) {
    const double middle = 0.5 * (low + high);
    if (cellOutputSlew(cell, input_slew_rise, input_slew_fall, middle)
        < target_slew) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return 0.5 * (low + high);
}

std::string portName(const std::string& pin)
{
  const auto separator = pin.find_last_of(":/");
  return separator == std::string::npos ? pin : pin.substr(separator + 1);
}

bool driverInputSlew(const DesignView& design,
                     const TimingView& timing,
                     const LibertyView& liberty,
                     const std::vector<std::vector<int>>& node_pins,
                     int driver,
                     double& input_slew_rise,
                     double& input_slew_fall)
{
  if (driver < 0 || driver >= design.num_pins
      || driver >= static_cast<int>(design.pin2node_map.size())) {
    return false;
  }
  const int node = design.pin2node_map[driver];
  if (node < 0 || node >= static_cast<int>(node_pins.size())) {
    return false;
  }
  const int cell_index = cellIndexForNode(design, node);
  if (cell_index < 0
      || cell_index >= static_cast<int>(liberty.cells.size())) {
    return false;
  }
  const auto& cell = liberty.cells[cell_index];
  bool found = false;
  double rise = 0.0;
  double fall = 0.0;
  for (const int pin : node_pins[node]) {
    if (pin == driver || pin < 0 || pin >= design.num_pins) {
      continue;
    }
    const int offset = design.pin_2_libpin_offset[pin];
    const std::string pin_port
        = pin < static_cast<int>(design.pin_names.size())
              ? portName(design.pin_names[pin])
              : std::string();
    for (const auto& arc : cell.arcs) {
      const bool matches
          = !arc.source_name.empty() && !pin_port.empty()
                ? arc.source_name == pin_port
                : arc.source == offset;
      if (matches) {
        rise = std::max(rise, timing.pin_slew_rise[pin]);
        fall = std::max(fall, timing.pin_slew_fall[pin]);
        found = true;
        break;
      }
    }
  }
  if (found) {
    input_slew_rise = rise;
    input_slew_fall = fall;
  }
  return found;
}

double cellOutputDelay(const LibertyCellView& cell,
                       double input_slew_rise,
                       double input_slew_fall,
                       double load_cap)
{
  double worst = 0.0;
  for (const auto& arc : cell.arcs) {
    worst = std::max(
        worst,
        lutFindValue(arc.rise_delay, input_slew_rise, load_cap));
    worst = std::max(
        worst,
        lutFindValue(arc.fall_delay, input_slew_fall, load_cap));
  }
  return worst;
}

double targetLoadForCell(const LibertyCellView& cell,
                         double target_slew_rise,
                         double target_slew_fall)
{
  const double target_slew = std::max(target_slew_rise, target_slew_fall);
  if (cell.max_cap <= 0.0 || target_slew <= 0.0) {
    return 0.0;
  }
  if (cellOutputSlew(cell, target_slew_rise, target_slew_fall, 0.0)
      >= target_slew) {
    return 0.0;
  }
  double low = 0.0;
  double high = cell.max_cap * 2.0;
  for (int iteration = 0; iteration < 48; ++iteration) {
    const double middle = 0.5 * (low + high);
    if (cellOutputSlew(cell, target_slew_rise, target_slew_fall, middle)
        < target_slew) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return 0.5 * (low + high);
}

SubtreeLoad subtreeLoad(const DesignView& design,
                        const TimingView& timing,
                        const SteinerView& steiner,
                        const RepairConfig& config,
                        const std::vector<std::vector<int>>& children,
                        int node,
                        int driver)
{
  SubtreeLoad result;
  if (node >= 0 && node < static_cast<int>(steiner.node_capacitance.size())) {
    result.load += steiner.node_capacitance[node];
  }
  if (node >= 0 && node < static_cast<int>(steiner.pin_fa.size())
      && steiner.pin_fa[node] >= 0) {
    result.load += 0.5 * edgeCapacitance(steiner, config, node);
  }
  if (node >= 0 && node < steiner.steiner_node_base && node != driver
      && node < static_cast<int>(timing.pin_cap.size())) {
    result.load += timing.pin_cap[node];
    result.sinks.push_back(node);
  }
  if (node >= 0 && node < static_cast<int>(children.size())) {
    for (const int child : children[node]) {
      const auto child_load = subtreeLoad(
          design, timing, steiner, config, children, child, driver);
      result.load += child_load.load;
      result.sinks.insert(
          result.sinks.end(), child_load.sinks.begin(), child_load.sinks.end());
    }
  }
  return result;
}

int chooseBufferCell(const LibertyView& liberty,
                     const CharacterizationResult& characterization,
                     double input_slew_rise,
                     double input_slew_fall,
                     double downstream_load)
{
  const auto& candidates = characterization.filtered_buffers.empty()
                               ? characterization.fast_buffers
                               : characterization.filtered_buffers;
  if (candidates.empty()) {
    return -1;
  }

  int lowest_main_id = -1;
  int lowest_vt = -1;
  if (characterization.lowest_drive_index >= 0
      && characterization.lowest_drive_index
             < static_cast<int>(characterization.filtered_buffers.size())) {
    const int cell_index = characterization.filtered_buffers[
        characterization.lowest_drive_index].cell_index;
    if (cell_index >= 0 && cell_index < static_cast<int>(liberty.cells.size())) {
      lowest_main_id = liberty.cells[cell_index].main_id;
      lowest_vt = liberty.cells[cell_index].vt;
    }
  }

  auto select = [&](auto&& candidate_cells) {
    int best = -1;
    double best_dist = kInf;
    double best_delay = kInf;
    int fallback = -1;
    double fallback_cap = -1.0;
    for (const auto* cell : candidate_cells) {
      if (cell == nullptr || !cell->is_buffer) {
        continue;
      }
      if (cell->max_cap > 0.0 && downstream_load > cell->max_cap) {
        continue;
      }
      const double output_slew = cellOutputSlew(
          *cell, input_slew_rise, input_slew_fall, downstream_load);
      if (cell->max_slew > 0.0 && output_slew > cell->max_slew) {
        continue;
      }
      if (cell->max_cap > fallback_cap) {
        fallback_cap = cell->max_cap;
        fallback = static_cast<int>(cell - liberty.cells.data());
      }
      const double target_load = targetLoadForCell(
          *cell, characterization.target_slew[0], characterization.target_slew[1]);
      const double distance = std::abs(downstream_load - target_load);
      const double delay = cellOutputDelay(
          *cell, input_slew_rise, input_slew_fall, downstream_load);
      if (best < 0
          || (delay < best_delay && distance < best_dist * 1.1)
          || (distance < best_dist && delay < best_delay * 1.1)) {
        best = static_cast<int>(cell - liberty.cells.data());
        best_dist = distance;
        best_delay = delay;
      }
    }
    return best >= 0 ? best : fallback;
  };

  if (lowest_main_id >= 0) {
    std::vector<const LibertyCellView*> family;
    family.reserve(liberty.cells.size());
    for (const auto& cell : liberty.cells) {
      if (cell.main_id == lowest_main_id
          && (lowest_vt < 0 || cell.vt == lowest_vt)) {
        family.push_back(&cell);
      }
    }
    const int selected = select(family);
    if (selected >= 0) {
      return selected;
    }
  }

  std::vector<const LibertyCellView*> filtered;
  filtered.reserve(candidates.size());
  for (const auto& buffer : candidates) {
    if (buffer.cell_index >= 0
        && buffer.cell_index < static_cast<int>(liberty.cells.size())) {
      filtered.push_back(&liberty.cells[buffer.cell_index]);
    }
  }
  return select(filtered);
}

bool isClockPin(const TimingView& timing, int pin)
{
  return std::find(timing.clock_pins.begin(), timing.clock_pins.end(), pin)
         != timing.clock_pins.end();
}

bool validLengths(const DesignView& design,
                  const TimingView& timing,
                  const SteinerView& steiner,
                  std::string& reason)
{
  if (design.num_nets <= 0 || design.num_pins <= 0 || design.num_nodes <= 0) {
    reason = "empty_design";
    return false;
  }
  if (static_cast<int>(timing.pin_slew_rise.size()) != design.num_pins
      || static_cast<int>(timing.pin_slew_fall.size()) != design.num_pins
      || static_cast<int>(timing.pin_load.size()) != design.num_pins
      || static_cast<int>(timing.pin_cap.size()) != design.num_pins
      || static_cast<int>(timing.pin_max_slew.size()) != design.num_pins
      || static_cast<int>(timing.pin_max_cap.size()) != design.num_pins) {
    reason = "timing_array_length_mismatch";
    return false;
  }
  if (static_cast<int>(steiner.net_flat_topo_sort_start.size())
          != design.num_nets + 1
      || static_cast<int>(steiner.pin_fa.size())
             < steiner.steiner_node_base
      || static_cast<int>(steiner.flat_pin_to_start.size())
             != static_cast<int>(steiner.pin_fa.size()) + 1) {
    reason = "steiner_array_length_mismatch";
    return false;
  }
  const std::size_t vertex_count = steiner.pin_fa.size();
  if (steiner.node_x.size() != vertex_count
      || steiner.node_y.size() != vertex_count
      || steiner.edge_resistance.size() != vertex_count
      || steiner.node_capacitance.size() != vertex_count) {
    reason = "steiner_vertex_length_mismatch";
    return false;
  }
  return true;
}

}  // namespace

int rebufferLibPinIndex(const DesignView& design, int pin)
{
  return libPinIndex(design, pin);
}

int rebufferCellIndexForNode(const DesignView& design, int node)
{
  return cellIndexForNode(design, node);
}

bool rebufferIsClockPin(const TimingView& timing, int pin)
{
  return isClockPin(timing, pin);
}

bool rebufferValidateLengths(const DesignView& design,
                             const TimingView& timing,
                             const SteinerView& steiner,
                             std::string& reason)
{
  return validLengths(design, timing, steiner, reason);
}

bool rebufferBuildNetChildren(const SteinerView& steiner,
                              int net_id,
                              std::vector<std::vector<int>>& children,
                              std::string& reason)
{
  if (net_id < 0
      || net_id + 1
             >= static_cast<int>(steiner.net_flat_topo_sort_start.size())) {
    reason = "invalid_net_id";
    return false;
  }
  const int topo_begin = steiner.net_flat_topo_sort_start[net_id];
  const int topo_end = steiner.net_flat_topo_sort_start[net_id + 1];
  int max_node = -1;
  for (int pos = topo_begin; pos < topo_end; ++pos) {
    max_node = std::max(max_node, steiner.net_flat_topo_sort[pos]);
  }
  if (max_node < 0) {
    reason = "empty_net_topology";
    return false;
  }
  children.assign(max_node + 1, {});
  for (int pos = topo_begin; pos < topo_end; ++pos) {
    const int node = steiner.net_flat_topo_sort[pos];
    if (node < 0 || node >= static_cast<int>(steiner.pin_fa.size())) {
      continue;
    }
    const int parent = steiner.pin_fa[node];
    if (parent >= 0 && parent <= max_node) {
      children[parent].push_back(node);
    }
  }
  return true;
}

SubtreeLoad rebufferSubtreeLoad(const DesignView& design,
                                const TimingView& timing,
                                const SteinerView& steiner,
                                const RepairConfig& config,
                                const std::vector<std::vector<int>>& children,
                                int node,
                                int driver)
{
  return subtreeLoad(design, timing, steiner, config, children, node, driver);
}

double rebufferEdgeCapacitance(const SteinerView& steiner,
                               const RepairConfig& config,
                               int node)
{
  return edgeCapacitance(steiner, config, node);
}

double rebufferCellOutputSlew(const LibertyCellView& cell,
                              double input_slew_rise,
                              double input_slew_fall,
                              double load_cap)
{
  return cellOutputSlew(cell, input_slew_rise, input_slew_fall, load_cap);
}

double rebufferCellOutputSlewRise(const LibertyCellView& cell,
                                  double input_slew_rise,
                                  double load_cap)
{
  return cellOutputSlewRise(cell, input_slew_rise, load_cap);
}

double rebufferCellOutputSlewFall(const LibertyCellView& cell,
                                  double input_slew_fall,
                                  double load_cap)
{
  return cellOutputSlewFall(cell, input_slew_fall, load_cap);
}

double rebufferCellOutputDelay(const LibertyCellView& cell,
                               double input_slew_rise,
                               double input_slew_fall,
                               double load_cap)
{
  return cellOutputDelay(cell, input_slew_rise, input_slew_fall, load_cap);
}

bool rebufferDriverInputSlew(const DesignView& design,
                             const TimingView& timing,
                             const LibertyView& liberty,
                             int driver,
                             double& input_slew_rise,
                             double& input_slew_fall)
{
  if (driver < 0 || driver >= design.num_pins
      || driver >= static_cast<int>(design.pin2node_map.size())) {
    return false;
  }
  const int node = design.pin2node_map[driver];
  std::vector<std::vector<int>> node_pins;
  if (node >= 0 && node < design.num_nodes) {
    node_pins.resize(design.num_nodes);
    for (int pin = 0; pin < design.num_pins; ++pin) {
      const int pin_node = design.pin2node_map[pin];
      if (pin_node == node) {
        node_pins[node].push_back(pin);
      }
    }
  }
  return driverInputSlew(design,
                         timing,
                         liberty,
                         node_pins,
                         driver,
                         input_slew_rise,
                         input_slew_fall);
}

double rebufferTargetLoadForCell(const LibertyCellView& cell,
                                 double target_slew_rise,
                                 double target_slew_fall)
{
  return targetLoadForCell(cell, target_slew_rise, target_slew_fall);
}

int rebufferChooseBufferCell(const LibertyView& liberty,
                             const CharacterizationResult& characterization,
                             double input_slew_rise,
                             double input_slew_fall,
                             double downstream_load)
{
  return chooseBufferCell(
      liberty, characterization, input_slew_rise, input_slew_fall, downstream_load);
}

bool planRepairDesign(const DesignView& design,
                      const TimingView& timing,
                      const SteinerView& steiner,
                      const LibertyView& liberty,
                      const CharacterizationResult& characterization,
                      const RepairConfig& config,
                      RepairDesignResult& result,
                      std::string& reason)
{
  result = RepairDesignResult{};
  if (!validLengths(design, timing, steiner, reason)) {
    return false;
  }
  if (design.inst_main_id.size() != static_cast<std::size_t>(design.num_nodes)
      || design.inst_libcell_offset.size()
             != static_cast<std::size_t>(design.num_nodes)
      || design.pin_2_libpin_offset.size()
             != static_cast<std::size_t>(design.num_pins)) {
    reason = "missing_design_electrical_metadata";
    return false;
  }

  std::vector<std::vector<int>> node_pins(design.num_nodes);
  for (int pin = 0; pin < design.num_pins; ++pin) {
    const int node = design.pin2node_map[pin];
    if (node >= 0 && node < design.num_nodes) {
      node_pins[node].push_back(pin);
    }
  }

  std::unordered_map<int, int> net_index_by_id;
  if (steiner.net_ids.empty()) {
    for (int net_id = 0; net_id < design.num_nets; ++net_id) {
      net_index_by_id.emplace(net_id, net_id);
    }
  } else {
    for (int index = 0; index < static_cast<int>(steiner.net_ids.size());
         ++index) {
      net_index_by_id.emplace(steiner.net_ids[index], index);
    }
  }

  int action_count = 0;
  for (int net_id = 0; net_id < design.num_nets; ++net_id) {
    if (action_count >= config.max_actions) {
      break;
    }
    const int driver = design.net2driver_pin_map[net_id];
    if (driver < 0 || driver >= design.num_pins) {
      continue;
    }
    if (isClockPin(timing, driver)) {
      continue;
    }
    const int pin_begin = design.flat_net2pin_start_map[net_id];
    const int pin_end = design.flat_net2pin_start_map[net_id + 1];
    if (pin_end - pin_begin < 2) {
      continue;
    }

    const double driver_slew = std::max(
        timing.pin_slew_rise[driver], timing.pin_slew_fall[driver]);
    const double driver_load = timing.pin_load[driver];
    const double driver_max_slew = timing.pin_max_slew[driver];
    const double driver_max_cap = timing.pin_max_cap[driver];
    double input_slew_rise = timing.pin_slew_rise[driver];
    double input_slew_fall = timing.pin_slew_fall[driver];
    driverInputSlew(design,
                    timing,
                    liberty,
                    node_pins,
                    driver,
                    input_slew_rise,
                    input_slew_fall);

    bool driver_slew_violation
        = driver_max_slew > 0.0
          && driver_slew > driver_max_slew * (1.0 - config.slew_margin);
    bool driver_cap_violation
        = driver_max_cap > 0.0
          && driver_load > driver_max_cap * (1.0 - config.cap_margin);

    bool load_slew_violation = false;
    bool load_cap_violation = false;
    for (int index = pin_begin; index < pin_end; ++index) {
      const int pin = design.flat_net2pin_map[index];
      if (pin == driver || pin < 0 || pin >= design.num_pins) {
        continue;
      }
      const double max_slew = timing.pin_max_slew[pin];
      if (max_slew > 0.0
          && std::max(timing.pin_slew_rise[pin], timing.pin_slew_fall[pin])
                 > max_slew * (1.0 - config.slew_margin)) {
        load_slew_violation = true;
      }
      const double max_cap = timing.pin_max_cap[pin];
      if (max_cap > 0.0
          && driver_load > max_cap * (1.0 - config.cap_margin)) {
        load_cap_violation = true;
      }
    }

    if (driver_slew_violation) {
      ++result.driver_slew_violations;
    }
    if (driver_cap_violation) {
      ++result.driver_cap_violations;
    }
    if (load_slew_violation) {
      ++result.load_slew_violations;
    }
    if (load_cap_violation) {
      ++result.load_cap_violations;
    }
    if (!driver_slew_violation && !driver_cap_violation
        && !load_slew_violation && !load_cap_violation) {
      continue;
    }

    bool repaired = false;
    const bool resize_slew_violation
        = driver_max_slew > 0.0
          && driver_slew
                 > driver_max_slew * (1.0 + config.resize_slew_threshold);
    if (resize_slew_violation && config.resize_drivers) {
      const int current_cell_index = cellIndexForNode(
          design, design.pin2node_map[driver]);
      if (current_cell_index >= 0
          && current_cell_index < static_cast<int>(liberty.cells.size())) {
        const auto& current_cell = liberty.cells[current_cell_index];
        int best_cell = -1;
        double best_area = kInf;
        int fallback_cell = -1;
        double fallback_area = -1.0;
        const int main_id = current_cell.main_id;
        if (main_id >= 0
            && main_id + 1
                   < static_cast<int>(design.main_id_2_cell_id_start.size())) {
          const int begin = design.main_id_2_cell_id_start[main_id];
          const int end = design.main_id_2_cell_id_start[main_id + 1];
          for (int cell_id = begin; cell_id < end; ++cell_id) {
            if (cell_id < 0
                || cell_id >= static_cast<int>(liberty.cells.size())) {
              continue;
            }
            const auto& cell = liberty.cells[cell_id];
            if (cell.vt != current_cell.vt) {
              continue;
            }
            if (cell.size <= current_cell.size) {
              continue;
            }
            if (cell.max_cap > 0.0
                && driver_load > cell.max_cap * (1.0 - config.cap_margin)) {
              continue;
            }
            const double output_slew = cellOutputSlew(
                cell, input_slew_rise, input_slew_fall, driver_load);
            if (cell.max_slew > 0.0
                && output_slew
                       > cell.max_slew * (1.0 - config.slew_margin)) {
              continue;
            }
            if (cell.area < best_area) {
              best_area = cell.area;
              best_cell = cell_id;
            }
            if (cell.area > fallback_area) {
              fallback_area = cell.area;
              fallback_cell = cell_id;
            }
          }
        }
        const int selected = best_cell >= 0 ? best_cell : fallback_cell;
        if (selected >= 0 && selected != current_cell_index) {
          result.resizes.push_back(
              ResizeAction{design.pin2node_map[driver], selected});
          repaired = true;
          const auto& selected_cell = liberty.cells[selected];
          const double selected_slew = cellOutputSlew(
              selected_cell, input_slew_rise, input_slew_fall, driver_load);
          driver_slew_violation
              = selected_cell.max_slew > 0.0
                && selected_slew
                       > selected_cell.max_slew
                             * (1.0 - config.slew_margin);
          driver_cap_violation
              = selected_cell.max_cap > 0.0
                && driver_load
                       > selected_cell.max_cap * (1.0 - config.cap_margin);
        }
      }
    }

    const int current_cell_index = cellIndexForNode(
        design, design.pin2node_map[driver]);
    double max_cap = kInf;
    if (current_cell_index >= 0
        && current_cell_index < static_cast<int>(liberty.cells.size())) {
      const auto& current_cell = liberty.cells[current_cell_index];
      if (driver_cap_violation) {
        max_cap = current_cell.max_cap * (1.0 - config.cap_margin);
      } else if (driver_slew_violation && current_cell.max_slew > 0.0) {
        max_cap = slewLoadCap(current_cell,
                              current_cell.max_slew
                                  * (1.0 - config.slew_margin),
                              input_slew_rise,
                              input_slew_fall);
      }
    }
    const bool needs_buffer
        = driver_cap_violation || load_slew_violation || load_cap_violation
          || (driver_slew_violation && std::isfinite(max_cap)
              && max_cap > 0.0);
    if (needs_buffer && config.insert_buffers) {
      const auto net_it = net_index_by_id.find(net_id);
      if (net_it == net_index_by_id.end()) {
        continue;
      }
      const int net_index = net_it->second;
      if (net_index < 0
          || net_index + 1
                 >= static_cast<int>(steiner.net_flat_topo_sort_start.size())) {
        continue;
      }
      std::vector<std::vector<int>> children;
      std::string child_reason;
      if (!rebufferBuildNetChildren(
              steiner, net_index, children, child_reason)) {
        continue;
      }
      if (driver >= static_cast<int>(children.size())
          || children[driver].empty()) {
        continue;
      }

      if (config.repair_net_tree) {
        std::vector<BufferAction> net_actions;
        std::string tree_reason;
        if (repairNetTree(design,
                          timing,
                          steiner,
                          liberty,
                          characterization,
                          config,
                          net_id,
                          max_cap,
                          driver,
                          children,
                          net_actions,
                          tree_reason)) {
          for (std::size_t index = 0; index < net_actions.size(); ++index) {
            auto& action = net_actions[index];
            if (action_count >= config.max_actions) {
              break;
            }
            // ``repairNetTree`` walks from the loads toward the driver, so
            // the emitted vector is downstream-first.  The native IDB
            // transaction applies smaller depths first; reverse the local
            // order so an upstream ancestor moves its loads before the
            // dependent repeater is applied to the new downstream net.
            action.tree_depth = static_cast<int>(net_actions.size() - index - 1);
            result.actions.push_back(std::move(action));
            ++action_count;
            repaired = true;
          }
        }
      } else {
      int selected_child = -1;
      SubtreeLoad selected_subtree;
      for (const int child : children[driver]) {
        const auto subtree = subtreeLoad(
            design, timing, steiner, config, children, child, driver);
        if (subtree.sinks.empty()) {
          continue;
        }
        if (selected_child < 0 || subtree.load > selected_subtree.load) {
          selected_child = child;
          selected_subtree = subtree;
        }
      }
      if (selected_child < 0 || selected_subtree.sinks.empty()) {
        continue;
      }

      const int buffer_cell = chooseBufferCell(
          liberty,
          characterization,
          timing.pin_slew_rise[driver],
          timing.pin_slew_fall[driver],
          selected_subtree.load);
      if (buffer_cell < 0) {
        continue;
      }

      int x_dbu = 0;
      int y_dbu = 0;
      // The Steiner tree is rooted at the driver *pin*.  ``pin2node_map``
      // identifies the owning cell and may point to its placement center,
      // which is a different coordinate domain from the electrical wire
      // endpoint.  Using the cell center here moves a repair_design action
      // off the active edge and forces the placeio_ecc overlay to project it
      // back onto the native topology.
      const int driver_node = driver;
      if (selected_child >= steiner.steiner_node_base
          && selected_child < static_cast<int>(steiner.node_x.size())
          && driver_node >= 0
          && driver_node < static_cast<int>(steiner.node_x.size())) {
        x_dbu = static_cast<int>(std::lround(steiner.node_x[selected_child]));
        y_dbu = static_cast<int>(std::lround(steiner.node_y[selected_child]));
      } else if (driver_node >= 0
                 && driver_node < static_cast<int>(steiner.node_x.size())
                 && selected_child >= 0
                 && selected_child < static_cast<int>(steiner.node_x.size())) {
        x_dbu = static_cast<int>(std::lround(
            0.5 * (steiner.node_x[driver_node]
                   + steiner.node_x[selected_child])));
        y_dbu = static_cast<int>(std::lround(
            0.5 * (steiner.node_y[driver_node]
                   + steiner.node_y[selected_child])));
      } else {
        continue;
      }

      BufferAction action;
      action.net_id = net_id;
      action.driver_pin = driver;
      action.downstream_pins = selected_subtree.sinks;
      action.buffer_cell_index = buffer_cell;
      action.x_dbu = x_dbu;
      action.y_dbu = y_dbu;
      action.downstream_load = selected_subtree.load;
      action.target_load = targetLoadForCell(
          liberty.cells[buffer_cell],
          characterization.target_slew[0],
          characterization.target_slew[1]);
      action.reason = "load_slew";
      result.actions.push_back(std::move(action));
      ++action_count;
      repaired = true;
      }
    }

    if (repaired) {
      ++result.repaired_nets;
    }
  }

  return true;
}

}  // namespace rebuffer
