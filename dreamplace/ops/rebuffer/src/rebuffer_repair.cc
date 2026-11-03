// SPDX-License-Identifier: BSD-3-Clause

#include "rebuffer_repair.hh"
#include "rebuffer_driver_input_slew.hh"
#include "rebuffer_cell_selection.hh"

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

bool isCellLoadPin(const DesignView& design, int pin);

bool driverInputSlew(const DesignView& design,
                     const TimingView& timing,
                     const LibertyView& liberty,
                     const std::vector<std::vector<int>>& node_pins,
                     int driver,
                     double& input_slew_rise,
                     double& input_slew_fall,
                     int cell_index_override = -1)
{
  if (driver < 0 || driver >= design.num_pins
      || driver >= static_cast<int>(design.pin2node_map.size())) {
    return false;
  }
  const int node = design.pin2node_map[driver];
  if (node < 0 || node >= static_cast<int>(node_pins.size())) {
    return false;
  }
  const int cell_index = cell_index_override >= 0
                             ? cell_index_override
                             : cellIndexForNode(design, node);
  if (cell_index < 0
      || cell_index >= static_cast<int>(liberty.cells.size())) {
    return false;
  }
  const auto& cell = liberty.cells[cell_index];
  return driverInputSlewFromPins(driver, design.num_pins, cell, node_pins[node],
      timing.pin_slew_rise, timing.pin_slew_fall,
      [&](const TimingArcView& arc, int pin) {
        const int offset = design.pin_2_libpin_offset[pin];
        const std::string pin_port = pin < static_cast<int>(design.pin_names.size())
            ? portName(design.pin_names[pin]) : std::string();
        return !arc.source_name.empty() && !pin_port.empty()
            ? arc.source_name == pin_port : arc.source == offset;
      },
      [&](int pin) { return isCellLoadPin(design, pin); },
      input_slew_rise, input_slew_fall);
}

bool cellPinCap(const DesignView& design,
                int cell_index,
                int pin,
                double& cap)
{
  if (cell_index < 0
      || cell_index + 1
             >= static_cast<int>(design.cell_id_2_libpin_id_start.size())
      || pin < 0 || pin >= static_cast<int>(design.pin_2_libpin_offset.size())
      || design.pin_cap.empty()) {
    return false;
  }
  const int lib_pin = design.cell_id_2_libpin_id_start[cell_index]
                      + design.pin_2_libpin_offset[pin];
  if (lib_pin < 0 || lib_pin >= static_cast<int>(design.pin_cap.size())) {
    return false;
  }
  cap = design.pin_cap[lib_pin];
  return std::isfinite(cap) && cap >= 0.0;
}

bool isCellLoadPin(const DesignView& design, int pin)
{
  if (pin < 0 || pin >= static_cast<int>(design.pin2net_map.size())) {
    return false;
  }
  const int net = design.pin2net_map[pin];
  return net >= 0 && net < static_cast<int>(design.net2driver_pin_map.size())
         && design.net2driver_pin_map[net] != pin;
}

bool refreshDriverOutputSlew(
    const DesignView& design,
    const LibertyView& liberty,
    const std::vector<std::vector<int>>& node_pins,
    const std::vector<int>& node_cell_indices,
    int driver,
    TimingView& timing)
{
  if (driver < 0 || driver >= design.num_pins
      || driver >= static_cast<int>(design.pin2node_map.size())
      || driver >= static_cast<int>(timing.pin_load.size())) {
    return false;
  }
  const int node = design.pin2node_map[driver];
  if (node < 0 || node >= static_cast<int>(node_cell_indices.size())) {
    return false;
  }
  const int cell_index = node_cell_indices[node];
  if (cell_index < 0 || cell_index >= static_cast<int>(liberty.cells.size())) {
    return false;
  }
  double input_rise = timing.pin_slew_rise[driver];
  double input_fall = timing.pin_slew_fall[driver];
  driverInputSlew(design,
                  timing,
                  liberty,
                  node_pins,
                  driver,
                  input_rise,
                  input_fall,
                  cell_index);
  const auto& cell = liberty.cells[cell_index];
  const double output_rise = cellOutputSlewRise(
      cell, input_rise, timing.pin_load[driver]);
  const double output_fall = cellOutputSlewFall(
      cell, input_fall, timing.pin_load[driver]);
  timing.pin_slew_rise[driver] = output_rise;
  timing.pin_slew_fall[driver] = output_fall;

  if (driver >= static_cast<int>(design.pin2net_map.size())) {
    return true;
  }
  const int net = design.pin2net_map[driver];
  if (net < 0 || net + 1 >= static_cast<int>(design.flat_net2pin_start_map.size())) {
    return true;
  }
  const int begin = design.flat_net2pin_start_map[net];
  const int end = design.flat_net2pin_start_map[net + 1];
  for (int index = begin; index < end; ++index) {
    if (index < 0 || index >= static_cast<int>(design.flat_net2pin_map.size())) {
      continue;
    }
    const int pin = design.flat_net2pin_map[index];
    if (pin < 0 || pin >= design.num_pins || pin == driver) {
      continue;
    }
    timing.pin_slew_rise[pin] = output_rise;
    timing.pin_slew_fall[pin] = output_fall;
  }
  return true;
}

bool updateResizedCellState(
    const DesignView& design,
    const LibertyView& liberty,
    const std::vector<std::vector<int>>& node_pins,
    const std::vector<int>& node_cell_indices,
    int node,
    int old_cell_index,
    int new_cell_index,
    TimingView& timing)
{
  if (node < 0 || node >= static_cast<int>(node_pins.size())) {
    return false;
  }
  for (const int pin : node_pins[node]) {
    if (!isCellLoadPin(design, pin)) {
      continue;
    }
    double old_cap = 0.0;
    double new_cap = 0.0;
    if (!cellPinCap(design, old_cell_index, pin, old_cap)
        || !cellPinCap(design, new_cell_index, pin, new_cap)) {
      return false;
    }
    const double delta = new_cap - old_cap;
    if (pin >= 0 && pin < static_cast<int>(timing.pin_cap.size())) {
      timing.pin_cap[pin] = new_cap;
    }
    const int net = pin < static_cast<int>(design.pin2net_map.size())
                        ? design.pin2net_map[pin]
                        : -1;
    if (net < 0 || net >= static_cast<int>(design.net2driver_pin_map.size())) {
      continue;
    }
    const int upstream_driver = design.net2driver_pin_map[net];
    if (upstream_driver < 0
        || upstream_driver >= static_cast<int>(timing.pin_load.size())) {
      continue;
    }
    timing.pin_load[upstream_driver] += delta;
    const int upstream_node = upstream_driver < design.num_pins
                                  && upstream_driver
                                         < static_cast<int>(design.pin2node_map.size())
                              ? design.pin2node_map[upstream_driver]
                              : -1;
    // A primary port can drive a resized cell input but has no Liberty cell
    // whose output slew can be recomputed.  The load delta still belongs to
    // the upstream net; only the cell-specific slew refresh is unavailable.
    if (upstream_node < 0
        || upstream_node >= static_cast<int>(node_cell_indices.size())
        || node_cell_indices[upstream_node] < 0) {
      continue;
    }
    if (!refreshDriverOutputSlew(design,
                                 liberty,
                                 node_pins,
                                 node_cell_indices,
                                 upstream_driver,
                                 timing)) {
      return false;
    }
  }
  return true;
}

void annotateRepairDesignLoadSlew(const DesignView& design,
                                  TimingView& timing)
{
  for (int net_id = 0; net_id < design.num_nets; ++net_id) {
    if (net_id + 1
        >= static_cast<int>(design.flat_net2pin_start_map.size())) {
      continue;
    }
    const int driver = design.net2driver_pin_map[net_id];
    const int begin = design.flat_net2pin_start_map[net_id];
    const int end = design.flat_net2pin_start_map[net_id + 1];
    for (int index = begin; index < end; ++index) {
      if (index < 0
          || index >= static_cast<int>(design.flat_net2pin_map.size())) {
        continue;
      }
      const int pin = design.flat_net2pin_map[index];
      if (pin < 0 || pin >= design.num_pins || pin == driver
          || pin >= static_cast<int>(timing.pin_max_slew.size())) {
        continue;
      }
      const double max_slew = timing.pin_max_slew[pin];
      if (!(max_slew > 0.0) || !std::isfinite(max_slew)) {
        continue;
      }
      if (timing.pin_slew_rise[pin] > max_slew) {
        timing.pin_slew_rise[pin] = max_slew;
      }
      if (timing.pin_slew_fall[pin] > max_slew) {
        timing.pin_slew_fall[pin] = max_slew;
      }
    }
  }
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

void splitFanoutRegion(const SteinerView& steiner,
                       std::vector<int> pins,
                       int max_fanout,
                       std::vector<std::vector<int>>& leaves)
{
  if (max_fanout <= 0 || static_cast<int>(pins.size()) <= max_fanout) {
    leaves.push_back(std::move(pins));
    return;
  }
  double min_x = kInf;
  double max_x = -kInf;
  double min_y = kInf;
  double max_y = -kInf;
  for (const int pin : pins) {
    if (pin >= 0 && pin < static_cast<int>(steiner.node_x.size())) {
      min_x = std::min(min_x, steiner.node_x[pin]);
      max_x = std::max(max_x, steiner.node_x[pin]);
      min_y = std::min(min_y, steiner.node_y[pin]);
      max_y = std::max(max_y, steiner.node_y[pin]);
    }
  }
  const bool split_x = (max_x - min_x) >= (max_y - min_y);
  std::stable_sort(pins.begin(), pins.end(), [&](int lhs, int rhs) {
    const double lhs_primary
        = lhs >= 0 && lhs < static_cast<int>(steiner.node_x.size())
              ? (split_x ? steiner.node_x[lhs] : steiner.node_y[lhs])
              : 0.0;
    const double rhs_primary
        = rhs >= 0 && rhs < static_cast<int>(steiner.node_x.size())
              ? (split_x ? steiner.node_x[rhs] : steiner.node_y[rhs])
              : 0.0;
    if (lhs_primary != rhs_primary) {
      return lhs_primary < rhs_primary;
    }
    const double lhs_secondary
        = lhs >= 0 && lhs < static_cast<int>(steiner.node_x.size())
              ? (split_x ? steiner.node_y[lhs] : steiner.node_x[lhs])
              : 0.0;
    const double rhs_secondary
        = rhs >= 0 && rhs < static_cast<int>(steiner.node_x.size())
              ? (split_x ? steiner.node_y[rhs] : steiner.node_x[rhs])
              : 0.0;
    if (lhs_secondary != rhs_secondary) {
      return lhs_secondary < rhs_secondary;
    }
    return lhs < rhs;
  });
  const auto middle = pins.begin() + static_cast<std::ptrdiff_t>(pins.size() / 2);
  std::vector<int> left(pins.begin(), middle);
  std::vector<int> right(middle, pins.end());
  splitFanoutRegion(steiner, std::move(left), max_fanout, leaves);
  splitFanoutRegion(steiner, std::move(right), max_fanout, leaves);
}

bool repairFanoutRegions(const DesignView& design,
                         const TimingView& timing,
                         const SteinerView& steiner,
                         const LibertyView& liberty,
                         const CharacterizationResult& characterization,
                         const RepairConfig& config,
                         int net_id,
                         int driver,
                         int pin_begin,
                         int pin_end,
                         std::vector<BufferAction>& actions)
{
  if (config.max_fanout <= 0) {
    return false;
  }
  std::vector<int> loads;
  for (int index = pin_begin; index < pin_end; ++index) {
    if (index < 0 || index >= static_cast<int>(design.flat_net2pin_map.size())) {
      continue;
    }
    const int pin = design.flat_net2pin_map[index];
    if (pin >= 0 && pin < design.num_pins && pin != driver) {
      loads.push_back(pin);
    }
  }
  if (static_cast<int>(loads.size()) <= config.max_fanout) {
    return false;
  }
  std::vector<std::vector<int>> regions;
  splitFanoutRegion(steiner, std::move(loads), config.max_fanout, regions);
  double input_slew_rise = timing.pin_slew_rise[driver];
  double input_slew_fall = timing.pin_slew_fall[driver];
  double driver_x = 0.0;
  double driver_y = 0.0;
  if (driver >= 0 && driver < static_cast<int>(steiner.node_x.size())) {
    driver_x = steiner.node_x[driver];
    driver_y = steiner.node_y[driver];
  }
  bool inserted = false;
  for (const auto& region : regions) {
    if (region.empty()) {
      continue;
    }
    double load = 0.0;
    double x = 0.0;
    double y = 0.0;
    int coordinate_count = 0;
    for (const int pin : region) {
      if (pin >= 0 && pin < static_cast<int>(timing.pin_cap.size())) {
        load += timing.pin_cap[pin];
      }
      if (pin >= 0 && pin < static_cast<int>(steiner.node_x.size())) {
        x += steiner.node_x[pin];
        y += steiner.node_y[pin];
        ++coordinate_count;
      }
    }
    const int buffer_cell = rebufferChooseBufferCell(
        liberty, characterization, input_slew_rise, input_slew_fall, load,
        config.repair_design_target_cell_math,
        config.repair_design_target_cell_selection);
    if (buffer_cell < 0) {
      continue;
    }
    if (coordinate_count > 0) {
      x /= coordinate_count;
      y /= coordinate_count;
    } else {
      x = 0.5 * (driver_x + x);
      y = 0.5 * (driver_y + y);
    }
    BufferAction action;
    action.net_id = net_id;
    action.driver_pin = driver;
    action.downstream_pins = region;
    action.buffer_cell_index = buffer_cell;
    action.x_dbu = static_cast<int>(std::lround(x));
    action.y_dbu = static_cast<int>(std::lround(y));
    action.downstream_load = load;
    action.target_load = rebufferTargetLoadForCell(
        liberty.cells[buffer_cell],
        characterization.target_slew[0],
        characterization.target_slew[1],
        config.repair_design_target_cell_math);
    action.reason = "fanout";
    actions.push_back(std::move(action));
    inserted = true;
  }
  return inserted;
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
  if (config.repair_design_reverse_topological) {
    if (timing.pin_rank.size() != static_cast<std::size_t>(design.num_pins)) {
      reason = "missing_repair_design_pin_rank";
      return false;
    }
    for (const int rank : timing.pin_rank) {
      if (rank < 0) {
        reason = "invalid_repair_design_pin_rank";
        return false;
      }
    }
  }
  if (config.repair_design_load_slew_annotation
      && !config.repair_design_reverse_topological) {
    reason = "repair_design_load_slew_annotation_requires_reverse_topological";
    return false;
  }
  if (config.repair_design_local_update
      && !config.repair_design_reverse_topological) {
    reason = "repair_design_local_update_requires_reverse_topological";
    return false;
  }
  if (config.repair_design_local_update) {
    if (design.pin_cap.empty()
        || design.cell_id_2_libpin_id_start.empty()
        || design.pin_cap.size()
               < static_cast<std::size_t>(
                   design.cell_id_2_libpin_id_start.back())) {
      reason = "missing_repair_design_pin_cap_payload";
      return false;
    }
  }
  if (config.repair_design_openroad_repeater_resize
      && !config.repair_net_tree) {
    reason = "repair_design_openroad_repeater_resize_requires_repair_net_tree";
    return false;
  }
  if (config.repair_design_target_cell_selection
      && (!config.repair_design_target_cell_math || !liberty.cell_metadata)) {
    reason = "repair_design_target_cell_selection_requires_math_and_library_metadata";
    return false;
  }

  std::vector<std::vector<int>> node_pins(design.num_nodes);
  for (int pin = 0; pin < design.num_pins; ++pin) {
    const int node = design.pin2node_map[pin];
    if (node >= 0 && node < design.num_nodes) {
      node_pins[node].push_back(pin);
    }
  }

  TimingView working_timing = timing;
  std::vector<int> node_cell_indices(design.num_nodes, -1);
  for (int node = 0; node < design.num_nodes; ++node) {
    node_cell_indices[node] = cellIndexForNode(design, node);
  }
  if (config.repair_design_load_slew_annotation) {
    annotateRepairDesignLoadSlew(design, working_timing);
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

  std::vector<int> net_order;
  net_order.reserve(design.num_nets);
  for (int net_id = 0; net_id < design.num_nets; ++net_id) {
    net_order.push_back(net_id);
  }
  if (config.repair_design_reverse_topological) {
    const auto rankForNet = [&](int net_id) {
      if (net_id < 0
          || net_id >= static_cast<int>(design.net2driver_pin_map.size())) {
        return std::numeric_limits<int>::min();
      }
      const int driver = design.net2driver_pin_map[net_id];
      if (driver < 0
          || driver >= static_cast<int>(working_timing.pin_rank.size())) {
        return std::numeric_limits<int>::min();
      }
      return working_timing.pin_rank[driver];
    };
    // net_order starts in physical net-id order, so stable_sort gives the
    // deterministic net-id tie-break used by the native planner.
    std::stable_sort(
        net_order.begin(),
        net_order.end(),
        [&](int lhs, int rhs) { return rankForNet(lhs) > rankForNet(rhs); });
  }

  int action_count = 0;
  const bool use_repair_net_tree
      = config.repair_net_tree
        || config.repair_design_max_wire_length_um > 0.0;
  for (const int net_id : net_order) {
    if (action_count >= config.max_actions) {
      break;
    }
    const int driver = design.net2driver_pin_map[net_id];
    if (driver < 0 || driver >= design.num_pins) {
      continue;
    }
    if (isClockPin(working_timing, driver)) {
      continue;
    }
    const int pin_begin = design.flat_net2pin_start_map[net_id];
    const int pin_end = design.flat_net2pin_start_map[net_id + 1];
    if (pin_end - pin_begin < 2) {
      continue;
    }
    int load_pin_count = 0;
    for (int index = pin_begin; index < pin_end; ++index) {
      if (index < 0 || index >= static_cast<int>(design.flat_net2pin_map.size())) {
        continue;
      }
      const int pin = design.flat_net2pin_map[index];
      if (pin >= 0 && pin < design.num_pins && pin != driver) {
        ++load_pin_count;
      }
    }
    const bool fanout_violation
        = config.repair_design_fanout_repair
          && config.max_fanout > 0 && load_pin_count > config.max_fanout;
    bool clock_net = isClockPin(working_timing, driver);
    if (!clock_net) {
      for (int index = pin_begin; index < pin_end; ++index) {
        if (index < 0
            || index >= static_cast<int>(design.flat_net2pin_map.size())) {
          continue;
        }
        if (isClockPin(working_timing, design.flat_net2pin_map[index])) {
          clock_net = true;
          break;
        }
      }
    }

    const double driver_load = working_timing.pin_load[driver];
    const double driver_max_slew = working_timing.pin_max_slew[driver];
    const double driver_max_cap = working_timing.pin_max_cap[driver];
    double input_slew_rise = working_timing.pin_slew_rise[driver];
    double input_slew_fall = working_timing.pin_slew_fall[driver];
    const int driver_node = design.pin2node_map[driver];
    const int initial_cell_index =
        driver_node >= 0 && driver_node < static_cast<int>(node_cell_indices.size())
            ? node_cell_indices[driver_node]
            : -1;
    driverInputSlew(design,
                    working_timing,
                    liberty,
                    node_pins,
                    driver,
                    input_slew_rise,
                    input_slew_fall,
                    initial_cell_index);
    if (config.repair_design_local_update
        || config.repair_design_load_slew_annotation) {
      refreshDriverOutputSlew(design,
                              liberty,
                              node_pins,
                              node_cell_indices,
                              driver,
                              working_timing);
    }
    const double driver_slew = std::max(
        working_timing.pin_slew_rise[driver],
        working_timing.pin_slew_fall[driver]);

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
      const double max_slew = working_timing.pin_max_slew[pin];
      if (max_slew > 0.0
          && std::max(working_timing.pin_slew_rise[pin],
                      working_timing.pin_slew_fall[pin])
                 > max_slew * (1.0 - config.slew_margin)) {
        load_slew_violation = true;
      }
      const double max_cap = working_timing.pin_max_cap[pin];
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
        && !load_slew_violation && !load_cap_violation
        && !fanout_violation
        && config.repair_design_max_wire_length_um <= 0.0) {
      continue;
    }

    bool repaired = false;
    if (fanout_violation && !clock_net && config.insert_buffers) {
      std::vector<BufferAction> fanout_actions;
      if (repairFanoutRegions(design,
                              working_timing,
                              steiner,
                              liberty,
                              characterization,
                              config,
                              net_id,
                              driver,
                              pin_begin,
                              pin_end,
                              fanout_actions)) {
        for (auto& action : fanout_actions) {
          if (action_count >= config.max_actions) {
            break;
          }
          result.actions.push_back(std::move(action));
          ++action_count;
          repaired = true;
        }
      }
    }

    int current_cell_index = initial_cell_index;
    const bool resize_slew_violation
        = driver_max_slew > 0.0
          && driver_slew
                 > driver_max_slew
                       * (config.repair_design_openroad_driver_resize
                              ? (1.0 - config.slew_margin)
                              : (1.0 + config.resize_slew_threshold));
    if (resize_slew_violation && config.resize_drivers) {
      if (current_cell_index >= 0
          && current_cell_index < static_cast<int>(liberty.cells.size())) {
        const auto& current_cell = liberty.cells[current_cell_index];
        int selected = -1;
        const int main_id = current_cell.main_id;
        if (main_id >= 0
            && main_id + 1
                   < static_cast<int>(design.main_id_2_cell_id_start.size())) {
          const int begin = design.main_id_2_cell_id_start[main_id];
          const int end = design.main_id_2_cell_id_start[main_id + 1];
          if (config.repair_design_openroad_driver_resize
              && config.repair_design_target_cell_selection
              && current_cell.is_buffer) {
            selected = findBufferSlewRepairCell(liberty, current_cell_index,
                input_slew_rise, input_slew_fall, driver_load, config.slew_margin);
          } else if (config.repair_design_openroad_driver_resize) {
            double best_violation = kInf;
            double best_area = kInf;
            for (int cell_id = begin; cell_id < end; ++cell_id) {
              if (cell_id < 0
                  || cell_id >= static_cast<int>(liberty.cells.size())) {
                continue;
              }
              const auto& cell = liberty.cells[cell_id];
              const double output_slew = cellOutputSlew(
                  cell, input_slew_rise, input_slew_fall, driver_load);
              const double limit = cell.max_slew > 0.0
                                       ? cell.max_slew
                                             * (1.0 - config.slew_margin)
                                       : kInf;
              const double violation
                  = std::isfinite(limit)
                        ? std::max(0.0, output_slew - limit)
                        : 0.0;
              const bool both_good = violation == 0.0
                                     && best_violation == 0.0;
              if (selected < 0 || violation < best_violation
                  || (both_good
                      && (cell.area < best_area
                          || (cell.area == best_area && cell_id < selected)))) {
                selected = cell_id;
                best_violation = violation;
                best_area = cell.area;
              }
            }
          } else {
            double best_area = kInf;
            double fallback_area = -1.0;
            int best_cell = -1;
            int fallback_cell = -1;
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
            selected = best_cell >= 0 ? best_cell : fallback_cell;
          }
        }
        if (selected >= 0 && selected != current_cell_index) {
          result.resizes.push_back(
              ResizeAction{design.pin2node_map[driver], selected});
          repaired = true;
          const auto& selected_cell = liberty.cells[selected];
          const double selected_slew_rise = cellOutputSlewRise(
              selected_cell, input_slew_rise, driver_load);
          const double selected_slew_fall = cellOutputSlewFall(
              selected_cell, input_slew_fall, driver_load);
          if (config.repair_design_local_update) {
            if (!updateResizedCellState(design,
                                        liberty,
                                        node_pins,
                                        node_cell_indices,
                                        driver_node,
                                        current_cell_index,
                                        selected,
                                        working_timing)) {
              reason = "repair_design_local_resize_update_failed";
              return false;
            }
            node_cell_indices[driver_node] = selected;
            working_timing.pin_slew_rise[driver] = selected_slew_rise;
            working_timing.pin_slew_fall[driver] = selected_slew_fall;
          }
          current_cell_index = selected;
          const double selected_slew_check = cellOutputSlew(
              selected_cell, input_slew_rise, input_slew_fall, driver_load);
          driver_slew_violation
              = selected_cell.max_slew > 0.0
                && selected_slew_check
                       > selected_cell.max_slew
                             * (1.0 - config.slew_margin);
          driver_cap_violation
              = selected_cell.max_cap > 0.0
                && driver_load
                       > selected_cell.max_cap * (1.0 - config.cap_margin);
        }
      }
    }

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
              && max_cap > 0.0)
          || config.repair_design_max_wire_length_um > 0.0;
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

      if (use_repair_net_tree) {
        std::vector<BufferAction> net_actions;
        std::string tree_reason;
        if (repairNetTree(design,
                          working_timing,
                          steiner,
                          liberty,
                          characterization,
                          config,
                          net_id,
                          max_cap,
                          driver,
                          current_cell_index,
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
            design, working_timing, steiner, config, children, child, driver);
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

      const int buffer_cell = rebufferChooseBufferCell(
          liberty,
          characterization,
          working_timing.pin_slew_rise[driver],
          working_timing.pin_slew_fall[driver],
          selected_subtree.load,
          config.repair_design_target_cell_math,
          config.repair_design_target_cell_selection);
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
      action.target_load = rebufferTargetLoadForCell(
          liberty.cells[buffer_cell],
          characterization.target_slew[0],
          characterization.target_slew[1],
          config.repair_design_target_cell_math);
      action.reason = "load_slew";
      result.actions.push_back(std::move(action));
      ++action_count;
      repaired = true;
      if (config.repair_design_local_update
          && buffer_cell >= 0
          && buffer_cell < static_cast<int>(liberty.cells.size())) {
        working_timing.pin_load[driver] = std::max(
            0.0,
            working_timing.pin_load[driver] - selected_subtree.load
                + liberty.cells[buffer_cell].input_cap);
        refreshDriverOutputSlew(design,
                                liberty,
                                node_pins,
                                node_cell_indices,
                                driver,
                                working_timing);
      }
      }
    }

    if (repaired) {
      ++result.repaired_nets;
    }
  }

  return true;
}

}  // namespace rebuffer
