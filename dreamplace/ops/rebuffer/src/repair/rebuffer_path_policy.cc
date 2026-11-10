// SPDX-License-Identifier: BSD-3-Clause

#include "rebuffer_path_policy.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace rebuffer {

namespace {

struct RankedDriver
{
  int pin = -1;
  int path_index = -1;
  double delay = 0.0;
};

const TimingArcView* pathArc(const LibertyCellView& cell, int local_id)
{
  for (const TimingArcView& arc : cell.arcs) {
    if (arc.local_id == local_id) {
      return &arc;
    }
  }
  // Older payloads did not preserve local arc ids.  Keep the compact-index
  // fallback for those fixtures, but prefer the identity-bearing lookup.
  if (local_id >= 0 && local_id < static_cast<int>(cell.arcs.size())) {
    return &cell.arcs[local_id];
  }
  return nullptr;
}

double loadDependentPathDelay(const TimingView& timing,
                              const LibertyCellView& cell,
                              int path_index,
                              int input_pin,
                              const PathDriverAccess& access,
                              bool openroad_intrinsic)
{
  const int edge = path_index - 1;
  if (edge < 0 || edge >= static_cast<int>(timing.path_arc_delays.size())
      || edge >= static_cast<int>(timing.path_arc_local_ids.size())
      || path_index >= static_cast<int>(timing.path_transitions.size())
      || edge >= static_cast<int>(timing.path_transitions.size())) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const double actual_delay = timing.path_arc_delays[edge];
  if (!std::isfinite(actual_delay)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const TimingArcView* arc
      = pathArc(cell, timing.path_arc_local_ids[edge]);
  if (arc == nullptr || input_pin < 0
      || input_pin >= static_cast<int>(access.rise_slew.size())
      || input_pin >= static_cast<int>(access.fall_slew.size())) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const bool input_rise = timing.path_transitions[edge] == 0;
  const bool output_rise = timing.path_transitions[path_index] == 0;
  const double input_slew = input_rise
                                ? access.rise_slew[input_pin]
                                : access.fall_slew[input_pin];
  const LutTable& delay_table
      = output_rise ? arc->rise_delay : arc->fall_delay;
  // OpenSTA TimingArc::intrinsicDelay evaluates at slew=0 and load=0.
  const double intrinsic = lutFindValue(
      delay_table, openroad_intrinsic ? 0.0 : input_slew, 0.0);
  return actual_delay - intrinsic;
}

int cellIndexForNode(const DesignView& design, int node)
{
  if (node < 0 || node >= static_cast<int>(design.inst_main_id.size())) {
    return -1;
  }
  const int main_id = design.inst_main_id[node];
  if (main_id < 0
      || main_id + 1
             >= static_cast<int>(design.main_id_2_cell_id_start.size())) {
    return -1;
  }
  return design.main_id_2_cell_id_start[main_id]
         + design.inst_libcell_offset[node];
}

}  // namespace

std::vector<int> rankPathDriverPins(const PathDriverAccess& access,
    const LibertyView& liberty, const std::vector<int>& path,
    const TimingView* path_context, bool openroad_intrinsic, std::vector<double>* scores)
{
  std::vector<RankedDriver> ranked;
  ranked.reserve(path.size());
  auto cell_for_driver = [&](int driver) {
    if (driver < 0 || driver >= access.num_pins
        || driver >= static_cast<int>(access.pin_net.size())
        || driver >= static_cast<int>(access.load.size())
        || driver >= static_cast<int>(access.pin_owner.size())) return -1;
    const int net = access.pin_net[driver];
    if (net < 0 || net >= static_cast<int>(access.net_drivers.size())
        || access.net_drivers[net] != driver || access.is_clock(driver)) return -1;
    const int cell = access.cell_for_node(access.pin_owner[driver]);
    return cell >= 0 && cell < static_cast<int>(liberty.cells.size()) ? cell : -1;
  };
  // The launching register output is ranked from its cell's input slews.
  if (!path.empty()) {
    const int driver = path.front();
    const int cell_index = cell_for_driver(driver);
    double rise = 0.0, fall = 0.0;
    if (cell_index >= 0 && access.input_slew(driver, rise, fall)) {
      const auto& cell = liberty.cells[cell_index];
      const double delay = rebufferCellOutputDelay(cell, rise, fall, access.load[driver])
                           - rebufferCellOutputDelay(cell, rise, fall, 0.0);
      ranked.push_back({driver, 0, delay});
    }
  }
  for (int index = 1; index < static_cast<int>(path.size()); ++index) {
    const int driver = path[index];
    const int cell = cell_for_driver(driver);
    if (cell < 0) continue;
    const int input = path[index - 1];
    if (input < 0 || input >= static_cast<int>(access.rise_slew.size())
        || input >= static_cast<int>(access.fall_slew.size())) continue;
    double delay = path_context == nullptr ? std::numeric_limits<double>::quiet_NaN()
        : loadDependentPathDelay(
              *path_context, liberty.cells[cell], index, input, access, openroad_intrinsic);
    if (!std::isfinite(delay)) {
      delay = rebufferCellOutputDelay(liberty.cells[cell], access.rise_slew[input],
                                     access.fall_slew[input], access.load[driver]);
    }
    ranked.push_back({driver, index, delay});
  }

  std::stable_sort(
      ranked.begin(),
      ranked.end(),
      [openroad_intrinsic](const RankedDriver& lhs, const RankedDriver& rhs) {
        if (openroad_intrinsic ? lhs.delay != rhs.delay
                               : std::abs(lhs.delay - rhs.delay) > 1.0e-12) {
          return lhs.delay > rhs.delay;
        }
        return lhs.path_index > rhs.path_index;
      });

  std::vector<int> result;
  result.reserve(ranked.size());
  if (scores) scores->clear();
  for (const auto& item : ranked) {
    result.push_back(item.pin);
    if (scores) scores->push_back(item.delay);
  }
  return result;
}

std::vector<int> rankPathDriverPins(const DesignView& design,
    const TimingView& timing, const LibertyView& liberty, const std::vector<int>& path,
    bool openroad_intrinsic)
{
  const PathDriverAccess access{design.num_pins, design.pin2net_map,
      design.net2driver_pin_map, design.pin2node_map, timing.pin_slew_rise,
      timing.pin_slew_fall, timing.pin_load,
      [&](int node) { return cellIndexForNode(design, node); },
      [&](int pin) { return rebufferIsClockPin(timing, pin); },
      [&](int pin, double& rise, double& fall) {
        return rebufferDriverInputSlew(design, timing, liberty, pin, rise, fall);
      }};
  return rankPathDriverPins(access, liberty, path, &timing, openroad_intrinsic);
}

}  // namespace rebuffer
