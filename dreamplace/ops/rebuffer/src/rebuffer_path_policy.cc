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
                              int output_pin)
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
      || input_pin >= static_cast<int>(timing.pin_slew_rise.size())
      || input_pin >= static_cast<int>(timing.pin_slew_fall.size())) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  const bool input_rise = timing.path_transitions[edge] == 0;
  const bool output_rise = timing.path_transitions[path_index] == 0;
  const double input_slew = input_rise
                                ? timing.pin_slew_rise[input_pin]
                                : timing.pin_slew_fall[input_pin];
  const LutTable& delay_table
      = output_rise ? arc->rise_delay : arc->fall_delay;
  const double intrinsic = lutFindValue(delay_table, input_slew, 0.0);
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

std::vector<int> rankPathDriverPins(const DesignView& design,
                                    const TimingView& timing,
                                    const LibertyView& liberty,
                                    const std::vector<int>& path)
{
  std::vector<RankedDriver> ranked;
  ranked.reserve(path.size());
  // The launching register output (path[0]) is a driver too.  The extracted
  // data path starts at Q, so there is no predecessor pin on the path; score
  // it from the cell's own input slews instead of skipping it.  OpenROAD's
  // expanded path keeps the clock pin ahead of Q and ranks it normally.
  if (!path.empty()) {
    const int driver = path.front();
    if (driver >= 0 && driver < design.num_pins
        && driver < static_cast<int>(design.pin2net_map.size())
        && driver < static_cast<int>(timing.pin_load.size())
        && driver < static_cast<int>(design.pin2node_map.size())) {
      const int net_id = design.pin2net_map[driver];
      const int cell_index = cellIndexForNode(design, design.pin2node_map[driver]);
      double slew_rise = 0.0;
      double slew_fall = 0.0;
      if (net_id >= 0 && net_id < static_cast<int>(design.net2driver_pin_map.size())
          && design.net2driver_pin_map[net_id] == driver
          && !rebufferIsClockPin(timing, driver)
          && cell_index >= 0 && cell_index < static_cast<int>(liberty.cells.size())
          && rebufferDriverInputSlew(design, timing, liberty, driver, slew_rise, slew_fall)) {
        const auto& cell = liberty.cells[cell_index];
        const double delay
            = rebufferCellOutputDelay(cell, slew_rise, slew_fall, timing.pin_load[driver])
              - rebufferCellOutputDelay(cell, slew_rise, slew_fall, 0.0);
        ranked.push_back({driver, 0, delay});
      }
    }
  }
  for (int index = 1; index < static_cast<int>(path.size()); ++index) {
    const int driver = path[index];
    if (driver < 0 || driver >= design.num_pins
        || driver >= static_cast<int>(design.pin2net_map.size())
        || driver >= static_cast<int>(timing.pin_load.size())) {
      continue;
    }
    const int net_id = design.pin2net_map[driver];
    if (net_id < 0 || net_id >= static_cast<int>(design.net2driver_pin_map.size())
        || design.net2driver_pin_map[net_id] != driver
        || rebufferIsClockPin(timing, driver)) {
      continue;
    }
    if (driver >= static_cast<int>(design.pin2node_map.size())) {
      continue;
    }
    const int cell_index
        = cellIndexForNode(design, design.pin2node_map[driver]);
    if (cell_index < 0 || cell_index >= static_cast<int>(liberty.cells.size())) {
      continue;
    }
    const int input = path[index - 1];
    if (input < 0 || input >= static_cast<int>(timing.pin_slew_rise.size())
        || input >= static_cast<int>(timing.pin_slew_fall.size())) {
      continue;
    }
    double delay = loadDependentPathDelay(
        timing, liberty.cells[cell_index], index, input, driver);
    if (!std::isfinite(delay)) {
      delay = rebufferCellOutputDelay(
        liberty.cells[cell_index],
        timing.pin_slew_rise[input],
        timing.pin_slew_fall[input],
        timing.pin_load[driver]);
    }
    ranked.push_back({driver, index, delay});
  }

  std::stable_sort(
      ranked.begin(),
      ranked.end(),
      [](const RankedDriver& lhs, const RankedDriver& rhs) {
        if (std::abs(lhs.delay - rhs.delay) > 1.0e-12) {
          return lhs.delay > rhs.delay;
        }
        return lhs.path_index > rhs.path_index;
      });

  std::vector<int> result;
  result.reserve(ranked.size());
  for (const auto& item : ranked) {
    result.push_back(item.pin);
  }
  return result;
}

}  // namespace rebuffer
