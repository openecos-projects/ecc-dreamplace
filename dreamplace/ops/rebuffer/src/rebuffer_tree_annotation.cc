// SPDX-License-Identifier: BSD-3-Clause
#include "rebuffer_tree_annotation.hh"

#include <algorithm>

namespace rebuffer {

void appendTreeLoads(std::vector<int>& target, const std::vector<int>& source)
{
  target.insert(target.end(), source.begin(), source.end());
  std::sort(target.begin(), target.end());
  target.erase(std::unique(target.begin(), target.end()), target.end());
}

void annotateTreeLoads(const DriverTreeNode& node, const TimingView& timing,
                       int driver, TreeOption& option)
{
  appendTreeLoads(option.loads, node.load_pin_ids);
  for (const int pin : node.load_pin_ids) {
    if (pin >= 0 && pin < static_cast<int>(timing.pin_cap.size())) {
      option.cap += timing.pin_cap[pin];
    }
    if (!timing.pin_arrival_rise.empty()) {
      const double rise = timing.pin_required_rise[pin] - timing.pin_arrival_rise[pin];
      const double fall = timing.pin_required_fall[pin] - timing.pin_arrival_fall[pin];
      const double budget = rise <= fall
          ? timing.pin_required_rise[pin] - timing.pin_arrival_rise[driver]
          : timing.pin_required_fall[pin] - timing.pin_arrival_fall[driver];
      option.slack = std::min(option.slack, budget);
    } else if (pin >= 0 && pin < static_cast<int>(timing.pin_slack.size())) {
      option.slack = std::min(option.slack, timing.pin_slack[pin]);
    } else {
      option.slack = std::min(option.slack, 0.0);
    }
    if (pin >= 0 && pin < static_cast<int>(timing.pin_slew_rise.size())) {
      option.slew_rise = std::max(option.slew_rise, timing.pin_slew_rise[pin]);
    }
    if (pin >= 0 && pin < static_cast<int>(timing.pin_slew_fall.size())) {
      option.slew_fall = std::max(option.slew_fall, timing.pin_slew_fall[pin]);
    }
    if (pin >= 0 && pin < static_cast<int>(timing.pin_max_slew.size())
        && timing.pin_max_slew[pin] > 0.0) {
      option.max_load_slew = std::min(option.max_load_slew, timing.pin_max_slew[pin]);
    }
    ++option.fanout;
  }
}

TreeRootTiming annotateTreeRoot(const DesignView& design,
                                const TimingView& timing,
                                const LibertyView& liberty, int driver)
{
  TreeRootTiming root;
  const int pin = rebufferLibPinIndex(design, driver);
  if (pin >= 0 && pin < static_cast<int>(design.pin_drive_resistance.size())) {
    root.resistance = design.pin_drive_resistance[pin];
  }
  if (root.resistance <= 0.0 && driver >= 0 && driver < design.num_pins) {
    const int cell = rebufferCellIndexForNode(design, design.pin2node_map[driver]);
    if (cell >= 0 && cell < static_cast<int>(liberty.cells.size())) {
      root.resistance = liberty.cells[cell].drive_resistance;
    }
  }
  if (driver >= 0 && driver < static_cast<int>(timing.pin_load.size())) {
    root.current_load = timing.pin_load[driver];
  }
  if (driver >= 0 && driver < static_cast<int>(timing.pin_cap.size())) {
    root.output_cap = timing.pin_cap[driver];
  }
  return root;
}

double treeRootCorrection(const TreeRootTiming& root, double cap)
{
  // Anchor the fixed driver model at the TP snapshot; d0 cancels.
  return root.resistance * (root.current_load - cap - root.output_cap);
}

}  // namespace rebuffer
