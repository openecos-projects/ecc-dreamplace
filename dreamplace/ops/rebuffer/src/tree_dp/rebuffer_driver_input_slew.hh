// SPDX-License-Identifier: BSD-3-Clause
// Canonical input-slew selection over one instance's pins; storage is borrowed.
#pragma once

#include <algorithm>
#include <vector>

#include "rebuffer_liberty.hh"

namespace rebuffer {

template <typename MatchesSource, typename IsLoad>
bool driverInputSlewFromPins(int driver, int num_pins,
    const LibertyCellView& cell, const std::vector<int>& pins,
    const std::vector<double>& rise_slew, const std::vector<double>& fall_slew,
    const MatchesSource& matches_source, const IsLoad& is_load,
    double& input_rise, double& input_fall)
{
  bool found = false;
  double rise = 0.0, fall = 0.0;
  for (int pin : pins) {
    if (pin == driver || pin < 0 || pin >= num_pins) continue;
    for (const auto& arc : cell.arcs) {
      if (matches_source(arc, pin)) {
        rise = std::max(rise, rise_slew[pin]);
        fall = std::max(fall, fall_slew[pin]);
        found = true;
        break;
      }
    }
  }
  // Lightweight metadata can omit Liberty source names/ids. A physical load
  // of another net remains the existing unambiguous input fallback.
  if (!found) {
    for (int pin : pins) {
      if (pin == driver || !is_load(pin) || pin < 0 || pin >= num_pins) continue;
      rise = std::max(rise, rise_slew[pin]);
      fall = std::max(fall, fall_slew[pin]);
      found = true;
    }
  }
  if (found) { input_rise = rise; input_fall = fall; }
  return found;
}

}  // namespace rebuffer
