// SPDX-License-Identifier: BSD-3-Clause
#include "rebuffer_tree_electrical.hh"

#include <algorithm>
#include <cmath>
#include <limits>

namespace rebuffer {

double treeMarginedLimit(double limit, double margin)
{
  return limit > 0.0 ? limit * (1.0 - margin / 100.0)
                     : std::numeric_limits<double>::infinity();
}

bool treeCanDrive(const TreeOption& option, double resistance,
                  double output_cap, double max_cap, const RepairConfig& config)
{
  if (!config.tree_enforce_electrical) return true;
  const double cap = option.cap + output_cap;
  const double slew = (resistance + config.r_unit * option.max_load_wire)
                      * cap * config.slew_shape_factor;
  return cap <= treeMarginedLimit(max_cap, config.cap_margin)
         && slew <= treeMarginedLimit(option.max_load_slew, config.slew_margin)
         && (config.max_fanout <= 0 || option.fanout <= config.max_fanout);
}

bool treeRootFeasible(const TreeOption& option, const TreeRootTiming& root,
                      const TimingView& timing, int driver,
                      const RepairConfig& config)
{
  const double max_cap = driver < static_cast<int>(timing.pin_max_cap.size())
                             ? timing.pin_max_cap[driver] : 0.0;
  if (!treeCanDrive(option, root.resistance, root.output_cap, max_cap, config)) {
    return false;
  }
  if (!config.tree_enforce_electrical) return true;
  const double max_slew = driver < static_cast<int>(timing.pin_max_slew.size())
                              ? timing.pin_max_slew[driver] : 0.0;
  const double observed = std::max(
      driver < static_cast<int>(timing.pin_slew_rise.size()) ? timing.pin_slew_rise[driver] : 0.0,
      driver < static_cast<int>(timing.pin_slew_fall.size()) ? timing.pin_slew_fall[driver] : 0.0);
  const double slew = std::max(0.0, observed
      - config.slew_shape_factor * treeRootCorrection(root, option.cap));
  return slew <= treeMarginedLimit(max_slew, config.slew_margin);
}

double treeWireStep(const LibertyView& liberty,
                    const CharacterizationResult& characterization,
                    const RepairConfig& config)
{
  double step = config.tree_max_wire_length_um > 0.0
                    ? config.tree_max_wire_length_um
                    : std::numeric_limits<double>::infinity();
  if (!config.tree_enforce_electrical) return step;
  // OpenROAD derives the shared step from the smallest fast buffer's
  // ability to drive another copy of itself under cap and load-slew limits.
  if (characterization.fast_buffers.empty()) return step;
  const auto& buffer = characterization.fast_buffers.front();
  if (buffer.cell_index < 0 || buffer.cell_index >= static_cast<int>(liberty.cells.size())) {
    return step;
  }
  const auto& cell = liberty.cells[buffer.cell_index];
  const double cap = buffer.input_cap + cell.output_cap;
  const double cap_limit = treeMarginedLimit(buffer.max_cap, config.cap_margin);
  if (config.c_unit > 0.0) step = std::min(step, std::max(0.0, (cap_limit - cap) / config.c_unit));
  const double limit = treeMarginedLimit(buffer.max_slew, config.slew_margin)
                       / config.slew_shape_factor;
  if (std::isfinite(limit)) {
    const double a = config.r_unit * config.c_unit;
    const double b = config.r_unit * cap + buffer.drive_resistance * config.c_unit;
    const double c = buffer.drive_resistance * cap - limit;
    // Stable positive root of a L^2 + b L + c = 0, also covering zero RC.
    const double length = c >= 0.0 ? 0.0
        : (a > 0.0 ? -2.0 * c / (b + std::sqrt(b * b - 4.0 * a * c))
                    : (b > 0.0 ? -c / b : std::numeric_limits<double>::infinity()));
    step = std::min(step, length);
  }
  return step;
}

}  // namespace rebuffer
