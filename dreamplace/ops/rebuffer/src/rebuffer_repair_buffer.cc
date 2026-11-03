// SPDX-License-Identifier: BSD-3-Clause
// repair_design buffer scoring; the legacy candidate domain stays explicit.

#include "rebuffer_repair.hh"
#include "rebuffer_cell_selection.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rebuffer {
namespace {
constexpr double kInf = std::numeric_limits<double>::infinity();

double legacyTargetLoadForCell(const LibertyCellView& cell,
                               double target_slew_rise,
                               double target_slew_fall)
{
  const double target_slew = std::max(target_slew_rise, target_slew_fall);
  if (cell.max_cap <= 0.0 || target_slew <= 0.0) {
    return 0.0;
  }
  if (rebufferCellOutputSlew(cell, target_slew_rise, target_slew_fall, 0.0)
      >= target_slew) {
    return 0.0;
  }
  double low = 0.0;
  double high = cell.max_cap * 2.0;
  for (int iteration = 0; iteration < 48; ++iteration) {
    const double middle = 0.5 * (low + high);
    if (rebufferCellOutputSlew(cell, target_slew_rise, target_slew_fall, middle)
        < target_slew) {
      low = middle;
    } else {
      high = middle;
    }
  }
  return 0.5 * (low + high);
}

}  // namespace

double rebufferTargetLoadForCell(const LibertyCellView& cell,
                                 double target_slew_rise,
                                 double target_slew_fall,
                                 bool target_cell_math)
{
  return target_cell_math
      ? cellTargetLoad(cell, {target_slew_rise, target_slew_fall})
      : legacyTargetLoadForCell(cell, target_slew_rise, target_slew_fall);
}

int rebufferChooseBufferCell(const LibertyView& liberty,
                             const CharacterizationResult& characterization,
                             double input_slew_rise,
                             double input_slew_fall,
                             double downstream_load,
                             bool target_cell_math,
                             bool target_cell_selection,
                             int initial_cell)
{
  if (target_cell_selection) {
    if (!target_cell_math) throw std::invalid_argument("target-cell selection requires canonical target-cell math");
    if (initial_cell < 0) {
      const int lowest = characterization.lowest_drive_index;
      if (lowest < 0 || lowest >= static_cast<int>(characterization.filtered_buffers.size())) {
        return -1;
      }
      initial_cell = characterization.filtered_buffers[lowest].cell_index;
    }
    return findBufferTargetCell(liberty, characterization, initial_cell, downstream_load);
  }
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
      const double output_slew = rebufferCellOutputSlew(
          *cell, input_slew_rise, input_slew_fall, downstream_load);
      if (cell->max_slew > 0.0 && output_slew > cell->max_slew) {
        continue;
      }
      if (cell->max_cap > fallback_cap) {
        fallback_cap = cell->max_cap;
        fallback = static_cast<int>(cell - liberty.cells.data());
      }
      const double target_load = rebufferTargetLoadForCell(
          *cell, characterization.target_slew[0], characterization.target_slew[1],
          target_cell_math);
      const double distance = std::abs(downstream_load - target_load);
      const double delay = target_cell_math
          ? cellTargetSlewDelay(*cell, characterization.target_slew, downstream_load)
          : rebufferCellOutputDelay(
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

RepeaterCellSelection rebufferChooseRepeaterCell(const LibertyView& liberty,
                               const CharacterizationResult& characterization,
                               double input_slew_rise, double input_slew_fall,
                               double initial_load, double repeater_load,
                               const RepairConfig& config)
{
  if (!config.repair_design_target_cell_selection) {
    return {rebufferChooseBufferCell(liberty, characterization,
        input_slew_rise, input_slew_fall,
        config.repair_design_openroad_repeater_resize ? repeater_load : initial_load,
        config.repair_design_target_cell_math), -1};
  }
  // findTargetCell first starts from the lowest-drive buffer. The insertion
  // resize then starts from that selected cell, whose eligibility may differ.
  const int initial = rebufferChooseBufferCell(liberty, characterization,
      input_slew_rise, input_slew_fall, initial_load, true, true);
  if (initial < 0 || !config.repair_design_openroad_repeater_resize) {
    return {initial, initial};
  }
  return {rebufferChooseBufferCell(liberty, characterization,
      input_slew_rise, input_slew_fall, repeater_load, true, true, initial), initial};
}

}  // namespace rebuffer
