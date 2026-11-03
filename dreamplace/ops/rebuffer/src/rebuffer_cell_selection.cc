// SPDX-License-Identifier: BSD-3-Clause
// OpenROAD target-load and driver-slew selection for scalar buffer cells.
#include "rebuffer_cell_selection.hh"
#include "rebuffer_library_metadata.hh"
#include "rebuffer_python_views.hh"
#include "rebuffer_repair.hh"

#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <utility>

#include "pybind11/stl.h"

namespace py = pybind11;
namespace rebuffer {

void initializeBufferCellSelection(LibertyView& liberty)
{
  if (!liberty.cell_metadata) return;
  const auto& records = liberty.cell_metadata->cells;
  std::map<std::string, std::vector<int>> equivalent;
  if (records.size() != liberty.cells.size()) throw std::invalid_argument("repair library metadata cell count differs from Liberty view");
  for (std::size_t i = 0; i < records.size(); ++i) {
    if (records[i].name != liberty.cells[i].name) throw std::invalid_argument("repair library metadata cell identity differs from Liberty view");
    if (!records[i].buffer_signature.empty() && !records[i].dont_use) {
      equivalent[records[i].buffer_signature].push_back(i);
    }
  }
  for (auto& [_, order] : equivalent) {
    std::sort(order.begin(), order.end(), [&](int a, int b) { return records[a].source_order < records[b].source_order; });
    std::sort(order.begin(), order.end(), [&](int a, int b) {
      return liberty.cells[a].drive_resistance > liberty.cells[b].drive_resistance;
    });
    liberty.buffer_equivalent_order.insert(liberty.buffer_equivalent_order.end(), order.begin(), order.end());
  }
}

std::vector<int> swappableBufferCells(const LibertyView& liberty, int source)
{
  if (!liberty.cell_metadata) throw std::invalid_argument("repair cell selection requires source Library/LEF metadata");
  const auto& metadata = liberty.cell_metadata->cells;
  const auto& current = metadata.at(source);
  if (current.buffer_signature.empty()) throw std::invalid_argument("target-cell source is outside the scalar buffer domain");
  if (!current.has_master || !current.master_is_core) return {};
  std::vector<int> equivalent;
  for (int candidate : liberty.buffer_equivalent_order) {
    if (metadata[candidate].buffer_signature == current.buffer_signature) equivalent.push_back(candidate);
  }
  // OpenSTA has no equivalence-set entry for a singleton or a dont-use
  // source. Resizer then returns the source without applying set filters.
  if (current.dont_use || equivalent.size() < 2) return {source};
  std::vector<int> result;
  const double source_area = liberty.cells[source].area;
  for (int candidate : equivalent) {
    const auto& target = metadata[candidate];
    if (!target.has_master) continue;
    if (source_area != 0 && liberty.cells[candidate].area / source_area > 4.0) continue;
    if (current.has_leakage && target.has_leakage && target.leakage / current.leakage > 4.0) continue;
    if (!current.function_class.empty() && current.function_class != target.function_class) continue;
    result.push_back(candidate);
  }
  return result;
}

int findBufferTargetCell(const LibertyView& liberty,
                         const CharacterizationResult& characterization,
                         int source, double load)
{
  if (!std::isfinite(load) || load < 0) throw std::invalid_argument("invalid target-cell load");
  const auto& cells = liberty.cells;
  int best = source;
  double distance = std::abs(load - cellTargetLoad(cells.at(source), characterization.target_slew));
  double delay = cellTargetSlewDelay(cells[source], characterization.target_slew, load);
  for (int candidate : swappableBufferCells(liberty, source)) {
    const double candidate_distance = std::abs(load - cellTargetLoad(cells[candidate], characterization.target_slew));
    const double candidate_delay = cellTargetSlewDelay(cells[candidate], characterization.target_slew, load);
    if ((candidate_delay < delay && candidate_distance < distance * 1.1)
        || (candidate_distance < distance && candidate_delay < delay * 1.1)) {
      best = candidate;
      distance = candidate_distance;
      delay = candidate_delay;
    }
  }
  return best;
}

int findBufferSlewRepairCell(const LibertyView& liberty,
                             int source,
                             double input_slew_rise,
                             double input_slew_fall,
                             double load,
                             double slew_margin)
{
  std::vector<std::pair<double, int>> sizes;
  for (int candidate : swappableBufferCells(liberty, source)) {
    const auto& cell = liberty.cells[candidate];
    const double slew = rebufferCellOutputSlew(
        cell, input_slew_rise, input_slew_fall, load);
    const double violation = cell.max_slew > 0.0
        ? std::max(0.0, slew - cell.max_slew * (1.0 - slew_margin))
        : 0.0;
    sizes.emplace_back(violation, candidate);
  }
  // RepairDesign sorts the ordered swappable sequence by violation, or by
  // area when both cells meet slew. Equal scores have no extra id tie-break.
  std::sort(sizes.begin(), sizes.end(), [&](const auto& a, const auto& b) {
    if (a.first == 0.0 && b.first == 0.0) {
      return liberty.cells[a.second].area < liberty.cells[b.second].area;
    }
    return a.first < b.first;
  });
  return sizes.empty() ? source : sizes.front().second;
}

void registerCellSelectionBindings(py::module_& module)
{
  module.def("validate_buffer_resize_pairs", [](const py::dict& payload,
                                                const std::vector<int>& sources,
                                                const std::vector<int>& targets) {
    if (sources.size() != targets.size()) {
      throw std::invalid_argument("buffer resize source/target counts differ");
    }
    const auto liberty = rebuffer_python::libertyFromDict(payload);
    for (std::size_t i = 0; i < sources.size(); ++i) {
      if (sources[i] < 0 || sources[i] >= static_cast<int>(liberty.cells.size())
          || targets[i] < 0 || targets[i] >= static_cast<int>(liberty.cells.size())) {
        throw std::invalid_argument("buffer resize cell index is outside Liberty");
      }
      const auto candidates = swappableBufferCells(liberty, sources[i]);
      if (std::find(candidates.begin(), candidates.end(), targets[i]) == candidates.end()) {
        throw std::invalid_argument("target is not a swappable buffer: "
            + liberty.cells[sources[i]].name + " -> " + liberty.cells[targets[i]].name);
      }
    }
  });
  module.def("repair_buffer_cell_candidates", [](const py::dict& payload, int source) {
    const auto liberty = rebuffer_python::libertyFromDict(payload);
    return swappableBufferCells(liberty, source);
  });
  module.def("repair_buffer_target_cell", [](const py::dict& payload, const py::dict& characterization,
                                              int source, double load) {
    const auto liberty = rebuffer_python::libertyFromDict(payload);
    return findBufferTargetCell(liberty, rebuffer_python::characterizationFromDict(characterization), source, load);
  });
}

}  // namespace rebuffer
