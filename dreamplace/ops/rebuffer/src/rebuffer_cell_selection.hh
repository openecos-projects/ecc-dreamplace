// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "rebuffer_liberty.hh"

namespace pybind11 { class module_; }

namespace rebuffer {

void initializeBufferCellSelection(LibertyView& liberty);
std::vector<int> swappableBufferCells(const LibertyView& liberty,
                                      int source,
                                      double sizing_area_limit = 4.0,
                                      double sizing_leakage_limit = 4.0);
int findBufferTargetCell(const LibertyView& liberty,
                         const CharacterizationResult& characterization,
                         int source,
                         double load,
                         double sizing_area_limit = 4.0,
                         double sizing_leakage_limit = 4.0);
int findBufferSlewRepairCell(const LibertyView& liberty,
                             int source,
                             double input_slew_rise,
                             double input_slew_fall,
                             double load,
                             double slew_margin,
                             double sizing_area_limit = 4.0,
                             double sizing_leakage_limit = 4.0);
void registerCellSelectionBindings(pybind11::module_& module);

}  // namespace rebuffer
