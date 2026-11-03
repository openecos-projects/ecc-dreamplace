// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <string_view>
#include <vector>

#include "pybind11/pybind11.h"

namespace rebuffer {
struct ElectricalPinObservation
{
  std::string_view name;
  double slew_rise, slew_fall, load_rise, load_fall, max_slew, max_cap;
  int fanout;
  bool driver, clock;
};

pybind11::dict observeElectricalPins(const std::vector<ElectricalPinObservation>& pins,
                                    int max_fanout, double cap_margin, double slew_margin);
bool electricalNonRegressing(const pybind11::dict& before, const pybind11::dict& after);
pybind11::dict compareElectricalObservations(const pybind11::dict& before,
                                            const pybind11::dict& after);
void registerElectricalObservationBindings(pybind11::module_& module);
}
