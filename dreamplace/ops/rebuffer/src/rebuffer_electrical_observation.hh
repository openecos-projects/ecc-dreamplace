// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "pybind11/pybind11.h"

namespace rebuffer {
bool electricalNonRegressing(const pybind11::dict& before, const pybind11::dict& after);
pybind11::dict compareElectricalObservations(const pybind11::dict& before,
                                            const pybind11::dict& after);
void registerElectricalObservationBindings(pybind11::module_& module);
}
