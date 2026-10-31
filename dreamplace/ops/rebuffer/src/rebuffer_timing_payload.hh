// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "pybind11/pybind11.h"

namespace rebuffer {

pybind11::tuple pinPairArcMapping(const pybind11::object& db);
pybind11::dict timingPinCapacitance(const pybind11::object& db);

}  // namespace rebuffer
