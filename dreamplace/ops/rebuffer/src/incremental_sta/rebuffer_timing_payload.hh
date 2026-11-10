// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace rebuffer {

pybind11::tuple pinPairArcMapping(const pybind11::object& db);
pybind11::dict timingPinCapacitance(const pybind11::object& db);

using TimingPayloadArray = pybind11::array_t<
    double, pybind11::array::c_style | pybind11::array::forcecast>;
pybind11::dict repairDesignElectricalPayload(
    const TimingPayloadArray& rise_load, const TimingPayloadArray& fall_load,
    const TimingPayloadArray& rise_cap, const TimingPayloadArray& fall_cap);

}  // namespace rebuffer
