// SPDX-License-Identifier: BSD-3-Clause
// Native repair_timing candidate dispatch over Python data payloads.

#pragma once

#include "pybind11/pybind11.h"

namespace rebuffer {

// ``timing_snapshot`` returns both queue items and a
// per-epoch ``planner_payloads`` list.  C++ looks up the payload for each
// request and invokes the native planner directly; there is no Python
// candidate/payload callback in this path.
pybind11::dict runRepairTimingNativeSnapshotOperator(
    const pybind11::object& timing_snapshot,
    const pybind11::object& checkpoint,
    const pybind11::object& apply,
    const pybind11::object& evaluate,
    const pybind11::object& restore,
    const pybind11::dict& config);

}  // namespace rebuffer
