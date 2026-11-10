// SPDX-License-Identifier: BSD-3-Clause
// Native outer repair_timing queue for placement-RC buffering.

#pragma once

#include "pybind11/pybind11.h"

namespace rebuffer {

// The snapshot callback exposes only timing data and native path records.  It
// does not select an endpoint or a net.  The C++ operator sorts and walks the
// queue, asks the bridge for a candidate for the selected net, and owns the
// complete accept/restore lifecycle.
pybind11::dict runRepairTimingQueueOperator(
    const pybind11::object& timing_snapshot,
    const pybind11::object& candidate_provider,
    const pybind11::object& checkpoint,
    const pybind11::object& apply,
    const pybind11::object& evaluate,
    const pybind11::object& restore,
    const pybind11::dict& config,
    const pybind11::object& release_checkpoint = pybind11::none());

}  // namespace rebuffer
