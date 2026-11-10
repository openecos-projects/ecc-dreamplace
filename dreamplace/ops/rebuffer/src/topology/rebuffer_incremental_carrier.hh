// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "pybind11/pybind11.h"

namespace rebuffer {

// Rebind one mutable overlay carrier from the preceding carrier snapshot.
// This does not run STA or rebuild Steiner; it only transports graph arrays.
pybind11::dict rebuildTimingCarrierIncremental(
    const pybind11::object& base,
    const pybind11::object& previous,
    const pybind11::object& physical);

}  // namespace rebuffer
