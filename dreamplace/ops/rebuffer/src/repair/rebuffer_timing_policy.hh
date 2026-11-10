// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "pybind11/pybind11.h"

namespace rebuffer {

bool acceptsTimingCandidate(const pybind11::dict& current,
                            const pybind11::dict& candidate,
                            const pybind11::dict& request,
                            const pybind11::dict& config);

}  // namespace rebuffer
