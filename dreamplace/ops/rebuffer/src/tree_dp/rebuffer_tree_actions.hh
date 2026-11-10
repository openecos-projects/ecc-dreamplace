// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include "rebuffer_driver_tree.hh"

namespace rebuffer {
std::vector<pybind11::dict> orderTreeBufferActions(const pybind11::list& actions);
}  // namespace rebuffer
