// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "pybind11/pybind11.h"

namespace rebuffer {
void registerVerifiedCommitBindings(pybind11::module_& module);
}
