// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <vector>

#include "rebuffer_liberty.hh"
#include "rebuffer_repair.hh"
#include "rebuffer_types.hh"

namespace rebuffer {

// Return driver pins on one timing path in descending load-dependent delay.
// This is the native counterpart of OpenROAD SetupLegacyBase::rankPathDrivers.
std::vector<int> rankPathDriverPins(const DesignView& design,
                                    const TimingView& timing,
                                    const LibertyView& liberty,
                                    const std::vector<int>& path);

}  // namespace rebuffer
