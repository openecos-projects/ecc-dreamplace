// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <map>

#include "rebuffer_tree_dp.hh"

namespace rebuffer {

// OpenROAD-style local tree rounds. Each round consumes the previous modeled
// tree; the outer queue evaluates the final complete bundle with Python TP.
bool optimizeCurrentDriverTree(
    const DesignView& design, const TimingView& timing, const LibertyView& liberty,
    const CharacterizationResult& characterization, const RepairConfig& config,
    const CurrentDriverTree& source, TreeVgdpResult& result, std::string& reason,
    std::map<std::string, double>* phase_timings = nullptr,
    bool record_trace = true,
    bool prune_worse_than_unbuffered = false);

}  // namespace rebuffer
