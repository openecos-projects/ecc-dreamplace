// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "rebuffer_tree_option.hh"

namespace rebuffer {

std::vector<TreeOption> treeWireCandidates(
    const DesignView& design, const LibertyView& liberty,
    const CharacterizationResult& characterization, const RepairConfig& config,
    const CurrentDriverTree& tree, const DriverTreeEdge& edge, int depth,
    const std::vector<TreeOption>& children, TreeVgdpResult& result,
    int& trace_id, std::string& reason);

}  // namespace rebuffer
