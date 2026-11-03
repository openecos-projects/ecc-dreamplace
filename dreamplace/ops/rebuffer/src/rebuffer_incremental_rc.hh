// SPDX-License-Identifier: BSD-3-Clause
// Compact RC storage and the shared Elmore/second-moment evaluator.
#pragma once

#include <array>
#include <vector>

namespace rebuffer {
struct IncrementalTopologyState;

struct CompactRcForest
{
  // External identities are labels only; every other index is local to this forest.
  std::vector<int> node_ids, parent, child_start, children, topo_start, topo;
  std::vector<double> resistance, edge_capacitance;
};

CompactRcForest compactRcForest(const IncrementalTopologyState& topology,
                                int original_pin_count);
std::vector<double> compactRcWireCaps(const CompactRcForest& forest);
// Input caps include the wire half-caps. Results: subtree load, delay, impulse.
std::array<std::vector<double>, 3> evaluateCompactRc(
    const CompactRcForest& forest, const std::vector<double>& caps);
}  // namespace rebuffer
