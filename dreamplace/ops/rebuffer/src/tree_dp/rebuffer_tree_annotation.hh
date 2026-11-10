// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "rebuffer_tree_option.hh"

namespace rebuffer {

struct TreeRootTiming
{
  double resistance = 0.0;
  double current_load = 0.0;
  double output_cap = 0.0;
};

void appendTreeLoads(std::vector<int>& target, const std::vector<int>& source);
void annotateTreeLoads(const DriverTreeNode& node, const TimingView& timing,
                       int driver, TreeOption& option);
TreeRootTiming annotateTreeRoot(const DesignView& design,
                                const TimingView& timing,
                                const LibertyView& liberty, int driver);
double treeRootCorrection(const TreeRootTiming& root, double cap);

}  // namespace rebuffer
