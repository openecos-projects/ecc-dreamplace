// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "rebuffer_tree_annotation.hh"

namespace rebuffer {

double treeMarginedLimit(double limit, double margin);
// Fixed RC/drive models only. Nonlinear timing is evaluated by Python TP
// after the complete candidate has been applied to the overlay.
bool treeCanDrive(const TreeOption& option, double resistance,
                  double output_cap, double max_cap, const RepairConfig& config);
bool treeRootFeasible(const TreeOption& option, const TreeRootTiming& root,
                      const TimingView& timing, int driver,
                      const RepairConfig& config);
double treeWireStep(const LibertyView& liberty,
                    const CharacterizationResult& characterization,
                    const RepairConfig& config);

}  // namespace rebuffer
