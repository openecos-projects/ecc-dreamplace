// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <limits>
#include "rebuffer_tree_dp.hh"

namespace rebuffer {

// Electrical state stops at the next buffer input. Physical terminal loads
// remain in loads for projecting the complete action bundle into placeio_ecc.
struct TreeOption
{
  double cap = 0.0;
  double slack = 0.0;
  double slew_rise = 0.0;
  double slew_fall = 0.0;
  double area = 0.0;
  int buffer_count = 0;
  int node_id = -1;
  double max_load_slew = std::numeric_limits<double>::infinity();
  double max_load_wire = 0.0;
  int fanout = 0;
  std::vector<int> loads;
  std::vector<BufferAction> actions;
};

void pruneTreeOptions(std::vector<TreeOption>& options, int node_id,
                      TreeVgdpResult& result, int& next_trace_id);

}  // namespace rebuffer
