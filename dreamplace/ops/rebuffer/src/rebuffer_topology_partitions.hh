// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <unordered_map>
#include <vector>

#include "rebuffer_incremental_topology.hh"

namespace rebuffer {

struct ElectricalPartitions
{
  std::unordered_map<int, int> node_partition;
  std::vector<std::vector<int>> loads;
  std::vector<int> input_partition;
};

ElectricalPartitions topologyPartitions(const IncrementalTopologyState& state);

}  // namespace rebuffer
