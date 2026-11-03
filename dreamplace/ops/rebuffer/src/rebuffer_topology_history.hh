// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "rebuffer_incremental_topology.hh"

namespace rebuffer {

void invalidateIncrementalTopologyCaches(IncrementalTopologyState& state);
bool undoLastTopologySplit(IncrementalTopologyState& state,
                           const IncrementalBuffer& buffer, std::string& reason);
bool redoTopologySplit(IncrementalTopologyState& state,
                       const IncrementalBuffer& buffer, std::string& reason);

}  // namespace rebuffer
