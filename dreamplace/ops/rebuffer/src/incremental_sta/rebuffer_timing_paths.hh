// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <cstddef>
#include "rebuffer_incremental_timing.hh"
#include "timing_setup_paths.h"

namespace rebuffer {

struct TimingSetupPaths
{
  dreamplace::timing::SetupPathBatch batch;
  std::size_t endpoint_record_count = 0;
};

struct TimingPathPredecessors
{
  const TimingGraphIndex& index;
  bool reordered;

  template <typename Visit>
  void operator()(int64_t pin, const Visit& visit) const
  {
    auto order = [&](int id) {
      return std::pair<int64_t, int64_t>{reordered ? index.path_level[index.source(id)] : 0, id};
    };
    auto cell = [&](int id) {
      if (id >= index.clock_end) {
        visit(dreamplace::timing::SetupPathPredecessor{
            index.source(id), id, index.cells[id][4], order(id)});
      }
    };
    const auto& incoming = index.cell_in[pin];
    if (reordered && incoming.size() > 1) {
      auto sorted = incoming;
      std::sort(sorted.begin(), sorted.end(), [&](int a, int b) { return order(a) < order(b); });
      for (int id : sorted) cell(id);
    } else {
      for (int id : incoming) cell(id);
    }
    for (int id : index.net_in[pin]) {
      visit(dreamplace::timing::SetupPathPredecessor{index.nets[id][0], -1, 0, {-1, -1}});
    }
  }
};

}  // namespace rebuffer
