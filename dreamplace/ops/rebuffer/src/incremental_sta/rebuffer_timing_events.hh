// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <map>
#include <set>
#include <string>
#include <vector>

#include "rebuffer_timing_program.hh"

namespace rebuffer {

struct TimingMutationState;
struct TimingStateUndo;

struct TimingEventSeeds
{
  std::set<int> forward, reverse, retired_arcs;
  bool require_full = false;
};

struct TimingEventResult
{
  std::map<std::string, std::size_t> coverage;
  std::vector<int64_t> changed_pins;
};

void buildTimingEventSchedule(TimingPropagationSchedule& schedule,
                              const TimingGraphIndex& index);

TimingEventResult propagateTimingEvents(
    const TimingPropagationProgram& program, const TimingGraphIndex& index,
    const TimingMutationState* mutation,
    std::map<std::string, std::vector<double>>& state,
    const TimingEventSeeds& seeds, TimingStateUndo& undo);

}  // namespace rebuffer
