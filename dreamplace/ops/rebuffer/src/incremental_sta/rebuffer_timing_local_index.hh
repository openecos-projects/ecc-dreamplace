// SPDX-License-Identifier: BSD-3-Clause
// Local adjacency, DAG ranks and per-pin event rows share one reversible delta.
#pragma once

#include <map>
#include <memory>
#include <vector>

#include "rebuffer_timing_program.hh"

namespace rebuffer {
struct TimingMutationState;
struct TimingTopologyUndo;

struct TimingIndexPinUndo
{
  int rank, path_level, net_arc;
  std::vector<int> successors, cell_out, cell_in, net_out, net_in;
  std::vector<TimingEventOperation> forward_inputs, reverse_outputs;
  std::vector<int> forward_successors, reverse_predecessors;
};

struct TimingIndexCellUndo
{
  int id;
  TimingCellArc row;
  unsigned char active;
};

struct TimingIndexNetUndo
{
  int id;
  TimingNetArc row;
  unsigned char active;
};

struct TimingIndexUndo
{
  std::size_t pins, cells, nets;
  bool canonical_schedule;
  std::map<int, TimingIndexPinUndo> rows;
  std::vector<TimingIndexCellUndo> cell_rows;
  std::vector<TimingIndexNetUndo> net_rows;
};

std::shared_ptr<TimingIndexUndo> captureTimingIndex(
    const TimingMutationState& state, const TimingGraphIndex& index);
void captureTimingIndexPin(TimingIndexUndo& undo, const TimingMutationState& state,
                            const TimingGraphIndex& index, int pin);
void initializeTimingLocalIndex(TimingMutationState& state, TimingGraphIndex& index,
                                 const TimingPropagationProgram& program);
void updateTimingLocalIndex(TimingMutationState& state, TimingGraphIndex& index,
    const TimingPropagationProgram& program, const TimingTopologyUndo& topology_undo);
void restoreTimingLocalIndex(TimingMutationState& state, TimingGraphIndex& index,
                              TimingIndexUndo& undo);
void updateTimingEventRows(TimingMutationState& state, const TimingGraphIndex& index,
    const TimingPropagationProgram& program, const std::vector<int>& pins,
    TimingIndexUndo& undo);
}  // namespace rebuffer
