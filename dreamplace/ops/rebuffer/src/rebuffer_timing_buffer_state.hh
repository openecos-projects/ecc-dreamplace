// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include "rebuffer_timing_mutation.hh"
#include "rebuffer_incremental_topology.hh"

namespace rebuffer {

enum class TimingTopologyOperation { insert, remove, resize };
struct TimingIndexUndo;

struct TimingPinUndo
{
  int id, net;
  unsigned char active;
  std::array<double, 5> values;
};

struct TimingNetUndo
{
  int id, driver;
  std::vector<int> pins;
};

struct TimingArcUndo
{
  int id;
  ProgramCellArc row;
  unsigned char active;
};

struct TimingTopologyUndo
{
  std::size_t pins, nodes, nets, arcs, buffers;
  std::vector<TimingPinUndo> previous_pins;
  std::vector<TimingNetUndo> previous_nets;
  std::vector<TimingArcUndo> previous_arcs;
  std::vector<std::pair<int, int>> previous_cells;
  std::vector<std::pair<int, TimingBufferRecord>> previous_buffers;
  std::shared_ptr<TimingIndexUndo> index_undo;
  int source_net = -1, previous_topology_epoch = -1;
};

std::shared_ptr<TimingTopologyUndo> captureTimingTopology(
    const TimingMutationState& state, const TimingGraphIndex& index, int source_net);
void restoreTimingTopology(TimingMutationState& state, TimingGraphIndex& index,
                           const TimingTopologyUndo& undo);
void synchronizeTimingBuffers(TimingMutationState& state, TimingGraphIndex& index,
                               const TimingPropagationProgram& program,
                               const IncrementalTopologyState& topology);
void requireSynchronizedTimingTopology(const TimingMutationState& state,
                                       const IncrementalTopologyState& topology);
void requireSynchronizedTimingTopologies(
    const TimingMutationState& state,
    const std::map<int, std::shared_ptr<IncrementalTopologyState>>& topologies);

struct TimingNetRcValues
{
  std::vector<int> pins;
  // rise/fall subtree cap, net delay, impulse
  std::array<std::vector<double>, 6> values;
};

}  // namespace rebuffer
