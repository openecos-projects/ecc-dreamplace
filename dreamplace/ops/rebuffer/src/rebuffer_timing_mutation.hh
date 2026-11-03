// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "rebuffer_arc_identity.hh"
#include "rebuffer_timing_program.hh"

namespace rebuffer {

struct TimingBufferRecord
{
  int source_net = -1, action_id = -1;
  int node = -1, input_pin = -1, output_pin = -1, output_net = -1;
  bool active = true;
  int logical_input = -1, logical_output = -1;
  double split_ratio = 0;
  std::vector<int> arcs;
};

struct TimingMutationState
{
  struct Statistics
  {
    std::size_t index_builds = 0, schedule_builds = 0, rc_builds = 0;
    std::size_t rc_nodes = 0, rc_pin_reads = 0;
    std::size_t local_updates = 0, rank_visits = 0, event_rows = 0;
    double index_ms = 0, schedule_ms = 0, rc_build_ms = 0, rc_evaluate_ms = 0;
    double local_index_ms = 0, local_events_ms = 0;
  } statistics;
  int original_node_count = 0;
  std::vector<ProgramCellArc> cells;
  std::vector<unsigned char> cell_active, pin_active;
  std::vector<TimingBufferRecord> buffers;
  std::vector<std::vector<int>> base_net_pins;
  std::vector<unsigned char> base_net_is_clock;
  std::vector<std::string> net_names;
  TimingEventSchedule events;
  std::vector<int> net_arc_by_pin;
  bool canonical_schedule = false;
  std::map<int, int> topology_epochs;
  std::vector<std::string> node_names, cell_names, pin_ports, library_pin_names;
  std::vector<int> node_cells, cell_mains, cell_arc_start, cell_pin_start;
  std::vector<int> pin_owner, pin_net, net_drivers;
  std::vector<std::vector<int>> node_pins, node_arcs, net_pins;
  std::vector<TimingLibraryArc> library_arcs;
  // generic/rise/fall pin capacitance, max slew, max capacitance
  std::array<std::vector<double>, 5> pin_values, library_pin_values;
};

void rebuildTimingMutationIndex(TimingMutationState& state, TimingGraphIndex& index,
                                const TimingPropagationProgram& program);
void buildTimingMutationIndex(const TimingMutationState& state, TimingGraphIndex& index);
std::shared_ptr<TimingPropagationSchedule> makeTimingMutationSchedule(
    const TimingMutationState& state, const TimingGraphIndex& index,
    const TimingPropagationProgram& program);

struct TimingResizeUndo
{
  int node = -1;
  int previous_cell = -1;
  std::vector<std::pair<int, ProgramCellArc>> arcs;
  std::vector<std::pair<int, std::array<double, 5>>> pins;
};

void restoreTimingResize(TimingMutationState& state, TimingGraphIndex& index,
                         const TimingResizeUndo& undo);

}  // namespace rebuffer
