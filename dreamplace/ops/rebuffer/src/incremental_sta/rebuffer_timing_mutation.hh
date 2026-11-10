// SPDX-License-Identifier: BSD-3-Clause
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <vector>

#include "lut_query.h"

#include "pybind11/pybind11.h"

#include "rebuffer_incremental_timing.hh"


namespace rebuffer {

using ProgramCellArc = std::array<int64_t, 8>;
using ProgramNetArc = std::array<int64_t, 2>;

struct ProgramLut
{
  std::vector<double> values, transitions, capacitances;
  std::vector<int64_t> transition_dims, capacitance_dims;
  int rows = 0;
  int value_stride = 0, transition_stride = 0, capacitance_stride = 0;

  TimingLutTable<double> view() const;
};

struct ProgramNetGroup
{
  std::vector<ProgramNetArc> arcs;
  std::vector<int64_t> drivers;
};

enum class TimingEventKind { clock, cell, net };

struct TimingEventOperation
{
  TimingEventKind kind;
  int source, sink, arc;
  bool operator==(const TimingEventOperation& other) const
  {
    return kind == other.kind && source == other.source && sink == other.sink && arc == other.arc;
  }
};

struct TimingEventSchedule
{
  std::vector<std::vector<TimingEventOperation>> forward_inputs, reverse_outputs;
  std::vector<std::vector<int>> forward_successors, reverse_predecessors;
};

struct TimingPropagationSchedule
{
  std::vector<ProgramCellArc> cells;
  std::vector<int64_t> arc_ids;
  std::vector<int64_t> level_start;
  std::vector<std::vector<int64_t>> level_inputs;
  std::vector<ProgramNetGroup> forward_nets, reverse_nets;
  ProgramNetGroup endpoint_nets;
  TimingEventSchedule events;
};

// Immutable original constraints and Liberty input copied once from Python TP.
struct TimingPropagationProgram
{
  int pin_count = 0;
  TimingPropagationSchedule schedule;
  std::vector<int64_t> start_pins, endpoint_pins, metric_pins;
  std::vector<double> start_rAAT, start_fAAT, start_rtran, start_ftran;
  std::vector<double> endpoint_rRAT, endpoint_fRAT, clock_rtran, clock_ftran;
  std::vector<ProgramCellArc> setup_arcs, recovery_arcs;
  std::unique_ptr<bool[]> setup_valid, recovery_valid, endpoint_valid;
  std::vector<int64_t> setup_pins, recovery_r_pins, recovery_f_pins;
  std::vector<int64_t> timing_check_pins;
  std::vector<int> start_row_by_pin, endpoint_row_by_pin;
  std::vector<std::vector<int>> setup_rows_by_pin, recovery_rows_by_pin;
  std::vector<double> recovery_clock_rtran, recovery_clock_ftran;
  std::vector<double> recovery_reference_rtran, recovery_reference_ftran;
  bool recovery_uses_live_slew = true;
  bool recovery_rat_is_capture_deadline = false;
  std::vector<double> recovery_deadline_r, recovery_deadline_f;
  std::unique_ptr<bool[]> recovery_keep_setup;
  bool pairwise_smooth = false;
  double pairwise_alpha = 20.0, slew_epsilon = 1e-8;
  std::array<ProgramLut, 6> luts;
};

std::shared_ptr<const TimingPropagationProgram> timingProgramFromPython(
    const pybind11::dict& values, const TimingGraphIndex& graph);

}  // namespace rebuffer

// --- rebuffer_timing_events.hh ---
#include <map>
#include <set>
#include <string>
#include <vector>


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

// --- rebuffer_timing_mutation.hh ---
#include <array>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>


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
  std::vector<std::vector<int>> buffers_by_source_net;
  std::unordered_map<int, int> buffer_id_by_action_id;
  bool has_rebuffer_prefix_in_base_nodes = false;
  std::unordered_map<std::string, int> base_node_ids_by_name;
  std::unordered_map<std::string, int> cell_ids_by_name;
  std::vector<unsigned char> base_node_constraint_flags;
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

inline int sourceNetForMutationNet(const TimingMutationState& state, int net)
{
  const int base_nets = static_cast<int>(state.base_net_pins.size());
  if (net < base_nets) return net;
  const int idx = net - base_nets;
  if (idx >= 0 && idx < static_cast<int>(state.buffers.size())
      && state.buffers[idx].active && state.buffers[idx].output_net == net) {
    return state.buffers[idx].source_net;
  }
  for (const auto& buffer : state.buffers) {
    if (buffer.active && buffer.output_net == net) return buffer.source_net;
  }
  return -1;
}

}  // namespace rebuffer

// --- rebuffer_timing_buffer_state.hh ---
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
