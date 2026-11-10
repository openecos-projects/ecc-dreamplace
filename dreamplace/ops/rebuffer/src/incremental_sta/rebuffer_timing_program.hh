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
