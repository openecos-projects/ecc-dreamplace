// SPDX-License-Identifier: BSD-3-Clause
// Full-seed sweep on the immutable carrier graph, using shared timing math.
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_program.hh"
#include "rebuffer_timing_mutation.hh"
#include "rebuffer_timing_buffer_state.hh"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

#include "net_aat_math.h"
#include "timing_hard_math.h"
#include "pybind11/stl.h"

namespace py = pybind11;
namespace rebuffer {
namespace {

template <std::size_t Width>
const int64_t* flat(const std::vector<std::array<int64_t, Width>>& rows)
{
  return rows.empty() ? nullptr : rows.front().data();
}

}  // namespace

py::dict PersistentTimingGraph::initializePropagation(const py::dict& values)
{
  if (propagation_program_ != nullptr) {
    throw py::value_error("propagation inputs are immutable; create a new graph to reconfigure");
  }
  // Parse into a separate owner: malformed inputs cannot change graph state.
  propagation_program_ = timingProgramFromPython(values, *timing_index_);
  return summary();
}

py::dict PersistentTimingGraph::propagate()
{
  if (!timing_state_initialized_ || propagation_program_ == nullptr) {
    throw py::value_error("propagation requires initialized timing state and inputs");
  }
  if (!net_states_.empty()) {
    if (mutation_state_ == nullptr) {
      throw py::value_error("full-seed propagation requires synchronized buffer timing records");
    }
    requireSynchronizedTimingTopologies(*mutation_state_, net_states_);
  }
  const auto& p = *propagation_program_;
  std::shared_ptr<TimingPropagationSchedule> generated;
  if (mutation_state_ && mutation_state_->canonical_schedule) {
    const auto started = std::chrono::steady_clock::now();
    TimingGraphIndex rebuilt;
    rebuilt.pins = timing_index_->pins; rebuilt.clocks = timing_index_->clocks;
    rebuilt.clock_end = timing_index_->clock_end;
    buildTimingMutationIndex(*mutation_state_, rebuilt);
    const auto indexed = std::chrono::steady_clock::now();
    ++mutation_state_->statistics.index_builds;
    mutation_state_->statistics.index_ms += std::chrono::duration<double, std::milli>(indexed - started).count();
    generated = makeTimingMutationSchedule(*mutation_state_, rebuilt, p);
    ++mutation_state_->statistics.schedule_builds;
    mutation_state_->statistics.schedule_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - indexed).count();
  }
  const auto& schedule = generated ? *generated : p.schedule;
  const auto& cells = mutation_state_ != nullptr && !generated
                          ? mutation_state_->cells : schedule.cells;
  const int pins = timing_index_->pins.size();
  const auto arcs = timing_index_->cells.size();
  std::map<std::string, std::vector<double>> output;
  auto& ra = output["pin_rAAT"] = std::vector<double>(pins, -2e8);
  auto& fa = output["pin_fAAT"] = std::vector<double>(pins, -2e8);
  auto& rs = output["pin_rtran"] = std::vector<double>(pins, 0);
  auto& fs = output["pin_ftran"] = std::vector<double>(pins, 0);
  auto& rr = output["pin_rRAT"] = std::vector<double>(pins, 2e8);
  auto& fr = output["pin_fRAT"] = std::vector<double>(pins, 2e8);
  auto& drr = output["cell_arc_rr_delays"] = std::vector<double>(arcs, 0);
  auto& dfr = output["cell_arc_fr_delays"] = std::vector<double>(arcs, 0);
  auto& drf = output["cell_arc_rf_delays"] = std::vector<double>(arcs, 0);
  auto& dff = output["cell_arc_ff_delays"] = std::vector<double>(arcs, 0);
  const auto& cap_r = state_.at("pin_net_cap_rise");
  const auto& cap_f = state_.at("pin_net_cap_fall");
  const auto& delay_r = state_.at("pin_net_delay_rise");
  const auto& delay_f = state_.at("pin_net_delay_fall");
  const auto& impulse_r = state_.at("pin_net_impulse_rise");
  const auto& impulse_f = state_.at("pin_net_impulse_fall");
  const auto rd = p.luts[0].view(), fd = p.luts[1].view();
  const auto rt = p.luts[2].view(), ft = p.luts[3].view();
  for (std::size_t i = 0; i < p.start_pins.size(); ++i) {
    const auto id = p.start_pins[i];
    ra[id] = p.start_rAAT[i]; fa[id] = p.start_fAAT[i];
    rs[id] = p.start_rtran[i]; fs[id] = p.start_ftran[i];
  }
  for (std::size_t i = 0; i < p.endpoint_pins.size(); ++i) {
    const auto id = p.endpoint_pins[i];
    rr[id] = p.endpoint_rRAT[i]; fr[id] = p.endpoint_fRAT[i];
  }

  dreamplace::timing::clk2qRows(
      flat(cells), schedule.level_start[1], 8, pins, cap_r.data(), cap_f.data(),
      p.clock_rtran.data(), p.clock_ftran.data(), rd, fd, rt, ft,
      p.pairwise_smooth, p.pairwise_alpha, ra.data(), fa.data(), rs.data(), fs.data());
  // Clock slots deliberately retain zero delays, matching the Python cache.
  std::vector<double> updates;
  std::size_t net_forward_rows = 0, net_reverse_rows = 0;
  auto forward_net = [&](const ProgramNetGroup& group) {
    updates.resize(group.arcs.size() * 4);
    for (std::size_t i = 0; i < group.arcs.size(); ++i) {
      double jacobian[6];
      netForwardRow(0, group.arcs[i].data(), 2, 1, pins,
                    ra.data(), fa.data(), rs.data(), fs.data(),
                    delay_r.data(), delay_f.data(), impulse_r.data(), impulse_f.data(),
                    updates.data() + i * 4, jacobian, p.slew_epsilon);
    }
    dreamplace::timing::aggregateNetRows(flat(group.arcs), updates.data(),
        group.arcs.size(), 2, ra.data(), fa.data(), rs.data(), fs.data());
    net_forward_rows += group.arcs.size();
  };
  forward_net(schedule.forward_nets[0]);
  for (std::size_t level = 1; level + 1 < schedule.level_start.size(); ++level) {
    const auto first = schedule.level_start[level], last = schedule.level_start[level + 1];
    updates.resize((last - first) * 8);
    for (auto i = first; i < last; ++i) {
      double jacobian[48];
      auto* row = updates.data() + (i - first) * 8;
      cellForwardRow(0, cells[i].data(), 8, 1, pins, ra.data(), fa.data(),
                     rs.data(), fs.data(), cap_r.data(), cap_f.data(), rd, fd, rt, ft,
                     row, jacobian, p.pairwise_smooth, p.pairwise_alpha);
      const auto slot = schedule.arc_ids[i];
      drr[slot] = row[4]; dfr[slot] = row[5]; drf[slot] = row[6]; dff[slot] = row[7];
    }
    if (first != last) {
      dreamplace::timing::aggregateCellRows(cells[first].data(), updates.data(),
          last - first, 8, ra.data(), fa.data(), rs.data(), fs.data());
    }
    forward_net(schedule.forward_nets[level]);
  }

  // Keep each batch's RAT input unchanged. Several rows can write one pin,
  // and self net arcs can read the same pin that they update.
  std::vector<double> scratch_r(pins), scratch_f(pins);
  auto prepare = [&](const std::vector<int64_t>& touched) {
    for (auto id : touched) { scratch_r[id] = rr[id]; scratch_f[id] = fr[id]; }
  };
  auto finish = [&](const std::vector<int64_t>& touched) {
    for (auto id : touched) { rr[id] = scratch_r[id]; fr[id] = scratch_f[id]; }
  };
  prepare(p.setup_pins);
  dreamplace::timing::setupRatRows(flat(p.setup_arcs), p.setup_arcs.size(), 8,
      rr.data(), fr.data(), p.clock_rtran.data(), p.clock_ftran.data(), rs.data(),
      fs.data(), rd, fd, p.setup_valid.get(), p.pairwise_smooth, p.pairwise_alpha,
      scratch_r.data(), scratch_f.data());
  finish(p.setup_pins);

  auto reference_r = p.recovery_reference_rtran;
  auto reference_f = p.recovery_reference_ftran;
  if (p.recovery_uses_live_slew) {
    for (std::size_t i = 0; i < p.recovery_arcs.size(); ++i) {
      reference_r[i] = rs[p.recovery_arcs[i][1]];
      reference_f[i] = fs[p.recovery_arcs[i][1]];
    }
  }
  dreamplace::timing::recoveryRatRows(flat(p.recovery_arcs), p.recovery_arcs.size(),
      8, pins, rr.data(), fr.data(), rs.data(), fs.data(),
      p.recovery_clock_rtran.data(), p.recovery_clock_ftran.data(),
      reference_r.data(), reference_f.data(), p.luts[4].view(), p.luts[5].view(),
      p.recovery_valid.get(), scratch_r.data(), scratch_f.data(),
      p.recovery_rat_is_capture_deadline, p.recovery_deadline_r.data(),
      p.recovery_deadline_f.data(), p.recovery_keep_setup.get());
  for (auto id : p.recovery_r_pins) rr[id] = scratch_r[id];
  for (auto id : p.recovery_f_pins) fr[id] = scratch_f[id];

  auto reverse_net = [&](const ProgramNetGroup& group) {
    prepare(group.drivers);
    dreamplace::timing::netRatRows(flat(group.arcs), group.arcs.size(), 2,
        rr.data(), fr.data(), delay_r.data(), delay_f.data(),
        scratch_r.data(), scratch_f.data());
    finish(group.drivers);
    net_reverse_rows += group.arcs.size();
  };
  reverse_net(schedule.endpoint_nets);
  for (int level = static_cast<int>(schedule.level_start.size()) - 2; level >= 1; --level) {
    const auto first = schedule.level_start[level], last = schedule.level_start[level + 1];
    prepare(schedule.level_inputs[level]);
    if (first != last) {
      for (auto row = first; row < last; ++row) {
        const auto slot = schedule.arc_ids[row];
        dreamplace::timing::cellRatRows(cells[row].data(), 1, 8,
            rr.data(), fr.data(), drr.data() + slot, dfr.data() + slot,
            drf.data() + slot, dff.data() + slot, scratch_r.data(), scratch_f.data());
      }
    }
    finish(schedule.level_inputs[level]);
    reverse_net(schedule.reverse_nets[level]);
  }

  py::dict coverage;
  coverage["clk2q_rows"] = schedule.level_start[1];
  coverage["cell_forward_rows"] = cells.size() - schedule.level_start[1];
  coverage["cell_reverse_rows"] = cells.size() - schedule.level_start[1];
  coverage["net_forward_rows"] = net_forward_rows;
  coverage["net_reverse_rows"] = net_reverse_rows;
  coverage["setup_rows"] = p.setup_arcs.size();
  coverage["recovery_rows"] = p.recovery_arcs.size();
  return finishPropagation(std::move(output), std::move(coverage), "full_seed");
}

}  // namespace rebuffer
