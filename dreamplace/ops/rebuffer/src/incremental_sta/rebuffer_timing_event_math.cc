// SPDX-License-Identifier: BSD-3-Clause
// Native changed-value worklists over the canonical Python TP equations.
#include "rebuffer_timing_events.hh"
#include "rebuffer_timing_mutation.hh"
#include "rebuffer_timing_state.hh"

#include <array>
#include <cmath>
#include <cstring>
#include <functional>
#include <queue>
#include <utility>

#include "net_aat_math.h"
#include "timing_hard_math.h"

namespace rebuffer {
namespace {

bool changed(double a, double b)
{
  return std::isnan(a) || std::isnan(b) || std::memcmp(&a, &b, sizeof(double));
}

template <std::size_t N>
bool changed(const std::array<double, N>& a, const std::array<double, N>& b)
{
  for (std::size_t i = 0; i < N; ++i) if (changed(a[i], b[i])) return true;
  return false;
}

}  // namespace

TimingEventResult propagateTimingEvents(
    const TimingPropagationProgram& p, const TimingGraphIndex& index,
    const TimingMutationState* mutation,
    std::map<std::string, std::vector<double>>& output,
    const TimingEventSeeds& seeds, TimingStateUndo& undo)
{
  TimingEventResult result;
  auto& ra = output.at("pin_rAAT"); auto& fa = output.at("pin_fAAT");
  auto& rs = output.at("pin_rtran"); auto& fs = output.at("pin_ftran");
  auto& rr = output.at("pin_rRAT"); auto& fr = output.at("pin_fRAT");
  auto& drr = output.at("cell_arc_rr_delays"); auto& dfr = output.at("cell_arc_fr_delays");
  auto& drf = output.at("cell_arc_rf_delays"); auto& dff = output.at("cell_arc_ff_delays");
  const auto& cap_r = output.at("pin_net_cap_rise");
  const auto& cap_f = output.at("pin_net_cap_fall");
  const auto& delay_r = output.at("pin_net_delay_rise");
  const auto& delay_f = output.at("pin_net_delay_fall");
  const auto& impulse_r = output.at("pin_net_impulse_rise");
  const auto& impulse_f = output.at("pin_net_impulse_fall");
  const auto& s = mutation && mutation->canonical_schedule ? mutation->events : p.schedule.events;
  const auto& cells = mutation ? mutation->cells : p.schedule.cells;
  const int pins = index.pins.size();
  const auto rd = p.luts[0].view(), fd = p.luts[1].view();
  const auto rt = p.luts[2].view(), ft = p.luts[3].view();
  auto& coverage = result.coverage;
  for (const auto* field : {"clk2q_rows", "cell_forward_rows", "cell_reverse_rows",
                             "net_forward_rows", "net_reverse_rows", "setup_rows",
                             "recovery_rows", "forward_pins", "reverse_pins"}) coverage[field] = 0;
  coverage["forward_seed_pins"] = seeds.forward.size();
  coverage["reverse_seed_pins"] = seeds.reverse.size();
  coverage["total_pins"] = pins;
  coverage["total_cell_arcs"] = index.cells.size();
  if (seeds.forward.empty() && seeds.reverse.empty() && seeds.retired_arcs.empty()) return result;
  constexpr std::array<const char*, 10> fields{
      "pin_rAAT", "pin_fAAT", "pin_rtran", "pin_ftran", "pin_rRAT", "pin_fRAT",
      "cell_arc_rr_delays", "cell_arc_fr_delays", "cell_arc_rf_delays", "cell_arc_ff_delays"};
  std::array<TimingFieldUndo*, 10> saved;
  std::array<std::vector<double>*, 10> arrays;
  for (std::size_t lane = 0; lane < fields.size(); ++lane) {
    arrays[lane] = &output.at(fields[lane]);
    saved[lane] = &undo.fields.emplace(fields[lane],
        TimingFieldUndo{arrays[lane]->size(), {}}).first->second;
  }
  auto capture = [&](int lane, int id) {
    saved[lane]->previous.push_back({static_cast<std::size_t>(id), (*arrays[lane])[id]});
  };
  using Event = std::pair<int, int>;
  std::priority_queue<Event, std::vector<Event>, std::greater<Event>> forward;
  std::priority_queue<Event> reverse;
  std::vector<unsigned char> forward_queued(pins, 0), reverse_queued(pins, 0);
  auto enqueue_forward = [&](int pin) {
    if (pin >= 0 && pin < pins && !forward_queued[pin]) {
      forward_queued[pin] = 1;
      forward.emplace(index.rank[pin], pin);
    }
  };
  auto enqueue_reverse = [&](int pin) {
    if (pin >= 0 && pin < pins && !reverse_queued[pin]) {
      reverse_queued[pin] = 1;
      reverse.emplace(index.rank[pin], pin);
    }
  };
  for (int pin : seeds.forward) enqueue_forward(pin);
  for (int pin : seeds.reverse) enqueue_reverse(pin);
  for (int slot : seeds.retired_arcs) {
    for (int lane = 6; lane < 10; ++lane) {
      capture(lane, slot);
      (*arrays[lane])[slot] = 0;
    }
  }
  auto active = [&](int pin) { return !mutation || mutation->pin_active.at(pin); };
  while (!forward.empty()) {
    const int pin = forward.top().second;
    forward.pop();
    ++coverage["forward_pins"];
    const std::array<double, 4> before{ra[pin], fa[pin], rs[pin], fs[pin]};
    for (int lane = 0; lane < 4; ++lane) capture(lane, pin);
    ra[pin] = fa[pin] = -2e8;
    rs[pin] = fs[pin] = 0;
    if (active(pin)) {
      const int start = pin < p.pin_count ? p.start_row_by_pin[pin] : -1;
      if (start >= 0) {
        ra[pin] = p.start_rAAT[start]; fa[pin] = p.start_fAAT[start];
        rs[pin] = p.start_rtran[start]; fs[pin] = p.start_ftran[start];
      }
      std::vector<ProgramCellArc> clocks;
      for (const auto& op : s.forward_inputs[pin]) {
        if (op.kind == TimingEventKind::clock) clocks.push_back(cells[op.arc]);
      }
      if (!clocks.empty()) {
        dreamplace::timing::clk2qRows(clocks.front().data(), clocks.size(), 8, pins,
            cap_r.data(), cap_f.data(), p.clock_rtran.data(), p.clock_ftran.data(), rd, fd, rt, ft,
            p.pairwise_smooth, p.pairwise_alpha, ra.data(), fa.data(), rs.data(), fs.data());
        coverage["clk2q_rows"] += clocks.size();
      }
      // Recompute all incoming contributions: the old winning edge may
      // have changed or disappeared. Keep the full schedule's row order.
      for (const auto& op : s.forward_inputs[pin]) {
        if (op.kind == TimingEventKind::clock) continue;
        if (op.kind == TimingEventKind::net) {
          const ProgramNetArc row{op.source, op.sink};
          double values[4], jacobian[6];
          netForwardRow(0, row.data(), 2, 1, pins, ra.data(), fa.data(), rs.data(), fs.data(),
              delay_r.data(), delay_f.data(), impulse_r.data(), impulse_f.data(),
              values, jacobian, p.slew_epsilon);
          dreamplace::timing::aggregateNetRows(row.data(), values, 1, 2,
                                               ra.data(), fa.data(), rs.data(), fs.data());
          ++coverage["net_forward_rows"];
        } else {
          const auto& row = cells[op.arc];
          double values[8], jacobian[48];
          cellForwardRow(0, row.data(), 8, 1, pins, ra.data(), fa.data(), rs.data(), fs.data(),
              cap_r.data(), cap_f.data(), rd, fd, rt, ft, values, jacobian,
              p.pairwise_smooth, p.pairwise_alpha);
          const std::array<double, 4> old_delay{drr[op.arc], dfr[op.arc], drf[op.arc], dff[op.arc]};
          const std::array<double, 4> new_delay{values[4], values[5], values[6], values[7]};
          for (int lane = 6; lane < 10; ++lane) capture(lane, op.arc);
          drr[op.arc] = values[4]; dfr[op.arc] = values[5];
          drf[op.arc] = values[6]; dff[op.arc] = values[7];
          if (changed(old_delay, new_delay)) enqueue_reverse(op.source);
          dreamplace::timing::aggregateCellRows(row.data(), values, 1, 8,
                                                ra.data(), fa.data(), rs.data(), fs.data());
          ++coverage["cell_forward_rows"];
        }
      }
    }
    if (!active(pin)) enqueue_reverse(pin);
    const std::array<double, 4> after{ra[pin], fa[pin], rs[pin], fs[pin]};
    if (std::memcmp(before.data(), after.data(), sizeof(before))) {
      result.changed_pins.push_back(pin);
      forward_queued[pin] = 2;
    }
    if (changed(before, after)) {
      for (int sink : s.forward_successors[pin]) enqueue_forward(sink);
      if (pin < p.pin_count && (!p.setup_rows_by_pin[pin].empty()
                                || !p.recovery_rows_by_pin[pin].empty())) enqueue_reverse(pin);
    }
  }

  std::vector<double> scratch_r(pins), scratch_f(pins);
  while (!reverse.empty()) {
    const int pin = reverse.top().second;
    reverse.pop();
    ++coverage["reverse_pins"];
    const std::array<double, 2> before{rr[pin], fr[pin]};
    capture(4, pin);
    capture(5, pin);
    rr[pin] = fr[pin] = 2e8;
    if (active(pin)) {
      const int endpoint = pin < p.pin_count ? p.endpoint_row_by_pin[pin] : -1;
      if (endpoint >= 0) {
        rr[pin] = p.endpoint_rRAT[endpoint]; fr[pin] = p.endpoint_fRAT[endpoint];
      }
      if (pin < p.pin_count) {
        scratch_r[pin] = rr[pin]; scratch_f[pin] = fr[pin];
        for (int row : p.setup_rows_by_pin[pin]) {
          dreamplace::timing::setupRatRows(p.setup_arcs[row].data(), 1, 8,
              rr.data(), fr.data(), p.clock_rtran.data(), p.clock_ftran.data(),
              rs.data(), fs.data(), rd, fd, p.setup_valid.get() + row * 2,
              p.pairwise_smooth, p.pairwise_alpha, scratch_r.data(), scratch_f.data());
          ++coverage["setup_rows"];
        }
        rr[pin] = scratch_r[pin]; fr[pin] = scratch_f[pin];
        const auto& recovery = p.recovery_rows_by_pin[pin];
        if (!recovery.empty()) {
          std::vector<ProgramCellArc> rows;
          std::vector<double> reference_r, reference_f;
          auto valid = std::make_unique<bool[]>(recovery.size() * 2);
          for (int row : recovery) {
            rows.push_back(p.recovery_arcs[row]);
            reference_r.push_back(p.recovery_uses_live_slew ? rs[pin] : p.recovery_reference_rtran[row]);
            reference_f.push_back(p.recovery_uses_live_slew ? fs[pin] : p.recovery_reference_ftran[row]);
            valid[(rows.size() - 1) * 2] = p.recovery_valid[row * 2];
            valid[(rows.size() - 1) * 2 + 1] = p.recovery_valid[row * 2 + 1];
          }
          dreamplace::timing::recoveryRatRows(rows.front().data(), rows.size(), 8, pins,
              rr.data(), fr.data(), rs.data(), fs.data(), p.recovery_clock_rtran.data(),
              p.recovery_clock_ftran.data(), reference_r.data(), reference_f.data(),
              p.luts[4].view(), p.luts[5].view(), valid.get(), scratch_r.data(), scratch_f.data(),
              p.recovery_rat_is_capture_deadline, p.recovery_deadline_r.data(),
              p.recovery_deadline_f.data(), p.recovery_keep_setup.get());
          rr[pin] = scratch_r[pin]; fr[pin] = scratch_f[pin];
          coverage["recovery_rows"] += rows.size();
        }
      }
      for (const auto& op : s.reverse_outputs[pin]) {
        scratch_r[pin] = rr[pin]; scratch_f[pin] = fr[pin];
        if (op.kind == TimingEventKind::cell) {
          dreamplace::timing::cellRatRows(cells[op.arc].data(), 1, 8, rr.data(), fr.data(),
              drr.data() + op.arc, dfr.data() + op.arc, drf.data() + op.arc, dff.data() + op.arc,
              scratch_r.data(), scratch_f.data());
          ++coverage["cell_reverse_rows"];
        } else {
          const ProgramNetArc row{op.source, op.sink};
          dreamplace::timing::netRatRows(row.data(), 1, 2, rr.data(), fr.data(),
              delay_r.data(), delay_f.data(), scratch_r.data(), scratch_f.data());
          ++coverage["net_reverse_rows"];
        }
        rr[pin] = scratch_r[pin]; fr[pin] = scratch_f[pin];
      }
    }
    const std::array<double, 2> after{rr[pin], fr[pin]};
    if (forward_queued[pin] != 2 && std::memcmp(before.data(), after.data(), sizeof(before))) {
      result.changed_pins.push_back(pin);
    }
    if (changed(before, after)) {
      for (int predecessor : s.reverse_predecessors[pin]) enqueue_reverse(predecessor);
    }
  }
  return result;
}

}  // namespace rebuffer
