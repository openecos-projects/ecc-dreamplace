// SPDX-License-Identifier: BSD-3-Clause
// Derive event seeds from native journal changes, including removed net edges.
#include "rebuffer_timing_graph.hh"

#include "rebuffer_timing_mutation.hh"


#include <algorithm>
#include <cmath>
#include <cstring>

#include "pybind11/stl.h"


namespace py = pybind11;
namespace rebuffer {

TimingEventSeeds PersistentTimingGraph::timingEventSeeds() const
{
  TimingEventSeeds seeds;
  if (!propagation_mark_.ready) {
    seeds.require_full = true;
    return seeds;
  }
  for (const auto& entry : journal_) {
    if (entry.sequence <= propagation_mark_.sequence) continue;
    if (entry.resize_undo) {
      for (const auto& [id, previous] : entry.resize_undo->arcs) {
        seeds.forward.insert(timing_index_->cells.at(id)[1]);
        seeds.reverse.insert(previous[0]);
      }
    } else if (entry.topology_undo) {
      const auto& undo = *entry.topology_undo;
      for (const auto& net : undo.previous_nets) {
        if (net.driver >= 0) seeds.reverse.insert(net.driver);
        seeds.forward.insert(net.pins.begin(), net.pins.end());
      }
      auto add_net = [&](int net) {
        const auto& pins = mutation_state_->net_pins.at(net);
        seeds.forward.insert(pins.begin(), pins.end());
        const int driver = mutation_state_->net_drivers.at(net);
        if (driver >= 0) seeds.reverse.insert(driver);
      };
      add_net(undo.source_net);
      for (const auto& row : mutation_state_->buffers) {
        if (row.source_net != undo.source_net) continue;
        if (row.active) add_net(row.output_net);
        for (int arc : row.arcs) {
          if (!mutation_state_->cell_active.at(arc)) seeds.retired_arcs.insert(arc);
        }
      }
      for (const auto& arc : undo.previous_arcs) {
        if (!mutation_state_->cell_active.at(arc.id)) seeds.retired_arcs.insert(arc.id);
      }
    } else if (entry.state_undo) {
      for (const auto& [name, field] : entry.state_undo->fields) {
        if (field.previous.empty()) continue;
        if (name.rfind("pin_net_", 0) != 0) {
          seeds.require_full = true;
          return seeds;
        }
        for (const auto& old : field.previous) {
          const int pin = old.index;
          seeds.forward.insert(pin);
          seeds.reverse.insert(pin);
          for (int edge : timing_index_->net_in.at(pin)) {
            seeds.reverse.insert(timing_index_->nets[edge][0]);
          }
        }
      }
    } else if (entry.name.rfind("pin_net_", 0) == 0 && entry.existed) {
      const auto& current = state_.at(entry.name);
      for (int pin = 0; pin < static_cast<int>(current.size()); ++pin) {
        if (pin >= static_cast<int>(entry.previous.size())
            || std::isnan(current[pin]) || std::isnan(entry.previous[pin])
            || std::memcmp(&current[pin], &entry.previous[pin], sizeof(double))) {
          seeds.forward.insert(pin);
          seeds.reverse.insert(pin);
          for (int edge : timing_index_->net_in.at(pin)) {
            seeds.reverse.insert(timing_index_->nets[edge][0]);
          }
        }
      }
    } else if (entry.existed && state_.count(entry.name)
               && (entry.name.rfind("pin_", 0) == 0
                   || entry.name.rfind("cell_arc_", 0) == 0)) {
      const auto& current = state_.at(entry.name);
      // Buffer mutation expands the canonical arrays without changing old
      // timing values. New slots are already covered by topology seeds.
      if (current.size() < entry.previous.size()
          || (!entry.previous.empty() && std::memcmp(current.data(), entry.previous.data(),
                                                     entry.previous.size() * sizeof(double)))) {
        seeds.require_full = true;
        return seeds;
      }
    } else {
      // Canonical timing was replaced outside propagation, or generic
      // scratch state was edited. Re-establish a native full-seed baseline.
      seeds.require_full = true;
      return seeds;
    }
  }
  return seeds;
}

py::dict PersistentTimingGraph::propagateIncremental()
{
  if (!timing_state_initialized_ || !propagation_program_) return propagate();
  if (! net_states_.empty()) {
    if (!mutation_state_) {
      throw py::value_error("event propagation requires synchronized native topology");
    }
    requireSynchronizedTimingTopologies(*mutation_state_, net_states_);
  }
  const auto seeds = timingEventSeeds();
  if (seeds.require_full) {
    auto result = propagate();
    result["fallback_reason"] = "native_full_seed_baseline_required";
    return result;
  }
  const auto checkpoint_size = journal_.size();
  const auto previous_slacks = endpointSlacks();
  try {
    // Register undo before the first in-place kernel write, including failures
    // that occur after forward propagation has already updated other pins.
    auto undo = beginStateUpdate();
    auto computed = propagateTimingEvents(*propagation_program_, *timing_index_,
                                          mutation_state_.get(), state_, seeds, *undo);
    py::dict coverage;
    for (const auto& [name, value] : computed.coverage) coverage[py::str(name)] = value;
    return finishSparsePropagation(undo, std::move(coverage), previous_slacks,
                                    std::move(computed.changed_pins));
  } catch (...) {
    restoreTo(checkpoint_size);
    throw;
  }
}

}  // namespace rebuffer

// --- rebuffer_timing_event_math.cc ---
// Native changed-value worklists over the canonical Python TP equations.

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

// --- rebuffer_timing_event_schedule.cc ---
// Preserve the full TP schedule's per-pin contribution order in event worklists.

#include <algorithm>

namespace rebuffer {

void buildTimingEventSchedule(TimingPropagationSchedule& s, const TimingGraphIndex& index)
{
  const auto pins = index.pins.size();
  auto& events = s.events;
  events.forward_inputs.assign(pins, {});
  events.reverse_outputs.assign(pins, {});
  events.forward_successors.assign(pins, {});
  events.reverse_predecessors.assign(pins, {});
  auto forward = [&](TimingEventOperation op) {
    events.forward_inputs.at(op.sink).push_back(op);
    if (op.kind != TimingEventKind::clock && op.source != op.sink) {
      events.forward_successors.at(op.source).push_back(op.sink);
    }
  };
  auto reverse = [&](TimingEventOperation op) {
    events.reverse_outputs.at(op.source).push_back(op);
    if (op.source != op.sink) events.reverse_predecessors.at(op.sink).push_back(op.source);
  };
  auto cell = [&](int row) {
    const int slot = s.arc_ids.at(row);
    return TimingEventOperation{slot < index.clock_end ? TimingEventKind::clock
                                                      : TimingEventKind::cell,
                               index.source(slot), static_cast<int>(s.cells[row][1]), slot};
  };
  for (int row = 0; row < s.level_start[1]; ++row) forward(cell(row));
  for (const auto& arc : s.forward_nets[0].arcs) {
    forward({TimingEventKind::net, static_cast<int>(arc[0]), static_cast<int>(arc[1]), -1});
  }
  for (int level = 1; level + 1 < static_cast<int>(s.level_start.size()); ++level) {
    for (int row = s.level_start[level]; row < s.level_start[level + 1]; ++row) forward(cell(row));
    for (const auto& arc : s.forward_nets[level].arcs) {
      forward({TimingEventKind::net, static_cast<int>(arc[0]), static_cast<int>(arc[1]), -1});
    }
  }
  for (const auto& arc : s.endpoint_nets.arcs) {
    reverse({TimingEventKind::net, static_cast<int>(arc[0]), static_cast<int>(arc[1]), -1});
  }
  for (int level = s.level_start.size() - 2; level >= 1; --level) {
    for (int row = s.level_start[level]; row < s.level_start[level + 1]; ++row) reverse(cell(row));
    for (const auto& arc : s.reverse_nets[level].arcs) {
      reverse({TimingEventKind::net, static_cast<int>(arc[0]), static_cast<int>(arc[1]), -1});
    }
  }
  for (auto* groups : {&events.forward_successors, &events.reverse_predecessors}) {
    for (auto& row : *groups) {
      std::sort(row.begin(), row.end());
      row.erase(std::unique(row.begin(), row.end()), row.end());
    }
  }
}

}  // namespace rebuffer

// --- rebuffer_timing_schedule.cc ---
// Derive an execution schedule while retaining canonical pin/cell slot ids.
#include "rebuffer_timing_local.hh"


#include <algorithm>
#include <chrono>
#include <map>

namespace rebuffer {
namespace {

std::vector<int64_t> unique(std::vector<int64_t> values)
{
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

}  // namespace

void buildTimingMutationIndex(const TimingMutationState& m, TimingGraphIndex& index)
{
  index.cells.resize(m.cells.size());
  for (std::size_t id = 0; id < m.cells.size(); ++id) {
    std::copy(m.cells[id].begin(), m.cells[id].end(), index.cells[id].begin());
  }
  index.cell_active = m.cell_active;
  index.nets.clear();
  for (std::size_t net = 0; net < m.net_drivers.size(); ++net) {
    if (m.net_drivers[net] < 0) continue;
    for (int pin : m.net_pins[net]) {
      if (pin != m.net_drivers[net]) index.nets.push_back({m.net_drivers[net], pin});
    }
  }
  index.net_active.assign(index.nets.size(), 1);
  index.build();
}

void rebuildTimingMutationIndex(TimingMutationState& m, TimingGraphIndex& index,
                                const TimingPropagationProgram& program)
{
  const auto started = std::chrono::steady_clock::now();
  buildTimingMutationIndex(m, index);
  const auto indexed = std::chrono::steady_clock::now();
  ++m.statistics.index_builds;
  m.statistics.index_ms += std::chrono::duration<double, std::milli>(indexed - started).count();
  initializeTimingLocalIndex(m, index, program);
  m.canonical_schedule = true;
}

std::shared_ptr<TimingPropagationSchedule> makeTimingMutationSchedule(
    const TimingMutationState& m, const TimingGraphIndex& index,
    const TimingPropagationProgram& program)
{
  auto result = std::make_shared<TimingPropagationSchedule>();
  auto& s = *result;
  std::map<int, std::vector<int>> levels;
  int count = 1;
  for (int id = 0; id < static_cast<int>(m.cells.size()); ++id) {
    if (!m.cell_active[id]) continue;
    const int level = id < index.clock_end ? 0 : index.rank[m.cells[id][0]] + 1;
    levels[level].push_back(id);
    count = std::max(count, level + 1);
  }
  s.level_start.push_back(0);
  for (int level = 0; level < count; ++level) {
    for (int id : levels[level]) {
      s.cells.push_back(m.cells[id]);
      s.arc_ids.push_back(id);
    }
    s.level_start.push_back(s.cells.size());
  }
  auto net_id = [&](int pin) {
    int net = m.pin_net.at(pin);
    if (net < 0) net += m.net_drivers.size();
    if (net < 0 || net >= static_cast<int>(m.net_drivers.size())) {
      throw pybind11::value_error("timing schedule pin has no net");
    }
    return net;
  };
  auto forward = [&](const std::vector<int64_t>& pins) {
    std::vector<int64_t> nets;
    for (int pin : pins) nets.push_back(net_id(pin));
    ProgramNetGroup group;
    for (int net : unique(std::move(nets))) {
      const int driver = m.net_drivers[net];
      if (driver < 0) continue;
      for (int pin : m.net_pins[net]) {
        if (pin != driver) group.arcs.push_back({driver, pin});
      }
    }
    return group;
  };
  auto reverse = [&](const std::vector<int64_t>& pins) {
    ProgramNetGroup group;
    for (int pin : pins) {
      const int driver = m.net_drivers[net_id(pin)];
      if (driver < 0) continue;
      group.arcs.push_back({driver, pin});
      group.drivers.push_back(driver);
    }
    group.drivers = unique(std::move(group.drivers));
    return group;
  };
  s.forward_nets.resize(count);
  s.reverse_nets.resize(count);
  s.level_inputs.resize(count);
  s.forward_nets[0] = forward(program.start_pins);
  s.endpoint_nets = reverse(program.endpoint_pins);
  for (int level = 1; level < count; ++level) {
    std::vector<int64_t> inputs, outputs;
    for (auto row = s.level_start[level]; row < s.level_start[level + 1]; ++row) {
      inputs.push_back(s.cells[row][0]);
      outputs.push_back(s.cells[row][1]);
    }
    s.level_inputs[level] = unique(std::move(inputs));
    s.forward_nets[level] = forward(outputs);
    s.reverse_nets[level] = reverse(s.level_inputs[level]);
  }
  buildTimingEventSchedule(s, index);
  return result;
}

}  // namespace rebuffer
