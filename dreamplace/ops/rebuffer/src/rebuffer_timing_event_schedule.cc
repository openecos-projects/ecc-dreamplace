// SPDX-License-Identifier: BSD-3-Clause
// Preserve the full TP schedule's per-pin contribution order in event worklists.
#include "rebuffer_timing_events.hh"

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
