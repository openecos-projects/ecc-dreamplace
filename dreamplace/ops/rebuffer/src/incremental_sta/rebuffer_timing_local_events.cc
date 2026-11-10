// SPDX-License-Identifier: BSD-3-Clause
// Reconstruct affected event rows in the full levelized oracle's exact order.
#include "rebuffer_timing_local_index.hh"
#include "rebuffer_timing_mutation.hh"

#include <algorithm>
#include <map>
#include <tuple>

namespace rebuffer {
namespace {
struct OrderedOperation
{
  // Forward: level, cell-before-net, id. Reverse: endpoint-before-level,
  // descending level, cell-before-net, id.
  std::tuple<int, int, int, int> order;
  TimingEventOperation operation;
};

std::vector<TimingEventOperation> ordered(std::vector<OrderedOperation> rows)
{
  std::sort(rows.begin(), rows.end(), [](const auto& a, const auto& b) { return a.order < b.order; });
  std::vector<TimingEventOperation> result;
  result.reserve(rows.size());
  for (const auto& row : rows) result.push_back(row.operation);
  return result;
}

void unique(std::vector<int>& values)
{
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
}
}  // namespace

void updateTimingEventRows(TimingMutationState& m, const TimingGraphIndex& index,
    const TimingPropagationProgram& program, const std::vector<int>& pins, TimingIndexUndo& undo)
{
  std::map<int, std::vector<int>> forward_levels;
  auto levels = [&](int net) -> const std::vector<int>& {
    const auto found = forward_levels.find(net);
    if (found != forward_levels.end()) return found->second;
    auto& result = forward_levels[net];
    for (int pin : m.net_pins.at(net)) {
      if (pin < program.pin_count && program.start_row_by_pin[pin] >= 0) {
        result.push_back(0);
        break;
      }
    }
    const int driver = m.net_drivers.at(net);
    if (driver >= 0) {
      for (int arc : index.cell_in.at(driver)) {
        if (arc >= index.clock_end) result.push_back(index.rank[index.source(arc)] + 1);
      }
    }
    unique(result);
    return result;
  };
  auto cell_input = [&](int pin) {
    return std::any_of(index.cell_out[pin].begin(), index.cell_out[pin].end(),
                       [&](int arc) { return arc >= index.clock_end; });
  };
  for (int pin : pins) {
    captureTimingIndexPin(undo, m, index, pin);
    std::vector<OrderedOperation> forward, reverse;
    std::vector<int> successors, predecessors;
    for (int arc : index.cell_in[pin]) {
      const int source = index.source(arc);
      const bool clock = arc < index.clock_end;
      forward.push_back({{clock ? 0 : index.rank[source] + 1, 0, arc, 0},
          {clock ? TimingEventKind::clock : TimingEventKind::cell, source, pin, arc}});
      if (!clock && source != pin) predecessors.push_back(source);
    }
    for (int arc : index.net_in[pin]) {
      const int source = index.nets[arc][0], net = m.pin_net[pin];
      for (int level : levels(net)) {
        forward.push_back({{level, 1, net, pin}, {TimingEventKind::net, source, pin, -1}});
      }
    }
    for (int arc : index.cell_out[pin]) {
      if (arc < index.clock_end) continue;
      const int sink = index.cells[arc][1];
      reverse.push_back({{1, -index.rank[pin] - 1, 0, arc}, {TimingEventKind::cell, pin, sink, arc}});
      if (sink != pin) successors.push_back(sink);
    }
    const int net = m.pin_net[pin];
    if (net >= 0 && m.net_drivers[net] == pin) {
      const bool forward_net = !levels(net).empty();
      for (int sink : m.net_pins[net]) {
        const int endpoint = sink < program.pin_count ? program.endpoint_row_by_pin[sink] : -1;
        if (endpoint >= 0) {
          reverse.push_back({{0, 0, 0, endpoint}, {TimingEventKind::net, pin, sink, -1}});
        }
        if (cell_input(sink)) {
          reverse.push_back({{1, -index.rank[sink] - 1, 1, sink},
                              {TimingEventKind::net, pin, sink, -1}});
        }
        if (forward_net && sink != pin) successors.push_back(sink);
      }
    }
    const bool endpoint = pin < program.pin_count && program.endpoint_row_by_pin[pin] >= 0;
    if (net >= 0 && (endpoint || cell_input(pin))) {
      const int driver = m.net_drivers[net];
      if (driver >= 0 && driver != pin) predecessors.push_back(driver);
    }
    unique(successors); unique(predecessors);
    m.events.forward_inputs[pin] = ordered(std::move(forward));
    m.events.reverse_outputs[pin] = ordered(std::move(reverse));
    m.events.forward_successors[pin] = std::move(successors);
    m.events.reverse_predecessors[pin] = std::move(predecessors);
  }
  m.statistics.event_rows += pins.size();
}

}  // namespace rebuffer
