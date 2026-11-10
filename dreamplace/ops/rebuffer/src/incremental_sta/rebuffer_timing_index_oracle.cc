// SPDX-License-Identifier: BSD-3-Clause
// Explicit diagnostic oracle: rebuild independently without altering the owner.
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_mutation.hh"

#include <algorithm>
#include <string>

#include "pybind11/stl.h"

namespace rebuffer {
namespace {
void require(bool condition, const std::string& field)
{
  if (!condition) throw pybind11::value_error("native timing index oracle mismatch: " + field);
}

std::vector<TimingNetArc> edges(const TimingGraphIndex& index, const std::vector<int>& ids)
{
  std::vector<TimingNetArc> result;
  for (int id : ids) {
    require(index.net_active.empty() || index.net_active.at(id), "inactive edge in adjacency");
    result.push_back(index.nets.at(id));
  }
  std::sort(result.begin(), result.end());
  return result;
}

std::vector<int> sorted(std::vector<int> values)
{
  std::sort(values.begin(), values.end());
  return values;
}

std::string operation(const std::vector<TimingEventOperation>& rows, std::size_t id)
{
  if (id >= rows.size()) return "end";
  const auto& row = rows[id];
  return std::to_string(static_cast<int>(row.kind)) + ":" + std::to_string(row.source)
      + ":" + std::to_string(row.sink) + ":" + std::to_string(row.arc);
}

void sameEvents(const std::vector<TimingEventOperation>& actual,
                 const std::vector<TimingEventOperation>& expected, const std::string& field)
{
  std::size_t id = 0;
  while (id < actual.size() && id < expected.size() && actual[id] == expected[id]) ++id;
  require(id == actual.size() && id == expected.size(), field + " row=" + std::to_string(id)
      + " actual=" + operation(actual, id) + " expected=" + operation(expected, id));
}
}  // namespace

pybind11::dict PersistentTimingGraph::mutationIndexOracle() const
{
  if (!mutation_state_ || !propagation_program_) {
    throw pybind11::value_error("mutation inputs are not initialized");
  }
  const auto& m = *mutation_state_;
  const auto& current = *timing_index_;
  TimingGraphIndex rebuilt;
  rebuilt.pins = current.pins; rebuilt.clocks = current.clocks; rebuilt.clock_end = current.clock_end;
  buildTimingMutationIndex(m, rebuilt);
  require(current.cells == rebuilt.cells && current.cell_active == rebuilt.cell_active, "cell slots");
  require(current.pin_ids == rebuilt.pin_ids, "pin identities");
  require(current.arc_ids == rebuilt.arc_ids, "arc identities");
  require(current.net_active.size() == current.nets.size(), "net active slots");
  auto schedule = makeTimingMutationSchedule(m, rebuilt, *propagation_program_);
  for (int pin = 0; pin < static_cast<int>(current.pins.size()); ++pin) {
    const auto label = " pin=" + std::to_string(pin) + " " + current.pins[pin];
    require(current.rank.at(pin) == rebuilt.rank.at(pin), "rank" + label);
    require(current.path_level.at(pin) == rebuilt.path_level.at(pin), "path level" + label);
    require(current.cell_out.at(pin) == rebuilt.cell_out.at(pin), "cell_out" + label);
    require(current.cell_in.at(pin) == rebuilt.cell_in.at(pin), "cell_in" + label);
    require(sorted(current.successors.at(pin)) == sorted(rebuilt.successors.at(pin)), "successors" + label);
    require(edges(current, current.net_out.at(pin)) == edges(rebuilt, rebuilt.net_out.at(pin)), "net_out" + label);
    require(edges(current, current.net_in.at(pin)) == edges(rebuilt, rebuilt.net_in.at(pin)), "net_in" + label);
    const int edge = m.net_arc_by_pin.at(pin);
    if (edge >= 0) require(current.nets.at(edge)[1] == pin, "net slot owner" + label);
    sameEvents(m.events.forward_inputs.at(pin), schedule->events.forward_inputs.at(pin), "forward inputs" + label);
    sameEvents(m.events.reverse_outputs.at(pin), schedule->events.reverse_outputs.at(pin), "reverse outputs" + label);
    require(m.events.forward_successors.at(pin) == schedule->events.forward_successors.at(pin), "event successors" + label);
    require(m.events.reverse_predecessors.at(pin) == schedule->events.reverse_predecessors.at(pin), "event predecessors" + label);
  }
  std::vector<TimingNetArc> active;
  for (std::size_t id = 0; id < current.nets.size(); ++id) if (current.net_active[id]) active.push_back(current.nets[id]);
  auto expected = rebuilt.nets;
  std::sort(active.begin(), active.end()); std::sort(expected.begin(), expected.end());
  require(active == expected, "active net edges");
  pybind11::dict result;
  result["status"] = "ok";
  result["pin_count"] = current.pins.size();
  result["cell_arc_count"] = current.cells.size();
  result["active_net_arc_count"] = active.size();
  return result;
}

}  // namespace rebuffer
