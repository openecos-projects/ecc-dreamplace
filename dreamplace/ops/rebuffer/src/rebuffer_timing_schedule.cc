// SPDX-License-Identifier: BSD-3-Clause
// Derive an execution schedule while retaining canonical pin/cell slot ids.
#include "rebuffer_timing_mutation.hh"
#include "rebuffer_timing_events.hh"
#include "rebuffer_timing_local_index.hh"

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
