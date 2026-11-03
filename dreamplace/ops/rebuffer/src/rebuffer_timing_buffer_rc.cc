// SPDX-License-Identifier: BSD-3-Clause
// Evaluate source-local RC directly, without dense namespaces or Python arrays.
#include "rebuffer_timing_buffer_state.hh"
#include "rebuffer_timing_graph.hh"
#include "rebuffer_incremental_rc.hh"

#include <algorithm>
#include <chrono>
#include <unordered_map>

namespace rebuffer {
namespace {

TimingNetRcValues timingNetRcValues(const TimingMutationState& m,
    const IncrementalTopologyState& topology, const CompactRcForest& forest,
    const std::array<std::vector<double>, 5>& pin_values, int original_pin_count)
{
  requireSynchronizedTimingTopology(m, topology);
  std::unordered_map<int, int> buffer_pins;
  for (const auto& buffer : topology.buffers) {
    const auto found = std::find_if(m.buffers.begin(), m.buffers.end(), [&](const auto& row) {
      return row.source_net == topology.net_id && row.action_id == buffer.action_id && row.active;
    });
    if (found == m.buffers.end()) throw pybind11::value_error("RC buffer has no active timing record");
    buffer_pins.emplace(buffer.input_node, found->input_pin);
    buffer_pins.emplace(buffer.output_node, found->output_pin);
  }
  for (const auto& [node, loads] : topology.node_loads) {
    if (loads.size() != 1 || loads.front() != node) {
      throw pybind11::value_error("native timing RC requires explicit physical pin vertices");
    }
  }
  TimingNetRcValues result;
  std::vector<int> original_physical, physical_pins(forest.node_ids.size(), -1), selected;
  for (int i = 0; i < static_cast<int>(forest.node_ids.size()); ++i) {
    const int node = forest.node_ids[i];
    const auto buffer = buffer_pins.find(node);
    if (buffer != buffer_pins.end()) physical_pins[i] = buffer->second;
    else if (node < original_pin_count) {
      original_physical.push_back(node);
      physical_pins[i] = node;
    }
    if (physical_pins[i] >= 0) {
      result.pins.push_back(physical_pins[i]);
      selected.push_back(i);
    }
  }
  if (original_physical != m.base_net_pins.at(topology.net_id)) {
    throw pybind11::value_error("timing RC topology physical pins do not match source net");
  }
  const auto wire_cap = compactRcWireCaps(forest);
  // Canonical timing state consumes rise/fall RC only. The public adapter
  // still computes generic mode for callers that request it.
  for (int mode = 0; mode < 2; ++mode) {
    std::vector<double> caps(forest.node_ids.size(), 0);
    for (int i = 0; i < static_cast<int>(caps.size()); ++i) {
      if (physical_pins[i] >= 0) caps[i] = pin_values[mode + 1].at(physical_pins[i]);
      caps[i] += wire_cap[i];
    }
    const auto values = evaluateCompactRc(forest, caps);
    for (int field = 0; field < 3; ++field) {
      auto& output = result.values[2 * field + mode];
      output.reserve(selected.size());
      for (int i : selected) output.push_back(values[field][i]);
    }
  }
  return result;
}

}  // namespace

TimingNetRcValues PersistentTimingGraph::recomputeTimingNetRc(
    const IncrementalTopologyState& topology,
    const std::array<std::vector<double>, 5>& pin_values)
{
  const auto started = std::chrono::steady_clock::now();
  const auto forest = compactRcForest(topology, propagation_program_->pin_count);
  const auto built = std::chrono::steady_clock::now();
  auto result = timingNetRcValues(*mutation_state_, topology, forest, pin_values,
                                  propagation_program_->pin_count);
  auto& stats = mutation_state_->statistics;
  ++stats.rc_builds;
  stats.rc_nodes += forest.node_ids.size();
  stats.rc_pin_reads += 2 * result.pins.size();
  stats.rc_build_ms += std::chrono::duration<double, std::milli>(built - started).count();
  stats.rc_evaluate_ms += std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - built).count();
  return result;
}

}  // namespace rebuffer
