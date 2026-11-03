// SPDX-License-Identifier: BSD-3-Clause
// Read current graph-owned timing directly; no carrier or state materialization.
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_buffer_state.hh"
#include "rebuffer_timing_events.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <set>

#include "timing_hard_math.h"
#include "rebuffer_timing_paths.hh"
#include "pybind11/stl.h"

namespace py = pybind11;
namespace rebuffer {

TimingSetupPaths PersistentTimingGraph::nativeSetupPaths(
    const std::optional<std::vector<int64_t>>& endpoint_pins,
    int64_t global_k, int64_t max_depth, double residual_tolerance_ps) const
{
  if (global_k < 0) throw py::value_error("global_k must be non-negative");
  if (!std::isfinite(residual_tolerance_ps) || residual_tolerance_ps <= 0) {
    throw py::value_error("residual_tolerance_ps must be finite and positive");
  }
  if (!timing_state_initialized_ || !propagation_program_) {
    throw py::value_error("setup paths require initialized timing state and propagation inputs");
  }
  const auto seeds = timingEventSeeds();
  if (seeds.require_full || !seeds.forward.empty() || !seeds.reverse.empty()
      || !seeds.retired_arcs.empty()) {
    throw py::value_error("setup paths require current native propagation");
  }
  for (const auto& [_, topology] : net_states_) {
    if (!mutation_state_) throw py::value_error("setup paths require synchronized native topology");
    requireSynchronizedTimingTopology(*mutation_state_, *topology);
  }
  const auto& p = *propagation_program_;
  const auto& index = *timing_index_;
  std::set<int64_t> requested;
  const bool filtered = endpoint_pins.has_value();
  if (filtered) {
    for (auto pin : *endpoint_pins) {
      if (pin < 0 || pin >= static_cast<int64_t>(index.pins.size())) {
        throw py::value_error("requested setup endpoint pin is out of range");
      }
      requested.insert(pin);
    }
  }
  auto included = [&](int64_t pin) { return !filtered || requested.count(pin); };
  const auto& ra = state_.at("pin_rAAT");
  const auto& fa = state_.at("pin_fAAT");
  const auto& rs = state_.at("pin_rtran");
  const auto& fs = state_.at("pin_ftran");
  const auto& rr = state_.at("pin_rRAT");
  const auto& fr = state_.at("pin_fRAT");
  const double inf = std::numeric_limits<double>::infinity();
  std::vector<dreamplace::timing::SetupPathState> states;
  std::set<int64_t> constraint_pins;
  std::size_t endpoint_records = 0;
  auto append = [&](int64_t pin, int64_t test, double rise, double fall) {
    if (!included(pin)) return;
    ++endpoint_records;
    dreamplace::timing::appendSetupPathStates(states, pin, test,
        p.endpoint_valid[pin * 2] ? rise : inf,
        p.endpoint_valid[pin * 2 + 1] ? fall : inf);
  };
  for (std::size_t row = 0; row < p.setup_arcs.size(); ++row) {
    const bool rise_valid = p.setup_valid[row * 2];
    const bool fall_valid = p.setup_valid[row * 2 + 1];
    if (!rise_valid && !fall_valid) continue;
    const auto& arc = p.setup_arcs[row];
    const auto pin = arc[1];
    constraint_pins.insert(pin);
    if (!included(pin)) continue;
    const int endpoint_row = p.endpoint_row_by_pin[pin];
    const auto candidate = dreamplace::timing::setupRatRow(arc.data(),
        endpoint_row < 0 ? 2e8 : p.endpoint_rRAT[endpoint_row],
        endpoint_row < 0 ? 2e8 : p.endpoint_fRAT[endpoint_row],
        p.clock_rtran.data(), p.clock_ftran.data(), rs.data(), fs.data(),
        p.luts[0].view(), p.luts[1].view(), p.pairwise_smooth, p.pairwise_alpha);
    append(pin, row, rise_valid ? candidate.first - ra[pin] : inf,
                     fall_valid ? candidate.second - fa[pin] : inf);
  }
  for (auto pin : p.endpoint_pins) {
    if (!p.endpoint_valid[pin * 2] && !p.endpoint_valid[pin * 2 + 1]) continue;
    // Match TP's actual no-setup-rows fallback, including recovery-only pins.
    if (!p.setup_arcs.empty()) {
      const bool check = p.timing_check_pins.empty() ? constraint_pins.count(pin)
          : std::binary_search(p.timing_check_pins.begin(), p.timing_check_pins.end(), pin);
      if (check) continue;
    }
    append(pin, -1, rr[pin] - ra[pin], fr[pin] - fa[pin]);
  }
  const dreamplace::timing::SetupPathTimingView view{
      ra.data(), fa.data(), state_.at("pin_net_delay_rise").data(),
      state_.at("pin_net_delay_fall").data(), state_.at("cell_arc_rr_delays").data(),
      state_.at("cell_arc_fr_delays").data(), state_.at("cell_arc_rf_delays").data(),
      state_.at("cell_arc_ff_delays").data()};
  const bool reordered = mutation_state_ && mutation_state_->canonical_schedule;
  const TimingPathPredecessors predecessors{index, reordered};
  auto extracted = dreamplace::timing::extractSetupPaths(std::move(states), view,
      index.pins.size(), global_k, max_depth, residual_tolerance_ps, predecessors,
      [&](int64_t pin) {
        return pin < p.pin_count && p.start_row_by_pin[pin] >= 0;
      });
  return {std::move(extracted), endpoint_records};
}

py::dict PersistentTimingGraph::setupPathDetails(const py::object& endpoint_pins,
    int64_t global_k, int64_t max_depth, double residual_tolerance_ps) const
{
  const auto started = std::chrono::steady_clock::now();
  const auto result_paths = nativeSetupPaths(endpoint_pins.is_none() ? std::nullopt
      : std::make_optional(endpoint_pins.cast<std::vector<int64_t>>()),
      global_k, max_depth, residual_tolerance_ps);
  const auto& extracted = result_paths.batch;
  const auto& ra = state_.at("pin_rAAT");
  const auto& fa = state_.at("pin_fAAT");
  py::dict paths;
  std::size_t valid_count = 0, path_pins = 0;
  for (std::size_t id = 0; id < extracted.states.size(); ++id) {
    const auto& state = extracted.states[id];
    const auto& trace = extracted.traces[id];
    valid_count += trace.valid != 0;
    path_pins += trace.pins.size();
    const auto key = py::int_(state.pin);
    // States are sorted by slack/transition/pin/test; strict '<' in the
    // existing consumer retains the first state for this pin on equal slack.
    if (paths.contains(key)) continue;
    py::dict record;
    record["path"] = trace.pins;
    record["transitions"] = trace.transitions;
    record["arc_ids"] = trace.arc_ids;
    record["endpoint_transition"] = state.transition;
    record["endpoint_slack"] = state.slack;
    record["valid"] = trace.valid != 0;
    record["topology_epoch"] = epoch_;
    record["endpoint_test_id"] = state.test_id;
    record["invalid_reason"] = trace.invalid_reason;
    record["max_residual_ps"] = trace.max_residual_ps;
    std::vector<double> arrivals;
    arrivals.reserve(trace.pins.size());
    for (std::size_t pin = 0; pin < trace.pins.size(); ++pin) {
      arrivals.push_back((trace.transitions[pin] == 0 ? ra : fa)[trace.pins[pin]]);
    }
    record["path_arrivals"] = std::move(arrivals);
    paths[key] = std::move(record);
  }
  py::dict statistics;
  statistics["backend"] = "graph_owned_cpp_transition_aware";
  statistics["topology_epoch"] = epoch_;
  statistics["endpoint_record_count"] = result_paths.endpoint_record_count;
  statistics["failing_state_count"] = extracted.failing_state_count;
  statistics["selected_state_count"] = extracted.states.size();
  statistics["valid_path_count"] = valid_count;
  statistics["invalid_path_count"] = extracted.states.size() - valid_count;
  statistics["path_pin_count"] = path_pins;
  statistics["timing_state_values_copied"] = 0;
  statistics["extraction_runtime_ms"] = std::chrono::duration<double, std::milli>(
      std::chrono::steady_clock::now() - started).count();
  py::dict result;
  result["paths"] = std::move(paths);
  result["statistics"] = std::move(statistics);
  return result;
}

}  // namespace rebuffer
