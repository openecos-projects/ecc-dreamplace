// SPDX-License-Identifier: BSD-3-Clause
// One observation contract for full and event propagation on graph-owned state.
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_program.hh"
#include "rebuffer_timing_state.hh"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <utility>

#include "timing_hard_math.h"
#include "pybind11/stl.h"

namespace py = pybind11;
namespace rebuffer {
namespace {
bool same(double left, double right)
{
  return std::memcmp(&left, &right, sizeof(double)) == 0;
}
double qualifiedSlack(const TimingPropagationProgram& p, double rise_rat,
                       double fall_rat, double rise_aat, double fall_aat, int64_t id)
{
  const double inf = std::numeric_limits<double>::infinity();
  return dreamplace::timing::hardMinimum(
      p.endpoint_valid[id * 2] ? rise_rat - rise_aat : inf,
      p.endpoint_valid[id * 2 + 1] ? fall_rat - fall_aat : inf);
}

std::pair<py::dict, std::vector<double>> observe(const TimingPropagationProgram& p,
    const std::map<std::string, std::vector<double>>& output,
    const std::vector<double>& cap_r, std::vector<int64_t> changed_pins,
    py::dict coverage, const char* mode, const std::vector<double>& previous_slacks)
{
  const auto& ra = output.at("pin_rAAT");
  const auto& fa = output.at("pin_fAAT");
  const auto& rr = output.at("pin_rRAT");
  const auto& fr = output.at("pin_fRAT");
  std::vector<double> slacks;
  std::vector<int64_t> changed_endpoints;
  const double inf = std::numeric_limits<double>::infinity();
  double wns = 0, tns = 0, worst = inf;
  int qualified = 0, violating = 0;
  std::size_t endpoint = 0;
  for (auto id : p.metric_pins) {
    const double value = qualifiedSlack(p, rr[id], fr[id], ra[id], fa[id], id);
    const double old = previous_slacks[endpoint++];
    if (!same(value, old)) changed_endpoints.push_back(id);
    slacks.push_back(value);
    worst = dreamplace::timing::hardMinimum(worst, value);
    const double negative = dreamplace::timing::hardMinimum(value, 0.0);
    wns = dreamplace::timing::hardMinimum(wns, negative);
    tns += negative;
    qualified += std::isfinite(value);
    violating += value < 0;
  }
  py::dict metrics;
  // Preserve TP's differentiable zero anchor, including nonfinite RC input.
  const double zero = std::accumulate(cap_r.begin(), cap_r.end(), 0.0) * 0.0;
  metrics["wns_ps"] = wns + zero;
  metrics["tns_ps"] = tns + zero;
  metrics["worst_slack_ps"] = std::isfinite(worst) ? worst : zero;
  metrics["endpoint_count"] = qualified;
  metrics["violating_endpoint_count"] = violating;
  py::dict result;
  result["status"] = "ok";
  result["mode"] = mode;
  result["metrics"] = std::move(metrics);
  result["coverage"] = std::move(coverage);
  result["endpoint_pins"] = p.metric_pins;
  result["changed_pins"] = std::move(changed_pins);
  result["changed_endpoints"] = std::move(changed_endpoints);
  return {std::move(result), std::move(slacks)};
}
}  // namespace

std::vector<double> PersistentTimingGraph::endpointSlacks() const
{
  if (cached_endpoint_slacks_valid_) {
    return cached_endpoint_slacks_;
  }
  const auto& p = *propagation_program_;
  const auto& ra = state_.at("pin_rAAT");
  const auto& fa = state_.at("pin_fAAT");
  const auto& rr = state_.at("pin_rRAT");
  const auto& fr = state_.at("pin_fRAT");
  std::vector<double> slacks;
  slacks.reserve(p.metric_pins.size());
  for (auto id : p.metric_pins) {
    slacks.push_back(qualifiedSlack(p, rr[id], fr[id], ra[id], fa[id], id));
  }
  cached_endpoint_slacks_ = slacks;
  cached_endpoint_slacks_valid_ = true;
  return slacks;
}

py::dict PersistentTimingGraph::finishPropagation(
    std::map<std::string, std::vector<double>> output, py::dict coverage, const char* mode)
{
  const int pins = timing_index_->pins.size();
  std::vector<int64_t> changed_pins;
  std::vector<unsigned char> changed(pins, 0);
  for (const auto* field : {"pin_rAAT", "pin_fAAT", "pin_rtran", "pin_ftran",
                            "pin_rRAT", "pin_fRAT"}) {
    const auto& current = output.at(field);
    const auto& previous = state_.at(field);
    for (int id = 0; id < pins; ++id) {
      changed[id] |= !same(current[id], previous[id]);
    }
  }
  for (int id = 0; id < pins; ++id) if (changed[id]) changed_pins.push_back(id);
  auto [result, slacks] = observe(*propagation_program_, output, state_.at("pin_net_cap_rise"),
      std::move(changed_pins), std::move(coverage), mode, endpointSlacks());
  result["endpoint_slacks"] = writeState(slacks);
  result["state"] = applyStateValues(std::move(output));
  cached_endpoint_slacks_ = std::move(slacks);
  cached_endpoint_slacks_valid_ = true;
  journal_.back().propagation_undo = propagation_mark_;
  propagation_mark_ = {true, journal_.back().sequence};
  return result;
}

py::dict PersistentTimingGraph::finishSparsePropagation(
    const std::shared_ptr<TimingStateUndo>& undo, py::dict coverage,
    const std::vector<double>& previous_slacks, std::vector<int64_t> changed)
{
  // Each pin is visited once, and retired arcs are absent from active input
  // rows. The worklist therefore guarantees unique field/index captures.
  compactTimingUndo(*undo, state_, true);
  std::sort(changed.begin(), changed.end());
  for (const int64_t pin : changed) markElectricalDirtyPin(static_cast<int>(pin));
  auto [result, slacks] = observe(*propagation_program_, state_, state_.at("pin_net_cap_rise"),
      std::move(changed), std::move(coverage), "events",
      previous_slacks);
  result["endpoint_slacks"] = writeState(slacks);
  cached_endpoint_slacks_ = std::move(slacks);
  cached_endpoint_slacks_valid_ = true;
  result["state"] = summary();
  journal_.back().propagation_undo = propagation_mark_;
  propagation_mark_ = {true, journal_.back().sequence};
  return result;
}

}  // namespace rebuffer
