// SPDX-License-Identifier: BSD-3-Clause
// One observation contract for full and event propagation on graph-owned state.
#include "rebuffer_timing_graph.hh"

#include "rebuffer_timing_mutation.hh"


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

// --- rebuffer_timing_electrical.cc ---
// Candidate acceptance reads the graph directly and shares the physical ERC.

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <set>


namespace py = pybind11;
namespace rebuffer {
namespace {

constexpr std::array<const char*, 3> kElectricalFields{"slew_ps", "cap_pf", "fanout"};
constexpr std::array<double, 3> kElectricalTolerances{1.0e-6, 1.0e-9, 0.0};

}  // namespace

struct ElectricalObservationCache
{
  bool full_dirty = true;
  int max_fanout = -1;
  double cap_margin = -1.0;
  double slew_margin = -1.0;
  struct PinState
  {
    unsigned char active = 0;
    unsigned char nonfinite = 0;
    unsigned char violates = 0;
    std::array<unsigned char, 3> checked{0, 0, 0};
    std::array<unsigned char, 3> violation{0, 0, 0};
    std::array<double, 3> amounts{0.0, 0.0, 0.0};
    py::str key;
    py::dict row;
  };
  std::vector<PinState> pins;
  std::vector<unsigned char> base_clock_pin;
  std::vector<int> dirty_pins;
  std::vector<unsigned char> dirty_mask;
  std::size_t active_pin_count = 0;
  std::array<int, 3> checked_count{0, 0, 0};
  std::array<int, 3> violation_count{0, 0, 0};
  std::set<int> violating_pins;
  std::set<int> nonfinite_pins;
  py::dict violations_by_pin;
};

void PersistentTimingGraph::markElectricalFullDirty() const
{
  if (electrical_cache_) {
    electrical_cache_->full_dirty = true;
    electrical_cache_->dirty_pins.clear();
    electrical_cache_->dirty_mask.clear();
  }
}

void PersistentTimingGraph::markElectricalDirtyPin(const int pin) const
{
  if (!electrical_cache_ || electrical_cache_->full_dirty || pin < 0) return;
  auto& cache = *electrical_cache_;
  if (pin >= static_cast<int>(cache.dirty_mask.size())) {
    cache.dirty_mask.resize(pin + 1, 0);
  }
  if (!cache.dirty_mask[pin]) {
    cache.dirty_mask[pin] = 1;
    cache.dirty_pins.push_back(pin);
  }
}

py::dict PersistentTimingGraph::evaluationObservation(
    int max_fanout, double cap_margin, double slew_margin) const
{
  if (!mutation_state_ || !propagation_program_ || !timing_state_initialized_) {
    throw py::value_error("evaluation observation requires initialized native timing");
  }
  const auto seeds = timingEventSeeds();
  if (seeds.require_full || !seeds.forward.empty() || !seeds.reverse.empty()
      || !seeds.retired_arcs.empty()) {
    throw py::value_error("evaluation observation requires current native propagation");
  }
  const auto& m = *mutation_state_;
  requireSynchronizedTimingTopologies(m, net_states_);
  if (!std::isfinite(cap_margin) || !std::isfinite(slew_margin)
      || cap_margin < 0.0 || cap_margin >= 100.0
      || slew_margin < 0.0 || slew_margin >= 100.0) {
    throw py::value_error("electrical margins must be in [0, 100)");
  }

  if (!electrical_cache_) {
    electrical_cache_ = std::make_shared<ElectricalObservationCache>();
  }
  auto& cache = *electrical_cache_;
  const auto& names = timing_index_->pins;
  const auto& rise = state_.at("pin_rtran");
  const auto& fall = state_.at("pin_ftran");
  const auto& rise_load = state_.at("pin_net_cap_rise");
  const auto& fall_load = state_.at("pin_net_cap_fall");
  const std::size_t pin_count = names.size();

  if (cache.base_clock_pin.empty() && !names.empty()) {
    cache.base_clock_pin.assign(names.size(), 0);
    for (const auto pin : timing_index_->clocks) {
      if (pin >= 0 && static_cast<std::size_t>(pin) < cache.base_clock_pin.size()) {
        cache.base_clock_pin[pin] = 1;
      }
    }
  }

  const bool need_full = cache.full_dirty
      || cache.max_fanout != max_fanout
      || cache.cap_margin != cap_margin
      || cache.slew_margin != slew_margin;

  std::vector<unsigned char> clock_net;
  auto ensure_clock_net = [&]() {
    if (!clock_net.empty()) return;
    clock_net.assign(m.net_names.size(), 0);
    std::copy(m.base_net_is_clock.begin(), m.base_net_is_clock.end(), clock_net.begin());
    for (const auto& buffer : m.buffers) {
      clock_net.at(buffer.output_net) = clock_net.at(buffer.source_net);
    }
  };

  auto remove_pin_contribution = [&](int pin) {
    auto& entry = cache.pins[pin];
    if (entry.active) --cache.active_pin_count;
    if (entry.nonfinite) cache.nonfinite_pins.erase(pin);
    for (int f = 0; f < 3; ++f) {
      cache.checked_count[f] -= entry.checked[f];
      cache.violation_count[f] -= entry.violation[f];
    }
  };

  auto update_pin = [&](int pin) {
    auto& entry = cache.pins[pin];
    const bool was_violating = entry.violates != 0;
    const auto old_amounts = entry.amounts;
    remove_pin_contribution(pin);
    entry.active = 0;
    entry.nonfinite = 0;
    entry.violates = 0;
    entry.checked = {0, 0, 0};
    entry.violation = {0, 0, 0};
    entry.amounts = {0.0, 0.0, 0.0};

    if (!m.pin_active[pin]) {
      if (was_violating) {
        cache.violating_pins.erase(pin);
        cache.violations_by_pin.attr("pop")(entry.key);
      }
      return;
    }
    entry.active = 1;
    ++cache.active_pin_count;
    const int net = m.pin_net[pin];
    const bool driver = net >= 0 && m.net_drivers[net] == pin;
    const int fanout = driver ? static_cast<int>(m.net_pins[net].size()) - 1 : 0;
    const double max_slew = m.pin_values[3][pin];
    const double max_cap = m.pin_values[4][pin];
    const double slew_r = rise[pin], slew_f = fall[pin];
    const double load_r = rise_load[pin], load_f = fall_load[pin];
    if ((max_slew > 0.0 && std::isfinite(max_slew)
         && (!std::isfinite(slew_r) || !std::isfinite(slew_f)))
        || (driver && max_cap > 0.0 && std::isfinite(max_cap)
            && (!std::isfinite(load_r) || !std::isfinite(load_f)))) {
      entry.nonfinite = 1;
      cache.nonfinite_pins.insert(pin);
      if (was_violating) {
        cache.violating_pins.erase(pin);
        cache.violations_by_pin.attr("pop")(entry.key);
      }
      return;
    }
    const bool is_clock =
        (pin < static_cast<int>(cache.base_clock_pin.size()) && cache.base_clock_pin[pin] != 0)
        || (net >= 0 && clock_net[net] != 0);
    if (is_clock) {
      if (was_violating) {
        cache.violating_pins.erase(pin);
        cache.violations_by_pin.attr("pop")(entry.key);
      }
      return;
    }
    const std::array<double, 3> limit{
        max_slew, driver ? max_cap : 0.0, driver ? static_cast<double>(max_fanout) : 0.0};
    const std::array<double, 3> observed{
        std::max(slew_r, slew_f), std::max(load_r, load_f), static_cast<double>(fanout)};
    const std::array<double, 3> margin{slew_margin, cap_margin, 0.0};
    for (int f = 0; f < 3; ++f) {
      if (!(limit[f] > 0.0) || !std::isfinite(limit[f])) continue;
      if (!std::isfinite(observed[f])) {
        entry.nonfinite = 1;
        cache.nonfinite_pins.insert(pin);
        if (was_violating) {
          cache.violating_pins.erase(pin);
          cache.violations_by_pin.attr("pop")(entry.key);
        }
        return;
      }
      const double amount =
          std::max(0.0, observed[f] - limit[f] * (1.0 - margin[f] / 100.0));
      entry.amounts[f] = amount;
      entry.checked[f] = 1;
      ++cache.checked_count[f];
      if (amount > kElectricalTolerances[f]) {
        entry.violation[f] = 1;
        ++cache.violation_count[f];
        entry.violates = 1;
      }
    }
    if (entry.violates) {
      if (!was_violating) {
        cache.violating_pins.insert(pin);
        if (!entry.key.ptr() || PyUnicode_GET_LENGTH(entry.key.ptr()) == 0) {
          std::string norm = names[pin];
          std::replace(norm.begin(), norm.end(), '/', ':');
          entry.key = py::str(norm);
        }
        if (cache.violations_by_pin.contains(entry.key)) {
          throw py::value_error("ambiguous electrical pin name");
        }
      }
      if (!was_violating || !entry.row.ptr()
          || std::memcmp(entry.amounts.data(), old_amounts.data(), sizeof(old_amounts)) != 0) {
        py::dict row;
        for (int f = 0; f < 3; ++f) row[kElectricalFields[f]] = entry.amounts[f];
        entry.row = std::move(row);
        cache.violations_by_pin[entry.key] = entry.row;
      }
    } else if (was_violating) {
      cache.violating_pins.erase(pin);
      cache.violations_by_pin.attr("pop")(entry.key);
    }
  };

  if (need_full) {
    ensure_clock_net();
    cache.pins.clear();
    cache.pins.resize(pin_count);
    cache.dirty_pins.clear();
    cache.dirty_mask.assign(pin_count, 0);
    cache.active_pin_count = 0;
    cache.checked_count = {0, 0, 0};
    cache.violation_count = {0, 0, 0};
    cache.violating_pins.clear();
    cache.nonfinite_pins.clear();
    cache.violations_by_pin = py::dict();
    cache.max_fanout = max_fanout;
    cache.cap_margin = cap_margin;
    cache.slew_margin = slew_margin;
    cache.full_dirty = false;
    for (std::size_t pin = 0; pin < pin_count; ++pin) {
      update_pin(static_cast<int>(pin));
    }
  } else {
    if (cache.pins.size() > pin_count) {
      for (std::size_t pin = pin_count; pin < cache.pins.size(); ++pin) {
        const bool was_violating = cache.pins[pin].violates != 0;
        remove_pin_contribution(static_cast<int>(pin));
        if (was_violating) {
          cache.violating_pins.erase(static_cast<int>(pin));
          cache.violations_by_pin.attr("pop")(cache.pins[pin].key);
        }
      }
      cache.pins.resize(pin_count);
    } else if (cache.pins.size() < pin_count) {
      const std::size_t old_size = cache.pins.size();
      cache.pins.resize(pin_count);
      for (std::size_t pin = old_size; pin < pin_count; ++pin) {
        markElectricalDirtyPin(static_cast<int>(pin));
      }
    }
    if (!cache.dirty_pins.empty()) {
      ensure_clock_net();
      for (const int pin : cache.dirty_pins) {
        if (pin >= 0 && static_cast<std::size_t>(pin) < cache.dirty_mask.size()) {
          cache.dirty_mask[pin] = 0;
        }
        if (pin >= 0 && static_cast<std::size_t>(pin) < pin_count) {
          update_pin(pin);
        }
      }
      cache.dirty_pins.clear();
    }
  }

  if (!cache.nonfinite_pins.empty()) {
    throw py::value_error("nonfinite electrical transition/load observation");
  }

  std::array<double, 3> total{0.0, 0.0, 0.0};
  for (const int pin : cache.violating_pins) {
    const auto& entry = cache.pins[pin];
    for (int f = 0; f < 3; ++f) {
      if (entry.violation[f]) total[f] += entry.amounts[f];
    }
  }

  py::dict electrical;
  electrical["violations_by_pin"] = cache.violations_by_pin.attr("copy")().cast<py::dict>();
  electrical["pin_count"] = cache.active_pin_count;
  electrical["configured_max_fanout"] = max_fanout;
  electrical["cap_margin_percent"] = cap_margin;
  electrical["slew_margin_percent"] = slew_margin;
  for (int f = 0; f < 3; ++f) {
    const std::string key = kElectricalFields[f];
    electrical[(key + "_checked_count").c_str()] = cache.checked_count[f];
    electrical[(key + "_violation_count").c_str()] = cache.violation_count[f];
    electrical[(key + "_total_excess").c_str()] = total[f];
  }

  const auto slacks = endpointSlacks();
  const auto& p = *propagation_program_;
  if (cached_endpoint_pin_names_.size() != p.metric_pins.size()) {
    cached_endpoint_pin_names_.clear();
    cached_endpoint_pin_names_.reserve(p.metric_pins.size());
    for (const auto pin : p.metric_pins) {
      cached_endpoint_pin_names_.emplace_back(names.at(pin));
    }
    cached_endpoint_slacks_by_name_ = py::dict();
    cached_endpoint_slacks_by_name_values_.clear();
  }
  if (cached_endpoint_slacks_by_name_values_.size() != slacks.size()) {
    cached_endpoint_slacks_by_name_ = py::dict();
    for (std::size_t row = 0; row < slacks.size(); ++row) {
      cached_endpoint_slacks_by_name_[cached_endpoint_pin_names_[row]] = slacks[row];
    }
    cached_endpoint_slacks_by_name_values_ = slacks;
  } else {
    for (std::size_t row = 0; row < slacks.size(); ++row) {
      if (std::memcmp(&slacks[row], &cached_endpoint_slacks_by_name_values_[row], sizeof(double)) != 0) {
        cached_endpoint_slacks_by_name_[cached_endpoint_pin_names_[row]] = slacks[row];
        cached_endpoint_slacks_by_name_values_[row] = slacks[row];
      }
    }
  }

  py::dict result;
  result["electrical"] = std::move(electrical);
  result["endpoint_slacks_by_name"] =
      cached_endpoint_slacks_by_name_.attr("copy")().cast<py::dict>();
  return result;
}

}  // namespace rebuffer

// --- rebuffer_electrical_observation.cc ---
// Check TP observations outside VGDP; no Liberty LUT evaluation belongs here.
#include "rebuffer_carrier.hh"


#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <string>
#include <vector>

namespace py = pybind11;
namespace rebuffer {
namespace {

constexpr std::array<const char*, 3> kFields{"slew_ps", "cap_pf", "fanout"};
constexpr std::array<double, 3> kTolerances{1.0e-6, 1.0e-9, 0.0};

std::vector<double> values(const py::dict& timing, const char* key, int count)
{
  const auto array = timing[key].cast<std::vector<double>>();
  if (static_cast<int>(array.size()) < count) {
    throw py::value_error(std::string("electrical observation size mismatch: ") + key);
  }
  return array;
}

double excess(double value, double limit, double margin)
{
  // PyDB exports zero/nonpositive as no limit; never turn it into a 0-unit
  // constraint. Counts below distinguish checked and unavailable limits.
  if (!(limit > 0.0) || !std::isfinite(limit)) return 0.0;
  if (!std::isfinite(value)) throw py::value_error("nonfinite electrical observation");
  return std::max(0.0, value - limit * (1.0 - margin / 100.0));
}

py::dict electricalSnapshot(const py::object& db, const py::dict& timing,
                             int max_fanout, double cap_margin, double slew_margin)
{
  const auto names = carrierArray<std::string>(db, "pin_names");
  const int count = names.size();
  const auto drivers = carrierArray<int>(db, "net2driver_pin_map");
  const auto starts = carrierArray<int>(db, "flat_net2pin_start_map");
  const auto pins = carrierArray<int>(db, "flat_net2pin_map");
  if (starts.size() != drivers.size() + 1 || starts.empty() || starts.front() != 0
      || starts.back() != static_cast<int>(pins.size())) {
    throw py::value_error("electrical observation net CSR mismatch");
  }
  const auto rise = values(timing, "pin_slew_rise", count);
  const auto fall = values(timing, "pin_slew_fall", count);
  const auto rise_load = values(timing, "pin_load_rise", count);
  const auto fall_load = values(timing, "pin_load_fall", count);
  const auto max_slew = values(timing, "pin_max_slew", count);
  const auto max_cap = values(timing, "pin_max_cap", count);
  std::vector<bool> clock(count, false), driver(count, false);
  std::vector<int> fanout(count, 0);
  if (py::hasattr(db, "clock_pins")) {
    for (const int pin : carrierArray<int>(db, "clock_pins")) {
      if (pin >= 0 && pin < count) clock[pin] = true;
    }
  }
  const auto net_names = carrierArray<std::string>(db, "net_names");
  std::set<std::string> clock_nets;
  if (py::hasattr(db, "clock_net_names")) {
    const auto clocks = carrierArray<std::string>(db, "clock_net_names");
    clock_nets.insert(clocks.begin(), clocks.end());
  }
  for (int net = 0; net < static_cast<int>(drivers.size()); ++net) {
    const int pin = drivers[net];
    if (pin < 0) continue;
    if (pin >= count || starts[net] < 0 || starts[net + 1] < starts[net]
        || starts[net + 1] > static_cast<int>(pins.size())) {
      throw py::value_error("electrical observation driver/CSR out of range");
    }
    // Match the planner's clock-net rule: a net is a clock net when its driver
    // is a clock pin or any pin on it is one.  Name-based matching alone missed
    // the pre-CTS clock buffer, whose driver then reported the whole clock load
    // as a cap/fanout violation that repair_design never repairs.
    bool net_is_clock = clock_nets.count(net_names.at(net)) != 0;
    if (!net_is_clock) {
      net_is_clock = clock[pin];
      for (int index = starts[net]; !net_is_clock && index < starts[net + 1]; ++index) {
        const int sink = pins[index];
        if (sink != pin && sink >= 0 && sink < count && clock[sink]) {
          net_is_clock = true;
        }
      }
    }
    if (net_is_clock) clock[pin] = true;
    driver[pin] = true;
    for (int index = starts[net]; index < starts[net + 1]; ++index) {
      if (pins[index] < 0 || pins[index] >= count) throw py::value_error("electrical net pin out of range");
      if (pins[index] != pin) ++fanout[pin];
      if (net_is_clock) clock[pins[index]] = true;
    }
  }
  std::vector<ElectricalPinObservation> observations;
  observations.reserve(count);
  for (int pin = 0; pin < count; ++pin) {
    observations.push_back({names[pin], rise[pin], fall[pin], rise_load[pin], fall_load[pin],
                            max_slew[pin], max_cap[pin], fanout[pin], driver[pin], clock[pin]});
  }
  return observeElectricalPins(observations, max_fanout, cap_margin, slew_margin);
}

}  // namespace

py::dict observeElectricalPins(const std::vector<ElectricalPinObservation>& pins,
                               int max_fanout, double cap_margin, double slew_margin)
{
  if (!std::isfinite(cap_margin) || !std::isfinite(slew_margin)
      || cap_margin < 0.0 || cap_margin >= 100.0 || slew_margin < 0.0 || slew_margin >= 100.0) {
    throw py::value_error("electrical margins must be in [0, 100)");
  }
  py::dict violations;
  std::array<int, 3> checked{0, 0, 0}, violation_count{0, 0, 0};
  std::array<double, 3> total{0.0, 0.0, 0.0};
  for (const auto& pin : pins) {
    if ((pin.max_slew > 0.0 && std::isfinite(pin.max_slew)
         && (!std::isfinite(pin.slew_rise) || !std::isfinite(pin.slew_fall)))
        || (pin.driver && pin.max_cap > 0.0 && std::isfinite(pin.max_cap)
            && (!std::isfinite(pin.load_rise) || !std::isfinite(pin.load_fall)))) {
      throw py::value_error("nonfinite electrical transition/load observation");
    }
    if (pin.clock) continue;
    const std::array<double, 3> limit{pin.max_slew, pin.driver ? pin.max_cap : 0.0,
                                    pin.driver ? static_cast<double>(max_fanout) : 0.0};
    const std::array<double, 3> observed{std::max(pin.slew_rise, pin.slew_fall),
        std::max(pin.load_rise, pin.load_fall), static_cast<double>(pin.fanout)};
    const std::array<double, 3> margin{slew_margin, cap_margin, 0.0};
    std::array<double, 3> amounts;
    bool violates = false;
    for (int field = 0; field < 3; ++field) {
      const double amount = excess(observed[field], limit[field], margin[field]);
      amounts[field] = amount;
      if (limit[field] > 0.0 && std::isfinite(limit[field])) ++checked[field];
      if (amount > kTolerances[field]) {
        ++violation_count[field];
        total[field] += amount;
        violates = true;
      }
    }
    if (violates) {
      py::dict row;
      for (int field = 0; field < 3; ++field) row[kFields[field]] = amounts[field];
      std::string name(pin.name);
      std::replace(name.begin(), name.end(), '/', ':');
      if (violations.contains(name.c_str())) throw py::value_error("ambiguous electrical pin name");
      violations[name.c_str()] = row;
    }
  }
  py::dict result;
  result["violations_by_pin"] = violations;
  result["pin_count"] = pins.size();
  result["configured_max_fanout"] = max_fanout;
  result["cap_margin_percent"] = cap_margin;
  result["slew_margin_percent"] = slew_margin;
  for (int field = 0; field < 3; ++field) {
    const std::string key = kFields[field];
    result[(key + "_checked_count").c_str()] = checked[field];
    result[(key + "_violation_count").c_str()] = violation_count[field];
    result[(key + "_total_excess").c_str()] = total[field];
  }
  return result;
}

py::dict compareElectricalObservations(const py::dict& before, const py::dict& after)
{
  py::dict result;
  py::list worsened;
  if (!before.contains("electrical") && !after.contains("electrical")) {
    result["accepted"] = true;
    result["checked"] = false;
    result["worsened_pins"] = worsened;
    return result;
  }
  result["checked"] = true;
  if (!before.contains("electrical") || !after.contains("electrical")) {
    result["accepted"] = false;
    result["reason"] = "electrical_observation_missing";
    result["worsened_pins"] = worsened;
    return result;
  }
  const auto previous = before["electrical"].cast<py::dict>()["violations_by_pin"].cast<py::dict>();
  const auto current = after["electrical"].cast<py::dict>()["violations_by_pin"].cast<py::dict>();
  for (const auto& pin : current) {
    PyObject* prev_obj = PyDict_GetItem(previous.ptr(), pin.first.ptr());
    if (prev_obj != nullptr && prev_obj == pin.second.ptr()) {
      continue;
    }
    const auto row = pin.second.cast<py::dict>();
    const auto old = previous.contains(pin.first) ? previous[pin.first].cast<py::dict>() : py::dict();
    for (int field = 0; field < 3; ++field) {
      const double value = py::cast<double>(row[kFields[field]]);
      const double prior = old.contains(kFields[field]) ? py::cast<double>(old[kFields[field]]) : 0.0;
      if (!std::isfinite(value) || value > prior + kTolerances[field]) {
        py::dict violation;
        violation["pin_name"] = pin.first;
        violation["check"] = kFields[field];
        violation["before_excess"] = prior;
        violation["after_excess"] = value;
        worsened.append(std::move(violation));
      }
    }
  }
  result["accepted"] = worsened.empty();
  result["worsened_pins"] = worsened;
  return result;
}

bool electricalNonRegressing(const py::dict& before, const py::dict& after)
{
  return py::cast<bool>(compareElectricalObservations(before, after)["accepted"]);
}

void registerElectricalObservationBindings(py::module_& module)
{
  module.def("electrical_snapshot", &electricalSnapshot, py::arg("pydb"), py::arg("timing"),
             py::arg("max_fanout") = 32, py::arg("cap_margin") = 0.0, py::arg("slew_margin") = 0.0);
  module.def("compare_electrical_observations", &compareElectricalObservations);
}

}  // namespace rebuffer
