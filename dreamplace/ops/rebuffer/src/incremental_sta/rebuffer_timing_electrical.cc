// SPDX-License-Identifier: BSD-3-Clause
// Candidate acceptance reads the graph directly and shares the physical ERC.
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_mutation.hh"
#include "rebuffer_timing_buffer_state.hh"
#include "rebuffer_timing_events.hh"
#include "rebuffer_electrical_observation.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <set>

#include "pybind11/stl.h"

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
