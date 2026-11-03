// SPDX-License-Identifier: BSD-3-Clause
// Check TP observations outside VGDP; no Liberty LUT evaluation belongs here.
#include "rebuffer_electrical_observation.hh"
#include "rebuffer_carrier.hh"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>
#include <string>
#include <vector>
#include "pybind11/stl.h"

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
    driver[pin] = true;
    for (int index = starts[net]; index < starts[net + 1]; ++index) {
      if (pins[index] < 0 || pins[index] >= count) throw py::value_error("electrical net pin out of range");
      if (pins[index] != pin) ++fanout[pin];
      if (clock_nets.count(net_names.at(net))) clock[pins[index]] = true;
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
    py::dict row;
    bool violates = false;
    for (int field = 0; field < 3; ++field) {
      const double amount = excess(observed[field], limit[field], margin[field]);
      row[kFields[field]] = amount;
      if (limit[field] > 0.0 && std::isfinite(limit[field])) ++checked[field];
      if (amount > kTolerances[field]) {
        ++violation_count[field];
        total[field] += amount;
        violates = true;
      }
    }
    if (violates) {
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
