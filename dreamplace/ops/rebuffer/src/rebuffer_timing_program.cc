// SPDX-License-Identifier: BSD-3-Clause
// Validate and own the original Python TP constraints, tables and schedule.
#include "rebuffer_timing_program.hh"
#include "rebuffer_timing_events.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <set>
#include <string>

#include "pybind11/numpy.h"
#include "pybind11/stl.h"

namespace py = pybind11;
namespace rebuffer {
namespace {

void require(bool condition, const std::string& message)
{
  if (!condition) throw py::value_error(message);
}

template <typename T>
py::array array(const py::handle& value, int ndim, const std::string& name)
{
  require(py::isinstance<py::array>(value), name + " must be a numpy array");
  auto result = py::reinterpret_borrow<py::array>(value);
  require(result.dtype().is(py::dtype::of<T>()), name + " has incorrect dtype");
  require(result.ndim() == ndim && (result.flags() & py::array::c_style),
          name + " has incorrect shape or layout");
  return result;
}

template <typename T>
std::vector<T> vector(const py::handle& value, const std::string& name)
{
  const auto input = array<T>(value, 1, name);
  if (input.size() == 0) return {};
  const auto* data = static_cast<const T*>(input.data());
  return {data, data + input.size()};
}

template <std::size_t Width>
std::vector<std::array<int64_t, Width>> rows(
    const py::handle& value, const std::string& name)
{
  const auto input = array<int64_t>(value, 2, name);
  require(input.shape(1) == Width, name + " has incorrect row width");
  std::vector<std::array<int64_t, Width>> result(input.shape(0));
  const auto* data = static_cast<const int64_t*>(input.data());
  for (std::size_t i = 0; i < result.size(); ++i) {
    std::copy_n(data + i * Width, Width, result[i].begin());
  }
  return result;
}

std::unique_ptr<bool[]> mask(const py::handle& value, std::size_t count,
                             const std::string& name)
{
  auto result = std::make_unique<bool[]>(count * 2);
  if (value.is_none()) {
    std::fill_n(result.get(), count * 2, true);
  } else {
    const auto input = array<bool>(value, 2, name);
    require(input.shape(0) == count && input.shape(1) == 2,
            name + " must have two states per row");
    std::copy_n(static_cast<const bool*>(input.data()), count * 2, result.get());
  }
  return result;
}

void offsets(const std::vector<int64_t>& values, std::size_t count,
             const std::string& name)
{
  require(values.size() >= 2 && values.front() == 0 && values.back() == count
              && std::is_sorted(values.begin(), values.end()),
          name + " must cover its rows in nondecreasing order");
}

void pin(int64_t value, int count, const std::string& name)
{
  require(value >= 0 && value < count, name + " pin id is out of range");
}

std::vector<int64_t> unique(std::vector<int64_t> values)
{
  std::sort(values.begin(), values.end());
  values.erase(std::unique(values.begin(), values.end()), values.end());
  return values;
}

ProgramLut lut(const py::handle& value)
{
  require(py::isinstance<py::list>(value), "LUT bundle must be a list");
  const auto fields = py::reinterpret_borrow<py::list>(value);
  require(fields.size() == 5, "LUT bundle must contain five arrays");
  ProgramLut result;
  const auto values = array<double>(fields[0], 2, "LUT values");
  const auto trans = array<double>(fields[1], 2, "LUT transitions");
  const auto caps = array<double>(fields[2], 2, "LUT capacitances");
  require(values.shape(0) <= std::numeric_limits<int>::max(), "too many LUT rows");
  result.rows = values.shape(0);
  require(trans.shape(0) == result.rows && caps.shape(0) == result.rows,
          "LUT table row counts must match");
  for (const auto& item : {values, trans, caps}) {
    require(item.shape(1) <= std::numeric_limits<int>::max(), "LUT stride overflow");
  }
  result.value_stride = values.shape(1);
  result.transition_stride = trans.shape(1);
  result.capacitance_stride = caps.shape(1);
  auto copy = [](const py::array& input) {
    if (input.size() == 0) return std::vector<double>{};
    const auto* data = static_cast<const double*>(input.data());
    return std::vector<double>(data, data + input.size());
  };
  result.values = copy(values);
  result.transitions = copy(trans);
  result.capacitances = copy(caps);
  result.transition_dims = vector<int64_t>(fields[3], "LUT transition dims");
  result.capacitance_dims = vector<int64_t>(fields[4], "LUT capacitance dims");
  require(result.transition_dims.size() == result.rows
              && result.capacitance_dims.size() == result.rows,
          "LUT dimension counts must match rows");
  for (int row = 0; row < result.rows; ++row) {
    const auto nt = result.transition_dims[row], nc = result.capacitance_dims[row];
    require(nt >= 0 && nc >= 0 && nt <= result.transition_stride
                && nc <= result.capacitance_stride,
            "LUT axis dimensions exceed storage");
    const auto needed = nt > 1 && nc > 1 ? nt * nc : std::max(nt, nc);
    require(needed <= result.value_stride, "LUT values exceed storage");
    for (const auto& axis : {std::make_pair(&result.transitions, nt),
                             std::make_pair(&result.capacitances, nc)}) {
      const int stride = axis.first == &result.transitions
                             ? result.transition_stride : result.capacitance_stride;
      for (int64_t i = 0; i < axis.second; ++i) {
        const auto x = (*axis.first)[row * stride + i];
        require(std::isfinite(x), "LUT axis must be finite");
        if (i) require(x >= (*axis.first)[row * stride + i - 1],
                       "LUT axis must be nondecreasing");
      }
    }
  }
  return result;
}

void lutRow(const ProgramCellArc& arc, const TimingPropagationProgram& program,
            int first, int last)
{
  for (int index = first; index < last; ++index) {
    const auto count = program.luts[index].rows;
    // The canonical LUT query supports Python negative row indexing.
    require(arc[3] >= -count && arc[3] < count, "LUT row id is out of range");
  }
}

}  // namespace

TimingLutTable<double> ProgramLut::view() const
{
  return {values.data(), transitions.data(), capacitances.data(),
          transition_dims.data(), capacitance_dims.data(), rows,
          value_stride, transition_stride, capacitance_stride};
}

std::shared_ptr<const TimingPropagationProgram> timingProgramFromPython(
    const py::dict& values, const TimingGraphIndex& graph)
{
  const std::set<std::string> keys{
      "schema_version", "timing_schema_version", "pin_names", "cell_arcs",
      "cell_level_start", "net_arcs", "net_arc_start", "pin_net", "net_drivers",
      "clock_pins", "clock_rtran", "clock_ftran", "start_pins", "start_rAAT",
      "start_fAAT", "start_rtran", "start_ftran", "endpoint_pins", "endpoint_rRAT",
      "endpoint_fRAT", "setup_arcs", "setup_valid", "check_arcs", "check_valid",
      "endpoint_valid", "backend_rSlew", "backend_fSlew", "aggregation_mode",
      "metric_scope", "recovery_uses_live_slew", "recovery_rat_is_capture_deadline",
      "pairwise_smooth", "pairwise_alpha",
      "slew_epsilon", "luts"};
  for (const auto& item : values) {
    require(keys.count(py::cast<std::string>(item.first)) != 0,
            "unknown propagation input field");
  }
  for (const auto& key : keys) {
    if (key == "recovery_rat_is_capture_deadline") continue;
    require(values.contains(py::str(key)), "missing propagation input field: " + key);
  }
  require(values["schema_version"].cast<int>() == 1
              && values["timing_schema_version"].cast<int>() == 2,
          "native propagation requires schema 2 timing inputs");
  require(values["aggregation_mode"].cast<std::string>() == "hard",
          "native propagation requires hard aggregation");
  const auto scope = values["metric_scope"].cast<std::string>();
  require(scope == "setup_only" || scope == "setup_plus_recovery",
          "unsupported timing metric scope");
  require(values["pin_names"].cast<std::vector<std::string>>() == graph.pins,
          "propagation pin namespace does not match graph");
  auto result = std::make_shared<TimingPropagationProgram>();
  auto& p = *result;
  p.pin_count = graph.pins.size();
  p.pairwise_smooth = values["pairwise_smooth"].cast<bool>();
  p.pairwise_alpha = values["pairwise_alpha"].cast<double>();
  p.slew_epsilon = values["slew_epsilon"].cast<double>();
  require(std::isfinite(p.pairwise_alpha) && p.pairwise_alpha > 0
              && std::isfinite(p.slew_epsilon) && p.slew_epsilon > 0,
          "pairwise alpha and slew epsilon must be finite and positive");
  const auto luts = values["luts"].cast<py::list>();
  require(luts.size() == p.luts.size(), "six LUT bundles are required");
  for (int i = 0; i < 6; ++i) p.luts[i] = lut(luts[i]);
  p.schedule.cells = rows<8>(values["cell_arcs"], "cell_arcs");
  p.schedule.arc_ids.resize(p.schedule.cells.size());
  std::iota(p.schedule.arc_ids.begin(), p.schedule.arc_ids.end(), 0);
  p.schedule.level_start = vector<int64_t>(values["cell_level_start"], "cell_level_start");
  offsets(p.schedule.level_start, p.schedule.cells.size(), "cell_level_start");
  require(p.schedule.cells.size() == graph.cells.size() && p.schedule.level_start[1] == graph.clock_end,
          "cell schedule does not match graph");
  for (std::size_t i = 0; i < p.schedule.cells.size(); ++i) {
    require(std::equal(p.schedule.cells[i].begin(), p.schedule.cells[i].end(), graph.cells[i].begin()),
            "cell arc does not match graph");
    pin(p.schedule.cells[i][0], i < graph.clock_end ? graph.clocks.size() : p.pin_count,
        i < graph.clock_end ? "clock-local" : "cell input");
    pin(p.schedule.cells[i][1], p.pin_count, "cell output");
    lutRow(p.schedule.cells[i], p, 0, 4);
  }
  const auto clocks = vector<int64_t>(values["clock_pins"], "clock_pins");
  require(clocks.size() == graph.clocks.size()
              && std::equal(clocks.begin(), clocks.end(), graph.clocks.begin()),
          "clock namespace does not match graph");
  p.clock_rtran = vector<double>(values["clock_rtran"], "clock_rtran");
  p.clock_ftran = vector<double>(values["clock_ftran"], "clock_ftran");
  require(p.clock_rtran.size() == clocks.size() && p.clock_ftran.size() == clocks.size(),
          "clock slew length does not match clock-local namespace");
  for (auto clock : clocks) pin(clock, p.pin_count, "physical clock");
  p.start_pins = vector<int64_t>(values["start_pins"], "start_pins");
  p.endpoint_pins = vector<int64_t>(values["endpoint_pins"], "endpoint_pins");
  for (const auto& pins : {p.start_pins, p.endpoint_pins}) {
    for (auto id : pins) pin(id, p.pin_count, "constraint");
    require(unique(pins).size() == pins.size(), "duplicate constraint pin");
  }
  auto constraint = [&](const char* name, std::size_t count) {
    auto data = vector<double>(values[name], name);
    require(data.size() == count, std::string(name) + " length does not match pins");
    return data;
  };
  p.start_rAAT = constraint("start_rAAT", p.start_pins.size());
  p.start_fAAT = constraint("start_fAAT", p.start_pins.size());
  p.start_rtran = constraint("start_rtran", p.start_pins.size());
  p.start_ftran = constraint("start_ftran", p.start_pins.size());
  p.endpoint_rRAT = constraint("endpoint_rRAT", p.endpoint_pins.size());
  p.endpoint_fRAT = constraint("endpoint_fRAT", p.endpoint_pins.size());
  p.endpoint_valid = mask(values["endpoint_valid"], p.pin_count, "endpoint_valid");

  const auto net_arcs = rows<2>(values["net_arcs"], "net_arcs");
  require(net_arcs.size() == graph.nets.size(), "net arcs do not match graph");
  for (std::size_t i = 0; i < net_arcs.size(); ++i) {
    require(std::equal(net_arcs[i].begin(), net_arcs[i].end(), graph.nets[i].begin()),
            "net arc does not match graph");
  }
  const auto starts = vector<int64_t>(values["net_arc_start"], "net_arc_start");
  offsets(starts, net_arcs.size(), "net_arc_start");
  auto pin_net = vector<int64_t>(values["pin_net"], "pin_net");
  const auto drivers = vector<int64_t>(values["net_drivers"], "net_drivers");
  require(drivers.size() + 1 == starts.size() && pin_net.size() == p.pin_count,
          "net mapping lengths do not match");
  for (auto& net : pin_net) {
    if (net < 0) net += drivers.size();
    require(net >= 0 && net < drivers.size(), "pin_net id is out of range");
  }
  for (auto driver : drivers) {
    require(driver == -1 || (driver >= 0 && driver < p.pin_count),
            "net driver is out of range");
  }
  auto forward = [&](const std::vector<int64_t>& pins) {
    std::vector<int64_t> nets;
    for (auto id : pins) nets.push_back(pin_net[id]);
    ProgramNetGroup group;
    for (auto net : unique(std::move(nets))) {
      group.arcs.insert(group.arcs.end(), net_arcs.begin() + starts[net],
                         net_arcs.begin() + starts[net + 1]);
    }
    return group;
  };
  auto reverse = [&](const std::vector<int64_t>& pins) {
    ProgramNetGroup group;
    for (auto id : pins) {
      const auto driver = drivers[pin_net[id]];
      if (driver == -1) continue;
      group.arcs.push_back({driver, id});
      group.drivers.push_back(driver);
    }
    group.drivers = unique(std::move(group.drivers));
    return group;
  };
  const auto levels = p.schedule.level_start.size() - 1;
  p.schedule.forward_nets.resize(levels);
  p.schedule.reverse_nets.resize(levels);
  p.schedule.level_inputs.resize(levels);
  p.schedule.forward_nets[0] = forward(p.start_pins);
  p.schedule.endpoint_nets = reverse(p.endpoint_pins);
  for (std::size_t level = 1; level < levels; ++level) {
    std::vector<int64_t> inputs, outputs;
    for (auto i = p.schedule.level_start[level]; i < p.schedule.level_start[level + 1]; ++i) {
      inputs.push_back(p.schedule.cells[i][0]);
      outputs.push_back(p.schedule.cells[i][1]);
    }
    p.schedule.level_inputs[level] = unique(std::move(inputs));
    p.schedule.forward_nets[level] = forward(outputs);
    p.schedule.reverse_nets[level] = reverse(p.schedule.level_inputs[level]);
  }

  p.setup_arcs = rows<8>(values["setup_arcs"], "setup_arcs");
  p.setup_valid = mask(values["setup_valid"], p.setup_arcs.size(), "setup_valid");
  std::set<int64_t> setup_pins, recovery_pins;
  for (const auto& arc : p.setup_arcs) {
    pin(arc[0], clocks.size(), "setup clock-local");
    pin(arc[1], p.pin_count, "setup data");
    lutRow(arc, p, 0, 2);
    p.setup_pins.push_back(arc[1]);
    setup_pins.insert(arc[1]);
  }
  p.setup_pins = unique(std::move(p.setup_pins));
  const auto checks = rows<8>(values["check_arcs"], "check_arcs");
  const auto valid = mask(values["check_valid"], checks.size(), "check_valid");
  std::vector<bool> recovery_valid;
  std::vector<double> clock_r(p.pin_count, 0), clock_f(p.pin_count, 0);
  for (std::size_t i = 0; i < clocks.size(); ++i) {
    clock_r[clocks[i]] = p.clock_rtran[i];
    clock_f[clocks[i]] = p.clock_ftran[i];
  }
  p.recovery_uses_live_slew = values["recovery_uses_live_slew"].cast<bool>()
                              || scope != "setup_plus_recovery";
  p.recovery_rat_is_capture_deadline = values.contains("recovery_rat_is_capture_deadline")
      && values["recovery_rat_is_capture_deadline"].cast<bool>();
  if (p.recovery_rat_is_capture_deadline) {
    p.recovery_deadline_r.assign(p.pin_count, 0);
    p.recovery_deadline_f.assign(p.pin_count, 0);
    p.recovery_keep_setup = std::make_unique<bool[]>(p.pin_count * 2);
    for (std::size_t i = 0; i < p.endpoint_pins.size(); ++i) {
      p.recovery_deadline_r[p.endpoint_pins[i]] = p.endpoint_rRAT[i];
      p.recovery_deadline_f[p.endpoint_pins[i]] = p.endpoint_fRAT[i];
    }
    for (std::size_t i = 0; i < p.setup_arcs.size(); ++i) {
      const auto pin = p.setup_arcs[i][1];
      for (int edge = 0; edge < 2; ++edge) {
        p.recovery_keep_setup[pin * 2 + edge] |= p.setup_valid[i * 2 + edge];
      }
    }
  }
  std::vector<double> reference_r(p.pin_count, 0), reference_f(p.pin_count, 0);
  const bool active_recovery = [&]() {
    for (std::size_t i = 0; i < checks.size(); ++i) {
      if (checks[i][6] == 3 && (valid[i * 2] || valid[i * 2 + 1])) return true;
    }
    return false;
  }();
  if (active_recovery && !p.recovery_uses_live_slew && !p.recovery_rat_is_capture_deadline) {
    const auto rise = constraint("backend_rSlew", p.endpoint_pins.size());
    const auto fall = constraint("backend_fSlew", p.endpoint_pins.size());
    for (std::size_t i = 0; i < p.endpoint_pins.size(); ++i) {
      reference_r[p.endpoint_pins[i]] = rise[i];
      reference_f[p.endpoint_pins[i]] = fall[i];
    }
  }
  for (std::size_t i = 0; i < checks.size(); ++i) {
    auto arc = checks[i];
    pin(arc[1], p.pin_count, "check data");
    p.timing_check_pins.push_back(arc[1]);
    if (arc[6] == 1) setup_pins.insert(arc[1]);
    if (arc[6] != 3 || !(valid[i * 2] || valid[i * 2 + 1])) continue;
    pin(arc[0], p.pin_count, "recovery physical clock");
    lutRow(arc, p, 4, 6);
    recovery_pins.insert(arc[1]);
    const auto clock_slew = arc[5] < 0 ? clock_f[arc[0]] : clock_r[arc[0]];
    p.recovery_clock_rtran.push_back(clock_slew);
    p.recovery_clock_ftran.push_back(clock_slew);
    p.recovery_reference_rtran.push_back(reference_r[arc[1]]);
    p.recovery_reference_ftran.push_back(reference_f[arc[1]]);
    if (valid[i * 2]) p.recovery_r_pins.push_back(arc[1]);
    if (valid[i * 2 + 1]) p.recovery_f_pins.push_back(arc[1]);
    arc[0] = p.recovery_arcs.size();  // per-row selected clock slew
    p.recovery_arcs.push_back(arc);
    recovery_valid.push_back(valid[i * 2]);
    recovery_valid.push_back(valid[i * 2 + 1]);
  }
  p.recovery_r_pins = unique(std::move(p.recovery_r_pins));
  p.recovery_f_pins = unique(std::move(p.recovery_f_pins));
  p.timing_check_pins = unique(std::move(p.timing_check_pins));
  p.recovery_valid = std::make_unique<bool[]>(recovery_valid.size());
  std::copy(recovery_valid.begin(), recovery_valid.end(), p.recovery_valid.get());
  for (auto id : p.endpoint_pins) {
    if (scope == "setup_plus_recovery" || !recovery_pins.count(id)
        || setup_pins.count(id)) p.metric_pins.push_back(id);
  }
  p.start_row_by_pin.assign(p.pin_count, -1);
  p.endpoint_row_by_pin.assign(p.pin_count, -1);
  p.setup_rows_by_pin.resize(p.pin_count);
  p.recovery_rows_by_pin.resize(p.pin_count);
  for (int row = 0; row < static_cast<int>(p.start_pins.size()); ++row) {
    p.start_row_by_pin[p.start_pins[row]] = row;
  }
  for (int row = 0; row < static_cast<int>(p.endpoint_pins.size()); ++row) {
    p.endpoint_row_by_pin[p.endpoint_pins[row]] = row;
  }
  for (int row = 0; row < static_cast<int>(p.setup_arcs.size()); ++row) {
    p.setup_rows_by_pin[p.setup_arcs[row][1]].push_back(row);
  }
  for (int row = 0; row < static_cast<int>(p.recovery_arcs.size()); ++row) {
    p.recovery_rows_by_pin[p.recovery_arcs[row][1]].push_back(row);
  }
  buildTimingEventSchedule(p.schedule, graph);
  return result;
}

}  // namespace rebuffer
