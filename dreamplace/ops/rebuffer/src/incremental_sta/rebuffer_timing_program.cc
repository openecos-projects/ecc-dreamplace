// SPDX-License-Identifier: BSD-3-Clause
// Validate and own the original Python TP constraints, tables and schedule.
#include "rebuffer_timing_mutation.hh"


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
  // Schema 3 adds the sparse constraint view; the propagation payload keys
  // consumed below are identical, so both versions are valid sources.
  const int timing_schema_version = values["timing_schema_version"].cast<int>();
  require(values["schema_version"].cast<int>() == 1
              && (timing_schema_version == 2 || timing_schema_version == 3),
          "native propagation requires schema 2 or 3 timing inputs");
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

// --- rebuffer_timing_graph_propagate.cc ---
// Full-seed sweep on the immutable carrier graph, using shared timing math.
#include "rebuffer_timing_graph.hh"


#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>

#include "net_aat_math.h"

#include "timing_hard_math.h"


namespace py = pybind11;
namespace rebuffer {
namespace {

template <std::size_t Width>
const int64_t* flat(const std::vector<std::array<int64_t, Width>>& rows)
{
  return rows.empty() ? nullptr : rows.front().data();
}

}  // namespace

py::dict PersistentTimingGraph::initializePropagation(const py::dict& values)
{
  if (propagation_program_ != nullptr) {
    throw py::value_error("propagation inputs are immutable; create a new graph to reconfigure");
  }
  // Parse into a separate owner: malformed inputs cannot change graph state.
  propagation_program_ = timingProgramFromPython(values, *timing_index_);
  return summary();
}

py::dict PersistentTimingGraph::propagate()
{
  if (!timing_state_initialized_ || propagation_program_ == nullptr) {
    throw py::value_error("propagation requires initialized timing state and inputs");
  }
  if (!net_states_.empty()) {
    if (mutation_state_ == nullptr) {
      throw py::value_error("full-seed propagation requires synchronized buffer timing records");
    }
    requireSynchronizedTimingTopologies(*mutation_state_, net_states_);
  }
  const auto& p = *propagation_program_;
  std::shared_ptr<TimingPropagationSchedule> generated;
  if (mutation_state_ && mutation_state_->canonical_schedule) {
    const auto started = std::chrono::steady_clock::now();
    TimingGraphIndex rebuilt;
    rebuilt.pins = timing_index_->pins; rebuilt.clocks = timing_index_->clocks;
    rebuilt.clock_end = timing_index_->clock_end;
    buildTimingMutationIndex(*mutation_state_, rebuilt);
    const auto indexed = std::chrono::steady_clock::now();
    ++mutation_state_->statistics.index_builds;
    mutation_state_->statistics.index_ms += std::chrono::duration<double, std::milli>(indexed - started).count();
    generated = makeTimingMutationSchedule(*mutation_state_, rebuilt, p);
    ++mutation_state_->statistics.schedule_builds;
    mutation_state_->statistics.schedule_ms += std::chrono::duration<double, std::milli>(
        std::chrono::steady_clock::now() - indexed).count();
  }
  const auto& schedule = generated ? *generated : p.schedule;
  const auto& cells = mutation_state_ != nullptr && !generated
                          ? mutation_state_->cells : schedule.cells;
  const int pins = timing_index_->pins.size();
  const auto arcs = timing_index_->cells.size();
  std::map<std::string, std::vector<double>> output;
  auto& ra = output["pin_rAAT"] = std::vector<double>(pins, -2e8);
  auto& fa = output["pin_fAAT"] = std::vector<double>(pins, -2e8);
  auto& rs = output["pin_rtran"] = std::vector<double>(pins, 0);
  auto& fs = output["pin_ftran"] = std::vector<double>(pins, 0);
  auto& rr = output["pin_rRAT"] = std::vector<double>(pins, 2e8);
  auto& fr = output["pin_fRAT"] = std::vector<double>(pins, 2e8);
  auto& drr = output["cell_arc_rr_delays"] = std::vector<double>(arcs, 0);
  auto& dfr = output["cell_arc_fr_delays"] = std::vector<double>(arcs, 0);
  auto& drf = output["cell_arc_rf_delays"] = std::vector<double>(arcs, 0);
  auto& dff = output["cell_arc_ff_delays"] = std::vector<double>(arcs, 0);
  const auto& cap_r = state_.at("pin_net_cap_rise");
  const auto& cap_f = state_.at("pin_net_cap_fall");
  const auto& delay_r = state_.at("pin_net_delay_rise");
  const auto& delay_f = state_.at("pin_net_delay_fall");
  const auto& impulse_r = state_.at("pin_net_impulse_rise");
  const auto& impulse_f = state_.at("pin_net_impulse_fall");
  const auto rd = p.luts[0].view(), fd = p.luts[1].view();
  const auto rt = p.luts[2].view(), ft = p.luts[3].view();
  for (std::size_t i = 0; i < p.start_pins.size(); ++i) {
    const auto id = p.start_pins[i];
    ra[id] = p.start_rAAT[i]; fa[id] = p.start_fAAT[i];
    rs[id] = p.start_rtran[i]; fs[id] = p.start_ftran[i];
  }
  for (std::size_t i = 0; i < p.endpoint_pins.size(); ++i) {
    const auto id = p.endpoint_pins[i];
    rr[id] = p.endpoint_rRAT[i]; fr[id] = p.endpoint_fRAT[i];
  }

  dreamplace::timing::clk2qRows(
      flat(cells), schedule.level_start[1], 8, pins, cap_r.data(), cap_f.data(),
      p.clock_rtran.data(), p.clock_ftran.data(), rd, fd, rt, ft,
      p.pairwise_smooth, p.pairwise_alpha, ra.data(), fa.data(), rs.data(), fs.data());
  // Clock slots deliberately retain zero delays, matching the Python cache.
  std::vector<double> updates;
  std::size_t net_forward_rows = 0, net_reverse_rows = 0;
  auto forward_net = [&](const ProgramNetGroup& group) {
    updates.resize(group.arcs.size() * 4);
    for (std::size_t i = 0; i < group.arcs.size(); ++i) {
      double jacobian[6];
      netForwardRow(0, group.arcs[i].data(), 2, 1, pins,
                    ra.data(), fa.data(), rs.data(), fs.data(),
                    delay_r.data(), delay_f.data(), impulse_r.data(), impulse_f.data(),
                    updates.data() + i * 4, jacobian, p.slew_epsilon);
    }
    dreamplace::timing::aggregateNetRows(flat(group.arcs), updates.data(),
        group.arcs.size(), 2, ra.data(), fa.data(), rs.data(), fs.data());
    net_forward_rows += group.arcs.size();
  };
  forward_net(schedule.forward_nets[0]);
  for (std::size_t level = 1; level + 1 < schedule.level_start.size(); ++level) {
    const auto first = schedule.level_start[level], last = schedule.level_start[level + 1];
    updates.resize((last - first) * 8);
    for (auto i = first; i < last; ++i) {
      double jacobian[48];
      auto* row = updates.data() + (i - first) * 8;
      cellForwardRow(0, cells[i].data(), 8, 1, pins, ra.data(), fa.data(),
                     rs.data(), fs.data(), cap_r.data(), cap_f.data(), rd, fd, rt, ft,
                     row, jacobian, p.pairwise_smooth, p.pairwise_alpha);
      const auto slot = schedule.arc_ids[i];
      drr[slot] = row[4]; dfr[slot] = row[5]; drf[slot] = row[6]; dff[slot] = row[7];
    }
    if (first != last) {
      dreamplace::timing::aggregateCellRows(cells[first].data(), updates.data(),
          last - first, 8, ra.data(), fa.data(), rs.data(), fs.data());
    }
    forward_net(schedule.forward_nets[level]);
  }

  // Keep each batch's RAT input unchanged. Several rows can write one pin,
  // and self net arcs can read the same pin that they update.
  std::vector<double> scratch_r(pins), scratch_f(pins);
  auto prepare = [&](const std::vector<int64_t>& touched) {
    for (auto id : touched) { scratch_r[id] = rr[id]; scratch_f[id] = fr[id]; }
  };
  auto finish = [&](const std::vector<int64_t>& touched) {
    for (auto id : touched) { rr[id] = scratch_r[id]; fr[id] = scratch_f[id]; }
  };
  prepare(p.setup_pins);
  dreamplace::timing::setupRatRows(flat(p.setup_arcs), p.setup_arcs.size(), 8,
      rr.data(), fr.data(), p.clock_rtran.data(), p.clock_ftran.data(), rs.data(),
      fs.data(), rd, fd, p.setup_valid.get(), p.pairwise_smooth, p.pairwise_alpha,
      scratch_r.data(), scratch_f.data());
  finish(p.setup_pins);

  auto reference_r = p.recovery_reference_rtran;
  auto reference_f = p.recovery_reference_ftran;
  if (p.recovery_uses_live_slew) {
    for (std::size_t i = 0; i < p.recovery_arcs.size(); ++i) {
      reference_r[i] = rs[p.recovery_arcs[i][1]];
      reference_f[i] = fs[p.recovery_arcs[i][1]];
    }
  }
  dreamplace::timing::recoveryRatRows(flat(p.recovery_arcs), p.recovery_arcs.size(),
      8, pins, rr.data(), fr.data(), rs.data(), fs.data(),
      p.recovery_clock_rtran.data(), p.recovery_clock_ftran.data(),
      reference_r.data(), reference_f.data(), p.luts[4].view(), p.luts[5].view(),
      p.recovery_valid.get(), scratch_r.data(), scratch_f.data(),
      p.recovery_rat_is_capture_deadline, p.recovery_deadline_r.data(),
      p.recovery_deadline_f.data(), p.recovery_keep_setup.get());
  for (auto id : p.recovery_r_pins) rr[id] = scratch_r[id];
  for (auto id : p.recovery_f_pins) fr[id] = scratch_f[id];

  auto reverse_net = [&](const ProgramNetGroup& group) {
    prepare(group.drivers);
    dreamplace::timing::netRatRows(flat(group.arcs), group.arcs.size(), 2,
        rr.data(), fr.data(), delay_r.data(), delay_f.data(),
        scratch_r.data(), scratch_f.data());
    finish(group.drivers);
    net_reverse_rows += group.arcs.size();
  };
  reverse_net(schedule.endpoint_nets);
  for (int level = static_cast<int>(schedule.level_start.size()) - 2; level >= 1; --level) {
    const auto first = schedule.level_start[level], last = schedule.level_start[level + 1];
    prepare(schedule.level_inputs[level]);
    if (first != last) {
      for (auto row = first; row < last; ++row) {
        const auto slot = schedule.arc_ids[row];
        dreamplace::timing::cellRatRows(cells[row].data(), 1, 8,
            rr.data(), fr.data(), drr.data() + slot, dfr.data() + slot,
            drf.data() + slot, dff.data() + slot, scratch_r.data(), scratch_f.data());
      }
    }
    finish(schedule.level_inputs[level]);
    reverse_net(schedule.reverse_nets[level]);
  }

  py::dict coverage;
  coverage["clk2q_rows"] = schedule.level_start[1];
  coverage["cell_forward_rows"] = cells.size() - schedule.level_start[1];
  coverage["cell_reverse_rows"] = cells.size() - schedule.level_start[1];
  coverage["net_forward_rows"] = net_forward_rows;
  coverage["net_reverse_rows"] = net_reverse_rows;
  coverage["setup_rows"] = p.setup_arcs.size();
  coverage["recovery_rows"] = p.recovery_arcs.size();
  return finishPropagation(std::move(output), std::move(coverage), "full_seed");
}

}  // namespace rebuffer

// --- rebuffer_timing_compact.cc ---
// Compact native timing slots at a committed boundary; preserve original ids.

#include <algorithm>
#include <map>


namespace py = pybind11;
namespace rebuffer {
namespace {

std::vector<int> liveMap(const std::vector<unsigned char>& live, std::size_t prefix)
{
  if (prefix > live.size()
      || !std::all_of(live.begin(), live.begin() + prefix, [](auto value) { return value != 0; })) {
    throw py::value_error("compact cannot remove original timing records");
  }
  std::vector<int> result(live.size(), -1);
  int next = 0;
  for (std::size_t id = 0; id < live.size(); ++id) if (live[id]) result[id] = next++;
  return result;
}

template <typename T>
std::vector<T> gather(const std::vector<T>& old, const std::vector<int>& mapping)
{
  if (old.size() != mapping.size()) throw py::value_error("compact metadata lengths do not align");
  std::vector<T> result(std::count_if(mapping.begin(), mapping.end(), [](int id) { return id >= 0; }));
  for (std::size_t id = 0; id < old.size(); ++id) {
    if (mapping[id] >= 0) result.at(mapping[id]) = old[id];
  }
  return result;
}

int mapped(int id, const std::vector<int>& mapping)
{
  if (id < 0 || id >= static_cast<int>(mapping.size()) || mapping[id] < 0) {
    throw py::value_error("compact active record references a removed slot");
  }
  return mapping[id];
}

py::array_t<int64_t> array(const std::vector<int>& values)
{
  py::array_t<int64_t> result({static_cast<py::ssize_t>(values.size())},
                              {static_cast<py::ssize_t>(sizeof(int64_t))});
  std::copy(values.begin(), values.end(), result.mutable_data());
  return result;
}

}  // namespace

py::dict PersistentTimingGraph::compact()
{
  if (!mutation_state_ || !propagation_program_ || !timing_state_initialized_) {
    throw py::value_error("compact requires initialized native mutation and timing state");
  }
  if (!journal_.empty()) {
    throw py::value_error("compact requires a committed journal boundary");
  }
  requireSynchronizedTimingTopologies(*mutation_state_, net_states_);
  const auto& old = *mutation_state_;
  const auto pins = liveMap(old.pin_active, propagation_program_->pin_count);
  const auto arcs = liveMap(old.cell_active, propagation_program_->schedule.cells.size());
  std::vector<unsigned char> node_live(old.node_names.size(), 0), net_live(old.net_names.size(), 0);
  for (std::size_t id = 0; id < node_live.size(); ++id) {
    node_live[id] = id < static_cast<std::size_t>(old.original_node_count) || old.node_cells[id] >= 0;
  }
  for (std::size_t id = 0; id < net_live.size(); ++id) {
    net_live[id] = id < old.base_net_pins.size() || old.net_drivers[id] >= 0;
  }
  const auto nodes = liveMap(node_live, old.original_node_count);
  const auto nets = liveMap(net_live, old.base_net_pins.size());
  const bool changed = std::find(pins.begin(), pins.end(), -1) != pins.end()
      || std::find(arcs.begin(), arcs.end(), -1) != arcs.end()
      || std::find(nodes.begin(), nodes.end(), -1) != nodes.end()
      || std::find(nets.begin(), nets.end(), -1) != nets.end()
      || std::find(timing_index_->net_active.begin(), timing_index_->net_active.end(), 0)
             != timing_index_->net_active.end();
  py::dict result;
  result["status"] = "ok";
  result["changed"] = changed;
  result["pin_old_to_new"] = array(pins);
  result["cell_arc_old_to_new"] = array(arcs);
  result["node_old_to_new"] = array(nodes);
  result["net_old_to_new"] = array(nets);
  if (!changed) {
    std::vector<int> net_arcs(timing_index_->nets.size());
    for (std::size_t id = 0; id < net_arcs.size(); ++id) net_arcs[id] = id;
    result["net_arc_old_to_new"] = array(net_arcs);
    result["state"] = summary();
    return result;
  }
  // Prepare and validate a complete candidate before changing the owner.
  // Compaction is infrequent; per-candidate mutation retains sparse metadata undo.
  auto next = std::make_shared<TimingMutationState>(old);
  auto& m = *next;
  auto index = std::make_shared<TimingGraphIndex>();
  index->pins = gather(timing_index_->pins, pins);
  index->clocks = timing_index_->clocks;
  index->clock_end = timing_index_->clock_end;
  m.cells = gather(old.cells, arcs);
  for (std::size_t slot = 0; slot < m.cells.size(); ++slot) {
    auto& row = m.cells[slot];
    if (slot >= static_cast<std::size_t>(index->clock_end)) row[0] = mapped(row[0], pins);
    row[1] = mapped(row[1], pins);
    row[7] = mapped(row[7], nodes);
  }
  m.cell_active.assign(m.cells.size(), 1);
  m.pin_active.assign(index->pins.size(), 1);
  m.pin_owner = gather(old.pin_owner, pins);
  m.pin_net = gather(old.pin_net, pins);
  m.pin_ports = gather(old.pin_ports, pins);
  for (int& node : m.pin_owner) node = mapped(node, nodes);
  for (int& net : m.pin_net) if (net >= 0) net = mapped(net, nets);
  for (int lane = 0; lane < 5; ++lane) m.pin_values[lane] = gather(old.pin_values[lane], pins);
  m.node_names = gather(old.node_names, nodes);
  m.node_cells = gather(old.node_cells, nodes);
  m.node_pins = gather(old.node_pins, nodes);
  m.node_arcs = gather(old.node_arcs, nodes);
  for (auto& row : m.node_pins) for (int& pin : row) pin = mapped(pin, pins);
  for (auto& row : m.node_arcs) for (int& arc : row) arc = mapped(arc, arcs);
  m.net_names = gather(old.net_names, nets);
  m.net_drivers = gather(old.net_drivers, nets);
  m.net_pins = gather(old.net_pins, nets);
  for (int& driver : m.net_drivers) if (driver >= 0) driver = mapped(driver, pins);
  for (auto& row : m.net_pins) for (int& pin : row) pin = mapped(pin, pins);
  m.buffers.clear();
  for (auto& vec : m.buffers_by_source_net) vec.clear();
  m.buffer_id_by_action_id.clear();
  for (const auto& previous : old.buffers) {
    if (!previous.active) continue;
    auto row = previous;
    row.node = mapped(row.node, nodes);
    row.input_pin = mapped(row.input_pin, pins); row.output_pin = mapped(row.output_pin, pins);
    row.output_net = mapped(row.output_net, nets);
    for (int& arc : row.arcs) arc = mapped(arc, arcs);
    const int new_id = static_cast<int>(m.buffers.size());
    m.buffers_by_source_net.at(row.source_net).push_back(new_id);
    m.buffer_id_by_action_id.emplace(row.action_id, new_id);
    m.buffers.push_back(std::move(row));
  }
  auto values = remappedTimingState(pins, arcs);
  rebuildTimingMutationIndex(m, *index, *propagation_program_);
  std::map<TimingNetArc, int> net_arc_ids;
  for (int id = 0; id < static_cast<int>(index->nets.size()); ++id) net_arc_ids.emplace(index->nets[id], id);
  std::vector<int> net_arc_map;
  for (std::size_t id = 0; id < timing_index_->nets.size(); ++id) {
    if (!timing_index_->net_active.empty() && !timing_index_->net_active[id]) {
      net_arc_map.push_back(-1);
      continue;
    }
    const auto& edge = timing_index_->nets[id];
    const TimingNetArc mapped_edge{mapped(edge[0], pins), mapped(edge[1], pins)};
    net_arc_map.push_back(net_arc_ids.at(mapped_edge));
  }
  result["net_arc_old_to_new"] = array(net_arc_map);
  detachStateViews();
  mutation_state_ = std::move(next);
  timing_index_ = std::move(index);
  state_ = std::move(values);
  cached_active_counts_valid_ = false;
  ++topology_revision_;
  markElectricalFullDirty();
  cached_endpoint_pin_names_.clear();
  cached_endpoint_slacks_by_name_values_.clear();
  ++epoch_;
  ++journal_generation_;
  result["state"] = summary();
  return result;
}

}  // namespace rebuffer
