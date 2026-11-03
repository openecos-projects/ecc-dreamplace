// SPDX-License-Identifier: BSD-3-Clause
// Canonical Python TP state import and atomic journal updates.

#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_state.hh"

#include <algorithm>
#include <array>
#include <utility>

#include "pybind11/stl.h"

namespace py = pybind11;

namespace rebuffer {
namespace {

constexpr std::array<TimingStateField, 16> kTimingFields{{
    {"pin_rAAT", "pin_rAAT_live_snapshot", true, "ps", -2e8},
    {"pin_fAAT", "pin_fAAT_live_snapshot", true, "ps", -2e8},
    {"pin_rtran", "pin_rtran_live", true, "ps"},
    {"pin_ftran", "pin_ftran_live", true, "ps"},
    {"pin_rRAT", "pin_rRAT_live_snapshot", true, "ps", 2e8},
    {"pin_fRAT", "pin_fRAT_live_snapshot", true, "ps", 2e8},
    {"pin_net_cap_rise", "pin_net_cap_rise_live", true, "pF"},
    {"pin_net_cap_fall", "pin_net_cap_fall_live", true, "pF"},
    {"pin_net_delay_rise", "pin_net_delay_rise_live", true, "ps"},
    {"pin_net_delay_fall", "pin_net_delay_fall_live", true, "ps"},
    {"pin_net_impulse_rise", "pin_net_impulse_rise_live", true, "ps^2"},
    {"pin_net_impulse_fall", "pin_net_impulse_fall_live", true, "ps^2"},
    {"cell_arc_rr_delays", "cell_arc_rr_delays", false, "ps"},
    {"cell_arc_fr_delays", "cell_arc_fr_delays", false, "ps"},
    {"cell_arc_rf_delays", "cell_arc_rf_delays", false, "ps"},
    {"cell_arc_ff_delays", "cell_arc_ff_delays", false, "ps"},
}};

const TimingStateField* findField(const std::string& name)
{
  const auto found = std::find_if(
      kTimingFields.begin(), kTimingFields.end(),
      [&](const TimingStateField& field) { return name == field.name; });
  return found == kTimingFields.end() ? nullptr : &*found;
}

std::vector<double> readTimingField(
    const py::handle& value, const TimingStateField& field,
    const std::size_t pin_count, const std::size_t cell_arc_count)
{
  const std::string prefix = std::string(field.name) + " must be ";
  if (!py::isinstance<py::array>(value)) {
    throw py::value_error(prefix + "a numpy array");
  }
  const auto array = py::reinterpret_borrow<py::array>(value);
  if (!array.dtype().is(py::dtype::of<double>())) {
    throw py::value_error(prefix + "native float64");
  }
  if (array.ndim() != 1 || !(array.flags() & py::array::c_style)) {
    throw py::value_error(prefix + "one-dimensional and C-contiguous");
  }
  const std::size_t expected = field.pin_indexed ? pin_count : cell_arc_count;
  if (static_cast<std::size_t>(array.size()) != expected) {
    throw py::value_error(std::string(field.name) + " length must equal "
                          + (field.pin_indexed ? "pin_count=" : "cell_arc_count=")
                          + std::to_string(expected));
  }
  if (expected == 0) return {};
  const auto* data = static_cast<const double*>(array.data());
  return {data, data + expected};
}

}  // namespace

const std::array<TimingStateField, 16>& timingStateFields()
{
  return kTimingFields;
}

py::dict persistentTimingStateSchema(
    const std::size_t pin_count, const std::size_t cell_arc_count)
{
  py::dict fields;
  for (const auto& field : kTimingFields) {
    py::dict spec;
    spec["domain"] = field.pin_indexed ? "pin" : "cell_arc";
    spec["length"] = field.pin_indexed ? pin_count : cell_arc_count;
    spec["dtype"] = "float64";
    spec["source"] = field.source;
    spec["unit"] = field.unit;
    fields[py::str(field.name)] = std::move(spec);
  }
  py::dict schema;
  schema["schema_version"] = 1;
  schema["pin_count"] = pin_count;
  schema["cell_arc_count"] = cell_arc_count;
  schema["fields"] = std::move(fields);
  return schema;
}

py::dict PersistentTimingGraph::timingStateSchema() const
{
  return persistentTimingStateSchema(
      timing_index_->pins.size(), timing_index_->cells.size());
}

std::vector<double> PersistentTimingGraph::readState(
    const py::handle& value, const char* name)
{
  const auto array
      = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(value);
  if (!array || array.ndim() != 1) {
    throw py::value_error(std::string(name) + " must be a one-dimensional array");
  }
  if (array.size() == 0) return {};
  const auto* data = array.data();
  return {data, data + array.size()};
}

py::array_t<double> PersistentTimingGraph::writeState(
    const std::vector<double>& value)
{
  py::array_t<double> result(
      {static_cast<py::ssize_t>(value.size())},
      {static_cast<py::ssize_t>(sizeof(double))});
  std::copy(value.begin(), value.end(), result.mutable_data());
  return result;
}

py::dict PersistentTimingGraph::applyStateValues(
    std::map<std::string, std::vector<double>> values)
{
  journal_.reserve(journal_.size() + values.size());
  if (!values.empty()) detachStateViews();
  for (auto& [name, value] : values) {
    const auto previous = state_.find(name);
    JournalEntry entry;
    entry.name = name;
    entry.existed = previous != state_.end();
    entry.previous_epoch = epoch_;
    entry.sequence = next_journal_sequence_++;
    if (entry.existed) entry.previous = std::move(previous->second);
    journal_.push_back(std::move(entry));
    state_[name] = std::move(value);
    ++epoch_;
  }
  return summary();
}

py::dict PersistentTimingGraph::setState(const py::dict& values)
{
  // Legacy callers also journal per-net RC scratch arrays. Canonical timing
  // names have a strict domain even when the full state is not initialized.
  std::map<std::string, std::vector<double>> parsed;
  for (const auto& item : values) {
    const auto name = py::cast<std::string>(item.first);
    const auto* field = findField(name);
    parsed.emplace(name, field != nullptr
        ? readTimingField(item.second, *field, timing_index_->pins.size(),
                           timing_index_->cells.size())
        : readState(item.second, name.c_str()));
  }
  if (!timing_state_initialized_ || parsed.empty()) return applyStateValues(std::move(parsed));
  const auto checkpoint_size = journal_.size();
  try {
    auto undo = beginStateUpdate();
    std::map<std::string, std::vector<double>> generic;
    for (auto& [name, values] : parsed) {
      if (findField(name) == nullptr) {
        generic.emplace(name, std::move(values));
        continue;
      }
      auto& current = state_.at(name);
      for (std::size_t id = 0; id < values.size(); ++id) {
        patchTimingValue(*undo, name.c_str(), current, id, values[id]);
      }
    }
    compactTimingUndo(*undo, state_);
    applyStateValues(std::move(generic));
    return summary();
  } catch (...) {
    restoreTo(checkpoint_size);
    throw;
  }
}

std::map<std::string, std::vector<double>> PersistentTimingGraph::remappedTimingState(
    const std::vector<int>& pins, const std::vector<int>& arcs) const
{
  if (state_.size() != kTimingFields.size()) {
    throw py::value_error("compact requires canonical timing state without generic scratch fields");
  }
  std::map<std::string, std::vector<double>> result;
  for (const auto& field : kTimingFields) {
    const auto& mapping = field.pin_indexed ? pins : arcs;
    const auto& old = state_.at(field.name);
    if (mapping.size() != old.size()) {
      throw py::value_error("compact timing state does not align with graph records");
    }
    std::vector<double> values(std::count_if(mapping.begin(), mapping.end(),
                                            [](int id) { return id >= 0; }));
    for (std::size_t id = 0; id < mapping.size(); ++id) {
      if (mapping[id] >= 0) values.at(mapping[id]) = old[id];
    }
    result.emplace(field.name, std::move(values));
  }
  return result;
}

py::dict PersistentTimingGraph::initializeTimingState(const py::dict& values)
{
  std::map<std::string, std::vector<double>> parsed;
  for (const auto& item : values) {
    const auto name = py::cast<std::string>(item.first);
    const auto* field = findField(name);
    if (field == nullptr) {
      throw py::value_error("unknown persistent timing state field: " + name);
    }
    parsed.emplace(name, readTimingField(
        item.second, *field, timing_index_->pins.size(), timing_index_->cells.size()));
  }
  for (const auto& field : kTimingFields) {
    if (parsed.count(field.name) == 0) {
      throw py::value_error(std::string("missing persistent timing state field: ")
                            + field.name);
    }
  }
  // No mutation occurs until every required field has passed validation.
  applyStateValues(std::move(parsed));
  timing_state_initialized_ = true;
  return summary();
}

}  // namespace rebuffer
