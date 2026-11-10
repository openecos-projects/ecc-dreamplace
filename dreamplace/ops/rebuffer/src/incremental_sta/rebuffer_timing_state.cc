// SPDX-License-Identifier: BSD-3-Clause
// Canonical Python TP state import and atomic journal updates.

#include "rebuffer_timing_graph.hh"


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
  if (!values.empty()) {
    detachStateViews();
    cached_endpoint_slacks_valid_ = false;
    markElectricalFullDirty();
  }
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
    markElectricalFullDirty();
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

// --- rebuffer_timing_state_view.cc ---
// Read-only zero-copy snapshots retain their storage across graph mutations.

#include <utility>

namespace py = pybind11;
namespace rebuffer {

struct TimingStateViewStorage
{
  // The graph owns current storage until the next write. On detach, retained
  // takes that exact allocation and the graph reference is released.
  std::shared_ptr<const PersistentTimingGraph> graph;
  std::vector<double> retained;
  double empty = 0;
};

py::dict PersistentTimingGraph::stateView(bool copy) const
{
  py::dict result;
  result["status"] = "ok";
  result["epoch"] = epoch_;
  for (const auto& [name, value] : state_) {
    if (copy) {
      result[py::str(name)] = writeState(value);
      continue;
    }
    auto& cached = state_views_[name];
    auto storage = cached.lock();
    if (!storage) {
      storage = std::make_shared<TimingStateViewStorage>();
      storage->graph = shared_from_this();
      cached = storage;
    }
    auto* holder = new std::shared_ptr<TimingStateViewStorage>(storage);
    py::capsule base(holder, [](void* pointer) {
      delete static_cast<std::shared_ptr<TimingStateViewStorage>*>(pointer);
    });
    const double* data = value.empty() ? &storage->empty : value.data();
    py::array_t<double> array({static_cast<py::ssize_t>(value.size())},
                              {static_cast<py::ssize_t>(sizeof(double))}, data, base);
    array.attr("setflags")(py::arg("write") = false);
    result[py::str(name)] = std::move(array);
    ++state_view_counters_.exports;
  }
  return result;
}

void PersistentTimingGraph::detachStateViews()
{
  cached_endpoint_slacks_valid_ = false;
  if (state_views_.empty()) return;
  // Ownership transfer below may release references from every view holder.
  // Keep this graph alive until the complete transfer has finished.
  const auto owner = shared_from_this();
  for (auto it = state_views_.begin(); it != state_views_.end();) {
    if (auto storage = it->second.lock()) {
      auto& current = state_.at(it->first);
      auto replacement = current;  // Allocate before changing either owner.
      storage->retained = std::move(current);
      current = std::move(replacement);
      ++state_view_counters_.detached_fields;
      state_view_counters_.detached_values += storage->retained.size();
      storage->graph.reset();
    }
    it = state_views_.erase(it);
  }
}

py::dict PersistentTimingGraph::stateViewStatistics() const
{
  py::dict result;
  result["exports"] = state_view_counters_.exports;
  result["detached_fields"] = state_view_counters_.detached_fields;
  result["detached_values"] = state_view_counters_.detached_values;
  std::size_t current = 0;
  for (const auto& [_, storage] : state_views_) if (!storage.expired()) ++current;
  result["current_view_fields"] = current;
  return result;
}

}  // namespace rebuffer

// --- rebuffer_timing_state_undo.cc ---
#include "rebuffer_timing_mutation.hh"


#include <algorithm>
#include <cstring>

namespace rebuffer {

void captureTimingValue(TimingStateUndo& undo, const char* field,
                         const std::vector<double>& values, std::size_t index)
{
  auto& saved = undo.fields.try_emplace(field, TimingFieldUndo{values.size(), {}}).first->second;
  // Newly appended slots are restored by truncating to previous_size.
  if (index < saved.previous_size) {
    saved.previous.push_back({index, values.at(index)});
    ++undo.value_count;
  }
}

void patchTimingValue(TimingStateUndo& undo, const char* field,
                       std::vector<double>& values, std::size_t index, double value)
{
  auto& current = values.at(index);
  if (std::memcmp(&current, &value, sizeof(double)) == 0) return;
  captureTimingValue(undo, field, values, index);
  current = value;
}

void compactTimingUndo(TimingStateUndo& undo,
    const std::map<std::string, std::vector<double>>& state,
    bool indices_are_unique)
{
  undo.value_count = 0;
  for (auto& [name, field] : undo.fields) {
    auto& values = field.previous;
    if (!indices_are_unique) {
      // Keep the first old value if an RC batch writes the same slot twice.
      std::stable_sort(values.begin(), values.end(),
                       [](const auto& a, const auto& b) { return a.index < b.index; });
      values.erase(std::unique(values.begin(), values.end(),
          [](const auto& a, const auto& b) { return a.index == b.index; }), values.end());
    }
    const auto& current = state.at(name);
    values.erase(std::remove_if(values.begin(), values.end(), [&](const auto& old) {
      return std::memcmp(&current.at(old.index), &old.value, sizeof(double)) == 0;
    }), values.end());
    undo.value_count += values.size();
  }
}

void restoreTimingState(std::map<std::string, std::vector<double>>& state,
                         const TimingStateUndo& undo)
{
  for (const auto& [name, field] : undo.fields) {
    auto& current = state.at(name);
    // Before compaction, repeated writes must unwind in reverse order.
    for (auto it = field.previous.rbegin(); it != field.previous.rend(); ++it) {
      current.at(it->index) = it->value;
    }
    current.resize(field.previous_size);
  }
}

std::shared_ptr<TimingStateUndo> PersistentTimingGraph::beginStateUpdate()
{
  auto undo = std::make_shared<TimingStateUndo>();
  detachStateViews();
  JournalEntry entry;
  entry.previous_epoch = epoch_;
  entry.sequence = next_journal_sequence_++;
  entry.state_undo = undo;
  journal_.push_back(std::move(entry));
  ++epoch_;
  return undo;
}

void PersistentTimingGraph::applyTimingRc(const std::vector<TimingNetRcValues>& changes,
                                          const std::vector<int>& inactive_pins)
{
  auto undo = beginStateUpdate();
  for (const auto& field : timingStateFields()) {
    auto& values = state_.at(field.name);
    const auto size = field.pin_indexed ? timing_index_->pins.size() : timing_index_->cells.size();
    if (values.size() == size) continue;
    undo->fields.emplace(field.name, TimingFieldUndo{values.size(), {}});
    values.resize(size, field.initial);
  }
  constexpr std::array<const char*, 6> names{
      "pin_net_cap_rise", "pin_net_cap_fall", "pin_net_delay_rise", "pin_net_delay_fall",
      "pin_net_impulse_rise", "pin_net_impulse_fall"};
  for (const auto& rc : changes) {
    for (int pin : rc.pins) markElectricalDirtyPin(pin);
    for (std::size_t lane = 0; lane < names.size(); ++lane) {
      auto& target = state_.at(names[lane]);
      for (std::size_t row = 0; row < rc.pins.size(); ++row) {
        patchTimingValue(*undo, names[lane], target, rc.pins[row], rc.values[lane].at(row));
      }
    }
  }
  for (int pin : inactive_pins) markElectricalDirtyPin(pin);
  for (const auto* name : names) {
    for (int pin : inactive_pins) patchTimingValue(*undo, name, state_.at(name), pin, 0);
  }
  compactTimingUndo(*undo, state_);
}

}  // namespace rebuffer
