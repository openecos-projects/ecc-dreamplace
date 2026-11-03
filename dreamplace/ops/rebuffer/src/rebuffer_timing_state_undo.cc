// SPDX-License-Identifier: BSD-3-Clause
#include "rebuffer_timing_state.hh"
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_buffer_state.hh"

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
    for (std::size_t lane = 0; lane < names.size(); ++lane) {
      auto& target = state_.at(names[lane]);
      for (std::size_t row = 0; row < rc.pins.size(); ++row) {
        patchTimingValue(*undo, names[lane], target, rc.pins[row], rc.values[lane].at(row));
      }
    }
  }
  for (const auto* name : names) {
    for (int pin : inactive_pins) patchTimingValue(*undo, name, state_.at(name), pin, 0);
  }
  compactTimingUndo(*undo, state_);
}

}  // namespace rebuffer
