// SPDX-License-Identifier: BSD-3-Clause
// Sparse undo for graph-owned canonical pin and cell-arc arrays.
#pragma once

#include <array>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace rebuffer {

struct TimingStateField
{
  const char* name;
  const char* source;
  bool pin_indexed;
  const char* unit;
  double initial = 0;
};

const std::array<TimingStateField, 16>& timingStateFields();

struct TimingIndexedValue
{
  std::size_t index;
  double value;
};

struct TimingFieldUndo
{
  std::size_t previous_size;
  std::vector<TimingIndexedValue> previous;
};

struct TimingStateUndo
{
  std::map<std::string, TimingFieldUndo> fields;
  std::size_t value_count = 0;
};

void captureTimingValue(TimingStateUndo& undo, const char* field,
                         const std::vector<double>& values, std::size_t index);
void patchTimingValue(TimingStateUndo& undo, const char* field,
                       std::vector<double>& values, std::size_t index, double value);
void compactTimingUndo(TimingStateUndo& undo,
    const std::map<std::string, std::vector<double>>& state,
    bool indices_are_unique = false);
void restoreTimingState(std::map<std::string, std::vector<double>>& state,
                         const TimingStateUndo& undo);

}  // namespace rebuffer
