// SPDX-License-Identifier: BSD-3-Clause
// Rebuild timing graph indices from physical PyPlaceDB without running STA.
#pragma once

#include <string>
#include <unordered_map>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace rebuffer {

struct TimingCarrierView
{
  pybind11::dict values;
  std::vector<std::string> node_names;
  std::vector<std::string> pin_names;
  std::vector<std::string> net_names;
  std::vector<int> old_node_ids;
  std::vector<int> old_pin_ids;
  std::vector<int> pin2node;
  std::vector<int> cell_by_node;
  std::vector<int> cell_arc_start;
  std::vector<std::vector<int>> net_pins;
  std::unordered_map<std::string, int> pin_ids;
};

template <typename T>
std::vector<T> carrierArray(const pybind11::object& source, const char* name)
{
  return source.attr(name).cast<std::vector<T>>();
}

inline int carrierIndex(int index, const std::vector<int>& mapping, const char* field)
{
  if (index < 0 || index >= static_cast<int>(mapping.size()) || mapping[index] < 0) {
    throw pybind11::value_error(std::string("unmapped timing identity: ") + field);
  }
  return mapping[index];
}

void compileCarrierGraph(const pybind11::object& base, TimingCarrierView& carrier);
pybind11::dict rebuildTimingCarrier(const pybind11::object& base,
                                   const pybind11::object& physical);

}  // namespace rebuffer
