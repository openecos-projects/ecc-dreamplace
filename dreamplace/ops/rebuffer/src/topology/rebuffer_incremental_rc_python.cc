// SPDX-License-Identifier: BSD-3-Clause
// Legacy dense dictionary/NumPy boundary for the shared compact native RC math.
#include "rebuffer_incremental_patch.hh"
#include "rebuffer_incremental_rc.hh"

#include <algorithm>
#include <cstdint>
#include <string>
#include <unordered_map>

#include "pybind11/stl.h"

namespace py = pybind11;
namespace rebuffer {

py::dict evaluateIncrementalRc(
    const py::dict& rc,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& generic_caps,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& rise_caps,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& fall_caps,
    const py::dict& buffer_caps, const int pin_count)
{
  const auto global_parent = rc["parent"].cast<std::vector<int>>();
  const auto global_child_start = rc["child_start"].cast<std::vector<int>>();
  const auto global_children = rc["child"].cast<std::vector<int>>();
  const auto global_topo_start = rc["topo_start"].cast<std::vector<int>>();
  const auto global_topo = rc["topo"].cast<std::vector<int>>();
  const auto global_resistance = rc["edge_resistance"].cast<std::vector<double>>();
  const auto global_edge_cap = rc["edge_capacitance"].cast<std::vector<double>>();
  CompactRcForest forest;
  auto& affected = forest.node_ids;
  affected = rc["affected_nodes"].cast<std::vector<int>>();
  std::unordered_map<int, int> local;
  local.reserve(affected.size());
  for (int i = 0; i < static_cast<int>(affected.size()); ++i) local.emplace(affected[i], i);
  forest.child_start.push_back(0);
  for (int node : affected) {
    forest.parent.push_back(local.at(global_parent[node]));
    forest.resistance.push_back(global_resistance[node]);
    forest.edge_capacitance.push_back(global_edge_cap[node]);
    for (int offset = global_child_start[node]; offset < global_child_start[node + 1]; ++offset) {
      forest.children.push_back(local.at(global_children[offset]));
    }
    forest.child_start.push_back(forest.children.size());
  }
  forest.topo_start = global_topo_start;
  for (int node : global_topo) forest.topo.push_back(local.at(node));
  const auto wire_cap = compactRcWireCaps(forest);
  auto modeCaps = [&](const auto& base, int mode) {
    std::vector<double> caps(affected.size(), 0);
    const auto values = base.template unchecked<1>();
    std::vector<std::vector<double>> slot_inputs, slot_outputs;
    if (buffer_caps.contains("slots")) {
      for (const auto& item : buffer_caps["slots"].cast<py::sequence>()) {
        const auto slot = item.cast<py::dict>();
        slot_inputs.push_back(slot["input"].cast<std::vector<double>>());
        slot_outputs.push_back(slot["output"].cast<std::vector<double>>());
      }
    } else {
      slot_inputs.push_back(buffer_caps["input"].cast<std::vector<double>>());
      slot_outputs.push_back(buffer_caps["output"].cast<std::vector<double>>());
    }
    for (int i = 0; i < static_cast<int>(affected.size()); ++i) {
      const int node = affected[i];
      if (node < pin_count && node < values.shape(0)) caps[i] = values(node);
      else if (node >= pin_count) {
        const int offset = node - pin_count, slot = offset / 2;
        if (slot >= 0 && slot < static_cast<int>(slot_inputs.size())) {
          caps[i] = (offset % 2 == 0 ? slot_inputs[slot] : slot_outputs[slot]).at(mode);
        }
      }
      caps[i] += wire_cap[i];
    }
    return caps;
  };
  py::dict result;
  py::array_t<int64_t> nodes({static_cast<py::ssize_t>(affected.size())},
                            {static_cast<py::ssize_t>(sizeof(int64_t))});
  std::copy(affected.begin(), affected.end(), nodes.mutable_data());
  result["affected_nodes"] = std::move(nodes);
  const std::array<const char*, 3> modes{"generic", "rise", "fall"};
  const std::array<const char*, 3> fields{"loads_", "delays_", "impulses_"};
  const std::array<const py::array_t<double, py::array::c_style | py::array::forcecast>*, 3>
      caps{&generic_caps, &rise_caps, &fall_caps};
  for (int mode = 0; mode < 3; ++mode) {
    const auto values = evaluateCompactRc(forest, modeCaps(*caps[mode], mode));
    for (int field = 0; field < 3; ++field) {
      result[(std::string(fields[field]) + modes[mode]).c_str()] = values[field];
    }
  }
  return result;
}

}  // namespace rebuffer
