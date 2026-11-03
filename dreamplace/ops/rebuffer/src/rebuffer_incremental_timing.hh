// SPDX-License-Identifier: BSD-3-Clause
#pragma once
#include <array>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include "pybind11/numpy.h"
#include "pybind11/pybind11.h"

namespace rebuffer {

using TimingCellArc = std::array<int, 8>;
using TimingNetArc = std::array<int, 2>;
using TimingArcIdentity = std::array<int, 6>;

struct TimingArcHash
{
  std::size_t operator()(const TimingArcIdentity& value) const;
};

// The index belongs to one immutable carrier snapshot, never to a mutable IDB.
struct TimingGraphIndex
{
  std::vector<std::string> pins;
  std::vector<TimingCellArc> cells;
  // Empty means all slots are active (the immutable carrier representation).
  std::vector<unsigned char> cell_active;
  std::vector<TimingNetArc> nets;
  // Empty means all edges are active; mutable owners retain retired net slots.
  std::vector<unsigned char> net_active;
  std::vector<int> clocks;
  int clock_end = 0;
  std::unordered_map<std::string, int> pin_ids;
  std::unordered_map<TimingArcIdentity, int, TimingArcHash> arc_ids;

  // Immutable topology structure shared by every affected propagation on
  // this carrier.  Seed masks, endpoint filtering, and worklists remain
  // per-call state in affectedTimingSchedule().
  std::vector<int> rank;
  // Carrier flat-arc order counts cell edges, while net edges have weight 0.
  std::vector<int> path_level;
  std::vector<std::vector<int>> successors;
  std::vector<std::vector<int>> cell_out;
  std::vector<std::vector<int>> cell_in;
  std::vector<std::vector<int>> net_out;
  std::vector<std::vector<int>> net_in;

  int source(int id) const;
  TimingArcIdentity key(int id) const;
  void build();
};

std::shared_ptr<TimingGraphIndex> timingGraphIndex(
    const pybind11::object& db, bool* cache_hit = nullptr);
std::shared_ptr<TimingGraphIndex> makeTimingGraphIndex(
    const std::vector<std::string>& pins, const std::vector<TimingCellArc>& cells,
    const std::vector<TimingNetArc>& nets, const std::vector<int>& clocks, int clock_end);

pybind11::dict affectedTimingSchedule(
    const pybind11::object& previous, const pybind11::object& current,
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>& previous_rc,
    const pybind11::array_t<double, pybind11::array::c_style | pybind11::array::forcecast>& current_rc,
    const pybind11::object& changed_pins = pybind11::none());
}  // namespace rebuffer
