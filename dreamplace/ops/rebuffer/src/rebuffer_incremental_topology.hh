// SPDX-License-Identifier: BSD-3-Clause
// Mutable per-net topology state for native rebuffer transactions.

#pragma once

#include <memory>
#include <array>
#include <string>
#include <unordered_map>
#include <vector>

#include "pybind11/pybind11.h"

namespace rebuffer {

struct SteinerView;

struct IncrementalEdge
{
  int parent = -1;
  int child = -1;
  double resistance = 0.0;
  double capacitance = 0.0;
};

struct IncrementalBuffer
{
  int action_id = -1;
  // Immutable Liberty identity for the inserted buffer.  Keeping the master
  // with the native slot lets a cumulative timing patch rebuild its cell arc
  // without relying on a Python-side action list.
  std::string master_name;
  int input_node = -1;
  int output_node = -1;
  int original_parent = -1;
  int original_child = -1;
  int upstream_node = -1;
  int downstream_node = -1;
  double split_ratio = 0.0;
  int next_node_before = -1;
  int epoch_before = 0;
  std::vector<IncrementalEdge> edges_before;
  std::unordered_map<int, std::vector<int>> loads_before;
};

struct IncrementalTopologyState
{
  int net_id = -1;
  int driver_pin_id = -1;
  int next_node_id = 0;
  int epoch = 0;
  std::vector<IncrementalEdge> edges;
  std::unordered_map<int, std::vector<int>> node_loads;
  std::unordered_map<int, std::array<double, 2>> coordinates;
  std::vector<IncrementalBuffer> buffers;
};

struct IncrementalTopologyCheckpoint
{
  IncrementalTopologyState state;
};

std::shared_ptr<IncrementalTopologyState> createIncrementalTopology(
    int net_id,
    int driver_pin_id,
    int next_node_id,
    const pybind11::dict& topology_values,
    std::string& reason);

std::shared_ptr<IncrementalTopologyState> incrementalTopologyFromSteiner(
    const SteinerView& steiner, int net_id, int driver_pin_id,
    double r_unit, double c_unit);

bool applyIncrementalBufferSplit(
    IncrementalTopologyState& state,
    const pybind11::dict& action,
    std::string& reason,
    pybind11::dict& result);

bool removeIncrementalBuffer(
    IncrementalTopologyState& state,
    int action_id,
    std::string& reason,
    pybind11::dict& result);

std::shared_ptr<IncrementalTopologyCheckpoint> checkpointIncrementalTopology(
    const IncrementalTopologyState& state);

bool restoreIncrementalTopology(
    IncrementalTopologyState& state,
    const IncrementalTopologyCheckpoint& checkpoint,
    std::string& reason);

bool validateIncrementalTopology(
    const IncrementalTopologyState& state,
    std::string& reason);

pybind11::dict incrementalTopologyView(const IncrementalTopologyState& state);

}  // namespace rebuffer
