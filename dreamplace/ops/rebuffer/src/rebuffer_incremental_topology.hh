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
struct IncrementalTopologyImage;

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
  // Journal entry for undoing the last split.  The previous implementation
  // copied the complete edge vector and load map into every buffer record;
  // that made a checkpoint and a long candidate chain grow quadratically.
  // The active edge is removed and three edges are appended, so this small
  // record is sufficient to restore the exact prior topology and ordering.
  IncrementalEdge original_edge;
  std::size_t original_edge_index = 0;
  bool has_coordinates = false;
  std::array<double, 2> split_location{0.0, 0.0};
};

struct IncrementalTopologyState
{
  int net_id = -1;
  // Source identity remains stable while the current compact carrier is
  // remapped.  It prevents equal per-net Steiner integers from being
  // mistaken for the same physical vertex.
  int source_net_id = -1;
  int driver_pin_id = -1;
  int next_node_id = 0;
  int epoch = 0;
  std::vector<IncrementalEdge> edges;
  std::unordered_map<int, std::vector<int>> node_loads;
  std::unordered_map<int, std::array<double, 2>> coordinates;
  std::unordered_map<int, int> source_node_by_node;
  std::vector<IncrementalBuffer> buffers;
  bool general_removal = false;
  mutable std::shared_ptr<const IncrementalTopologyImage> checkpoint_image;

  // One immutable base copy lets restore rebuild a divergent checkpoint
  // history without retaining a full edge/load snapshot in every checkpoint.
  // It is paid once per native net, rather than once per candidate.
  std::vector<IncrementalEdge> base_edges;
  std::unordered_map<int, std::vector<int>> base_node_loads;
  std::unordered_map<int, std::array<double, 2>> base_coordinates;
  std::unordered_map<int, int> base_source_node_by_node;
  int base_next_node_id = 0;
  int base_epoch = 0;
  bool base_captured = false;

  // Optional exact schedule cache owned by this topology version.  The
  // incremental patch builder fills it only under its explicit diagnostic
  // switch; any topology mutation clears it before changing the epoch.
  pybind11::dict schedule_cache;
  std::vector<int> schedule_cache_changed_pins;
  int schedule_cache_epoch = -1;
  int schedule_cache_slot = -1;
  int schedule_cache_pin_count = -1;
  int schedule_cache_row_count = -1;
  int schedule_cache_net_count = -1;
  int schedule_cache_event = -1;
};

// Only a topology with a non-LIFO removal needs a full per-source image.
// Every checkpoint of one unchanged revision shares the same immutable image.
struct IncrementalTopologyImage
{
  explicit IncrementalTopologyImage(const IncrementalTopologyState& source) : state(source)
  {
    state.checkpoint_image.reset();
  }
  IncrementalTopologyState state;
};

struct IncrementalTopologyCheckpoint
{
  int net_id = -1;
  int driver_pin_id = -1;
  std::size_t buffer_count = 0;
  int next_node_id = 0;
  int epoch = 0;
  std::vector<int> action_ids;
  std::vector<IncrementalBuffer> buffers;
  std::shared_ptr<const IncrementalTopologyImage> image;
  // A pybind dict assignment only retains the existing NumPy array objects;
  // it does not copy their storage.  Keeping this handle preserves an exact
  // topology-version schedule cache across journal restore.
  pybind11::dict schedule_cache;
  std::vector<int> schedule_cache_changed_pins;
  int schedule_cache_epoch = -1;
  int schedule_cache_slot = -1;
  int schedule_cache_pin_count = -1;
  int schedule_cache_row_count = -1;
  int schedule_cache_net_count = -1;
  int schedule_cache_event = -1;
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

// Copy a per-net state into the node namespace of a compact parent carrier.
// Physical pin ids remain stable; Steiner and synthetic buffer node ids are
// shifted together so a later cross-net patch can coexist with prior slots.
std::shared_ptr<IncrementalTopologyState> remapIncrementalTopology(
    const std::shared_ptr<IncrementalTopologyState>& state,
    int original_pin_count,
    int node_shift,
    int target_net_id = -1,
    int synthetic_node_base = -1,
    const pybind11::dict& source_node_mapping = pybind11::dict());

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

bool removeIncrementalBufferAnywhere(
    IncrementalTopologyState& state, int action_id,
    std::string& reason, pybind11::dict& result);

bool resizeIncrementalBuffer(
    IncrementalTopologyState& state,
    const pybind11::dict& action,
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
