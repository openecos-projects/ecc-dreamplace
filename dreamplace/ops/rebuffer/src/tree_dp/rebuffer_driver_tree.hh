// SPDX-License-Identifier: BSD-3-Clause
#pragma once

// Canonical input-slew selection over one instance's pins; storage is borrowed.

#include <algorithm>
#include <vector>

#include "rebuffer_liberty.hh"


namespace rebuffer {

template <typename MatchesSource, typename IsLoad>
bool driverInputSlewFromPins(int driver, int num_pins,
    const LibertyCellView& cell, const std::vector<int>& pins,
    const std::vector<double>& rise_slew, const std::vector<double>& fall_slew,
    const MatchesSource& matches_source, const IsLoad& is_load,
    double& input_rise, double& input_fall)
{
  bool found = false;
  double rise = 0.0, fall = 0.0;
  for (int pin : pins) {
    if (pin == driver || pin < 0 || pin >= num_pins) continue;
    for (const auto& arc : cell.arcs) {
      if (matches_source(arc, pin)) {
        rise = std::max(rise, rise_slew[pin]);
        fall = std::max(fall, fall_slew[pin]);
        found = true;
        break;
      }
    }
  }
  // Lightweight metadata can omit Liberty source names/ids. A physical load
  // of another net remains the existing unambiguous input fallback.
  if (!found) {
    for (int pin : pins) {
      if (pin == driver || !is_load(pin) || pin < 0 || pin >= num_pins) continue;
      rise = std::max(rise, rise_slew[pin]);
      fall = std::max(fall, fall_slew[pin]);
      found = true;
    }
  }
  if (found) { input_rise = rise; input_fall = fall; }
  return found;
}

}  // namespace rebuffer

// --- rebuffer_driver_tree_links.hh ---
#include <string>
#include <unordered_map>
#include <vector>
#include "rebuffer_types.hh"


namespace rebuffer {
struct CurrentDriverBufferLink
{
  int instance_id = -1;
  int input_pin_id = -1;
  int output_pin_id = -1;
  int input_net_id = -1;
  int output_net_id = -1;
};

bool collectCurrentDriverBufferLinks(const DesignView& design,
    std::vector<CurrentDriverBufferLink>& links,
    std::unordered_map<int, CurrentDriverBufferLink>& by_input_pin, std::string& reason);
}  // namespace rebuffer

// --- rebuffer_driver_tree.hh ---
// Native current driver-tree import/export for placement-RC buffering.


#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pybind11/pybind11.h"


#include "rebuffer_repair.hh"


namespace rebuffer {

enum class DriverTreeNodeKind : std::uint8_t
{
  driver = 0,
  sink = 1,
  logic_load = 2,
  steiner = 3,
  buffer_input = 4,
  buffer_output = 5,
  buffer = 6,
};

enum class DriverTreeEdgeKind : std::uint8_t
{
  wire = 0,
  buffer = 1,
};

enum class DriverTreeMutationKind : std::uint8_t
{
  keep = 0,
  remove = 1,
  rewrite = 2,
};

struct DriverTreeMutation
{
  int kind = static_cast<int>(DriverTreeMutationKind::keep);
  int node_id = -1;
  int source_instance_id = -1;
  int parent_node_id = -1;
  int child_node_id = -1;
  int target_cell_index = -1;
  int target_main_id = -1;
};

struct DriverTreeNode
{
  int id = -1;
  int source_node_id = -1;
  int source_pin_id = -1;
  int source_instance_id = -1;
  int master_id = -1;
  // ``master_id`` is the stable main-cell family identity.  ``cell_id`` is
  // the concrete Liberty variant currently used by the instance.  Keep them
  // separate: a rewrite changes the latter while preserving the former.
  int cell_id = -1;
  int net_id = -1;
  int partition_id = -1;
  int parent_node_id = -1;
  int kind = static_cast<int>(DriverTreeNodeKind::logic_load);
  bool existing_buffer = false;
  bool fixed = false;
  double x = 0.0;
  double y = 0.0;
  std::vector<int> child_node_ids;
  std::vector<int> load_pin_ids;
};

struct DriverTreeEdge
{
  int id = -1;
  int parent_node_id = -1;
  int child_node_id = -1;
  int net_id = -1;
  int partition_id = -1;
  int kind = static_cast<int>(DriverTreeEdgeKind::wire);
};

struct CurrentDriverTree
{
  int root_net_id = -1;
  int driver_pin_id = -1;
  int root_node_id = -1;
  int topology_epoch = 0;
  std::string tree_digest;
  std::vector<int> net_ids;
  std::vector<DriverTreeNode> nodes;
  std::vector<DriverTreeEdge> edges;
};

struct CurrentDriverTreeCheckpoint
{
  CurrentDriverTree tree;
};

class CurrentDriverTreeTransaction
{
 public:
  explicit CurrentDriverTreeTransaction(const CurrentDriverTree& source);

  pybind11::dict applyMutation(const pybind11::dict& values);
  pybind11::list areaRecoveryCandidates(const pybind11::dict& design,
                                        const pybind11::dict& liberty) const;
  pybind11::dict applyCandidateTree(
      const std::shared_ptr<CurrentDriverTree>& candidate,
      const std::string& source_tree_digest);
  pybind11::dict activeTree() const;
  pybind11::dict evaluateCandidate(const pybind11::dict& result);
  pybind11::dict acceptCandidate();
  std::shared_ptr<CurrentDriverTreeCheckpoint> checkpoint() const;
  pybind11::dict restore(
      const std::shared_ptr<CurrentDriverTreeCheckpoint>& checkpoint);
  pybind11::dict summary() const;

  const CurrentDriverTree& activeTreeRef() const { return active_; }

 private:
  CurrentDriverTree source_;
  CurrentDriverTree active_;
  std::string state_ = "CHECKPOINTED";
};

std::shared_ptr<CurrentDriverTreeTransaction> createCurrentDriverTreeTransaction(
    const std::shared_ptr<CurrentDriverTree>& source);

bool importCurrentDriverTree(const DesignView& design,
                             const TopologyView& topology,
                             int root_net_id,
                             CurrentDriverTree& tree,
                             std::string& reason,
                             bool stop_at_existing_buffers = false);
bool importCurrentDriverTree(const DesignView& design, const TopologyView& topology,
    int root_net_id, const std::vector<CurrentDriverBufferLink>& buffer_links,
    CurrentDriverTree& tree, std::string& reason,
    bool stop_at_existing_buffers = false);

bool validateCurrentDriverTree(const CurrentDriverTree& tree,
                               std::string& reason);

std::string currentDriverTreeDigest(const CurrentDriverTree& tree);

pybind11::dict currentDriverTreeToDict(const CurrentDriverTree& tree);
bool currentDriverTreeFromDict(const pybind11::dict& values,
                               CurrentDriverTree& tree,
                               std::string& reason);

std::vector<DriverTreeMutation> enumerateCurrentDriverTreeMutations(
    const CurrentDriverTree& tree,
    const DesignView& design,
    const LibertyView& liberty,
    std::string& reason);

bool applyCurrentDriverTreeMutation(const CurrentDriverTree& source,
                                    const DriverTreeMutation& mutation,
                                    CurrentDriverTree& target,
                                    std::string& reason);

bool applyCurrentDriverTreeBufferActions(
    const CurrentDriverTree& source,
    const LibertyView& liberty,
    const std::vector<BufferAction>& actions,
    CurrentDriverTree& target,
    std::string& reason,
    bool compute_digest = true);

pybind11::dict driverTreeMutationToDict(const DriverTreeMutation& mutation);
DriverTreeMutation driverTreeMutationFromDict(const pybind11::dict& values);

}  // namespace rebuffer

// --- rebuffer_tree_actions.hh ---

namespace rebuffer {
std::vector<pybind11::dict> orderTreeBufferActions(const pybind11::list& actions);
}  // namespace rebuffer

// --- rebuffer_tree_recovery.hh ---
// Native placement-RC tree area recovery candidate generation.


#include <vector>


namespace rebuffer {

struct TreeRecoveryCandidate
{
  DriverTreeMutation mutation;
  double area_saved = 0.0;
  int buffer_count_delta = -1;
  const char* recovery_kind = "remove_buffer_pass_through";
};

// Enumerate timing-preserving recovery candidates structurally.  Actual timing
// acceptance remains the Python TimingPropagation transaction callback; this
// function only generates valid one-buffer pass-through mutations and orders
// them by native cell area.
std::vector<TreeRecoveryCandidate> enumerateTreeAreaRecovery(
    const CurrentDriverTree& tree,
    const DesignView& design,
    const LibertyView& liberty,
    std::string& reason);

// Run the native recovery transaction. Python callbacks only own the physical
// placeio_ecc carrier mutation and Python TimingPropagation observation;
// candidate order,
// native tree lifecycle, envelope acceptance, restore and ledger stay native.
// physical_checkpoint receives (label, mutation) so the bridge can snapshot
// the stable physical instance identity before applying the mutation.
pybind11::dict runTreeAreaRecovery(
    const std::shared_ptr<CurrentDriverTreeTransaction>& transaction,
    const pybind11::dict& design,
    const pybind11::dict& liberty,
    const pybind11::object& physical_checkpoint,
    const pybind11::object& physical_apply,
    const pybind11::object& evaluate,
    const pybind11::object& physical_restore,
    const pybind11::dict& config);

}  // namespace rebuffer
