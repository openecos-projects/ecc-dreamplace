// SPDX-License-Identifier: BSD-3-Clause
// Native current driver-tree import/export for placement-RC buffering.

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "pybind11/pybind11.h"

#include "rebuffer_types.hh"
#include "rebuffer_liberty.hh"
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
                             std::string& reason);

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
    std::string& reason);

pybind11::dict driverTreeMutationToDict(const DriverTreeMutation& mutation);
DriverTreeMutation driverTreeMutationFromDict(const pybind11::dict& values);

}  // namespace rebuffer
