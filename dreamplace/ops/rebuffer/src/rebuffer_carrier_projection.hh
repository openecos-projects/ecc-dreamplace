// SPDX-License-Identifier: BSD-3-Clause
// Owned placement/tree projection shared by Python export and native planning.
#pragma once

#include <map>
#include <string>
#include <utility>
#include <vector>

#include "rebuffer_repair.hh"

namespace rebuffer {

struct OverlayActionProjection
{
  std::vector<int> original_net_by_net, state_node_by_vertex, state_net_by_vertex;
  std::map<std::pair<int, int>, std::pair<int, int>> state_provenance;
  bool has_partition_provenance = true;
};

struct ProjectedBufferIdentity
{
  int action_id, cell, node, input_pin, output_pin, output_net;
};

struct OverlayCarrierProjection
{
  DesignView design;
  SteinerView steiner;
  std::vector<std::string> node_names, node_master_names, net_names, pin_direct;
  std::vector<double> node_size_x, node_size_y;
  std::vector<std::vector<int>> net_pins, topologies, children;
  std::vector<ProjectedBufferIdentity> buffers;
  OverlayActionProjection actions;
};

}  // namespace rebuffer
