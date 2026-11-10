// SPDX-License-Identifier: BSD-3-Clause

#include "rebuffer_types.hh"

#include <algorithm>
#include <cstddef>
#include <unordered_map>
#include <vector>

namespace rebuffer {

namespace {

bool valid_topology(const DesignView& design,
                    const TopologyView& topology,
                    std::string& reason)
{
  if (topology.steiner_node_base < 0) {
    reason = "invalid_steiner_node_base";
    return false;
  }
  if (topology.net_flat_topo_sort_start.size() < 2
      || topology.net_flat_topo_sort_start.front() != 0
      || topology.net_flat_topo_sort_start.back()
             != static_cast<int>(topology.net_flat_topo_sort.size())) {
    reason = "invalid_topology_csr";
    return false;
  }
  if (topology.net_flat_topo_sort_start.size() - 1 != topology.net_ids.size()) {
    reason = "topology_net_count_mismatch";
    return false;
  }
  if (topology.pin_fa.empty()) {
    reason = "empty_pin_fa";
    return false;
  }
  for (std::size_t index = 1; index < topology.net_flat_topo_sort_start.size();
       ++index) {
    if (topology.net_flat_topo_sort_start[index]
        < topology.net_flat_topo_sort_start[index - 1]) {
      reason = "topology_csr_not_monotonic";
      return false;
    }
  }
  std::vector<bool> seen(design.num_nets, false);
  for (const int net_id : topology.net_ids) {
    if (net_id < 0 || net_id >= design.num_nets || seen[net_id]) {
      reason = "invalid_or_duplicate_topology_net";
      return false;
    }
    seen[net_id] = true;
  }
  for (const int node : topology.net_flat_topo_sort) {
    if (node < 0 || node >= static_cast<int>(topology.pin_fa.size())) {
      reason = "topology_node_out_of_range";
      return false;
    }
  }
  return true;
}

}  // namespace

bool build_forest(const DesignView& design,
                  const TopologyView& topology,
                  ForestResult& forest,
                  std::string& reason)
{
  if (!validate_design(design, reason)) {
    return false;
  }
  if (!valid_topology(design, topology, reason)) {
    return false;
  }

  forest = ForestResult{};
  forest.num_nets = static_cast<int>(topology.net_ids.size());
  forest.net_node_start.reserve(forest.num_nets + 1);
  forest.net_node_start.push_back(0);
  forest.net_summaries.reserve(forest.num_nets);

  for (int net_index = 0; net_index < forest.num_nets; ++net_index) {
    const int net_id = topology.net_ids[net_index];
    const int begin = topology.net_flat_topo_sort_start[net_index];
    const int end = topology.net_flat_topo_sort_start[net_index + 1];
    if (begin == end) {
      reason = "empty_topology_net";
      return false;
    }

    const int node_count = end - begin;
    std::unordered_map<int, int> local_index;
    local_index.reserve(node_count);
    for (int local = 0; local < node_count; ++local) {
      const int global = topology.net_flat_topo_sort[begin + local];
      if (!local_index.emplace(global, local).second) {
        reason = "duplicate_topology_node_in_net";
        return false;
      }
    }

    NetForestSummary summary;
    summary.net_id = net_id;
    summary.driver_pin = design.net2driver_pin_map[net_id];
    summary.node_count = node_count;

    std::vector<int> depth(node_count, -1);
    std::vector<int> fanout(node_count, 0);
    int root_index = -1;
    for (int local = 0; local < node_count; ++local) {
      const int global = topology.net_flat_topo_sort[begin + local];
      const int parent = topology.pin_fa[global];
      if (parent < 0) {
        if (root_index >= 0) {
          reason = "multiple_topology_roots";
          return false;
        }
        root_index = local;
        depth[local] = 0;
        summary.root_node = global;
        continue;
      }
      const auto parent_it = local_index.find(parent);
      if (parent_it == local_index.end() || parent_it->second >= local) {
        reason = "parent_not_before_child_in_topology";
        return false;
      }
      depth[local] = depth[parent_it->second] + 1;
      ++fanout[parent_it->second];
    }
    if (root_index < 0) {
      reason = "missing_topology_root";
      return false;
    }

    for (int local = 0; local < node_count; ++local) {
      const int global = topology.net_flat_topo_sort[begin + local];
      const bool is_pin = global < topology.steiner_node_base;
      summary.pin_node_count += is_pin ? 1 : 0;
      summary.steiner_node_count += is_pin ? 0 : 1;
      summary.max_depth = std::max(summary.max_depth, depth[local]);
      summary.max_fanout = std::max(summary.max_fanout, fanout[local]);
      if (global == summary.driver_pin) {
        summary.driver_in_topology = true;
      }
    }
    summary.edge_count = node_count - 1;

    for (int local = 0; local < node_count; ++local) {
      const int global = topology.net_flat_topo_sort[begin + local];
      const int parent = topology.pin_fa[global];
      forest.net_node_ids.push_back(global);
      forest.node_parent.push_back(parent);
      forest.node_depth.push_back(depth[local]);
      forest.node_fanout.push_back(fanout[local]);
    }
    forest.net_node_start.push_back(
        static_cast<int>(forest.net_node_ids.size()));

    forest.num_nodes += summary.node_count;
    forest.num_pin_nodes += summary.pin_node_count;
    forest.num_steiner_nodes += summary.steiner_node_count;
    forest.num_edges += summary.edge_count;
    forest.max_depth = std::max(forest.max_depth, summary.max_depth);
    forest.max_fanout = std::max(forest.max_fanout, summary.max_fanout);
    forest.net_summaries.push_back(summary);
  }

  return true;
}

}  // namespace rebuffer
