// SPDX-License-Identifier: BSD-3-Clause
// Native data contracts shared by the rebuffer planner stages.

#pragma once

#include <string>
#include <cstdint>
#include <unordered_map>
#include <vector>

namespace rebuffer {

struct DesignView
{
  int dbu = 0;
  int num_nets = 0;
  int num_pins = 0;
  int num_nodes = 0;

  std::vector<int> flat_net2pin_map;
  std::vector<int> flat_net2pin_start_map;
  std::vector<int> net2driver_pin_map;
  std::vector<int> pin2node_map;
  std::vector<int> pin2net_map;
  std::vector<double> pin_offset_x;
  std::vector<double> pin_offset_y;
  std::vector<double> node_x;
  std::vector<double> node_y;
  std::vector<std::string> pin_names;

  // Optional electrical metadata.  Empty vectors mean the caller did not
  // provide the field; the native planner must fail closed rather than guess.
  std::vector<double> pin_cap_limit;
  std::vector<double> pin_slew_limit;
  std::vector<double> pin_drive_resistance;
  std::vector<int> inst_main_id;
  std::vector<int> inst_libcell_offset;
  std::vector<int> main_id_2_cell_id_start;
  std::vector<int> cell_id_2_libpin_id_start;
  std::vector<int> pin_2_libpin_offset;
  std::vector<std::uint8_t> node_is_buffer;
  std::vector<std::uint8_t> node_is_fixed;
};

struct TopologyView
{
  // Topology node ids are global pin ids for pin nodes and ids at or above
  // steiner_node_base for synthesized Steiner nodes.
  int steiner_node_base = 0;
  std::vector<int> net_ids;
  std::vector<int> net_flat_topo_sort_start;
  std::vector<int> net_flat_topo_sort;
  std::vector<int> pin_fa;
  std::unordered_map<int, double> node_x;
  std::unordered_map<int, double> node_y;
};

struct NetForestSummary
{
  int net_id = -1;
  int driver_pin = -1;
  int root_node = -1;
  int node_count = 0;
  int pin_node_count = 0;
  int steiner_node_count = 0;
  int edge_count = 0;
  int max_depth = 0;
  int max_fanout = 0;
  bool driver_in_topology = false;
};

struct ForestResult
{
  int num_nets = 0;
  int num_nodes = 0;
  int num_pin_nodes = 0;
  int num_steiner_nodes = 0;
  int num_edges = 0;
  int max_depth = 0;
  int max_fanout = 0;

  std::vector<int> net_node_start;
  std::vector<int> net_node_ids;
  std::vector<int> node_parent;
  std::vector<int> node_depth;
  std::vector<int> node_fanout;
  std::vector<NetForestSummary> net_summaries;
};

bool validate_design(const DesignView& design, std::string& reason);
bool build_forest(const DesignView& design,
                  const TopologyView& topology,
                  ForestResult& forest,
                  std::string& reason);

}  // namespace rebuffer
