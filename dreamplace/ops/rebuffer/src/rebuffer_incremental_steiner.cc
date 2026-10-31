// SPDX-License-Identifier: BSD-3-Clause
// Import one existing native Steiner tree into a mutable electrical view.

#include "rebuffer_incremental_topology.hh"
#include "rebuffer_repair.hh"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

namespace rebuffer {

std::shared_ptr<IncrementalTopologyState> incrementalTopologyFromSteiner(
    const SteinerView& steiner, int net_id, int driver_pin_id,
    double r_unit, double c_unit)
{
  if (!std::isfinite(r_unit) || !std::isfinite(c_unit)
      || r_unit <= 0.0 || c_unit < 0.0) {
    throw std::invalid_argument("invalid_wire_rc_units");
  }
  int index = net_id;
  if (!steiner.net_ids.empty()) {
    const auto found = std::find(steiner.net_ids.begin(), steiner.net_ids.end(), net_id);
    index = static_cast<int>(found - steiner.net_ids.begin());
    if (found == steiner.net_ids.end()) index = -1;
  }
  if (index < 0 || index + 1 >= static_cast<int>(steiner.net_flat_topo_sort_start.size())) {
    throw std::invalid_argument("unknown_steiner_net_id");
  }
  const int begin = steiner.net_flat_topo_sort_start[index];
  const int end = steiner.net_flat_topo_sort_start[index + 1];
  if (begin < 0 || end <= begin || end > static_cast<int>(steiner.net_flat_topo_sort.size())) {
    throw std::invalid_argument("invalid_steiner_net_range");
  }
  const std::unordered_set<int> nodes(
      steiner.net_flat_topo_sort.begin() + begin,
      steiner.net_flat_topo_sort.begin() + end);
  if (nodes.size() != static_cast<std::size_t>(end - begin)
      || nodes.count(driver_pin_id) == 0) {
    throw std::invalid_argument("invalid_steiner_net_nodes_or_driver");
  }
  auto state = std::make_shared<IncrementalTopologyState>();
  state->net_id = net_id;
  state->driver_pin_id = driver_pin_id;
  for (int offset = begin; offset < end; ++offset) {
    const int node = steiner.net_flat_topo_sort[offset];
    if (node < 0 || node >= static_cast<int>(steiner.pin_fa.size())
        || node >= static_cast<int>(steiner.node_x.size())
        || node >= static_cast<int>(steiner.node_y.size())
        || node >= static_cast<int>(steiner.edge_resistance.size())) {
      throw std::invalid_argument("steiner_node_arrays_do_not_align");
    }
    state->next_node_id = std::max(state->next_node_id, node + 1);
    state->coordinates[node] = {steiner.node_x[node], steiner.node_y[node]};
    if (node == driver_pin_id) continue;
    const int parent = steiner.pin_fa[node];
    if (nodes.count(parent) == 0) {
      throw std::invalid_argument("steiner_parent_outside_selected_net");
    }
    const double resistance = steiner.edge_resistance[node];
    state->edges.push_back({parent, node, resistance, resistance / r_unit * c_unit});
    if (node < steiner.steiner_node_base) state->node_loads[node] = {node};
  }
  std::string reason;
  if (!validateIncrementalTopology(*state, reason)) throw std::invalid_argument(reason);
  return state;
}

}  // namespace rebuffer
