// SPDX-License-Identifier: BSD-3-Clause
// Build only selected wire partitions, preserving the full projection's IDs.
#include "rebuffer_timing_local.hh"
#include "rebuffer_timing_graph.hh"
#include "rebuffer_timing_buffer_state.hh"

#include <algorithm>
#include <chrono>
#include <functional>
#include <numeric>
#include <cmath>

namespace rebuffer {

std::shared_ptr<LocalPlanningView> NativeTimingLocalPlanningSnapshot::view(int root)
{
  if (last_view_ != nullptr && last_view_root_ == root) return last_view_;
  requireCurrent();
  requireLocal(root >= 0 && root < net_count_, "local tree root is out of range");
  const auto existing_tree = trees_.find(root);
  const bool already_visited = existing_tree != trees_.end();
  const int prev_filled_vertices = filled_vertices_;
  const int prev_filled_pins = filled_pins_;
  std::shared_ptr<LocalPlanningView> result;
  if (last_view_ != nullptr && last_view_.use_count() == 1) {
    resetLocalPlanningView(*last_view_);
    result = std::move(last_view_);
    last_view_root_ = -1;
  } else if (!inputs_->local_view_pool.empty()) {
    result = std::move(inputs_->local_view_pool.back());
    inputs_->local_view_pool.pop_back();
  } else {
    result = std::make_shared<LocalPlanningView>();
  }
  const auto nets = reachable(root);
  result->nets.insert(nets.begin(), nets.end());
  auto started = std::chrono::steady_clock::now();
  buildTopology(*result);
  timings_["local_topology"] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  started = std::chrono::steady_clock::now();
  buildViews(*result);
  timings_["local_views"] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  if (already_visited) {
    filled_vertices_ = prev_filled_vertices;
    filled_pins_ = prev_filled_pins;
    result->tree = existing_tree->second;
  } else {
    started = std::chrono::steady_clock::now();
    TopologyView topology;
    const auto& p = result->projection;
    topology.steiner_node_base = pin_count_;
    topology.net_ids = nets;
    topology.net_flat_topo_sort_start.push_back(0);
    topology.pin_fa = std::move(result->projection.steiner.pin_fa);
    for (const int net : nets) {
      const auto& nodes = p.topologies.at(net);
      topology.net_flat_topo_sort.insert(topology.net_flat_topo_sort.end(), nodes.begin(), nodes.end());
      for (const int node : nodes) {
        topology.node_x.emplace(node, p.steiner.node_x[node]);
        topology.node_y.emplace(node, p.steiner.node_y[node]);
      }
      topology.net_flat_topo_sort_start.push_back(topology.net_flat_topo_sort.size());
    }
    result->tree = std::make_shared<CurrentDriverTree>();
    std::string reason;
    const bool stop_at_existing_buffers = config_.contains("native_tree_stop_at_existing_buffers")
        && config_["native_tree_stop_at_existing_buffers"].cast<bool>();
    std::vector<std::size_t> local_link_indices;
    for (const int net : result->nets) {
      if (net >= 0 && net < static_cast<int>(identity_->link_by_output.size())
          && identity_->link_by_output[net] >= 0) {
        local_link_indices.push_back(static_cast<std::size_t>(identity_->link_by_output[net]));
      }
      auto it = std::lower_bound(
          identity_->links.begin(), identity_->links.end(), net,
          [](const CurrentDriverBufferLink& link, int target) {
            return link.input_net_id < target;
          });
      for (; it != identity_->links.end() && it->input_net_id == net; ++it) {
        local_link_indices.push_back(static_cast<std::size_t>(it - identity_->links.begin()));
      }
    }
    std::sort(local_link_indices.begin(), local_link_indices.end());
    local_link_indices.erase(
        std::unique(local_link_indices.begin(), local_link_indices.end()),
        local_link_indices.end());
    std::vector<CurrentDriverBufferLink> local_links;
    local_links.reserve(local_link_indices.size());
    for (const std::size_t idx : local_link_indices) {
      local_links.push_back(identity_->links[idx]);
    }
    const bool imported = importCurrentDriverTree(
        p.design, topology, root, local_links, *result->tree, reason, stop_at_existing_buffers);
    result->projection.steiner.pin_fa = std::move(topology.pin_fa);
    requireLocal(imported, reason);
    timings_["tree_import"] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
    trees_.emplace(root, result->tree);
  }
  last_view_ = result;
  last_view_root_ = root;
  return result;
}

void NativeTimingLocalPlanningSnapshot::buildTopology(LocalPlanningView& out)
{
  const auto& m = *graph_->mutation_state_;
  const int base_nets = inputs_->design.num_nets;
  auto& p = out.projection;
  p.topologies.resize(net_count_);
  p.steiner.steiner_node_base = pin_count_;
  p.steiner.pin_fa.resize(vertex_count_, -1);
  p.steiner.node_x.resize(vertex_count_, 0.0);
  p.steiner.node_y.resize(vertex_count_, 0.0);
  if (static_cast<int>(p.actions.original_net_by_net.size()) < base_nets) {
    p.actions.original_net_by_net.resize(base_nets);
    std::iota(p.actions.original_net_by_net.begin(),
              p.actions.original_net_by_net.begin() + base_nets, 0);
  }
  if (out.cached_topology_revision != identity_->topology_revision
      || static_cast<int>(p.actions.original_net_by_net.size()) != net_count_) {
    p.actions.original_net_by_net.resize(net_count_);
    for (std::size_t i = 0; i < identity_->original_by_net.size(); ++i) {
      p.actions.original_net_by_net[base_nets + static_cast<int>(i)] = identity_->original_by_net[i];
    }
  }
  p.actions.state_node_by_vertex.resize(vertex_count_, -1);
  p.actions.state_net_by_vertex.resize(vertex_count_, -1);
  for (const int net : out.nets) {
    const int owner = originalNet(net), native_net = graphNet(net);
    const int driver = m.net_drivers.at(native_net);
    requireLocal(driver >= 0, "local partition has no driver");
    int source_root = driver;
    if (driver >= inputs_->design.num_pins) {
      const int canonical_pin = pinId(driver);
      const auto& source = identity_->source_by_pin.at(canonical_pin - inputs_->design.num_pins);
      requireLocal(source.first == owner, "local driver has a different native topology owner");
      source_root = source.second;
    }
    auto add = [&](int source, int parent, double x, double y) {
      const int id = vertex(owner, source);
      requireLocal(id >= 0 && id < vertex_count_, "local vertex identity is out of range");
      p.topologies[net].push_back(id);
      p.steiner.pin_fa[id] = parent < 0 ? -1 : vertex(owner, parent);
      p.steiner.node_x[id] = x; p.steiner.node_y[id] = y;
      p.actions.state_node_by_vertex[id] = source;
      p.actions.state_net_by_vertex[id] = owner;
      p.actions.state_provenance.push_back({{net, id}, {owner, source}});
      out.touched_vertices.push_back(id);
      if (id < pin_count_) out.pins.insert(id);
    };
    const auto current = graph_->net_states_.find(owner);
    if (current == graph_->net_states_.end()) {
      requireLocal(net == owner, "synthetic partition has no mutable native topology");
      const auto& s = inputs_->steiner;
      for (int pos = s.net_flat_topo_sort_start.at(owner); pos < s.net_flat_topo_sort_start.at(owner + 1); ++pos) {
        const int source = s.net_flat_topo_sort.at(pos);
        if (source != source_root) {
          const double resistance = s.edge_resistance.at(source);
          const double capacitance = resistance / inputs_->r_unit * inputs_->c_unit;
          requireLocal(std::isfinite(resistance) && resistance >= 0
                           && std::isfinite(capacitance) && capacitance >= 0,
                       "local wire edge has invalid RC value");
        }
        add(source, source == source_root ? -1 : s.pin_fa.at(source), s.node_x.at(source), s.node_y.at(source));
      }
    } else {
      const auto& state = *current->second;
      std::string reason;
      requireLocal(validateIncrementalTopology(state, reason), reason);
      std::set<std::pair<int, int>> cells;
      for (const auto& buffer : state.buffers) cells.emplace(buffer.input_node, buffer.output_node);
      std::map<int, std::vector<int>> children;
      for (const auto& edge : state.edges) {
        if (!cells.count({edge.parent, edge.child})) children[edge.parent].push_back(edge.child);
      }
      std::set<int> seen;
      std::function<void(int, int)> visit = [&](int source, int parent) {
        requireLocal(seen.insert(source).second, "local wire partition contains a cycle");
        const auto& point = state.coordinates.at(source);
        add(source, parent, point[0], point[1]);
        for (const int child : children[source]) visit(child, source);
      };
      visit(source_root, -1);
    }
    std::vector<int> actual, expected;
    for (const int id : p.topologies[net]) {
      if (id < pin_count_) actual.push_back(id);
    }
    for (const int pin : m.net_pins.at(native_net)) if (m.pin_active.at(pin)) expected.push_back(pinId(pin));
    std::sort(actual.begin(), actual.end()); std::sort(expected.begin(), expected.end());
    requireLocal(actual == expected, "local partition pins differ from native timing owner");
    requireLocal(std::find(actual.begin(), actual.end(), pinId(driver)) != actual.end(), "local driver is absent from its partition");
  }
  normalizePartitionProvenance(p.actions.state_provenance);
  filled_vertices_ += p.actions.state_provenance.size();
}

}  // namespace rebuffer
