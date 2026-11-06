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
  const auto cached = views_.find(root);
  if (cached != views_.end()) return cached->second;
  requireCurrent();
  requireLocal(root >= 0 && root < net_count_, "local tree root is out of range");
  auto result = std::make_shared<LocalPlanningView>();
  const auto nets = reachable(root);
  result->nets.insert(nets.begin(), nets.end());
  auto started = std::chrono::steady_clock::now();
  buildTopology(*result);
  timings_["local_topology"] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  started = std::chrono::steady_clock::now();
  buildViews(*result);
  timings_["local_views"] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
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
  requireLocal(importCurrentDriverTree(p.design, topology, root, links_, *result->tree, reason,
                                       stop_at_existing_buffers), reason);
  timings_["tree_import"] += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started).count();
  views_.emplace(root, result);
  return result;
}

void NativeTimingLocalPlanningSnapshot::buildTopology(LocalPlanningView& out)
{
  const auto& m = *graph_->mutation_state_;
  auto& p = out.projection;
  p.topologies.resize(net_count_);
  p.steiner.steiner_node_base = pin_count_;
  p.steiner.pin_fa.assign(vertex_count_, -1);
  p.steiner.node_x.resize(vertex_count_);
  p.steiner.node_y.resize(vertex_count_);
  p.actions.original_net_by_net.resize(net_count_);
  std::iota(p.actions.original_net_by_net.begin(), p.actions.original_net_by_net.end(), 0);
  for (const auto& [net, owner] : original_by_net_) p.actions.original_net_by_net[net] = owner;
  p.actions.state_node_by_vertex.assign(vertex_count_, -1);
  p.actions.state_net_by_vertex.assign(vertex_count_, -1);
  for (const int net : out.nets) {
    const int owner = originalNet(net), native_net = graphNet(net);
    const int driver = m.net_drivers.at(native_net);
    requireLocal(driver >= 0, "local partition has no driver");
    int source_root = driver;
    if (driver >= inputs_->design.num_pins) {
      const auto& source = source_by_pin_.at(pinId(driver));
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
