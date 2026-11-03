// SPDX-License-Identifier: BSD-3-Clause
#include "rebuffer_incremental_rc.hh"
#include "rebuffer_incremental_topology.hh"

#include <algorithm>
#include <cstdint>
#include <stdexcept>
#include <unordered_map>
#include <utility>

namespace rebuffer {

CompactRcForest compactRcForest(const IncrementalTopologyState& state,
                                int original_pin_count)
{
  CompactRcForest result;
  auto& ids = result.node_ids;
  ids.push_back(state.driver_pin_id);
  for (const auto& edge : state.edges) {
    ids.push_back(edge.parent); ids.push_back(edge.child);
  }
  for (const auto& [node, _] : state.node_loads) ids.push_back(node);
  std::unordered_map<int, int> buffer_slots, buffer_outputs;
  for (int slot = 0; slot < static_cast<int>(state.buffers.size()); ++slot) {
    const auto& buffer = state.buffers[slot];
    ids.push_back(buffer.input_node); ids.push_back(buffer.output_node);
    if (!buffer_slots.emplace(buffer.input_node, 2 * slot).second
        || !buffer_slots.emplace(buffer.output_node, 2 * slot + 1).second) {
      throw std::invalid_argument("native timing RC node namespace collides");
    }
    buffer_outputs.emplace(buffer.input_node, buffer.output_node);
  }
  std::sort(ids.begin(), ids.end());
  ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
  if (ids.front() < 0) throw std::invalid_argument("persistent timing graph RC node is negative");
  // Preserve the former dense adapter's wire-cap accumulation order: original
  // physical pins, current buffer slots, then shifted Steiner identities.
  auto key = [&](int node) -> std::int64_t {
    const auto slot = buffer_slots.find(node);
    if (slot != buffer_slots.end()) return static_cast<std::int64_t>(original_pin_count) + slot->second;
    return node < original_pin_count ? node
        : static_cast<std::int64_t>(node) + 2 * state.buffers.size();
  };
  std::sort(ids.begin(), ids.end(), [&](int a, int b) { return key(a) < key(b); });
  for (std::size_t i = 1; i < ids.size(); ++i) {
    if (key(ids[i - 1]) == key(ids[i])) {
      throw std::invalid_argument("native timing RC node namespace collides");
    }
  }
  std::unordered_map<int, int> local;
  for (int i = 0; i < static_cast<int>(ids.size()); ++i) local.emplace(ids[i], i);
  result.parent.assign(ids.size(), -1);
  result.resistance.assign(ids.size(), 0);
  result.edge_capacitance.assign(ids.size(), 0);
  std::vector<std::vector<int>> children(ids.size());
  for (const auto& edge : state.edges) {
    const auto output = buffer_outputs.find(edge.parent);
    if (output != buffer_outputs.end() && output->second == edge.child) continue;
    const int source = local.at(edge.parent), sink = local.at(edge.child);
    if (result.parent[sink] >= 0 && result.parent[sink] != source) {
      throw std::invalid_argument("persistent timing graph RC node has multiple parents");
    }
    result.parent[sink] = source;
    children[source].push_back(sink);
    result.resistance[sink] = edge.resistance;
    result.edge_capacitance[sink] = edge.capacitance;
  }
  std::vector<int> roots{state.driver_pin_id};
  for (const auto& buffer : state.buffers) roots.push_back(buffer.output_node);
  std::vector<unsigned char> visited(ids.size(), 0);
  result.topo_start.push_back(0);
  for (int root_id : roots) {
    const int root = local.at(root_id);
    if (result.parent[root] >= 0 && result.parent[root] != root) {
      throw std::invalid_argument("persistent timing graph RC root has a parent");
    }
    result.parent[root] = root;
    if (visited[root] == 2) continue;
    std::vector<std::pair<int, bool>> pending{{root, false}};
    while (!pending.empty()) {
      const auto [node, leaving] = pending.back(); pending.pop_back();
      if (leaving) { visited[node] = 2; continue; }
      if (visited[node] == 2) continue;
      if (visited[node] == 1) throw std::invalid_argument("persistent timing graph RC forest has a cycle");
      visited[node] = 1;
      result.topo.push_back(node);
      pending.emplace_back(node, true);
      for (auto it = children[node].rbegin(); it != children[node].rend(); ++it) {
        if (visited[*it] == 1) throw std::invalid_argument("persistent timing graph RC forest has a cycle");
        pending.emplace_back(*it, false);
      }
    }
    result.topo_start.push_back(result.topo.size());
  }
  if (std::find(visited.begin(), visited.end(), 0) != visited.end()) {
    throw std::invalid_argument("persistent timing graph RC forest is disconnected");
  }
  result.child_start.push_back(0);
  for (const auto& row : children) {
    result.children.insert(result.children.end(), row.begin(), row.end());
    result.child_start.push_back(result.children.size());
  }
  return result;
}

std::vector<double> compactRcWireCaps(const CompactRcForest& forest)
{
  std::vector<double> caps(forest.parent.size(), 0);
  for (int node = 0; node < static_cast<int>(caps.size()); ++node) {
    if (forest.parent[node] == node) continue;
    const double half = forest.edge_capacitance[node] * 0.5;
    caps[node] += half;
    caps[forest.parent[node]] += half;
  }
  return caps;
}

std::array<std::vector<double>, 3> evaluateCompactRc(
    const CompactRcForest& forest, const std::vector<double>& caps)
{
  const int count = forest.parent.size();
  std::vector<double> load(count), delay(count), ldelay(count), beta(count), impulse(count);
  for (std::size_t tree = 0; tree + 1 < forest.topo_start.size(); ++tree) {
    const int first = forest.topo_start[tree], last = forest.topo_start[tree + 1];
    for (int row = last; row-- > first;) {
      const int node = forest.topo[row];
      double value = caps[node];
      for (int edge = forest.child_start[node]; edge < forest.child_start[node + 1]; ++edge) {
        value += load[forest.children[edge]];
      }
      load[node] = value;
    }
    for (int row = first; row < last; ++row) {
      const int node = forest.topo[row], parent = forest.parent[node];
      delay[node] = parent == node ? 0.0 : delay[parent] + forest.resistance[node] * load[node];
    }
    for (int row = last; row-- > first;) {
      const int node = forest.topo[row];
      double value = caps[node] * delay[node];
      for (int edge = forest.child_start[node]; edge < forest.child_start[node + 1]; ++edge) {
        value += ldelay[forest.children[edge]];
      }
      ldelay[node] = value;
    }
    for (int row = first; row < last; ++row) {
      const int node = forest.topo[row], parent = forest.parent[node];
      beta[node] = parent == node ? 0.0 : beta[parent] + forest.resistance[node] * ldelay[node];
      impulse[node] = 2.0 * beta[node] - delay[node] * delay[node];
    }
  }
  return {std::move(load), std::move(delay), std::move(impulse)};
}

}  // namespace rebuffer
