// SPDX-License-Identifier: BSD-3-Clause
// Native current driver-tree import/export for placement-RC buffering.

#include "rebuffer_driver_tree.hh"
#include "rebuffer_driver_tree_links.hh"

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <map>
#include <queue>
#include <set>
#include <sstream>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "pybind11/stl.h"
#include "rebuffer_tree_recovery.hh"

namespace py = pybind11;

namespace rebuffer_python {
rebuffer::DesignView designFromDict(const py::dict& values);
rebuffer::LibertyView libertyFromDict(const py::dict& values);
}  // namespace rebuffer_python

namespace rebuffer {
namespace {

using BufferLink = CurrentDriverBufferLink;

using NetSlice = std::vector<int>;
using NodeKey = std::pair<int, int>;

std::string nodeKindName(const int kind)
{
  switch (static_cast<DriverTreeNodeKind>(kind)) {
    case DriverTreeNodeKind::driver:
      return "driver";
    case DriverTreeNodeKind::sink:
      return "sink";
    case DriverTreeNodeKind::logic_load:
      return "logic_load";
    case DriverTreeNodeKind::steiner:
      return "steiner";
    case DriverTreeNodeKind::buffer_input:
      return "buffer_input";
    case DriverTreeNodeKind::buffer_output:
      return "buffer_output";
    case DriverTreeNodeKind::buffer:
      return "buffer";
  }
  return "unknown";
}

void hashText(std::uint64_t& hash, const std::string& value)
{
  constexpr std::uint64_t kPrime = 1099511628211ULL;
  for (const unsigned char byte : value) {
    hash ^= byte;
    hash *= kPrime;
  }
  hash ^= 0xffU;
  hash *= kPrime;
}

template <typename T>
void hashValue(std::uint64_t& hash, const T value)
{
  hashText(hash, std::to_string(value));
}

void hashValue(std::uint64_t& hash, const double value)
{
  std::ostringstream stream;
  stream << std::setprecision(17) << value;
  hashText(hash, stream.str());
}

int topologySlice(const TopologyView& topology, int net_id)
{
  for (int index = 0; index < static_cast<int>(topology.net_ids.size()); ++index) {
    if (topology.net_ids[index] == net_id) {
      return index;
    }
  }
  return -1;
}

bool validateTopologyContract(const DesignView& design,
                              const TopologyView& topology,
                              std::map<int, NetSlice>& slices,
                              std::string& reason)
{
  if (design.dbu <= 0 || design.num_nets <= 0 || design.num_pins <= 0
      || design.num_nodes <= 0) {
    reason = "invalid_design_dimensions";
    return false;
  }
  if (design.net2driver_pin_map.size() != static_cast<std::size_t>(design.num_nets)
      || design.pin2node_map.size() != static_cast<std::size_t>(design.num_pins)
      || design.pin2net_map.size() != static_cast<std::size_t>(design.num_pins)
      || design.node_x.size() != static_cast<std::size_t>(design.num_nodes)
      || design.node_y.size() != static_cast<std::size_t>(design.num_nodes)) {
    reason = "design_identity_array_length_mismatch";
    return false;
  }
  if (design.node_is_buffer.size() != static_cast<std::size_t>(design.num_nodes)) {
    reason = "missing_node_is_buffer";
    return false;
  }
  if (!design.node_is_fixed.empty()
      && design.node_is_fixed.size() != static_cast<std::size_t>(design.num_nodes)) {
    reason = "node_is_fixed_length_mismatch";
    return false;
  }
  if (topology.net_flat_topo_sort_start.size() != topology.net_ids.size() + 1
      || topology.net_flat_topo_sort_start.empty()
      || topology.net_flat_topo_sort_start.front() != 0
      || topology.net_flat_topo_sort_start.back()
             != static_cast<int>(topology.net_flat_topo_sort.size())
      || topology.pin_fa.empty()) {
    reason = "invalid_topology_csr";
    return false;
  }

  std::set<int> seen_nets;
  for (int slice_index = 0;
       slice_index < static_cast<int>(topology.net_ids.size());
       ++slice_index) {
    const int net_id = topology.net_ids[slice_index];
    if (net_id < 0 || net_id >= design.num_nets || !seen_nets.insert(net_id).second) {
      reason = "invalid_or_duplicate_topology_net";
      return false;
    }
    const int begin = topology.net_flat_topo_sort_start[slice_index];
    const int end = topology.net_flat_topo_sort_start[slice_index + 1];
    if (begin >= end || begin < 0 || end > static_cast<int>(topology.net_flat_topo_sort.size())) {
      reason = "empty_or_invalid_topology_net";
      return false;
    }
    NetSlice nodes(topology.net_flat_topo_sort.begin() + begin,
                   topology.net_flat_topo_sort.begin() + end);
    std::unordered_set<int> node_set;
    for (const int node : nodes) {
      if (node < 0 || node >= static_cast<int>(topology.pin_fa.size())
          || !node_set.insert(node).second) {
        reason = "invalid_or_duplicate_topology_node";
        return false;
      }
    }
    for (const int node : nodes) {
      const int parent = topology.pin_fa[node];
      if (parent >= 0 && node_set.find(parent) == node_set.end()) {
        reason = "topology_parent_outside_net_slice";
        return false;
      }
    }
    slices.emplace(net_id, std::move(nodes));
  }
  return true;
}

bool pinCoordinate(const DesignView& design,
                   const TopologyView& topology,
                   const int node,
                   double& x,
                   double& y)
{
  const auto topology_x = topology.node_x.find(node);
  const auto topology_y = topology.node_y.find(node);
  if (topology_x != topology.node_x.end() && topology_y != topology.node_y.end()) {
    x = topology_x->second;
    y = topology_y->second;
    return true;
  }
  if (node < 0 || node >= design.num_pins) {
    return false;
  }
  const int instance = design.pin2node_map[node];
  if (instance < 0 || instance >= design.num_nodes) {
    return false;
  }
  x = design.node_x[instance];
  y = design.node_y[instance];
  if (node < static_cast<int>(design.pin_offset_x.size())) {
    x += design.pin_offset_x[node];
  }
  if (node < static_cast<int>(design.pin_offset_y.size())) {
    y += design.pin_offset_y[node];
  }
  return true;
}

bool orientNet(const DesignView& design,
               const TopologyView& topology,
               const NetSlice& nodes,
               const int net_id,
               std::vector<int>& order,
               std::unordered_map<int, int>& parent_by_node,
               std::string& reason)
{
  const int driver = design.net2driver_pin_map[net_id];
  const std::unordered_set<int> node_set(nodes.begin(), nodes.end());
  if (node_set.find(driver) == node_set.end()) {
    reason = "driver_not_in_topology_net";
    return false;
  }
  std::unordered_map<int, std::vector<int>> adjacency;
  for (const int node : nodes) {
    const int parent = topology.pin_fa[node];
    if (parent < 0) {
      continue;
    }
    adjacency[parent].push_back(node);
    adjacency[node].push_back(parent);
  }
  for (auto& [node, neighbors] : adjacency) {
    std::sort(neighbors.begin(), neighbors.end());
  }
  std::queue<int> queue;
  std::unordered_set<int> visited;
  queue.push(driver);
  visited.insert(driver);
  parent_by_node.clear();
  parent_by_node[driver] = -1;
  order.clear();
  while (!queue.empty()) {
    const int node = queue.front();
    queue.pop();
    order.push_back(node);
    for (const int next : adjacency[node]) {
      if (visited.insert(next).second) {
        parent_by_node[next] = node;
        queue.push(next);
      }
    }
  }
  if (visited.size() != node_set.size()) {
    reason = "current_net_tree_not_connected";
    return false;
  }
  int topology_edge_count = 0;
  for (const auto& [node, parent] : parent_by_node) {
    if (parent >= 0) {
      ++topology_edge_count;
    }
  }
  if (topology_edge_count != static_cast<int>(nodes.size()) - 1) {
    reason = "current_net_topology_has_cycle";
    return false;
  }
  return true;
}

int addNode(CurrentDriverTree& tree, DriverTreeNode node)
{
  node.id = static_cast<int>(tree.nodes.size());
  tree.nodes.push_back(std::move(node));
  return tree.nodes.back().id;
}

void addEdge(CurrentDriverTree& tree,
             const int parent,
             const int child,
             const int net_id,
             const int kind)
{
  DriverTreeEdge edge;
  edge.id = static_cast<int>(tree.edges.size());
  edge.parent_node_id = parent;
  edge.child_node_id = child;
  edge.net_id = net_id;
  edge.partition_id = net_id;
  edge.kind = kind;
  tree.edges.push_back(edge);
  tree.nodes[parent].child_node_ids.push_back(child);
  tree.nodes[child].parent_node_id = parent;
}

int findTreeEdge(const CurrentDriverTree& tree, const int parent, const int child)
{
  for (int index = 0; index < static_cast<int>(tree.edges.size()); ++index) {
    if (tree.edges[index].parent_node_id == parent
        && tree.edges[index].child_node_id == child) {
      return index;
    }
  }
  return -1;
}

int instanceCellId(const DesignView& design, const int instance_id)
{
  if (instance_id < 0
      || instance_id >= static_cast<int>(design.inst_main_id.size())
      || instance_id >= static_cast<int>(design.inst_libcell_offset.size())) {
    return -1;
  }
  const int main_id = design.inst_main_id[instance_id];
  if (main_id < 0
      || main_id + 1 >= static_cast<int>(design.main_id_2_cell_id_start.size())) {
    return -1;
  }
  const int first_cell = design.main_id_2_cell_id_start[main_id];
  const int cell_offset = design.inst_libcell_offset[instance_id];
  return cell_offset < 0 ? -1 : first_cell + cell_offset;
}

int graphNodeFor(const std::map<NodeKey, int>& graph_nodes,
                const int net_id,
                const int source_node_id)
{
  const auto found = graph_nodes.find({net_id, source_node_id});
  return found == graph_nodes.end() ? -1 : found->second;
}

std::string canonicalNodeText(const CurrentDriverTree& tree,
                              const int node_id,
                              std::map<int, std::string>& memo,
                              std::set<int>& active)
{
  const auto cached = memo.find(node_id);
  if (cached != memo.end()) {
    return cached->second;
  }
  if (!active.insert(node_id).second) {
    return "<cycle>";
  }
  const auto& node = tree.nodes[node_id];
  std::ostringstream text;
  text << "N|" << node.kind << '|'
       // Pin ids are stable identity. Synthetic Steiner ids and instance
       // indices are database allocation details and may shift after a
       // committed buffer adds pins/nets.
       << node.source_pin_id << '|'
       << node.master_id << '|' << node.cell_id << '|'
       << node.net_id << '|' << node.partition_id << '|'
       << (node.existing_buffer ? 1 : 0) << '|'
       << (node.fixed ? 1 : 0) << '|'
       << std::setprecision(17) << node.x << '|' << node.y << '|';
  std::vector<int> loads = node.load_pin_ids;
  std::sort(loads.begin(), loads.end());
  for (const int load : loads) {
    text << load << ',';
  }
  std::vector<std::string> children;
  for (const auto& edge : tree.edges) {
    if (edge.parent_node_id != node_id) {
      continue;
    }
    std::ostringstream child;
    child << "E|" << edge.kind << '|' << edge.net_id << '|'
          << edge.partition_id << '|'
          << canonicalNodeText(tree, edge.child_node_id, memo, active);
    children.push_back(child.str());
  }
  std::sort(children.begin(), children.end());
  text << "C|";
  for (const auto& child : children) {
    text << child << ';';
  }
  active.erase(node_id);
  const std::string result = text.str();
  memo.emplace(node_id, result);
  return result;
}

std::string digestFor(const CurrentDriverTree& tree)
{
  std::uint64_t hash = 1469598103934665603ULL;
  hashValue(hash, tree.root_net_id);
  hashValue(hash, tree.driver_pin_id);
  hashValue(hash, tree.topology_epoch);
  std::vector<int> net_ids = tree.net_ids;
  std::sort(net_ids.begin(), net_ids.end());
  for (const int net_id : net_ids) {
    hashValue(hash, net_id);
  }
  std::map<int, std::string> memo;
  std::set<int> active;
  hashText(hash, canonicalNodeText(tree, tree.root_node_id, memo, active));
  std::ostringstream result;
  result << std::hex << std::setw(16) << std::setfill('0') << hash;
  return result.str();
}

std::string partitionDigestFor(const CurrentDriverTree& tree)
{
  std::uint64_t hash = 1469598103934665603ULL;
  hashValue(hash, tree.root_net_id);
  for (const int net_id : tree.net_ids) {
    hashValue(hash, net_id);
  }
  for (const auto& node : tree.nodes) {
    hashValue(hash, node.net_id);
    hashValue(hash, node.partition_id);
  }
  for (const auto& edge : tree.edges) {
    hashValue(hash, edge.net_id);
    hashValue(hash, edge.partition_id);
  }
  std::ostringstream result;
  result << std::hex << std::setw(16) << std::setfill('0') << hash;
  return result.str();
}

py::dict treeIdentity(const CurrentDriverTree& tree)
{
  py::dict result;
  result["driver_pin_id"] = tree.driver_pin_id;
  result["source_root_instance_id"] =
      tree.root_node_id >= 0 && tree.root_node_id < static_cast<int>(tree.nodes.size())
          ? tree.nodes[tree.root_node_id].source_instance_id
          : -1;
  result["source_tree_digest"] = tree.tree_digest;
  result["topology_epoch"] = tree.topology_epoch;
  result["tree_digest_algorithm"] = "canonical_v2";
  result["current_net_partition_digest"] = partitionDigestFor(tree);
  return result;
}

template <typename T>
T requiredValue(const py::dict& values, const char* field)
{
  if (!values.contains(field)) {
    throw py::value_error(std::string("current driver tree export is missing ") + field);
  }
  return py::cast<T>(values[field]);
}

}  // namespace

bool importCurrentDriverTree(const DesignView& design,
                             const TopologyView& topology,
                             const int root_net_id,
                             CurrentDriverTree& tree,
                             std::string& reason)
{
  tree = CurrentDriverTree{};
  std::map<int, NetSlice> slices;
  if (!validateTopologyContract(design, topology, slices, reason)) {
    return false;
  }
  if (slices.find(root_net_id) == slices.end()) {
    reason = "root_net_missing_from_topology";
    return false;
  }
  const int root_driver = design.net2driver_pin_map[root_net_id];

  std::vector<BufferLink> buffer_links;
  std::unordered_map<int, BufferLink> buffer_by_input;
  if (!collectCurrentDriverBufferLinks(design, buffer_links, buffer_by_input, reason)) {
    return false;
  }
  std::vector<int> net_order;
  std::set<int> reachable_nets;
  std::queue<int> pending_nets;
  pending_nets.push(root_net_id);
  reachable_nets.insert(root_net_id);
  while (!pending_nets.empty()) {
    const int net_id = pending_nets.front();
    pending_nets.pop();
    net_order.push_back(net_id);
    for (const auto& link : buffer_links) {
      if (link.input_net_id != net_id) {
        continue;
      }
      if (slices.find(link.output_net_id) == slices.end()) {
        reason = "buffer_output_net_missing_from_topology";
        return false;
      }
      if (!reachable_nets.insert(link.output_net_id).second) {
        reason = "current_buffer_tree_contains_cycle";
        return false;
      }
      pending_nets.push(link.output_net_id);
    }
  }

  tree.root_net_id = root_net_id;
  tree.driver_pin_id = root_driver;
  tree.topology_epoch = 0;
  tree.net_ids = net_order;
  std::map<NodeKey, int> graph_nodes;
  for (const int net_id : net_order) {
    std::vector<int> order;
    std::unordered_map<int, int> parent_by_node;
    if (!orientNet(design,
                   topology,
                   slices.at(net_id),
                   net_id,
                   order,
                   parent_by_node,
                   reason)) {
      return false;
    }
    for (const int source_node : order) {
      const bool is_pin = source_node < topology.steiner_node_base;
      const int source_pin = is_pin ? source_node : -1;
      const int source_instance = is_pin ? design.pin2node_map[source_pin] : -1;
      const bool is_buffer_input = buffer_by_input.find(source_pin) != buffer_by_input.end();
      bool is_buffer_output = false;
      for (const auto& link : buffer_links) {
        if (link.output_pin_id == source_pin) {
          is_buffer_output = true;
          break;
        }
      }
      DriverTreeNode node;
      node.source_node_id = source_node;
      node.source_pin_id = source_pin;
      node.source_instance_id = source_instance;
      node.net_id = net_id;
      node.partition_id = net_id;
      node.existing_buffer = is_buffer_input || is_buffer_output;
      if (!is_pin) {
        node.kind = static_cast<int>(DriverTreeNodeKind::steiner);
      } else if (is_buffer_input) {
        node.kind = static_cast<int>(DriverTreeNodeKind::buffer_input);
      } else if (is_buffer_output) {
        node.kind = static_cast<int>(DriverTreeNodeKind::buffer_output);
      } else if (source_pin == design.net2driver_pin_map[net_id]) {
        node.kind = static_cast<int>(DriverTreeNodeKind::driver);
      } else {
        node.kind = static_cast<int>(DriverTreeNodeKind::logic_load);
        node.load_pin_ids.push_back(source_pin);
      }
      if (source_instance >= 0 && source_instance < design.num_nodes) {
        node.master_id = source_instance < static_cast<int>(design.inst_main_id.size())
                              ? design.inst_main_id[source_instance]
                              : -1;
        node.cell_id = instanceCellId(design, source_instance);
        node.fixed = !design.node_is_fixed.empty()
                     && design.node_is_fixed[source_instance] != 0;
      }
      if (!pinCoordinate(design, topology, source_node, node.x, node.y)) {
        reason = "missing_current_tree_node_coordinate";
        return false;
      }
      const int graph_id = addNode(tree, std::move(node));
      graph_nodes.emplace(NodeKey{net_id, source_node}, graph_id);
      if (net_id == root_net_id && source_node == root_driver) {
        tree.root_node_id = graph_id;
      }
    }
  }

  for (const int net_id : net_order) {
    std::unordered_map<int, int> parent_by_node;
    std::vector<int> order;
    if (!orientNet(design,
                   topology,
                   slices.at(net_id),
                   net_id,
                   order,
                   parent_by_node,
                   reason)) {
      return false;
    }
    for (const int source_node : order) {
      const int parent = parent_by_node.at(source_node);
      if (parent < 0) {
        continue;
      }
      const int parent_graph = graphNodeFor(graph_nodes, net_id, parent);
      const int child_graph = graphNodeFor(graph_nodes, net_id, source_node);
      if (parent_graph < 0 || child_graph < 0) {
        reason = "topology_edge_node_missing_from_current_tree";
        return false;
      }
      addEdge(tree,
              parent_graph,
              child_graph,
              net_id,
              static_cast<int>(DriverTreeEdgeKind::wire));
    }
  }

  for (const auto& link : buffer_links) {
    if (reachable_nets.find(link.input_net_id) == reachable_nets.end()) {
      continue;
    }
    const int input_graph = graphNodeFor(graph_nodes, link.input_net_id, link.input_pin_id);
    const int output_graph = graphNodeFor(graph_nodes, link.output_net_id, link.output_pin_id);
    if (input_graph < 0 || output_graph < 0) {
      reason = "buffer_bridge_pin_missing_from_current_tree";
      return false;
    }
    DriverTreeNode buffer;
    buffer.source_node_id = link.instance_id;
    buffer.source_instance_id = link.instance_id;
    buffer.master_id = link.instance_id < static_cast<int>(design.inst_main_id.size())
                           ? design.inst_main_id[link.instance_id]
                           : -1;
    buffer.cell_id = instanceCellId(design, link.instance_id);
    buffer.net_id = link.input_net_id;
    buffer.partition_id = link.input_net_id;
    buffer.kind = static_cast<int>(DriverTreeNodeKind::buffer);
    buffer.existing_buffer = true;
    buffer.fixed = !design.node_is_fixed.empty()
                   && design.node_is_fixed[link.instance_id] != 0;
    if (!pinCoordinate(design,
                       topology,
                       link.input_pin_id,
                       buffer.x,
                       buffer.y)) {
      reason = "missing_buffer_coordinate";
      return false;
    }
    const int buffer_graph = addNode(tree, std::move(buffer));
    addEdge(tree,
            input_graph,
            buffer_graph,
            link.input_net_id,
            static_cast<int>(DriverTreeEdgeKind::buffer));
    addEdge(tree,
            buffer_graph,
            output_graph,
            link.output_net_id,
            static_cast<int>(DriverTreeEdgeKind::buffer));
  }

  if (tree.root_node_id < 0) {
    reason = "current_tree_root_not_found";
    return false;
  }
  if (!validateCurrentDriverTree(tree, reason)) {
    return false;
  }
  tree.tree_digest = currentDriverTreeDigest(tree);
  return true;
}

bool validateCurrentDriverTree(const CurrentDriverTree& tree,
                               std::string& reason)
{
  if (tree.root_net_id < 0 || tree.driver_pin_id < 0 || tree.root_node_id < 0
      || tree.root_node_id >= static_cast<int>(tree.nodes.size())) {
    reason = "invalid_current_tree_root";
    return false;
  }
  if (tree.nodes.empty() || tree.edges.size() + 1 != tree.nodes.size()) {
    reason = "current_tree_must_be_single_tree";
    return false;
  }
  std::vector<int> parents(tree.nodes.size(), -1);
  std::vector<std::vector<int>> children(tree.nodes.size());
  for (const auto& edge : tree.edges) {
    if (edge.parent_node_id < 0
        || edge.parent_node_id >= static_cast<int>(tree.nodes.size())
        || edge.child_node_id < 0
        || edge.child_node_id >= static_cast<int>(tree.nodes.size())
        || edge.parent_node_id == edge.child_node_id) {
      reason = "current_tree_edge_out_of_range";
      return false;
    }
    if (parents[edge.child_node_id] >= 0) {
      reason = "current_tree_node_has_multiple_parents";
      return false;
    }
    parents[edge.child_node_id] = edge.parent_node_id;
    children[edge.parent_node_id].push_back(edge.child_node_id);
  }
  if (parents[tree.root_node_id] >= 0) {
    reason = "current_tree_root_has_parent";
    return false;
  }
  std::vector<int> stack{tree.root_node_id};
  std::vector<bool> visited(tree.nodes.size(), false);
  while (!stack.empty()) {
    const int node = stack.back();
    stack.pop_back();
    if (visited[node]) {
      reason = "current_tree_contains_cycle";
      return false;
    }
    visited[node] = true;
    for (const int child : children[node]) {
      stack.push_back(child);
    }
  }
  if (std::any_of(visited.begin(), visited.end(), [](const bool value) { return !value; })) {
    reason = "current_tree_has_disconnected_node";
    return false;
  }
  if (tree.nodes[tree.root_node_id].source_pin_id != tree.driver_pin_id) {
    reason = "current_tree_root_driver_pin_mismatch";
    return false;
  }
  return true;
}

std::string currentDriverTreeDigest(const CurrentDriverTree& tree)
{
  return digestFor(tree);
}

py::dict currentDriverTreeToDict(const CurrentDriverTree& tree)
{
  py::dict result;
  result["status"] = "ok";
  result["artifact"] = "rebuffer_current_driver_tree";
  result["artifact_version"] = 1;
  result["root_net_id"] = tree.root_net_id;
  result["driver_pin_id"] = tree.driver_pin_id;
  result["root_node_id"] = tree.root_node_id;
  result["topology_epoch"] = tree.topology_epoch;
  result["tree_digest"] = tree.tree_digest;
  result["tree_digest_algorithm"] = "canonical_v2";
  result["tree_identity"] = treeIdentity(tree);
  result["net_ids"] = tree.net_ids;
  py::list nodes;
  for (const auto& node : tree.nodes) {
    py::dict value;
    value["id"] = node.id;
    value["source_node_id"] = node.source_node_id;
    value["source_pin_id"] = node.source_pin_id;
    value["source_instance_id"] = node.source_instance_id;
    value["master_id"] = node.master_id;
    value["cell_id"] = node.cell_id;
    value["net_id"] = node.net_id;
    value["partition_id"] = node.partition_id;
    value["parent_node_id"] = node.parent_node_id;
    value["kind"] = node.kind;
    value["kind_name"] = nodeKindName(node.kind);
    value["existing_buffer"] = node.existing_buffer;
    value["fixed"] = node.fixed;
    value["x"] = node.x;
    value["y"] = node.y;
    value["child_node_ids"] = node.child_node_ids;
    value["load_pin_ids"] = node.load_pin_ids;
    nodes.append(std::move(value));
  }
  py::list edges;
  for (const auto& edge : tree.edges) {
    py::dict value;
    value["id"] = edge.id;
    value["parent_node_id"] = edge.parent_node_id;
    value["child_node_id"] = edge.child_node_id;
    value["net_id"] = edge.net_id;
    value["partition_id"] = edge.partition_id;
    value["kind"] = edge.kind;
    edges.append(std::move(value));
  }
  result["nodes"] = std::move(nodes);
  result["edges"] = std::move(edges);
  result["node_count"] = static_cast<int>(tree.nodes.size());
  result["edge_count"] = static_cast<int>(tree.edges.size());
  result["buffer_count"] = static_cast<int>(std::count_if(
      tree.nodes.begin(), tree.nodes.end(), [](const DriverTreeNode& node) {
        return node.kind == static_cast<int>(DriverTreeNodeKind::buffer);
      }));
  result["load_count"] = static_cast<int>(std::count_if(
      tree.nodes.begin(), tree.nodes.end(), [](const DriverTreeNode& node) {
        return !node.load_pin_ids.empty();
      }));
  return result;
}

bool currentDriverTreeFromDict(const py::dict& values,
                               CurrentDriverTree& tree,
                               std::string& reason)
{
  try {
    tree = CurrentDriverTree{};
    tree.root_net_id = requiredValue<int>(values, "root_net_id");
    tree.driver_pin_id = requiredValue<int>(values, "driver_pin_id");
    tree.root_node_id = requiredValue<int>(values, "root_node_id");
    tree.topology_epoch = requiredValue<int>(values, "topology_epoch");
    tree.net_ids = requiredValue<std::vector<int>>(values, "net_ids");
    for (const auto& item : values["nodes"].cast<py::sequence>()) {
      const py::dict value = item.cast<py::dict>();
      DriverTreeNode node;
      node.id = requiredValue<int>(value, "id");
      node.source_node_id = requiredValue<int>(value, "source_node_id");
      node.source_pin_id = requiredValue<int>(value, "source_pin_id");
      node.source_instance_id = requiredValue<int>(value, "source_instance_id");
      node.master_id = requiredValue<int>(value, "master_id");
      node.cell_id = requiredValue<int>(value, "cell_id");
      node.net_id = requiredValue<int>(value, "net_id");
      node.partition_id = requiredValue<int>(value, "partition_id");
      node.parent_node_id = requiredValue<int>(value, "parent_node_id");
      node.kind = requiredValue<int>(value, "kind");
      node.existing_buffer = requiredValue<bool>(value, "existing_buffer");
      node.fixed = requiredValue<bool>(value, "fixed");
      node.x = requiredValue<double>(value, "x");
      node.y = requiredValue<double>(value, "y");
      node.child_node_ids = requiredValue<std::vector<int>>(value, "child_node_ids");
      node.load_pin_ids = requiredValue<std::vector<int>>(value, "load_pin_ids");
      tree.nodes.push_back(std::move(node));
    }
    for (const auto& item : values["edges"].cast<py::sequence>()) {
      const py::dict value = item.cast<py::dict>();
      DriverTreeEdge edge;
      edge.id = requiredValue<int>(value, "id");
      edge.parent_node_id = requiredValue<int>(value, "parent_node_id");
      edge.child_node_id = requiredValue<int>(value, "child_node_id");
      edge.net_id = requiredValue<int>(value, "net_id");
      edge.partition_id = requiredValue<int>(value, "partition_id");
      edge.kind = requiredValue<int>(value, "kind");
      tree.edges.push_back(std::move(edge));
    }
    if (!validateCurrentDriverTree(tree, reason)) {
      return false;
    }
    tree.tree_digest = currentDriverTreeDigest(tree);
    if (values.contains("tree_digest")
        && py::cast<std::string>(values["tree_digest"]) != tree.tree_digest) {
      reason = "current_tree_digest_mismatch";
      return false;
    }
    return true;
  } catch (const std::exception& error) {
    reason = error.what();
    return false;
  }
}

std::vector<DriverTreeMutation> enumerateCurrentDriverTreeMutations(
    const CurrentDriverTree& tree,
    const DesignView& design,
    const LibertyView& liberty,
    std::string& reason)
{
  if (!validateCurrentDriverTree(tree, reason)) {
    return {};
  }
  std::vector<DriverTreeMutation> mutations;
  for (const auto& node : tree.nodes) {
    if (node.kind != static_cast<int>(DriverTreeNodeKind::buffer)) {
      continue;
    }
    std::vector<int> parents;
    std::vector<int> children;
    for (const auto& edge : tree.edges) {
      if (edge.child_node_id == node.id) {
        parents.push_back(edge.parent_node_id);
      }
      if (edge.parent_node_id == node.id) {
        children.push_back(edge.child_node_id);
      }
    }
    if (parents.size() != 1 || children.size() != 1) {
      reason = "existing_buffer_requires_one_parent_and_one_child";
      return {};
    }
    DriverTreeMutation keep;
    keep.kind = static_cast<int>(DriverTreeMutationKind::keep);
    keep.node_id = node.id;
    keep.source_instance_id = node.source_instance_id;
    keep.parent_node_id = parents.front();
    keep.child_node_id = children.front();
    mutations.push_back(keep);

    DriverTreeMutation remove = keep;
    remove.kind = static_cast<int>(DriverTreeMutationKind::remove);
    mutations.push_back(remove);

    if (node.source_instance_id < 0
        || node.source_instance_id >= static_cast<int>(design.inst_main_id.size())) {
      reason = "existing_buffer_instance_identity_missing";
      return {};
    }
    const int main_id = design.inst_main_id[node.source_instance_id];
    for (int cell = 0; cell < static_cast<int>(liberty.cells.size()); ++cell) {
      if (liberty.cells[cell].main_id != main_id
          || !liberty.cells[cell].is_buffer) {
        continue;
      }
      DriverTreeMutation rewrite = keep;
      rewrite.kind = static_cast<int>(DriverTreeMutationKind::rewrite);
      rewrite.target_cell_index = cell;
      rewrite.target_main_id = liberty.cells[cell].main_id;
      mutations.push_back(rewrite);
    }
  }
  return mutations;
}

bool applyCurrentDriverTreeMutation(const CurrentDriverTree& source,
                                    const DriverTreeMutation& mutation,
                                    CurrentDriverTree& target,
                                    std::string& reason)
{
  if (!validateCurrentDriverTree(source, reason)) {
    return false;
  }
  if (mutation.node_id < 0
      || mutation.node_id >= static_cast<int>(source.nodes.size())
      || source.nodes[mutation.node_id].kind
             != static_cast<int>(DriverTreeNodeKind::buffer)) {
    reason = "mutation_target_is_not_existing_buffer";
    return false;
  }
  if (mutation.kind == static_cast<int>(DriverTreeMutationKind::keep)) {
    target = source;
  } else if (mutation.kind == static_cast<int>(DriverTreeMutationKind::rewrite)) {
    if (mutation.target_cell_index < 0) {
      reason = "rewrite_target_cell_is_missing";
      return false;
    }
    if (mutation.target_main_id >= 0
        && mutation.target_main_id != source.nodes[mutation.node_id].master_id) {
      reason = "rewrite_target_main_id_mismatch";
      return false;
    }
    target = source;
    target.nodes[mutation.node_id].cell_id = mutation.target_cell_index;
  } else if (mutation.kind == static_cast<int>(DriverTreeMutationKind::remove)) {
    std::vector<int> parents;
    std::vector<int> children;
    for (const auto& edge : source.edges) {
      if (edge.child_node_id == mutation.node_id) {
        parents.push_back(edge.parent_node_id);
      }
      if (edge.parent_node_id == mutation.node_id) {
        children.push_back(edge.child_node_id);
      }
    }
    if (parents.size() != 1 || children.size() != 1
        || parents.front() != mutation.parent_node_id
        || children.front() != mutation.child_node_id) {
      reason = "remove_mutation_parent_child_mismatch";
      return false;
    }
    target = CurrentDriverTree{};
    target.root_net_id = source.root_net_id;
    target.driver_pin_id = source.driver_pin_id;
    target.topology_epoch = source.topology_epoch + 1;
    target.net_ids = source.net_ids;
    std::vector<int> remap(source.nodes.size(), -1);
    for (const auto& node : source.nodes) {
      if (node.id == mutation.node_id) {
        continue;
      }
      DriverTreeNode copy = node;
      copy.child_node_ids.clear();
      copy.parent_node_id = -1;
      copy.id = static_cast<int>(target.nodes.size());
      remap[node.id] = copy.id;
      target.nodes.push_back(std::move(copy));
    }
    for (const auto& edge : source.edges) {
      if (edge.parent_node_id == mutation.node_id
          || edge.child_node_id == mutation.node_id) {
        continue;
      }
      DriverTreeEdge copy = edge;
      copy.id = static_cast<int>(target.edges.size());
      copy.parent_node_id = remap[edge.parent_node_id];
      copy.child_node_id = remap[edge.child_node_id];
      target.edges.push_back(copy);
    }
    DriverTreeEdge bypass;
    bypass.id = static_cast<int>(target.edges.size());
    bypass.parent_node_id = remap[mutation.parent_node_id];
    bypass.child_node_id = remap[mutation.child_node_id];
    bypass.net_id = source.nodes[mutation.parent_node_id].net_id;
    bypass.partition_id = source.nodes[mutation.parent_node_id].partition_id;
    bypass.kind = static_cast<int>(DriverTreeEdgeKind::wire);
    target.edges.push_back(bypass);
    target.root_node_id = remap[source.root_node_id];
    for (auto& node : target.nodes) {
      node.child_node_ids.clear();
      node.parent_node_id = -1;
    }
    for (const auto& edge : target.edges) {
      target.nodes[edge.parent_node_id].child_node_ids.push_back(
          edge.child_node_id);
      target.nodes[edge.child_node_id].parent_node_id = edge.parent_node_id;
    }
  } else {
    reason = "unsupported_current_tree_mutation_kind";
    return false;
  }
  if (!validateCurrentDriverTree(target, reason)) {
    return false;
  }
  target.tree_digest = currentDriverTreeDigest(target);
  return true;
}

py::dict driverTreeMutationToDict(const DriverTreeMutation& mutation)
{
  py::dict result;
  result["kind"] = mutation.kind;
  result["node_id"] = mutation.node_id;
  result["source_instance_id"] = mutation.source_instance_id;
  result["parent_node_id"] = mutation.parent_node_id;
  result["child_node_id"] = mutation.child_node_id;
  result["target_cell_index"] = mutation.target_cell_index;
  result["target_main_id"] = mutation.target_main_id;
  return result;
}

DriverTreeMutation driverTreeMutationFromDict(const py::dict& values)
{
  DriverTreeMutation mutation;
  mutation.kind = requiredValue<int>(values, "kind");
  mutation.node_id = requiredValue<int>(values, "node_id");
  mutation.source_instance_id = requiredValue<int>(values, "source_instance_id");
  mutation.parent_node_id = requiredValue<int>(values, "parent_node_id");
  mutation.child_node_id = requiredValue<int>(values, "child_node_id");
  mutation.target_cell_index = requiredValue<int>(values, "target_cell_index");
  mutation.target_main_id = values.contains("target_main_id")
                                ? py::cast<int>(values["target_main_id"])
                                : -1;
  return mutation;
}

CurrentDriverTreeTransaction::CurrentDriverTreeTransaction(
    const CurrentDriverTree& source)
    : source_(source), active_(source)
{
  std::string reason;
  if (!validateCurrentDriverTree(source_, reason)) {
    throw py::value_error(reason);
  }
  source_.tree_digest = currentDriverTreeDigest(source_);
  active_.tree_digest = source_.tree_digest;
}

py::list CurrentDriverTreeTransaction::areaRecoveryCandidates(
    const py::dict& design_values,
    const py::dict& liberty_values) const
{
  const auto design = rebuffer_python::designFromDict(design_values);
  const auto liberty = rebuffer_python::libertyFromDict(liberty_values);
  std::string reason;
  const auto candidates = enumerateTreeAreaRecovery(
      active_, design, liberty, reason);
  if (!reason.empty()) {
    throw py::value_error(reason);
  }
  py::list output;
  for (const auto& candidate : candidates) {
    py::dict value = driverTreeMutationToDict(candidate.mutation);
    value["area_saved"] = candidate.area_saved;
    value["buffer_count_delta"] = candidate.buffer_count_delta;
    value["recovery_kind"] = candidate.recovery_kind;
    output.append(std::move(value));
  }
  return output;
}

py::dict CurrentDriverTreeTransaction::applyCandidateTree(
    const std::shared_ptr<CurrentDriverTree>& candidate,
    const std::string& source_tree_digest)
{
  if (candidate == nullptr) {
    throw py::value_error("current tree candidate is null");
  }
  if (source_tree_digest != source_.tree_digest) {
    throw py::value_error("stale_source_tree_identity");
  }
  std::string reason;
  if (!validateCurrentDriverTree(*candidate, reason)) {
    throw py::value_error(reason);
  }
  if (candidate->root_net_id != source_.root_net_id
      || candidate->driver_pin_id != source_.driver_pin_id
      || candidate->net_ids != source_.net_ids) {
    throw py::value_error("candidate_tree_root_or_partition_mismatch");
  }
  active_ = *candidate;
  active_.tree_digest = currentDriverTreeDigest(active_);
  state_ = "APPLIED";
  py::dict result = currentDriverTreeToDict(active_);
  result["transaction_state"] = state_;
  result["source_tree_digest"] = source_.tree_digest;
  result["candidate_tree_digest"] = active_.tree_digest;
  return result;
}

py::dict CurrentDriverTreeTransaction::activeTree() const
{
  return currentDriverTreeToDict(active_);
}

py::dict CurrentDriverTreeTransaction::evaluateCandidate(const py::dict& values)
{
  if (state_ != "APPLIED") {
    throw py::value_error("current tree candidate must be APPLIED before evaluation");
  }
  const std::string source_digest = requiredValue<std::string>(values, "source_tree_digest");
  const std::string candidate_digest =
      requiredValue<std::string>(values, "candidate_tree_digest");
  if (source_digest != source_.tree_digest) {
    throw py::value_error("stale_source_tree_identity");
  }
  if (candidate_digest != active_.tree_digest) {
    throw py::value_error("stale_candidate_tree_identity");
  }
  const std::string status = requiredValue<std::string>(values, "status");
  if (status != "evaluated" && status != "rejected" && status != "failed") {
    throw py::value_error("invalid_candidate_evaluation_status");
  }
  const int timing_epoch = requiredValue<int>(values, "timing_epoch");
  if (timing_epoch < 0) {
    throw py::value_error("invalid_timing_epoch");
  }
  state_ = "EVALUATED";
  py::dict result = values;
  result["transaction_state"] = state_;
  result["source_tree_digest"] = source_.tree_digest;
  result["candidate_tree_digest"] = active_.tree_digest;
  result["tree_identity"] = treeIdentity(active_);
  return result;
}

py::dict CurrentDriverTreeTransaction::acceptCandidate()
{
  if (state_ != "EVALUATED") {
    throw py::value_error("current tree candidate must be EVALUATED before accept");
  }
  state_ = "ACCEPTED";
  py::dict result = currentDriverTreeToDict(active_);
  result["transaction_state"] = state_;
  result["source_tree_digest"] = source_.tree_digest;
  result["candidate_tree_digest"] = active_.tree_digest;
  return result;
}

py::dict CurrentDriverTreeTransaction::applyMutation(
    const py::dict& values)
{
  const auto mutation = driverTreeMutationFromDict(values);
  CurrentDriverTree candidate;
  std::string reason;
  if (!applyCurrentDriverTreeMutation(active_, mutation, candidate, reason)) {
    throw py::value_error(reason);
  }
  active_ = std::move(candidate);
  state_ = "APPLIED";
  py::dict result = currentDriverTreeToDict(active_);
  result["transaction_state"] = state_;
  result["source_tree_digest"] = source_.tree_digest;
  result["candidate_tree_digest"] = active_.tree_digest;
  return result;
}

std::shared_ptr<CurrentDriverTreeCheckpoint>
CurrentDriverTreeTransaction::checkpoint() const
{
  auto result = std::make_shared<CurrentDriverTreeCheckpoint>();
  result->tree = active_;
  return result;
}

py::dict CurrentDriverTreeTransaction::restore(
    const std::shared_ptr<CurrentDriverTreeCheckpoint>& checkpoint)
{
  if (checkpoint == nullptr) {
    throw py::value_error("current driver tree checkpoint is null");
  }
  std::string reason;
  if (!validateCurrentDriverTree(checkpoint->tree, reason)) {
    throw py::value_error(reason);
  }
  active_ = checkpoint->tree;
  active_.tree_digest = currentDriverTreeDigest(active_);
  state_ = "RESTORED";
  py::dict result = currentDriverTreeToDict(active_);
  result["transaction_state"] = state_;
  result["source_tree_digest"] = source_.tree_digest;
  result["candidate_tree_digest"] = active_.tree_digest;
  return result;
}

py::dict CurrentDriverTreeTransaction::summary() const
{
  py::dict result;
  result["transaction_state"] = state_;
  result["source_tree_digest"] = source_.tree_digest;
  result["active_tree_digest"] = active_.tree_digest;
  result["source_node_count"] = static_cast<int>(source_.nodes.size());
  result["active_node_count"] = static_cast<int>(active_.nodes.size());
  result["source_edge_count"] = static_cast<int>(source_.edges.size());
  result["active_edge_count"] = static_cast<int>(active_.edges.size());
  return result;
}

std::shared_ptr<CurrentDriverTreeTransaction>
createCurrentDriverTreeTransaction(
    const std::shared_ptr<CurrentDriverTree>& source)
{
  if (source == nullptr) {
    throw py::value_error("current driver tree is null");
  }
  return std::make_shared<CurrentDriverTreeTransaction>(*source);
}

}  // namespace rebuffer
