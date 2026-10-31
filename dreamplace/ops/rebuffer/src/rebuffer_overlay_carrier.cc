// SPDX-License-Identifier: BSD-3-Clause
// Export mutable net partitions as physical pins/nets and existing RC vertices.
#include "rebuffer_overlay_carrier.hh"
#include "rebuffer_carrier.hh"

#include <algorithm>
#include <functional>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <unordered_set>

namespace py = pybind11;
namespace rebuffer {
namespace {

struct BufferRow
{
  const IncrementalTopologyState* state;
  const IncrementalBuffer* buffer;
  int cell;
  int input_pin;
  int output_pin;
  int output_net;
  std::string name;
  std::string input_port;
  std::string output_port;
};

void flatRows(py::dict& out, const char* rows_name, const char* values_name,
              const char* offsets_name, const std::vector<std::vector<int>>& rows)
{
  std::vector<int> values, offsets{0};
  for (const auto& row : rows) {
    values.insert(values.end(), row.begin(), row.end());
    offsets.push_back(values.size());
  }
  out[rows_name] = rows;
  out[values_name] = values;
  out[offsets_name] = offsets;
}

}  // namespace

py::dict exportOverlayCarrier(
    const py::object& base, const SteinerView& steiner,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const py::list& actions, const double r_unit, const double c_unit)
{
  if (!std::isfinite(r_unit) || r_unit <= 0.0 || !std::isfinite(c_unit)
      || c_unit < 0.0) {
    throw py::value_error("invalid_overlay_wire_rc_units");
  }
  const auto base_pins = carrierArray<std::string>(base, "pin_names");
  auto nodes = carrierArray<std::string>(base, "node_names");
  auto masters = carrierArray<std::string>(base, "node_master_names");
  auto nets = carrierArray<std::string>(base, "net_names");
  auto pin_owner = carrierArray<int>(base, "pin2node_map");
  auto pin_names = base_pins;
  auto pin_x = carrierArray<double>(base, "pin_offset_x");
  auto pin_y = carrierArray<double>(base, "pin_offset_y");
  auto pin_orient = carrierArray<std::string>(base, "pin_direct");
  const auto cell_names = carrierArray<std::string>(base, "flat_libcell_names");
  const auto cell_arc_start = carrierArray<int>(base, "cell_id_2_arc_id_start");
  const auto lib_arcs = base.attr("flat_libarc_info").cast<py::sequence>();
  const int pin_count = base_pins.size(), net_count = nets.size();
  if (steiner.steiner_node_base != pin_count || steiner.node_x.size() != steiner.node_y.size()) {
    throw py::value_error("overlay requires the base Steiner vertex domain");
  }
  std::map<int, const IncrementalTopologyState*> by_net;
  std::map<int, std::pair<const IncrementalTopologyState*, const IncrementalBuffer*>> by_action;
  for (const auto& state : states) {
    std::string reason;
    if (!state || !validateIncrementalTopology(*state, reason)) throw py::value_error(reason);
    if (state->net_id < 0 || state->net_id >= net_count || !by_net.emplace(state->net_id, state.get()).second) {
      throw py::value_error("duplicate or invalid overlay net identity");
    }
    for (const auto& buffer : state->buffers) {
      if (!by_action.emplace(buffer.action_id, std::make_pair(state.get(), &buffer)).second) {
        throw py::value_error("duplicate overlay action identity");
      }
    }
  }
  std::map<int, py::dict> action_rows;
  for (const auto& item : actions) {
    const auto action = item.cast<py::dict>();
    const int id = py::cast<int>(action["session_buffer_id"]);
    if (!action_rows.emplace(id, action).second || !by_action.count(id)) {
      throw py::value_error("overlay action has no current native buffer");
    }
  }
  if (action_rows.size() != by_action.size()) throw py::value_error("native overlay buffer has no action metadata");

  std::vector<BufferRow> buffers;
  std::set<std::string> node_names(nodes.begin(), nodes.end()), net_names(nets.begin(), nets.end());
  for (const auto& [id, entry] : by_action) {
    const auto& action = action_rows.at(id);
    const auto master = py::cast<std::string>(action["buffer_master_name"]);
    const auto found = std::find(cell_names.begin(), cell_names.end(), master);
    if (found == cell_names.end()) throw py::value_error("overlay master is absent from Liberty");
    const int cell = found - cell_names.begin();
    const int begin = cell_arc_start.at(cell), end = cell_arc_start.at(cell + 1);
    if (begin == end) throw py::value_error("overlay buffer has no Liberty arc");
    const auto first = lib_arcs[begin].cast<py::sequence>();
    const auto source = py::cast<std::string>(first[0]), sink = py::cast<std::string>(first[1]);
    for (int lut = begin; lut < end; ++lut) {
      const auto info = lib_arcs[lut].cast<py::sequence>();
      if (py::cast<std::string>(info[0]) != source || py::cast<std::string>(info[1]) != sink
          || py::cast<int>(info[4]) != 1) throw py::value_error("overlay requires a noninverting buffer master");
    }
    const std::string name = "__dreamplace_rebuffer_" + std::to_string(id);
    const std::string net_name = name + "_out";
    if (!node_names.insert(name).second || !net_names.insert(net_name).second) {
      throw py::value_error("overlay identity collides with the base design");
    }
    const int index = buffers.size();
    buffers.push_back({entry.first, entry.second, cell, pin_count + 2 * index,
                       pin_count + 2 * index + 1, net_count + index, name, source, sink});
    nets.push_back(net_name);
  }
  const int extra_pins = buffers.size() * 2;
  const int physical_pins = pin_count + extra_pins;
  const int vertex_count = steiner.node_x.size() + extra_pins;
  const int movable = nodes.size() - py::cast<int>(base.attr("num_terminals"))
      - py::cast<int>(base.attr("num_terminal_NIs"));
  if (movable < 0) throw py::value_error("invalid base placement node counts");
  for (int& node : pin_owner) if (node >= movable) node += buffers.size();
  auto node_x = carrierArray<double>(base, "node_x");
  auto node_y = carrierArray<double>(base, "node_y");
  auto width = carrierArray<double>(base, "node_size_x");
  auto height = carrierArray<double>(base, "node_size_y");
  const auto lib_width = carrierArray<double>(base, "flat_libcell_width");
  const auto lib_height = carrierArray<double>(base, "flat_libcell_height");
  std::vector<int> original_net(nets.size());
  std::iota(original_net.begin(), original_net.begin() + net_count, 0);
  std::map<std::pair<int, int>, int> virtual_pins;
  for (int i = 0; i < static_cast<int>(buffers.size()); ++i) {
    const auto& row = buffers[i];
    const auto point = row.state->coordinates.at(row.buffer->input_node);
    const int node = movable + i;
    nodes.insert(nodes.begin() + node, row.name);
    masters.insert(masters.begin() + node, cell_names[row.cell]);
    node_x.insert(node_x.begin() + node, point[0]);
    node_y.insert(node_y.begin() + node, point[1]);
    width.insert(width.begin() + node, lib_width.at(row.cell));
    height.insert(height.begin() + node, lib_height.at(row.cell));
    pin_names.push_back(row.name + ":" + row.input_port);
    pin_names.push_back(row.name + ":" + row.output_port);
    pin_owner.insert(pin_owner.end(), {node, node});
    pin_x.insert(pin_x.end(), {0.0, 0.0});
    pin_y.insert(pin_y.end(), {0.0, 0.0});
    pin_orient.insert(pin_orient.end(), {"R0", "R0"});
    original_net[row.output_net] = row.state->net_id;
    virtual_pins[{row.state->net_id, row.buffer->input_node}] = row.input_pin;
    virtual_pins[{row.state->net_id, row.buffer->output_node}] = row.output_pin;
  }
  auto baseVertex = [&](int node) { return node < pin_count ? node : node + extra_pins; };
  auto vertex = [&](int net, int node) {
    const auto found = virtual_pins.find({net, node});
    return found == virtual_pins.end() ? baseVertex(node) : found->second;
  };
  std::vector<double> x(vertex_count), y(vertex_count);
  for (int node = 0; node < static_cast<int>(steiner.node_x.size()); ++node) {
    x[baseVertex(node)] = steiner.node_x[node];
    y[baseVertex(node)] = steiner.node_y[node];
  }
  for (const auto& row : buffers) {
    const auto point = row.state->coordinates.at(row.buffer->input_node);
    x[row.input_pin] = x[row.output_pin] = point[0];
    y[row.input_pin] = y[row.output_pin] = point[1];
  }
  std::vector<std::vector<int>> topo(nets.size()), net_pins(nets.size()), children(vertex_count);
  std::vector<int> parent(vertex_count, -1), state_node(vertex_count, -1), state_net(vertex_count, -1);
  std::map<std::pair<int, int>, std::pair<int, int>> state_provenance;
  auto addNode = [&](int net, int mapped, int original, int original_net_id) {
    if (mapped < 0 || mapped >= vertex_count) throw py::value_error("overlay RC vertex is out of range");
    topo[net].push_back(mapped);
    state_node[mapped] = original;
    state_net[mapped] = original_net_id;
    state_provenance[{net, mapped}] = {original_net_id, original};
    if (mapped < physical_pins) net_pins[net].push_back(mapped);
  };
  for (int net = 0; net < net_count; ++net) {
    if (by_net.count(net)) continue;
    const int begin = steiner.net_flat_topo_sort_start.at(net), end = steiner.net_flat_topo_sort_start.at(net + 1);
    for (int i = begin; i < end; ++i) {
      const int node = steiner.net_flat_topo_sort.at(i), mapped = baseVertex(node);
      addNode(net, mapped, node, net);
      if (i != begin) {
        parent[mapped] = baseVertex(steiner.pin_fa.at(node));
        children.at(parent[mapped]).push_back(mapped);
      }
    }
  }
  for (const auto& [net, state] : by_net) {
    std::map<int, int> roots{{state->driver_pin_id, net}};
    std::set<std::pair<int, int>> cell_edges;
    for (const auto& row : buffers) {
      if (row.state != state) continue;
      roots[row.buffer->output_node] = row.output_net;
      cell_edges.insert({row.buffer->input_node, row.buffer->output_node});
    }
    std::map<int, std::vector<int>> wire_children;
    for (const auto& edge : state->edges) {
      if (!cell_edges.count({edge.parent, edge.child})) wire_children[edge.parent].push_back(edge.child);
    }
    std::unordered_set<int> visited;
    std::function<void(int, int)> visit = [&](int node, int partition) {
      if (!visited.insert(node).second) throw py::value_error("overlay vertex belongs to multiple partitions");
      const int mapped = vertex(net, node);
      addNode(partition, mapped, node, net);
      for (const int child : wire_children[node]) {
        const int mapped_child = vertex(net, child);
        parent.at(mapped_child) = mapped;
        children[mapped].push_back(mapped_child);
        visit(child, partition);
      }
    };
    for (const auto& [root, partition] : roots) visit(root, partition);
    if (visited.size() != state->coordinates.size()) throw py::value_error("overlay contains an unexported RC vertex");
  }

  // The native planner consumes the same vertex-domain RC convention as
  // RCTiming: edge resistance is stored on the child vertex and distributed
  // wire capacitance is split equally between the edge endpoints.  The
  // carrier used to export only topology, which made the fresh Steiner path
  // fail at steinerFromDict() and encouraged callers to use a stale base RC
  // view.  Reconstruct the arrays from the current graph after all buffer
  // partitions have been materialized.
  std::vector<double> edge_resistance(vertex_count, 0.0);
  std::vector<double> node_capacitance(vertex_count, 0.0);
  std::map<std::tuple<int, int, int>, std::pair<double, double>> state_rc;
  for (const auto& [net, state] : by_net) {
    for (const auto& edge : state->edges) {
      state_rc[{net, edge.parent, edge.child}]
          = {edge.resistance, edge.capacitance};
    }
  }

  for (int parent = 0; parent < vertex_count; ++parent) {
    for (const int child : children[parent]) {
      if (parent < 0 || child < 0 || child >= vertex_count
          || state_node[parent] < 0 || state_node[child] < 0
          || state_net[parent] < 0 || state_net[child] < 0
          || state_net[parent] != state_net[child]) {
        throw py::value_error("overlay RC edge identity is invalid");
      }
      const int net = state_net[child];
      const int parent_node = state_node[parent];
      const int child_node = state_node[child];
      double resistance = 0.0;
      double capacitance = 0.0;
      const auto state_edge = state_rc.find({net, parent_node, child_node});
      if (state_edge != state_rc.end()) {
        resistance = state_edge->second.first;
        capacitance = state_edge->second.second;
      } else {
        // Unmodified partitions retain the base Steiner identity.  Its
        // incoming resistance is indexed by the child vertex; derive the
        // corresponding half-edge capacitance using the active placement-RC
        // units, matching RCTiming's geometry contract.
        if (net < 0 || net >= static_cast<int>(steiner.net_flat_topo_sort_start.size()) - 1
            || child_node < 0
            || child_node >= static_cast<int>(steiner.edge_resistance.size())) {
          throw py::value_error("overlay RC edge is absent from current state");
        }
        resistance = steiner.edge_resistance[child_node];
        capacitance = resistance / r_unit * c_unit;
      }
      if (!std::isfinite(resistance) || resistance < 0.0
          || !std::isfinite(capacitance) || capacitance < 0.0) {
        throw py::value_error("overlay RC edge has invalid value");
      }
      edge_resistance[child] = resistance;
      node_capacitance[parent] += 0.5 * capacitance;
      node_capacitance[child] += 0.5 * capacitance;
    }
  }
  py::dict physical;
  for (const char* field : {"dbu", "num_terminals", "num_terminal_NIs", "clock_net_names",
                          "xl", "yl", "xh", "yh", "row_height", "site_width", "rows", "regions"}) {
    if (py::hasattr(base, field)) physical[field] = base.attr(field);
  }
  std::map<std::string, int> node_ids;
  for (int node = 0; node < static_cast<int>(nodes.size()); ++node) node_ids[nodes[node]] = node;
  physical["num_nodes"] = nodes.size();
  physical["node_names"] = nodes; physical["node_name2id_map"] = node_ids;
  physical["node_master_names"] = masters;
  physical["node_x"] = node_x; physical["node_y"] = node_y;
  physical["node_size_x"] = width; physical["node_size_y"] = height;
  physical["pin_names"] = pin_names; physical["pin2node_map"] = pin_owner;
  physical["pin_offset_x"] = pin_x; physical["pin_offset_y"] = pin_y;
  physical["pin_direct"] = pin_orient; physical["net_names"] = nets;
  flatRows(physical, "net2pin_map", "flat_net2pin_map", "flat_net2pin_start_map", net_pins);
  py::dict current;
  flatRows(current, "net_topologies", "net_flat_topo_sort", "net_flat_topo_sort_start", topo);
  flatRows(current, "children", "flat_pin_to", "flat_pin_to_start", children);
  std::vector<int> from;
  for (int node = 0; node < vertex_count; ++node) for (const int child : children[node]) from.push_back(node);
  current["pin_fa"] = parent; current["flat_pin_from"] = from;
  current["node_x"] = x; current["node_y"] = y;
  current["steiner_node_base"] = physical_pins;
  current["edge_resistance"] = edge_resistance;
  current["node_capacitance"] = node_capacitance;
  py::dict output;
  output["physical"] = physical; output["steiner"] = current;
  output["original_net_by_net"] = original_net;
  output["state_node_by_vertex"] = state_node;
  output["state_net_by_vertex"] = state_net;
  py::list provenance;
  for (const auto& [key, value] : state_provenance) {
    py::dict row;
    row["partition_id"] = key.first;
    row["vertex_id"] = key.second;
    row["state_net_id"] = value.first;
    row["state_node_id"] = value.second;
    provenance.append(std::move(row));
  }
  output["state_provenance"] = std::move(provenance);
  return output;
}

}  // namespace rebuffer
