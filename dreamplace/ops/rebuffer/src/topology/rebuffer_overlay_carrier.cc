// SPDX-License-Identifier: BSD-3-Clause
// Export mutable net partitions as physical pins/nets and existing RC vertices.
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

OverlayCarrierProjection buildOverlayCarrierProjection(
    const py::object& base, const SteinerView& steiner,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const py::list& actions, const double r_unit, const double c_unit)
{
  return buildOverlayCarrierProjection(overlayCarrierInputs(base), steiner, states,
                                         actions, r_unit, c_unit);
}

OverlayCarrierProjection buildOverlayCarrierProjection(
    const OverlayCarrierInputs& base, const SteinerView& steiner,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const py::list& actions, const double r_unit, const double c_unit)
{
  if (!std::isfinite(r_unit) || r_unit <= 0.0 || !std::isfinite(c_unit)
      || c_unit < 0.0) {
    throw py::value_error("invalid_overlay_wire_rc_units");
  }
  const auto& base_pins = base.identity.pins;
  auto nodes = base.nodes;
  auto masters = base.masters;
  auto nets = base.identity.nets;
  auto pin_owner = base.identity.pin2node;
  auto pin_names = base_pins;
  auto pin_x = base.pin_x;
  auto pin_y = base.pin_y;
  auto pin_orient = base.pin_direct;
  const auto& cell_names = base.identity.cells;
  const auto& cell_arc_start = base.cell_arc_start;
  const auto& lib_arcs = base.library_arcs;
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
    if (action.contains("action_kind")
        && py::cast<std::string>(action["action_kind"]) == "resize_cell") {
      continue;
    }
    const int id = py::cast<int>(action["session_buffer_id"]);
    if (!action_rows.emplace(id, action).second || !by_action.count(id)) {
      throw py::value_error("overlay action has no current native buffer");
    }
  }
  if (action_rows.size() != by_action.size()) throw py::value_error("native overlay buffer has no action metadata");

  std::vector<BufferRow> buffers;
  for (const auto& [id, entry] : by_action) {
    const auto& action = action_rows.at(id);
    const auto master = py::cast<std::string>(action["buffer_master_name"]);
    const auto found = std::find(cell_names.begin(), cell_names.end(), master);
    if (found == cell_names.end()) throw py::value_error("overlay master is absent from Liberty");
    const int cell = found - cell_names.begin();
    const int begin = cell_arc_start.at(cell), end = cell_arc_start.at(cell + 1);
    if (begin == end) throw py::value_error("overlay buffer has no Liberty arc");
    const auto& first = lib_arcs.at(begin);
    const auto& source = first.source;
    const auto& sink = first.sink;
    for (int lut = begin; lut < end; ++lut) {
      const auto& info = lib_arcs.at(lut);
      if (info.source != source || info.sink != sink || info.sense != 1) {
        throw py::value_error("overlay requires a noninverting buffer master");
      }
    }
    const std::string name = "__dreamplace_rebuffer_" + std::to_string(id);
    const std::string net_name = name + "_out";
    // Unique action ids give distinct generated names; base membership is
    // immutable and indexed once when importing the campaign's physical DB.
    if (base.node_name_set.count(name) || base.net_name_set.count(net_name)) {
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
  const int movable = nodes.size() - base.num_terminals - base.num_terminal_nis;
  if (movable < 0) throw py::value_error("invalid base placement node counts");
  for (int& node : pin_owner) if (node >= movable) node += buffers.size();
  auto node_x = base.node_x;
  auto node_y = base.node_y;
  auto width = base.width;
  auto height = base.height;
  const auto& lib_width = base.lib_width;
  const auto& lib_height = base.lib_height;
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
  PartitionProvenance state_provenance;
  state_provenance.reserve(vertex_count);
  auto addNode = [&](int net, int mapped, int original, int original_net_id) {
    if (mapped < 0 || mapped >= vertex_count) throw py::value_error("overlay RC vertex is out of range");
    topo[net].push_back(mapped);
    state_node[mapped] = original;
    state_net[mapped] = original_net_id;
    state_provenance.push_back({{net, mapped}, {original_net_id, original}});
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
  OverlayCarrierProjection data;
  auto& design = data.design;
  design.dbu = base.dbu;
  design.num_nodes = nodes.size();
  design.num_base_nodes = base.nodes.size();
  design.num_pins = physical_pins;
  design.num_nets = nets.size();
  design.num_terminals = base.num_terminals;
  design.num_terminal_nis = base.num_terminal_nis;
  design.pin_names = std::move(pin_names);
  design.pin2node_map = std::move(pin_owner);
  design.pin_offset_x = std::move(pin_x);
  design.pin_offset_y = std::move(pin_y);
  design.node_x = std::move(node_x);
  design.node_y = std::move(node_y);
  design.pin2net_map.assign(physical_pins, -1);
  design.net2driver_pin_map.assign(nets.size(), -1);
  design.flat_net2pin_start_map.push_back(0);
  for (std::size_t net = 0; net < nets.size(); ++net) {
    const bool has_driver =
        net >= static_cast<std::size_t>(net_count) || base.identity.drivers.at(net) >= 0;
    if (has_driver && !topo[net].empty()) {
      design.net2driver_pin_map[net] = topo[net].front();
    } else if (net < static_cast<std::size_t>(net_count)) {
      design.net2driver_pin_map[net] = base.identity.drivers.at(net);
    }
    for (const int pin : net_pins[net]) {
      design.pin2net_map.at(pin) = net;
      design.flat_net2pin_map.push_back(pin);
    }
    design.flat_net2pin_start_map.push_back(design.flat_net2pin_map.size());
  }
  for (const auto& row : buffers) {
    data.buffers.push_back({row.buffer->action_id, row.cell,
        movable + static_cast<int>(data.buffers.size()), row.input_pin, row.output_pin, row.output_net});
  }
  data.node_names = std::move(nodes);
  data.node_master_names = std::move(masters);
  data.net_names = std::move(nets);
  data.pin_direct = std::move(pin_orient);
  data.node_size_x = std::move(width);
  data.node_size_y = std::move(height);
  data.net_pins = std::move(net_pins);
  data.topologies = std::move(topo);
  data.children = std::move(children);
  data.actions.original_net_by_net = std::move(original_net);
  data.actions.state_node_by_vertex = std::move(state_node);
  data.actions.state_net_by_vertex = std::move(state_net);
  normalizePartitionProvenance(state_provenance);
  data.actions.state_provenance = std::move(state_provenance);
  auto& current = data.steiner;
  current.steiner_node_base = physical_pins;
  current.net_ids.resize(data.net_names.size());
  std::iota(current.net_ids.begin(), current.net_ids.end(), 0);
  current.net_flat_topo_sort_start.push_back(0);
  for (const auto& row : data.topologies) {
    current.net_flat_topo_sort.insert(current.net_flat_topo_sort.end(), row.begin(), row.end());
    current.net_flat_topo_sort_start.push_back(current.net_flat_topo_sort.size());
  }
  current.pin_fa = std::move(parent);
  current.node_x = std::move(x);
  current.node_y = std::move(y);
  current.edge_resistance = std::move(edge_resistance);
  current.node_capacitance = std::move(node_capacitance);
  return data;
}

py::dict exportOverlayCarrier(
    const py::object& base, const SteinerView& steiner,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const py::list& actions, const double r_unit, const double c_unit)
{
  const auto data = buildOverlayCarrierProjection(base, steiner, states, actions, r_unit, c_unit);
  const auto& nodes = data.node_names;
  const auto& masters = data.node_master_names;
  const auto& nets = data.net_names;
  const auto& design = data.design;
  const auto& node_x = design.node_x;
  const auto& node_y = design.node_y;
  const auto& width = data.node_size_x;
  const auto& height = data.node_size_y;
  const auto& pin_names = design.pin_names;
  const auto& pin_owner = design.pin2node_map;
  const auto& pin_x = design.pin_offset_x;
  const auto& pin_y = design.pin_offset_y;
  const auto& pin_orient = data.pin_direct;
  const auto& net_pins = data.net_pins;
  const auto& topo = data.topologies;
  const auto& children = data.children;
  const auto& parent = data.steiner.pin_fa;
  const auto& x = data.steiner.node_x;
  const auto& y = data.steiner.node_y;
  const auto& edge_resistance = data.steiner.edge_resistance;
  const auto& node_capacitance = data.steiner.node_capacitance;
  const auto& original_net = data.actions.original_net_by_net;
  const auto& state_node = data.actions.state_node_by_vertex;
  const auto& state_net = data.actions.state_net_by_vertex;
  const auto& state_provenance = data.actions.state_provenance;
  const int vertex_count = x.size(), physical_pins = design.num_pins;
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

// --- rebuffer_overlay_inputs.cc ---

namespace py = pybind11;
namespace rebuffer {

OverlayActionInputs overlayActionInputs(const py::object& base)
{
  OverlayActionInputs out{
      carrierArray<std::string>(base, "flat_libcell_names"),
      carrierArray<std::string>(base, "pin_names"),
      carrierArray<std::string>(base, "net_names"),
      carrierArray<int>(base, "pin2node_map"),
      carrierArray<int>(base, "pin2net_map"),
      carrierArray<int>(base, "net2driver_pin_map"),
      {}};
  int max_node = -1;
  for (int node : out.pin2node) {
    if (node > max_node) max_node = node;
  }
  std::vector<int> node_unique_input_pin(static_cast<std::size_t>(max_node + 1), -1);
  for (int pin = 0; pin < static_cast<int>(out.pin2node.size()); ++pin) {
    const int node = out.pin2node[pin];
    const int net = pin < static_cast<int>(out.pin2net.size()) ? out.pin2net[pin] : -1;
    if (node < 0 || net < 0 || net >= static_cast<int>(out.drivers.size())
        || out.drivers[net] == pin) {
      continue;
    }
    if (node_unique_input_pin[node] == -1) {
      node_unique_input_pin[node] = pin;
    } else {
      node_unique_input_pin[node] = -2;
    }
  }
  out.buffer_input_pin_by_net.assign(out.drivers.size(), -1);
  for (int net = 0; net < static_cast<int>(out.drivers.size()); ++net) {
    const int driver = out.drivers[net];
    if (driver < 0 || driver >= static_cast<int>(out.pin2node.size())) continue;
    const int node = out.pin2node[driver];
    if (node >= 0 && node < static_cast<int>(node_unique_input_pin.size())
        && node_unique_input_pin[node] >= 0) {
      out.buffer_input_pin_by_net[net] = node_unique_input_pin[node];
    }
  }
  return out;
}

OverlayCarrierInputs overlayCarrierInputs(const py::object& base)
{
  OverlayCarrierInputs result;
  result.identity = overlayActionInputs(base);
  result.dbu = py::cast<int>(base.attr("dbu"));
  result.num_terminals = py::cast<int>(base.attr("num_terminals"));
  result.num_terminal_nis = py::cast<int>(base.attr("num_terminal_NIs"));
  result.nodes = carrierArray<std::string>(base, "node_names");
  result.node_name_set.insert(result.nodes.begin(), result.nodes.end());
  result.net_name_set.insert(result.identity.nets.begin(), result.identity.nets.end());
  for (const auto& name : result.nodes) {
    if (name.rfind("__dreamplace_rebuffer_", 0) == 0) {
      result.has_rebuffer_prefix = true;
      break;
    }
  }
  if (!result.has_rebuffer_prefix) {
    for (const auto& name : result.identity.nets) {
      if (name.rfind("__dreamplace_rebuffer_", 0) == 0) {
        result.has_rebuffer_prefix = true;
        break;
      }
    }
  }
  result.masters = carrierArray<std::string>(base, "node_master_names");
  result.pin_direct = carrierArray<std::string>(base, "pin_direct");
  result.cell_arc_start = carrierArray<int>(base, "cell_id_2_arc_id_start");
  for (const auto& item : base.attr("flat_libarc_info").cast<py::sequence>()) {
    const auto row = item.cast<py::sequence>();
    result.library_arcs.push_back({row[0].cast<std::string>(), row[1].cast<std::string>(),
                                    row[4].cast<int>()});
  }
  result.pin_x = carrierArray<double>(base, "pin_offset_x");
  result.pin_y = carrierArray<double>(base, "pin_offset_y");
  result.node_x = carrierArray<double>(base, "node_x");
  result.node_y = carrierArray<double>(base, "node_y");
  result.width = carrierArray<double>(base, "node_size_x");
  result.height = carrierArray<double>(base, "node_size_y");
  result.lib_width = carrierArray<double>(base, "flat_libcell_width");
  result.lib_height = carrierArray<double>(base, "flat_libcell_height");
  return result;
}

py::list projectOverlayActionsFromProjection(
    const py::object& base, const OverlayActionProjection& projection,
    const std::vector<std::shared_ptr<IncrementalTopologyState>>& states,
    const py::list& actions)
{
  return projectOverlayActionsFromProjection(overlayActionInputs(base), projection, states, actions);
}

}  // namespace rebuffer
