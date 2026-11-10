// SPDX-License-Identifier: BSD-3-Clause
#include "rebuffer_carrier.hh"

#include <algorithm>
#include <unordered_set>

namespace py = pybind11;
namespace rebuffer {
namespace {

std::unordered_map<std::string, int> nameIndex(const std::vector<std::string>& names)
{
  std::unordered_map<std::string, int> result;
  for (int i = 0; i < static_cast<int>(names.size()); ++i) {
    if (!result.emplace(names[i], i).second) throw py::value_error("duplicate physical identity");
  }
  return result;
}

void exportRows(py::dict& output, const char* rows_name, const char* flat_name,
                const char* start_name, const std::vector<std::vector<int>>& rows)
{
  std::vector<int> flat;
  std::vector<int> start{0};
  for (const auto& row : rows) {
    flat.insert(flat.end(), row.begin(), row.end());
    start.push_back(flat.size());
  }
  output[rows_name] = rows;
  output[flat_name] = flat;
  output[start_name] = start;
}

}  // namespace

py::dict rebuildTimingCarrier(const py::object& base, const py::object& physical)
{
  if (!py::hasattr(base, "flat_lib_pin_names") || !py::hasattr(physical, "node_master_names")) {
    throw py::value_error("timing carrier requires Liberty pin names and physical node masters");
  }
  TimingCarrierView state;
  state.node_names = carrierArray<std::string>(physical, "node_names");
  state.pin_names = carrierArray<std::string>(physical, "pin_names");
  state.net_names = carrierArray<std::string>(physical, "net_names");
  state.pin2node = carrierArray<int>(physical, "pin2node_map");
  const auto masters = carrierArray<std::string>(physical, "node_master_names");
  const auto node_ids = nameIndex(state.node_names);
  state.pin_ids = nameIndex(state.pin_names);
  auto net_ids = nameIndex(state.net_names);
  const auto base_nodes = carrierArray<std::string>(base, "node_names");
  const auto base_pins = carrierArray<std::string>(base, "pin_names");
  const auto base_node_ids = nameIndex(base_nodes);
  const auto clock_names = carrierArray<std::string>(base, "clock_net_names");
  const std::unordered_set<std::string> clocks(clock_names.begin(), clock_names.end());
  auto x = carrierArray<double>(physical, "pin_offset_x");
  auto y = carrierArray<double>(physical, "pin_offset_y");
  auto orientation = carrierArray<std::string>(physical, "pin_direct");
  if (masters.size() != state.node_names.size() || state.pin2node.size() != state.pin_names.size()
      || x.size() != state.pin_names.size() || y.size() != state.pin_names.size()
      || orientation.size() != state.pin_names.size()) {
    throw py::value_error("physical carrier arrays do not align");
  }
  state.old_node_ids.assign(base_nodes.size(), -1);
  for (int node = 0; node < static_cast<int>(base_nodes.size()); ++node) {
    const auto found = node_ids.find(base_nodes[node]);
    if (found != node_ids.end()) state.old_node_ids[node] = found->second;
  }
  // Physical exports must include clocks. Missing timing pins are rejected
  // by the graph compiler rather than resurrected from stale connectivity.
  state.old_pin_ids.assign(base_pins.size(), -1);
  for (int pin = 0; pin < static_cast<int>(base_pins.size()); ++pin) {
    const auto found = state.pin_ids.find(base_pins[pin]);
    if (found != state.pin_ids.end()) state.old_pin_ids[pin] = found->second;
  }
  const auto physical_flat = carrierArray<int>(physical, "flat_net2pin_map");
  const auto physical_start = carrierArray<int>(physical, "flat_net2pin_start_map");
  if (physical_start.size() != state.net_names.size() + 1 || physical_start.front() != 0
      || physical_start.back() != static_cast<int>(physical_flat.size())) {
    throw py::value_error("physical net CSR does not align");
  }
  for (int net = 0; net < static_cast<int>(state.net_names.size()); ++net) {
    const int begin = physical_start[net], end = physical_start[net + 1];
    if (begin < 0 || end < begin || end > static_cast<int>(physical_flat.size())) {
      throw py::value_error("invalid physical net CSR range");
    }
    state.net_pins.emplace_back(physical_flat.begin() + begin, physical_flat.begin() + end);
  }
  const auto lib_names = carrierArray<std::string>(base, "flat_libcell_names");
  const auto lib_ids = nameIndex(lib_names);
  const auto main_start = carrierArray<int>(base, "main_id_2_cell_id_start");
  const auto lib_pin_start = carrierArray<int>(base, "cell_id_2_libpin_id_start");
  const auto lib_pin_names = carrierArray<std::string>(base, "flat_lib_pin_names");
  state.cell_arc_start = carrierArray<int>(base, "cell_id_2_arc_id_start");
  std::vector<int> main_id(state.node_names.size(), -1), cell_offset(main_id.size(), 0);
  state.cell_by_node.assign(state.node_names.size(), -1);
  const auto old_main = carrierArray<int>(base, "inst_main_id");
  for (int node = 0; node < static_cast<int>(masters.size()); ++node) {
    if (masters[node].empty()) continue;  // IO and synthetic physical obstacles.
    const auto found = lib_ids.find(masters[node]);
    if (found == lib_ids.end()) {
      const auto old = base_node_ids.find(state.node_names[node]);
      if (old == base_node_ids.end() || old_main.at(old->second) >= 0) {
        throw py::value_error("physical master is absent from Liberty: " + masters[node]);
      }
      continue;  // Existing physical-only cells stay outside the timing graph.
    }
    const int cell = found->second;
    const int main = std::upper_bound(main_start.begin(), main_start.end(), cell) - main_start.begin() - 1;
    state.cell_by_node[node] = cell;
    main_id[node] = main;
    cell_offset[node] = cell - main_start.at(main);
  }
  std::vector<int> lib_offset(state.pin_names.size(), -1);
  for (int pin = 0; pin < static_cast<int>(state.pin_names.size()); ++pin) {
    const int owner = state.pin2node.at(pin);
    const int cell = state.cell_by_node.at(owner);
    if (cell < 0) continue;
    const auto prefix = state.node_names[owner] + ":";
    if (state.pin_names[pin].rfind(prefix, 0) != 0) throw py::value_error("pin owner identity mismatch");
    const std::string port = state.pin_names[pin].substr(prefix.size());
    for (int i = lib_pin_start.at(cell); i < lib_pin_start.at(cell + 1); ++i) {
      if (lib_pin_names.at(i) == port) lib_offset[pin] = i - lib_pin_start[cell];
    }
    if (lib_offset[pin] < 0) throw py::value_error("physical signal pin is absent from Liberty");
  }
  std::vector<int> pin_net(state.pin_names.size(), -1);
  std::vector<double> weights;
  for (int net = 0; net < static_cast<int>(state.net_pins.size()); ++net) {
    weights.push_back(clocks.count(state.net_names[net]) ? 0.0 : 1.0);
    for (const int pin : state.net_pins[net]) {
      if (pin < 0 || pin >= static_cast<int>(pin_net.size()) || pin_net[pin] != -1) {
        throw py::value_error("physical pin belongs to multiple nets or is out of range");
      }
      pin_net[pin] = net;
    }
  }
  if (std::find(pin_net.begin(), pin_net.end(), -1) != pin_net.end()) {
    throw py::value_error("physical carrier contains an unconnected pin");
  }
  std::vector<std::vector<int>> node_pins(state.node_names.size());
  int movable_pins = 0;
  const int movable_nodes = py::cast<int>(physical.attr("num_nodes"))
      - py::cast<int>(physical.attr("num_terminals")) - py::cast<int>(physical.attr("num_terminal_NIs"));
  for (int pin = 0; pin < static_cast<int>(state.pin_names.size()); ++pin) {
    node_pins.at(state.pin2node[pin]).push_back(pin);
    movable_pins += state.pin2node[pin] < movable_nodes;
  }
  auto& out = state.values;
  out["pin_names"] = state.pin_names;
  out["pin_name2id_map"] = state.pin_ids;
  out["num_pins"] = state.pin_names.size();
  out["num_nets"] = state.net_names.size();
  out["net_names"] = state.net_names;
  out["net_name2id_map"] = net_ids;
  out["net_weights"] = weights;
  out["pin2node_map"] = state.pin2node;
  out["pin_to_node_id"] = state.pin2node;
  out["pin_to_inst_id"] = state.pin2node;
  out["pin2net_map"] = pin_net;
  out["pin_offset_x"] = x;
  out["pin_offset_y"] = y;
  out["pin_direct"] = orientation;
  out["num_movable_pins"] = movable_pins;
  out["inst_main_id"] = main_id;
  out["inst_libcell_offset"] = cell_offset;
  out["pin_2_libpin_offset"] = lib_offset;
  exportRows(out, "net2pin_map", "flat_net2pin_map", "flat_net2pin_start_map", state.net_pins);
  exportRows(out, "node2pin_map", "flat_node2pin_map", "flat_node2pin_start_map", node_pins);
  compileCarrierGraph(base, state);
  return out;
}

}  // namespace rebuffer
