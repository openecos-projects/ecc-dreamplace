// SPDX-License-Identifier: BSD-3-Clause
#include "rebuffer_overlay_inputs.hh"
#include "rebuffer_carrier.hh"
#include "rebuffer_overlay_actions.hh"

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
