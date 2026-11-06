// SPDX-License-Identifier: BSD-3-Clause
#include "rebuffer_overlay_inputs.hh"
#include "rebuffer_carrier.hh"
#include "rebuffer_overlay_actions.hh"

namespace py = pybind11;
namespace rebuffer {

OverlayActionInputs overlayActionInputs(const py::object& base)
{
  return {carrierArray<std::string>(base, "flat_libcell_names"),
          carrierArray<std::string>(base, "pin_names"),
          carrierArray<std::string>(base, "net_names"),
          carrierArray<int>(base, "pin2node_map"),
          carrierArray<int>(base, "pin2net_map"),
          carrierArray<int>(base, "net2driver_pin_map")};
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
