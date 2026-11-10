// SPDX-License-Identifier: BSD-3-Clause
// Python dict -> native view conversion for the rebuffer bindings.

#include <cmath>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "pybind11/numpy.h"

#include "pybind11/pybind11.h"

#include "pybind11/stl.h"

#include "rebuffer_liberty.hh"

#include "rebuffer_repair.hh"

#include "rebuffer_types.hh"


namespace py = pybind11;

namespace rebuffer_python {

template <typename T>
std::vector<T> asVector(const py::handle& value, const char* field)
{
  const auto array = py::array_t<T, py::array::c_style | py::array::forcecast>::ensure(
      value);
  if (!array || array.ndim() != 1) {
    throw py::value_error(std::string(field) + " must be one-dimensional");
  }
  if (array.size() == 0) {
    return {};
  }
  const auto* pointer = array.data();
  return std::vector<T>(pointer, pointer + array.size());
}

template <typename T>
std::vector<T> optionalVector(const py::dict& values, const char* field)
{
  if (!values.contains(field) || values[field].is_none()) {
    return {};
  }
  return asVector<T>(values[field], field);
}

int asInt(const py::dict& values, const char* field)
{
  if (!values.contains(field)) {
    throw py::value_error(std::string("missing ") + field);
  }
  return py::cast<int>(values[field]);
}

std::vector<std::string> namesFromDict(const py::dict& values, const char* field);

rebuffer::DesignView designFromDict(const py::dict& values)
{
  rebuffer::DesignView design;
  design.dbu = asInt(values, "dbu");
  design.num_nets = asInt(values, "num_nets");
  design.num_pins = asInt(values, "num_pins");
  design.num_nodes = asInt(values, "num_nodes");
  design.num_base_nodes = values.contains("num_base_nodes")
      ? py::cast<int>(values["num_base_nodes"]) : design.num_nodes;
  design.num_terminals = values.contains("num_terminals")
      ? py::cast<int>(values["num_terminals"]) : 0;
  design.num_terminal_nis = values.contains("num_terminal_NIs")
      ? py::cast<int>(values["num_terminal_NIs"]) : 0;
  design.flat_net2pin_map
      = asVector<int>(values["flat_net2pin_map"], "flat_net2pin_map");
  design.flat_net2pin_start_map = asVector<int>(
      values["flat_net2pin_start_map"], "flat_net2pin_start_map");
  design.net2driver_pin_map
      = asVector<int>(values["net2driver_pin_map"], "net2driver_pin_map");
  design.pin2node_map
      = asVector<int>(values["pin2node_map"], "pin2node_map");
  design.pin2net_map = asVector<int>(values["pin2net_map"], "pin2net_map");
  design.pin_offset_x
      = asVector<double>(values["pin_offset_x"], "pin_offset_x");
  design.pin_offset_y
      = asVector<double>(values["pin_offset_y"], "pin_offset_y");
  design.node_x = asVector<double>(values["node_x"], "node_x");
  design.node_y = asVector<double>(values["node_y"], "node_y");
  design.pin_cap_limit
      = optionalVector<double>(values, "flat_lib_pin_cap_limit");
  design.pin_cap = optionalVector<double>(values, "flat_lib_pin_cap");
  design.pin_slew_limit
      = optionalVector<double>(values, "flat_lib_pin_slew_limit");
  design.pin_drive_resistance
      = optionalVector<double>(values, "flat_lib_pin_drive_resistance");
  design.inst_main_id = optionalVector<int>(values, "inst_main_id");
  design.inst_libcell_offset
      = optionalVector<int>(values, "inst_libcell_offset");
  design.main_id_2_cell_id_start
      = optionalVector<int>(values, "main_id_2_cell_id_start");
  design.cell_id_2_libpin_id_start
      = optionalVector<int>(values, "cell_id_2_libpin_id_start");
  design.pin_2_libpin_offset
      = optionalVector<int>(values, "pin_2_libpin_offset");
  design.node_is_buffer
      = optionalVector<std::uint8_t>(values, "node_is_buffer");
  design.node_is_fixed
      = optionalVector<std::uint8_t>(values, "node_is_fixed");
  if (values.contains("names")) {
    const auto names = values["names"].cast<py::dict>();
    design.pin_names = namesFromDict(names, "pin");
  }
  return design;
}

rebuffer::TopologyView topologyFromDict(const py::dict& values)
{
  rebuffer::TopologyView topology;
  topology.steiner_node_base = asInt(values, "steiner_node_base");
  topology.net_ids = asVector<int>(values["net_ids"], "net_ids");
  topology.net_flat_topo_sort_start = asVector<int>(
      values["net_flat_topo_sort_start"], "net_flat_topo_sort_start");
  topology.net_flat_topo_sort = asVector<int>(
      values["net_flat_topo_sort"], "net_flat_topo_sort");
  topology.pin_fa = asVector<int>(values["pin_fa"], "pin_fa");
  auto coordinateMap = [&](const char* field) {
    std::unordered_map<int, double> result;
    if (!values.contains(field) || values[field].is_none()) {
      return result;
    }
    if (!py::isinstance<py::dict>(values[field])) {
      // Dense per-node array: keep only the nodes listed by this payload.
      const auto dense = asVector<double>(values[field], field);
      result.reserve(topology.net_flat_topo_sort.size());
      for (const int node : topology.net_flat_topo_sort) {
        if (node < 0 || node >= static_cast<int>(dense.size())) {
          throw py::value_error(std::string(field) + " is shorter than the topology node ids");
        }
        result.emplace(node, dense[node]);
      }
      return result;
    }
    for (const auto& item : values[field].cast<py::dict>()) {
      result.emplace(py::cast<int>(item.first), py::cast<double>(item.second));
    }
    return result;
  };
  topology.node_x = coordinateMap("node_x");
  topology.node_y = coordinateMap("node_y");
  return topology;
}

std::vector<std::string> namesFromDict(const py::dict& values, const char* field)
{
  std::vector<std::string> names;
  if (!values.contains(field)) {
    return names;
  }
  for (const auto& item : values[field].cast<py::sequence>()) {
    names.push_back(py::cast<std::string>(item));
  }
  return names;
}

rebuffer::LutTable lutFromArc(
    const std::vector<int>& value_start,
    const std::vector<int>& trans_start,
    const std::vector<int>& cap_start,
    const std::vector<double>& trans_axis,
    const std::vector<double>& cap_axis,
    const std::vector<double>& values,
    int arc_index,
    int table_index,
    int dim0,
    int dim1)
{
  rebuffer::LutTable table;
  table.dim0 = dim0;
  table.dim1 = dim1;
  table.trans_axis.assign(
      trans_axis.begin() + trans_start[arc_index],
      trans_axis.begin() + trans_start[arc_index + 1]);
  table.cap_axis.assign(
      cap_axis.begin() + cap_start[arc_index],
      cap_axis.begin() + cap_start[arc_index + 1]);
  const int table_size = dim0 * dim1;
  const int begin = value_start[arc_index] + table_index * table_size;
  table.values.assign(
      values.begin() + begin,
      values.begin() + begin + table_size);
  return table;
}

std::vector<std::string> stringsFromValue(const py::handle& value)
{
  std::vector<std::string> result;
  if (value.is_none()) {
    return result;
  }
  for (const auto& item : value.cast<py::sequence>()) {
    result.push_back(py::cast<std::string>(item));
  }
  return result;
}

rebuffer::LibertyView libertyFromDict(const py::dict& values)
{
  rebuffer::LibertyView liberty;
  liberty.dbu = asInt(values, "dbu");
  const auto names = namesFromDict(values, "names");
  const auto main_id = asVector<int>(values["cell_main_id"], "cell_main_id");
  const auto size = asVector<int>(values["cell_size"], "cell_size");
  const auto vt = asVector<int>(values["cell_vt"], "cell_vt");
  const auto area = asVector<double>(values["cell_area"], "cell_area");
  const auto leakage
      = asVector<double>(values["cell_leakage"], "cell_leakage");
  const auto input_cap
      = asVector<double>(values["cell_input_cap"], "cell_input_cap");
  const auto output_cap = values.contains("cell_output_cap")
                              ? asVector<double>(values["cell_output_cap"],
                                                 "cell_output_cap")
                              : std::vector<double>{};
  const auto intrinsic_delay = optionalVector<double>(values, "cell_intrinsic_delay");
  const auto drive_resistance = optionalVector<double>(values, "cell_drive_resistance");
  const auto max_cap
      = asVector<double>(values["cell_max_cap"], "cell_max_cap");
  const auto max_slew
      = asVector<double>(values["cell_max_slew"], "cell_max_slew");
  const auto is_buffer
      = asVector<bool>(values["cell_is_buffer"], "cell_is_buffer");
  const auto cell_arc_start
      = asVector<int>(values["cell_arc_start"], "cell_arc_start");
  const auto arc_source = values.contains("arc_source")
                              ? stringsFromValue(values["arc_source"])
                              : std::vector<std::string>{};
  const auto arc_sink = values.contains("arc_sink")
                            ? stringsFromValue(values["arc_sink"])
                            : std::vector<std::string>{};
  const auto arc_local_id = values.contains("arc_local_id")
                                ? asVector<int>(values["arc_local_id"],
                                                "arc_local_id")
                                : std::vector<int>{};
  const auto arc_sense = asVector<int>(values["arc_sense"], "arc_sense");
  const auto arc_trigger
      = asVector<int>(values["arc_trigger"], "arc_trigger");
  const auto arc_dim = asVector<int>(values["arc_dim"], "arc_dim");
  const auto arc_trans_start
      = asVector<int>(values["arc_trans_start"], "arc_trans_start");
  const auto arc_trans_axis
      = asVector<double>(values["arc_trans_axis"], "arc_trans_axis");
  const auto arc_cap_start
      = asVector<int>(values["arc_cap_start"], "arc_cap_start");
  const auto arc_cap_axis
      = asVector<double>(values["arc_cap_axis"], "arc_cap_axis");
  const auto arc_value_start
      = asVector<int>(values["arc_value_start"], "arc_value_start");
  const auto arc_values
      = asVector<double>(values["arc_values"], "arc_values");

  if (names.size() != main_id.size() || names.size() != size.size()
      || names.size() != vt.size() || names.size() != area.size()
      || names.size() != leakage.size()
      || names.size() != input_cap.size()
      || (!output_cap.empty() && names.size() != output_cap.size())
      || (!intrinsic_delay.empty() && names.size() != intrinsic_delay.size())
      || (!drive_resistance.empty() && names.size() != drive_resistance.size())
      || names.size() != max_cap.size()
      || names.size() != max_slew.size()
      || names.size() != is_buffer.size()
      || names.size() + 1 != cell_arc_start.size()) {
    throw py::value_error("Liberty cell arrays have inconsistent lengths");
  }
  const int num_arcs = static_cast<int>(arc_sense.size());
  if (arc_trigger.size() != arc_sense.size()
      || (!arc_source.empty() && arc_source.size() != arc_sense.size())
      || (!arc_sink.empty() && arc_sink.size() != arc_sense.size())
      || (!arc_local_id.empty() && arc_local_id.size() != arc_sense.size())
      || arc_dim.size() != 2 * num_arcs
      || arc_value_start.size() != num_arcs + 1
      || arc_trans_start.size() != num_arcs + 1
      || arc_cap_start.size() != num_arcs + 1) {
    throw py::value_error("Liberty arc arrays have inconsistent lengths");
  }

  liberty.cells.resize(names.size());
  for (std::size_t cell_index = 0; cell_index < names.size(); ++cell_index) {
    auto& cell = liberty.cells[cell_index];
    cell.name = names[cell_index];
    cell.main_id = main_id[cell_index];
    cell.size = size[cell_index];
    cell.vt = vt[cell_index];
    cell.area = area[cell_index];
    cell.leakage = leakage[cell_index];
    cell.input_cap = input_cap[cell_index];
    cell.output_cap = output_cap.empty() ? 0.0 : output_cap[cell_index];
    cell.max_cap = max_cap[cell_index];
    cell.max_slew = max_slew[cell_index];
    cell.is_buffer = is_buffer[cell_index];
    const int begin = cell_arc_start[cell_index];
    const int end = cell_arc_start[cell_index + 1];
    for (int arc_index = begin; arc_index < end; ++arc_index) {
      const int dim0 = arc_dim[2 * arc_index];
      const int dim1 = arc_dim[2 * arc_index + 1];
      if (dim0 <= 0 || dim1 <= 0) {
        continue;
      }
      rebuffer::TimingArcView arc;
      if (!arc_local_id.empty()) {
        arc.local_id = arc_local_id[arc_index];
      }
      if (!arc_source.empty()) {
        arc.source_name = arc_source[arc_index];
      }
      if (!arc_sink.empty()) {
        arc.sink_name = arc_sink[arc_index];
      }
      arc.sense = arc_sense[arc_index];
      arc.trigger = arc_trigger[arc_index];
      arc.fall_delay = lutFromArc(
          arc_value_start,
          arc_trans_start,
          arc_cap_start,
          arc_trans_axis,
          arc_cap_axis,
          arc_values,
          arc_index,
          0,
          dim0,
          dim1);
      arc.rise_delay = lutFromArc(
          arc_value_start,
          arc_trans_start,
          arc_cap_start,
          arc_trans_axis,
          arc_cap_axis,
          arc_values,
          arc_index,
          1,
          dim0,
          dim1);
      arc.fall_transition = lutFromArc(
          arc_value_start,
          arc_trans_start,
          arc_cap_start,
          arc_trans_axis,
          arc_cap_axis,
          arc_values,
          arc_index,
          2,
          dim0,
          dim1);
      arc.rise_transition = lutFromArc(
          arc_value_start,
          arc_trans_start,
          arc_cap_start,
          arc_trans_axis,
          arc_cap_axis,
          arc_values,
          arc_index,
          3,
          dim0,
          dim1);
      cell.arcs.push_back(std::move(arc));
    }
    cell.intrinsic_delay = intrinsic_delay.empty()
                               ? rebuffer::cellIntrinsicDelay(cell)
                               : intrinsic_delay[cell_index];
    cell.drive_resistance = drive_resistance.empty()
                                ? rebuffer::cellDriveResistance(cell)
                                : drive_resistance[cell_index];
  }
  if (values.contains("_native_cell_metadata") && !values["_native_cell_metadata"].is_none()) {
    liberty.cell_metadata = values["_native_cell_metadata"].cast<std::shared_ptr<rebuffer::RepairLibraryMetadata>>();
    rebuffer::initializeBufferCellSelection(liberty);
  }
  return liberty;
}

py::dict characterizationResult(
    const rebuffer::CharacterizationResult& result)
{
  auto bufferDict = [](const rebuffer::CharacterizedBuffer& buffer) {
    py::dict row;
    row["cell_id"] = buffer.cell_index;
    row["name"] = buffer.name;
    row["main_id"] = buffer.main_id;
    row["area"] = buffer.area;
    row["leakage"] = buffer.leakage;
    row["input_cap"] = buffer.input_cap;
    row["max_cap"] = buffer.max_cap;
    row["max_slew"] = buffer.max_slew;
    row["drive_resistance"] = buffer.drive_resistance;
    row["intrinsic_delay"] = buffer.intrinsic_delay;
    row["output_cap"] = buffer.output_cap;
    row["target_load"] = buffer.target_load;
    row["margined_max_cap"] = buffer.margined_max_cap;
    return row;
  };
  py::dict output;
  output["status"] = "ok";
  output["target_slew_rise"] = result.target_slew[0];
  output["target_slew_fall"] = result.target_slew[1];
  py::list filtered;
  for (const auto& buffer : result.filtered_buffers) {
    filtered.append(bufferDict(buffer));
  }
  py::list fast;
  for (const auto& buffer : result.fast_buffers) {
    fast.append(bufferDict(buffer));
  }
  output["filtered_buffers"] = std::move(filtered);
  output["fast_buffers"] = std::move(fast);
  if (result.lowest_drive_index >= 0
      && result.lowest_drive_index
          < static_cast<int>(result.filtered_buffers.size())) {
    output["buffer_lowest_drive"]
        = result.filtered_buffers[result.lowest_drive_index].name;
  } else {
    output["buffer_lowest_drive"] = "";
  }
  return output;
}

rebuffer::TimingView timingFromDict(const py::dict& values)
{
  rebuffer::TimingView timing;
  timing.pin_slew_rise
      = asVector<double>(values["pin_slew_rise"], "pin_slew_rise");
  timing.pin_slew_fall
      = asVector<double>(values["pin_slew_fall"], "pin_slew_fall");
  timing.pin_load = asVector<double>(values["pin_load"], "pin_load");
  timing.pin_cap = asVector<double>(values["pin_cap"], "pin_cap");
  timing.pin_max_slew
      = asVector<double>(values["pin_max_slew"], "pin_max_slew");
  timing.pin_max_cap
      = asVector<double>(values["pin_max_cap"], "pin_max_cap");
  timing.pin_slack = optionalVector<double>(values, "pin_slack");
  timing.pin_arrival_rise = optionalVector<double>(values, "pin_arrival_rise");
  timing.pin_arrival_fall = optionalVector<double>(values, "pin_arrival_fall");
  timing.pin_required_rise = optionalVector<double>(values, "pin_required_rise");
  timing.pin_required_fall = optionalVector<double>(values, "pin_required_fall");
  timing.clock_pins = optionalVector<int>(values, "clock_pins");
  timing.pin_rank = optionalVector<int>(values, "pin_rank");
  if (values.contains("path_driver_input_pin")) {
    timing.path_driver_input_pin = py::cast<int>(values["path_driver_input_pin"]);
  }
  if (values.contains("path_driver_input_transition")) {
    timing.path_driver_input_transition
        = py::cast<int>(values["path_driver_input_transition"]);
  }
  if (values.contains("path_driver_output_transition")) {
    timing.path_driver_output_transition
        = py::cast<int>(values["path_driver_output_transition"]);
  }
  if (values.contains("path_driver_lib_arc_id")) {
    timing.path_driver_lib_arc_id
        = py::cast<int>(values["path_driver_lib_arc_id"]);
  }
  if (values.contains("path_arc_delays")) {
    timing.path_arc_delays
        = optionalVector<double>(values, "path_arc_delays");
  }
  if (values.contains("path_arc_local_ids")) {
    timing.path_arc_local_ids
        = optionalVector<int>(values, "path_arc_local_ids");
  }
  if (values.contains("path_transitions")) {
    timing.path_transitions
        = optionalVector<int>(values, "path_transitions");
  }
  if (values.contains("path_driver_input_arrival")) {
    timing.path_driver_input_arrival
        = py::cast<double>(values["path_driver_input_arrival"]);
  }
  if (values.contains("path_driver_output_arrival")) {
    timing.path_driver_output_arrival
        = py::cast<double>(values["path_driver_output_arrival"]);
  }
  if (values.contains("path_driver_arrival_valid")) {
    timing.path_driver_arrival_valid
        = py::cast<bool>(values["path_driver_arrival_valid"]);
  }
  return timing;
}

rebuffer::SteinerView steinerFromDict(const py::dict& values)
{
  rebuffer::SteinerView steiner;
  steiner.steiner_node_base = asInt(values, "steiner_node_base");
  steiner.net_ids = optionalVector<int>(values, "net_ids");
  steiner.net_flat_topo_sort_start = asVector<int>(
      values["net_flat_topo_sort_start"], "net_flat_topo_sort_start");
  steiner.net_flat_topo_sort = asVector<int>(
      values["net_flat_topo_sort"], "net_flat_topo_sort");
  steiner.pin_fa = asVector<int>(values["pin_fa"], "pin_fa");
  steiner.flat_pin_to_start = asVector<int>(
      values["flat_pin_to_start"], "flat_pin_to_start");
  steiner.flat_pin_to
      = asVector<int>(values["flat_pin_to"], "flat_pin_to");
  steiner.node_x = asVector<double>(values["node_x"], "node_x");
  steiner.node_y = asVector<double>(values["node_y"], "node_y");
  steiner.edge_resistance = asVector<double>(
      values["edge_resistance"], "edge_resistance");
  steiner.node_capacitance = asVector<double>(
      values["node_capacitance"], "node_capacitance");
  return steiner;
}

rebuffer::RepairConfig configFromDict(const py::dict& values)
{
  rebuffer::RepairConfig config;
  auto number = [&](const char* key, double fallback) {
    return values.contains(key) ? py::cast<double>(values[key]) : fallback;
  };
  auto integer = [&](const char* key, int fallback) {
    return values.contains(key) ? py::cast<int>(values[key]) : fallback;
  };
  auto boolean = [&](const char* key, bool fallback) {
    return values.contains(key) ? py::cast<bool>(values[key]) : fallback;
  };
  config.slew_margin = number("slew_margin", config.slew_margin);
  config.cap_margin = number("cap_margin", config.cap_margin);
  config.r_unit = number("r_unit", config.r_unit);
  config.c_unit = number("c_unit", config.c_unit);
  config.max_fanout = integer("max_fanout", config.max_fanout);
  config.tree_max_buffers = integer("tree_max_buffers", config.tree_max_buffers);
  config.insert_buffers = boolean("insert_buffers", config.insert_buffers);
  config.resize_drivers = boolean("resize_drivers", config.resize_drivers);
  config.resize_slew_threshold
      = number("resize_slew_threshold", config.resize_slew_threshold);
  config.repair_design_reverse_topological = boolean(
      "repair_design_reverse_topological",
      config.repair_design_reverse_topological);
  config.repair_design_local_update = boolean(
      "repair_design_local_update", config.repair_design_local_update);
  config.repair_design_load_slew_annotation = boolean(
      "repair_design_load_slew_annotation",
      config.repair_design_load_slew_annotation);
  config.repair_design_openroad_driver_resize = boolean(
      "repair_design_openroad_driver_resize",
      config.repair_design_openroad_driver_resize);
  config.repair_design_fanout_repair = boolean(
      "repair_design_fanout_repair", config.repair_design_fanout_repair);
  config.repair_design_resize_to_target_slew = boolean(
      "repair_design_resize_to_target_slew",
      config.repair_design_resize_to_target_slew);
  config.repair_design_target_cell_math = boolean(
      "repair_design_target_cell_math", config.repair_design_target_cell_math);
  config.repair_design_target_cell_selection = boolean(
      "repair_design_target_cell_selection", config.repair_design_target_cell_selection);
  config.repair_design_openroad_repeater_resize = boolean(
      "repair_design_openroad_repeater_resize",
      config.repair_design_openroad_repeater_resize);
  config.sizing_area_limit = number("sizing_area_limit", config.sizing_area_limit);
  config.sizing_leakage_limit
      = number("sizing_leakage_limit", config.sizing_leakage_limit);
  config.repair_design_max_wire_length_um = values.contains(
      "repair_design_max_wire_length_um")
      ? number("repair_design_max_wire_length_um",
               config.repair_design_max_wire_length_um)
      : number("max_wire_length", config.repair_design_max_wire_length_um);
  config.repair_net_tree = boolean("repair_net_tree", config.repair_net_tree);
  config.slew_rc_factor = number("slew_rc_factor", config.slew_rc_factor);
  config.slew_shape_factor
      = number("slew_shape_factor", config.slew_rc_factor);
  config.min_slack_improvement
      = number("min_slack_improvement", config.min_slack_improvement);
  config.enforce_timing_buffer_move_fanout = boolean(
      "enforce_timing_buffer_move_fanout",
      config.enforce_timing_buffer_move_fanout);
  config.timing_max_fanout
      = integer("timing_max_fanout", config.timing_max_fanout);
  config.emit_candidate_trace
      = boolean("emit_candidate_trace", config.emit_candidate_trace);
  config.use_downstream_slew_delay_bonus = boolean(
      "use_downstream_slew_delay_bonus",
      config.use_downstream_slew_delay_bonus);
  config.max_timing_alternatives
      = integer("max_timing_alternatives", config.max_timing_alternatives);
  config.tree_enforce_electrical = boolean("tree_enforce_electrical", config.tree_enforce_electrical);
  config.tree_max_wire_length_um = number("tree_max_wire_length_um", config.tree_max_wire_length_um);
  config.tree_timing_rounds = integer("tree_timing_rounds", config.tree_timing_rounds);
  config.tree_recovery_rounds = integer("tree_recovery_rounds", config.tree_recovery_rounds);
  if (config.tree_timing_rounds < 1 || config.tree_timing_rounds > 3
      || config.tree_recovery_rounds < 0 || config.tree_recovery_rounds > 5
      || !std::isfinite(config.r_unit) || config.r_unit < 0.0
      || !std::isfinite(config.c_unit) || config.c_unit < 0.0
      || !std::isfinite(config.tree_max_wire_length_um) || config.tree_max_wire_length_um < 0.0
      || !std::isfinite(config.repair_design_max_wire_length_um)
             || config.repair_design_max_wire_length_um < 0.0
      || config.slew_shape_factor <= 0.0 || !std::isfinite(config.slew_shape_factor)
      || !std::isfinite(config.cap_margin) || config.cap_margin < 0.0 || config.cap_margin >= 100.0
      || !std::isfinite(config.slew_margin) || config.slew_margin < 0.0 || config.slew_margin >= 100.0) {
    throw py::value_error("invalid_current_tree_search_config");
  }
  return config;
}

rebuffer::CharacterizationResult characterizationFromDict(
    const py::dict& values)
{
  rebuffer::CharacterizationResult result;
  auto parse = [](const py::object& value) {
    std::vector<rebuffer::CharacterizedBuffer> buffers;
    if (value.is_none()) {
      return buffers;
    }
    for (const auto& item : value.cast<py::sequence>()) {
      const auto row = item.cast<py::dict>();
      rebuffer::CharacterizedBuffer buffer;
      buffer.cell_index = py::cast<int>(row["cell_id"]);
      buffer.name = py::cast<std::string>(row["name"]);
      buffer.main_id = py::cast<int>(row["main_id"]);
      buffer.area = py::cast<double>(row["area"]);
      buffer.leakage = py::cast<double>(row["leakage"]);
      buffer.input_cap = py::cast<double>(row["input_cap"]);
      buffer.max_cap = py::cast<double>(row["max_cap"]);
      buffer.max_slew = py::cast<double>(row["max_slew"]);
      buffer.drive_resistance
          = py::cast<double>(row["drive_resistance"]);
      if (row.contains("intrinsic_delay")) {
        buffer.intrinsic_delay = py::cast<double>(row["intrinsic_delay"]);
      }
      if (row.contains("output_cap")) {
        buffer.output_cap = py::cast<double>(row["output_cap"]);
      }
      buffer.target_load = py::cast<double>(row["target_load"]);
      buffer.margined_max_cap
          = py::cast<double>(row["margined_max_cap"]);
      buffers.push_back(std::move(buffer));
    }
    return buffers;
  };
  result.filtered_buffers = parse(values["filtered_buffers"]);
  result.fast_buffers = parse(values["fast_buffers"]);
  result.target_slew[0] = py::cast<double>(values["target_slew_rise"]);
  result.target_slew[1] = py::cast<double>(values["target_slew_fall"]);
  if (values.contains("buffer_lowest_drive")) {
    const auto lowest
        = py::cast<std::string>(values["buffer_lowest_drive"]);
    for (int index = 0;
         index < static_cast<int>(result.filtered_buffers.size());
         ++index) {
      if (result.filtered_buffers[index].name == lowest) {
        result.lowest_drive_index = index;
        break;
      }
    }
  }
  return result;
}

py::list netSummaries(const rebuffer::ForestResult& forest)
{
  py::list result;
  for (const auto& summary : forest.net_summaries) {
    py::dict row;
    row["net_id"] = summary.net_id;
    row["driver_pin"] = summary.driver_pin;
    row["root_node"] = summary.root_node;
    row["node_count"] = summary.node_count;
    row["pin_node_count"] = summary.pin_node_count;
    row["steiner_node_count"] = summary.steiner_node_count;
    row["edge_count"] = summary.edge_count;
    row["max_depth"] = summary.max_depth;
    row["max_fanout"] = summary.max_fanout;
    row["driver_in_topology"] = summary.driver_in_topology;
    result.append(std::move(row));
  }
  return result;
}

py::dict bufferActionDict(const rebuffer::BufferAction& action)
{
  py::dict row;
  row["net_id"] = action.net_id;
  row["driver_pin"] = action.driver_pin;
  row["downstream_pins"] = action.downstream_pins;
  row["buffer_cell_index"] = action.buffer_cell_index;
  if (action.parent_node_id >= 0 && action.child_node_id >= 0) {
    row["parent_node_id"] = action.parent_node_id;
    row["child_node_id"] = action.child_node_id;
    if (action.source_parent_node_id >= 0
        && action.source_child_node_id >= 0) {
      row["source_parent_node_id"] = action.source_parent_node_id;
      row["source_child_node_id"] = action.source_child_node_id;
    }
    if (action.split_ratio > 0.0 && action.split_ratio < 1.0) {
      row["split_ratio"] = action.split_ratio;
    }
    if (action.reason == "tree_vgdp_insert" && action.split_ratio >= 0.0) {
      row["source_edge_fraction"] = action.split_ratio;
    }
  }
  row["segment_tree_depth"] = action.tree_depth;
  row["x_dbu"] = action.x_dbu;
  row["y_dbu"] = action.y_dbu;
  row["downstream_load"]
      = std::round(action.downstream_load * 1.0e12) / 1.0e12;
  row["target_load"] = std::round(action.target_load * 1.0e12) / 1.0e12;
  if (action.initial_buffer_cell_index >= 0) {
    row["initial_buffer_cell_index"] = action.initial_buffer_cell_index;
    row["initial_downstream_load"] = action.initial_downstream_load;
  }
  row["reason"] = action.reason;
  return row;
}

}  // namespace rebuffer_python
