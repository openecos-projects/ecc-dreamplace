#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openroad_place_io_bridge.h"
#include "openroad_pyplacedb_export.h"

namespace py = pybind11;

namespace {

using dreamplace::placeio_openroad::OpenRoadPlaceIOBridge;
using dreamplace::placeio_openroad::PyPlaceDB;

std::vector<std::string> parseFlagValues(const py::list& args, const std::string& flag)
{
  std::vector<std::string> values;
  const auto arg_count = py::len(args);
  for (py::ssize_t i = 0; i < arg_count; ++i) {
    if (py::cast<std::string>(args[i]) == flag && i + 1 < arg_count) {
      values.push_back(py::cast<std::string>(args[i + 1]));
    }
  }
  return values;
}

int parseFlagInt(const py::list& args, const std::string& flag, int default_value)
{
  const auto values = parseFlagValues(args, flag);
  if (values.empty()) {
    return default_value;
  }
  try {
    return std::max(1, std::stoi(values.back()));
  } catch (const std::exception&) {
    return default_value;
  }
}

using BridgePtr = std::shared_ptr<OpenRoadPlaceIOBridge>;

BridgePtr forward(const py::list& args)
{
  const auto lef_files = parseFlagValues(args, "--lef_input");
  const auto def_files = parseFlagValues(args, "--def_input");
  const auto liberty_files = parseFlagValues(args, "--lib_input");
  const auto sdc_files = parseFlagValues(args, "--sdc_input");
  auto vt_suffixes = parseFlagValues(args, "--vt_suffix");
  const int thread_count = parseFlagInt(args, "--num_threads", 8);
  if (def_files.empty()) {
    throw std::runtime_error("placeio_openroad requires exactly one DEF input");
  }
  if (sdc_files.size() > 1) {
    throw std::runtime_error("placeio_openroad accepts at most one SDC input");
  }
  return std::make_shared<OpenRoadPlaceIOBridge>(
      lef_files,
      def_files.front(),
      liberty_files,
      sdc_files.empty() ? std::string() : sdc_files.front(),
      vt_suffixes,
      thread_count);
}

PyPlaceDB pydb(const BridgePtr& bridge)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->exportPyDBView();
}

void apply(const BridgePtr& bridge, const py::object& node_x, const py::object& node_y)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  bridge->syncToOpenRoad(node_x, node_y);
}

py::dict apply_sizing(const BridgePtr& bridge,
                      const py::object& inst_cell_ids,
                      const py::object& cell_master_names)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->applySizing(inst_cell_ids, cell_master_names);
}

py::dict query_diff_guided_batch_timing_metrics(const BridgePtr& bridge)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->queryDiffGuidedBatchTimingMetrics();
}

py::dict query_diff_guided_batch_timing_samples(const BridgePtr& bridge,
                                                const py::dict& sample_request)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->queryDiffGuidedBatchTimingSamples(sample_request);
}

py::dict query_diff_guided_batch_dynamic_conflict_signature(const BridgePtr& bridge,
                                                            const py::dict& config)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->queryDiffGuidedBatchDynamicConflictSignature(config);
}

py::dict evaluate_diff_guided_batch_actions(const BridgePtr& bridge,
                                            const std::vector<py::dict>& actions,
                                            const py::dict& config)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->evaluateDiffGuidedBatchActions(actions, config);
}

py::dict apply_diff_guided_batch_transaction(const BridgePtr& bridge,
                                             const std::vector<py::dict>& action_results,
                                             const py::dict& config)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->applyDiffGuidedBatchTransaction(action_results, config);
}

py::dict run_diff_guided_batch_loop(const BridgePtr& bridge,
                                    const std::vector<py::dict>& seed_queue,
                                    const py::dict& conflict_precompute,
                                    const py::dict& config)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->runDiffGuidedBatchLoop(seed_queue, conflict_precompute, config);
}

py::dict run_diff_guided_batch_loop_compact(const BridgePtr& bridge,
                                            const py::dict& action_buffers,
                                            const py::dict& conflict_precompute,
                                            const py::dict& config)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->runDiffGuidedBatchLoopCompact(action_buffers, conflict_precompute, config);
}

py::dict run_one_net_buffer(const BridgePtr& bridge,
                            const std::string& net_name,
                            const py::dict& config)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->runOneNetBuffer(net_name, config);
}

py::dict run_coordinate_buffer_insert(const BridgePtr& bridge,
                                      const py::dict& action,
                                      const py::dict& config)
{
  if (bridge == nullptr) {
    throw std::runtime_error("placeio_openroad bridge is null");
  }
  return bridge->runCoordinateBufferInsert(action, config);
}

void write(const BridgePtr& bridge,
           const std::string& filename,
           int,
           const py::object& node_x,
           const py::object& node_y)
{
  apply(bridge, node_x, node_y);
  bridge->writeDef(filename);
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
  py::class_<OpenRoadPlaceIOBridge, BridgePtr>(m, "OpenRoadPlaceIOBridge")
      .def("lefUnit", &OpenRoadPlaceIOBridge::lefUnit)
      .def("defUnit", &OpenRoadPlaceIOBridge::defUnit)
      .def("setNodeOrient", &OpenRoadPlaceIOBridge::setNodeOrient)
      .def("export_pydb_view", &OpenRoadPlaceIOBridge::exportPyDBView)
      .def("sync_from_openroad", &OpenRoadPlaceIOBridge::syncFromOpenRoad)
      .def("sync_to_openroad", &OpenRoadPlaceIOBridge::syncToOpenRoad)
      .def("apply_sizing", &OpenRoadPlaceIOBridge::applySizing)
      .def("refresh_timing", &OpenRoadPlaceIOBridge::refreshTiming)
      .def("query_diff_guided_batch_timing_metrics",
           &OpenRoadPlaceIOBridge::queryDiffGuidedBatchTimingMetrics)
      .def("query_diff_guided_batch_timing_samples",
           &OpenRoadPlaceIOBridge::queryDiffGuidedBatchTimingSamples,
           py::arg("sample_request") = py::dict())
      .def("query_diff_guided_batch_dynamic_conflict_signature",
           &OpenRoadPlaceIOBridge::queryDiffGuidedBatchDynamicConflictSignature,
           py::arg("config") = py::dict())
      .def("evaluate_diff_guided_batch_actions",
           &OpenRoadPlaceIOBridge::evaluateDiffGuidedBatchActions,
           py::arg("actions"),
           py::arg("config") = py::dict())
      .def("apply_diff_guided_batch_transaction",
           &OpenRoadPlaceIOBridge::applyDiffGuidedBatchTransaction,
           py::arg("action_results"),
           py::arg("config") = py::dict())
      .def("run_diff_guided_batch_loop",
           &OpenRoadPlaceIOBridge::runDiffGuidedBatchLoop,
           py::arg("seed_queue"),
           py::arg("conflict_precompute") = py::dict(),
           py::arg("config") = py::dict())
      .def("run_diff_guided_batch_loop_compact",
           &OpenRoadPlaceIOBridge::runDiffGuidedBatchLoopCompact,
           py::arg("action_buffers"),
           py::arg("conflict_precompute") = py::dict(),
           py::arg("config") = py::dict())
      .def("eval_tcl_string", &OpenRoadPlaceIOBridge::evalTclString)
      .def("run_buffer_insertion", &OpenRoadPlaceIOBridge::runBufferInsertion)
      .def("run_one_net_buffer",
           &OpenRoadPlaceIOBridge::runOneNetBuffer,
           py::arg("net_name"),
           py::arg("config") = py::dict())
      .def("run_coordinate_buffer_insert",
           &OpenRoadPlaceIOBridge::runCoordinateBufferInsert,
           py::arg("action"),
           py::arg("config") = py::dict())
      .def("write_def", &OpenRoadPlaceIOBridge::writeDef);

  py::class_<PyPlaceDB>(m, "PyPlaceDB")
      .def(py::init<>())
      .def_readwrite("num_nodes", &PyPlaceDB::num_nodes)
      .def_readwrite("num_terminals", &PyPlaceDB::num_terminals)
      .def_readwrite("num_terminal_NIs", &PyPlaceDB::num_terminal_NIs)
      .def_readwrite("node_name2id_map", &PyPlaceDB::node_name2id_map)
      .def_readwrite("node_names", &PyPlaceDB::node_names)
      .def_readwrite("node_master_names", &PyPlaceDB::node_master_names)
      .def_readwrite("node_is_buffer", &PyPlaceDB::node_is_buffer)
      .def_readwrite("node_is_hard_macro", &PyPlaceDB::node_is_hard_macro)
      .def_readwrite("macro_writeback_candidate", &PyPlaceDB::macro_writeback_candidate)
      .def_readwrite("node_x", &PyPlaceDB::node_x)
      .def_readwrite("node_y", &PyPlaceDB::node_y)
      .def_readwrite("node_orient", &PyPlaceDB::node_orient)
      .def_readwrite("node_size_x", &PyPlaceDB::node_size_x)
      .def_readwrite("node_size_y", &PyPlaceDB::node_size_y)
      .def_readwrite("node2orig_node_map", &PyPlaceDB::node2orig_node_map)
      .def_readwrite("pin_direct", &PyPlaceDB::pin_direct)
      .def_readwrite("pin_offset_x", &PyPlaceDB::pin_offset_x)
      .def_readwrite("pin_offset_y", &PyPlaceDB::pin_offset_y)
      .def_readwrite("pin_names", &PyPlaceDB::pin_names)
      .def_readwrite("net_name2id_map", &PyPlaceDB::net_name2id_map)
      .def_readwrite("pin_name2id_map", &PyPlaceDB::pin_name2id_map)
      .def_readwrite("net_names", &PyPlaceDB::net_names)
      .def_readwrite("net2pin_map", &PyPlaceDB::net2pin_map)
      .def_readwrite("flat_net2pin_map", &PyPlaceDB::flat_net2pin_map)
      .def_readwrite("flat_net2pin_start_map", &PyPlaceDB::flat_net2pin_start_map)
      .def_readwrite("net_weights", &PyPlaceDB::net_weights)
      .def_readwrite("net_weight_deltas", &PyPlaceDB::net_weight_deltas)
      .def_readwrite("net_criticality", &PyPlaceDB::net_criticality)
      .def_readwrite("net_criticality_deltas", &PyPlaceDB::net_criticality_deltas)
      .def_readwrite("node2pin_map", &PyPlaceDB::node2pin_map)
      .def_readwrite("flat_node2pin_map", &PyPlaceDB::flat_node2pin_map)
      .def_readwrite("flat_node2pin_start_map", &PyPlaceDB::flat_node2pin_start_map)
      .def_readwrite("pin2node_map", &PyPlaceDB::pin2node_map)
      .def_readwrite("pin2net_map", &PyPlaceDB::pin2net_map)
      .def_readwrite("rows", &PyPlaceDB::rows)
      .def_readwrite("regions", &PyPlaceDB::regions)
      .def_readwrite("flat_region_boxes", &PyPlaceDB::flat_region_boxes)
      .def_readwrite("flat_region_boxes_start", &PyPlaceDB::flat_region_boxes_start)
      .def_readwrite("node2fence_region_map", &PyPlaceDB::node2fence_region_map)
      .def_readwrite("num_routing_grids_x", &PyPlaceDB::num_routing_grids_x)
      .def_readwrite("num_routing_grids_y", &PyPlaceDB::num_routing_grids_y)
      .def_readwrite("routing_grid_xl", &PyPlaceDB::routing_grid_xl)
      .def_readwrite("routing_grid_yl", &PyPlaceDB::routing_grid_yl)
      .def_readwrite("routing_grid_xh", &PyPlaceDB::routing_grid_xh)
      .def_readwrite("routing_grid_yh", &PyPlaceDB::routing_grid_yh)
      .def_readwrite("dbu", &PyPlaceDB::dbu)
      .def_readwrite("unit_horizontal_capacities", &PyPlaceDB::unit_horizontal_capacities)
      .def_readwrite("unit_vertical_capacities", &PyPlaceDB::unit_vertical_capacities)
      .def_readwrite("initial_horizontal_demand_map", &PyPlaceDB::initial_horizontal_demand_map)
      .def_readwrite("initial_vertical_demand_map", &PyPlaceDB::initial_vertical_demand_map)
      .def_readwrite("net2driver_pin_map", &PyPlaceDB::net2driver_pin_map)
      .def_readwrite("start_points", &PyPlaceDB::start_points)
      .def_readwrite("end_points", &PyPlaceDB::end_points)
      .def_readwrite("clock_pins", &PyPlaceDB::clock_pins)
      .def_readwrite("FF_ids", &PyPlaceDB::FF_ids)
      .def_readwrite("clk_pin_r_aat", &PyPlaceDB::clk_pin_r_aat)
      .def_readwrite("clk_pin_f_aat", &PyPlaceDB::clk_pin_f_aat)
      .def_readwrite("clk_pin_rtran", &PyPlaceDB::clk_pin_rtran)
      .def_readwrite("clk_pin_ftran", &PyPlaceDB::clk_pin_ftran)
      .def_readwrite("clk_pin_names", &PyPlaceDB::clk_pin_names)
      .def_readwrite("flat_cells_by_level", &PyPlaceDB::flat_cells_by_level)
      .def_readwrite("flat_cells_by_reverse_level", &PyPlaceDB::flat_cells_by_reverse_level)
      .def_readwrite("flat_cells_by_level_start", &PyPlaceDB::flat_cells_by_level_start)
      .def_readwrite("flat_cells_by_reverse_level_start", &PyPlaceDB::flat_cells_by_reverse_level_start)
      .def_readwrite("flat_inst_arcs_by_level", &PyPlaceDB::flat_inst_arcs_by_level)
      .def_readwrite("flat_inst_arcs_by_level_start", &PyPlaceDB::flat_inst_arcs_by_level_start)
      .def_readwrite("flat_pin_to_graph", &PyPlaceDB::flat_pin_to_graph)
      .def_readwrite("flat_pin_to_graph_start", &PyPlaceDB::flat_pin_to_graph_start)
      .def_readwrite("flat_pin_to_graph_reverse", &PyPlaceDB::flat_pin_to_graph_reverse)
      .def_readwrite("flat_pin_to_graph_start_reverse", &PyPlaceDB::flat_pin_to_graph_start_reverse)
      .def_readwrite("pin_pair_arc_keys", &PyPlaceDB::pin_pair_arc_keys)
      .def_readwrite("flat_pin_pair_arc_start", &PyPlaceDB::flat_pin_pair_arc_start)
      .def_readwrite("flat_pin_pair_arc_indices", &PyPlaceDB::flat_pin_pair_arc_indices)
      .def_readwrite("arc_level_start", &PyPlaceDB::arc_level_start)
      .def_readwrite("arc_src_pin", &PyPlaceDB::arc_src_pin)
      .def_readwrite("arc_dst_pin", &PyPlaceDB::arc_dst_pin)
      .def_readwrite("arc_inst_id", &PyPlaceDB::arc_inst_id)
      .def_readwrite("arc_libcell_id", &PyPlaceDB::arc_libcell_id)
      .def_readwrite("arc_libarc_id", &PyPlaceDB::arc_libarc_id)
      .def_readwrite("arc_sense", &PyPlaceDB::arc_sense)
      .def_readwrite("arc_type", &PyPlaceDB::arc_type)
      .def_readwrite("arc_offset", &PyPlaceDB::arc_offset)
      .def_readwrite("pin_pred_start", &PyPlaceDB::pin_pred_start)
      .def_readwrite("pin_pred_pin", &PyPlaceDB::pin_pred_pin)
      .def_readwrite("pin_pred_arc_id", &PyPlaceDB::pin_pred_arc_id)
      .def_readwrite("pin_succ_start", &PyPlaceDB::pin_succ_start)
      .def_readwrite("pin_succ_pin", &PyPlaceDB::pin_succ_pin)
      .def_readwrite("pin_succ_arc_id", &PyPlaceDB::pin_succ_arc_id)
      .def_readwrite("endpoint_pin_ids", &PyPlaceDB::endpoint_pin_ids)
      .def_readwrite("start_pin_ids", &PyPlaceDB::start_pin_ids)
      .def_readwrite("pin_to_inst_id", &PyPlaceDB::pin_to_inst_id)
      .def_readwrite("pin_to_node_id", &PyPlaceDB::pin_to_node_id)
      .def_readwrite("inst_topo_start", &PyPlaceDB::inst_topo_start)
      .def_readwrite("inst_topo_ids", &PyPlaceDB::inst_topo_ids)
      .def_readwrite("diff_guided_batch_endpoint_groups", &PyPlaceDB::diff_guided_batch_endpoint_groups)
      .def_readwrite("diff_guided_batch_top_path_groups", &PyPlaceDB::diff_guided_batch_top_path_groups)
      .def_readwrite("diff_guided_batch_net_neighborhood_groups", &PyPlaceDB::diff_guided_batch_net_neighborhood_groups)
      .def_readwrite("diff_guided_batch_fanin_fanout_groups", &PyPlaceDB::diff_guided_batch_fanin_fanout_groups)
      .def_readwrite("inrdelays", &PyPlaceDB::inrdelays)
      .def_readwrite("infdelays", &PyPlaceDB::infdelays)
      .def_readwrite("inrtrans", &PyPlaceDB::inrtrans)
      .def_readwrite("inftrans", &PyPlaceDB::inftrans)
      .def_readwrite("outcaps", &PyPlaceDB::outcaps)
      .def_readwrite("endpoints_rRAT", &PyPlaceDB::endpoints_rRAT)
      .def_readwrite("endpoints_fRAT", &PyPlaceDB::endpoints_fRAT)
      .def_readwrite("backend_endpoint_rAAT", &PyPlaceDB::backend_endpoint_rAAT)
      .def_readwrite("backend_endpoint_fAAT", &PyPlaceDB::backend_endpoint_fAAT)
      .def_readwrite("backend_endpoint_rRAT", &PyPlaceDB::backend_endpoint_rRAT)
      .def_readwrite("backend_endpoint_fRAT", &PyPlaceDB::backend_endpoint_fRAT)
      .def_readwrite("backend_endpoint_min_rAAT", &PyPlaceDB::backend_endpoint_min_rAAT)
      .def_readwrite("backend_endpoint_min_fAAT", &PyPlaceDB::backend_endpoint_min_fAAT)
      .def_readwrite("backend_endpoint_min_rRAT", &PyPlaceDB::backend_endpoint_min_rRAT)
      .def_readwrite("backend_endpoint_min_fRAT", &PyPlaceDB::backend_endpoint_min_fRAT)
      .def_readwrite("net_flat_arcs_start", &PyPlaceDB::net_flat_arcs_start)
      .def_readwrite("net_flat_arcs", &PyPlaceDB::net_flat_arcs)
      .def_readwrite("inst_flat_arcs_start", &PyPlaceDB::inst_flat_arcs_start)
      .def_readwrite("inst_flat_arcs", &PyPlaceDB::inst_flat_arcs)
      .def_readwrite("endpoints_constraint_arcs", &PyPlaceDB::endpoints_constraint_arcs)
      .def_readwrite("endpoints_timing_check_arcs", &PyPlaceDB::endpoints_timing_check_arcs)
      .def_readwrite("main_id_2_cell_id_start", &PyPlaceDB::main_id_2_cell_id_start)
      .def_readwrite("cell_id_2_arc_id_start", &PyPlaceDB::cell_id_2_arc_id_start)
      .def_readwrite("inst_main_id", &PyPlaceDB::inst_main_id)
      .def_readwrite("inst_libcell_offset", &PyPlaceDB::inst_libcell_offset)
      .def_readwrite("inst_size", &PyPlaceDB::inst_size)
      .def_readwrite("cell_id_2_libpin_id_start", &PyPlaceDB::cell_id_2_libpin_id_start)
      .def_readwrite("pin_2_libpin_offset", &PyPlaceDB::pin_2_libpin_offset)
      .def_readwrite("flat_lib_pin_offset_x", &PyPlaceDB::flat_lib_pin_offset_x)
      .def_readwrite("flat_lib_pin_offset_y", &PyPlaceDB::flat_lib_pin_offset_y)
      .def_readwrite("flat_lib_pin_cap", &PyPlaceDB::flat_lib_pin_cap)
      .def_readwrite("flat_lib_pin_rcap", &PyPlaceDB::flat_lib_pin_rcap)
      .def_readwrite("flat_lib_pin_fcap", &PyPlaceDB::flat_lib_pin_fcap)
      .def_readwrite("flat_lib_pin_cap_limit", &PyPlaceDB::flat_lib_pin_cap_limit)
      .def_readwrite("flat_lib_pin_slew_limit", &PyPlaceDB::flat_lib_pin_slew_limit)
      .def_readwrite("flat_libarc_info", &PyPlaceDB::flat_libarc_info)
      .def_readwrite("flat_libcell_names", &PyPlaceDB::flat_libcell_names)
      .def_readwrite("flat_libcell_info", &PyPlaceDB::flat_libcell_info)
      .def_readwrite("flat_libcell_width", &PyPlaceDB::flat_libcell_width)
      .def_readwrite("flat_libcell_height", &PyPlaceDB::flat_libcell_height)
      .def_readwrite("flat_libcell_leakage", &PyPlaceDB::flat_libcell_leakage)
      .def_readwrite("flat_libcell_main_id2size_vt_limit", &PyPlaceDB::flat_libcell_main_id2size_vt_limit)
      .def_readwrite("main_id_is_sizeable", &PyPlaceDB::main_id_is_sizeable)
      .def_readwrite("buffer_main_type_index", &PyPlaceDB::buffer_main_type_index)
      .def_readwrite("buffer_main_type_candidate_indices", &PyPlaceDB::buffer_main_type_candidate_indices)
      .def_readwrite("buffer_main_type_status", &PyPlaceDB::buffer_main_type_status)
      .def_readwrite("f_delay_flat_luts_values", &PyPlaceDB::f_delay_flat_luts_values)
      .def_readwrite("f_delay_flat_luts_trans_table", &PyPlaceDB::f_delay_flat_luts_trans_table)
      .def_readwrite("f_delay_flat_luts_cap_table", &PyPlaceDB::f_delay_flat_luts_cap_table)
      .def_readwrite("f_delay_flat_luts_dim", &PyPlaceDB::f_delay_flat_luts_dim)
      .def_readwrite("r_delay_flat_luts_values", &PyPlaceDB::r_delay_flat_luts_values)
      .def_readwrite("r_delay_flat_luts_trans_table", &PyPlaceDB::r_delay_flat_luts_trans_table)
      .def_readwrite("r_delay_flat_luts_cap_table", &PyPlaceDB::r_delay_flat_luts_cap_table)
      .def_readwrite("r_delay_flat_luts_dim", &PyPlaceDB::r_delay_flat_luts_dim)
      .def_readwrite("f_trans_flat_luts_values", &PyPlaceDB::f_trans_flat_luts_values)
      .def_readwrite("f_trans_flat_luts_trans_table", &PyPlaceDB::f_trans_flat_luts_trans_table)
      .def_readwrite("f_trans_flat_luts_cap_table", &PyPlaceDB::f_trans_flat_luts_cap_table)
      .def_readwrite("f_trans_flat_luts_dim", &PyPlaceDB::f_trans_flat_luts_dim)
      .def_readwrite("r_trans_flat_luts_values", &PyPlaceDB::r_trans_flat_luts_values)
      .def_readwrite("r_trans_flat_luts_trans_table", &PyPlaceDB::r_trans_flat_luts_trans_table)
      .def_readwrite("r_trans_flat_luts_cap_table", &PyPlaceDB::r_trans_flat_luts_cap_table)
      .def_readwrite("r_trans_flat_luts_dim", &PyPlaceDB::r_trans_flat_luts_dim)
      .def_readwrite("export_profile", &PyPlaceDB::export_profile)
      .def_readwrite("c_unit", &PyPlaceDB::c_unit)
      .def_readwrite("r_unit", &PyPlaceDB::r_unit)
      .def_readwrite("xl", &PyPlaceDB::xl)
      .def_readwrite("yl", &PyPlaceDB::yl)
      .def_readwrite("xh", &PyPlaceDB::xh)
      .def_readwrite("yh", &PyPlaceDB::yh)
      .def_readwrite("row_height", &PyPlaceDB::row_height)
      .def_readwrite("site_width", &PyPlaceDB::site_width)
      .def_readwrite("total_space_area", &PyPlaceDB::total_space_area)
      .def_readwrite("total_fixed_node_area", &PyPlaceDB::total_fixed_node_area)
      .def_readwrite("num_movable_pins", &PyPlaceDB::num_movable_pins);

  m.def("forward", &forward, "Read LEF/DEF through OpenROAD in-process");
  m.def("pydb", &pydb, "Build a Python placement DB from the OpenROAD bridge");
  m.def("apply", &apply, "Apply node coordinates to the OpenROAD bridge");
  m.def("apply_sizing", &apply_sizing, "Apply sizing masters to the OpenROAD bridge");
  m.def("query_diff_guided_batch_timing_metrics",
        &query_diff_guided_batch_timing_metrics,
        "Query in-process OpenSTA metrics for diff-guided batch sizing");
  m.def("query_diff_guided_batch_timing_samples",
        &query_diff_guided_batch_timing_samples,
        py::arg("bridge"),
        py::arg("sample_request") = py::dict(),
        "Query sampled in-process OpenSTA endpoint/pin timing values for parity debugging");
  m.def("query_diff_guided_batch_dynamic_conflict_signature",
        &query_diff_guided_batch_dynamic_conflict_signature,
        py::arg("bridge"),
        py::arg("config") = py::dict(),
        "Query OpenSTA top-path dynamic conflict signature for diff-guided batch selection");
  m.def("evaluate_diff_guided_batch_actions",
        &evaluate_diff_guided_batch_actions,
        py::arg("bridge"),
        py::arg("actions"),
        py::arg("config") = py::dict(),
        "Evaluate diff-guided batch actions with in-process OpenSTA trial resize and rollback");
  m.def("apply_diff_guided_batch_transaction",
        &apply_diff_guided_batch_transaction,
        py::arg("bridge"),
        py::arg("action_results"),
        py::arg("config") = py::dict(),
        "Apply accepted diff-guided batch actions to the in-process OpenSTA/OpenROAD state");
  m.def("run_diff_guided_batch_loop",
        &run_diff_guided_batch_loop,
        py::arg("bridge"),
        py::arg("seed_queue"),
        py::arg("conflict_precompute") = py::dict(),
        py::arg("config") = py::dict(),
        "Run the diff-guided 3-5 batch loop inside the in-process OpenSTA/OpenROAD bridge");
  m.def("run_diff_guided_batch_loop_compact",
        &run_diff_guided_batch_loop_compact,
        py::arg("bridge"),
        py::arg("action_buffers"),
        py::arg("conflict_precompute") = py::dict(),
        py::arg("config") = py::dict(),
        "Run the diff-guided 3-5 batch loop from compact action proposal buffers");
  m.def("run_one_net_buffer",
        &run_one_net_buffer,
        py::arg("bridge"),
        py::arg("net_name"),
        py::arg("config") = py::dict(),
        "Run one-net buffer insertion through the in-process OpenROAD bridge");
  m.def("run_coordinate_buffer_insert",
        &run_coordinate_buffer_insert,
        py::arg("bridge"),
        py::arg("action"),
        py::arg("config") = py::dict(),
        "Run coordinate-specific buffer insertion through the in-process OpenROAD bridge");
  m.def(
      "write",
      [](const BridgePtr& bridge,
         const std::string& filename,
         int nodemap_input_flag,
         const py::object& node_x,
         const py::object& node_y) { write(bridge, filename, nodemap_input_flag, node_x, node_y); },
      "Write a DEF after applying node coordinates");
  m.attr("OpenRoadRawDB") = m.attr("OpenRoadPlaceIOBridge");
}
