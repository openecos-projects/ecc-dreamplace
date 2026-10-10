#include "timing_propagation_paths.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
// #include <torch/extension.h>

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "Timing propagation helper bindings";

  py::class_<dreamplace::TraversalPruningRefreshResult>(
      m, "TraversalPruningRefreshResult")
      .def_readonly("kept_flat_arc_indices",
                    &dreamplace::TraversalPruningRefreshResult::kept_flat_arc_indices)
      .def_readonly("kept_level_offsets",
                    &dreamplace::TraversalPruningRefreshResult::kept_level_offsets)
      .def_readonly("kept_counts_by_level",
                    &dreamplace::TraversalPruningRefreshResult::kept_counts_by_level)
      .def_readonly("active_inst_count",
                    &dreamplace::TraversalPruningRefreshResult::active_inst_count)
      .def_readonly("active_arc_count",
                    &dreamplace::TraversalPruningRefreshResult::active_arc_count)
      .def_readonly("dropped_arc_count",
                    &dreamplace::TraversalPruningRefreshResult::dropped_arc_count)
      .def_readonly("parallel_task_count",
                    &dreamplace::TraversalPruningRefreshResult::parallel_task_count)
      .def_readonly("preparation_runtime_ms",
                    &dreamplace::TraversalPruningRefreshResult::preparation_runtime_ms);

  py::class_<dreamplace::EndpointIncidenceResult>(
      m, "EndpointIncidenceResult")
      .def_readonly("pin_endpoint_incidence_count",
                    &dreamplace::EndpointIncidenceResult::pin_endpoint_incidence_count)
      .def_readonly("arc_endpoint_incidence_count",
                    &dreamplace::EndpointIncidenceResult::arc_endpoint_incidence_count)
      .def_readonly("active_endpoint_count",
                    &dreamplace::EndpointIncidenceResult::active_endpoint_count)
      .def_readonly("pin_incidence_nonzero_count",
                    &dreamplace::EndpointIncidenceResult::pin_incidence_nonzero_count)
      .def_readonly("arc_incidence_nonzero_count",
                    &dreamplace::EndpointIncidenceResult::arc_incidence_nonzero_count)
      .def_readonly("incidence_runtime_ms",
                    &dreamplace::EndpointIncidenceResult::incidence_runtime_ms);

  py::class_<dreamplace::CriticalPathBatch>(m, "CriticalPathBatch")
      .def_readonly("path_offsets", &dreamplace::CriticalPathBatch::path_offsets)
      .def_readonly("path_pins", &dreamplace::CriticalPathBatch::path_pins)
      .def_readonly("path_transitions", &dreamplace::CriticalPathBatch::path_transitions)
      .def_readonly("path_arc_ids", &dreamplace::CriticalPathBatch::path_arc_ids)
      .def_readonly("endpoint_pins", &dreamplace::CriticalPathBatch::endpoint_pins)
      .def_readonly("endpoint_test_ids", &dreamplace::CriticalPathBatch::endpoint_test_ids)
      .def_readonly("endpoint_transitions", &dreamplace::CriticalPathBatch::endpoint_transitions)
      .def_readonly("endpoint_slacks", &dreamplace::CriticalPathBatch::endpoint_slacks)
      .def_readonly("path_valid", &dreamplace::CriticalPathBatch::path_valid)
      .def_readonly("invalid_reason", &dreamplace::CriticalPathBatch::invalid_reason)
      .def_readonly("max_residual_ps", &dreamplace::CriticalPathBatch::max_residual_ps)
      .def_readonly("failing_state_count", &dreamplace::CriticalPathBatch::failing_state_count)
      .def_readonly("selected_state_count", &dreamplace::CriticalPathBatch::selected_state_count)
      .def_readonly("valid_path_count", &dreamplace::CriticalPathBatch::valid_path_count)
      .def_readonly("invalid_path_count", &dreamplace::CriticalPathBatch::invalid_path_count)
      .def_readonly("topology_epoch", &dreamplace::CriticalPathBatch::topology_epoch)
      .def_readonly("extraction_runtime_ms", &dreamplace::CriticalPathBatch::extraction_runtime_ms);

  py::class_<dreamplace::SetupCriticalPathExtractor>(
      m, "SetupCriticalPathExtractor")
      .def(py::init<
           const at::Tensor&,
           const at::Tensor&,
           const at::Tensor&,
           const at::Tensor&,
           const at::Tensor&,
           int64_t>(),
           py::arg("flat_inst_arcs_by_level"),
           py::arg("pin_pred_start"),
           py::arg("pin_pred_pin"),
           py::arg("pin_pred_arc_id"),
           py::arg("start_points"),
           py::arg("topology_epoch"))
      .def("extract",
           &dreamplace::SetupCriticalPathExtractor::extract,
           py::arg("endpoint_pins"),
           py::arg("endpoint_test_ids"),
           py::arg("endpoint_rise_slack"),
           py::arg("endpoint_fall_slack"),
           py::arg("pin_rise_aat"),
           py::arg("pin_fall_aat"),
           py::arg("pin_net_delay_rise"),
           py::arg("pin_net_delay_fall"),
           py::arg("cell_delay_rr"),
           py::arg("cell_delay_fr"),
           py::arg("cell_delay_rf"),
           py::arg("cell_delay_ff"),
           py::arg("global_k") = static_cast<int64_t>(0),
           py::arg("max_depth") = static_cast<int64_t>(0),
           py::arg("residual_tolerance_ps") = 1e-3)
      .def("topology_epoch", &dreamplace::SetupCriticalPathExtractor::topology_epoch)
      .def("num_pins", &dreamplace::SetupCriticalPathExtractor::num_pins)
      .def("num_cell_arcs", &dreamplace::SetupCriticalPathExtractor::num_cell_arcs);

  py::class_<dreamplace::CriticalEndpointTraversalPruner>(
      m, "CriticalEndpointTraversalPruner")
      .def(py::init<
           const at::Tensor&,
           const at::Tensor&,
           const at::Tensor&,
           const at::Tensor&,
           const at::Tensor&,
           const at::Tensor&,
           const at::Tensor&,
           const at::Tensor&,
           const at::Tensor&>(),
           py::arg("flat_inst_arcs_by_level"),
           py::arg("flat_inst_arcs_by_level_start"),
           py::arg("flat_pin_to_graph_reverse"),
           py::arg("flat_pin_to_graph_start_reverse"),
           py::arg("pin_pair_arc_keys"),
           py::arg("flat_pin_pair_arc_start"),
           py::arg("flat_pin_pair_arc_indices"),
           py::arg("start_points"),
           py::arg("pin2node_map"))
      .def(py::init<
           const at::Tensor&,
           const at::Tensor&,
           const at::Tensor&,
           const at::Tensor&,
           const at::Tensor&,
           const at::Tensor&,
           const at::Tensor&>(),
           py::arg("flat_inst_arcs_by_level"),
           py::arg("flat_inst_arcs_by_level_start"),
           py::arg("pin_pred_start"),
           py::arg("pin_pred_pin"),
           py::arg("pin_pred_arc_id"),
           py::arg("start_points"),
           py::arg("pin2node_map"))
      .def("refresh",
           &dreamplace::CriticalEndpointTraversalPruner::refresh,
           py::arg("active_endpoint_ids"))
      .def("endpoint_incidence",
           &dreamplace::CriticalEndpointTraversalPruner::endpoint_incidence,
           py::arg("active_endpoint_ids"))
      .def("num_flat_arcs",
           &dreamplace::CriticalEndpointTraversalPruner::num_flat_arcs)
      .def("num_levels",
           &dreamplace::CriticalEndpointTraversalPruner::num_levels);

  m.def(
      "extract_critical_paths",
      &dreamplace::extract_critical_paths,
      py::arg("endpoints"),
      py::arg("k"),
      py::arg("start_points"),
      py::arg("pin_rAAT"),
      py::arg("pin_fAAT"),
      py::arg("pin_rslack"),
      py::arg("pin_fslack"),
      py::arg("pin_slack"),
      py::arg("flat_pin_to_graph_reverse"),
      py::arg("flat_pin_to_graph_start_reverse"),
      py::arg("pin_pair_arc_keys") ,
      py::arg("flat_pin_pair_arc_start"),
      py::arg("flat_pin_pair_arc_indices"),
      py::arg("max_depth") = static_cast<int64_t>(0),
      py::arg("slack_epsilon") = 1e-3,
      "Extract a greedy critical path for each endpoint." );
}
