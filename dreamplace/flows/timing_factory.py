"""Build timing operators from the shared placement graph and current state."""

import torch

from dreamplace.ops.timing_propagation.crash_stage_marker import write_crash_stage_marker
from dreamplace.ops.timing_propagation.timing_propagation import TimingPropagation


def build_timing_propagation_op(params, placedb, data_collections, cell_modeling_op):
    write_crash_stage_marker(
        "timing_propagation_op_construction",
        "start",
        timing_propagation_device=getattr(params, "timing_propagation_device", "inherit"),
        params_gpu=getattr(params, "gpu", 0),
        params_gpu_id=getattr(params, "gpu_id", 0),
    )
    arguments = dict(
        inrdelays=data_collections.inrdelays,
        infdelays=data_collections.infdelays,
        inrtrans=data_collections.inrtrans,
        inftrans=data_collections.inftrans,
        outcaps=data_collections.outcaps,
        pin_net=data_collections.pin2net_map,
        start_points=data_collections.start_points,
        end_points=data_collections.end_points,
        clock_pins=data_collections.clock_pins,
        FF_ids=data_collections.FF_ids,
        clk_pin_rtran=data_collections.clk_pin_rtran,
        clk_pin_ftran=data_collections.clk_pin_ftran,
        net_flat_arcs_start=data_collections.net_flat_arcs_start,
        net_flat_arcs=data_collections.net_flat_arcs,
        net2driver_pin_map=data_collections.net2driver_pin_map,
        arcs_info=data_collections.arcs_info,
        inst_flat_arcs_start=data_collections.inst_flat_arcs_start,
        inst_flat_arcs=data_collections.inst_flat_arcs,
        endpoints_constraint_arcs=data_collections.endpoints_constraint_arcs,
        flat_inst_arcs_by_level=data_collections.flat_inst_arcs_by_level,
        flat_inst_arcs_by_level_start=data_collections.flat_inst_arcs_by_level_start,
        endpoints_rRAT=placedb.endpoints_rRAT,
        endpoints_fRAT=placedb.endpoints_fRAT,
        flat_pin_to_graph=data_collections.flat_pin_to_graph,
        flat_pin_to_graph_start=data_collections.flat_pin_to_graph_start,
        flat_pin_to_graph_reverse=data_collections.flat_pin_to_graph_reverse,
        flat_pin_to_graph_start_reverse=data_collections.flat_pin_to_graph_start_reverse,
        endpoints_timing_check_arcs=getattr(data_collections, "endpoints_timing_check_arcs", None),
        sparse_timing_view=getattr(placedb, "sparse_timing_view", None),
        timing_model_lease=getattr(data_collections, "timing_model_lease", None),
        hold_timing_data=getattr(placedb, "hold_timing_data", None),
        endpoints_max_valid=getattr(data_collections, "endpoints_max_valid", None),
        endpoints_constraint_max_valid=getattr(
            data_collections, "endpoints_constraint_max_valid", None
        ),
        endpoints_timing_check_max_valid=getattr(
            data_collections, "endpoints_timing_check_max_valid", None
        ),
        backend_endpoint_rSlew=torch.from_numpy(placedb.backend_endpoint_rSlew).to(
            data_collections.end_points.device
        )
        if getattr(placedb, "backend_endpoint_rSlew", None) is not None
        and len(placedb.backend_endpoint_rSlew) > 0
        else None,
        backend_endpoint_fSlew=torch.from_numpy(placedb.backend_endpoint_fSlew).to(
            data_collections.end_points.device
        )
        if getattr(placedb, "backend_endpoint_fSlew", None) is not None
        and len(placedb.backend_endpoint_fSlew) > 0
        else None,
        pin_pair_arc_keys=data_collections.pin_pair_arc_keys,
        flat_pin_pair_arc_start=data_collections.flat_pin_pair_arc_start,
        flat_pin_pair_arc_indices=data_collections.flat_pin_pair_arc_indices,
        pin_pred_start=getattr(data_collections, "pin_pred_start", None),
        pin_pred_pin=getattr(data_collections, "pin_pred_pin", None),
        pin_pred_arc_id=getattr(data_collections, "pin_pred_arc_id", None),
        pin_succ_start=getattr(data_collections, "pin_succ_start", None),
        pin_succ_pin=getattr(data_collections, "pin_succ_pin", None),
        pin_succ_arc_id=getattr(data_collections, "pin_succ_arc_id", None),
        arc_level_start=getattr(data_collections, "arc_level_start", None),
        arc_src_pin=getattr(data_collections, "arc_src_pin", None),
        arc_dst_pin=getattr(data_collections, "arc_dst_pin", None),
        arc_inst_id=getattr(data_collections, "arc_inst_id", None),
        endpoint_pin_ids=getattr(data_collections, "endpoint_pin_ids", None),
        start_pin_ids=getattr(data_collections, "start_pin_ids", None),
        pin_to_inst_id=getattr(data_collections, "pin_to_inst_id", None),
        pin_to_node_id=getattr(data_collections, "pin_to_node_id", None),
        inst_topo_start=getattr(data_collections, "inst_topo_start", None),
        inst_topo_ids=getattr(data_collections, "inst_topo_ids", None),
        cell_modeling_op=cell_modeling_op,
        lib_arc_offsets=data_collections.flat_libarc_info[:, 3].long()
        if getattr(data_collections, "flat_libarc_info", None) is not None
        else None,
        pin2node_map=data_collections.pin2node_map,
        inst_main_id=data_collections.inst_main_id,
        inst_is_sizeable=getattr(data_collections, "inst_is_sizeable", None),
        inst_size_init=getattr(data_collections, "inst_size_init", None),
        inst_vt_init=getattr(data_collections, "inst_vt_init", None),
        size_var_getter=getattr(data_collections, "get_size_var", None),
        vt_var_getter=getattr(data_collections, "get_vt_var", None),
        critical_endpoint_pruning_mode=getattr(params, "critical_endpoint_pruning_mode", "off"),
        critical_endpoint_top_k=getattr(params, "critical_endpoint_top_k", 256),
        critical_endpoint_slack_window_ps=getattr(
            params, "critical_endpoint_slack_window_ps", 100.0
        ),
        critical_endpoint_refresh_interval=getattr(
            params, "critical_endpoint_refresh_interval", 10
        ),
        critical_endpoint_hysteresis_interval=getattr(
            params, "critical_endpoint_hysteresis_interval", 3
        ),
        critical_endpoint_full_refresh_interval=getattr(
            params, "critical_endpoint_full_refresh_interval", 50
        ),
        timing_propagation_profile=getattr(params, "timing_propagation_profile", False),
        timing_propagation_device=getattr(params, "timing_propagation_device", "inherit"),
        timing_propagation_global_gpu=getattr(params, "gpu", 0),
        timing_propagation_global_gpu_id=getattr(params, "gpu_id", 0),
        timing_propagation_parity_check=getattr(params, "timing_propagation_parity_check", False),
        timing_propagation_parity_atol_ps=getattr(
            params, "timing_propagation_parity_atol_ps", 1e-3
        ),
        timing_propagation_parity_rtol=getattr(params, "timing_propagation_parity_rtol", 1e-5),
        timing_aggregation_mode=getattr(params, "timing_aggregation_mode", "smooth"),
        timing_aggregation_tau_ps=getattr(params, "timing_aggregation_tau_ps", 2.0),
        timing_metric_scope=getattr(params, "timing_metric_scope", "setup_plus_recovery"),
        production_fast_loop=getattr(params, "production_fast_loop", False),
        timing_lut_2d_native_op=getattr(params, "timing_lut_2d_native_op", "auto"),
    )
    timing_propagation_op = TimingPropagation(**arguments)
    if placedb.hold_timing_data is not None:
        from dreamplace.flows.hold_rc import HoldRC

        timing_propagation_op.hold_rc = HoldRC(placedb.hold_timing_data)
    if placedb.hold_timing_corner is not None:
        from dreamplace.flows.timing_corner_runtime import build_early_timing_view

        timing_propagation_op.early_timing_view = build_early_timing_view(
            arguments,
            data_collections,
            placedb.hold_timing_corner,
        )
        timing_propagation_op.hold_liberty_paths = placedb.hold_timing_corner.liberty_paths
    write_crash_stage_marker(
        "timing_propagation_op_construction",
        "done",
        resolved_device=str(getattr(timing_propagation_op, "device", "")),
    )
    return timing_propagation_op
