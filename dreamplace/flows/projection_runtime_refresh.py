"""Apply projected sizing/VT state to runtime geometry and timing data."""

import logging
import time
from dataclasses import dataclass, replace
from typing import Callable

import torch

from dreamplace.flows.metric_state import timing_scalar
from dreamplace.flows.projection_artifacts import projected_vt_distribution


def apply_projected_runtime_refresh(engine, params, placedb):
    projection_frame = getattr(engine, "last_projection_frame", None)
    projection_result = getattr(engine, "last_projection_result", None)
    if projection_frame is not None:
        if projection_result is not None and projection_frame.projection_result is not projection_result:
            projection_frame = None
        else:
            projection_result = projection_frame.projection_result
    if projection_result is None:
        summary = {
            "runtime_refresh_applied": False,
            "num_candidate_instances": 0,
            "num_changed_instances": 0,
            "num_runtime_updated_instances": 0,
            "num_runtime_updated_node_geometries": 0,
            "num_runtime_updated_pins": 0,
            "runtime_geometry_refresh_applied": False,
            "runtime_pin_offset_consistency_applied": False,
            "runtime_topology_rebuilt": False,
            "metadata": {
                "artifact_scope": "post_projection_runtime",
                "placement_sizing_mode": engine._placement_sizing_mode(params),
            },
        }
        engine._write_projection_runtime_refresh_summary(params, summary)
        pin_summary = engine._write_projection_pin_offset_runtime_summary(
            params,
            runtime_pin_offset_consistency_applied=False,
            num_runtime_updated_pins=0,
            num_runtime_updated_instances=0,
        )
        summary["pin_offset_summary_path"] = getattr(
            engine,
            "last_projection_pin_offset_runtime_summary_path",
            None,
        )
        engine.last_projection_runtime_refresh_summary = summary
        engine.last_projection_runtime_refresh_summary_path = getattr(
            engine,
            "last_projection_runtime_refresh_summary_path",
            None,
        )
        engine.last_projection_pin_offset_runtime_summary = pin_summary
        return summary

    continuous_size_sync = engine._sync_projected_real_size(projection_result)

    changed_mask = (
        projection_frame.changed_candidate_mask
        if projection_frame is not None
        else engine._projection_changed_inst_mask(projection_result)
    )
    if changed_mask is None:
        summary = {
            "runtime_refresh_applied": False,
            "num_candidate_instances": int(
                getattr(projection_result, "inst_ids", torch.empty(0)).numel()
            ),
            "num_changed_instances": 0,
            "num_runtime_updated_instances": 0,
            "num_runtime_updated_node_geometries": 0,
            "num_runtime_updated_pins": 0,
            "runtime_geometry_refresh_applied": False,
            "runtime_pin_offset_consistency_applied": False,
            "runtime_topology_rebuilt": False,
            "continuous_size_sync": continuous_size_sync,
            "metadata": {
                "artifact_scope": "post_projection_runtime",
                "placement_sizing_mode": engine._placement_sizing_mode(params),
            },
        }
        engine._write_projection_runtime_refresh_summary(params, summary)
        pin_summary = engine._write_projection_pin_offset_runtime_summary(
            params,
            runtime_pin_offset_consistency_applied=False,
            num_runtime_updated_pins=0,
            num_runtime_updated_instances=0,
        )
        summary["pin_offset_summary_path"] = getattr(
            engine,
            "last_projection_pin_offset_runtime_summary_path",
            None,
        )
        engine.last_projection_runtime_refresh_summary = summary
        engine.last_projection_pin_offset_runtime_summary = pin_summary
        return summary

    changed_inst_ids = (
        projection_frame.changed_inst_ids
        if projection_frame is not None
        else projection_result.inst_ids[changed_mask].long()
    )
    summary = {
        "runtime_refresh_applied": False,
        "num_candidate_instances": int(projection_result.inst_ids.numel()),
        "num_changed_instances": int(changed_inst_ids.numel()),
        "num_runtime_updated_instances": 0,
        "num_runtime_updated_node_geometries": 0,
        "num_runtime_updated_pins": 0,
        "runtime_geometry_refresh_applied": False,
        "runtime_pin_offset_consistency_applied": False,
        "runtime_topology_rebuilt": False,
        "continuous_size_sync": continuous_size_sync,
        "metadata": {
            "artifact_scope": "post_projection_runtime",
            "placement_sizing_mode": engine._placement_sizing_mode(params),
        },
    }

    if changed_inst_ids.numel() == 0:
        pin_summary = engine._write_projection_pin_offset_runtime_summary(
            params,
            runtime_pin_offset_consistency_applied=False,
            num_runtime_updated_pins=0,
            num_runtime_updated_instances=0,
        )
        summary["pin_offset_summary_path"] = getattr(
            engine,
            "last_projection_pin_offset_runtime_summary_path",
            None,
        )
        engine.last_projection_pin_offset_runtime_summary = pin_summary
        engine._write_projection_runtime_refresh_summary(params, summary)
        return summary

    defer_placedb_sync = engine._defer_real_size_transition_placedb_sync()
    changed_inst_ids_cpu = (
        None if defer_placedb_sync else changed_inst_ids.detach().cpu().numpy()
    )
    if projection_frame is not None:
        projected_cell_ids = projection_frame.projected_cell_id_global[
            changed_inst_ids.to(projection_frame.projected_cell_id_global.device)
        ].long()
        projected_offsets = projection_frame.projected_libcell_offset_global[
            changed_inst_ids.to(
                projection_frame.projected_libcell_offset_global.device
            )
        ].long()
    else:
        projected_cell_ids = projection_result.projected_cell_id[changed_mask].long()
        projected_offsets = projection_result.projected_libcell_offset[changed_mask].long()

    runtime_cell_apply_started_at = time.perf_counter()
    with torch.no_grad():
        if getattr(engine.data_collections, "inst_cell_id", None) is not None:
            changed_inst_ids_device = changed_inst_ids.to(
                engine.data_collections.inst_cell_id.device
            )
            projected_cell_ids_device = projected_cell_ids.to(
                engine.data_collections.inst_cell_id.device,
                dtype=engine.data_collections.inst_cell_id.dtype,
            )
            engine.data_collections.inst_cell_id[changed_inst_ids_device] = projected_cell_ids_device
        if getattr(engine.data_collections, "inst_libcell_offset", None) is not None:
            changed_inst_ids_device = changed_inst_ids.to(
                engine.data_collections.inst_libcell_offset.device
            )
            projected_offsets_device = projected_offsets.to(
                engine.data_collections.inst_libcell_offset.device,
                dtype=engine.data_collections.inst_libcell_offset.dtype,
            )
            engine.data_collections.inst_libcell_offset[changed_inst_ids_device] = projected_offsets_device
        if (
            projection_frame is not None
            and projection_frame.projected_vt_global is not None
            and getattr(engine.data_collections, "vt_logits", None) is not None
        ):
            vt_logits = engine.data_collections.vt_logits
            vt_ids = changed_inst_ids.to(device=vt_logits.device)
            projected_vt_distribution = projection_frame.projected_vt_global[
                changed_inst_ids.to(
                    projection_frame.projected_vt_global.device
                )
            ].to(device=vt_logits.device, dtype=vt_logits.dtype)
            projected_vt_logits = torch.log(
                projected_vt_distribution.clamp(min=1e-6)
            )
            vt_logits[vt_ids] = projected_vt_logits
            summary_vt_updated = int(vt_ids.numel())
        else:
            summary_vt_updated = 0

    if not defer_placedb_sync and getattr(placedb, "inst_cell_id", None) is not None:
        placedb.inst_cell_id[changed_inst_ids_cpu] = projected_cell_ids.detach().cpu().numpy()
    if not defer_placedb_sync and getattr(placedb, "inst_libcell_offset", None) is not None:
        placedb.inst_libcell_offset[changed_inst_ids_cpu] = projected_offsets.detach().cpu().numpy()
    if not defer_placedb_sync and getattr(placedb, "inst_cell_id", None) is not None and getattr(placedb, "flat_libcell_names", None) is not None:
        preview = []
        for idx in range(min(10, len(placedb.inst_cell_id))):
            cell_id = int(placedb.inst_cell_id[idx])
            if 0 <= cell_id < len(placedb.flat_libcell_names):
                preview.append((cell_id, str(placedb.flat_libcell_names[cell_id])))
        logging.info(
            "projection runtime refresh preview placedb_id=%s first_cells=%s",
            id(placedb),
            preview,
        )

    flat_libcell_width = getattr(engine.data_collections, "flat_libcell_width", None)
    flat_libcell_height = getattr(engine.data_collections, "flat_libcell_height", None)
    if (
        projection_frame is not None
        and projection_frame.projected_node_size_x is not None
        and projection_frame.projected_node_size_y is not None
    ):
        frame_ids = changed_inst_ids.to(
            projection_frame.projected_node_size_x.device
        )
        projected_width = projection_frame.projected_node_size_x[frame_ids]
        projected_height = projection_frame.projected_node_size_y[
            changed_inst_ids.to(projection_frame.projected_node_size_y.device)
        ]

        with torch.no_grad():
            changed_inst_ids_device = changed_inst_ids.to(
                engine.data_collections.node_size_x.device
            )
            engine.data_collections.node_size_x[changed_inst_ids_device] = projected_width.to(
                device=engine.data_collections.node_size_x.device,
                dtype=engine.data_collections.node_size_x.dtype,
            )
            engine.data_collections.node_size_y[changed_inst_ids_device] = projected_height.to(
                device=engine.data_collections.node_size_y.device,
                dtype=engine.data_collections.node_size_y.dtype,
            )
            if getattr(engine.data_collections, "original_node_size_x", None) is not None:
                engine.data_collections.original_node_size_x[changed_inst_ids_device] = projected_width.to(
                    device=engine.data_collections.original_node_size_x.device,
                    dtype=engine.data_collections.original_node_size_x.dtype,
                )
                engine.data_collections.original_node_size_y[changed_inst_ids_device] = projected_height.to(
                    device=engine.data_collections.original_node_size_y.device,
                    dtype=engine.data_collections.original_node_size_y.dtype,
                )
            if getattr(engine.data_collections, "node_areas", None) is not None:
                engine.data_collections.node_areas[changed_inst_ids_device] = (
                    projected_width.to(
                        device=engine.data_collections.node_areas.device,
                        dtype=engine.data_collections.node_areas.dtype,
                    )
                    * projected_height.to(
                        device=engine.data_collections.node_areas.device,
                        dtype=engine.data_collections.node_areas.dtype,
                    )
                )
        summary["runtime_geometry_refresh_applied"] = True
        summary["num_runtime_updated_node_geometries"] = int(changed_inst_ids.numel())
        if projection_frame.projected_inst_size_init is not None:
            with torch.no_grad():
                init_ids = changed_inst_ids.to(
                    projection_frame.projected_inst_size_init.device
                )
                engine.data_collections.inst_size_init[init_ids] = (
                    projection_frame.projected_inst_size_init[init_ids].to(
                        device=engine.data_collections.inst_size_init.device,
                        dtype=engine.data_collections.inst_size_init.dtype,
                    )
                )
        if (
            not defer_placedb_sync
            and placedb is not None
            and getattr(placedb, "node_size_x", None) is not None
        ):
            changed_cpu = changed_inst_ids.detach().cpu().numpy()
            placedb.node_size_x[changed_cpu] = projected_width.detach().cpu().numpy()
            placedb.node_size_y[changed_cpu] = projected_height.detach().cpu().numpy()
            if getattr(placedb, "inst_size_init", None) is not None:
                init_values = projection_frame.projected_inst_size_init[
                    changed_inst_ids.to(
                        projection_frame.projected_inst_size_init.device
                    )
                ]
                placedb.inst_size_init[changed_cpu] = init_values.detach().cpu().numpy()
    elif flat_libcell_width is not None and flat_libcell_height is not None:
        projected_width = flat_libcell_width[projected_cell_ids].to(
            engine.data_collections.node_size_x.device,
            dtype=engine.data_collections.node_size_x.dtype,
        )
        projected_height = flat_libcell_height[projected_cell_ids].to(
            engine.data_collections.node_size_y.device,
            dtype=engine.data_collections.node_size_y.dtype,
        )

        with torch.no_grad():
            changed_inst_ids_device = changed_inst_ids.to(
                engine.data_collections.node_size_x.device
            )
            engine.data_collections.node_size_x[changed_inst_ids_device] = projected_width
            engine.data_collections.node_size_y[changed_inst_ids_device] = projected_height
            if getattr(engine.data_collections, "original_node_size_x", None) is not None:
                original_changed_inst_ids_device = changed_inst_ids.to(
                    engine.data_collections.original_node_size_x.device
                )
                engine.data_collections.original_node_size_x[original_changed_inst_ids_device] = projected_width
                engine.data_collections.original_node_size_y[original_changed_inst_ids_device] = projected_height
            if getattr(engine.data_collections, "node_areas", None) is not None:
                projected_area = projected_width * projected_height
                area_changed_inst_ids_device = changed_inst_ids.to(
                    engine.data_collections.node_areas.device
                )
                engine.data_collections.node_areas[area_changed_inst_ids_device] = projected_area

        if not defer_placedb_sync and getattr(placedb, "node_size_x", None) is not None:
            placedb.node_size_x[changed_inst_ids_cpu] = projected_width.detach().cpu().numpy()
        if not defer_placedb_sync and getattr(placedb, "node_size_y", None) is not None:
            placedb.node_size_y[changed_inst_ids_cpu] = projected_height.detach().cpu().numpy()

        summary["runtime_geometry_refresh_applied"] = True
        summary["num_runtime_updated_node_geometries"] = int(changed_inst_ids.numel())

    engine._record_real_size_transition_stage(
        "runtime_cell_apply_ms",
        runtime_cell_apply_started_at,
    )
    summary["num_runtime_updated_vt_instances"] = summary_vt_updated

    pin_summary = engine._apply_projected_pin_offset_consistency(
        params,
        placedb,
        projection_result=projection_result,
        changed_inst_ids=changed_inst_ids,
        projection_frame=projection_frame,
    )
    if isinstance(pin_summary, dict):
        summary["runtime_pin_offset_consistency_applied"] = bool(
            pin_summary.get("runtime_pin_offset_consistency_applied", False)
        )
        summary["num_runtime_updated_pins"] = int(
            pin_summary.get("num_runtime_updated_pins", 0)
        )
        summary["pin_offset_summary_path"] = getattr(
            engine,
            "last_projection_pin_offset_runtime_summary_path",
            None,
        )

    if (
        getattr(engine.op_collections, "steiner_topo_op", None) is not None
        and getattr(engine.op_collections, "pin_pos_op", None) is not None
        and hasattr(engine, "pos")
    ):
        topology = engine._refresh_live_timing_topology(engine.pos[0].data)
        update_kind = str(topology.get("topology_update_kind", "unknown"))
        summary["runtime_topology_refresh_kind"] = update_kind
        summary["runtime_topology_rebuilt"] = update_kind == "topology_rebuild"
        summary["runtime_topology_geometry_refreshed"] = update_kind != "gr_snapshot_unchanged"

    data_collections = engine.data_collections
    data_collections.runtime_cell_state_generation = int(
        getattr(data_collections, "runtime_cell_state_generation", 0) or 0
    ) + 1
    timing_model_refresh_started_at = time.perf_counter()
    data_collections.timing_model_generation = int(
        getattr(data_collections, "timing_model_generation", 0) or 0
    ) + 1
    model = getattr(engine, "last_global_place_model", None)
    virtual_density_cache_invalidated = False
    if model is not None:
        model._timing_geometry_cache = None
        virtual_density_view = getattr(model, "_virtual_cell_density_view", None)
        density_ops = getattr(virtual_density_view, "_density_ops", None)
        if isinstance(density_ops, dict):
            density_ops.clear()
            virtual_density_cache_invalidated = True
    engine._record_real_size_transition_stage(
        "timing_model_refresh_ms",
        timing_model_refresh_started_at,
    )

    summary["runtime_refresh_applied"] = True
    summary["num_runtime_updated_instances"] = int(changed_inst_ids.numel())
    summary["runtime_cell_state_generation"] = int(
        getattr(data_collections, "runtime_cell_state_generation", 0) or 0
    )
    summary["timing_model_generation"] = int(
        getattr(data_collections, "timing_model_generation", 0) or 0
    )
    if projection_frame is not None:
        engine.last_projection_frame = replace(
            projection_frame,
            topology_generation_after=int(
                getattr(
                    getattr(engine.op_collections, "steiner_topo_op", None),
                    "topology_generation",
                    projection_frame.topology_generation_before,
                )
                or projection_frame.topology_generation_before
            ),
            timing_model_generation_after=int(
                getattr(data_collections, "timing_model_generation", 0) or 0
            ),
        )
        summary["projection_frame_state_digest"] = engine.last_projection_frame.state_digest
        summary["projection_frame_topology_generation_before"] = (
            projection_frame.topology_generation_before
        )
        summary["projection_frame_topology_generation_after"] = (
            engine.last_projection_frame.topology_generation_after
        )
        summary["projection_frame_timing_model_generation_before"] = (
            projection_frame.timing_model_generation_before
        )
        summary["projection_frame_timing_model_generation_after"] = (
            engine.last_projection_frame.timing_model_generation_after
        )
    summary["timing_cache_invalidated"] = model is not None
    summary["virtual_density_cache_invalidated"] = virtual_density_cache_invalidated
    engine._write_projection_runtime_refresh_summary(params, summary)
    logging.info(
        "applied projected runtime refresh to %d changed instances (%d pins, geometry=%s, topology=%s)",
        summary["num_runtime_updated_instances"],
        summary["num_runtime_updated_pins"],
        summary["runtime_geometry_refresh_applied"],
        summary["runtime_topology_rebuilt"],
    )
    return summary


@dataclass
class ProjectionTimingSnapshotContext:
    """Runtime views and owners temporarily used for projected-state timing."""

    data_collections: object
    op_collections: object
    model: object
    pos: torch.Tensor | None
    projection_result: object
    capture_timing_metrics: Callable
    restore_timing_metrics: Callable
    enrich_record: Callable


def compute_post_projection_timing_snapshot(
    context,
    params,
    place_record,
    legalization_record,
    projection_summary,
):
    projection_result = context.projection_result
    model = context.model
    data_collections = context.data_collections
    if projection_result is None or model is None or data_collections is None:
        return None

    size_var_getter = getattr(data_collections, "get_size_var", None)
    vt_var_getter = getattr(data_collections, "get_vt_var", None)
    if size_var_getter is None or vt_var_getter is None:
        return None

    current_size = size_var_getter()
    current_vt = vt_var_getter()
    if current_size is None or current_vt is None:
        return None

    legal_mask = projection_result.has_legal_candidate.bool()
    if projection_result.inst_ids.numel() == 0 or not torch.any(legal_mask):
        return None

    projected_size = current_size.detach().clone().float()
    legal_inst_ids = projection_result.inst_ids[legal_mask].long()
    projected_size[legal_inst_ids] = (
        projection_result.projected_size[legal_mask]
        .detach()
        .to(
            projected_size.device,
            dtype=projected_size.dtype,
        )
    )
    projected_vt = projected_vt_distribution(
        current_vt,
        legal_inst_ids,
        projection_result.projected_vt[legal_mask].detach(),
    )

    original_size_var_getter = data_collections.get_size_var
    original_vt_var_getter = data_collections.get_vt_var
    had_inst_libcell_offset = hasattr(data_collections, "inst_libcell_offset")
    had_inst_cell_id = hasattr(data_collections, "inst_cell_id")
    original_inst_libcell_offset = getattr(
        data_collections, "inst_libcell_offset", None
    )
    original_inst_cell_id = getattr(data_collections, "inst_cell_id", None)
    projected_inst_libcell_offset = None
    projected_inst_cell_id = None
    if original_inst_libcell_offset is not None:
        projected_inst_libcell_offset = (
            original_inst_libcell_offset.detach().clone().long()
        )
        projected_inst_libcell_offset[legal_inst_ids] = (
            projection_result.projected_libcell_offset[legal_mask]
            .detach()
            .to(projected_inst_libcell_offset.device)
        )
    if original_inst_cell_id is not None:
        projected_inst_cell_id = original_inst_cell_id.detach().clone().long()
        projected_inst_cell_id[legal_inst_ids] = (
            projection_result.projected_cell_id[legal_mask]
            .detach()
            .to(projected_inst_cell_id.device)
        )

    timing_op = getattr(context.op_collections, "timing_propagation_op", None)
    original_last_timing_metrics = context.capture_timing_metrics()
    original_timing_scalars = {}
    if timing_op is not None:
        for name in (
            "last_total_slew_violation",
            "last_total_cap_violation",
            "last_total_leakage",
            "last_runtime_seconds",
        ):
            original_timing_scalars[name] = getattr(timing_op, name, None)
    try:
        data_collections.get_size_var = lambda: projected_size
        data_collections.get_vt_var = lambda: projected_vt
        if projected_inst_libcell_offset is not None:
            data_collections.inst_libcell_offset = projected_inst_libcell_offset
        if projected_inst_cell_id is not None:
            data_collections.inst_cell_id = projected_inst_cell_id

        if (
            getattr(context.op_collections, "steiner_topo_op", None) is not None
            and getattr(context.op_collections, "pin_pos_op", None) is not None
            and context.pos is not None
        ):
            pin_pos_for_topology = (
                context.op_collections.pin_pos_op(context.pos)
                .detach()
                .cpu()
                .contiguous()
            )
            (
                context.data_collections.net_flat_topo_sort,
                context.data_collections.net_flat_topo_sort_start,
                context.data_collections.pin_fa,
                context.data_collections.flat_pin_to,
                context.data_collections.flat_pin_to_start,
                context.data_collections.flat_pin_from,
            ) = context.op_collections.steiner_topo_op.rebuild_tree(
                pin_pos_for_topology
            )

        pos = context.pos
        wns, tns, ws, ts = model.timing_obj(pos)

        leakage = projection_summary.get("total_projected_leakage")
        if leakage is None and timing_op is not None:
            leakage = getattr(timing_op, "last_total_leakage", None)
        record = context.enrich_record(
            {
                "wns": timing_scalar(wns),
                "tns": timing_scalar(tns),
                "timing_objective": timing_scalar(ws),
                "slew_violation": None
                if timing_op is None
                else getattr(timing_op, "last_total_slew_violation", None),
                "cap_violation": None
                if timing_op is None
                else getattr(timing_op, "last_total_cap_violation", None),
                "leakage": leakage,
                "stage_note": "projected_discrete_state_retimed_before_legalization",
            }
        )
        return {
            "record": record,
            "endpoint_payload": {
                "source": "backend_endpoint_debug",
                "available": False,
                "summary_path": None,
                "csv_path": None,
                "wns": None,
                "tns": None,
                "num_negative_endpoints": None,
                "python_wns": None,
                "python_tns": None,
            },
            "top_endpoints": [],
        }
    except Exception:
        logging.exception("failed to compute dedicated post-projection timing snapshot")
        return None
    finally:
        data_collections.get_size_var = original_size_var_getter
        data_collections.get_vt_var = original_vt_var_getter
        if had_inst_libcell_offset:
            data_collections.inst_libcell_offset = original_inst_libcell_offset
        if had_inst_cell_id:
            data_collections.inst_cell_id = original_inst_cell_id
        if original_last_timing_metrics is not None:
            context.restore_timing_metrics(original_last_timing_metrics)
        if timing_op is not None:
            for name, value in original_timing_scalars.items():
                setattr(timing_op, name, value)
