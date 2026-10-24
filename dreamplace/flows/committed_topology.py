"""Rebuild runtime topology and placement RC after a native buffer commit."""

import copy
import logging
import math
import os
import time
from enum import Enum

import dreamplace.NonLinearPlace as NonLinearPlace


def finalize_inflation_s5b1(engine):
    """One native commit/refresh followed by legalization and placement-RC STA."""
    return finalize_discrete_window(engine, engine.placer.inflation_s5b1)


class MaterializationAreaPolicy(Enum):
    EXACT = "exact_virtual_area"
    CAPACITY = "bounded_after_native_sizing_and_buffer_commit"


def validate_materialized_area(physical_area, summary, accepted, area_policy):
    """Validate physical commit accounting independently of artifact naming."""
    expected_area = float(summary["gp_area"]["movable"])
    if area_policy is MaterializationAreaPolicy.EXACT:
        if not math.isclose(physical_area, expected_area, rel_tol=1e-6):
            raise RuntimeError("virtual-to-real materialization changed physical movable area")
    elif area_policy is MaterializationAreaPolicy.CAPACITY:
        rebuild = dict(summary.get("refresh", {}).get("rebuild", {}) or {})
        pre_nodes = int((rebuild.get("pre_counts", {}) or {}).get("nodes", 0) or 0)
        post_nodes = int((rebuild.get("post_counts", {}) or {}).get("nodes", 0) or 0)
        if post_nodes != pre_nodes + accepted:
            raise RuntimeError("terminal rebuild node count does not match committed buffers")
        capacity = float(summary["gp_area"].get("capacity", 0.0) or 0.0)
        if not math.isfinite(physical_area) or physical_area <= 0.0:
            raise RuntimeError("terminal movable area is non-finite or empty")
        if capacity > 0.0 and physical_area > capacity * (1.0 + 1e-6):
            raise RuntimeError("terminal movable area exceeds placement capacity")
    else:
        raise ValueError(f"unknown materialization area policy: {area_policy!r}")
    summary["materialized_area_consistency"] = area_policy.value


def finalize_discrete_window(
    engine, owner, *, artifact_stem="inflation_s5b1",
    area_policy=MaterializationAreaPolicy.EXACT,
):
    """Finalize a virtual electrical window using the canonical ECC lifecycle."""
    from dreamplace.ops.placeio_common.physical_mutation import write_native_def
    from dreamplace.ops.routability.gpugr_context import write_back_movable_lpos

    request = owner.terminal_request
    if request is None or owner.terminal_commit_started:
        raise RuntimeError("S5B1 terminal commit is missing or was already attempted")
    owner.terminal_commit_started = True
    started = time.perf_counter()
    params, placedb = engine.params, engine.placedb
    result_dir = str(params.result_dir)
    stem = os.path.join(result_dir, params.design_name() + "_" + artifact_stem)
    summary = {"status": "pending", **owner.terminal_summary, "request": request.to_summary(),
               "atomic_across_sizing_and_buffer": False}
    try:
        sync_native_sizing = getattr(
            owner,
            "sync_native_sizing",
            getattr(owner, "_sync_native_sizing", None),
        )
        if not callable(sync_native_sizing):
            raise RuntimeError("S5B1 terminal owner has no native sizing sync")
        summary["native_sizing"] = sync_native_sizing()
        write_back_movable_lpos(engine.placer.pos[0], params, placedb)
        backend = engine._physical_mutation_backend()
        commit = backend.commit_buffers(request, output_dir=result_dir)
        summary["commit"] = commit
        accepted = int(commit.get("accepted_action_count", 0))
        if accepted != len(request.actions) or commit.get("status") not in {"accepted", "skipped"}:
            raise RuntimeError(f"S5B1 final buffer transaction failed: {commit}")
        # The virtual entries have now been materialized in the native design.
        # Preserve the pre-commit count for diagnostics while reporting the
        # terminal state consumed by regression checks.
        summary["virtual_count_before_commit"] = int(summary.get("virtual_count", 0) or 0)
        summary["virtual_count"] = 0

        # This is the terminal physical stage. Filler is no longer needed and
        # must not be regenerated from the pre-window target density on rebuild.
        fillers_enabled = params.enable_fillers
        params.enable_fillers = 0
        try:
            summary["refresh"] = backend.refresh(
                refresh_mode="full_rebuild", rebuild_mode="full_rebuild",
            )
        finally:
            params.enable_fillers = fillers_enabled
        if summary["refresh"].get("status") != "ok":
            raise RuntimeError(f"S5B1 terminal full refresh failed: {summary['refresh']}")
        physical_area = float(placedb.total_movable_node_area)
        expected_area = float(summary["gp_area"]["movable"])
        summary["materialized_movable_area"] = physical_area
        summary["materialized_area_delta_from_gp"] = physical_area - expected_area
        summary["materialized_area_relative_delta_from_gp"] = (
            (physical_area - expected_area) / expected_area
            if expected_area
            else None
        )
        validate_materialized_area(physical_area, summary, accepted, area_policy)
        placedb.cooptimization_geometry = None
        owner.ops.steiner_topo_op.frozen_nets.segment_state = None
        owner.data.buffer_optimization_state = None
        owner.data.buffer_segment_count_state = None
        owner.data.buffer_segment_count_payload = None
        owner.data.buffer_relaxed_timing_payload = None
        owner.lane.buffer_optimization_state = None
        owner.ops.virtual_cell_density_op = None
        summary["virtual_view_removed"] = True

        terminal_params = copy.copy(params)
        terminal_params.flow_kind = "placement"
        terminal_params.joint_quality_profile = ""
        terminal_params.global_place_flag = 0
        terminal_params.legalize_flag = 1
        terminal_params.timing_opt_enabled = 0
        terminal_params.routability_opt_flag = 0
        terminal_params.l_shape_routability_flag = 0
        terminal_params.sizing_parameterization = "logits"
        terminal_params.placement_sizing_mode = "place_only"
        terminal_params.enable_relaxed_buffer_timing = False
        terminal_params.enable_fillers = 0
        terminal_params.result_dir = stem + "_terminal"
        os.makedirs(terminal_params.result_dir, exist_ok=True)
        new_placer = NonLinearPlace.NonLinearPlace(terminal_params, placedb, None)
        engine.placer = new_placer
        engine.rsmt, engine.hpwl, engine.metrics = new_placer(terminal_params, placedb)
        if not new_placer.op_collections.legality_check_op(new_placer.pos[0]):
            raise RuntimeError("S5B1 terminal placement is not legal")
        _, _, timing_metrics = new_placer.run_sta_only(terminal_params, placedb)
        summary["final_timing"] = timing_metrics["timing_stage_summary"]["sta"]
        if not all(math.isfinite(float(summary["final_timing"][key])) for key in ("wns", "tns")):
            raise RuntimeError("S5B1 terminal placement timing is non-finite")
        summary["final_debug"] = new_placer.last_placement_debug_summary
        summary["final_def"] = stem + ".def"
        summary["final_verilog"] = stem + ".v"
        write_native_def(placedb, summary["final_def"])
        engine.write_verilog(summary["final_verilog"])
        summary.update(status="ok", legal=True, placement_rc_model="DreamPlace_LEF_FLUTE",
                       native_sta_role="diagnostic_no_SPEF", new_gp_steps=0,
                       backend_capabilities=dict(placedb.backend_caps))
        engine.last_run_result = {"status": "ok", "rsmt": engine.rsmt,
                                  "hpwl": engine.hpwl, artifact_stem + "_terminal": summary}
        return engine.last_run_result
    except Exception as exc:
        summary.update(status="failed", error=str(exc))
        raise
    finally:
        summary["runtime_ms"] = (time.perf_counter() - started) * 1000.
        engine._write_json_artifact(stem + "_terminal.json", summary)


def refresh_committed_topology(engine, control_result):
    refresh_mode = control_result.get("refresh_mode", "topo")
    rebuild_mode = control_result.get("rebuild_mode", "topo")
    logging.info(
        "buffering commit mutated topology; refreshing native database -> pydb "
        "with refresh_mode=%s rebuild_mode=%s before constructing a fresh NonLinearPlace",
        refresh_mode,
        rebuild_mode,
    )
    old_placer_id = id(engine.placer)
    buffering_summary = dict(getattr(engine.placer, "last_buffering_lane_summary", None) or {})
    buffering_optimization_path = None
    if buffering_summary:
        buffering_optimization_path = os.path.join(
            engine.params.result_dir,
            f"{engine._design_name()}_pre_commit_buffering_optimization.json",
        )
        engine._write_json_artifact(
            buffering_optimization_path,
            {
                key: buffering_summary[key]
                for key in ("inner_loop", "segment_count", "state_kind", "mode")
                if key in buffering_summary
            },
        )
    joint_flow_artifact_path = getattr(
        engine.placer,
        "last_joint_flow_artifact_path",
        None,
    )
    buffering_lane = getattr(engine.placer, "last_buffering_lane", None)
    if buffering_lane is None:
        coordinator = getattr(engine.placer, "joint_coordinator", None)
        buffering_lane = getattr(coordinator, "buffering_lane", None)
    post_commit_pysta_path = str(
        getattr(getattr(buffering_lane, "config", None), "post_commit_pysta_json", "") or ""
    )
    openroad_eco_summary = engine._run_joint_post_commit_openroad_eco(control_result)
    openroad_eco_summary_path = (openroad_eco_summary or {}).get("summary_path")
    if openroad_eco_summary_path:
        engine._reference_joint_post_commit_openroad_eco_summary(
            control_result,
            openroad_eco_summary_path,
            joint_flow_artifact_path=joint_flow_artifact_path,
        )
    mutation_backend = getattr(engine, "last_physical_mutation_backend", None)
    if mutation_backend is None:
        mutation_backend = engine._physical_mutation_backend()
    refresh_summary = mutation_backend.refresh(
        refresh_mode=refresh_mode,
        rebuild_mode=rebuild_mode,
    )
    if str(refresh_summary.get("status") or "") != "ok":
        raise RuntimeError(
            f"committed topology refresh failed: {refresh_summary.get('reason') or refresh_summary}"
        )
    engine.placer = None
    new_placer = NonLinearPlace.NonLinearPlace(engine.params, engine.placedb, None)
    engine.placer = new_placer
    post_rebuild_result = {
        "status": "ok",
        "continuation_policy": "rebuild_placer_outer_loop",
        "old_nonlinear_place_id": old_placer_id,
        "new_nonlinear_place_id": id(new_placer),
        "fresh_nonlinear_place_constructed": id(new_placer) != old_placer_id,
        "topology_generation": int(getattr(engine.placedb, "topology_generation", 0) or 0),
        "pre_commit_buffering_optimization_path": buffering_optimization_path,
        "openroad_eco_summary": dict(openroad_eco_summary or {}),
        "openroad_eco_status": (openroad_eco_summary or {}).get("status"),
        "openroad_eco_mode": (openroad_eco_summary or {}).get("mode"),
        "openroad_eco_summary_path": (openroad_eco_summary or {}).get("summary_path"),
    }
    try:
        timing_started_at = time.perf_counter()
        relaxed_timing_was_enabled = getattr(
            engine.params,
            "enable_relaxed_buffer_timing",
            False,
        )
        segment_gradient_was_enabled = getattr(
            engine.params,
            "buffering_segment_count_tns_gradient",
            None,
        )
        engine.params.enable_relaxed_buffer_timing = False
        if segment_gradient_was_enabled is not None:
            engine.params.buffering_segment_count_tns_gradient = 0
        try:
            sta_rsmt, sta_hpwl, sta_metrics = new_placer.run_sta_only(
                engine.params,
                engine.placedb,
            )
        finally:
            engine.params.enable_relaxed_buffer_timing = relaxed_timing_was_enabled
            if segment_gradient_was_enabled is not None:
                engine.params.buffering_segment_count_tns_gradient = segment_gradient_was_enabled
        post_rebuild_result.update(
            {
                "post_rebuild_timing_status": "ok",
                "post_rebuild_timing_mode": "committed_topology_plain_pysta",
                "placement_rc_model": "DreamPlace_LEF_FLUTE",
                "post_rebuild_timing_wall_ms": float(
                    (time.perf_counter() - timing_started_at) * 1000.0
                ),
                "rsmt": sta_rsmt,
                "hpwl": sta_hpwl,
            }
        )
        if isinstance(sta_metrics, dict):
            for key in ("wns", "tns", "objective", "overflow", "density"):
                if key in sta_metrics:
                    post_rebuild_result[key] = sta_metrics[key]
            timing_stage_summary = sta_metrics.get("timing_stage_summary")
            timing_record = (
                timing_stage_summary.get("sta", {})
                if isinstance(timing_stage_summary, dict)
                else {}
            )
            if isinstance(timing_record, dict):
                for key in ("wns", "tns"):
                    if key in timing_record:
                        post_rebuild_result[key] = timing_record[key]
                if "timing_objective" in timing_record:
                    post_rebuild_result["worst_slack"] = timing_record["timing_objective"]
        if not all(math.isfinite(float(post_rebuild_result[key])) for key in ("wns", "tns")):
            raise RuntimeError("committed-topology timing contains non-finite WNS/TNS")
    except Exception as exc:
        post_rebuild_result["status"] = "failed"
        post_rebuild_result["post_rebuild_timing_status"] = "failed"
        post_rebuild_result["error"] = str(exc)
        logging.exception("post-commit fresh NonLinearPlace timing check failed")
        raise
    if post_commit_pysta_path:
        post_commit_pysta = {
            "artifact_version": 2,
            "artifact_scope": "buffering_post_commit_pysta",
            "status": post_rebuild_result["post_rebuild_timing_status"],
            "timing_mode": post_rebuild_result["post_rebuild_timing_mode"],
            "wns": post_rebuild_result.get("wns"),
            "tns": post_rebuild_result.get("tns"),
            "worst_slack": post_rebuild_result.get("worst_slack"),
            "wall_ms": post_rebuild_result.get("post_rebuild_timing_wall_ms"),
        }
        try:
            engine._write_json_artifact(post_commit_pysta_path, post_commit_pysta)
            post_rebuild_result["post_commit_pysta_json_path"] = post_commit_pysta_path
        except Exception as exc:  # pragma: no cover - configured diagnostic path
            post_rebuild_result["post_commit_pysta_write_error"] = str(exc)
            logging.exception("Failed to write committed-topology PySTA artifact")
    summary_path = engine._write_joint_post_rebuild_summary(
        control_result,
        refresh_summary,
        post_rebuild_result,
    )
    if summary_path:
        post_rebuild_result["summary_path"] = summary_path
    refresh_summary_path = engine._write_buffer_commit_refresh_summary(
        control_result,
        refresh_summary,
        post_rebuild_result,
    )
    if refresh_summary_path:
        post_rebuild_result["buffer_commit_pydb_refresh_summary_path"] = refresh_summary_path
        engine._reference_buffer_commit_refresh_summary(
            control_result,
            refresh_summary_path,
            joint_flow_artifact_path=joint_flow_artifact_path,
        )
        engine._write_joint_post_rebuild_summary(
            control_result,
            refresh_summary,
            post_rebuild_result,
        )
    engine.last_joint_post_rebuild_result = post_rebuild_result
    return post_rebuild_result
