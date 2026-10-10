"""Buffering lane orchestration for one placement iteration."""

import logging
import time

import torch


def run_buffering_inner_loop(engine, params, model):
    if not engine._buffering_flow_enabled(params):
        return None
    runtime_profile_enabled = bool(
        getattr(params, "buffering_runtime_profile", False)
    )
    runtime_profile_started_at = time.perf_counter()
    runtime_profile = {
        "enabled": True,
        "clock": "perf_counter_wall",
        "live_timing_topology_refresh_ms": 0.0,
        "lane_build_ms": 0.0,
        "lane_prepare_ms": 0.0,
        "state_attach_ms": 0.0,
        "virtual_scheduler_ms": 0.0,
        "commit_request_ms": 0.0,
        "summary_build_ms": 0.0,
    }

    def record_runtime_stage(name, started_at):
        if runtime_profile_enabled:
            runtime_profile[name] = (
                time.perf_counter() - started_at
            ) * 1000.0

    data_collections = getattr(model, "data_collections", None)
    positions = (
        getattr(data_collections, "pos", None)
        if data_collections is not None
        else None
    )
    op_collections = getattr(engine, "op_collections", None)
    if (
        positions is not None
        and len(positions) > 0
        and hasattr(engine, "_refresh_live_timing_topology")
        and getattr(op_collections, "pin_pos_op", None) is not None
        and getattr(op_collections, "steiner_topo_op", None) is not None
    ):
        runtime_stage_started_at = time.perf_counter()
        engine._refresh_live_timing_topology(positions[0])
        record_runtime_stage(
            "live_timing_topology_refresh_ms",
            runtime_stage_started_at,
        )

    runtime_stage_started_at = time.perf_counter()
    lane = engine._build_buffering_lane(params)
    record_runtime_stage("lane_build_ms", runtime_stage_started_at)
    runtime_stage_started_at = time.perf_counter()
    lane.prepare(model=model, params=params)
    record_runtime_stage("lane_prepare_ms", runtime_stage_started_at)
    if runtime_profile_enabled:
        lane_prepare_profile = dict(
            getattr(lane, "summarize", lambda: {})().get(
                "lane_prepare_runtime_profile",
                {},
            )
            or {}
        )
        runtime_profile["lane_prepare_profile"] = lane_prepare_profile
    if data_collections is not None:
        runtime_stage_started_at = time.perf_counter()
        lane.attach_state(data_collections)
        record_runtime_stage("state_attach_ms", runtime_stage_started_at)

    mode = str(getattr(lane.config, "mode", "segment") or "segment")
    strategy = str(
        getattr(
            lane.config,
            "active_strategy",
            (
                getattr(lane.config, "segment_strategy", "continuous")
                if mode == "segment"
                else getattr(lane.config, "candidate_strategy", "continuous")
            ),
        )
        or "continuous"
    )
    if strategy == "discrete_net_gradient":
        state = getattr(lane, "_current_state", lambda: None)()
        expected_state_attr = "z_param" if mode == "segment" else "activation_param"
        if state is None or not hasattr(state, expected_state_attr):
            summary = lane.summarize()
            summary.update(
                {
                    "status": "skipped",
                    "reason": (
                        "missing_segment_count_state"
                        if mode == "segment"
                        else "missing_discrete_candidate_state"
                    ),
                    "strategy": strategy,
                }
            )
            engine.last_buffering_lane = lane
            engine.last_buffering_lane_summary = summary
            engine._write_buffering_inner_loop_artifact(params, summary)
            logging.info(
                "Buffering inner-loop summary: %s",
                engine._compact_buffering_inner_loop_log_summary(summary),
            )
            return summary
        runtime_stage_started_at = time.perf_counter()
        if mode == "candidate":
            from dreamplace.ops.buffer_insertion.discrete_candidate_buffer_runner import (
                run_discrete_candidate_buffer_scheduler,
            )

            loop_summary = run_discrete_candidate_buffer_scheduler(
                lane=lane,
                model=model,
                rounds=int(getattr(lane.config, "continuous_steps", 0)),
            )
        else:
            from dreamplace.ops.buffer_insertion.discrete_virtual_buffer_runner import (
                run_discrete_virtual_buffer_scheduler,
            )

            loop_summary = run_discrete_virtual_buffer_scheduler(
                lane=lane,
                model=model,
                rounds=int(getattr(lane.config, "continuous_steps", 0)),
            )
        record_runtime_stage("virtual_scheduler_ms", runtime_stage_started_at)
    else:
        trainable_params = lane.optimizer_parameters()
        if not trainable_params:
            summary = lane.summarize()
            summary.update(
                {
                    "status": "skipped",
                    "reason": "missing_buffer_trainable_parameters",
                }
            )
            engine.last_buffering_lane = lane
            engine.last_buffering_lane_summary = summary
            engine._write_buffering_inner_loop_artifact(params, summary)
            logging.info(
                "Buffering inner-loop summary: %s",
                engine._compact_buffering_inner_loop_log_summary(summary),
            )
            return summary

        from dreamplace.ops.buffer_insertion.buffering_inner_loop import (
            run_buffering_inner_loop,
        )

        optimizer = torch.optim.Adam(
            trainable_params,
            lr=float(getattr(lane.config, "continuous_lr", 0.05)),
        )
        runtime_stage_started_at = time.perf_counter()
        loop_summary = run_buffering_inner_loop(
            lane=lane,
            model=model,
            optimizer=optimizer,
            steps=int(getattr(lane.config, "continuous_steps", 0)),
            project_interval=0,
        )
        record_runtime_stage("virtual_scheduler_ms", runtime_stage_started_at)
    runtime_refresh = dict(
        getattr(lane, "last_runtime_refresh_summary", {}) or {}
    )
    runtime_stage_started_at = time.perf_counter()
    commit_request = lane.build_commit_request(
        loop_summary.final_projection,
        iteration=int(loop_summary.iterations),
        reason=(
            (
                "buffering_candidate_discrete_final_commit"
                if mode == "candidate"
                else "buffering_discrete_virtual_final_commit"
            )
            if strategy == "discrete_net_gradient"
            else "buffering_inner_loop_final_commit"
        ),
        runtime_refresh=runtime_refresh,
    )
    commit_request_summary = commit_request.to_summary()
    record_runtime_stage("commit_request_ms", runtime_stage_started_at)
    if commit_request.commit_enabled and not commit_request.is_noop:
        commit_result = {
            "status": "request_ready",
            "iteration": int(loop_summary.iterations),
            "projected_buffer_count": int(
                getattr(
                    loop_summary.final_projection,
                    "projected_buffer_count",
                    0,
                )
                or 0
            ),
            "runtime_refresh": runtime_refresh,
            "commit_request": commit_request_summary,
        }
        commit_summary = {
            "status": "pending_buffer_commit_request",
            "reason": "physical_buffer_commit_owned_by_placement_engine",
            "commit_enabled": True,
            "action_count": len(commit_request.actions),
        }
        engine.last_buffer_commit_request = commit_request
        engine.last_post_commit_control_result = engine._joint_post_commit_control_result(
            commit_result
        )
    else:
        if not commit_request.commit_enabled:
            commit_summary = {
                "status": "disabled",
                "reason": "commit_not_requested",
                "commit_enabled": False,
                "action_count": len(commit_request.actions),
            }
        else:
            commit_summary = {
                "status": "skipped",
                "reason": "no_projected_buffer_actions",
                "commit_enabled": True,
                "action_count": 0,
            }
        engine.last_buffer_commit_request = None
        engine.last_post_commit_control_result = None
    runtime_stage_started_at = time.perf_counter()
    summary = lane.summarize()
    summary.update(
        {
            "status": "completed",
            "strategy": getattr(loop_summary, "strategy", strategy),
            "terminal_reason": getattr(loop_summary, "terminal_reason", None),
            "trace_path": str(getattr(loop_summary, "trace_path", "") or ""),
            "trace_sha256": str(getattr(loop_summary, "trace_sha256", "") or ""),
            "inner_loop": {
                "iterations": loop_summary.iterations,
                "strategy": getattr(loop_summary, "strategy", strategy),
                "terminal_reason": getattr(loop_summary, "terminal_reason", None),
                "terminal_metrics": dict(
                    getattr(loop_summary, "terminal_metrics", {}) or {}
                ),
                "trace_path": str(getattr(loop_summary, "trace_path", "") or ""),
                "trace_sha256": str(
                    getattr(loop_summary, "trace_sha256", "") or ""
                ),
                "projection_count": len(loop_summary.projection_trace),
                "periodic_integer_projection_count": len(
                    loop_summary.periodic_integer_projection_trace
                ),
                "final_projection_wall_ms": loop_summary.final_projection_wall_ms,
                "metrics_trace": list(loop_summary.metrics_trace),
                "periodic_integer_projection_trace": list(
                    loop_summary.periodic_integer_projection_trace
                ),
            },
            "commit": commit_summary,
            "commit_request": commit_request_summary,
        }
    )
    record_runtime_stage("summary_build_ms", runtime_stage_started_at)
    if runtime_profile_enabled:
        runtime_profile["total_pre_artifact_wall_ms"] = (
            time.perf_counter() - runtime_profile_started_at
        ) * 1000.0
        summary["runtime_profile"] = runtime_profile
    engine.last_buffering_lane = lane
    engine.last_buffering_inner_loop_summary = loop_summary
    engine.last_buffering_lane_summary = summary
    engine._write_buffering_inner_loop_artifact(params, summary)
    logging.info(
        "Buffering inner-loop summary: %s",
        engine._compact_buffering_inner_loop_log_summary(summary),
    )
    return summary
