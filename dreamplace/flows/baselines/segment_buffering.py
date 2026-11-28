"""ECC segment sizing frame owner; live state is supplied per call."""


from dataclasses import dataclass
from typing import Callable
import hashlib
import time
import torch
import os
import json
import logging


UNSET = object()

@dataclass
class SegmentSizingFrameContext:
    coordinator: object
    data_collections: object
    digest: Callable
    freeze_frame: Callable
    profile_clock: Callable
    profile_enabled: Callable
    select: Callable
    sync_cells: Callable

def freeze_segment_milestone_sizing_frame(
    context,
    *,
    params,
    model,
    pos,
    event,
):
    coordinator = getattr(context, "coordinator", None)
    if (
        event is None
        or coordinator is None
        or not coordinator.uses_overflow_milestone_actions
    ):
        return None
    data_collections = context.data_collections
    size_logits = getattr(data_collections, "size_logits", None)
    sizing_enabled = bool(getattr(params, "joint_segment_sizing_enabled", True))
    if sizing_enabled and size_logits is None:
        raise RuntimeError("milestone sizing requires size_logits")
    size_grad = None
    if size_logits is not None and size_logits.grad is not None:
        size_grad = size_logits.grad.detach().clone()
    if sizing_enabled and size_grad is None:
        raise RuntimeError(
            "milestone sizing objective did not populate size_logits.grad"
        )
    if size_grad is not None and not bool(torch.isfinite(size_grad).all()):
        raise RuntimeError("milestone size_logits.grad contains non-finite values")

    state = None
    if coordinator.buffering_lane is not None:
        state = coordinator.buffering_lane._current_state()
    z_param = None if state is None else getattr(state, "z_param", None)
    state_digest = context.digest(
        (
            ("pos", pos),
            ("size_logits", size_logits),
            ("inst_cell_id", getattr(data_collections, "inst_cell_id", None)),
            ("z_param", z_param),
        )
    )
    grad_digest = context.digest((("size_grad", size_grad),))
    summary = {
        "status": "ready" if sizing_enabled else "disabled",
        "event_id": int(event["event_id"]),
        "iteration": int(event["iteration"]),
        "milestone": float(event["milestone"]),
        "frame": "pre_nesterov_x_t",
        "objective_mode": "normal_joint_objective",
        "timing_objective_lane": str(
            getattr(params, "timing_objective_lane", "timing_slew_cap")
        ),
        "timing_wns_coeff": float(getattr(params, "timing_wns_coeff", 0.0)),
        "timing_tns_coeff": float(getattr(params, "timing_tns_coeff", 0.0)),
        "size_density_area_weight": float(
            getattr(model, "size_density_area_weight", 0.1)
        ),
        "state_digest": state_digest,
        "gradient_digest": grad_digest,
        "gradient_norm": (
            None
            if size_grad is None
            else float(size_grad.float().norm().cpu().item())
        ),
        "gradient_nonzero_count": (
            0
            if size_grad is None
            else int((size_grad != 0.0).sum().cpu().item())
        ),
        "gradient_positive_count": (
            0
            if size_grad is None
            else int((size_grad > 0.0).sum().cpu().item())
        ),
        "gradient_negative_count": (
            0
            if size_grad is None
            else int((size_grad < 0.0).sum().cpu().item())
        ),
        "topology_generation": coordinator.segment_milestones.frozen_topology_generation,
        "enabled": sizing_enabled,
    }
    return {"grad": size_grad, "summary": summary}


def capture_segment_milestone_sizing_frame(
    context,
    *,
    params,
    model,
    pos,
    event,
    evaluate_objective,
    advance_stage=True,
):
    profile_enabled = context.profile_enabled(params)

    def profile_clock():
        return context.profile_clock(
            profile_enabled,
            pos,
        )

    started_at = profile_clock()
    sizing_enabled = bool(
        getattr(params, "joint_segment_sizing_enabled", True)
    )
    evaluate_objective = bool(evaluate_objective and sizing_enabled)
    objective_evidence = {
        "objective_mode": "disabled",
        "timing_surrogate_mode": None,
        "placement_timing_surrogate_mode": str(
            getattr(params, "timing_surrogate_mode", "lut_only")
        ),
    }
    if evaluate_objective:
        size_logits = getattr(context.data_collections, "size_logits", None)
        if size_logits is None:
            raise RuntimeError("milestone sizing objective requires size_logits")
        timing_started_at = profile_clock()
        wns, tns, ws, ts = model.timing_obj(
            pos,
            surrogate_mode="surrogate_only",
        )
        timing_finished_at = profile_clock()
        sizing_loss = model._timing_loss(wns, tns, ws, ts)
        area_penalty = model._continuous_density_area_penalty()
        if area_penalty is not None:
            sizing_loss = sizing_loss + area_penalty
        loss_finished_at = profile_clock()
        size_grad = torch.autograd.grad(
            sizing_loss,
            size_logits,
            allow_unused=True,
        )[0]
        autograd_finished_at = profile_clock()
        model._timing_geometry_cache = None
        if size_grad is None:
            raise RuntimeError(
                "milestone surrogate timing objective did not depend on size_logits"
            )
        size_logits.grad = size_grad.detach()
        objective_evidence = {
            "objective_mode": "milestone_sizing_surrogate_plus_area",
            "timing_surrogate_mode": "surrogate_only",
            "placement_timing_surrogate_mode": str(
                getattr(params, "timing_surrogate_mode", "lut_only")
            ),
            "objective_value": float(sizing_loss.detach().cpu().item()),
            "wns": float(wns.detach().cpu().item()),
            "tns": float(tns.detach().cpu().item()),
            "area_penalty": (
                None
                if area_penalty is None
                else float(area_penalty.detach().cpu().item())
            ),
        }
    else:
        size_logits = getattr(context.data_collections, "size_logits", None)
        if size_logits is not None:
            size_logits.grad = None
        timing_started_at = started_at
        timing_finished_at = started_at
        loss_finished_at = started_at
        autograd_finished_at = started_at
    freeze_started_at = profile_clock()
    frame = context.freeze_frame(
        params=params,
        model=model,
        pos=pos,
        event=event,
    )
    finished_at = profile_clock()
    frame["summary"].update(objective_evidence)
    frame["summary"]["runtime_ms"] = float((finished_at - started_at) * 1000.0)
    if profile_enabled:
        frame["summary"]["runtime_profile"] = {
            "enabled": True,
            "synchronized": bool(pos.is_cuda),
            "timing_obj_ms": float(
                (timing_finished_at - timing_started_at) * 1000.0
            ),
            "loss_and_area_ms": float(
                (loss_finished_at - timing_finished_at) * 1000.0
            ),
            "autograd_ms": float(
                (autograd_finished_at - loss_finished_at) * 1000.0
            ),
            "frame_freeze_ms": float(
                (finished_at - freeze_started_at) * 1000.0
            ),
            "total_ms": float((finished_at - started_at) * 1000.0),
        }
    if advance_stage:
        context.coordinator.advance_segment_milestone(
            event=event,
            stage="sizing_gradient_frozen",
            evidence=dict(frame.get("summary", {}) or {}),
        )
    return frame


def apply_segment_milestone_discrete_sizing(
    context,
    *,
    params,
    event,
    sizing_frame,
):
    started_at = time.perf_counter()
    if sizing_frame is None:
        raise RuntimeError("milestone sizing frame is unavailable")
    frame_summary = dict(sizing_frame.get("summary", {}) or {})
    if int(frame_summary.get("event_id", -1)) != int(event["event_id"]):
        raise RuntimeError("milestone sizing frame event identity mismatch")
    if not bool(frame_summary.get("enabled", False)):
        return {
            "status": "disabled",
            "reason": "joint_segment_sizing_disabled",
            "frame": frame_summary,
            "num_changed_instances": 0,
            "applied_instance_ids": [],
            "applied_cell_ids": [],
            "area_delta_internal": 0.0,
            "input_state_digest": frame_summary.get("state_digest"),
            "output_state_digest": frame_summary.get("state_digest"),
            "runtime_ms": float(
                (time.perf_counter() - started_at) * 1000.0
            ),
        }

    size_logits = getattr(context.data_collections, "size_logits", None)
    size_grad = sizing_frame.get("grad")
    if size_logits is None or size_grad is None:
        raise RuntimeError("enabled milestone sizing requires logits and gradient")
    if context.digest((("size_grad", size_grad),)) != frame_summary.get(
        "gradient_digest"
    ):
        raise RuntimeError("milestone sizing gradient frame changed after capture")

    target_logits, _target_sizes, summary, _candidate_table = (
        context.select(
            params=params,
            size_logits=size_logits,
            size_grad=size_grad,
            policy={
                "up_percent": float(
                    getattr(params, "joint_segment_sizing_up_percent", 10.0)
                ),
                "down_percent": 0.0,
                "preserve_vt": True,
                "ranking_mode": "real_size_taylor",
                "step_mode": "direct_one_step",
            },
        )
    )
    with torch.no_grad():
        size_logits.copy_(
            target_logits.to(device=size_logits.device, dtype=size_logits.dtype)
        )
    summary.update(
        {
            "status": "applied",
            "event_id": int(event["event_id"]),
            "iteration": int(event["iteration"]),
            "milestone": float(event["milestone"]),
            "frame": frame_summary,
            "policy": {
                "up_percent": float(
                    getattr(params, "joint_segment_sizing_up_percent", 10.0)
                ),
                "down_percent": 0.0,
                "preserve_vt": True,
                "ranking_mode": "real_size_taylor",
                "step_mode": "direct_one_step",
            },
        }
    )
    action_digest = hashlib.sha256()
    for inst_id, cell_id in zip(
        summary.get("applied_instance_ids", ()),
        summary.get("applied_cell_ids", ()),
    ):
        action_digest.update(f"{int(inst_id)}:{int(cell_id)};".encode("ascii"))
    summary["action_digest"] = "sha256:" + action_digest.hexdigest()
    summary["input_state_digest"] = frame_summary.get("state_digest")
    summary.update(context.sync_cells(params, summary))
    summary["output_state_digest"] = context.digest(
        (
            ("size_logits", size_logits),
            ("inst_cell_id", context.data_collections.inst_cell_id),
            (
                "inst_libcell_offset",
                getattr(context.data_collections, "inst_libcell_offset", None),
            ),
            ("node_size_x", getattr(context.data_collections, "node_size_x", None)),
            ("node_size_y", getattr(context.data_collections, "node_size_y", None)),
        )
    )
    summary["runtime_ms"] = float(
        (time.perf_counter() - started_at) * 1000.0
    )
    return summary


@dataclass(frozen=True)
class SegmentGradientContext:
    joint_coordinator: object
    data_collections: object
    digest: Callable
    profile_enabled: Callable
    profile_clock: Callable
    refresh_topology: Callable

def compute_refreshed_segment_buffer_gradient(
    context,
    *,
    params,
    model,
    pos,
    event,
    refresh_topology=True,
    frame="post_placement_post_sizing_x_next",
    expected_topology_generation=None,
):
    coordinator = context.joint_coordinator
    state = coordinator.buffering_lane._current_state()
    z_param = state.z_param
    if not coordinator.segment_route_b_enabled:
        return {
            "status": "disabled",
            "reason": "joint_segment_route_b_disabled",
            "event_id": int(event["event_id"]),
            "state_digest": context.digest(
                (("pos", pos), ("z_param", z_param))
            ),
        }

    if z_param.grad is not None:
        z_param.grad = None
    size_logits = getattr(context.data_collections, "size_logits", None)
    if size_logits is not None and size_logits.grad is not None:
        size_logits.grad = None
    pos_before = pos.detach().clone()
    profile_enabled = context.profile_enabled(params)

    def profile_clock():
        return context.profile_clock(
            profile_enabled,
            pos,
        )

    started_at = profile_clock()
    if refresh_topology:
        context.refresh_topology(pos)
    timing_started_at = profile_clock()
    wns, tns, ws, ts = model.timing_obj(pos)
    timing_finished_at = profile_clock()
    timing_loss = model._timing_loss(wns, tns, ws, ts)
    loss_finished_at = profile_clock()
    z_grad = torch.autograd.grad(
        timing_loss,
        z_param,
        retain_graph=False,
        create_graph=False,
        allow_unused=False,
    )[0]
    autograd_finished_at = profile_clock()
    if z_grad is None or not bool(torch.isfinite(z_grad).all()):
        finite = (
            torch.zeros_like(z_param, dtype=torch.bool)
            if z_grad is None
            else torch.isfinite(z_grad)
        )
        bad_indices = torch.nonzero(~finite, as_tuple=False).flatten()
        sample_indices = bad_indices[:32]
        sample_rows = []
        for index in sample_indices.detach().cpu().tolist():
            row = {"state_index": int(index)}
            for key, values in (
                ("segment_id", getattr(state, "segment_ids", None)),
                ("net_id", getattr(state, "segment_net_id", None)),
                ("parent_node_id", getattr(state, "parent_node_id", None)),
                ("child_node_id", getattr(state, "child_node_id", None)),
                ("z", getattr(state, "z_param", None)),
                ("bsu_index", getattr(state, "bsu_index_param", None)),
            ):
                if torch.is_tensor(values) and int(index) < int(values.numel()):
                    value = values[int(index)].detach().cpu().item()
                    row[key] = float(value) if values.dtype.is_floating_point else int(value)
            if z_grad is not None:
                row["gradient"] = float(z_grad[int(index)].detach().cpu().item())
            for key, attr in (
                ("per_size_input_cap", "buffer_segment_count_per_size_input_cap"),
                ("per_size_delay", "buffer_segment_count_per_size_delay"),
                (
                    "per_size_output_slew",
                    "buffer_segment_count_per_size_output_slew",
                ),
            ):
                values = getattr(context.data_collections, attr, None)
                if torch.is_tensor(values) and int(values.shape[0]) > 0:
                    row_index = 0 if int(values.shape[0]) == 1 else int(index)
                    if row_index >= int(values.shape[0]):
                        continue
                    tensor_row = values[row_index].detach().float().cpu()
                    row[key] = [float(value) for value in tensor_row.tolist()]
                    row[f"{key}_finite"] = bool(torch.isfinite(tensor_row).all())
            if hasattr(state, "segment_record"):
                record = state.segment_record(
                    int(index), materialization_kind="selected"
                )
                for key in (
                    "parent_x_dbu",
                    "parent_y_dbu",
                    "child_x_dbu",
                    "child_y_dbu",
                ):
                    if record.get(key) is not None:
                        row[key] = int(record[key])
                edge_rc = dict(record.get("edge_rc", {}) or {})
                if edge_rc:
                    row["edge_resistance"] = float(edge_rc.get("r", 0.0))
                    row["edge_capacitance"] = float(edge_rc.get("c", 0.0))
            sample_rows.append(row)
        failure = {
            "artifact": "segment_count_nonfinite_gradient_failure",
            "artifact_version": 2,
            "event_id": int(event["event_id"]),
            "frame": str(frame),
            "gradient_count": int(z_param.numel()),
            "nonfinite_count": int(bad_indices.numel()),
            "samples": sample_rows,
        }
        failure_path = os.path.join(
            str(params.result_dir),
            f"{params.design_name()}_segment_count_nonfinite_gradient.json",
        )
        os.makedirs(os.path.dirname(os.path.abspath(failure_path)), exist_ok=True)
        with open(failure_path, "w", encoding="utf-8") as stream:
            json.dump(failure, stream, indent=2, sort_keys=True)
            stream.write("\n")
        raise RuntimeError(
            "refreshed segment z gradient is missing or non-finite: "
            f"count={int(bad_indices.numel())}, artifact={failure_path}"
        )
    if not torch.equal(pos.detach(), pos_before):
        raise RuntimeError("refreshed segment timing pass changed placement")
    z_param.grad = z_grad.detach().clone()
    model._timing_geometry_cache = None
    expected_generation = expected_topology_generation
    if expected_generation is None:
        expected_generation = (
            coordinator.segment_milestones.frozen_topology_generation
        )
    topology = context.data_collections.buffering_timing_topology
    actual_generation = int(topology.get("topology_generation", 0) or 0)
    if actual_generation != int(expected_generation):
        raise RuntimeError("refreshed segment timing changed frozen topology generation")
    state_digest = context.digest(
        (
            ("pos", pos),
            ("size_logits", size_logits),
            ("inst_cell_id", context.data_collections.inst_cell_id),
            ("z_param", z_param),
        )
    )
    finished_at = profile_clock()
    result = {
        "status": "ready",
        "event_id": int(event["event_id"]),
        "iteration": int(event["iteration"]),
        "frame": str(frame),
        "state_digest": state_digest,
        "topology_generation": actual_generation,
        "runtime_cell_state_generation": int(
            getattr(context.data_collections, "runtime_cell_state_generation", 0)
            or 0
        ),
        "wns": float(wns),
        "tns": float(tns),
        "timing_loss": float(timing_loss.detach().cpu().item()),
        "gradient_norm": float(z_grad.detach().float().norm().cpu().item()),
        "gradient_nonzero_count": int((z_grad != 0.0).sum().cpu().item()),
        "runtime_ms": float((finished_at - started_at) * 1000.0),
    }
    if profile_enabled:
        result["runtime_profile"] = {
            "enabled": True,
            "synchronized": bool(pos.is_cuda),
            "timing_obj_ms": float(
                (timing_finished_at - timing_started_at) * 1000.0
            ),
            "timing_loss_ms": float(
                (loss_finished_at - timing_finished_at) * 1000.0
            ),
            "autograd_ms": float(
                (autograd_finished_at - loss_finished_at) * 1000.0
            ),
            "post_gradient_validation_ms": float(
                (finished_at - autograd_finished_at) * 1000.0
            ),
            "total_ms": float((finished_at - started_at) * 1000.0),
        }
    return result


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
