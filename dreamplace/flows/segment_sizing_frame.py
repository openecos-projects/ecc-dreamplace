"""ECC segment sizing frame owner; live state is supplied per call."""
from dataclasses import dataclass
from typing import Callable
import hashlib
import time
import torch

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
            getattr(params, "timing_objective_lane", "timing_only")
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

