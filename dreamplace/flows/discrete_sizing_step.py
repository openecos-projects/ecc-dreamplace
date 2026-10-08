"""ECC discrete sizing step owner; live state is supplied per call."""
from dataclasses import dataclass
from typing import Callable
import logging
import torch
from dreamplace.ops.discrete_gradient_topk import apply_discrete_gradient_topk_update, apply_quad_gradient_from_data_collections, build_quad_candidate_table
from dreamplace.ops.discrete_gradient_topk.oscillation import CellOscillationState

UNSET = object()

@dataclass
class DiscreteSizingStepContext:
    cache: Callable
    cache_key: Callable
    candidate_table: Callable
    config: Callable
    data_collections: object
    oscillation: Callable
    publish_candidate_cache: Callable
    publish_oscillation: Callable
    select: Callable
    sizing_mode: Callable
    skip_artifact: Callable
    sync_cells: Callable
    write_artifact: Callable
    summary: object = UNSET
    topk_summary: object = UNSET

def apply_discrete_sizing_dynamics(context, params, before_snapshot, iteration, total_iterations, config, summary, size_logits, before_logits, phase, progress, raw_delta_max_abs):
    data_collections = getattr(context, "data_collections", None)
    size_grad = None if size_logits.grad is None else size_logits.grad.detach().clone()
    vt_logits = getattr(data_collections, "vt_logits", None)
    before_vt_logits = (
        None if before_snapshot is None else before_snapshot.get("vt_logits")
    )
    vt_grad = (
        None
        if vt_logits is None or vt_logits.grad is None
        else vt_logits.grad.detach().clone()
    )
    vt_percent = float(config["discrete_gradient_topk_vt_percent"])
    shared_budget = config["discrete_gradient_topk_shared_budget"]
    if shared_budget and context.sizing_mode(params) != "size_only":
        raise ValueError("shared cell budget is currently supported only in size_only")
    quad_gradient_enabled = (
        (vt_percent > 0.0 or shared_budget)
        and vt_logits is not None
        and vt_logits.ndim == 2
        and (int(vt_logits.shape[1]) > 1 or shared_budget)
    )
    if shared_budget and not quad_gradient_enabled:
        raise RuntimeError("shared cell budget requires VT metadata")
    oscillation_state = None
    blocked_instances = None
    if shared_budget:
        oscillation_state = context.oscillation()
        if oscillation_state is None:
            oscillation_state = CellOscillationState()
            context.publish_oscillation(oscillation_state)
        if config["discrete_gradient_topk_oscillation_veto"]:
            blocked_instances = oscillation_state.blocked(
                phase="timing",
                iteration=iteration,
                count=int(data_collections.inst_cell_id.numel()),
                device=size_logits.device,
            )
    with torch.no_grad():
        size_logits.copy_(before_logits.detach())
        if quad_gradient_enabled:
            if vt_logits is None or before_vt_logits is None or (
                vt_grad is None and int(vt_logits.shape[1]) > 1
            ):
                raise RuntimeError(
                    "discrete quad-gradient requires vt_logits "
                    "and a populated vt gradient"
                )
            if vt_grad is None:
                vt_grad = torch.zeros_like(vt_logits)
            vt_logits.copy_(before_vt_logits.detach())
    if quad_gradient_enabled:
        cache_key = (
            "quad_gradient",
            str(size_logits.device),
            str(size_logits.dtype),
            tuple(int(dim) for dim in data_collections.flat_libcell_info.shape),
        )
        if context.cache_key() != cache_key:
            context.publish_candidate_cache(
                build_quad_candidate_table(
                    data_collections.flat_libcell_info,
                    data_collections.flat_libcell_leakage,
                ),
                cache_key,
            )
        target_logits, target_vt_logits, topk_summary = (
            apply_quad_gradient_from_data_collections(
                data_collections=data_collections,
                size_logits=before_logits.detach().to(
                    device=size_logits.device,
                    dtype=size_logits.dtype,
                ),
                size_grad=size_grad,
                vt_logits=before_vt_logits.detach().to(
                    device=vt_logits.device,
                    dtype=vt_logits.dtype,
                ),
                vt_grad=vt_grad,
                size_up_percent=config[
                    "discrete_gradient_topk_up_percent"
                ],
                size_down_percent=config[
                    "discrete_gradient_topk_down_percent"
                ],
                vt_percent=vt_percent,
                shared_budget_percent=(
                    config["discrete_gradient_topk_shared_budget_percent"]
                    if shared_budget
                    else None
                ),
                blocked_instances=blocked_instances,
                candidate_table=context.cache(),
            )
        )
    else:
        target_vt_logits = None
        target_logits, _target_sizes, topk_summary, _instance_candidate_table = (
            context.select(
                params=params,
                size_logits=before_logits.detach().to(
                    device=size_logits.device,
                    dtype=size_logits.dtype,
                ),
                size_grad=size_grad,
                policy={
                    "up_percent": config[
                        "discrete_gradient_topk_up_percent"
                    ],
                    "down_percent": config[
                        "discrete_gradient_topk_down_percent"
                    ],
                    "preserve_vt": config[
                        "discrete_gradient_topk_preserve_vt"
                    ],
                    "ranking_mode": config[
                        "discrete_gradient_topk_ranking_mode"
                    ],
                    "step_mode": config[
                        "discrete_gradient_topk_step_mode"
                    ],
                },
            )
        )
    with torch.no_grad():
        size_logits.copy_(target_logits.to(device=size_logits.device, dtype=size_logits.dtype))
        if quad_gradient_enabled:
            vt_logits.copy_(
                target_vt_logits.to(
                    device=vt_logits.device,
                    dtype=vt_logits.dtype,
                )
            )
        elif (
            config["discrete_gradient_topk_preserve_vt"]
            and vt_logits is not None
            and before_vt_logits is not None
        ):
            vt_logits.copy_(before_vt_logits.detach())
    applied_delta = size_logits.detach() - before_logits.detach()
    applied_delta_max_abs = (
        0.0
        if applied_delta.numel() == 0
        else float(applied_delta.abs().max().item())
    )
    topk_summary.update(
        {
            "iteration": int(iteration),
            "total_iterations": int(total_iterations),
            "phase": phase,
            "phase_progress": progress,
            "phase_lr_scale": 0.0,
            "raw_continuous_delta_max_abs": raw_delta_max_abs,
            "applied_delta_max_abs": applied_delta_max_abs,
            "config": {
                "up_percent": config["discrete_gradient_topk_up_percent"],
                "down_percent": config["discrete_gradient_topk_down_percent"],
                "vt_percent": vt_percent,
                "shared_budget": shared_budget,
                "oscillation_veto": config["discrete_gradient_topk_oscillation_veto"],
                "shared_budget_percent": (
                    config["discrete_gradient_topk_shared_budget_percent"]
                    if shared_budget
                    else None
                ),
                "preserve_vt": (
                    False
                    if quad_gradient_enabled
                    else config["discrete_gradient_topk_preserve_vt"]
                ),
                "ranking_mode": (
                    "quad_gradient_taylor"
                    if quad_gradient_enabled
                    else config["discrete_gradient_topk_ranking_mode"]
                ),
                "step_mode": config["discrete_gradient_topk_step_mode"],
                "lambda_update_mode": config["discrete_gradient_topk_lambda_update_mode"],
                "lambda_update_scale": config["discrete_gradient_topk_lambda_update_scale"],
                "lambda_min": config["discrete_gradient_topk_lambda_min"],
                "lambda_max": config["discrete_gradient_topk_lambda_max"],
            },
        }
    )
    logging.info(
        "discrete sizing update iteration=%d upsize_cells=%d "
        "downsize_cells=%d vt_cells=%d changed_cells=%d "
        "selected_up=%d selected_down=%d",
        int(iteration),
        int(topk_summary.get("num_up_applied", 0) or 0),
        int(topk_summary.get("num_down_applied", 0) or 0),
        int(topk_summary.get("num_vt_applied", 0) or 0),
        int(topk_summary.get("num_changed_instances", 0) or 0),
        int(topk_summary.get("num_up_selected", 0) or 0),
        int(topk_summary.get("num_down_selected", 0) or 0),
    )
    previous_cell_ids = []
    if oscillation_state is not None and topk_summary["applied_instance_ids"]:
        previous_cell_ids = data_collections.inst_cell_id[
            topk_summary["applied_instance_ids"]
        ].detach().cpu().tolist()
    topk_summary.update(context.sync_cells(params, topk_summary))
    if oscillation_state is not None:
        oscillation_state.record(
            phase="timing",
            iteration=iteration,
            instance_ids=topk_summary["applied_instance_ids"],
            previous_ids=previous_cell_ids,
            target_ids=topk_summary["applied_cell_ids"],
        )
        topk_summary["reversals"] = oscillation_state.reversals
        topk_summary["repeated_reversals"] = oscillation_state.repeated_reversals
        logging.info(
            "discrete oscillation iteration=%d frozen_cells=%d "
            "prevented_moves=%d replacement_actions=%d reversals=%d repeated_reversals=%d",
            int(iteration),
            topk_summary["num_blocked_instances"],
            topk_summary["num_prevented_moves"],
            topk_summary["num_replacement_actions"],
            oscillation_state.reversals,
            oscillation_state.repeated_reversals,
        )
    if context.skip_artifact(params):
        topk_summary["artifact_write_skipped_by_fast_loop"] = True
        artifact_info = None
        context.topk_summary = dict(topk_summary)
    else:
        topk_summary["artifact_write_skipped_by_fast_loop"] = False
        artifact_info = context.write_artifact(params, topk_summary)
    if artifact_info is not None:
        topk_summary["latest_path"] = artifact_info["latest_path"]
        topk_summary["trace_path"] = artifact_info["trace_path"]
    summary.update(
        {
            "enabled": True,
            "phase": phase,
            "phase_progress": progress,
            "phase_lr_scale": 0.0,
            "raw_delta_max_abs": raw_delta_max_abs,
            "trust_ratio": 1.0,
            "applied_delta_max_abs": applied_delta_max_abs,
            "discrete_gradient_topk": topk_summary,
        }
    )
    for key, value in topk_summary.items():
        if key not in ("ranking", "step", "size", "config"):
            summary[f"discrete_gradient_topk_{key}"] = value
    ranking_summary = topk_summary.get("ranking") or {}
    for key, value in ranking_summary.items():
        summary[f"discrete_gradient_topk_{key}"] = value
    step_summary = topk_summary.get("step") or {}
    for key, value in step_summary.items():
        summary[f"discrete_gradient_topk_{key}"] = value
    size_summary = topk_summary.get("size") or {}
    for key, value in size_summary.items():
        summary[f"discrete_gradient_topk_size_{key}"] = value
    context.summary = summary
    return summary


def select_discrete_gradient_topk(
    context,
    *,
    params,
    size_logits,
    size_grad,
    policy,
):
    data_collections = getattr(context, "data_collections", None)
    ranking_mode = str(policy["ranking_mode"])
    required_tensors = {
        "size_grad": size_grad,
        "inst_size_lower": getattr(data_collections, "inst_size_lower", None),
        "inst_size_upper": getattr(data_collections, "inst_size_upper", None),
        "inst_is_sizeable": getattr(data_collections, "inst_is_sizeable", None),
        "inst_cell_id": getattr(data_collections, "inst_cell_id", None),
        "flat_libcell_info": getattr(data_collections, "flat_libcell_info", None),
        "flat_libcell_leakage": (
            getattr(data_collections, "flat_libcell_leakage", None)
            if ranking_mode == "leakage_taylor"
            else torch.tensor(0)
        ),
    }
    missing_tensors = [
        name for name, value in required_tensors.items() if value is None
    ]
    if missing_tensors:
        raise RuntimeError(
            "discrete gradient top-k requires missing tensors: "
            + ", ".join(missing_tensors)
        )

    config = context.config(params)
    cache_config = dict(config)
    cache_config["discrete_gradient_topk_preserve_vt"] = bool(
        policy["preserve_vt"]
    )
    candidate_table = context.candidate_table(
        cache_config,
        data_collections,
        size_logits.device,
        size_logits.dtype,
    )
    target_logits, target_sizes, summary = apply_discrete_gradient_topk_update(
        size_logits=size_logits.detach(),
        size_grad=size_grad.to(
            device=size_logits.device,
            dtype=size_logits.dtype,
        ),
        inst_size_lower=data_collections.inst_size_lower.to(
            device=size_logits.device,
            dtype=size_logits.dtype,
        ),
        inst_size_upper=data_collections.inst_size_upper.to(
            device=size_logits.device,
            dtype=size_logits.dtype,
        ),
        inst_is_sizeable=data_collections.inst_is_sizeable.to(
            device=size_logits.device
        ).bool(),
        inst_cell_id=data_collections.inst_cell_id.to(
            device=size_logits.device
        ).long(),
        flat_libcell_info=data_collections.flat_libcell_info.to(
            device=size_logits.device,
            dtype=size_logits.dtype,
        ),
        flat_libcell_leakage=(
            data_collections.flat_libcell_leakage.to(
                device=size_logits.device,
                dtype=size_logits.dtype,
            )
            if ranking_mode == "leakage_taylor"
            else None
        ),
        up_percent=float(policy["up_percent"]),
        down_percent=float(policy["down_percent"]),
        preserve_vt=bool(policy["preserve_vt"]),
        ranking_mode=ranking_mode,
        step_mode=str(policy["step_mode"]),
        lambda_update_mode=config["discrete_gradient_topk_lambda_update_mode"],
        lambda_update_scale=config["discrete_gradient_topk_lambda_update_scale"],
        lambda_min=config["discrete_gradient_topk_lambda_min"],
        lambda_max=config["discrete_gradient_topk_lambda_max"],
        instance_candidate_table=candidate_table,
    )
    return target_logits, target_sizes, summary, candidate_table

