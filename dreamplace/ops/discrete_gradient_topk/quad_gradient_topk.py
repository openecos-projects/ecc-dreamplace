import math
from collections import defaultdict
from dataclasses import dataclass

import torch

from .discrete_gradient_topk import (
    _logit_grad_to_real_size_grad,
    sizes_to_logits,
)

ACTION_NAMES = ("size_up", "size_down", "vt_slowdown", "vt_speedup")
SIZE_UP = 0
SIZE_DOWN = 1
VT_SLOWDOWN = 2
VT_SPEEDUP = 3


@dataclass(frozen=True)
class QuadCandidateTable:
    """One-step legal neighbors for every library cell."""

    target_cell_ids: torch.Tensor

    def to(self, *, device):
        return QuadCandidateTable(target_cell_ids=self.target_cell_ids.to(device=device))


def build_quad_candidate_table(flat_libcell_info, flat_libcell_leakage=None):
    """Build the four legal (size-/VT-axis) neighbors for each cell.

    A VT move is legal only when the destination has exactly the same timing
    coordinate, so a single action never changes both axes. When leakage is
    available, it defines the slow-to-fast VT order without relying on backend
    VT integer labels.
    """

    info = flat_libcell_info.detach().cpu()
    leakage = None if flat_libcell_leakage is None else flat_libcell_leakage.detach().cpu()
    num_cells = int(info.shape[0])
    target_cell_ids = torch.full((num_cells, len(ACTION_NAMES)), -1, dtype=torch.long)
    families = defaultdict(list)
    for cell_id in range(num_cells):
        main_id = int(round(float(info[cell_id, 1])))
        size = float(info[cell_id, 2])
        vt = int(round(float(info[cell_id, 3])))
        families[main_id].append((size, vt, cell_id))

    for main_id, rows in families.items():
        pair_to_cell = {}
        for size, vt, cell_id in rows:
            key = (size, vt)
            if key in pair_to_cell:
                raise ValueError(
                    "quad-gradient candidate lattice has duplicate "
                    f"(timing_coordinate, vt) entries for main_id={main_id}: {key}"
                )
            pair_to_cell[key] = cell_id

        by_vt = defaultdict(list)
        by_size = defaultdict(list)
        for size, vt, cell_id in rows:
            by_vt[vt].append((size, cell_id))
            by_size[size].append((vt, cell_id))

        for size_rows in by_vt.values():
            size_rows.sort(key=lambda item: (item[0], item[1]))
            for index, (_size, cell_id) in enumerate(size_rows):
                if index + 1 < len(size_rows):
                    target_cell_ids[cell_id, SIZE_UP] = size_rows[index + 1][1]
                if index > 0:
                    target_cell_ids[cell_id, SIZE_DOWN] = size_rows[index - 1][1]

        for vt_rows in by_size.values():
            if leakage is None:
                vt_rows.sort(key=lambda item: (item[0], item[1]))
            else:
                vt_rows.sort(
                    key=lambda item: (
                        float(leakage[item[1]]),
                        item[0],
                        item[1],
                    )
                )
            for index, (_vt, cell_id) in enumerate(vt_rows):
                if index > 0:
                    target_cell_ids[cell_id, VT_SLOWDOWN] = vt_rows[index - 1][1]
                if index + 1 < len(vt_rows):
                    target_cell_ids[cell_id, VT_SPEEDUP] = vt_rows[index + 1][1]

    return QuadCandidateTable(target_cell_ids=target_cell_ids)


def _select_count(num_candidates, percent):
    if num_candidates <= 0 or percent <= 0.0:
        return 0
    return min(
        num_candidates,
        max(1, int(math.ceil(num_candidates * float(percent) / 100.0))),
    )


def _select_direction(scores, valid, action, percent):
    selected = torch.zeros_like(valid)
    improving = valid[:, action] & (scores[:, action] < 0.0)
    count = _select_count(int(torch.count_nonzero(improving).item()), percent)
    if count <= 0:
        return selected
    values = torch.where(
        improving,
        scores[:, action],
        torch.full_like(scores[:, action], float("inf")),
    )
    rows = torch.argsort(values, stable=True)[:count]
    selected[rows, action] = True
    return selected


def _select_vt(scores, valid, percent):
    selected = torch.zeros_like(valid)
    vt_scores = torch.where(
        valid[:, VT_SLOWDOWN : VT_SPEEDUP + 1],
        scores[:, VT_SLOWDOWN : VT_SPEEDUP + 1],
        torch.full_like(scores[:, VT_SLOWDOWN : VT_SPEEDUP + 1], float("inf")),
    )
    best_scores, best_local_actions = torch.min(vt_scores, dim=1)
    improving = torch.isfinite(best_scores) & (best_scores < 0.0)
    count = _select_count(int(torch.count_nonzero(improving).item()), percent)
    if count <= 0:
        return selected
    ranked = torch.where(improving, best_scores, torch.full_like(best_scores, float("inf")))
    rows = torch.argsort(ranked, stable=True)[:count]
    selected[rows, best_local_actions[rows] + VT_SLOWDOWN] = True
    return selected


def _select_shared_budget(scores, valid, sizeable, percent):
    if not math.isfinite(percent) or not 0.0 <= percent <= 100.0:
        raise ValueError("shared budget percentage must be in [0, 100]")
    budget = _select_count(int(torch.count_nonzero(sizeable).item()), percent)
    improving = valid & torch.isfinite(scores) & (scores < 0.0)
    best_scores, best_actions = torch.min(
        torch.where(improving, scores, float("inf")), dim=1
    )
    ranked = torch.argsort(best_scores, stable=True)[:budget]
    ranked = ranked[torch.isfinite(best_scores[ranked])]
    selected = torch.zeros_like(valid)
    selected[ranked, best_actions[ranked]] = True
    return selected, budget


def _target_vt_logits(vt_ids, num_vts, *, dtype, device, eps):
    logits = torch.full(
        (int(vt_ids.numel()), num_vts),
        math.log(eps),
        dtype=dtype,
        device=device,
    )
    logits.scatter_(1, vt_ids.reshape(-1, 1), 0.0)
    return logits


def apply_discrete_quad_gradient_topk_update(
    *,
    size_logits,
    size_grad,
    vt_logits,
    vt_grad,
    inst_size_lower,
    inst_size_upper,
    inst_vt_mask,
    inst_is_sizeable,
    inst_cell_id,
    flat_libcell_info,
    flat_libcell_leakage=None,
    size_up_percent,
    size_down_percent,
    vt_percent,
    shared_budget_percent=None,
    blocked_instances=None,
    candidate_table=None,
    eps=1.0e-6,
    size_parameterization="logits",
):
    """Select and commit one legal size/VT neighbor per chosen instance.

    Size actions use the existing real-size Taylor score. VT actions recover a
    directional derivative in probability space from the softmax-logit
    gradient, avoiding the near-one-hot softmax attenuation.
    With real_size, the legacy size_logits argument and first return value
    contain real size coordinates, and size_grad is already in that coordinate.
    """
    if size_parameterization not in ("logits", "real_size"):
        raise ValueError("quad-gradient size parameterization must be logits or real_size")

    if candidate_table is None:
        candidate_table = build_quad_candidate_table(
            flat_libcell_info,
            flat_libcell_leakage,
        )
        cache_mode = "rebuilt"
    else:
        cache_mode = "cached"
    candidate_table = candidate_table.to(device=inst_cell_id.device)

    num_instances = int(inst_cell_id.numel())
    if vt_logits.ndim != 2 or int(vt_logits.shape[0]) != num_instances:
        raise ValueError("vt_logits must have shape [num_instances, num_vt_classes]")
    if vt_grad.shape != vt_logits.shape:
        raise ValueError("vt_grad shape must match vt_logits")
    if inst_vt_mask.shape != vt_logits.shape:
        raise ValueError("inst_vt_mask shape must match vt_logits")

    current_cell_ids = inst_cell_id.to(device=flat_libcell_info.device).long()
    sizeable = inst_is_sizeable.to(device=current_cell_ids.device).bool()
    valid_current_cell = (current_cell_ids >= 0) & (current_cell_ids < flat_libcell_info.shape[0])
    if bool(torch.any(sizeable & ~valid_current_cell)):
        raise ValueError(
            "quad-gradient requires a valid current cell id for every sizeable instance"
        )
    safe_current_cell_ids = torch.clamp(
        current_cell_ids,
        min=0,
        max=max(int(flat_libcell_info.shape[0]) - 1, 0),
    )
    targets = candidate_table.target_cell_ids[safe_current_cell_ids]
    valid = targets >= 0
    valid &= valid_current_cell.unsqueeze(1)
    safe_targets = torch.clamp(targets, min=0)
    target_info = flat_libcell_info[safe_targets]
    target_sizes = target_info[:, :, 2].to(dtype=size_logits.dtype)
    target_vts = target_info[:, :, 3].long()

    sizeable = sizeable.to(device=valid.device)
    valid &= sizeable.unsqueeze(1)
    valid &= torch.isfinite(size_grad).to(device=valid.device).unsqueeze(1)
    valid[:, VT_SLOWDOWN : VT_SPEEDUP + 1] &= torch.isfinite(vt_grad).all(dim=1).unsqueeze(1)

    size_real_grad = (
        size_grad if size_parameterization == "real_size"
        else _logit_grad_to_real_size_grad(
            size_logits=size_logits,
            size_grad=size_grad,
            lower=inst_size_lower,
            upper=inst_size_upper,
        )
    )
    current_sizes = flat_libcell_info[safe_current_cell_ids, 2].to(dtype=size_logits.dtype)
    scores = torch.full(
        (num_instances, len(ACTION_NAMES)),
        float("inf"),
        dtype=size_logits.dtype,
        device=size_logits.device,
    )
    scores[:, SIZE_UP : SIZE_DOWN + 1] = size_real_grad.unsqueeze(1) * (
        target_sizes[:, SIZE_UP : SIZE_DOWN + 1] - current_sizes.unsqueeze(1)
    )

    vt_mask = inst_vt_mask.to(device=vt_logits.device).bool()
    vt_probs = torch.softmax(vt_logits.masked_fill(~vt_mask, -1.0e9), dim=1)
    probability_grad = vt_grad / torch.clamp(vt_probs, min=eps)
    probability_baseline = torch.sum(probability_grad * vt_probs, dim=1)
    vt_target_ids = target_vts[:, VT_SLOWDOWN : VT_SPEEDUP + 1]
    safe_vt_target_ids = torch.clamp(vt_target_ids, min=0)
    valid[:, VT_SLOWDOWN : VT_SPEEDUP + 1] &= torch.gather(
        vt_mask,
        1,
        safe_vt_target_ids,
    )
    vt_target_grad = torch.gather(probability_grad, 1, safe_vt_target_ids)
    scores[:, VT_SLOWDOWN : VT_SPEEDUP + 1] = vt_target_grad - probability_baseline.unsqueeze(1)
    scores = torch.where(valid, scores, torch.full_like(scores, float("inf")))

    if shared_budget_percent is None:
        selected = _select_direction(scores, valid, SIZE_UP, size_up_percent)
        selected |= _select_direction(scores, valid, SIZE_DOWN, size_down_percent)
        selected |= _select_vt(scores, valid, vt_percent)
        budget = None
        num_blocked = 0
        num_prevented_moves = 0
        num_replacement_actions = 0
    else:
        percent = float(shared_budget_percent)
        if blocked_instances is not None:
            baseline_selected, _ = _select_shared_budget(scores, valid, sizeable, percent)
            blocked = blocked_instances.to(device=valid.device, dtype=torch.bool)
            if blocked.shape != sizeable.shape:
                raise ValueError("blocked_instances must match instance shape")
            num_blocked = int(blocked.sum().item())
            num_prevented_moves = int(
                (valid[blocked] & (scores[blocked] < 0)).any(dim=1).sum().item()
            )
            valid &= ~blocked.unsqueeze(1)
        else:
            num_blocked = 0
            num_prevented_moves = 0
        selected, budget = _select_shared_budget(scores, valid, sizeable, percent)
        num_replacement_actions = (
            int((selected.any(dim=1) & ~baseline_selected.any(dim=1)).sum().item())
            if blocked_instances is not None else 0
        )

    selected_scores = torch.where(selected, scores, torch.full_like(scores, float("inf")))
    best_scores, best_actions = torch.min(selected_scores, dim=1)
    changed = torch.isfinite(best_scores)
    changed_rows = torch.nonzero(changed, as_tuple=False).flatten()
    changed_actions = best_actions[changed_rows]
    changed_target_cells = targets[changed_rows, changed_actions]
    changed_target_sizes = flat_libcell_info[changed_target_cells, 2].to(dtype=size_logits.dtype)
    changed_target_vts = flat_libcell_info[changed_target_cells, 3].long()

    next_size_logits = size_logits.detach().clone()
    next_vt_logits = vt_logits.detach().clone()
    if int(changed_rows.numel()) > 0:
        next_size_logits[changed_rows] = (
            changed_target_sizes if size_parameterization == "real_size"
            else sizes_to_logits(
                changed_target_sizes,
                inst_size_lower[changed_rows],
                inst_size_upper[changed_rows],
                eps,
            )
        )
        next_vt_logits[changed_rows] = _target_vt_logits(
            changed_target_vts,
            int(vt_logits.shape[1]),
            dtype=vt_logits.dtype,
            device=vt_logits.device,
            eps=eps,
        )

    action_names = [ACTION_NAMES[int(action)] for action in changed_actions.detach().cpu().tolist()]
    applied_instance_ids = [int(value) for value in changed_rows.detach().cpu().tolist()]
    applied_cell_ids = [int(value) for value in changed_target_cells.detach().cpu().tolist()]
    applied_sizes = [float(value) for value in changed_target_sizes.detach().cpu().tolist()]
    applied_vts = [int(value) for value in changed_target_vts.detach().cpu().tolist()]
    applied_scores = [float(value) for value in best_scores[changed].detach().cpu().tolist()]
    applied_axis_deltas = [
        1 if action in ("size_up", "vt_speedup") else -1 for action in action_names
    ]
    selected_items = [
        {
            "instance_id": inst_id,
            "target_cell_id": cell_id,
            "target_size": size,
            "target_vt": vt,
            "action": action,
            "predicted_delta_obj": score,
            "predicted_improvement": -score,
        }
        for inst_id, cell_id, size, vt, action, score in zip(
            applied_instance_ids,
            applied_cell_ids,
            applied_sizes,
            applied_vts,
            action_names,
            applied_scores,
            strict=True,
        )
    ]
    action_counts = {
        action: sum(selected_action == action for selected_action in action_names)
        for action in ACTION_NAMES
    }
    finite_candidates = valid & torch.isfinite(scores)
    improving_candidates = finite_candidates & (scores < 0.0)
    summary = {
        "enabled": True,
        "mode": "discrete_gradient_topk",
        "ranking_mode": "quad_gradient_taylor",
        "preserve_vt": False,
        "step_mode": "direct_one_step",
        "up_percent": float(size_up_percent),
        "down_percent": float(size_down_percent),
        "vt_percent": float(vt_percent),
        "selection_policy": "shared_cell_budget" if budget is not None else "per_direction_percent",
        "shared_budget_percent": None if budget is None else float(shared_budget_percent),
        "shared_budget_cells": budget,
        "num_blocked_instances": num_blocked,
        "num_prevented_moves": num_prevented_moves,
        "num_replacement_actions": num_replacement_actions,
        "num_sizeable_instances": int(torch.count_nonzero(sizeable).item()),
        "num_up_selected": action_counts["size_up"],
        "num_down_selected": action_counts["size_down"],
        "num_up_applied": action_counts["size_up"],
        "num_down_applied": action_counts["size_down"],
        "num_vt_selected": action_counts["vt_slowdown"] + action_counts["vt_speedup"],
        "num_vt_applied": action_counts["vt_slowdown"] + action_counts["vt_speedup"],
        "num_vt_slowdown_applied": action_counts["vt_slowdown"],
        "num_vt_speedup_applied": action_counts["vt_speedup"],
        "num_changed_instances": int(changed_rows.numel()),
        "num_candidate_moves": int(torch.count_nonzero(finite_candidates).item()),
        "num_improving_candidates": int(torch.count_nonzero(improving_candidates).item()),
        "applied_instance_ids": applied_instance_ids,
        "applied_cell_ids": applied_cell_ids,
        "applied_sizes": applied_sizes,
        "applied_vts": applied_vts,
        "applied_actions": action_names,
        "applied_legal_index_deltas": applied_axis_deltas,
        "selected_scores": selected_items,
        "candidate_table_cache_mode": cache_mode,
        "ranking": {
            "mode": "quad_gradient_taylor",
            "selection_backend": "tensor",
            "score_coordinate": "real_size_and_vt_probability",
            "delta_value_name": "delta_real_size_or_vt_onehot",
            "direction_coordinate": "size_and_vt_axes",
            "num_candidate_moves": int(torch.count_nonzero(finite_candidates).item()),
            "num_improving_candidates": int(torch.count_nonzero(improving_candidates).item()),
            "num_non_improving_selected": 0,
            "predicted_delta_obj_min": (
                float(scores[finite_candidates].min().item())
                if bool(torch.any(finite_candidates).item())
                else None
            ),
            "predicted_delta_obj_mean": (
                float(scores[finite_candidates].mean().item())
                if bool(torch.any(finite_candidates).item())
                else None
            ),
            "predicted_delta_obj_max": (
                float(scores[finite_candidates].max().item())
                if bool(torch.any(finite_candidates).item())
                else None
            ),
        },
        "step": {
            "direct_step_num_instances": int(changed_rows.numel()),
            "mean_abs_legal_index_delta": 1.0 if int(changed_rows.numel()) else 0.0,
            "max_abs_legal_index_delta": 1 if int(changed_rows.numel()) else 0,
        },
        "size": {
            "mean_after": float(
                (
                    next_size_logits if size_parameterization == "real_size" else (
                        inst_size_lower
                        + torch.sigmoid(next_size_logits) * (inst_size_upper - inst_size_lower)
                    )
                )
                .double()
                .mean()
                .item()
            )
            if num_instances
            else None,
        },
    }
    return next_size_logits, next_vt_logits, summary


def apply_quad_gradient_from_data_collections(
    *,
    data_collections,
    size_logits,
    size_grad,
    vt_logits,
    vt_grad,
    size_up_percent,
    size_down_percent,
    vt_percent,
    candidate_table,
    shared_budget_percent=None,
    blocked_instances=None,
    size_parameterization="logits",
):
    required_tensors = {
        "size_grad": size_grad,
        "vt_logits": vt_logits,
        "vt_grad": vt_grad,
        "inst_size_lower": getattr(data_collections, "inst_size_lower", None),
        "inst_size_upper": getattr(data_collections, "inst_size_upper", None),
        "inst_vt_mask": getattr(data_collections, "inst_vt_mask", None),
        "inst_is_sizeable": getattr(data_collections, "inst_is_sizeable", None),
        "inst_cell_id": getattr(data_collections, "inst_cell_id", None),
        "flat_libcell_info": getattr(data_collections, "flat_libcell_info", None),
        "flat_libcell_leakage": getattr(data_collections, "flat_libcell_leakage", None),
    }
    missing_tensors = [name for name, value in required_tensors.items() if value is None]
    if missing_tensors:
        raise RuntimeError(
            "discrete quad-gradient requires missing tensors: " + ", ".join(missing_tensors)
        )

    return apply_discrete_quad_gradient_topk_update(
        size_logits=size_logits.detach(),
        size_grad=size_grad.to(
            device=size_logits.device,
            dtype=size_logits.dtype,
        ),
        vt_logits=vt_logits.detach(),
        vt_grad=vt_grad.to(
            device=vt_logits.device,
            dtype=vt_logits.dtype,
        ),
        inst_size_lower=data_collections.inst_size_lower.to(
            device=size_logits.device,
            dtype=size_logits.dtype,
        ),
        inst_size_upper=data_collections.inst_size_upper.to(
            device=size_logits.device,
            dtype=size_logits.dtype,
        ),
        inst_vt_mask=data_collections.inst_vt_mask.to(device=vt_logits.device),
        inst_is_sizeable=data_collections.inst_is_sizeable.to(device=size_logits.device),
        inst_cell_id=data_collections.inst_cell_id.to(device=size_logits.device),
        flat_libcell_info=data_collections.flat_libcell_info.to(
            device=size_logits.device,
            dtype=size_logits.dtype,
        ),
        flat_libcell_leakage=data_collections.flat_libcell_leakage.to(
            device=size_logits.device,
            dtype=size_logits.dtype,
        ),
        size_up_percent=float(size_up_percent),
        size_down_percent=float(size_down_percent),
        vt_percent=float(vt_percent),
        shared_budget_percent=shared_budget_percent,
        blocked_instances=blocked_instances,
        candidate_table=candidate_table,
        size_parameterization=size_parameterization,
    )
