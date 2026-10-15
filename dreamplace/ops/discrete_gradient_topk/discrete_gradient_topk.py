import math
from collections import defaultdict

import torch


RANKING_MODES = {
    "raw_gradient",
    "taylor_delta_logit",
    "real_size_taylor",
    "timing_coordinate_taylor",
    "leakage_taylor",
}
STEP_MODES = {"direct_one_step", "lagrangian_relaxation"}
SELECTION_BACKENDS = {"python", "tensor"}


def sizes_to_logits(target_sizes, lower, upper, eps=1e-6):
    denom = torch.clamp(upper - lower, min=torch.finfo(target_sizes.dtype).eps)
    norm = (target_sizes - lower) / denom
    norm = torch.clamp(norm, min=eps, max=1.0 - eps)
    return torch.log(norm / (1.0 - norm))


def _logit_grad_to_real_size_grad(size_logits, size_grad, lower, upper, eps=1e-12):
    size_norm = torch.sigmoid(size_logits)
    derivative = size_norm * (1.0 - size_norm) * (upper - lower)
    safe_derivative = torch.where(
        derivative.abs() > eps,
        derivative,
        torch.full_like(derivative, eps),
    )
    real_size_grad = size_grad / safe_derivative
    return torch.where(
        torch.isfinite(real_size_grad),
        real_size_grad,
        torch.zeros_like(real_size_grad),
    )


def _canonical_ranking_mode(ranking_mode):
    if ranking_mode == "timing_coordinate_taylor":
        return "real_size_taylor"
    return ranking_mode


def build_log_leakage_coord(flat_libcell_leakage, eps=None):
    leakage = flat_libcell_leakage.detach()
    if eps is None:
        positive = leakage[torch.isfinite(leakage) & (leakage > 0)]
        if int(positive.numel()) > 0:
            eps = max(float(torch.min(positive).item()) * 1.0e-6, 1.0e-30)
        else:
            eps = 1.0e-30
    eps_tensor = torch.as_tensor(eps, dtype=leakage.dtype, device=leakage.device)
    safe_leakage = torch.where(
        torch.isfinite(leakage) & (leakage >= 0),
        leakage,
        torch.zeros_like(leakage),
    )
    return torch.log(torch.clamp(safe_leakage + eps_tensor, min=eps_tensor)), float(eps)


def _stats(values):
    if not values:
        return None, None, None
    tensor = torch.stack([torch.as_tensor(value, dtype=torch.float64) for value in values])
    return float(torch.min(tensor)), float(torch.mean(tensor)), float(torch.max(tensor))


def _tensor_stats(values, mask):
    if values.numel() == 0 or not bool(torch.any(mask).item()):
        return None, None, None
    selected = values[mask].detach().double()
    return (
        float(torch.min(selected).item()),
        float(torch.mean(selected).item()),
        float(torch.max(selected).item()),
    )


def build_family_candidate_table(flat_libcell_info, preserve_vt=True):
    info = flat_libcell_info.detach().cpu()
    families = defaultdict(list)
    for cell_id in range(info.shape[0]):
        main_id = int(round(float(info[cell_id, 1])))
        # Column 2 is the unified timing coordinate. Historical "size" names
        # are kept so the continuous sizing path has one source of truth.
        size = float(info[cell_id, 2])
        vt = int(round(float(info[cell_id, 3]))) if info.shape[1] > 3 else 0
        key = (main_id, vt) if preserve_vt else (main_id, None)
        families[key].append((size, cell_id, vt, main_id))

    table = {}
    for key, rows in families.items():
        rows = sorted(rows, key=lambda item: (item[0], item[1]))
        table[key] = {
            "sizes": torch.tensor([row[0] for row in rows], dtype=flat_libcell_info.dtype),
            "cell_ids": torch.tensor([row[1] for row in rows], dtype=torch.long),
            "vts": torch.tensor([row[2] for row in rows], dtype=torch.long),
            "main_ids": torch.tensor([row[3] for row in rows], dtype=torch.long),
        }
    return table


def build_instance_candidate_table(
    *,
    family_candidate_table=None,
    family_table=None,
    inst_cell_id,
    flat_libcell_info,
    inst_size_lower,
    inst_size_upper,
    eps=1e-6,
    preserve_vt=True,
):
    if family_candidate_table is None:
        family_candidate_table = family_table
    if family_candidate_table is None:
        raise ValueError("family_candidate_table is required")
    device = inst_cell_id.device
    dtype = inst_size_lower.dtype
    flat_info_cpu = flat_libcell_info.detach().cpu()
    num_instances = int(inst_cell_id.numel())
    max_candidates = max((int(row["sizes"].numel()) for row in family_candidate_table.values()), default=1)
    legal_sizes = torch.zeros((num_instances, max_candidates), dtype=dtype, device=device)
    legal_logits = torch.zeros((num_instances, max_candidates), dtype=dtype, device=device)
    legal_cell_ids = torch.full((num_instances, max_candidates), -1, dtype=torch.long, device=device)
    candidate_mask = torch.zeros((num_instances, max_candidates), dtype=torch.bool, device=device)
    current_legal_index = torch.full((num_instances,), -1, dtype=torch.long, device=device)

    inst_cell_cpu = inst_cell_id.detach().cpu().long()
    for inst_idx in range(num_instances):
        cell_id = int(inst_cell_cpu[inst_idx])
        main_id = int(round(float(flat_info_cpu[cell_id, 1])))
        vt = int(round(float(flat_info_cpu[cell_id, 3]))) if flat_info_cpu.shape[1] > 3 else 0
        key = (main_id, vt) if preserve_vt else (main_id, None)
        family = family_candidate_table.get(key)
        if family is None:
            continue
        sizes = family["sizes"].to(device=device, dtype=dtype)
        cell_ids = family["cell_ids"].to(device=device)
        count = int(sizes.numel())
        legal_sizes[inst_idx, :count] = sizes
        legal_cell_ids[inst_idx, :count] = cell_ids
        candidate_mask[inst_idx, :count] = True
        lower = inst_size_lower[inst_idx].expand(count)
        upper = inst_size_upper[inst_idx].expand(count)
        legal_logits[inst_idx, :count] = sizes_to_logits(sizes, lower, upper, eps)
        matches = torch.nonzero(cell_ids == cell_id, as_tuple=False).flatten()
        if int(matches.numel()) > 0:
            current_legal_index[inst_idx] = matches[0].to(device)

    return {
        "legal_sizes": legal_sizes,
        "legal_cell_ids": legal_cell_ids,
        "legal_logits": legal_logits,
        "candidate_mask": candidate_mask,
        "current_legal_index": current_legal_index,
    }


def _refresh_current_legal_index(instance_table, inst_cell_id):
    legal_cell_ids = instance_table["legal_cell_ids"]
    candidate_mask = instance_table["candidate_mask"]
    matches = (legal_cell_ids == inst_cell_id.to(legal_cell_ids.device).long().unsqueeze(1)) & candidate_mask
    any_match = torch.any(matches, dim=1)
    current_legal_index = torch.argmax(matches.to(torch.long), dim=1)
    current_legal_index = torch.where(
        any_match,
        current_legal_index,
        torch.full_like(current_legal_index, -1),
    )
    refreshed = dict(instance_table)
    refreshed["current_legal_index"] = current_legal_index.to(inst_cell_id.device)
    return refreshed


def build_discrete_gradient_topk_candidate_cache(
    *,
    inst_cell_id,
    flat_libcell_info,
    inst_size_lower,
    inst_size_upper,
    eps=1e-6,
    preserve_vt=True,
):
    family_table = build_family_candidate_table(flat_libcell_info, preserve_vt=preserve_vt)
    return build_instance_candidate_table(
        family_candidate_table=family_table,
        inst_cell_id=inst_cell_id,
        flat_libcell_info=flat_libcell_info,
        inst_size_lower=inst_size_lower,
        inst_size_upper=inst_size_upper,
        eps=eps,
        preserve_vt=preserve_vt,
    )


def _select_count(num_candidates, percent):
    if num_candidates <= 0 or percent <= 0.0:
        return 0
    return min(num_candidates, max(1, int(math.ceil(num_candidates * float(percent) / 100.0))))


def _build_summary(
    *,
    mode,
    up_percent,
    down_percent,
    preserve_vt,
    step_mode,
    num_sizeable_instances,
    num_up_eligible,
    num_down_eligible,
    up_selected,
    down_selected,
    target_sizes,
    before_sizes,
    candidate_scores,
    selected_scores,
    applied_moves,
    candidate_stats=None,
    selection_backend="python",
):
    selected_predicted_values = [item["predicted_delta_obj"] for item in selected_scores]
    if candidate_stats is None:
        predicted_values = [item["predicted_delta_obj"] for item in candidate_scores]
        delta_abs = [abs(item["delta_logit"]) for item in candidate_scores]
        pred_min, pred_mean, pred_max = _stats(predicted_values)
        _delta_min, delta_mean, delta_max = _stats(delta_abs)
        num_candidate_moves = int(len(candidate_scores))
        num_improving_candidates = int(
            sum(1 for item in candidate_scores if item["predicted_delta_obj"] < 0.0)
        )
        num_non_improving_candidates = int(
            sum(1 for item in candidate_scores if item["predicted_delta_obj"] >= 0.0)
        )
        num_up_improving_instances = len(
            {
                int(item["instance_id"])
                for item in candidate_scores
                if item["direction"] == "up"
                and item["predicted_delta_obj"] < 0.0
            }
        )
        num_down_improving_instances = len(
            {
                int(item["instance_id"])
                for item in candidate_scores
                if item["direction"] == "down"
                and item["predicted_delta_obj"] < 0.0
            }
        )
    else:
        pred_min = candidate_stats["predicted_delta_obj_min"]
        pred_mean = candidate_stats["predicted_delta_obj_mean"]
        pred_max = candidate_stats["predicted_delta_obj_max"]
        delta_mean = candidate_stats["delta_logit_mean_abs"]
        delta_max = candidate_stats["delta_logit_max_abs"]
        num_candidate_moves = int(candidate_stats["num_candidate_moves"])
        num_improving_candidates = int(candidate_stats["num_improving_candidates"])
        num_non_improving_candidates = int(candidate_stats["num_non_improving_candidates"])
        num_up_improving_instances = int(
            candidate_stats["num_up_improving_instances"]
        )
        num_down_improving_instances = int(
            candidate_stats["num_down_improving_instances"]
        )
    score_coordinate = (
        None if candidate_stats is None else candidate_stats.get("score_coordinate")
    )
    delta_value_name = (
        None if candidate_stats is None else candidate_stats.get("delta_value_name")
    )
    direction_coordinate = (
        None if candidate_stats is None else candidate_stats.get("direction_coordinate")
    )
    real_size_grad_min = (
        None if candidate_stats is None else candidate_stats.get("real_size_grad_min")
    )
    real_size_grad_mean = (
        None if candidate_stats is None else candidate_stats.get("real_size_grad_mean")
    )
    real_size_grad_max = (
        None if candidate_stats is None else candidate_stats.get("real_size_grad_max")
    )
    delta_real_size_mean_abs = (
        None if candidate_stats is None else candidate_stats.get("delta_real_size_mean_abs")
    )
    delta_real_size_max_abs = (
        None if candidate_stats is None else candidate_stats.get("delta_real_size_max_abs")
    )
    leakage_coord_mode = (
        None if candidate_stats is None else candidate_stats.get("leakage_coord_mode")
    )
    leakage_eps = None if candidate_stats is None else candidate_stats.get("leakage_eps")
    delta_log_leakage_mean_abs = (
        None if candidate_stats is None else candidate_stats.get("delta_log_leakage_mean_abs")
    )
    delta_log_leakage_max_abs = (
        None if candidate_stats is None else candidate_stats.get("delta_log_leakage_max_abs")
    )
    num_invalid_leakage_candidates = (
        0 if candidate_stats is None else int(candidate_stats.get("num_invalid_leakage_candidates") or 0)
    )
    num_equal_leakage_candidates = (
        0 if candidate_stats is None else int(candidate_stats.get("num_equal_leakage_candidates") or 0)
    )
    selected_up_delta_log_leakage = [
        item["delta_log_leakage"]
        for item in up_selected
        if "delta_log_leakage" in item
    ]
    selected_down_delta_log_leakage = [
        item["delta_log_leakage"]
        for item in down_selected
        if "delta_log_leakage" in item
    ]
    _, selected_up_mean_delta_log_leakage, _ = _stats(selected_up_delta_log_leakage)
    _, selected_down_mean_delta_log_leakage, _ = _stats(selected_down_delta_log_leakage)
    selected_min, _, _ = _stats(selected_predicted_values)
    changed = target_sizes != before_sizes
    size_after = target_sizes.detach().double()
    applied_index_deltas = [abs(int(item["legal_index_delta"])) for item in applied_moves]
    num_up_applied = sum(
        1 for item in applied_moves if int(item["legal_index_delta"]) > 0
    )
    num_down_applied = sum(
        1 for item in applied_moves if int(item["legal_index_delta"]) < 0
    )
    mean_abs_legal_index_delta = (
        float(sum(applied_index_deltas)) / float(len(applied_index_deltas))
        if applied_index_deltas
        else 0.0
    )
    max_abs_legal_index_delta = max(applied_index_deltas) if applied_index_deltas else 0
    return {
        "enabled": True,
        "mode": "discrete_gradient_topk",
        "up_percent": float(up_percent),
        "down_percent": float(down_percent),
        "preserve_vt": bool(preserve_vt),
        "ranking_mode": mode,
        "ranking_mode_note": (
            "real_size_taylor is a legacy mode name; the coordinate source is "
            "flat_libcell_info[:,2], which now stores timing_coordinate."
            if mode == "real_size_taylor"
            else None
        ),
        "step_mode": step_mode,
        "lambda_update_mode": "fixed_gradient_price",
        "lagrangian_relaxation_enabled": step_mode == "lagrangian_relaxation",
        "num_sizeable_instances": int(num_sizeable_instances),
        "num_up_eligible": int(num_up_eligible),
        "num_down_eligible": int(num_down_eligible),
        "num_up_improving_instances": int(num_up_improving_instances),
        "num_down_improving_instances": int(num_down_improving_instances),
        "num_up_selected": int(len(up_selected)),
        "num_down_selected": int(len(down_selected)),
        "num_up_applied": int(num_up_applied),
        "num_down_applied": int(num_down_applied),
        "num_changed_instances": int(torch.count_nonzero(changed).item()),
        "applied_instance_ids": [int(item["instance_id"]) for item in applied_moves],
        "applied_cell_ids": [int(item["target_cell_id"]) for item in applied_moves],
        "applied_sizes": [float(item["target_size"]) for item in applied_moves],
        "applied_legal_index_deltas": [int(item["legal_index_delta"]) for item in applied_moves],
        "selected_scores": list(selected_scores),
        "step": {
            "direct_step_num_instances": int(torch.count_nonzero(changed).item())
            if step_mode == "direct_one_step"
            else 0,
            "lagrangian_relaxation_step_num_instances": 0,
            "mean_abs_legal_index_delta": mean_abs_legal_index_delta,
            "max_abs_legal_index_delta": max_abs_legal_index_delta,
            "lambda_source": None,
            "lambda_update_scale": 1.0,
            "lambda_min": None,
            "lambda_mean": None,
            "lambda_max": None,
            "num_lambda_updated": 0,
            "candidate_cost_min": None,
            "candidate_cost_mean": None,
            "candidate_cost_max": None,
        },
        "ranking": {
            "mode": mode,
            "selection_backend": selection_backend,
            "score_coordinate": score_coordinate or "logit",
            "delta_value_name": delta_value_name or "delta_logit",
            "direction_coordinate": direction_coordinate or "legal_index",
            "num_candidate_moves": num_candidate_moves,
            "num_improving_candidates": num_improving_candidates,
            "num_non_improving_candidates": num_non_improving_candidates,
            "num_up_improving_instances": int(num_up_improving_instances),
            "num_down_improving_instances": int(num_down_improving_instances),
            "num_non_improving_selected": int(
                sum(1 for item in selected_scores if item["predicted_delta_obj"] >= 0.0)
            ),
            "predicted_delta_obj_min": pred_min,
            "predicted_delta_obj_mean": pred_mean,
            "predicted_delta_obj_max": pred_max,
            "predicted_improvement_threshold": (
                -float(selected_min) if selected_min is not None else None
            ),
            "delta_logit_mean_abs": delta_mean,
            "delta_logit_max_abs": delta_max,
            "real_size_grad_min": real_size_grad_min,
            "real_size_grad_mean": real_size_grad_mean,
            "real_size_grad_max": real_size_grad_max,
            "delta_real_size_mean_abs": delta_real_size_mean_abs,
            "delta_real_size_max_abs": delta_real_size_max_abs,
            "leakage_coord_mode": leakage_coord_mode,
            "leakage_eps": leakage_eps,
            "delta_log_leakage_mean_abs": delta_log_leakage_mean_abs,
            "delta_log_leakage_max_abs": delta_log_leakage_max_abs,
            "num_invalid_leakage_candidates": num_invalid_leakage_candidates,
            "num_equal_leakage_candidates": num_equal_leakage_candidates,
            "selected_up_mean_delta_log_leakage": selected_up_mean_delta_log_leakage,
            "selected_down_mean_delta_log_leakage": selected_down_mean_delta_log_leakage,
        },
        "size": {
            "mean_before": float(before_sizes.detach().double().mean().item())
            if before_sizes.numel()
            else None,
            "mean_after": float(size_after.mean().item()) if target_sizes.numel() else None,
            "min_after": float(size_after.min().item()) if target_sizes.numel() else None,
            "max_after": float(size_after.max().item()) if target_sizes.numel() else None,
        },
    }


def _current_sizes_from_table(instance_table):
    legal_sizes = instance_table["legal_sizes"]
    current_idx = instance_table["current_legal_index"]
    safe_idx = torch.clamp(current_idx, min=0)
    return legal_sizes[torch.arange(legal_sizes.shape[0], device=legal_sizes.device), safe_idx]


def _candidate_dict(
    *,
    inst_idx,
    direction,
    target_index,
    target_cell_id,
    current_logit,
    target_logit,
    gradient,
    instance_table=None,
):
    delta_logit = target_logit - current_logit
    predicted_delta_obj = gradient * delta_logit
    current_legal_index = None
    current_cell_id = None
    if instance_table is not None:
        current_legal_index = int(instance_table["current_legal_index"][inst_idx])
        if current_legal_index >= 0:
            current_cell_id = int(
                instance_table["legal_cell_ids"][inst_idx, current_legal_index]
            )
    item = {
        "instance_id": int(inst_idx),
        "direction": direction,
        "current_legal_index": current_legal_index,
        "current_cell_id": current_cell_id,
        "target_legal_index": int(target_index),
        "target_cell_id": int(target_cell_id),
        "current_logit": float(current_logit),
        "target_logit": float(target_logit),
        "delta_logit": float(delta_logit),
        "gradient": float(gradient),
        "predicted_delta_obj": float(predicted_delta_obj),
        "predicted_improvement": float(-predicted_delta_obj),
    }
    if instance_table is not None:
        candidate_mask = instance_table["candidate_mask"][inst_idx].detach().cpu().bool()
        valid_count = int(torch.count_nonzero(candidate_mask).item())
        legal_cell_ids = instance_table["legal_cell_ids"][inst_idx, :valid_count].detach().cpu().long()
        legal_sizes = instance_table["legal_sizes"][inst_idx, :valid_count].detach().cpu()
        item["legal_cell_id_candidates"] = [int(value) for value in legal_cell_ids.tolist()]
        item["legal_size_idx_candidates"] = list(range(valid_count))
        item["legal_size_candidates"] = [float(value) for value in legal_sizes.tolist()]
    return item


def _enumerate_candidates(
    *,
    size_grad,
    inst_is_sizeable,
    instance_table,
    ranking_mode,
):
    legal_logits = instance_table["legal_logits"]
    legal_cell_ids = instance_table["legal_cell_ids"]
    candidate_mask = instance_table["candidate_mask"]
    current_index = instance_table["current_legal_index"]
    candidates = []
    up_eligible = set()
    down_eligible = set()
    for inst_idx in range(int(size_grad.numel())):
        if not bool(inst_is_sizeable[inst_idx]):
            continue
        grad = size_grad[inst_idx]
        if not bool(torch.isfinite(grad)):
            continue
        cur_idx = int(current_index[inst_idx])
        if cur_idx < 0:
            continue
        valid_count = int(torch.count_nonzero(candidate_mask[inst_idx]).item())
        if valid_count <= 1:
            continue
        current_logit = legal_logits[inst_idx, cur_idx]
        if cur_idx < valid_count - 1:
            up_eligible.add(inst_idx)
        if cur_idx > 0:
            down_eligible.add(inst_idx)
        target_indices = range(valid_count)
        if ranking_mode == "taylor_delta_logit":
            adjacent = []
            if cur_idx > 0:
                adjacent.append(cur_idx - 1)
            if cur_idx < valid_count - 1:
                adjacent.append(cur_idx + 1)
            target_indices = adjacent
        for target_idx in target_indices:
            if target_idx == cur_idx:
                continue
            direction = "up" if target_idx > cur_idx else "down"
            target_logit = legal_logits[inst_idx, target_idx]
            candidates.append(
                _candidate_dict(
                    inst_idx=inst_idx,
                    direction=direction,
                    target_index=target_idx,
                    target_cell_id=legal_cell_ids[inst_idx, target_idx],
                    current_logit=current_logit,
                    target_logit=target_logit,
                    gradient=grad,
                    instance_table=instance_table,
                )
            )
    return candidates, up_eligible, down_eligible


def _legacy_raw_gradient_select(
    *,
    size_grad,
    up_eligible,
    down_eligible,
    up_percent,
    down_percent,
    instance_table,
):
    selected = []
    up_indices = sorted(up_eligible, key=lambda idx: float(size_grad[idx]))
    up_indices = [idx for idx in up_indices if float(size_grad[idx]) < 0.0]
    down_indices = sorted(down_eligible, key=lambda idx: float(size_grad[idx]), reverse=True)
    down_indices = [idx for idx in down_indices if float(size_grad[idx]) > 0.0]
    up_indices = up_indices[: _select_count(len(up_indices), up_percent)]
    down_indices = down_indices[: _select_count(len(down_indices), down_percent)]
    for inst_idx in up_indices:
        cur_idx = int(instance_table["current_legal_index"][inst_idx])
        item = _candidate_dict(
            inst_idx=inst_idx,
            direction="up",
            target_index=cur_idx + 1,
            target_cell_id=instance_table["legal_cell_ids"][inst_idx, cur_idx + 1],
            current_logit=instance_table["legal_logits"][inst_idx, cur_idx],
            target_logit=instance_table["legal_logits"][inst_idx, cur_idx + 1],
            gradient=size_grad[inst_idx],
            instance_table=instance_table,
        )
        selected.append(item)
    for inst_idx in down_indices:
        cur_idx = int(instance_table["current_legal_index"][inst_idx])
        item = _candidate_dict(
            inst_idx=inst_idx,
            direction="down",
            target_index=cur_idx - 1,
            target_cell_id=instance_table["legal_cell_ids"][inst_idx, cur_idx - 1],
            current_logit=instance_table["legal_logits"][inst_idx, cur_idx],
            target_logit=instance_table["legal_logits"][inst_idx, cur_idx - 1],
            gradient=size_grad[inst_idx],
            instance_table=instance_table,
        )
        selected.append(item)
    return selected


def _taylor_select(candidates, up_percent, down_percent):
    best_by_instance_and_direction = {}
    for item in candidates:
        if item["predicted_delta_obj"] >= 0.0:
            continue
        key = (item["instance_id"], item["direction"])
        previous = best_by_instance_and_direction.get(key)
        if previous is None or item["predicted_delta_obj"] < previous["predicted_delta_obj"]:
            best_by_instance_and_direction[key] = item
    up = [item for item in best_by_instance_and_direction.values() if item["direction"] == "up"]
    down = [item for item in best_by_instance_and_direction.values() if item["direction"] == "down"]
    up.sort(key=lambda item: item["predicted_delta_obj"])
    down.sort(key=lambda item: item["predicted_delta_obj"])
    return (
        up[: _select_count(len(up), up_percent)]
        + down[: _select_count(len(down), down_percent)]
    )


def _build_taylor_tensor_candidates(
    *,
    size_logits,
    size_grad,
    inst_size_lower,
    inst_size_upper,
    inst_is_sizeable,
    instance_table,
    ranking_mode,
    step_mode=None,
    flat_libcell_leakage=None,
):
    legal_logits = instance_table["legal_logits"]
    candidate_mask = instance_table["candidate_mask"]
    current_index = instance_table["current_legal_index"].to(device=legal_logits.device)
    num_instances, max_candidates = legal_logits.shape
    legal_indices = torch.arange(max_candidates, device=legal_logits.device).view(1, -1)
    row_indices = torch.arange(num_instances, device=legal_logits.device)
    has_current = current_index >= 0
    safe_current_index = torch.clamp(current_index, min=0)
    current_logits = legal_logits[row_indices, safe_current_index]
    delta_logits = legal_logits - current_logits.unsqueeze(1)
    legal_sizes = instance_table["legal_sizes"]
    legal_cell_ids = instance_table["legal_cell_ids"].to(device=legal_logits.device)
    current_sizes = legal_sizes[row_indices, safe_current_index]
    delta_real_sizes = legal_sizes - current_sizes.unsqueeze(1)
    leakage_coord = None
    delta_log_leakage = None
    current_leakage_coord = None
    leakage_valid = None
    current_leakage_valid = None
    if ranking_mode == "leakage_taylor":
        if flat_libcell_leakage is None:
            raise ValueError("leakage_taylor requires flat_libcell_leakage")
        flat_leakage = flat_libcell_leakage.to(device=legal_logits.device, dtype=legal_logits.dtype)
        if int(torch.max(torch.clamp(legal_cell_ids, min=0)).item()) >= int(flat_leakage.numel()):
            raise ValueError("leakage_taylor flat_libcell_leakage length does not cover legal_cell_ids")
        flat_leakage_valid = torch.isfinite(flat_leakage) & (flat_leakage > 0)
        log_leakage, leakage_eps = build_log_leakage_coord(flat_leakage)
        safe_cell_ids = torch.clamp(legal_cell_ids, min=0)
        leakage_coord = log_leakage[safe_cell_ids]
        leakage_valid = flat_leakage_valid[safe_cell_ids] & (legal_cell_ids >= 0)
        current_leakage_coord = leakage_coord[row_indices, safe_current_index]
        current_leakage_valid = leakage_valid[row_indices, safe_current_index] & has_current
        delta_log_leakage = leakage_coord - current_leakage_coord.unsqueeze(1)
    else:
        leakage_eps = None
    finite_grad = torch.isfinite(size_grad).to(device=legal_logits.device)
    valid_inst = (
        inst_is_sizeable.to(device=legal_logits.device).bool()
        & finite_grad
        & has_current
    )
    if ranking_mode == "leakage_taylor":
        valid_inst = valid_inst & current_leakage_valid
    not_current = legal_indices != safe_current_index.unsqueeze(1)
    valid_mask = candidate_mask & valid_inst.unsqueeze(1) & not_current
    invalid_leakage_mask = None
    equal_leakage_mask = None
    if ranking_mode == "leakage_taylor":
        invalid_leakage_mask = (
            candidate_mask
            & not_current
            & (~leakage_valid | ~current_leakage_valid.unsqueeze(1))
        )
        valid_mask = valid_mask & leakage_valid
        equal_leakage_mask = valid_mask & (delta_log_leakage == 0)
    if step_mode == "direct_one_step" and ranking_mode == "leakage_taylor":
        inf = torch.full_like(delta_log_leakage, float("inf"))
        positive_delta = valid_mask & (delta_log_leakage > 0)
        negative_delta = valid_mask & (delta_log_leakage < 0)
        nearest_up_delta, nearest_up_idx = torch.min(
            torch.where(positive_delta, delta_log_leakage, inf),
            dim=1,
        )
        nearest_down_delta, nearest_down_idx = torch.min(
            torch.where(negative_delta, -delta_log_leakage, inf),
            dim=1,
        )
        leakage_one_step_mask = torch.zeros_like(valid_mask)
        has_nearest_up = torch.isfinite(nearest_up_delta)
        has_nearest_down = torch.isfinite(nearest_down_delta)
        leakage_one_step_mask[row_indices[has_nearest_up], nearest_up_idx[has_nearest_up]] = True
        leakage_one_step_mask[row_indices[has_nearest_down], nearest_down_idx[has_nearest_down]] = True
        valid_mask = valid_mask & leakage_one_step_mask
    elif ranking_mode == "taylor_delta_logit" or step_mode == "direct_one_step":
        valid_mask = valid_mask & (
            (legal_indices == (safe_current_index - 1).unsqueeze(1))
            | (legal_indices == (safe_current_index + 1).unsqueeze(1))
        )
    real_size_grad = None
    if ranking_mode == "real_size_taylor":
        real_size_grad = _logit_grad_to_real_size_grad(
            size_logits=size_logits.to(device=legal_logits.device, dtype=legal_logits.dtype),
            size_grad=size_grad.to(device=legal_logits.device, dtype=legal_logits.dtype),
            lower=inst_size_lower.to(device=legal_logits.device, dtype=legal_logits.dtype),
            upper=inst_size_upper.to(device=legal_logits.device, dtype=legal_logits.dtype),
        )
        predicted = real_size_grad.unsqueeze(1) * delta_real_sizes
    elif ranking_mode == "leakage_taylor":
        predicted = size_grad.to(device=legal_logits.device).unsqueeze(1) * delta_log_leakage
    else:
        predicted = size_grad.to(device=legal_logits.device).unsqueeze(1) * delta_logits
    improving_mask = valid_mask & (predicted < 0)
    if ranking_mode == "leakage_taylor":
        up_mask = valid_mask & (delta_log_leakage > 0)
        down_mask = valid_mask & (delta_log_leakage < 0)
        direction_coordinate = "log_leakage"
    else:
        up_mask = valid_mask & (legal_indices > safe_current_index.unsqueeze(1))
        down_mask = valid_mask & (legal_indices < safe_current_index.unsqueeze(1))
        direction_coordinate = "legal_index"
    return {
        "predicted": predicted,
        "delta_logits": delta_logits,
        "delta_real_sizes": delta_real_sizes,
        "delta_log_leakage": delta_log_leakage,
        "leakage_coord": leakage_coord,
        "leakage_eps": leakage_eps,
        "invalid_leakage_mask": invalid_leakage_mask,
        "equal_leakage_mask": equal_leakage_mask,
        "real_size_grad": real_size_grad,
        "score_coordinate": (
            "real_size"
            if ranking_mode == "real_size_taylor"
            else "log_leakage"
            if ranking_mode == "leakage_taylor"
            else "logit"
        ),
        "delta_value_name": (
            "delta_real_size"
            if ranking_mode == "real_size_taylor"
            else "delta_log_leakage"
            if ranking_mode == "leakage_taylor"
            else "delta_logit"
        ),
        "direction_coordinate": direction_coordinate,
        "valid_mask": valid_mask,
        "improving_mask": improving_mask,
        "up_mask": up_mask,
        "down_mask": down_mask,
        "current_index": current_index,
    }


def _tensor_candidate_stats(tensor_candidates):
    valid_mask = tensor_candidates["valid_mask"]
    improving_mask = tensor_candidates["improving_mask"]
    pred_min, pred_mean, pred_max = _tensor_stats(tensor_candidates["predicted"], valid_mask)
    _delta_min, delta_mean, delta_max = _tensor_stats(
        tensor_candidates["delta_logits"].abs(),
        valid_mask,
    )
    if tensor_candidates.get("real_size_grad") is not None:
        valid_inst = torch.any(valid_mask, dim=1)
        real_size_grad_min, real_size_grad_mean, real_size_grad_max = _tensor_stats(
            tensor_candidates["real_size_grad"],
            valid_inst,
        )
        _delta_real_size_min, delta_real_size_mean, delta_real_size_max = _tensor_stats(
            tensor_candidates["delta_real_sizes"].abs(),
            valid_mask,
        )
    else:
        real_size_grad_min = real_size_grad_mean = real_size_grad_max = None
        delta_real_size_mean = delta_real_size_max = None
    if tensor_candidates.get("delta_log_leakage") is not None:
        _delta_log_leakage_min, delta_log_leakage_mean, delta_log_leakage_max = _tensor_stats(
            tensor_candidates["delta_log_leakage"].abs(),
            valid_mask,
        )
        invalid_leakage_mask = tensor_candidates.get("invalid_leakage_mask")
        equal_leakage_mask = tensor_candidates.get("equal_leakage_mask")
        num_invalid_leakage_candidates = (
            int(torch.count_nonzero(invalid_leakage_mask).item())
            if invalid_leakage_mask is not None
            else 0
        )
        num_equal_leakage_candidates = (
            int(torch.count_nonzero(equal_leakage_mask).item())
            if equal_leakage_mask is not None
            else 0
        )
    else:
        delta_log_leakage_mean = delta_log_leakage_max = None
        num_invalid_leakage_candidates = 0
        num_equal_leakage_candidates = 0
    num_candidate_moves = int(torch.count_nonzero(valid_mask).item())
    num_improving_candidates = int(torch.count_nonzero(improving_mask).item())
    num_up_improving_instances = int(
        torch.count_nonzero(
            torch.any(improving_mask & tensor_candidates["up_mask"], dim=1)
        ).item()
    )
    num_down_improving_instances = int(
        torch.count_nonzero(
            torch.any(improving_mask & tensor_candidates["down_mask"], dim=1)
        ).item()
    )
    num_up_eligible, num_down_eligible = _eligible_counts_from_tensor_candidates(tensor_candidates)
    return {
        "num_up_eligible": num_up_eligible,
        "num_down_eligible": num_down_eligible,
        "num_candidate_moves": num_candidate_moves,
        "num_improving_candidates": num_improving_candidates,
        "num_non_improving_candidates": num_candidate_moves - num_improving_candidates,
        "num_up_improving_instances": num_up_improving_instances,
        "num_down_improving_instances": num_down_improving_instances,
        "predicted_delta_obj_min": pred_min,
        "predicted_delta_obj_mean": pred_mean,
        "predicted_delta_obj_max": pred_max,
        "delta_logit_mean_abs": delta_mean,
        "delta_logit_max_abs": delta_max,
        "score_coordinate": tensor_candidates.get("score_coordinate", "logit"),
        "delta_value_name": tensor_candidates.get("delta_value_name", "delta_logit"),
        "direction_coordinate": tensor_candidates.get("direction_coordinate", "legal_index"),
        "real_size_grad_min": real_size_grad_min,
        "real_size_grad_mean": real_size_grad_mean,
        "real_size_grad_max": real_size_grad_max,
        "delta_real_size_mean_abs": delta_real_size_mean,
        "delta_real_size_max_abs": delta_real_size_max,
        "leakage_coord_mode": "log_leakage" if tensor_candidates.get("delta_log_leakage") is not None else None,
        "leakage_eps": tensor_candidates.get("leakage_eps"),
        "delta_log_leakage_mean_abs": delta_log_leakage_mean,
        "delta_log_leakage_max_abs": delta_log_leakage_max,
        "num_invalid_leakage_candidates": num_invalid_leakage_candidates,
        "num_equal_leakage_candidates": num_equal_leakage_candidates,
    }


def _flatten_ranked_indices(values, mask):
    flat_indices = torch.nonzero(mask.reshape(-1), as_tuple=False).flatten()
    if int(flat_indices.numel()) == 0:
        return []
    flat_values = values.reshape(-1)[flat_indices]
    order = torch.argsort(flat_values, stable=True)
    return [int(item) for item in flat_indices[order].detach().cpu().tolist()]


def _tensor_selected_items_from_mask(
    *,
    selected_mask,
    tensor_candidates,
    instance_table,
    size_grad,
):
    legal_logits = instance_table["legal_logits"]
    legal_cell_ids = instance_table["legal_cell_ids"]
    num_instances, max_candidates = legal_logits.shape
    flat_indices = _flatten_ranked_indices(tensor_candidates["predicted"], selected_mask)
    selected = []
    for flat_idx in flat_indices:
        inst_idx = flat_idx // max_candidates
        target_idx = flat_idx % max_candidates
        cur_idx = int(tensor_candidates["current_index"][inst_idx])
        if tensor_candidates.get("delta_log_leakage") is not None:
            delta_log_leakage = float(tensor_candidates["delta_log_leakage"][inst_idx, target_idx])
            direction = "up" if delta_log_leakage > 0.0 else "down"
        else:
            direction = "up" if target_idx > cur_idx else "down"
        item = _candidate_dict(
            inst_idx=inst_idx,
            direction=direction,
            target_index=target_idx,
            target_cell_id=legal_cell_ids[inst_idx, target_idx],
            current_logit=legal_logits[inst_idx, cur_idx],
            target_logit=legal_logits[inst_idx, target_idx],
            gradient=size_grad[inst_idx],
            instance_table=instance_table,
        )
        if tensor_candidates.get("real_size_grad") is not None:
            item["gradient_coordinate"] = "real_size"
            item["real_size_gradient"] = float(tensor_candidates["real_size_grad"][inst_idx])
            item["delta_real_size"] = float(tensor_candidates["delta_real_sizes"][inst_idx, target_idx])
            item["predicted_delta_obj"] = float(tensor_candidates["predicted"][inst_idx, target_idx])
            item["predicted_improvement"] = float(-tensor_candidates["predicted"][inst_idx, target_idx])
        if tensor_candidates.get("delta_log_leakage") is not None:
            item["gradient_coordinate"] = "log_leakage"
            item["delta_log_leakage"] = float(tensor_candidates["delta_log_leakage"][inst_idx, target_idx])
            item["predicted_delta_obj"] = float(tensor_candidates["predicted"][inst_idx, target_idx])
            item["predicted_improvement"] = float(-tensor_candidates["predicted"][inst_idx, target_idx])
        selected.append(item)
    return selected


def _taylor_tensor_select_direction(
    *,
    tensor_candidates,
    instance_table,
    size_grad,
    direction_mask,
    percent,
):
    if float(percent) <= 0.0:
        return []
    pred = tensor_candidates["predicted"]
    improving_direction_mask = tensor_candidates["improving_mask"] & direction_mask
    if not bool(torch.any(improving_direction_mask).item()):
        return []
    masked_pred = torch.where(improving_direction_mask, pred, torch.full_like(pred, float("inf")))
    per_instance_values, per_instance_indices = torch.min(masked_pred, dim=1)
    has_best = torch.isfinite(per_instance_values)
    num_to_select = _select_count(int(torch.count_nonzero(has_best).item()), percent)
    if num_to_select <= 0:
        return []
    order = torch.argsort(per_instance_values, stable=True)
    chosen_rows = order[:num_to_select]
    chosen_rows = chosen_rows[per_instance_indices[chosen_rows] >= 0]
    selected_mask = torch.zeros_like(tensor_candidates["improving_mask"])
    selected_mask[chosen_rows, per_instance_indices[chosen_rows]] = True
    return _tensor_selected_items_from_mask(
        selected_mask=selected_mask,
        tensor_candidates=tensor_candidates,
        instance_table=instance_table,
        size_grad=size_grad,
    )


def _taylor_tensor_select(
    *,
    size_logits,
    size_grad,
    inst_size_lower,
    inst_size_upper,
    inst_is_sizeable,
    instance_table,
    ranking_mode,
    step_mode,
    up_percent,
    down_percent,
    flat_libcell_leakage=None,
):
    tensor_candidates = _build_taylor_tensor_candidates(
        size_logits=size_logits,
        size_grad=size_grad,
        inst_size_lower=inst_size_lower,
        inst_size_upper=inst_size_upper,
        inst_is_sizeable=inst_is_sizeable,
        instance_table=instance_table,
        ranking_mode=ranking_mode,
        step_mode=step_mode,
        flat_libcell_leakage=flat_libcell_leakage,
    )
    up_selected = _taylor_tensor_select_direction(
        tensor_candidates=tensor_candidates,
        instance_table=instance_table,
        size_grad=size_grad,
        direction_mask=tensor_candidates["up_mask"],
        percent=up_percent,
    ) if float(up_percent) > 0.0 else []
    down_selected = (
        _taylor_tensor_select_direction(
            tensor_candidates=tensor_candidates,
            instance_table=instance_table,
            size_grad=size_grad,
            direction_mask=tensor_candidates["down_mask"],
            percent=down_percent,
        )
        if float(down_percent) > 0.0
        else []
    )
    return up_selected + down_selected, _tensor_candidate_stats(tensor_candidates)


def _raw_gradient_tensor_select(
    *,
    size_logits,
    size_grad,
    inst_size_lower,
    inst_size_upper,
    inst_is_sizeable,
    instance_table,
    up_percent,
    down_percent,
):
    selection_candidates = _build_taylor_tensor_candidates(
        size_logits=size_logits,
        size_grad=size_grad,
        inst_size_lower=inst_size_lower,
        inst_size_upper=inst_size_upper,
        inst_is_sizeable=inst_is_sizeable,
        instance_table=instance_table,
        ranking_mode="taylor_delta_logit",
        step_mode="direct_one_step",
    )
    stats_candidates = _build_taylor_tensor_candidates(
        size_logits=size_logits,
        size_grad=size_grad,
        inst_size_lower=inst_size_lower,
        inst_size_upper=inst_size_upper,
        inst_is_sizeable=inst_is_sizeable,
        instance_table=instance_table,
        ranking_mode="taylor_delta_logit",
    )
    current_index = selection_candidates["current_index"]
    candidate_mask = instance_table["candidate_mask"]
    num_candidates = torch.count_nonzero(candidate_mask, dim=1)
    finite_grad = torch.isfinite(size_grad).to(device=current_index.device)
    valid = inst_is_sizeable.to(device=current_index.device).bool() & finite_grad & (current_index >= 0)
    up_rows = valid & (current_index < (num_candidates - 1)) & (size_grad.to(current_index.device) < 0)
    down_rows = valid & (current_index > 0) & (size_grad.to(current_index.device) > 0)

    selected_mask = torch.zeros_like(selection_candidates["valid_mask"])
    num_up = _select_count(int(torch.count_nonzero(up_rows).item()), up_percent)
    if num_up > 0:
        up_values = torch.where(up_rows, size_grad.to(current_index.device), torch.full_like(size_grad.to(current_index.device), float("inf")))
        up_order = torch.argsort(up_values, stable=True)[:num_up]
        selected_mask[up_order, current_index[up_order] + 1] = True
    num_down = (
        _select_count(int(torch.count_nonzero(down_rows).item()), down_percent)
        if float(down_percent) > 0.0
        else 0
    )
    if num_down > 0:
        down_values = torch.where(down_rows, size_grad.to(current_index.device), torch.full_like(size_grad.to(current_index.device), float("-inf")))
        down_order = torch.argsort(down_values, descending=True, stable=True)[:num_down]
        selected_mask[down_order, current_index[down_order] - 1] = True

    selected = _tensor_selected_items_from_mask(
        selected_mask=selected_mask,
        tensor_candidates=selection_candidates,
        instance_table=instance_table,
        size_grad=size_grad,
    )
    return selected, _tensor_candidate_stats(stats_candidates)


def _eligible_counts_from_instance_table(size_grad, inst_is_sizeable, instance_table):
    candidate_mask = instance_table["candidate_mask"]
    current_index = instance_table["current_legal_index"].to(device=candidate_mask.device)
    num_candidates = torch.count_nonzero(candidate_mask, dim=1)
    finite_grad = torch.isfinite(size_grad).to(device=candidate_mask.device)
    valid = inst_is_sizeable.to(device=candidate_mask.device).bool() & finite_grad & (current_index >= 0)
    up_eligible = valid & (current_index < (num_candidates - 1))
    down_eligible = valid & (current_index > 0)
    return int(torch.count_nonzero(up_eligible).item()), int(torch.count_nonzero(down_eligible).item())


def _eligible_counts_from_tensor_candidates(tensor_candidates):
    return (
        int(torch.count_nonzero(torch.any(tensor_candidates["up_mask"], dim=1)).item()),
        int(torch.count_nonzero(torch.any(tensor_candidates["down_mask"], dim=1)).item()),
    )


def apply_discrete_gradient_topk_update(
    *,
    size_logits,
    size_grad,
    inst_size_lower,
    inst_size_upper,
    inst_is_sizeable,
    inst_cell_id,
    flat_libcell_info,
    flat_libcell_leakage=None,
    up_percent,
    down_percent,
    preserve_vt=True,
    ranking_mode="real_size_taylor",
    step_mode="direct_one_step",
    lagrangian_relaxation_candidate_cost_fn=None,
    lagrangian_relaxation_config=None,
    arc_lambda=None,
    lambda_source=None,
    lambda_update_state=None,
    lambda_update_mode="fixed_gradient_price",
    lambda_update_scale=1.0,
    lambda_min=0.0,
    lambda_max=1.0e6,
    instance_candidate_table=None,
    selection_backend="tensor",
    eps=1e-6,
):
    if ranking_mode == "taylor_delta_candidate":
        raise ValueError("taylor_delta_candidate has been retired; use real_size_taylor")
    if ranking_mode not in RANKING_MODES:
        raise ValueError(f"invalid discrete gradient top-k ranking mode: {ranking_mode}")
    requested_ranking_mode = ranking_mode
    ranking_mode = _canonical_ranking_mode(ranking_mode)
    if step_mode not in STEP_MODES:
        raise ValueError(f"invalid discrete gradient top-k step mode: {step_mode}")
    if selection_backend not in SELECTION_BACKENDS:
        raise ValueError(f"invalid discrete gradient top-k selection backend: {selection_backend}")
    if ranking_mode in {"real_size_taylor", "leakage_taylor"} and selection_backend != "tensor":
        raise ValueError(f"{ranking_mode} ranking mode requires tensor selection backend")
    if ranking_mode == "leakage_taylor" and flat_libcell_leakage is None:
        raise ValueError("leakage_taylor requires flat_libcell_leakage")
    if step_mode == "lagrangian_relaxation":
        raise NotImplementedError(
            "lagrangian_relaxation step mode requires a candidate evaluator and is future work"
        )

    if instance_candidate_table is None:
        instance_table = build_discrete_gradient_topk_candidate_cache(
            inst_cell_id=inst_cell_id,
            flat_libcell_info=flat_libcell_info,
            inst_size_lower=inst_size_lower,
            inst_size_upper=inst_size_upper,
            eps=eps,
            preserve_vt=preserve_vt,
        )
        cache_mode = "rebuilt"
    else:
        instance_table = _refresh_current_legal_index(instance_candidate_table, inst_cell_id)
        cache_mode = "cached"
    before_sizes = _current_sizes_from_table(instance_table)
    target_sizes = before_sizes.clone()
    target_logits = size_logits.detach().clone()
    candidates = []
    candidate_stats = None
    if ranking_mode == "raw_gradient" and selection_backend == "python":
        candidates, up_eligible, down_eligible = _enumerate_candidates(
            size_grad=size_grad,
            inst_is_sizeable=inst_is_sizeable,
            instance_table=instance_table,
            ranking_mode=ranking_mode,
        )
        selected = _legacy_raw_gradient_select(
            size_grad=size_grad,
            up_eligible=up_eligible,
            down_eligible=down_eligible,
            up_percent=up_percent,
            down_percent=down_percent,
            instance_table=instance_table,
        )
        actual_selection_backend = "python"
    elif ranking_mode == "raw_gradient":
        num_up_eligible, num_down_eligible = _eligible_counts_from_instance_table(
            size_grad,
            inst_is_sizeable,
            instance_table,
        )
        selected, candidate_stats = _raw_gradient_tensor_select(
            size_logits=size_logits,
            size_grad=size_grad,
            inst_size_lower=inst_size_lower,
            inst_size_upper=inst_size_upper,
            inst_is_sizeable=inst_is_sizeable,
            instance_table=instance_table,
            up_percent=up_percent,
            down_percent=down_percent,
        )
        actual_selection_backend = "tensor"
    elif selection_backend == "tensor":
        selected, candidate_stats = _taylor_tensor_select(
            size_logits=size_logits,
            size_grad=size_grad,
            inst_size_lower=inst_size_lower,
            inst_size_upper=inst_size_upper,
            inst_is_sizeable=inst_is_sizeable,
            instance_table=instance_table,
            ranking_mode=ranking_mode,
            step_mode=step_mode,
            up_percent=up_percent,
            down_percent=down_percent,
            flat_libcell_leakage=flat_libcell_leakage,
        )
        num_up_eligible = candidate_stats.get("num_up_eligible", 0)
        num_down_eligible = candidate_stats.get("num_down_eligible", 0)
        actual_selection_backend = "tensor"
    else:
        effective_ranking_mode = (
            "taylor_delta_logit" if step_mode == "direct_one_step" else ranking_mode
        )
        candidates, up_eligible, down_eligible = _enumerate_candidates(
            size_grad=size_grad,
            inst_is_sizeable=inst_is_sizeable,
            instance_table=instance_table,
            ranking_mode=effective_ranking_mode,
        )
        num_up_eligible = len(up_eligible)
        num_down_eligible = len(down_eligible)
        selected = _taylor_select(candidates, up_percent, down_percent)
        actual_selection_backend = "python"
    if ranking_mode == "raw_gradient" and actual_selection_backend == "python":
        num_up_eligible = len(up_eligible)
        num_down_eligible = len(down_eligible)

    for item in selected:
        inst_idx = item["instance_id"]
        target_idx = item["target_legal_index"]
        cur_idx = int(instance_table["current_legal_index"][inst_idx])
        if step_mode == "direct_one_step":
            if target_idx > cur_idx:
                target_idx = cur_idx + 1
            elif target_idx < cur_idx:
                target_idx = cur_idx - 1
        target_sizes[inst_idx] = instance_table["legal_sizes"][inst_idx, target_idx]
        target_logits[inst_idx] = instance_table["legal_logits"][inst_idx, target_idx]
        item["applied_legal_index"] = int(target_idx)
        item["applied_cell_id"] = int(instance_table["legal_cell_ids"][inst_idx, target_idx])
        item["applied_size"] = float(instance_table["legal_sizes"][inst_idx, target_idx])
        item["applied_legal_index_delta"] = int(target_idx - cur_idx)

    up_selected = [item for item in selected if item["direction"] == "up"]
    down_selected = [item for item in selected if item["direction"] == "down"]
    applied_moves = [
        {
            "instance_id": item["instance_id"],
            "target_cell_id": item["applied_cell_id"],
            "target_size": item["applied_size"],
            "legal_index_delta": item["applied_legal_index_delta"],
        }
        for item in selected
        if int(item.get("applied_legal_index_delta", 0)) != 0
    ]
    summary = _build_summary(
        mode=requested_ranking_mode,
        up_percent=up_percent,
        down_percent=down_percent,
        preserve_vt=preserve_vt,
        step_mode=step_mode,
        num_sizeable_instances=int(torch.count_nonzero(inst_is_sizeable).item()),
        num_up_eligible=num_up_eligible,
        num_down_eligible=num_down_eligible,
        up_selected=up_selected,
        down_selected=down_selected,
        target_sizes=target_sizes,
        before_sizes=before_sizes,
        candidate_scores=candidates,
        selected_scores=selected,
        applied_moves=applied_moves,
        candidate_stats=candidate_stats,
        selection_backend=actual_selection_backend,
    )
    summary["candidate_table_cache_mode"] = cache_mode
    return target_logits.to(dtype=size_logits.dtype), target_sizes.to(dtype=inst_size_lower.dtype), summary
