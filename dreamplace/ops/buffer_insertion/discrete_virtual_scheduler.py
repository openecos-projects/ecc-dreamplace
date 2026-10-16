from dataclasses import dataclass
import math

import torch


ROUTE_B_AFFECTED_NETS_PER_ACTION = 1000


@dataclass(frozen=True)
class DiscreteVirtualSchedulerTransition:
    next_counts: torch.Tensor
    selected_segment_indices: torch.Tensor
    selected_net_indices: torch.Tensor
    selected_segment_ids: torch.Tensor
    selected_raw_grads: torch.Tensor
    selected_predicted_improvements: torch.Tensor
    affected_net_count: int
    eligible_net_count: int
    prefix_size: int
    positive_prefix_count: int

    @property
    def accepted_count(self):
        return int(self.selected_segment_indices.numel())


def take_transition_prefix(transition, *, counts, selected_count):
    """Return the same ranked transition restricted to its first actions."""
    selected_count = int(selected_count)
    if selected_count < 0 or selected_count > transition.accepted_count:
        raise ValueError("selected_count exceeds the scheduled transition")
    selected_segment_indices = transition.selected_segment_indices[:selected_count]
    next_counts = counts.detach().clone()
    next_counts[selected_segment_indices] += 1.0
    return DiscreteVirtualSchedulerTransition(
        next_counts=next_counts,
        selected_segment_indices=selected_segment_indices,
        selected_net_indices=transition.selected_net_indices[:selected_count],
        selected_segment_ids=transition.selected_segment_ids[:selected_count],
        selected_raw_grads=transition.selected_raw_grads[:selected_count],
        selected_predicted_improvements=(
            transition.selected_predicted_improvements[:selected_count]
        ),
        affected_net_count=transition.affected_net_count,
        eligible_net_count=transition.eligible_net_count,
        prefix_size=transition.prefix_size,
        positive_prefix_count=transition.positive_prefix_count,
    )


def transition_backtracking_sizes(selected_count):
    """Return deterministic largest-prefix-first trial sizes."""
    selected_count = int(selected_count)
    if selected_count < 0:
        raise ValueError("selected_count must be nonnegative")
    sizes = []
    while selected_count > 0:
        sizes.append(selected_count)
        if selected_count == 1:
            break
        selected_count = (selected_count + 1) // 2
    return tuple(sizes)


def _require_1d(name, value, *, length=None, device=None):
    if not torch.is_tensor(value) or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional torch tensor")
    if length is not None and int(value.numel()) != int(length):
        raise ValueError(f"{name} must have length {length}")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must reside on {device}")


def _validate_inputs(
    *,
    segment_net_index,
    net_ids,
    segment_ids,
    counts,
    raw_grad,
    max_count,
):
    _require_1d("counts", counts)
    segment_count = int(counts.numel())
    device = counts.device
    for name, value in (
        ("segment_net_index", segment_net_index),
        ("segment_ids", segment_ids),
        ("raw_grad", raw_grad),
    ):
        _require_1d(name, value, length=segment_count, device=device)
    _require_1d("net_ids", net_ids, device=device)
    if segment_net_index.dtype != torch.long or segment_ids.dtype != torch.long:
        raise ValueError("segment_net_index and segment_ids must use torch.long")
    if net_ids.dtype != torch.long:
        raise ValueError("net_ids must use torch.long")
    if int(max_count) < 1:
        raise ValueError("max_count must be positive")
    if not bool(torch.isfinite(counts).all()):
        raise ValueError("counts must be finite")
    if not bool(torch.isfinite(raw_grad).all()):
        raise ValueError("raw_grad must be finite")
    if not bool((counts == torch.round(counts)).all()):
        raise ValueError("counts must be integer-valued")
    if not bool(((counts >= 0.0) & (counts <= float(max_count))).all()):
        raise ValueError("counts must lie within [0, max_count]")
    if not bool((segment_ids >= 0).all()):
        raise ValueError("segment_ids must be nonnegative")
    affected_net_count = int(net_ids.numel())
    if affected_net_count == 0:
        if segment_count:
            raise ValueError("empty net_ids requires an empty segment state")
        return
    if not bool(
        ((segment_net_index >= 0) & (segment_net_index < affected_net_count)).all()
    ):
        raise ValueError("segment_net_index contains an out-of-range compact net row")


def _stable_rank_rows(rows, *, segment_net_index, net_ids, segment_ids, score):
    if int(rows.numel()) <= 1:
        return rows
    # Stable least-significant-to-most-significant ordering implements
    # (-score, segment_id, net_id) without moving full tensors to the CPU.
    order = torch.argsort(net_ids[segment_net_index[rows]], stable=True)
    rows = rows[order]
    order = torch.argsort(segment_ids[rows], stable=True)
    rows = rows[order]
    order = torch.argsort(score[rows], descending=True, stable=True)
    return rows[order]


def schedule_net_gradient_actions(
    *,
    segment_net_index,
    net_ids,
    segment_ids,
    counts,
    raw_grad,
    max_count,
    selection_fraction=1.0 / ROUTE_B_AFFECTED_NETS_PER_ACTION,
):
    """Return the atomic Route B count transition for one full gradient vector."""
    _validate_inputs(
        segment_net_index=segment_net_index,
        net_ids=net_ids,
        segment_ids=segment_ids,
        counts=counts,
        raw_grad=raw_grad,
        max_count=max_count,
    )
    selection_fraction = float(selection_fraction)
    if not math.isfinite(selection_fraction) or not 0.0 < selection_fraction <= 1.0:
        raise ValueError("selection_fraction must lie within (0, 1]")
    device = counts.device
    affected_net_count = int(net_ids.numel())
    if affected_net_count == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_value = torch.empty(0, dtype=raw_grad.dtype, device=device)
        return DiscreteVirtualSchedulerTransition(
            next_counts=counts.detach().clone(),
            selected_segment_indices=empty_long,
            selected_net_indices=empty_long,
            selected_segment_ids=empty_long,
            selected_raw_grads=empty_value,
            selected_predicted_improvements=empty_value,
            affected_net_count=0,
            eligible_net_count=0,
            prefix_size=0,
            positive_prefix_count=0,
        )

    predicted_improvement = -raw_grad.detach()
    eligible_segment = counts.detach() < float(max_count)
    negative_infinity = torch.tensor(
        float("-inf"), dtype=predicted_improvement.dtype, device=device
    )
    candidate_score = torch.where(
        eligible_segment,
        predicted_improvement,
        negative_infinity,
    )
    best_score = torch.full(
        (affected_net_count,),
        float("-inf"),
        dtype=predicted_improvement.dtype,
        device=device,
    )
    best_score.scatter_reduce_(
        0,
        segment_net_index,
        candidate_score,
        reduce="amax",
        include_self=True,
    )
    has_eligible = torch.isfinite(best_score)
    eligible_net_count = int(has_eligible.sum().item())

    max_segment_id = torch.iinfo(segment_ids.dtype).max
    best_segment_id = torch.full_like(net_ids, max_segment_id)
    best_segment_id.scatter_reduce_(
        0,
        segment_net_index,
        torch.where(
            eligible_segment & (candidate_score == best_score[segment_net_index]),
            segment_ids,
            torch.full_like(segment_ids, max_segment_id),
        ),
        reduce="amin",
        include_self=True,
    )
    selected_row_mask = eligible_segment & (
        candidate_score == best_score[segment_net_index]
    ) & (segment_ids == best_segment_id[segment_net_index])
    selected_rows = torch.nonzero(selected_row_mask, as_tuple=False).flatten()
    selected_rows = _stable_rank_rows(
        selected_rows,
        segment_net_index=segment_net_index,
        net_ids=net_ids,
        segment_ids=segment_ids,
        score=predicted_improvement,
    )

    prefix_size = max(1, int(math.ceil(affected_net_count * selection_fraction)))
    prefix_rows = selected_rows[:prefix_size]
    accepted_rows = prefix_rows[predicted_improvement[prefix_rows] > 0.0]
    next_counts = counts.detach().clone()
    next_counts[accepted_rows] += 1.0
    selected_net_indices = segment_net_index[accepted_rows]
    return DiscreteVirtualSchedulerTransition(
        next_counts=next_counts,
        selected_segment_indices=accepted_rows,
        selected_net_indices=selected_net_indices,
        selected_segment_ids=segment_ids[accepted_rows],
        selected_raw_grads=raw_grad.detach()[accepted_rows],
        selected_predicted_improvements=predicted_improvement[accepted_rows],
        affected_net_count=affected_net_count,
        eligible_net_count=eligible_net_count,
        prefix_size=prefix_size,
        positive_prefix_count=int((predicted_improvement[prefix_rows] > 0.0).sum().item()),
    )
