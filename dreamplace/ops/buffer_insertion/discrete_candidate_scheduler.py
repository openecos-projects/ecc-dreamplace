from dataclasses import dataclass
import math

import torch

from .discrete_virtual_scheduler import ROUTE_B_AFFECTED_NETS_PER_ACTION


@dataclass(frozen=True)
class DiscreteCandidateTransition:
    next_activations: torch.Tensor
    selected_candidate_indices: torch.Tensor
    selected_net_indices: torch.Tensor
    selected_candidate_ids: torch.Tensor
    selected_raw_grads: torch.Tensor
    selected_predicted_improvements: torch.Tensor
    affected_net_count: int
    eligible_net_count: int
    prefix_size: int
    positive_prefix_count: int

    @property
    def accepted_count(self):
        return int(self.selected_candidate_indices.numel())


@dataclass(frozen=True)
class TypedDiscreteCandidateTransition:
    next_activations: torch.Tensor
    selected_candidate_indices: torch.Tensor
    selected_drive_type_ids: torch.Tensor
    selected_net_indices: torch.Tensor
    selected_candidate_ids: torch.Tensor
    selected_raw_grads: torch.Tensor
    selected_predicted_improvements: torch.Tensor
    affected_net_count: int
    eligible_net_count: int
    prefix_size: int
    positive_prefix_count: int

    @property
    def accepted_count(self):
        return int(self.selected_candidate_indices.numel())


def _require_1d(name, value, *, length=None, device=None):
    if not torch.is_tensor(value) or value.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional torch tensor")
    if length is not None and int(value.numel()) != int(length):
        raise ValueError(f"{name} must have length {length}")
    if device is not None and value.device != device:
        raise ValueError(f"{name} must reside on {device}")


def _validate_inputs(
    *,
    candidate_net_index,
    net_ids,
    candidate_ids,
    activations,
    raw_grad,
):
    _require_1d("activations", activations)
    candidate_count = int(activations.numel())
    device = activations.device
    for name, value in (
        ("candidate_net_index", candidate_net_index),
        ("candidate_ids", candidate_ids),
        ("raw_grad", raw_grad),
    ):
        _require_1d(name, value, length=candidate_count, device=device)
    _require_1d("net_ids", net_ids, device=device)
    if candidate_net_index.dtype != torch.long or candidate_ids.dtype != torch.long:
        raise ValueError("candidate_net_index and candidate_ids must use torch.long")
    if net_ids.dtype != torch.long:
        raise ValueError("net_ids must use torch.long")
    if not bool(torch.isfinite(activations).all()):
        raise ValueError("activations must be finite")
    if not bool(torch.isfinite(raw_grad).all()):
        raise ValueError("raw_grad must be finite")
    if not bool((activations == torch.round(activations)).all()):
        raise ValueError("activations must be binary")
    if not bool(((activations >= 0.0) & (activations <= 1.0)).all()):
        raise ValueError("activations must be binary")
    if not bool((candidate_ids >= 0).all()):
        raise ValueError("candidate_ids must be nonnegative")
    affected_net_count = int(net_ids.numel())
    if affected_net_count == 0:
        if candidate_count:
            raise ValueError("empty net_ids requires an empty candidate state")
        return
    if not bool(
        ((candidate_net_index >= 0) & (candidate_net_index < affected_net_count)).all()
    ):
        raise ValueError("candidate_net_index contains an out-of-range compact net row")


def _stable_rank_rows(rows, *, candidate_net_index, net_ids, candidate_ids, score):
    if int(rows.numel()) <= 1:
        return rows
    order = torch.argsort(net_ids[candidate_net_index[rows]], stable=True)
    rows = rows[order]
    order = torch.argsort(candidate_ids[rows], stable=True)
    rows = rows[order]
    order = torch.argsort(score[rows], descending=True, stable=True)
    return rows[order]


def schedule_candidate_gradient_actions(
    *,
    candidate_net_index,
    net_ids,
    candidate_ids,
    activations,
    raw_grad,
    selection_fraction=1.0 / ROUTE_B_AFFECTED_NETS_PER_ACTION,
):
    """Select one positive inactive candidate per ranked net for one round."""
    _validate_inputs(
        candidate_net_index=candidate_net_index,
        net_ids=net_ids,
        candidate_ids=candidate_ids,
        activations=activations,
        raw_grad=raw_grad,
    )
    selection_fraction = float(selection_fraction)
    if not math.isfinite(selection_fraction) or not 0.0 < selection_fraction <= 1.0:
        raise ValueError("selection_fraction must lie within (0, 1]")
    device = activations.device
    affected_net_count = int(net_ids.numel())
    if affected_net_count == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_value = torch.empty(0, dtype=raw_grad.dtype, device=device)
        return DiscreteCandidateTransition(
            next_activations=activations.detach().clone(),
            selected_candidate_indices=empty_long,
            selected_net_indices=empty_long,
            selected_candidate_ids=empty_long,
            selected_raw_grads=empty_value,
            selected_predicted_improvements=empty_value,
            affected_net_count=0,
            eligible_net_count=0,
            prefix_size=0,
            positive_prefix_count=0,
        )

    predicted_improvement = -raw_grad.detach()
    eligible_candidate = activations.detach() == 0.0
    candidate_score = torch.where(
        eligible_candidate,
        predicted_improvement,
        torch.tensor(
            float("-inf"),
            dtype=predicted_improvement.dtype,
            device=device,
        ),
    )
    best_score = torch.full(
        (affected_net_count,),
        float("-inf"),
        dtype=predicted_improvement.dtype,
        device=device,
    )
    best_score.scatter_reduce_(
        0,
        candidate_net_index,
        candidate_score,
        reduce="amax",
        include_self=True,
    )
    has_eligible = torch.isfinite(best_score)
    eligible_net_count = int(has_eligible.sum().item())

    max_candidate_id = torch.iinfo(candidate_ids.dtype).max
    best_candidate_id = torch.full_like(net_ids, max_candidate_id)
    best_candidate_id.scatter_reduce_(
        0,
        candidate_net_index,
        torch.where(
            eligible_candidate
            & (candidate_score == best_score[candidate_net_index]),
            candidate_ids,
            torch.full_like(candidate_ids, max_candidate_id),
        ),
        reduce="amin",
        include_self=True,
    )
    selected_row_mask = (
        eligible_candidate
        & (candidate_score == best_score[candidate_net_index])
        & (candidate_ids == best_candidate_id[candidate_net_index])
    )
    selected_rows = torch.nonzero(selected_row_mask, as_tuple=False).flatten()
    selected_rows = _stable_rank_rows(
        selected_rows,
        candidate_net_index=candidate_net_index,
        net_ids=net_ids,
        candidate_ids=candidate_ids,
        score=predicted_improvement,
    )
    prefix_size = max(1, int(math.ceil(affected_net_count * selection_fraction)))
    prefix_rows = selected_rows[:prefix_size]
    accepted_rows = prefix_rows[predicted_improvement[prefix_rows] > 0.0]
    next_activations = activations.detach().clone()
    next_activations[accepted_rows] = 1.0
    return DiscreteCandidateTransition(
        next_activations=next_activations,
        selected_candidate_indices=accepted_rows,
        selected_net_indices=candidate_net_index[accepted_rows],
        selected_candidate_ids=candidate_ids[accepted_rows],
        selected_raw_grads=raw_grad.detach()[accepted_rows],
        selected_predicted_improvements=predicted_improvement[accepted_rows],
        affected_net_count=affected_net_count,
        eligible_net_count=eligible_net_count,
        prefix_size=prefix_size,
        positive_prefix_count=int(
            (predicted_improvement[prefix_rows] > 0.0).sum().item()
        ),
    )


def schedule_typed_candidate_gradient_actions(
    *,
    candidate_net_index,
    net_ids,
    candidate_ids,
    activations,
    raw_grad,
    selection_fraction=1.0 / ROUTE_B_AFFECTED_NETS_PER_ACTION,
):
    """Select one positive position/type action per ranked net for one round.

    ``activations`` and ``raw_grad`` have shape ``[candidate_count, type_count]``.
    A row is the physical candidate position; one-hot activation enforces that
    only one drive type can be selected at that position.
    """

    if not torch.is_tensor(activations) or activations.ndim != 2:
        raise ValueError("typed activations must have shape [candidate_count, type_count]")
    if not torch.is_tensor(raw_grad) or raw_grad.shape != activations.shape:
        raise ValueError("typed raw_grad must match typed activation shape")
    selection_fraction = float(selection_fraction)
    if not math.isfinite(selection_fraction) or not 0.0 < selection_fraction <= 1.0:
        raise ValueError("selection_fraction must lie within (0, 1]")
    candidate_count, drive_type_count = activations.shape
    if drive_type_count <= 0:
        raise ValueError("typed candidate state requires at least one drive type")
    device = activations.device
    for name, value in (
        ("candidate_net_index", candidate_net_index),
        ("candidate_ids", candidate_ids),
    ):
        _require_1d(name, value, length=candidate_count, device=device)
    _require_1d("net_ids", net_ids, device=device)
    if candidate_net_index.dtype != torch.long or candidate_ids.dtype != torch.long:
        raise ValueError("candidate_net_index and candidate_ids must use torch.long")
    if net_ids.dtype != torch.long:
        raise ValueError("net_ids must use torch.long")
    if not bool(torch.isfinite(activations).all()):
        raise ValueError("typed activations must be finite")
    if not bool(torch.isfinite(raw_grad).all()):
        raise ValueError("typed raw_grad must be finite")
    if not bool((activations == torch.round(activations)).all()):
        raise ValueError("typed activations must be binary")
    if not bool(((activations >= 0.0) & (activations <= 1.0)).all()):
        raise ValueError("typed activations must be binary")
    if not bool((activations.sum(dim=1) <= 1.0).all()):
        raise ValueError("typed activations violate position one-hot")
    affected_net_count = int(net_ids.numel())
    if affected_net_count == 0:
        empty_long = torch.empty(0, dtype=torch.long, device=device)
        empty_value = torch.empty(0, dtype=raw_grad.dtype, device=device)
        return TypedDiscreteCandidateTransition(
            next_activations=activations.detach().clone(),
            selected_candidate_indices=empty_long,
            selected_drive_type_ids=empty_long,
            selected_net_indices=empty_long,
            selected_candidate_ids=empty_long,
            selected_raw_grads=empty_value,
            selected_predicted_improvements=empty_value,
            affected_net_count=0,
            eligible_net_count=0,
            prefix_size=0,
            positive_prefix_count=0,
        )
    if not bool(
        ((candidate_net_index >= 0) & (candidate_net_index < affected_net_count)).all()
    ):
        raise ValueError("candidate_net_index contains an out-of-range compact net row")

    predicted = -raw_grad.detach()
    active_position = activations.detach().sum(dim=1) > 0.0
    eligible_action = (~active_position).view(-1, 1).expand(-1, drive_type_count)
    action_score = torch.where(
        eligible_action,
        predicted,
        torch.tensor(float("-inf"), dtype=predicted.dtype, device=device),
    )
    action_net_index = candidate_net_index.view(-1, 1).expand(-1, drive_type_count).reshape(-1)
    action_candidate_ids = candidate_ids.view(-1, 1).expand(-1, drive_type_count).reshape(-1)
    action_drive_type_ids = torch.arange(
        drive_type_count,
        dtype=torch.long,
        device=device,
    ).view(1, -1).expand(candidate_count, -1).reshape(-1)
    action_score_flat = action_score.reshape(-1)

    best_score = torch.full(
        (affected_net_count,),
        float("-inf"),
        dtype=predicted.dtype,
        device=device,
    )
    best_score.scatter_reduce_(
        0,
        action_net_index,
        action_score_flat,
        reduce="amax",
        include_self=True,
    )
    has_eligible = torch.isfinite(best_score)
    eligible_net_count = int(has_eligible.sum().item())
    max_candidate_id = torch.iinfo(candidate_ids.dtype).max
    best_candidate_id = torch.full_like(net_ids, max_candidate_id)
    best_candidate_id.scatter_reduce_(
        0,
        action_net_index,
        torch.where(
            eligible_action.reshape(-1)
            & (action_score_flat == best_score[action_net_index]),
            action_candidate_ids,
            torch.full_like(action_candidate_ids, max_candidate_id),
        ),
        reduce="amin",
        include_self=True,
    )
    best_type_id = torch.full_like(net_ids, drive_type_count)
    best_type_id.scatter_reduce_(
        0,
        action_net_index,
        torch.where(
            eligible_action.reshape(-1)
            & (action_score_flat == best_score[action_net_index])
            & (action_candidate_ids == best_candidate_id[action_net_index]),
            action_drive_type_ids,
            torch.full_like(action_drive_type_ids, drive_type_count),
        ),
        reduce="amin",
        include_self=True,
    )
    selected_mask = (
        eligible_action.reshape(-1)
        & (action_score_flat == best_score[action_net_index])
        & (action_candidate_ids == best_candidate_id[action_net_index])
        & (action_drive_type_ids == best_type_id[action_net_index])
    )
    selected_actions = torch.nonzero(selected_mask, as_tuple=False).flatten()
    if int(selected_actions.numel()) > 1:
        order = torch.argsort(
            net_ids[action_net_index[selected_actions]],
            stable=True,
        )
        selected_actions = selected_actions[order]
        order = torch.argsort(action_candidate_ids[selected_actions], stable=True)
        selected_actions = selected_actions[order]
        selected_actions = selected_actions[
            torch.argsort(action_drive_type_ids[selected_actions], stable=True)
        ]
        selected_actions = selected_actions[
            torch.argsort(action_score_flat[selected_actions], descending=True, stable=True)
        ]
    prefix_size = max(1, int(math.ceil(affected_net_count * selection_fraction)))
    prefix_actions = selected_actions[:prefix_size]
    accepted_actions = prefix_actions[action_score_flat[prefix_actions] > 0.0]
    next_activations = activations.detach().clone()
    if int(accepted_actions.numel()):
        candidate_rows = torch.div(
            accepted_actions,
            drive_type_count,
            rounding_mode="floor",
        )
        type_rows = action_drive_type_ids[accepted_actions]
        next_activations[candidate_rows, type_rows] = 1.0
    candidate_rows = torch.div(
        accepted_actions,
        drive_type_count,
        rounding_mode="floor",
    )
    return TypedDiscreteCandidateTransition(
        next_activations=next_activations,
        selected_candidate_indices=candidate_rows,
        selected_drive_type_ids=action_drive_type_ids[accepted_actions],
        selected_net_indices=action_net_index[accepted_actions],
        selected_candidate_ids=action_candidate_ids[accepted_actions],
        selected_raw_grads=raw_grad.detach().reshape(-1)[accepted_actions],
        selected_predicted_improvements=predicted.reshape(-1)[accepted_actions],
        affected_net_count=affected_net_count,
        eligible_net_count=eligible_net_count,
        prefix_size=prefix_size,
        positive_prefix_count=int(
            (action_score_flat[prefix_actions] > 0.0).sum().item()
        ),
    )
