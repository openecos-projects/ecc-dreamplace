from dataclasses import dataclass

import torch
from torch import nn


def _candidate_int(candidate, key, *, fallback_key=None, fallback_keys=None):
    value = candidate.get(key)
    if value is None and fallback_key is not None:
        value = candidate.get(fallback_key)
    if value is None:
        for extra_key in fallback_keys or ():
            value = candidate.get(extra_key)
            if value is not None:
                break
    if value is None:
        raise ValueError(f"candidate missing {key}")
    return int(value)


def _index_select_to(value, indices, device):
    index = indices.to(device=value.device, dtype=torch.long)
    return value.index_select(0, index).to(device=device)


@dataclass
class CandidateStateView:
    candidate_ids: torch.Tensor
    candidate_node_id: torch.Tensor
    candidate_net_id: torch.Tensor
    candidate_bu_tensor: torch.Tensor
    bsu_index_tensor: torch.Tensor
    legal_buffer_count: int
    buffer_main_type_index: int

    def candidate_bu(self):
        return self.candidate_bu_tensor

    def relaxed_bu(self):
        return self.candidate_bu()

    def bsu_index(self):
        return self.bsu_index_tensor


@dataclass
class BufferOptimizationState:
    candidate_ids: torch.Tensor
    candidate_node_id: torch.Tensor
    candidate_net_id: torch.Tensor
    bu_logits: nn.Parameter
    bsu_index_param: nn.Parameter
    legal_buffer_count: int
    buffer_main_type_index: int
    candidate_records: tuple

    def candidate_bu(self):
        return torch.sigmoid(self.bu_logits)

    def relaxed_bu(self):
        return self.candidate_bu()

    def bsu_index(self):
        return torch.clamp(self.bsu_index_param, 0.0, float(self.legal_buffer_count - 1))

    def project_bsu_index_(self):
        with torch.no_grad():
            self.bsu_index_param.clamp_(0.0, float(self.legal_buffer_count - 1))
        return self

    def slice_candidates(self, indices, *, candidate_node_id, device):
        index = indices.to(dtype=torch.long)
        bu_logits = _index_select_to(self.bu_logits, index, device)
        bsu_index_param = _index_select_to(self.bsu_index_param, index, device)
        return CandidateStateView(
            candidate_ids=_index_select_to(self.candidate_ids, index, device),
            candidate_node_id=candidate_node_id.to(
                device=device,
                dtype=self.candidate_node_id.dtype,
            ),
            candidate_net_id=_index_select_to(
                self.candidate_net_id,
                index,
                device,
            ),
            candidate_bu_tensor=torch.sigmoid(bu_logits),
            bsu_index_tensor=torch.clamp(
                bsu_index_param,
                0.0,
                float(self.legal_buffer_count - 1),
            ),
            legal_buffer_count=int(self.legal_buffer_count),
            buffer_main_type_index=int(self.buffer_main_type_index),
        )


@dataclass
class DiscreteCandidateState:
    candidate_ids: torch.Tensor
    candidate_node_id: torch.Tensor
    candidate_net_id: torch.Tensor
    candidate_net_index: torch.Tensor
    net_ids: torch.Tensor
    activation_param: nn.Parameter
    fixed_bsu_index: int
    fixed_bsu_tensor: torch.Tensor
    legal_buffer_count: int
    buffer_main_type_index: int
    candidate_records: tuple

    def candidate_bu(self):
        return self.activation_param

    def relaxed_bu(self):
        return self.candidate_bu()

    def bsu_index(self):
        return self.fixed_bsu_tensor

    def validate_binary_(self):
        activation = self.activation_param.detach()
        if activation.ndim != 1 or activation.numel() != self.candidate_ids.numel():
            raise ValueError("candidate activation shape mismatch")
        if not bool(torch.isfinite(activation).all()):
            raise ValueError("candidate activation must be finite")
        if not bool((activation == torch.round(activation)).all()):
            raise ValueError("candidate activation must be binary")
        if not bool(((activation >= 0.0) & (activation <= 1.0)).all()):
            raise ValueError("candidate activation must lie in [0, 1]")
        if not bool((self.fixed_bsu_tensor == float(self.fixed_bsu_index)).all()):
            raise ValueError("candidate bsu state must stay fixed")
        return self

    def slice_candidates(self, indices, *, candidate_node_id, device):
        index = indices.to(dtype=torch.long)
        return CandidateStateView(
            candidate_ids=_index_select_to(self.candidate_ids, index, device),
            candidate_node_id=candidate_node_id.to(
                device=device,
                dtype=self.candidate_node_id.dtype,
            ),
            candidate_net_id=_index_select_to(
                self.candidate_net_id,
                index,
                device,
            ),
            candidate_bu_tensor=_index_select_to(
                self.activation_param,
                index,
                device,
            ),
            bsu_index_tensor=_index_select_to(
                self.fixed_bsu_tensor,
                index,
                device,
            ),
            legal_buffer_count=int(self.legal_buffer_count),
            buffer_main_type_index=int(self.buffer_main_type_index),
        )


def _capture_optional_tensor(tensor):
    if tensor is None:
        return None
    return tensor.detach().clone()


def _tensor_grad_stats(tensor):
    if tensor is None or tensor.grad is None:
        return None, None
    grad = tensor.grad.detach()
    if grad.numel() == 0:
        return 0.0, 0.0
    return float(grad.norm().item()), float(grad.abs().max().item())


def capture_buffer_optimization_snapshot(state):
    if state is None:
        return {
            "buffer_bu_logits": None,
            "buffer_z_param": None,
            "buffer_bsu_index_param": None,
            "buffer_bu_var": None,
            "buffer_z_var": None,
            "buffer_bsu_index_var": None,
        }
    bu_logits = getattr(state, "bu_logits", None)
    z_param = getattr(state, "z_param", None)
    relaxed_bu = state.relaxed_bu() if hasattr(state, "relaxed_bu") else None
    z_value = state.z_value() if hasattr(state, "z_value") else None
    return {
        "buffer_bu_logits": _capture_optional_tensor(bu_logits),
        "buffer_z_param": _capture_optional_tensor(z_param),
        "buffer_bsu_index_param": _capture_optional_tensor(
            getattr(state, "bsu_index_param", None)
        ),
        "buffer_bu_var": _capture_optional_tensor(relaxed_bu),
        "buffer_z_var": _capture_optional_tensor(z_value),
        "buffer_bsu_index_var": _capture_optional_tensor(state.bsu_index()),
    }


def restore_buffer_optimization_snapshot(state, snapshot):
    summary = {
        "restored": False,
        "restore_reason": None,
    }
    if state is None:
        summary["restore_reason"] = "buffer_state_unavailable"
        return summary
    if not isinstance(snapshot, dict):
        summary["restore_reason"] = "buffer_snapshot_unavailable"
        return summary
    restored_keys = []
    for key, target in (
        ("buffer_bu_logits", getattr(state, "bu_logits", None)),
        ("buffer_z_param", getattr(state, "z_param", None)),
        ("buffer_bsu_index_param", getattr(state, "bsu_index_param", None)),
    ):
        source = snapshot.get(key)
        if source is None or target is None:
            continue
        with torch.no_grad():
            target.copy_(source.to(device=target.device, dtype=target.dtype))
        restored_keys.append(key)
    summary["restored"] = bool(restored_keys)
    summary["restore_reason"] = (
        "buffer_snapshot_restore" if restored_keys else "buffer_snapshot_empty"
    )
    summary["restored_keys"] = restored_keys
    return summary


def buffer_optimization_grad_stats(state):
    if state is None:
        return {
            "buffer_bu_grad_norm": None,
            "buffer_bu_grad_max_abs": None,
            "buffer_z_grad_norm": None,
            "buffer_z_grad_max_abs": None,
            "buffer_bsu_index_grad_norm": None,
            "buffer_bsu_index_grad_max_abs": None,
        }
    bu_norm, bu_max = _tensor_grad_stats(getattr(state, "bu_logits", None))
    z_norm, z_max = _tensor_grad_stats(getattr(state, "z_param", None))
    bsu_norm, bsu_max = _tensor_grad_stats(getattr(state, "bsu_index_param", None))
    return {
        "buffer_bu_grad_norm": bu_norm,
        "buffer_bu_grad_max_abs": bu_max,
        "buffer_z_grad_norm": z_norm,
        "buffer_z_grad_max_abs": z_max,
        "buffer_bsu_index_grad_norm": bsu_norm,
        "buffer_bsu_index_grad_max_abs": bsu_max,
    }


def build_buffer_optimization_state(
    candidates,
    *,
    buffer_main_type_index,
    legal_buffer_count,
    initial_bu_logit=0.0,
    initial_bsu_index=0.5,
    dtype=torch.float32,
    device=None,
):
    candidates = list(candidates or [])
    buffer_main_type_index = int(buffer_main_type_index)
    legal_buffer_count = int(legal_buffer_count)
    if legal_buffer_count < 2:
        raise ValueError("legal_buffer_count must be at least 2 for bsu interpolation")

    candidate_ids = []
    candidate_node_ids = []
    candidate_net_ids = []
    records = []
    for candidate in candidates:
        candidate_family = _candidate_int(candidate, "buffer_main_type_index")
        if candidate_family != buffer_main_type_index:
            raise ValueError(
                "candidate buffer_main_type_index does not match requested "
                f"buffer_main_type_index: {candidate_family} != {buffer_main_type_index}"
            )
        candidate_ids.append(_candidate_int(candidate, "candidate_id"))
        candidate_node_ids.append(
            _candidate_int(
                candidate,
                "candidate_node_id",
                fallback_key="tree_node_id",
                fallback_keys=("synthetic_node_id",),
            )
        )
        candidate_net_ids.append(_candidate_int(candidate, "net_id"))
        # Records stay read-only until projection copies the selected entries.
        records.append(candidate)

    tensor_device = torch.device(device) if device is not None else None
    num_candidates = len(candidates)
    bu_logits = nn.Parameter(
        torch.full(
            (num_candidates,),
            float(initial_bu_logit),
            dtype=dtype,
            device=tensor_device,
        )
    )
    bsu_index_param = nn.Parameter(
        torch.full(
            (num_candidates,),
            float(initial_bsu_index),
            dtype=dtype,
            device=tensor_device,
        )
    )

    return BufferOptimizationState(
        candidate_ids=torch.tensor(candidate_ids, dtype=torch.long, device=tensor_device),
        candidate_node_id=torch.tensor(candidate_node_ids, dtype=torch.long, device=tensor_device),
        candidate_net_id=torch.tensor(candidate_net_ids, dtype=torch.long, device=tensor_device),
        bu_logits=bu_logits,
        bsu_index_param=bsu_index_param,
        legal_buffer_count=legal_buffer_count,
        buffer_main_type_index=buffer_main_type_index,
        candidate_records=tuple(records),
    )


def build_discrete_candidate_state(
    candidates,
    *,
    buffer_main_type_index,
    legal_buffer_count,
    fixed_bsu_index,
    dtype=torch.float32,
    device=None,
):
    candidates = list(candidates or [])
    buffer_main_type_index = int(buffer_main_type_index)
    legal_buffer_count = int(legal_buffer_count)
    fixed_bsu_index = int(fixed_bsu_index)
    if legal_buffer_count < 2:
        raise ValueError("legal_buffer_count must be at least 2")
    if not 0 <= fixed_bsu_index < legal_buffer_count:
        raise ValueError("fixed_bsu_index is out of range")

    candidate_ids = []
    candidate_node_ids = []
    candidate_net_ids = []
    records = []
    for candidate in candidates:
        candidate_family = _candidate_int(candidate, "buffer_main_type_index")
        if candidate_family != buffer_main_type_index:
            raise ValueError(
                "candidate buffer_main_type_index does not match requested "
                f"buffer_main_type_index: {candidate_family} != {buffer_main_type_index}"
            )
        has_coordinate = (
            candidate.get("x_dbu") is not None
            and candidate.get("y_dbu") is not None
        ) or (
            candidate.get("candidate_location_x") is not None
            and candidate.get("candidate_location_y") is not None
        )
        if not has_coordinate:
            raise ValueError("candidate is missing coordinate payload")
        candidate_ids.append(_candidate_int(candidate, "candidate_id"))
        candidate_node_ids.append(
            _candidate_int(
                candidate,
                "candidate_node_id",
                fallback_key="tree_node_id",
                fallback_keys=("synthetic_node_id",),
            )
        )
        candidate_net_ids.append(_candidate_int(candidate, "net_id"))
        # Records stay read-only until projection copies the selected entries.
        records.append(candidate)

    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_ids must be unique")
    net_ids = sorted(set(candidate_net_ids))
    net_index_by_id = {net_id: index for index, net_id in enumerate(net_ids)}
    candidate_net_index = [net_index_by_id[net_id] for net_id in candidate_net_ids]
    tensor_device = torch.device(device) if device is not None else None
    num_candidates = len(candidate_ids)
    activation_param = nn.Parameter(
        torch.zeros(num_candidates, dtype=dtype, device=tensor_device)
    )
    fixed_bsu_tensor = torch.full(
        (num_candidates,),
        float(fixed_bsu_index),
        dtype=dtype,
        device=tensor_device,
    )
    state = DiscreteCandidateState(
        candidate_ids=torch.tensor(
            candidate_ids,
            dtype=torch.long,
            device=tensor_device,
        ),
        candidate_node_id=torch.tensor(
            candidate_node_ids,
            dtype=torch.long,
            device=tensor_device,
        ),
        candidate_net_id=torch.tensor(
            candidate_net_ids,
            dtype=torch.long,
            device=tensor_device,
        ),
        candidate_net_index=torch.tensor(
            candidate_net_index,
            dtype=torch.long,
            device=tensor_device,
        ),
        net_ids=torch.tensor(net_ids, dtype=torch.long, device=tensor_device),
        activation_param=activation_param,
        fixed_bsu_index=fixed_bsu_index,
        fixed_bsu_tensor=fixed_bsu_tensor,
        legal_buffer_count=legal_buffer_count,
        buffer_main_type_index=buffer_main_type_index,
        candidate_records=tuple(records),
    )
    return state.validate_binary_()


def _validate_per_size_tensor(name, value, *, bsu_index):
    if not torch.is_tensor(value):
        value = torch.as_tensor(value, dtype=bsu_index.dtype, device=bsu_index.device)
    value = value.to(device=bsu_index.device, dtype=bsu_index.dtype)
    if value.ndim != 2:
        raise ValueError(f"{name} must have shape [num_candidates, legal_buffer_count]")
    if value.shape[0] != bsu_index.numel():
        raise ValueError(f"{name} candidate dimension does not match bsu_index")
    if value.shape[1] < 2:
        raise ValueError(f"{name} must contain at least two legal buffer sizes")
    if not torch.isfinite(value).all():
        raise ValueError(f"{name} contains non-finite values")
    return value


def _clamp_status(raw_index, clipped_index, legal_buffer_count, coordinate_source):
    low = bool(torch.any(raw_index < 0.0).detach().cpu().item())
    high = bool(torch.any(raw_index > float(legal_buffer_count - 1)).detach().cpu().item())
    if low and high:
        clamp = "mixed"
    elif low:
        clamp = "low"
    elif high:
        clamp = "high"
    else:
        clamp = "none"
    return {
        "coordinate_source": str(coordinate_source),
        "clamped": bool(low or high),
        "clamp": clamp,
        "axis_min": 0.0,
        "axis_max": float(legal_buffer_count - 1),
        "raw_min": float(raw_index.detach().min().cpu().item()) if raw_index.numel() else None,
        "raw_max": float(raw_index.detach().max().cpu().item()) if raw_index.numel() else None,
        "clipped_min": float(clipped_index.detach().min().cpu().item()) if clipped_index.numel() else None,
        "clipped_max": float(clipped_index.detach().max().cpu().item()) if clipped_index.numel() else None,
    }


def interpolate_buffer_values_by_bsu_index(
    bsu_index,
    *,
    per_size_input_cap,
    per_size_delay,
    per_size_output_slew,
    coordinate_source="current_relaxed",
):
    if not torch.is_tensor(bsu_index):
        bsu_index = torch.as_tensor(bsu_index, dtype=torch.float32)
    if bsu_index.ndim != 1:
        raise ValueError("bsu_index must have shape [num_candidates]")

    per_size_input_cap = _validate_per_size_tensor(
        "per_size_input_cap",
        per_size_input_cap,
        bsu_index=bsu_index,
    )
    per_size_delay = _validate_per_size_tensor(
        "per_size_delay",
        per_size_delay,
        bsu_index=bsu_index,
    )
    per_size_output_slew = _validate_per_size_tensor(
        "per_size_output_slew",
        per_size_output_slew,
        bsu_index=bsu_index,
    )
    legal_buffer_count = int(per_size_delay.shape[1])
    for name, value in (
        ("per_size_input_cap", per_size_input_cap),
        ("per_size_output_slew", per_size_output_slew),
    ):
        if int(value.shape[1]) != legal_buffer_count:
            raise ValueError(f"{name} legal-size dimension does not match per_size_delay")

    raw_index = bsu_index
    clipped_index = torch.clamp(raw_index, 0.0, float(legal_buffer_count - 1))
    lo_index = torch.floor(clipped_index).to(dtype=torch.long)
    hi_index = torch.ceil(clipped_index).to(dtype=torch.long)
    alpha = clipped_index - lo_index.to(dtype=clipped_index.dtype)

    def interpolate(table):
        lo = torch.gather(table, 1, lo_index.view(-1, 1)).view(-1)
        hi = torch.gather(table, 1, hi_index.view(-1, 1)).view(-1)
        return (1.0 - alpha) * lo + alpha * hi

    return {
        "buffer_input_cap": interpolate(per_size_input_cap),
        "buffer_delay": interpolate(per_size_delay),
        "buffer_output_slew": interpolate(per_size_output_slew),
        "bsu_index": clipped_index,
        "bsu_index_lo": lo_index,
        "bsu_index_hi": hi_index,
        "bsu_index_alpha": alpha,
        "bsu_index_status": _clamp_status(
            raw_index,
            clipped_index,
            legal_buffer_count,
            coordinate_source,
        ),
    }


def _candidate_has_coordinate(candidate):
    return (
        candidate.get("x_dbu") is not None
        and candidate.get("y_dbu") is not None
    ) or (
        candidate.get("candidate_location_x") is not None
        and candidate.get("candidate_location_y") is not None
    )


def project_buffer_optimization_state(
    state,
    *,
    top_k=None,
    threshold=None,
):
    from dreamplace.ops.physical_action import VirtualBufferState

    scores = state.relaxed_bu().detach().cpu()
    bsu_index = state.bsu_index().detach().cpu()
    candidate_ids = state.candidate_ids.detach().cpu().tolist()
    if threshold is not None:
        selected = [
            index for index, score in enumerate(scores.tolist())
            if float(score) >= float(threshold)
        ]
    else:
        selected = list(range(len(candidate_ids)))
    selected = sorted(
        selected,
        key=lambda index: (-float(scores[index]), int(candidate_ids[index])),
    )
    if top_k is not None:
        selected = selected[: int(top_k)]

    virtual_state = VirtualBufferState()
    max_bsu = int(state.legal_buffer_count) - 1
    for index in selected:
        candidate = dict(state.candidate_records[index])
        if not _candidate_has_coordinate(candidate):
            raise ValueError("selected buffer candidate is missing coordinate payload")
        raw_bsu = int(torch.round(torch.clamp(bsu_index[index], 0.0, float(max_bsu))).item())
        bsu = max(0, min(max_bsu, raw_bsu))
        candidate["buffer_projection_score"] = float(scores[index])
        candidate["bsu_selection_mode"] = "relaxed_index_round_clip"
        candidate["bsu_index_value"] = float(bsu_index[index])
        virtual_state.insert(candidate, bsu=bsu)
    return virtual_state


def project_discrete_candidate_state(state):
    from dreamplace.ops.physical_action import VirtualBufferState

    state.validate_binary_()
    active_indices = torch.nonzero(
        state.activation_param.detach() == 1.0,
        as_tuple=False,
    ).flatten()
    if int(active_indices.numel()) > 1:
        order = torch.argsort(
            state.candidate_ids.index_select(0, active_indices),
            stable=True,
        )
        active_indices = active_indices[order]
    virtual_state = VirtualBufferState()
    for index in active_indices.detach().cpu().tolist():
        candidate = dict(state.candidate_records[index])
        if not _candidate_has_coordinate(candidate):
            raise ValueError("selected buffer candidate is missing coordinate payload")
        candidate["buffer_projection_score"] = 1.0
        candidate["bsu_selection_mode"] = "fixed_discrete_candidate"
        candidate["bsu_index_value"] = float(state.fixed_bsu_index)
        virtual_state.insert(candidate, bsu=int(state.fixed_bsu_index))
    return virtual_state
