from dataclasses import dataclass

import torch
from torch import nn

from .buffer_interpolation import interpolate_buffer_values_by_bsu_index


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
class TypedCandidateStateView(CandidateStateView):
    activation_param: torch.Tensor
    drive_type_bsu_index: torch.Tensor
    _shared_gradient_probe: bool = True


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


@dataclass
class TypedDiscreteCandidateState:
    """Discrete candidate state with one-hot drive-type actions per position.

    ``activation_param`` has shape ``[candidate_count, drive_type_count]``.
    The timing provider still sees one candidate row per physical position;
    ``candidate_bu`` collapses the one-hot row and ``bsu_index`` selects the
    corresponding legal family index.  During typed gradient probing,
    ``_gradient_drive_type`` temporarily assigns the requested type only to
    inactive positions, while retaining already accepted types.
    """

    candidate_ids: torch.Tensor
    candidate_node_id: torch.Tensor
    candidate_net_id: torch.Tensor
    candidate_net_index: torch.Tensor
    net_ids: torch.Tensor
    activation_param: nn.Parameter
    drive_type_bsu_index: torch.Tensor
    drive_type_ids: torch.Tensor
    drive_type_master_names: tuple
    drive_type_cell_ids: torch.Tensor
    legal_buffer_count: int
    buffer_main_type_index: int
    candidate_records: tuple
    drive_type_table_metadata: dict
    _gradient_drive_type: int | None = None
    _shared_gradient_probe: bool = False
    timing_active_only: bool = False

    def timing_affected_net_ids(self):
        """Between probes, let ordinary RC own nets without accepted buffers."""
        if (not self.timing_active_only or self._gradient_drive_type is not None
                or self._shared_gradient_probe):
            return None
        return tuple(sorted(set(
            self.candidate_net_id[self._active_mask()].detach().cpu().tolist()
        )))

    @property
    def is_typed_candidate(self):
        return True

    @property
    def fixed_bsu_index(self):
        return None

    @property
    def drive_type_count(self):
        return int(self.activation_param.shape[1])

    def _active_mask(self):
        return self.activation_param.detach().sum(dim=1) > 0.0

    def _selected_drive_type(self):
        return torch.argmax(self.activation_param.detach(), dim=1)

    def _bsu_for_types(self, type_ids):
        return self.drive_type_bsu_index.to(device=type_ids.device).index_select(
            0,
            type_ids.to(dtype=torch.long),
        )

    def candidate_bu(self):
        return self._candidate_bu(self.activation_param)

    def _candidate_bu(self, activation):
        if self._gradient_drive_type is None:
            return activation.sum(dim=1)
        active = activation.detach().sum(dim=1) > 0.0
        typed = activation[:, int(self._gradient_drive_type)]
        return torch.where(active, activation.sum(dim=1), typed)

    def relaxed_bu(self):
        return self.candidate_bu()

    def bsu_index(self):
        return self._bsu_index(self.activation_param)

    def _bsu_index(self, activation):
        selected_types = torch.argmax(activation.detach(), dim=1)
        selected_bsu = self._bsu_for_types(selected_types).to(
            dtype=activation.dtype,
        )
        if self._gradient_drive_type is None:
            return selected_bsu
        active = activation.detach().sum(dim=1) > 0.0
        typed_bsu = self.drive_type_bsu_index.to(
            device=selected_bsu.device,
            dtype=selected_bsu.dtype,
        )[int(self._gradient_drive_type)]
        return torch.where(active, selected_bsu, typed_bsu.expand_as(selected_bsu))

    def set_gradient_drive_type(self, drive_type_id):
        drive_type_id = int(drive_type_id)
        if not 0 <= drive_type_id < self.drive_type_count:
            raise ValueError("drive_type_id is outside typed candidate state")
        self._gradient_drive_type = drive_type_id
        self._shared_gradient_probe = False
        return self

    def begin_shared_gradient_probe(self):
        self._gradient_drive_type = None
        self._shared_gradient_probe = True
        return self

    def clear_gradient_drive_type(self):
        self._gradient_drive_type = None
        self._shared_gradient_probe = False
        return self

    def selected_drive_type(self):
        return self._selected_drive_type()

    def validate_binary_(self):
        activation = self.activation_param.detach()
        if activation.ndim != 2:
            raise ValueError("typed candidate activation must be two-dimensional")
        if activation.shape[0] != self.candidate_ids.numel():
            raise ValueError("typed candidate activation candidate dimension mismatch")
        if activation.shape[1] != self.drive_type_count:
            raise ValueError("typed candidate activation type dimension mismatch")
        if not bool(torch.isfinite(activation).all()):
            raise ValueError("typed candidate activation must be finite")
        if not bool((activation == torch.round(activation)).all()):
            raise ValueError("typed candidate activation must be binary")
        if not bool(((activation >= 0.0) & (activation <= 1.0)).all()):
            raise ValueError("typed candidate activation must lie in [0, 1]")
        if not bool((activation.sum(dim=1) <= 1.0).all()):
            raise ValueError("typed candidate activation violates position one-hot")
        if self.drive_type_bsu_index.numel() != self.drive_type_count:
            raise ValueError("typed candidate BSU table length mismatch")
        if not bool(
            ((self.drive_type_bsu_index >= 0)
             & (self.drive_type_bsu_index < self.legal_buffer_count)).all()
        ):
            raise ValueError("typed candidate BSU table contains an invalid index")
        return self

    def slice_candidates(self, indices, *, candidate_node_id, device):
        # Active-view indices are packed on CPU, while typed activations may
        # live on CUDA.  Advanced indexing must use the activation device;
        # the previous CPU index tensor made CUDA candidate windows fail as
        # soon as an existing typed buffer was included in the cap overlay.
        index = indices.to(device=self.activation_param.device, dtype=torch.long)
        activation = self.activation_param.index_select(0, index)
        candidate_bu = self._candidate_bu(activation).to(device=device)
        bsu_index = self._bsu_index(activation).to(device=device)
        view = CandidateStateView(
            candidate_ids=_index_select_to(self.candidate_ids, index, device),
            candidate_node_id=candidate_node_id.to(
                device=device,
                dtype=self.candidate_node_id.dtype,
            ),
            candidate_net_id=_index_select_to(self.candidate_net_id, index, device),
            candidate_bu_tensor=candidate_bu,
            bsu_index_tensor=bsu_index,
            legal_buffer_count=int(self.legal_buffer_count),
            buffer_main_type_index=int(self.buffer_main_type_index),
        )
        if self._shared_gradient_probe:
            return TypedCandidateStateView(
                **vars(view),
                activation_param=activation.to(device=device),
                drive_type_bsu_index=self.drive_type_bsu_index.to(device=device),
            )
        return view


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


def build_typed_discrete_candidate_state(
    candidates,
    *,
    buffer_main_type_index,
    legal_buffer_count,
    drive_type_table,
    dtype=torch.float32,
    device=None,
):
    """Build a position-by-drive-type one-hot candidate state."""

    candidates = list(candidates or [])
    buffer_main_type_index = int(buffer_main_type_index)
    legal_buffer_count = int(legal_buffer_count)
    entries = tuple(getattr(drive_type_table, "entries", ()) or ())
    if len(entries) != 4:
        raise ValueError("typed discrete candidate buffering requires four drive types")
    if legal_buffer_count < 2:
        raise ValueError("legal_buffer_count must be at least 2")

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
        records.append(candidate)

    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate_ids must be unique")
    net_ids = sorted(set(candidate_net_ids))
    net_index_by_id = {net_id: index for index, net_id in enumerate(net_ids)}
    candidate_net_index = [net_index_by_id[net_id] for net_id in candidate_net_ids]
    tensor_device = torch.device(device) if device is not None else None
    num_candidates = len(candidate_ids)
    activation_param = nn.Parameter(
        torch.zeros(
            (num_candidates, len(entries)),
            dtype=dtype,
            device=tensor_device,
        )
    )
    state = TypedDiscreteCandidateState(
        candidate_ids=torch.tensor(candidate_ids, dtype=torch.long, device=tensor_device),
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
        drive_type_bsu_index=torch.tensor(
            [int(entry.bsu_index) for entry in entries],
            dtype=torch.long,
            device=tensor_device,
        ),
        drive_type_ids=torch.arange(
            len(entries),
            dtype=torch.long,
            device=tensor_device,
        ),
        drive_type_master_names=tuple(str(entry.master_name) for entry in entries),
        drive_type_cell_ids=torch.tensor(
            [int(entry.cell_id) for entry in entries],
            dtype=torch.long,
            device=tensor_device,
        ),
        legal_buffer_count=legal_buffer_count,
        buffer_main_type_index=buffer_main_type_index,
        candidate_records=records,
        drive_type_table_metadata=drive_type_table.to_metadata(),
    )
    return state.validate_binary_()


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


def project_typed_discrete_candidate_state(state):
    """Project typed one-hot candidate actions to physical virtual entries."""

    from dreamplace.ops.physical_action import VirtualBufferState

    state.validate_binary_()
    active_rows = torch.nonzero(
        state.activation_param.detach().sum(dim=1) == 1.0,
        as_tuple=False,
    ).flatten()
    if int(active_rows.numel()) > 1:
        order = torch.argsort(
            state.candidate_ids.index_select(0, active_rows),
            stable=True,
        )
        active_rows = active_rows[order]
    selected_types = state.selected_drive_type().detach()
    virtual_state = VirtualBufferState()
    for row in active_rows.detach().cpu().tolist():
        candidate = dict(state.candidate_records[row])
        if not _candidate_has_coordinate(candidate):
            raise ValueError("selected buffer candidate is missing coordinate payload")
        drive_type_id = int(selected_types[row].detach().cpu().item())
        drive_type = state.drive_type_table_metadata["masters"][drive_type_id]
        bsu = int(state.drive_type_bsu_index[drive_type_id].detach().cpu().item())
        candidate.update(
            {
                "buffer_projection_score": 1.0,
                "bsu_selection_mode": "drive_type_table",
                "bsu_index_value": float(bsu),
                "drive_type_id": drive_type_id,
                "drive_type_master_name": str(drive_type["master_name"]),
                "drive_type_cell_id": int(drive_type["cell_id"]),
                "drive_type_main_type_index": int(
                    drive_type.get(
                        "buffer_main_type_index",
                        state.buffer_main_type_index,
                    )
                ),
                "candidate_action_id": int(row * state.drive_type_count + drive_type_id),
            }
        )
        virtual_state.insert(candidate, bsu=bsu)
    return virtual_state
