"""Reuse constant candidate slices while discrete activations are frozen in GP."""

from typing import NamedTuple

import torch

from dreamplace.ops.buffer_insertion.optimization_state import (
    CandidateStateView,
    TypedDiscreteCandidateState,
)


class _Entry(NamedTuple):
    indices: torch.Tensor
    nodes: torch.Tensor
    value: CandidateStateView


def _signature(tensor):
    return (
        id(tensor),
        tensor.data_ptr(),
        tensor._version,
        tuple(tensor.shape),
        tuple(tensor.stride()),
        tensor.device,
        tensor.dtype,
    )


class FrozenCandidateViews:
    def __init__(self):
        self._sources = ()
        self._signature = None
        self._views = {}

    def get(self, state, *, indices, nodes, device):
        if (
            not isinstance(state, TypedDiscreteCandidateState)
            or state.activation_param.requires_grad
            or state._shared_gradient_probe
            or state._gradient_drive_type is not None
        ):
            self._views.clear()
            self._sources = ()
            self._signature = None
            return None
        sources = (
            state.activation_param,
            state.drive_type_bsu_index,
            state.candidate_ids,
            state.candidate_net_id,
            state.candidate_node_id,
        )
        if any(value.is_inference() for value in (*sources, indices, nodes)):
            return None
        signature = (
            id(state),
            state.legal_buffer_count,
            state.buffer_main_type_index,
            tuple(_signature(value) for value in sources),
        )
        if signature != self._signature:
            self._views.clear()
            self._sources = sources
            self._signature = signature
        device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        stream = torch.cuda.current_stream(device).cuda_stream if device.type == "cuda" else None
        key = (_signature(indices), _signature(nodes), device, stream)
        entry = self._views.get(key)
        if entry is None:
            view = state.slice_candidates(
                indices, candidate_node_id=nodes.to(device=device), device=device
            )
            entry = _Entry(indices, nodes, view)
            self._views[key] = entry
        return entry.value
