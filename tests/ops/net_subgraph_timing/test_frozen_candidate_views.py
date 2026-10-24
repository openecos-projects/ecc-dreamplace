"""Frozen GP slices must never retain a graph or stale accepted actions."""

import pytest
import torch

from dreamplace.ops.buffer_insertion.drive_type_table import DriveTypeEntry, DriveTypeTable
from dreamplace.ops.buffer_insertion.optimization_state import build_typed_discrete_candidate_state
from dreamplace.ops.net_subgraph_timing.frozen_candidate_views import FrozenCandidateViews


def _state(device):
    candidates = tuple(
        {
            "candidate_id": i,
            "candidate_node_id": i,
            "net_id": i // 2,
            "buffer_main_type_index": 4,
            "candidate_location_x": float(i),
            "candidate_location_y": 0.0,
        }
        for i in range(3)
    )
    table = DriveTypeTable(
        entries=tuple(DriveTypeEntry(i, f"BUF{i}", i + 1.0, i, i) for i in range(4)),
        source="fixture",
    )
    state = build_typed_discrete_candidate_state(
        candidates,
        buffer_main_type_index=4,
        legal_buffer_count=4,
        drive_type_table=table,
        device=device,
    )
    with torch.no_grad():
        state.activation_param[0, 1] = 1
        state.activation_param[2, 2] = 1
    state.activation_param.requires_grad_(False)
    return state


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_cached_views_follow_action_and_index_edits(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    state = _state(device)
    cache = FrozenCandidateViews()
    indices = torch.tensor([2, 0])
    nodes = torch.tensor([8, 9])
    first = cache.get(state, indices=indices, nodes=nodes, device=device)
    assert cache.get(state, indices=indices, nodes=nodes, device=device) is first
    assert not first.candidate_bu().requires_grad
    torch.testing.assert_close(first.bsu_index().cpu(), torch.tensor([2.0, 1.0]))

    with torch.no_grad():
        state.activation_param[0, 1] = 0
        state.activation_param[0, 3] = 1
    updated = cache.get(state, indices=indices, nodes=nodes, device=device)
    assert updated is not first
    torch.testing.assert_close(updated.bsu_index().cpu(), torch.tensor([2.0, 3.0]))
    state.drive_type_bsu_index[3] = 0
    refreshed = cache.get(state, indices=indices, nodes=nodes, device=device)
    torch.testing.assert_close(refreshed.bsu_index().cpu(), torch.tensor([2.0, 0.0]))

    nodes[0] = 10
    changed_nodes = cache.get(state, indices=indices, nodes=nodes, device=device)
    torch.testing.assert_close(
        changed_nodes.candidate_node_id.cpu(),
        torch.tensor([10, 9], dtype=state.candidate_node_id.dtype),
    )
    indices[0] = 1
    changed_indices = cache.get(state, indices=indices, nodes=nodes, device=device)
    torch.testing.assert_close(changed_indices.candidate_bu().cpu(), torch.tensor([0.0, 1.0]))


def test_window_probes_bypass_cache_and_keep_activation_gradients():
    state = _state("cpu")
    cache = FrozenCandidateViews()
    indices = torch.tensor([2, 0])
    nodes = torch.tensor([8, 9])
    frozen = cache.get(state, indices=indices, nodes=nodes, device="cpu")
    state.activation_param.requires_grad_(True)
    assert cache.get(state, indices=indices, nodes=nodes, device="cpu") is None
    view = state.slice_candidates(indices, candidate_node_id=nodes, device="cpu")
    (gradient,) = torch.autograd.grad(view.candidate_bu().sum(), state.activation_param)
    torch.testing.assert_close(
        gradient, torch.tensor([[1.0, 1.0, 1.0, 1.0], [0.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]])
    )

    state.activation_param.requires_grad_(False)
    state.set_gradient_drive_type(0)
    assert cache.get(state, indices=indices, nodes=nodes, device="cpu") is None
    state.begin_shared_gradient_probe()
    assert cache.get(state, indices=indices, nodes=nodes, device="cpu") is None
    state.clear_gradient_drive_type()
    assert cache.get(state, indices=indices, nodes=nodes, device="cpu") is not frozen


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cached_views_are_local_to_their_cuda_stream():
    state = _state("cuda")
    cache = FrozenCandidateViews()
    indices = torch.tensor([2, 0])
    nodes = torch.tensor([8, 9])
    first = cache.get(state, indices=indices, nodes=nodes, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        second = cache.get(state, indices=indices, nodes=nodes, device="cuda")
        assert second is not first
        assert cache.get(state, indices=indices, nodes=nodes, device="cuda") is second
    stream.synchronize()
    torch.testing.assert_close(first.bsu_index(), second.bsu_index())
