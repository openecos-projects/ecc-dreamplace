import pytest
import torch

from dreamplace.ops.buffer_insertion.optimization_state import (
    build_discrete_candidate_state,
)


def _candidate(candidate_id, net_id, node_id, x_dbu):
    return {
        "candidate_id": candidate_id,
        "net_id": net_id,
        "candidate_node_id": node_id,
        "buffer_main_type_index": 4,
        "x_dbu": x_dbu,
        "y_dbu": 20,
    }


def test_discrete_candidate_state_starts_exactly_zero_with_compact_net_mapping():
    candidates = [
        _candidate(11, 5, 2, 10),
        _candidate(12, 5, 3, 20),
        _candidate(21, 7, 4, 30),
    ]
    state = build_discrete_candidate_state(
        candidates,
        buffer_main_type_index=4,
        legal_buffer_count=3,
        fixed_bsu_index=1,
    )

    assert state.activation_param.requires_grad
    torch.testing.assert_close(state.activation_param, torch.zeros(3))
    torch.testing.assert_close(state.candidate_bu(), torch.zeros(3))
    torch.testing.assert_close(state.bsu_index(), torch.ones(3))
    assert state.net_ids.tolist() == [5, 7]
    assert state.candidate_net_index.tolist() == [0, 0, 1]
    assert state.candidate_records[0] is candidates[0]
    state.validate_binary_()


def test_discrete_candidate_view_keeps_global_activation_gradient_identity():
    state = build_discrete_candidate_state(
        [
            _candidate(11, 5, 2, 10),
            _candidate(12, 5, 3, 20),
            _candidate(21, 7, 4, 30),
        ],
        buffer_main_type_index=4,
        legal_buffer_count=3,
        fixed_bsu_index=2,
    )
    indices = torch.tensor([2, 0], dtype=torch.long)
    view = state.slice_candidates(
        indices,
        candidate_node_id=torch.tensor([40, 20], dtype=torch.long),
        device=torch.device("cpu"),
    )

    assert view.candidate_ids.tolist() == [21, 11]
    assert view.candidate_node_id.tolist() == [40, 20]
    assert view.candidate_net_id.tolist() == [7, 5]
    torch.testing.assert_close(view.bsu_index(), torch.full((2,), 2.0))
    loss = (view.candidate_bu() * torch.tensor([3.0, 2.0])).sum()
    grad, = torch.autograd.grad(loss, state.activation_param)
    torch.testing.assert_close(grad, torch.tensor([2.0, 0.0, 3.0]))


@pytest.mark.parametrize(
    "candidates,fixed_bsu_index,message",
    (
        (
            [_candidate(11, 5, 2, 10), _candidate(11, 7, 4, 30)],
            1,
            "candidate_ids must be unique",
        ),
        (
            [{**_candidate(11, 5, 2, 10), "x_dbu": None}],
            1,
            "missing coordinate payload",
        ),
        (
            [_candidate(11, 5, 2, 10)],
            3,
            "fixed_bsu_index is out of range",
        ),
    ),
)
def test_discrete_candidate_state_rejects_invalid_identity_or_fixed_bsu(
    candidates,
    fixed_bsu_index,
    message,
):
    with pytest.raises(ValueError, match=message):
        build_discrete_candidate_state(
            candidates,
            buffer_main_type_index=4,
            legal_buffer_count=3,
            fixed_bsu_index=fixed_bsu_index,
        )
