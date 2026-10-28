import pytest
import torch
from torch import nn

from dreamplace.ops.buffer_insertion.discrete_candidate_scheduler import (
    schedule_typed_candidate_gradient_actions,
)
from dreamplace.ops.buffer_insertion.optimization_state import (
    TypedDiscreteCandidateState,
)


def test_typed_scheduler_selects_one_type_per_net_and_position():
    transition = schedule_typed_candidate_gradient_actions(
        candidate_net_index=torch.tensor([0, 0, 1], dtype=torch.long),
        net_ids=torch.tensor([10, 20], dtype=torch.long),
        candidate_ids=torch.tensor([100, 101, 200], dtype=torch.long),
        activations=torch.zeros((3, 4)),
        raw_grad=torch.tensor(
            [
                [-2.0, -4.0, -1.0, -3.0],
                [-5.0, -1.0, -2.0, -3.0],
                [-0.5, -0.25, -0.75, -1.0],
            ]
        ),
    )

    # The route-B fraction is 0.001, so two eligible nets produce one action.
    assert transition.accepted_count == 1
    assert transition.selected_candidate_ids.tolist() == [101]
    assert transition.selected_drive_type_ids.tolist() == [0]
    assert transition.next_activations[1].tolist() == [1.0, 0.0, 0.0, 0.0]
    assert transition.next_activations[[0, 2]].sum().item() == 0.0


def test_typed_scheduler_does_not_select_second_type_for_active_position():
    activations = torch.zeros((2, 4))
    activations[0, 2] = 1.0
    transition = schedule_typed_candidate_gradient_actions(
        candidate_net_index=torch.tensor([0, 0], dtype=torch.long),
        net_ids=torch.tensor([10], dtype=torch.long),
        candidate_ids=torch.tensor([100, 101], dtype=torch.long),
        activations=activations,
        raw_grad=torch.full((2, 4), -10.0),
    )

    assert transition.accepted_count == 1
    assert transition.selected_candidate_ids.tolist() == [101]
    assert transition.next_activations[0].tolist() == [0.0, 0.0, 1.0, 0.0]
    assert transition.next_activations[1].sum().item() == 1.0


def test_typed_scheduler_honors_selection_fraction():
    net_count = 1001
    raw_grad = torch.ones((net_count, 4))
    best_rows = torch.tensor([1000, 900, 800, 700, 600, 500])
    best_types = torch.tensor([3, 2, 1, 0, 3, 2])
    raw_grad[best_rows, best_types] = -torch.arange(6, 0, -1).float()
    transition = schedule_typed_candidate_gradient_actions(
        candidate_net_index=torch.arange(net_count, dtype=torch.long),
        net_ids=torch.arange(net_count, dtype=torch.long),
        candidate_ids=torch.arange(net_count, dtype=torch.long),
        activations=torch.zeros((net_count, 4)),
        raw_grad=raw_grad,
        selection_fraction=0.005,
    )

    assert transition.prefix_size == 6
    assert transition.accepted_count == 6
    torch.testing.assert_close(transition.selected_net_indices, best_rows)
    torch.testing.assert_close(transition.selected_drive_type_ids, best_types)


def test_typed_scheduler_rejects_invalid_selection_fraction():
    with pytest.raises(ValueError, match="selection_fraction"):
        schedule_typed_candidate_gradient_actions(
            candidate_net_index=torch.tensor([0], dtype=torch.long),
            net_ids=torch.tensor([10], dtype=torch.long),
            candidate_ids=torch.tensor([100], dtype=torch.long),
            activations=torch.zeros((1, 4)),
            raw_grad=torch.full((1, 4), -1.0),
            selection_fraction=0.0,
        )


@pytest.mark.parametrize(
    "activations,raw_grad,message",
    (
        (
            torch.tensor([[0.0, 0.5]]),
            torch.tensor([[-1.0, -1.0]]),
            "typed activations must be binary",
        ),
        (
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([[-1.0, -1.0]]),
            "typed activations violate position one-hot",
        ),
    ),
)
def test_typed_scheduler_rejects_non_binary_or_non_one_hot_state(
    activations,
    raw_grad,
    message,
):
    with pytest.raises(ValueError, match=message):
        schedule_typed_candidate_gradient_actions(
            candidate_net_index=torch.tensor([0], dtype=torch.long),
            net_ids=torch.tensor([10], dtype=torch.long),
            candidate_ids=torch.tensor([100], dtype=torch.long),
            activations=activations,
            raw_grad=raw_grad,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_typed_candidate_slice_moves_active_view_indices_to_cuda():
    device = torch.device("cuda")
    state = TypedDiscreteCandidateState(
        candidate_ids=torch.tensor([100, 101, 102], dtype=torch.long),
        candidate_node_id=torch.tensor([10, 11, 12], dtype=torch.long),
        candidate_net_id=torch.tensor([20, 21, 22], dtype=torch.long),
        candidate_net_index=torch.tensor([0, 1, 2], dtype=torch.long),
        net_ids=torch.tensor([20, 21, 22], dtype=torch.long),
        activation_param=nn.Parameter(
            torch.tensor(
                [[1.0, 0.0, 0.0, 0.0],
                 [0.0, 0.0, 1.0, 0.0],
                 [0.0, 0.0, 0.0, 0.0]],
                device=device,
            )
        ),
        drive_type_bsu_index=torch.tensor([0, 1, 2, 3], dtype=torch.long, device=device),
        drive_type_ids=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        drive_type_master_names=("B0", "B1", "B2", "B3"),
        drive_type_cell_ids=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        legal_buffer_count=4,
        buffer_main_type_index=0,
        candidate_records=(),
        drive_type_table_metadata={},
    )

    view = state.slice_candidates(
        torch.tensor([1, 2], dtype=torch.long),
        candidate_node_id=torch.tensor([11, 12], dtype=torch.long, device=device),
        device=device,
    )

    assert view.candidate_bu_tensor.device == device
    assert view.bsu_index_tensor.device == device
    assert view.candidate_bu_tensor.tolist() == [1.0, 0.0]
