import pytest
import torch

from dreamplace.ops.buffer_insertion.discrete_candidate_scheduler import (
    schedule_candidate_gradient_actions,
)


def test_candidate_scheduler_uses_per_net_best_and_one_tenth_percent_prefix():
    net_ids = torch.arange(1000, dtype=torch.long)
    candidate_net_index = torch.cat(
        (torch.tensor([0, 0], dtype=torch.long), torch.arange(1, 1000, dtype=torch.long))
    )
    candidate_ids = torch.cat(
        (torch.tensor([7, 3], dtype=torch.long), torch.arange(1000, 1999, dtype=torch.long))
    )
    activations = torch.ones(1001)
    activations[:2] = 0.0
    raw_grad = torch.zeros(1001)
    raw_grad[:2] = -2.0

    transition = schedule_candidate_gradient_actions(
        candidate_net_index=candidate_net_index,
        net_ids=net_ids,
        candidate_ids=candidate_ids,
        activations=activations,
        raw_grad=raw_grad,
    )

    assert transition.affected_net_count == 1000
    assert transition.prefix_size == 1
    assert transition.accepted_count == 1
    assert transition.selected_candidate_indices.tolist() == [1]
    assert transition.selected_candidate_ids.tolist() == [3]
    assert transition.selected_net_indices.tolist() == [0]
    expected = activations.clone()
    expected[1] = 1.0
    torch.testing.assert_close(transition.next_activations, expected)


def test_candidate_scheduler_excludes_active_candidate_and_can_choose_same_net_later():
    kwargs = {
        "candidate_net_index": torch.tensor([0, 0], dtype=torch.long),
        "net_ids": torch.tensor([10], dtype=torch.long),
        "candidate_ids": torch.tensor([1, 2], dtype=torch.long),
        "raw_grad": torch.tensor([-3.0, -2.0]),
    }
    first = schedule_candidate_gradient_actions(
        activations=torch.tensor([0.0, 0.0]),
        **kwargs,
    )
    second = schedule_candidate_gradient_actions(
        activations=first.next_activations,
        **kwargs,
    )

    assert first.selected_candidate_ids.tolist() == [1]
    assert second.selected_candidate_ids.tolist() == [2]
    torch.testing.assert_close(second.next_activations, torch.ones(2))


def test_candidate_scheduler_filters_nonpositive_prefix_without_filling_from_tail():
    transition = schedule_candidate_gradient_actions(
        candidate_net_index=torch.arange(1000, dtype=torch.long),
        net_ids=torch.arange(1000, dtype=torch.long),
        candidate_ids=torch.arange(1000, dtype=torch.long),
        activations=torch.zeros(1000),
        raw_grad=torch.ones(1000),
    )

    assert transition.prefix_size == 1
    assert transition.positive_prefix_count == 0
    assert transition.accepted_count == 0
    torch.testing.assert_close(transition.next_activations, torch.zeros(1000))


@pytest.mark.parametrize(
    "activations,raw_grad,message",
    (
        (torch.tensor([0.5]), torch.tensor([-1.0]), "activations must be binary"),
        (torch.tensor([0.0]), torch.tensor([float("nan")]), "raw_grad must be finite"),
    ),
)
def test_candidate_scheduler_rejects_invalid_state_without_mutation(
    activations,
    raw_grad,
    message,
):
    before = activations.clone()
    with pytest.raises(ValueError, match=message):
        schedule_candidate_gradient_actions(
            candidate_net_index=torch.tensor([0], dtype=torch.long),
            net_ids=torch.tensor([10], dtype=torch.long),
            candidate_ids=torch.tensor([1], dtype=torch.long),
            activations=activations,
            raw_grad=raw_grad,
        )
    torch.testing.assert_close(activations, before)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_candidate_scheduler_stays_on_cuda():
    device = torch.device("cuda")
    transition = schedule_candidate_gradient_actions(
        candidate_net_index=torch.tensor([0, 1], dtype=torch.long, device=device),
        net_ids=torch.tensor([10, 20], dtype=torch.long, device=device),
        candidate_ids=torch.tensor([1, 2], dtype=torch.long, device=device),
        activations=torch.zeros(2, device=device),
        raw_grad=torch.tensor([-2.0, -1.0], device=device),
    )

    assert transition.next_activations.device.type == "cuda"
    assert transition.selected_candidate_indices.device.type == "cuda"
