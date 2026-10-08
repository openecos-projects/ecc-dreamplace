import pytest
import torch

from dreamplace.ops.buffer_insertion.discrete_virtual_scheduler import (
    schedule_net_gradient_actions,
    take_transition_prefix,
    transition_backtracking_sizes,
)


def test_scheduler_uses_one_tenth_percent_affected_net_prefix_and_segment_id_tiebreak():
    net_ids = torch.arange(1000, dtype=torch.long)
    segment_net_index = torch.cat(
        (torch.tensor([0, 0], dtype=torch.long), torch.arange(1, 1000, dtype=torch.long))
    )
    segment_ids = torch.cat(
        (torch.tensor([7, 3], dtype=torch.long), torch.arange(1000, 1999, dtype=torch.long))
    )
    counts = torch.full((1001,), 3.0)
    counts[:2] = 0.0
    raw_grad = torch.zeros(1001)
    raw_grad[:2] = -2.0

    transition = schedule_net_gradient_actions(
        segment_net_index=segment_net_index,
        net_ids=net_ids,
        segment_ids=segment_ids,
        counts=counts,
        raw_grad=raw_grad,
        max_count=3,
    )

    assert transition.affected_net_count == 1000
    assert transition.prefix_size == 1
    assert transition.accepted_count == 1
    assert transition.selected_segment_indices.tolist() == [1]
    assert transition.selected_segment_ids.tolist() == [3]
    torch.testing.assert_close(
        transition.next_counts,
        torch.cat((torch.tensor([0.0, 1.0]), torch.full((999,), 3.0))),
    )


def test_scheduler_filters_nonpositive_scores_and_preserves_input_counts():
    net_ids = torch.arange(100, dtype=torch.long)
    segment_net_index = torch.arange(100, dtype=torch.long)
    segment_ids = torch.arange(100, dtype=torch.long)
    counts = torch.zeros(100)
    counts[0] = 3.0
    counts_before = counts.clone()
    raw_grad = torch.ones(100)
    raw_grad[0] = -10.0
    raw_grad[1] = 0.0

    transition = schedule_net_gradient_actions(
        segment_net_index=segment_net_index,
        net_ids=net_ids,
        segment_ids=segment_ids,
        counts=counts,
        raw_grad=raw_grad,
        max_count=3,
    )

    assert transition.prefix_size == 1
    assert transition.accepted_count == 0
    torch.testing.assert_close(counts, counts_before)
    torch.testing.assert_close(transition.next_counts, counts_before)


def test_scheduler_uses_explicit_affected_net_selection_fraction():
    net_count = 1000
    transition = schedule_net_gradient_actions(
        segment_net_index=torch.arange(net_count, dtype=torch.long),
        net_ids=torch.arange(net_count, dtype=torch.long),
        segment_ids=torch.arange(net_count, dtype=torch.long),
        counts=torch.zeros(net_count),
        raw_grad=-torch.arange(1, net_count + 1, dtype=torch.float32),
        max_count=3,
        selection_fraction=0.01,
    )

    assert transition.prefix_size == 10
    assert transition.accepted_count == 10


def test_scheduler_rejects_nonfinite_gradients_without_mutating_input():
    counts = torch.tensor([0.0])
    with pytest.raises(ValueError, match="raw_grad must be finite"):
        schedule_net_gradient_actions(
            segment_net_index=torch.tensor([0], dtype=torch.long),
            net_ids=torch.tensor([10], dtype=torch.long),
            segment_ids=torch.tensor([5], dtype=torch.long),
            counts=counts,
            raw_grad=torch.tensor([float("nan")]),
            max_count=3,
        )
    torch.testing.assert_close(counts, torch.tensor([0.0]))


def test_transition_prefix_preserves_ranked_action_order():
    counts = torch.zeros(2000)
    transition = schedule_net_gradient_actions(
        segment_net_index=torch.arange(2000, dtype=torch.long),
        net_ids=torch.arange(2000, dtype=torch.long),
        segment_ids=torch.arange(2000, dtype=torch.long),
        counts=counts,
        raw_grad=torch.cat((torch.tensor([-2.0, -1.0]), torch.ones(1998))),
        max_count=3,
    )

    prefix = take_transition_prefix(transition, counts=counts, selected_count=1)

    assert transition.accepted_count == 2
    assert prefix.accepted_count == 1
    assert prefix.selected_segment_ids.tolist() == [0]
    assert prefix.next_counts[:2].tolist() == [1.0, 0.0]
    assert transition_backtracking_sizes(28) == (28, 14, 7, 4, 2, 1)
