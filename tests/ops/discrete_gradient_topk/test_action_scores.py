from types import SimpleNamespace

import torch

from dreamplace.ops.discrete_gradient_topk.quad_gradient_topk import (
    ACTION_NAMES,
    QuadCandidateTable,
    apply_quad_gradient_from_action_scores,
)


def test_action_scores_are_consumed_without_size_delta_scaling():
    info = torch.tensor(
        [
            [0.0, 10.0, 1.0, 0.0],
            [1.0, 10.0, 2.0, 0.0],
            [2.0, 10.0, 1.0, 1.0],
            [3.0, 10.0, 2.0, 1.0],
        ],
        dtype=torch.float64,
    )
    candidates = QuadCandidateTable(
        target_cell_ids=torch.tensor(
            [
                [1, -1, 2, -1],
                [-1, 0, 3, 2],
                [3, -1, -1, 0],
                [-1, 2, -1, 1],
            ]
        )
    )
    data = SimpleNamespace(
        inst_cell_id=torch.tensor([0, 1]),
        inst_is_sizeable=torch.ones(2, dtype=torch.bool),
        flat_libcell_info=info,
        inst_size_lower=torch.zeros(2, dtype=torch.float64),
        inst_size_upper=torch.full((2,), 2.0, dtype=torch.float64),
    )
    scores = torch.full((2, 4), float("inf"), dtype=torch.float64)
    scores[0, 0] = -5.0
    scores[1, 3] = -3.0
    valid = torch.isfinite(scores)

    next_size, next_vt, summary = apply_quad_gradient_from_action_scores(
        action_scores=scores,
        valid_instances=valid,
        data_collections=data,
        size_logits=torch.tensor([1.0, 1.0], dtype=torch.float64),
        vt_logits=torch.zeros((2, 2), dtype=torch.float64),
        candidate_table=candidates,
        shared_budget_percent=50.0,
    )

    torch.testing.assert_close(next_size, torch.tensor([2.0, 1.0], dtype=torch.float64))
    assert next_vt.argmax(dim=1).tolist() == [0, 0]
    assert summary["num_changed_instances"] == 1
    assert summary["applied_instance_ids"] == [0]
    assert summary["applied_cell_ids"] == [1]
    assert summary["applied_actions"] == ["size_up"]
    assert summary["ranking_mode"] == "lut_action_delta"


def test_invalid_action_score_is_never_selected():
    info = torch.tensor(
        [
            [0.0, 10.0, 1.0, 0.0],
            [1.0, 10.0, 2.0, 0.0],
        ],
        dtype=torch.float64,
    )
    candidates = QuadCandidateTable(
        target_cell_ids=torch.tensor([[1, -1, -1, -1], [-1, 0, -1, -1]])
    )
    data = SimpleNamespace(
        inst_cell_id=torch.tensor([0]),
        inst_is_sizeable=torch.ones(1, dtype=torch.bool),
        flat_libcell_info=info,
        inst_size_lower=torch.zeros(1, dtype=torch.float64),
        inst_size_upper=torch.ones(1, dtype=torch.float64),
    )
    scores = torch.tensor([[-100.0, -100.0, -100.0, -100.0]], dtype=torch.float64)
    valid = torch.zeros((1, 4), dtype=torch.bool)

    _, _, summary = apply_quad_gradient_from_action_scores(
        action_scores=scores,
        valid_instances=valid,
        data_collections=data,
        size_logits=torch.zeros(1, dtype=torch.float64),
        vt_logits=torch.zeros((1, 2), dtype=torch.float64),
        candidate_table=candidates,
        shared_budget_percent=100.0,
    )

    assert summary["num_changed_instances"] == 0
    assert summary["applied_actions"] == []
    assert ACTION_NAMES == ("size_up", "size_down", "vt_slowdown", "vt_speedup")


def test_temporary_cooldown_does_not_mutate_action_legality():
    info = torch.tensor([[0.0, 10.0, 1.0, 0.0], [1.0, 10.0, 2.0, 0.0]])
    candidates = QuadCandidateTable(
        target_cell_ids=torch.tensor([[1, -1, -1, -1], [-1, 0, -1, -1]])
    )
    data = SimpleNamespace(
        inst_cell_id=torch.tensor([0]),
        inst_is_sizeable=torch.ones(1, dtype=torch.bool),
        flat_libcell_info=info,
    )
    valid = torch.tensor([[True, False, False, False]])
    original_valid = valid.clone()
    summaries = []
    for blocked in (True, False):
        _, _, summary = apply_quad_gradient_from_action_scores(
            action_scores=torch.tensor([[-1.0, 0.0, 0.0, 0.0]]),
            valid_instances=valid,
            data_collections=data,
            size_logits=torch.tensor([1.0]),
            vt_logits=torch.zeros((1, 2)),
            candidate_table=candidates,
            shared_budget_percent=100.0,
            blocked_instances=torch.tensor([blocked]),
        )
        summaries.append(summary["applied_actions"])
    assert summaries == [[], ["size_up"]]
    torch.testing.assert_close(valid, original_valid, atol=0, rtol=0)
