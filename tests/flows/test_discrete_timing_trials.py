"""One-action diagnostics keep selection and metric derivatives separate."""

from types import SimpleNamespace

import torch
from dreamplace.flows.discrete_timing_owner import DiscreteTimingOwner
from dreamplace.ops.discrete_gradient_topk.oscillation import CellOscillationState


def test_trial_ranks_by_selection_but_compares_the_tns_loss_derivative(monkeypatch):
    owner = DiscreteTimingOwner.__new__(DiscreteTimingOwner)
    size = torch.zeros(2)
    owner.data = SimpleNamespace(
        inst_cell_id=torch.tensor([0, 2]),
        inst_is_sizeable=torch.ones(2, dtype=torch.bool),
        vt_logits=torch.zeros(2, 3),
        get_continuous_size_parameter=lambda: size,
    )
    owner.placedb = SimpleNamespace(flat_libcell_names=["A", "A_up", "B", "B_up"])
    owner.params = SimpleNamespace(
        discrete_gradient_topk_shared_budget_percent=1.0,
        timing_action_trial_candidates=1,
    )
    owner.oscillation = CellOscillationState()
    owner.refresh_candidates_after_sizing = False
    model = SimpleNamespace(_diff_opt_last_timing_metrics={"tns": -1000.0, "wns": -10.0})
    context = SimpleNamespace(
        valid_instances=torch.tensor([[True, False, False, False]] * 2),
        target_cell_ids=torch.tensor([[1, -1, -1, -1], [3, -1, -1, -1]]),
    )
    tns_scores = torch.tensor([[-20.0, 0.0, 0.0, 0.0], [6.0, 0.0, 0.0, 0.0]])
    selection_scores = torch.tensor([[-1.0, 0.0, 0.0, 0.0], [-40.0, 0.0, 0.0, 0.0]])
    wns_scores = torch.tensor([[1.0, 0.0, 0.0, 0.0], [-2.0, 0.0, 0.0, 0.0]])

    def sync(_model, summary, _geometry):
        owner.data.inst_cell_id[summary["applied_instance_ids"]] = torch.tensor(
            summary["applied_cell_ids"]
        )

    def loss(current, _pos):
        assert owner.data.inst_cell_id.tolist() == [0, 3]
        current._diff_opt_last_timing_metrics = {"tns": -1006.0, "wns": -12.0}
        return torch.tensor(11030.0)

    monkeypatch.setattr(owner, "_sync", sync)
    monkeypatch.setattr(owner, "_loss", loss)
    monkeypatch.setattr(owner, "_trial_action_features", lambda *_args: {})
    pos = torch.tensor([1.0, 2.0])
    result = owner._trial_tns_action_scores(
        model,
        pos,
        context,
        tns_scores,
        geometry=None,
        base_loss=10000.0,
        selection_scores=selection_scores,
        wns_scores=wns_scores,
    )

    assert result["trial_records"] == [
        {
            "instance_id": 1,
            "action": "size_up",
            "local_score": 6.0,
            "selection_score": -40.0,
            "local_wns_score": -2.0,
            "old_cell": 2,
            "target_cell": 3,
            "old_cell_name": "B",
            "target_cell_name": "B_up",
            "base_tns": -1000.0,
            "trial_tns": -1006.0,
            "delta_tns": 6.0,
            "base_wns": -10.0,
            "trial_wns": -12.0,
            "delta_wns": -2.0,
            "delta_wns_loss": 2.0,
            "base_loss": 10000.0,
            "trial_loss": 11030.0,
            "delta_loss": -1030.0,
            "features": {},
            "trial_features": {},
        }
    ]
    assert result["accuracy"]["size_up"] == {
        "sample_count": 1,
        "nonzero_exact_count": 1,
        "sign_ok": 1,
        "sign_zero": 0,
        "sign_bad": 0,
        "median_abs_error": 0.0,
        "median_relative_error": 0.0,
        "exact_improve_100_count": 0,
        "exact_worsen_100_count": 0,
        "local_zero_worsen_100_count": 0,
    }
    torch.testing.assert_close(owner.data.inst_cell_id, torch.tensor([0, 2]))
    torch.testing.assert_close(pos, torch.tensor([1.0, 2.0]))
