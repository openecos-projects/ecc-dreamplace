from types import SimpleNamespace

import pytest
from dreamplace.flows.diff_opt import DiffOptController
from dreamplace.flows.diff_opt_spec import parse_diff_opt_flows


def test_diff_opt_default_modes_are_step_local():
    mode, flows = parse_diff_opt_flows("buffering")
    assert mode == "buffering"
    assert [flow.to_list() for flow in flows] == [
        ["bf", 10, {"wns": 500.0, "tns": 5.0, "slew": 1.0, "cap": 1.0}]
    ]

    mode, flows = parse_diff_opt_flows("joint")
    assert mode == "joint"
    assert [flow.action for flow in flows] == ["sz", "bf"]


def test_diff_opt_rejects_mode_action_mismatch_and_empty_flow():
    with pytest.raises(ValueError, match="does not allow"):
        parse_diff_opt_flows("sizing", [["bf", 1, [1, 1, 1, 1]]])
    with pytest.raises(ValueError, match="non-empty"):
        parse_diff_opt_flows("joint", [])


def test_diff_opt_coefficients_are_canonicalized():
    _mode, flows = parse_diff_opt_flows(
        "joint",
        [["sz", 2, [100, 1, 1, 1]], ["bf", 1, {"wns": 500, "tns": 5}]],
    )
    assert flows[0].coefficients.to_dict() == {
        "wns": 100.0,
        "tns": 1.0,
        "slew": 1.0,
        "cap": 1.0,
    }
    assert flows[1].coefficients.to_dict() == {
        "wns": 500.0,
        "tns": 5.0,
        "slew": 1.0,
        "cap": 1.0,
    }


def test_diff_opt_buffer_phase_rebuilds_full_domain_and_binds_timing_loss():
    calls = []

    class Lane:
        timing_loss = None

        @staticmethod
        def _current_state():
            return object()

    lane = Lane()

    class Owner:
        def prepare(self, model, pos, *, active_only):
            calls.append(active_only)
            self.lane = lane

    controller = DiffOptController.__new__(DiffOptController)
    controller.owner = Owner()
    controller.params = SimpleNamespace(enable_relaxed_buffer_timing=False)
    def timing_loss(wns, tns, worst_slack, total_slack):
        return wns + tns
    model = SimpleNamespace(_timing_loss=timing_loss)

    controller._ensure_lane(model, object())

    assert calls == [False]
    assert lane.timing_loss is timing_loss
    assert controller.params.enable_relaxed_buffer_timing is True


def test_diff_opt_each_phase_uses_its_four_terms_and_restores_model_state(monkeypatch, tmp_path):
    import torch
    from dreamplace.flows import diff_opt
    from dreamplace.flows.discrete_timing_owner import DiscreteTimingOwner
    from dreamplace.PlaceObj import PlaceObj

    params = SimpleNamespace(
        placement_sizing_mode="place_only",
        timing_objective_lane="timing_only",
        differentiable_timing_obj=0,
        timing_opt_sizing_rounds=2,
        buffering_continuous_steps=3,
        diff_optimization_mode="joint",
    )
    data = SimpleNamespace(
        inst_cell_id=torch.tensor([0]),
        real_size=torch.nn.Parameter(torch.tensor([1.0]), requires_grad=False),
        vt_logits=torch.nn.Parameter(torch.zeros(1, 1), requires_grad=False),
    )
    data.get_continuous_size_parameter = lambda: data.real_size
    timing = SimpleNamespace(
        last_total_slew_violation_tensor=torch.tensor(1000.0),
        last_total_cap_violation_tensor=torch.tensor(1000.0),
    )
    model = PlaceObj.__new__(PlaceObj)
    model.params = params
    model.op_collections = SimpleNamespace(timing_propagation_op=timing)
    model.timing_wns_coeff, model.timing_tns_coeff = 7.0, 8.0
    model.timing_slew_weight, model.timing_cap_weight = 9.0, 10.0
    model.timing_grad_balance_weight, model.timing_leakage_weight = 11.0, 0.0
    expected_original = (7.0, 8.0, 9.0, 10.0, 11.0)
    model.timing_obj = lambda pos: tuple(torch.tensor(v) for v in (-2.0, -3.0, 0.0, 0.0))
    observed = []
    preparations = []

    class PhaseOwner(DiscreteTimingOwner):
        def __init__(self, *args, **kwargs):
            self.params, self.data, self.capacity = params, data, 100.0
            self.lane = None

        def prepare(self, model, pos, *, active_only):
            preparations.append(active_only)
            self.lane = SimpleNamespace(_current_state=lambda: object())

        def probe(self, model, pos, capacity):
            assert params.timing_objective_lane == "timing_slew_cap"
            assert params.differentiable_timing_obj == 1
            assert data.real_size.requires_grad and data.vt_logits.requires_grad
            loss = model._timing_loss(*model.timing_obj(pos))
            terms = model.last_timing_objective_terms
            observed.append(dict(terms["weighted_terms"]))
            assert terms["weighted_terms"]["slew"] > 0
            assert terms["weighted_terms"]["cap"] > 0
            return {"best_restore": {"best_score": float(loss)}}

        def run_sizing(self, model, pos, geometry, capacity):
            return self.probe(model, pos, capacity)

        def run_buffering(self, model, pos, capacity):
            import hashlib
            import json

            assert self.lane.timing_loss == model._timing_loss
            result = self.probe(model, pos, capacity)
            path = tmp_path / "candidate_discrete_scheduler_trace.json"
            path.write_text(json.dumps(observed[-1]))
            result.update(
                trace_path=str(path), trace_sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            )
            return result

        def retain_buffered_nets(self, model):
            pass

        def publish_geometry(self, model):
            return {"status": "ok"}

        def sync_native_sizing(self):
            return {"ok": True}

        def prepare_terminal_request(self, *args, **kwargs):
            return {"status": "request_ready"}

    monkeypatch.setattr(diff_opt, "DiscreteTimingOwner", PhaseOwner)
    monkeypatch.setattr(
        "dreamplace.ops.routability.cooptimization_area.SizingWindowGeometry",
        lambda *_args: None,
    )
    controller = DiffOptController(params, None, data, None, None)
    summary = controller.run(
        model,
        torch.zeros(2),
        flows=[["sz", 1, [100, 1, 2, 3]], ["bf", 1, [200, 2, 4, 6]], ["bf", 1, [300, 3, 6, 9]]],
    )
    assert len(observed) == 3
    assert preparations == [True, False, False]
    for name in ("timing", "slew", "cap"):
        assert [terms[name] for terms in observed] == pytest.approx(
            [observed[0][name] * factor for factor in (1, 2, 3)]
        )
    assert [phase["action"] for phase in summary["phases"]] == ["sz", "bf", "bf"]
    assert (
        model.timing_wns_coeff,
        model.timing_tns_coeff,
        model.timing_slew_weight,
        model.timing_cap_weight,
        model.timing_grad_balance_weight,
    ) == expected_original
    assert (
        params.timing_objective_lane,
        params.differentiable_timing_obj,
        params.placement_sizing_mode,
    ) == ("timing_only", 0, "place_only")
    assert not data.real_size.requires_grad and not data.vt_logits.requires_grad

    import hashlib
    from pathlib import Path

    traces = [phase for phase in summary["phases"] if phase["action"] == "bf"]
    assert len({phase["trace_path"] for phase in traces}) == 2
    for phase in traces:
        assert (
            hashlib.sha256(Path(phase["trace_path"]).read_bytes()).hexdigest()
            == phase["trace_sha256"]
        )
