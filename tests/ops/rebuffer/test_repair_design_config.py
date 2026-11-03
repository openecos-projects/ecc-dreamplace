"""The direct planner builds Python TP with the same RC as its native inputs."""

from types import SimpleNamespace

import pytest
from dreamplace.ops.rebuffer import repair_design
from test_repair_design import _characterization, _design, _liberty, _steiner, _timing


@pytest.mark.parametrize("supplied_model", [False, True])
def test_direct_planner_and_timing_model_share_wire_rc(monkeypatch, supplied_model):
    pydb = SimpleNamespace(
        net_names=["n"],
        pin_names=["driver:Y", "load:A"],
        node_names=["driver", "load"],
        flat_libcell_names=["DRV", "BUF", "SINK"],
    )
    captured = []

    def model_factory(current_pydb, **kwargs):
        captured.append((current_pydb, kwargs))
        return SimpleNamespace(baseline=SimpleNamespace(as_dict=lambda: {}))

    monkeypatch.setattr(repair_design, "RebufferTimingModel", model_factory)
    monkeypatch.setattr(
        repair_design, "build_liberty_view_from_pydb", lambda _: {"status": "ok", **_liberty()}
    )
    monkeypatch.setattr(
        repair_design,
        "characterize_buffers_from_pydb",
        lambda _: {"status": "ok", **_characterization()},
    )
    monkeypatch.setattr(
        repair_design,
        "steiner_payload",
        lambda *args, **kwargs: {"status": "ok", **_steiner()},
    )
    monkeypatch.setattr(repair_design, "timing_payload", lambda *args, **kwargs: _timing())
    design = _design()
    design["status"] = "ok"
    monkeypatch.setattr(repair_design, "build_rebuffer_design_from_pydb", lambda _: design)
    wire_rc = {"r_unit": 0.653, "c_unit": 1.8889e-4}
    model = (
        SimpleNamespace(**wire_rc, baseline=SimpleNamespace(as_dict=lambda: {}))
        if supplied_model
        else None
    )

    result = repair_design.plan_repair_design_from_pydb(
        pydb, model, config=None if supplied_model else wire_rc
    )

    assert result["status"] == "ok", result
    assert captured == ([] if supplied_model else [(pydb, wire_rc)])
    assert {key: result["config"][key] for key in wire_rc} == wire_rc
    if supplied_model:
        with pytest.raises(ValueError, match="r_unit does not match its timing model"):
            repair_design.plan_repair_design_from_pydb(pydb, model, config={"r_unit": 0.914})
