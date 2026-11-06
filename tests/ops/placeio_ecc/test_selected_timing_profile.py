"""The selected provider reaches native timing and fresh commit evaluation."""

from types import SimpleNamespace

import pytest
from dreamplace.flows import repair_buffering
from dreamplace.ops.placeio_ecc.buffer_session import PlaceIOBufferSession
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from tests.ops.placeio_ecc.test_native_timing_campaign import _action
from tests.ops.placeio_ecc.test_native_timing_queue import _export
from tests.ops.placeio_ecc.test_repair_buffering_stage import (
    _ACTION,
    _Placedb,
    _install_common,
    _install_timing_plan,
)
from tests.ops.placeio_ecc.test_timing_carrier import _base


def test_timing_initial_and_verified_reader_share_selected_provider_and_wire_rc(monkeypatch):
    _install_common(monkeypatch)
    _, verified_calls = _install_timing_plan(monkeypatch, [_ACTION])
    captured = []

    def factory(pydb, **kwargs):
        captured.append((pydb, kwargs))
        return SimpleNamespace(baseline=SimpleNamespace(as_dict=lambda: {}))

    monkeypatch.setattr("dreamplace.ops.rebuffer.timing_model.RebufferTimingModel", factory)
    monkeypatch.setattr(
        "dreamplace.ops.placeio_ecc.electrical_observation.electrical_observation",
        lambda *args, **kwargs: {},
    )
    config = {"steiner_provider": "openroad_stt", "r_unit": 0.653, "c_unit": 1.8889e-4}
    result = repair_buffering.run_repair_buffering_stage(
        _Placedb(), mode="repair_timing", config=config
    )
    verified_calls[0][1]["reader"]("committed")

    assert result["status"] == "ok"
    assert captured == [
        (
            "base",
            {
                "use_pin_offset": False,
                "use_persistent_graph": False,
                "use_native_timing_graph": True,
                "use_native_timing_evaluation": True,
                **config,
            },
        ),
        ("committed", {"use_pin_offset": False, **config}),
    ]


@pytest.mark.parametrize("native_graph", [False, True])
def test_selected_timing_candidate_and_materialized_fresh_model_keep_provider(native_graph):
    model = RebufferTimingModel(
        _base(),
        steiner_provider="openroad_stt",
        use_pin_offset=False,
        use_native_timing_graph=native_graph,
        use_native_timing_evaluation=native_graph,
    )
    session = PlaceIOBufferSession(SimpleNamespace(pydb=model.pydb), timing_model=model)
    session.begin()
    root = session.checkpoint()["checkpoint_id"]
    initial = session.evaluate()
    session.apply([_action(session.overlay)])
    current = session.evaluate()
    runtime = model.force_full_overlay_runtime()
    assert runtime.model.steiner_provider == "openroad_stt"
    assert runtime.model.use_pin_offset is False
    expected = RebufferTimingModel(
        runtime.model.pydb, steiner_provider="openroad_stt", use_pin_offset=False
    )
    assert runtime.model.baseline.as_dict() == pytest.approx(expected.baseline.as_dict(), abs=1e-6)
    if native_graph:
        _, frozen, _ = _export(model, session.overlay, session.overlay.native_timing_bridge)
        assert {key: current[key] for key in frozen.baseline.as_dict()} == pytest.approx(
            frozen.baseline.as_dict(), abs=1e-6
        )
    session.restore(root)
    restored = session.evaluate()
    assert {key: restored[key] for key in model.baseline.as_dict()} == {
        key: initial[key] for key in model.baseline.as_dict()
    }


def test_selected_fresh_tp_supports_a_buffer_inserted_at_its_driver():
    model = RebufferTimingModel(_base(), steiner_provider="openroad_stt", use_pin_offset=False)
    session = PlaceIOBufferSession(SimpleNamespace(pydb=model.pydb), timing_model=model)
    session.begin()
    action = _action(session.overlay)
    action["candidate_location_x_dbu"] = 10000
    session.apply([action])
    session.evaluate()
    runtime = model.force_full_overlay_runtime()
    fresh = RebufferTimingModel(
        runtime.model.pydb, steiner_provider="openroad_stt", use_pin_offset=False
    )
    assert fresh.baseline == runtime.model.baseline
    assert float(fresh.loads["rise"][2]) > 0
