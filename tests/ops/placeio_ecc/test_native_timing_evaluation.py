"""Candidate evaluation can use the native TP owner before carrier materialization."""

from types import SimpleNamespace

import pytest
from dreamplace.flows.baselines.repair_buffering import resolve_repair_timing_config
from dreamplace.ops.placeio_ecc.buffer_session import PlaceIOBufferSession
from dreamplace.ops.placeio_ecc.repair_timing import plan_timing_repair
from dreamplace.ops.rebuffer.timing_model import RebufferTimingModel
from tests.ops.placeio_ecc.test_native_timing_campaign import _action
from tests.ops.placeio_ecc.test_native_timing_queue import _export
from tests.ops.placeio_ecc.test_timing_carrier import _base


def _session():
    model = RebufferTimingModel(
        _base(),
        use_pin_offset=False,
        use_native_timing_graph=True,
        use_native_timing_evaluation=True,
    )
    session = PlaceIOBufferSession(SimpleNamespace(pydb=model.pydb), timing_model=model)
    session.begin()
    return model, session, session.overlay.native_timing_bridge


def _forbidden(*_args, **_kwargs):
    raise AssertionError("native candidate evaluation must not materialize Python TP")


def test_native_candidate_evaluation_and_restore_do_not_materialize(monkeypatch):
    model, session, bridge = _session()
    root = session.checkpoint()["checkpoint_id"]
    original = session.evaluate()
    monkeypatch.setattr(
        "dreamplace.ops.placeio_ecc.current_timing.evaluate_current_overlay", _forbidden
    )
    session.apply([_action(session.overlay)])

    observation = session.evaluate()
    _, fresh, _ = _export(model, session.overlay, bridge)

    assert {key: observation[key] for key in fresh.baseline.as_dict()} == pytest.approx(
        fresh.baseline.as_dict(), abs=1e-6, rel=0
    )
    assert observation["topology_source"] == "native_timing_graph"
    assert model._overlay_runtime is None
    assert bridge.statistics()["python_materialization_count"] == 0
    stats = bridge.statistics()
    assert stats["evaluation_counts"]["full_seed"] == 1
    assert stats["evaluation_counts"]["events"] > 0
    assert stats["full_fallback_count"] == 0
    assert sum(stats["evaluation_counts"].values()) == stats["propagation_count"]
    session.restore(root)
    restored = session.evaluate()
    assert {key: restored[key] for key in model.baseline.as_dict()} == {
        key: original[key] for key in model.baseline.as_dict()
    }


def test_native_full_seed_fallback_records_reason_epoch_and_phase_without_materializing():
    model, session, bridge = _session()
    before = bridge.statistics()
    bridge.graph.set_state({"pin_rAAT": bridge.graph.state_view()["pin_rAAT"] + 1.0})

    result = bridge.evaluate(session.overlay)

    assert result["metrics"] == pytest.approx(model.baseline.as_dict(), abs=1e-6, rel=0)
    assert result["fallback_reason"] == "native_full_seed_baseline_required"
    stats = bridge.statistics()
    assert stats["full_fallback_count"] == 1
    assert stats["python_materialization_count"] == 0
    fallback = stats["full_fallbacks"][0]
    assert {key: value for key, value in fallback.items() if key != "elapsed_ms"} == {
        "reason": result["fallback_reason"],
        "epoch": result["epoch"],
        "phase": "native_propagation",
    }
    assert fallback["elapsed_ms"] >= 0
    assert stats["propagation_count"] == before["propagation_count"] + 1
    fallback["reason"] = "caller mutation"
    assert bridge.statistics()["full_fallbacks"][0]["reason"] == result["fallback_reason"]


def test_native_restore_uses_state_api_and_retains_explicit_campaign_diagnostics(monkeypatch):
    from tests.ops.rebuffer.test_persistent_timing_resize import _same_state

    model, session, bridge = _session()
    root = session.checkpoint()["checkpoint_id"]
    before = bridge.graph.state_view()
    active_ledger = bridge.campaign.summary()["action_ledger"]

    def forbidden(*_args, **_kwargs):
        raise AssertionError("native rollback requested serialized campaign restore")

    monkeypatch.setattr(type(bridge.campaign), "restore", forbidden)
    for variant in range(50):
        bridge.campaign.record_candidate(0, {"variant_index": variant}, "screened")
    session.apply([_action(session.overlay)])
    session.evaluate()

    session.restore(root)

    _same_state(bridge.graph.state_view(), before, bitwise=True)
    observation = session.evaluate()
    assert {key: observation[key] for key in model.baseline.as_dict()} == model.baseline.as_dict()
    summary = bridge.campaign.summary()
    assert summary["action_ledger"] == active_ledger
    assert len(summary["ledger_history"]) >= 50
    assert bridge.statistics()["python_materialization_count"] == 0


def test_python_carrier_is_materialized_only_when_explicitly_requested():
    model, session, bridge = _session()
    session.apply([_action(session.overlay)])
    session.evaluate()
    assert model._overlay_runtime is None

    runtime = model.current_graph()

    assert runtime is not None
    assert len(runtime.model.pydb.pin_names) == 6
    assert model.current_graph() is runtime
    assert bridge.statistics()["python_materialization_count"] == 1
    assert model.force_full_overlay_runtime() is not None
    assert bridge.statistics()["python_materialization_count"] == 2


def test_native_repair_evaluation_skips_legacy_electrical_and_slack_callbacks(monkeypatch):
    model, session, bridge = _session()
    monkeypatch.setattr("dreamplace.ops.placeio_ecc.repair_timing._slack_vector", _forbidden)
    monkeypatch.setattr(
        "dreamplace.ops.placeio_ecc.electrical_observation.electrical_observation", _forbidden
    )

    result = plan_timing_repair(
        session,
        model,
        max_passes=1,
        config=resolve_repair_timing_config(
            {
                "use_native_timing_graph": True,
                "use_native_timing_queue": True,
                "use_native_timing_evaluation": True,
            }
        ),
    )

    assert result["status"] == "ok"
    assert result["baseline"]["electrical"] == bridge.graph.evaluation_observation()["electrical"]
    assert (
        result["native_timing_graph_statistics"]["mode"]
        == "native_campaign_with_native_timing_consumer"
    )


def test_native_evaluation_requires_native_graph_owner():
    with pytest.raises(ValueError, match="requires a native timing graph"):
        RebufferTimingModel(_base(), use_pin_offset=False, use_native_timing_evaluation=True)


def test_native_repair_evaluation_requires_native_queue():
    model, session, _ = _session()

    with pytest.raises(ValueError, match="requires the native timing queue"):
        plan_timing_repair(session, model, max_passes=1, config={})
