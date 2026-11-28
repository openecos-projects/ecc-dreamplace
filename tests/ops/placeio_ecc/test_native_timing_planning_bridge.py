"""The production planning bridge must not materialize a Python carrier."""

import pytest
from dreamplace.flows.baselines.repair_buffering import resolve_repair_timing_config
from dreamplace.ops.placeio_ecc.repair_timing import plan_timing_repair
from dreamplace.ops.rebuffer.rebuffer_timing import RebufferPlanContext
from tests.ops.placeio_ecc.test_native_timing_evaluation import _session


def _forbidden(*_args, **_kwargs):
    raise AssertionError("native planning must not use the legacy carrier consumer")


def _config():
    return resolve_repair_timing_config(
        {
            "use_native_timing_graph": True,
            "use_native_timing_queue": True,
            "use_native_timing_evaluation": True,
            "use_native_timing_planning": True,
        }
    )


def test_native_planning_consumes_owned_queue_tree_and_projection(monkeypatch):
    model, session, bridge = _session()
    config = _config()
    context = RebufferPlanContext(model.pydb, model, config=config)
    for name in ("native_queue_views", "native_current_tree_payload", "map_native_tree_actions"):
        monkeypatch.setattr(context, name, _forbidden)
    monkeypatch.setattr(
        "dreamplace.ops.placeio_ecc.current_timing.evaluate_current_overlay", _forbidden
    )
    monkeypatch.setattr(model, "force_full_overlay_runtime", _forbidden)

    result = plan_timing_repair(session, model, max_passes=1, config=config, context=context)

    assert result["status"] == "ok"
    statistics = result["native_timing_graph_statistics"]
    assert statistics["planning_snapshot_count"] == 1
    assert statistics["python_materialization_count"] == 0
    assert statistics["native_planning"]["planned_requests"] > 0
    assert statistics["native_planning"]["imported_tree_count"] > 0
    assert model._overlay_runtime is None
    assert bridge.graph.state_view_statistics()["exports"] == 0


def test_native_planning_requires_native_primary_evaluation():
    model, session, _ = _session()
    model.use_native_timing_evaluation = False

    with pytest.raises(ValueError, match="requires native timing evaluation and queue"):
        plan_timing_repair(session, model, max_passes=1, config=_config())


def test_native_planning_inputs_reject_a_different_campaign_config():
    model, session, bridge = _session()
    config = _config()
    context = RebufferPlanContext(model.pydb, model, config=config)
    bridge.initialize_planning(context)
    another = RebufferPlanContext(model.pydb, model, config={**config, "tree_timing_rounds": 1})

    with pytest.raises(ValueError, match="same campaign configuration"):
        bridge.initialize_planning(another)


def test_native_planning_rejects_materialization_before_changing_state(monkeypatch):
    model, session, bridge = _session()
    context = RebufferPlanContext(model.pydb, model, config=_config())
    bridge.initialize_planning(context)
    checkpoint = bridge.graph.checkpoint()
    before = bridge.statistics()
    monkeypatch.setattr(
        "dreamplace.ops.placeio_ecc.current_timing.evaluate_current_overlay", _forbidden
    )

    with pytest.raises(RuntimeError, match="cannot materialize a Python timing carrier"):
        bridge.materialize(session.overlay)

    assert bridge.statistics() == before
    bridge.graph.restore(checkpoint)
    assert model._overlay_runtime is None
