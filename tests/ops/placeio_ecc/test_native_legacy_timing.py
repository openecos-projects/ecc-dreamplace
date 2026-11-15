"""The LEGACY production entry keeps queries, candidates and rollback native."""

import numpy as np
import pytest
from dreamplace.flows.baselines.repair_buffering import resolve_repair_timing_config
from dreamplace.ops.placeio_ecc.repair_timing import plan_timing_repair
from dreamplace.ops.rebuffer import rebuffer_cpp
from tests.ops.placeio_ecc.test_native_timing_evaluation import _session
from tests.ops.placeio_ecc.test_timing_carrier import _base
from tests.ops.rebuffer.test_persistent_timing_queue import _initialize
from tests.ops.rebuffer.test_persistent_timing_resize import _graph


def _config(**extra):
    return resolve_repair_timing_config(
        {
            "timing_scheduler": "openroad_legacy",
            "use_native_timing_graph": True,
            "use_native_timing_queue": True,
            "use_native_timing_evaluation": True,
            "use_native_timing_planning": True,
            "legacy_max_passes_per_endpoint": 2,
            **extra,
        }
    )


def test_production_legacy_entry_uses_native_owner_without_materialization(monkeypatch):
    model, session, bridge = _session()

    def forbidden(*_args, **_kwargs):
        pytest.fail("LEGACY called Python carrier or path materialization")

    monkeypatch.setattr(
        "dreamplace.ops.placeio_ecc.current_timing.evaluate_current_overlay", forbidden
    )
    monkeypatch.setattr(model, "native_setup_path_details", forbidden)
    monkeypatch.setattr(model, "force_full_overlay_runtime", forbidden)
    result = plan_timing_repair(session, model, max_passes=3, config=_config())

    assert result["status"] == "ok"
    assert result["scheduler"] == "openroad_legacy"
    assert result["native_candidate_planner"]
    assert result["attempt_count"] <= 3
    statistics = result["native_timing_graph_statistics"]
    assert statistics["python_materialization_count"] == statistics["full_fallback_count"] == 0
    assert bridge.graph.state_view_statistics()["exports"] == 0
    for row in result["passes"]:
        if "path_statistics" in row:
            assert row["path_statistics"]["traced_endpoint_count"] <= 1
            assert row["path_statistics"]["active_endpoint_count"] <= 1
    assert any(
        "score_ps" in driver for row in result["passes"] for driver in row.get("drivers", [])
    )
    # Trial/root checkpoints are owned by the native scheduler only for their
    # restore window.  A long LEGACY campaign must not retain one Python
    # overlay checkpoint per candidate.
    assert session.overlay._checkpoints == {}


def test_filtered_path_query_matches_full_queue_and_rejects_nonendpoint():
    base = _base()
    base.end_points = base.endpoint_pin_ids = [1, 3]
    base.endpoints_rRAT = base.endpoints_fRAT = [-1000.0, 0.0]
    base.outcaps = [0.0, 0.0]
    base.endpoints_max_valid = np.ones((2, 2), dtype=bool)
    base.endpoints_max_reason = np.zeros((2, 2), dtype=np.int32)
    owner, graph = _graph(base)
    graph.propagate()
    _initialize(owner, graph)
    full = rebuffer_cpp.select_timing_queue_items(graph.timing_queue_state([0, 1]), {})

    for endpoint in (1, 3):
        selected = graph.timing_queue_state([0, 1], endpoint_pin=endpoint)
        rows = rebuffer_cpp.select_timing_queue_items(selected, {})
        # Snapshot positions are implementation indices; all semantic fields match.
        assert rows == [row for row in full if row["endpoint_pin"] == endpoint]
        assert selected["graph_queue_statistics"]["active_endpoint_count"] == 1
    with pytest.raises(ValueError, match="not a timing endpoint"):
        graph.timing_queue_state([0, 1], endpoint_pin=0)


@pytest.mark.parametrize(
    "override",
    [
        {"use_native_timing_graph": False},
        {"use_native_timing_evaluation": False},
        {"repair_gradient_ordering": True},
        {"max_repairs_per_pass": 2},
    ],
)
def test_legacy_rejects_incompatible_consumer_or_policy(override):
    with pytest.raises(ValueError):
        _config(**override)


def test_legacy_resize_drivers_upsizes_driver_and_restores_cleanly():
    model, session, bridge = _session()
    root = session.checkpoint()["checkpoint_id"]
    baseline_obs = session.evaluate()

    result = plan_timing_repair(
        session,
        model,
        max_passes=1,
        config=_config(resize_drivers=True),
    )

    assert result["status"] == "ok"
    assert result["final"]["wns_ps"] > baseline_obs["wns_ps"]
    commit_actions = session.overlay.commit_actions()
    resize_actions = [
        {k: v for k, v in a.items() if k != "tree_bundle_id"}
        for a in commit_actions
        if a.get("action_kind") == "resize_cell"
    ]
    assert resize_actions == [
        {
            "action_kind": "resize_cell",
            "commit_order": 0,
            "driver_pin": 2,
            "net_id": 1,
            "net_name": "n1",
            "node_id": 0,
            "instance_name": "u",
            "previous_cell_index": 0,
            "previous_master_name": "BUF1",
            "target_cell_index": 1,
            "target_master_name": "BUF2",
        }
    ]
    assert bridge.graph.mutation_view()["node_master_names"][0] == "BUF2"

    session.restore(root)
    assert session.overlay.commit_actions() == []
    assert bridge.graph.mutation_view()["node_master_names"][0] == "BUF1"
    restored_obs = session.evaluate()
    assert restored_obs["wns_ps"] == pytest.approx(baseline_obs["wns_ps"], abs=1e-6)

