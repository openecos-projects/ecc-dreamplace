"""Synthetic native-transaction boundary, fresh views and bounded continuation."""

import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from dreamplace.flows import midplace_repair as repair
from dreamplace.flows.midplace_observation import (
    committed_seed,
    runtime_topology,
    validate_position_seed,
)
from dreamplace.flows.midplace_schedule import MidplaceSchedule
from dreamplace.flows.midplace_state import PlacementCursor
from dreamplace.flows.timing_opt_schedule import TimingOptWindowSchedule
from dreamplace.Placer import PlacementEngine
from dreamplace.placer_cli import _load_params_json


def database(count):
    node_names = ["a", "b", "new_buffer"][:count]
    pins = np.arange(count * 2, dtype=np.int32)
    starts = np.arange(count + 1, dtype=np.int32) * 2
    pydb = SimpleNamespace(
        num_nodes=count,
        native_mutation_epoch=0,
        native_state_dirty=False,
        node_names=node_names,
        pin_names=[f"p{i}" for i in pins],
        net_names=[f"n{i}" for i in range(count)],
        flat_inst_arcs_by_level=[()] * count,
        node_is_buffer=[False, False, True][:count],
        node2orig_node_map=np.arange(count),
        node2fence_region_map=np.full(count, np.iinfo(np.int32).max),
    )
    return SimpleNamespace(
        rawdb=SimpleNamespace(pydb=pydb, refresh_generation=0),
        node_names=node_names,
        num_nodes=count,
        num_physical_nodes=count,
        num_movable_nodes=count,
        num_filler_nodes=0,
        dtype=np.float64,
        node_x=np.arange(count, dtype=float),
        node_y=np.arange(count, dtype=float) + 10,
        node_size_x=np.ones(count),
        node_size_y=np.ones(count),
        node2orig_node_map=np.arange(count),
        node2fence_region_map=np.full(count, np.iinfo(np.int32).max),
        regions=[],
        xl=0.0,
        yl=0.0,
        xh=100.0,
        yh=100.0,
        topology_generation=0,
        pin2node_map=np.repeat(np.arange(count, dtype=np.int32), 2),
        pin2net_map=np.repeat(np.arange(count, dtype=np.int32), 2),
        flat_node2pin_map=pins,
        flat_node2pin_start_map=starts,
        flat_net2pin_map=pins,
        flat_net2pin_start_map=starts,
        pending_continuation_seed=None,
    )


def placer_for(db):
    data = SimpleNamespace(
        **{
            key: torch.as_tensor(getattr(db, key))
            for key in (
                "pin2node_map",
                "pin2net_map",
                "node2fence_region_map",
                "flat_node2pin_map",
                "flat_node2pin_start_map",
                "flat_net2pin_map",
                "flat_net2pin_start_map",
                "node_size_x",
                "node_size_y",
            )
        },
        node_areas=torch.ones(db.num_nodes),
    )
    db.pending_continuation_seed = None
    return SimpleNamespace(
        data_collections=data,
        pos=[torch.tensor(np.concatenate((db.node_x, db.node_y)))],
        op_collections=SimpleNamespace(timing_propagation_op=object()),
        _routability_controller=SimpleNamespace(_gp_steps=3, _last_inflation_step=1),
    )


@pytest.fixture
def setup(tmp_path, monkeypatch):
    params = _load_params_json(None)
    params.fromJson(
        {
            "with_sta": 1,
            "place_io_engine": "ecc",
            "result_dir": str(tmp_path),
            "base_design_name": "synthetic",
            "repair_buffering_midplace_enable": True,
            "repair_buffering_midplace_bounded_steps": 2,
        }
    )
    engine = SimpleNamespace(params=params, placedb=database(2))
    session = repair.create_session(engine)
    old = placer_for(engine.placedb)
    model = SimpleNamespace(
        Lgamma_iteration=100,
        Llambda_density_weight_iteration=1,
        Lsub_iteration=1,
        use_timing_obj=True,
        use_l_shape_routability=True,
        last_timing_objective_terms={"total_loss": 2.0, "grad_balance": {}},
        op_collections=SimpleNamespace(
            timing_propagation_op=SimpleNamespace(num_pins=4, pin_net=torch.tensor([0, 0, 1, 1]))
        ),
        l_shape_routability_op=SimpleNamespace(placedb=engine.placedb),
    )
    metric = SimpleNamespace(
        iteration=0,
        objective=torch.tensor(1.0),
        timing_objective=2.0,
        l_shape_cost=3.0,
        overflow=torch.tensor(0.7),
    )
    sync = Mock()
    monkeypatch.setattr(repair, "write_back_movable_lpos", sync)
    monkeypatch.setattr(repair, "post_step_overflow", lambda *args: 0.5)
    return engine, session, old, model, metric, sync


@pytest.mark.parametrize(
    "inserted,resized,kind",
    [(0, 0, "noop"), (0, 1, "resize-only"), (1, 0, "insert"), (1, 1, "resize+insert")],
)
def test_once_commit_fresh_runtime_and_bounded_steps(setup, monkeypatch, inserted, resized, kind):
    engine, session, old, model, metric, sync = setup

    def stage(db, **kwargs):
        if inserted or resized:
            updated = database(2 + inserted)
            db.__dict__.update(updated.__dict__)
            db.topology_generation = 1
        return {
            "status": "ok",
            "stages": {
                "repair_design": {
                    "status": "ok" if inserted or resized else "noop",
                    "inserted_count": inserted,
                    "resized_count": resized,
                    "plan": {"actions": [{}] * inserted, "resizes": [{}] * resized},
                }
            },
        }

    run = Mock(side_effect=stage)
    monkeypatch.setattr(repair, "run_repair_buffering_stage", run)
    if inserted or resized:
        with pytest.raises(repair.MidplaceRestart):
            session.after_step(old, model, metric, {}, PlacementCursor(), 2.0)
        seed = engine.placedb.pending_continuation_seed
        assert seed["movable_node_names"] == engine.placedb.node_names
        fresh = session.build_placer(lambda: placer_for(engine.placedb))
        session.runtime_ready(fresh, None, fresh._routability_controller)
        assert fresh is not old
        assert session.record["runtime_views"]["csr_matches_committed_placedb"]
        assert session.record["position_seed_consumed"]
    else:
        session.after_step(old, model, metric, {}, PlacementCursor(), 2.0)
        fresh = old
        assert not session.restarted
        assert engine.placedb.pending_continuation_seed is None
    assert session.record["mutation_kind"] == kind
    model.op_collections.timing_propagation_op = SimpleNamespace(
        num_pins=len(engine.placedb.pin2net_map),
        pin_net=torch.as_tensor(engine.placedb.pin2net_map),
    )
    sync.assert_called_once()
    run.assert_called_once()
    fresh.pos[0][0] += 1
    if inserted:
        fresh.pos[0][2] += 0.1
    session.after_step(fresh, model, metric, {}, PlacementCursor(gamma=1), 3.0)
    with pytest.raises(repair.MidplaceBoundedStop):
        session.after_step(fresh, model, metric, {}, PlacementCursor(gamma=2), 4.0)
    assert {
        key: session.record[key]
        for key in ("status", "gate_tier", "gate_status", "continuation_steps_completed")
    } == {
        "status": "bounded_completed",
        "gate_tier": "B",
        "gate_status": "passed",
        "continuation_steps_completed": 2,
    }
    assert run.call_count == 1


@pytest.mark.parametrize("phase", ["timing_carrier_rebuild_ms", "placedb_rebuild_ms"])
def test_committed_refresh_failure_cannot_restart_old_runtime(setup, monkeypatch, phase):
    _, session, old, model, metric, _ = setup

    def failure(db, *, profile, **kwargs):
        profile.commits["sizing"] = {"ok": True, "accepted_count": 1}
        with profile.measure(phase):
            raise RuntimeError("failed after native commit")

    run = Mock(side_effect=failure)
    monkeypatch.setattr(repair, "run_repair_buffering_stage", run)
    with pytest.raises(RuntimeError, match="after native commit") as exc:
        session.after_step(old, model, metric, {}, PlacementCursor(), 1.0)
    session.fail(exc.value)
    assert session.record["failure_stage"] == phase
    assert session.record["gate_status"] == "failed"
    assert session.tracker.consumed
    assert not session.restarted
    assert run.call_count == 1


def test_sizing_then_buffer_rejection_is_partial_commit(setup, monkeypatch):
    _, session, old, model, metric, _ = setup

    def reject(db, *, profile, **kwargs):
        db.rawdb.pydb.native_mutation_epoch = 1
        db.rawdb.pydb.native_state_dirty = True
        profile.commits.update(sizing={"accepted_count": 1}, buffer={"status": "rejected"})
        return {"status": "failed", "stages": {"repair_design": {"status": "commit_failed"}}}

    monkeypatch.setattr(repair, "run_repair_buffering_stage", reject)
    with pytest.raises(RuntimeError, match="repair failed") as exc:
        session.after_step(old, model, metric, {}, PlacementCursor(), 1.0)
    session.fail(exc.value)
    assert session.record["status"] == "partial_commit"
    assert session.record["failure_native_state"] == {
        "pydb_identity": id(session.engine.placedb.rawdb.pydb),
        "refresh_generation": 0,
        "native_mutation_epoch": 1,
        "native_state_dirty": True,
    }
    assert not session.restarted


def test_seed_uses_committed_names_and_coordinates_not_old_indices():
    db = database(3)
    db.node_names = ["b", "new_buffer", "a"]
    old = {
        "movable_node_names": ["a", "b"],
        "movable_x": [100.0, 200.0],
        "filler_x": [7.0],
        "filler_y": [8.0],
        "fixed_positions": {},
    }
    assert committed_seed(db, old) == {
        "movable_node_names": ["b", "new_buffer", "a"],
        "movable_x": [0.0, 1.0, 2.0],
        "movable_y": [10.0, 11.0, 12.0],
        "filler_x": [7.0],
        "filler_y": [8.0],
        "fixed_positions": {},
    }
    old["overflow_reference_area"] = 123.0
    assert committed_seed(db, old)["overflow_reference_area"] == 123.0


def test_reinitialized_position_seed_is_rejected_before_any_optimizer_step():
    db = database(3)
    seed = committed_seed(db, {"filler_x": [], "filler_y": []})
    fresh = placer_for(db)
    assert validate_position_seed(fresh, db, seed) == {
        "movable_positions_checked": 3,
        "fixed_positions_checked": 0,
        "max_seed_displacement": 0.0,
    }
    fresh.pos[0][1] += 1
    with pytest.raises(RuntimeError, match="preserve the committed position seed"):
        validate_position_seed(fresh, db, seed)


def test_stale_csr_and_fixed_insertions_are_rejected():
    db = database(3)
    fresh = placer_for(db)
    fresh.data_collections.pin2node_map = torch.zeros(6)
    with pytest.raises(RuntimeError, match="stale pin2node_map"):
        runtime_topology(fresh, db, {"a", "b"})
    fresh = placer_for(db)
    db.num_movable_nodes = 2
    with pytest.raises(RuntimeError, match="not a movable buffer"):
        runtime_topology(fresh, db, {"a", "b"})


def test_schedule_preserves_consumed_windows_and_discards_virtual_cache():
    schedule = TimingOptWindowSchedule([0.8, 0.5, 0.2], 3)
    schedule.next_overflow_event(0.7, iteration=3, stage=0)
    old = SimpleNamespace(
        inflation_s5b1=SimpleNamespace(schedule=schedule, round_index=1, summaries=[{"window": 1}]),
        _routability_controller=SimpleNamespace(_gp_steps=5, _last_inflation_step=4),
    )
    saved = MidplaceSchedule()
    saved.capture(
        old,
        SimpleNamespace(inflation_state=SimpleNamespace(num_area_adjust=1), Lsub_iteration=2),
        0,
    )
    owner = SimpleNamespace(lane=None)
    controller = SimpleNamespace()
    summary = saved.restore(owner, controller)
    assert owner.schedule.pending_milestones == [0.5, 0.2]
    assert owner.schedule.window_count == 1
    assert owner.schedule is not schedule
    assert summary["inflation_rounds_completed"] == {0: 1}
    assert controller._last_inflation_step == 4
    owner.lane = object()
    with pytest.raises(RuntimeError, match="retained virtual topology"):
        saved.restore(owner, controller)


def test_outer_owner_unwinds_before_restart_and_bounded_stop_skips_terminal(setup, monkeypatch):
    fixture_engine, session, old, _, _, _ = setup
    engine = object.__new__(PlacementEngine)
    engine.params, engine.placedb = fixture_engine.params, fixture_engine.placedb
    engine.setup_placedb = Mock()
    engine.placer = old
    seen_sessions = []

    def place():
        seen_sessions.append(engine.midplace_repair)
        if len(seen_sessions) == 1:
            raise repair.MidplaceRestart()
        assert engine.placer is None
        session.record.update(status="bounded_completed", continuation_steps_completed=2)
        raise repair.MidplaceBoundedStop()

    engine.place = Mock(side_effect=place)
    monkeypatch.setattr("dreamplace.flows.midplace_state.create_session", lambda owner: session)
    finalize = Mock(
        side_effect=AssertionError("bounded lane must not enter terminal materialization")
    )
    monkeypatch.setattr("dreamplace.flows.committed_topology.finalize_inflation_s5b1", finalize)
    assert engine.run()["status"] == "bounded_completed"
    assert seen_sessions == [session, session]
    finalize.assert_not_called()


def test_poststep_overflow_uses_live_position_and_initial_area_normalization():
    from dreamplace.flows.midplace_observation import post_step_overflow

    db = SimpleNamespace(total_movable_node_area=10.0, overflow_reference_area=20.0)
    placer = SimpleNamespace(pos=[torch.tensor([1.0])], data_collections=SimpleNamespace())
    ops = {"overflow": lambda pos: (pos * 4.0, torch.tensor([2.0]))}
    assert post_step_overflow(placer, db, ops, 0) == pytest.approx(0.2)
    placer.pos[0].fill_(3.0)
    assert post_step_overflow(placer, db, ops, 0) == pytest.approx(0.6)


def test_native_queue_endpoint_sets_are_stable_in_handoff_artifacts(setup):
    _, session, _, _, _, _ = setup
    session.record["native_config"] = {"retired_endpoints": {7, 2}}
    session.write()
    assert json.loads(session.path.read_text())["native_config"] == {"retired_endpoints": [2, 7]}


def test_initial_metric_survives_an_s5b1_window_before_the_first_stable_step(setup, monkeypatch):
    _, session, old, model, metric, _ = setup
    metric.overflow = torch.tensor(0.5)
    session.observe_initial(metric, PlacementCursor())
    metric.overflow.fill_(0.8)
    monkeypatch.setattr(repair, "post_step_overflow", lambda *args: 0.9)
    stage = Mock(
        return_value={
            "status": "ok",
            "stages": {
                "repair_design": {
                    "status": "noop",
                    "inserted_count": 0,
                    "resized_count": 0,
                    "plan": {},
                }
            },
        }
    )
    monkeypatch.setattr(repair, "run_repair_buffering_stage", stage)
    assert not session.tracker.consumed
    session.after_step(old, model, metric, {}, PlacementCursor(), 1.0)
    assert session.record["request"]["crossing"]["overflow"] == pytest.approx(0.5)
    assert session.record["request"]["execution"]["overflow"] == pytest.approx(0.9)
    stage.assert_called_once()


@pytest.mark.parametrize("key", ["node2orig_node_map", "node2fence_region_map"])
@pytest.mark.parametrize("value", [-1, 999])
def test_inserted_buffer_native_identity_and_region_are_checked(key, value):
    db = database(3)
    db.__dict__[key][2] = value
    with pytest.raises(RuntimeError, match=key):
        runtime_topology(placer_for(db), db, {"a", "b"})


def test_committed_seed_rejects_fixed_position_change():
    db = database(3)
    db.num_movable_nodes = 2
    old = repair.capture_fillers(db, placer_for(db).pos[0])
    db.node_x[2] += 1
    with pytest.raises(RuntimeError, match="fixed position"):
        committed_seed(db, old)


@pytest.mark.parametrize("missing", ["timing", "routability"])
def test_enabled_objective_must_have_executed_after_handoff(setup, missing):
    engine, _, placer, model, metric, _ = setup
    if missing == "timing":
        model.last_timing_objective_terms = {}
    else:
        metric.l_shape_cost = None
    with pytest.raises(RuntimeError, match=missing):
        repair.step_record(placer, engine.placedb, model, metric, 0.5, 1, 2.0, [])


def test_first_step_rejects_stale_timing_consumer(setup):
    engine, _, placer, model, metric, _ = setup
    model.op_collections.timing_propagation_op.pin_net[0] = 1
    with pytest.raises(RuntimeError, match="timing.*pin"):
        repair.step_record(placer, engine.placedb, model, metric, 0.5, 1, 2.0, [])


@pytest.mark.parametrize("fallbacks", [0, 2])
def test_timing_fallback_status_comes_from_the_completed_native_campaign(fallbacks):
    stage = {
        "stages": {
            "repair_timing": {
                "status": "ok",
                "plan": {
                    "native_timing_graph_statistics": {"full_fallback_count": fallbacks},
                },
            }
        }
    }
    result = repair.timing_validation(stage)
    assert result["incremental_full_fallback"] is bool(fallbacks)
    assert result["native_full_fallbacks"] == fallbacks


def test_missing_native_statistics_do_not_claim_no_fallback():
    stage = {"stages": {"repair_timing": {"status": "ok", "plan": {}}}}
    assert repair.timing_validation(stage)["incremental_full_fallback"] is None


def test_full_continuation_keeps_sparse_checkpoints_and_the_last_step(setup):
    _, session, placer, model, metric, _ = setup
    session.config = replace(session.config, bounded_steps=0)
    session.active = True
    for iteration in range(103):
        session.after_step(placer, model, metric, {}, PlacementCursor(gamma=iteration), 1.0)
    session.placement_finished()
    assert [step["step"] for step in session.record["steps"]] == [1, 5, 20, 50, 100, 103]
    assert session.record["continuation_steps_completed"] == 103


def test_objectives_and_windows_wait_for_the_physical_repair(setup):
    from dreamplace.flows.inflation_s5b1 import InflationS5B1
    from dreamplace.flows.timing_objective_policy import diff_tdp_gate_status
    from dreamplace.ops.routability.l_shape_policy import LShapePolicy
    from dreamplace.ops.routability.routability_controller import RoutabilityController

    engine, _, _, _, _, _ = setup
    params = engine.params
    params.differentiable_timing_obj = params.diff_timing_driven_placement = 1
    params._midplace_objectives_pending = True
    model = SimpleNamespace(enable_l_shape_routability=False)
    policy = LShapePolicy(params)
    window = object.__new__(InflationS5B1)
    window.params = params
    controller = object.__new__(RoutabilityController)
    assert not diff_tdp_gate_status(params, 10, 0.01)["timing_objective_active"]
    policy.maybe_reenable(model, 0.01)
    assert not model.enable_l_shape_routability
    window.before_step(None, None, None, None, None, iteration=10, stage=0)
    assert window.before_inflation(None, None, iteration=10, stage=0) is None
    assert controller.inflation_triggers(params, 0, 3, 0.01) == (False, False)
    params._midplace_objectives_pending = False
    assert diff_tdp_gate_status(params, 10, 0.01)["timing_objective_active"]
    policy.maybe_reenable(model, 0.01)
    assert model.enable_l_shape_routability


def test_terminal_failure_keeps_the_handoff_failure_stage(setup):
    _, session, _, _, _, _ = setup
    with pytest.raises(RuntimeError, match="terminal failed"):
        session.finish_terminal(Mock(side_effect=RuntimeError("terminal failed")))
    assert session.record["terminal_status"] == "failed"
    assert session.record["failure_stage"] == "terminal_materialization_legalization_ms"


def test_disabling_midplace_clears_suspension_from_a_previous_failed_session(setup):
    engine, _, _, _, _, _ = setup
    engine.params._midplace_objectives_pending = True
    engine.params.repair_buffering_midplace_enable = False
    assert repair.create_session(engine) is None
    assert not engine.params._midplace_objectives_pending
