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
        node_areas=torch.ones(db.num_nodes, dtype=torch.float64),
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
    monkeypatch.setattr(repair, "post_step_overflow", lambda *args: 0.35)
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
    metric.overflow = torch.tensor(0.35)
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
    assert session.record["request"]["crossing"]["overflow"] == pytest.approx(0.35)
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


def test_size_only_repair_syncs_runtime_inplace_without_restart(setup, monkeypatch):
    engine, session, old, model, metric, sync = setup
    db = engine.placedb
    db.scale_factor = 1.0
    db.cell_padding_x = 0.0
    db.area = 100.0
    db.total_fixed_node_area = 0.0
    db.total_space_area = 100.0
    db.total_movable_node_area = 2.0
    db.total_movable_macro_area = 0.0
    db.total_movable_cell_area = 2.0
    db.total_filler_node_area = 0.0
    db.movable_slice = slice(0, 2)
    db.movable_macro_mask = np.zeros(2, dtype=bool)
    db.inst_main_id = np.array([0, 0], dtype=np.int32)
    db.main_id_2_cell_id_start = np.array([0, 2], dtype=np.int32)
    db.inst_libcell_offset = np.array([0, 0], dtype=np.int32)
    db.inst_cell_id = np.array([0, 0], dtype=np.int32)
    db.flat_libcell_names = ["BUF1", "BUF2"]
    db.flat_libcell_info = np.array([[0, 0, 1], [0, 0, 2]], dtype=np.int32)
    db.flat_libcell_width = np.array([1.0, 2.0], dtype=np.float64)
    db.flat_libcell_height = np.array([1.0, 1.0], dtype=np.float64)
    db.flat_libcell_leakage = np.array([0.1, 0.2], dtype=np.float64)
    db.cell_id_2_libpin_id_start = np.array([0, 2, 4], dtype=np.int32)
    db.pin_2_libpin_offset = np.array([0, 1, 0, 1], dtype=np.int32)
    db.flat_lib_pin_offset_x = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64)
    db.flat_lib_pin_offset_y = np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64)
    db.flat_lib_pin_cap = np.array([0.01, 0.0, 0.02, 0.0], dtype=np.float64)
    db.flat_lib_pin_rcap = np.array([0.01, 0.0, 0.02, 0.0], dtype=np.float64)
    db.flat_lib_pin_fcap = np.array([0.01, 0.0, 0.02, 0.0], dtype=np.float64)
    db.flat_lib_pin_cap_limit = np.array([0.0, 1.0, 0.0, 1.0], dtype=np.float64)
    db.flat_lib_pin_slew_limit = np.array([10.0] * 4, dtype=np.float64)
    db.cell_id_2_arc_id_start = np.array([0, 1, 2], dtype=np.int32)
    db.flat_libarc_info = np.array([[0, 1, 0, 0, 1, 0], [0, 1, 1, 0, 1, 0]], dtype=np.int32)
    db.flat_inst_arcs_by_level = np.array([[0, 1, 0, 0, 1, 0, 0, 0]], dtype=np.int32)
    db.pin_offset_x = np.array([0.1, 0.2, 0.1, 0.2], dtype=np.float64)
    db.pin_offset_y = np.array([0.5, 0.6, 0.5, 0.6], dtype=np.float64)
    db.rawdb.pydb.inst_libcell_offset = [0, 0]

    for key in (
        "inst_main_id",
        "main_id_2_cell_id_start",
        "inst_libcell_offset",
        "inst_cell_id",
        "flat_libcell_info",
        "flat_libcell_width",
        "flat_libcell_height",
        "flat_libcell_leakage",
        "cell_id_2_libpin_id_start",
        "pin_2_libpin_offset",
        "flat_lib_pin_offset_x",
        "flat_lib_pin_offset_y",
        "flat_lib_pin_cap",
        "flat_lib_pin_rcap",
        "flat_lib_pin_fcap",
        "flat_lib_pin_cap_limit",
        "flat_lib_pin_slew_limit",
        "cell_id_2_arc_id_start",
        "flat_libarc_info",
        "flat_inst_arcs_by_level",
        "pin_offset_x",
        "pin_offset_y",
    ):
        setattr(old.data_collections, key, torch.as_tensor(getattr(db, key)).clone())
    old.data_collections.inst_size_cur = torch.tensor([1, 1], dtype=torch.int32)
    old.data_collections.inst_vt_cur = torch.tensor([0, 0], dtype=torch.int32)
    old.data_collections.inst_leakage_cur = torch.tensor([0.1, 0.1], dtype=torch.float64)
    old.data_collections.target_density = torch.tensor([0.5], dtype=torch.float64)
    old.data_collections.sorted_node_map = torch.arange(2, dtype=torch.int32)

    def stage(placedb_arg, **kwargs):
        assert kwargs.get("skip_size_only_rebuild") is True
        placedb_arg.rawdb.pydb.inst_libcell_offset = [1, 0]
        return {
            "status": "ok",
            "stages": {
                "repair_design": {
                    "status": "ok",
                    "inserted_count": 0,
                    "resized_count": 1,
                    "plan": {
                        "actions": [],
                        "resizes": [{"node_id": 0, "target_cell_index": 1}],
                    },
                }
            },
        }

    monkeypatch.setattr(repair, "run_repair_buffering_stage", Mock(side_effect=stage))
    # Must NOT raise MidplaceRestart when size-only syncs in-place!
    session.after_step(old, model, metric, {}, PlacementCursor(), 2.0)

    assert not session.restarted
    assert session.active is True
    assert session.record["mutation_kind"] == "resize-only"
    assert session.record["runtime_reused"] is True
    assert session.record["runtime_sizing_sync"]["runtime_cell_state_synced"] is True
    assert int(db.inst_libcell_offset[0]) == 1
    assert float(db.node_size_x[0]) == pytest.approx(2.0)
    assert float(db.total_movable_node_area) == pytest.approx(3.0)
    assert float(old.data_collections.node_size_x[0]) == pytest.approx(2.0)
    assert float(old.data_collections.original_node_size_x[0]) == pytest.approx(2.0)
    assert float(old.data_collections.pin_offset_x[0]) == pytest.approx(0.3)
    assert int(old.data_collections.flat_inst_arcs_by_level[0, 2]) == 1


def test_macroplacedb_incremental_rebuild_reuses_static_arrays_with_exact_parity(tmp_path):
    from copy import deepcopy

    from dreamplace.macroPlaceDB import MacroPlaceDB

    def make_pydb(num_movable=2):
        total_nodes = num_movable + 1
        num_pins = total_nodes * 2
        node_names = [f"u{i}" for i in range(num_movable)] + ["OUT"]
        pin_names = [f"p{i}" for i in range(num_pins)]
        pydb = SimpleNamespace(
            timing_schema_version=2,
            dbu=1000,
            num_nodes=total_nodes,
            num_terminals=0,
            num_terminal_NIs=1,
            num_place_blockages=0,
            num_movable_pins=num_movable * 2,
            node_name2id_map={name: idx for idx, name in enumerate(node_names)},
            node_names=node_names,
            node_x=[100.0 * i for i in range(total_nodes)],
            node_y=[200.0 * i for i in range(total_nodes)],
            node_orient=["N"] * total_nodes,
            node_size_x=[100.0] * num_movable + [0.0],
            node_size_y=[200.0] * num_movable + [0.0],
            node_is_hard_macro=[False] * total_nodes,
            macro_writeback_candidate=[False] * total_nodes,
            node2orig_node_map=list(range(total_nodes)),
            pin_direct=["I", "O"] * total_nodes,
            pin_offset_x=[10.0] * num_pins,
            pin_offset_y=[20.0] * num_pins,
            pin_names=pin_names,
            net_name2id_map={f"n{i}": i for i in range(total_nodes)},
            pin_name2id_map={name: idx for idx, name in enumerate(pin_names)},
            net_names=[f"n{i}" for i in range(total_nodes)],
            clock_net_names=[],
            net2pin_map=[np.array([2 * i, 2 * i + 1], dtype=np.int32) for i in range(total_nodes)],
            flat_net2pin_map=list(range(num_pins)),
            flat_net2pin_start_map=[2 * i for i in range(total_nodes + 1)],
            net_weights=[1.0] * total_nodes,
            node2pin_map=[np.array([2 * i, 2 * i + 1], dtype=np.int32) for i in range(total_nodes)],
            flat_node2pin_map=list(range(num_pins)),
            flat_node2pin_start_map=[2 * i for i in range(total_nodes + 1)],
            pin2node_map=[i // 2 for i in range(num_pins)],
            pin2net_map=[i // 2 for i in range(num_pins)],
            rows=[[0.0, 0.0, 1000.0, 200.0]],
            regions=[],
            flat_region_boxes=[],
            flat_region_boxes_start=[0],
            node2fence_region_map=[np.iinfo(np.int32).max] * total_nodes,
            xl=0.0,
            yl=0.0,
            xh=1000.0,
            yh=1000.0,
            row_height=200.0,
            site_width=10.0,
            total_fixed_node_area=0.0,
            total_space_area=1000000.0,
            routing_grid_xl=0.0,
            routing_grid_yl=0.0,
            routing_grid_xh=1000.0,
            routing_grid_yh=1000.0,
            num_routing_grids_x=4,
            num_routing_grids_y=4,
            unit_horizontal_capacities=[1.0, 2.0],
            unit_vertical_capacities=[1.5, 2.5],
            min_wire_widths=[10.0, 10.0],
            min_wire_spacings=[10.0, 10.0],
            initial_horizontal_demand_map=[0.5] * 32,
            initial_vertical_demand_map=[0.25] * 32,
            m2_pg_rail_density_boxes=[[0.0, 0.0, 100.0, 10.0]],
            m2_pg_rail_boxes=[[0.0, 0.0, 100.0, 10.0]],
            start_points=[0],
            end_points=[num_pins - 1],
            endpoints_max_valid=np.ones((1, 2), dtype=bool),
            endpoints_max_reason=np.zeros((1, 2), dtype=np.int32),
            endpoints_constraint_max_valid=np.empty((0, 2), dtype=bool),
            endpoints_constraint_max_reason=np.empty((0, 2), dtype=np.int32),
            endpoints_timing_check_max_valid=np.empty((0, 2), dtype=bool),
            endpoints_timing_check_max_reason=np.empty((0, 2), dtype=np.int32),
            clock_pins=[],
            FF_ids=[],
            clk_pin_r_aat=[],
            clk_pin_f_aat=[],
            clk_pin_rtran=[],
            clk_pin_ftran=[],
            clk_pin_names=[],
            flat_inst_arcs_by_level=[[0, 1, 0, 0, 1, 0, 0, 0]],
            flat_inst_arcs_by_level_start=[0, 1],
            flat_pin_to_graph=[],
            flat_pin_to_graph_start=[0] * (num_pins + 1),
            flat_pin_to_graph_reverse=[],
            flat_pin_to_graph_start_reverse=[0] * (num_pins + 1),
            pin_pair_arc_keys=[],
            flat_pin_pair_arc_start=[0],
            flat_pin_pair_arc_indices=[],
            arc_level_start=[0, 1],
            arc_src_pin=[0],
            arc_dst_pin=[1],
            arc_inst_id=[0],
            arc_libcell_id=[0],
            arc_libarc_id=[0],
            arc_sense=[1],
            arc_type=[0],
            arc_offset=[0],
            pin_pred_start=[0] * (num_pins + 1),
            pin_pred_pin=[],
            pin_pred_arc_id=[],
            pin_succ_start=[0] * (num_pins + 1),
            pin_succ_pin=[],
            pin_succ_arc_id=[],
            endpoint_pin_ids=[num_pins - 1],
            start_pin_ids=[0],
            pin_to_inst_id=[i // 2 for i in range(num_pins)],
            pin_to_node_id=[i // 2 for i in range(num_pins)],
            inst_topo_start=[0, num_movable],
            inst_topo_ids=list(range(num_movable)),
            flat_cells_by_level=list(range(num_movable)),
            flat_cells_by_reverse_level=list(range(num_movable)),
            flat_cells_by_level_start=[0, num_movable],
            flat_cells_by_reverse_level_start=[0, num_movable],
            net2driver_pin_map=[2 * i for i in range(total_nodes)],
            inrdelays=[0.0],
            infdelays=[0.0],
            inrtrans=[10.0],
            inftrans=[10.0],
            outcaps=[0.0],
            endpoints_rRAT=[100.0],
            endpoints_fRAT=[100.0],
            net_flat_arcs_start=list(range(total_nodes + 1)),
            net_flat_arcs=[[2 * i, 2 * i + 1] for i in range(total_nodes)],
            inst_flat_arcs_start=[0] + [1] * total_nodes,
            inst_flat_arcs=[0],
            endpoints_constraint_arcs=[],
            endpoints_timing_check_arcs=[],
            main_id_2_cell_id_start=[0, 2],
            cell_id_2_arc_id_start=[0, 1, 2],
            buffer_main_type_index=0,
            buffer_main_type_status="ready",
            buffer_main_type_candidate_indices=[0],
            inst_main_id=[0] * num_movable + [-1],
            inst_libcell_offset=[0] * total_nodes,
            flat_libarc_info=[["A", "Y", 0, 0, 1, 0], ["A", "Y", 1, 0, 1, 0]],
            flat_libcell_info=[["BUF1", 0, 1, 0], ["BUF2", 0, 2, 0]],
            flat_libcell_width=[100.0, 200.0],
            flat_libcell_height=[200.0, 200.0],
            flat_libcell_leakage=[1.0, 2.0],
            flat_libcell_main_id2size_vt_limit=[["BUF", 2, 1]],
            main_id_is_sizeable=[True],
            cell_id_2_libpin_id_start=[0, 2, 4],
            flat_lib_pin_names=["A", "Y", "A", "Y"],
            pin_2_libpin_offset=[0, 1] * num_movable + [-1, -1],
            flat_lib_pin_offset_x=[10.0, 20.0, 30.0, 40.0],
            flat_lib_pin_offset_y=[15.0, 25.0, 35.0, 45.0],
            flat_lib_pin_cap=[0.01, 0.0, 0.02, 0.0],
            flat_lib_pin_rcap=[0.01, 0.0, 0.02, 0.0],
            flat_lib_pin_fcap=[0.01, 0.0, 0.02, 0.0],
            flat_lib_pin_cap_limit=[0.0, 1.0, 0.0, 1.0],
            flat_lib_pin_slew_limit=[100.0] * 4,
            r_unit=0.001,
            c_unit=0.001,
        )
        for prefix in ("r_delay", "f_delay", "r_trans", "f_trans", "r_check", "f_check"):
            setattr(pydb, f"{prefix}_flat_luts_values", [[1.0, 2.0, 3.0, 4.0]] * 2)
            setattr(pydb, f"{prefix}_flat_luts_trans_table", [[0.0, 10.0]] * 2)
            setattr(pydb, f"{prefix}_flat_luts_cap_table", [[0.0, 1.0]] * 2)
            setattr(pydb, f"{prefix}_flat_luts_dim", [[2, 2]] * 2)
        return pydb

    def make_params():
        p = _load_params_json(None)
        p.fromJson(
            {
                "with_sta": 1,
                "place_io_engine": "ecc",
                "result_dir": str(tmp_path),
                "base_design_name": "synthetic",
                "target_density": 0.8,
                "num_bins_x": 16,
                "num_bins_y": 16,
                "random_center_init_flag": 0,
            }
        )
        return p

    pydb0 = make_pydb(2)
    params_inc = make_params()
    db_inc = MacroPlaceDB(SimpleNamespace())
    db_inc.pydb = pydb0
    db_inc.rawdb = SimpleNamespace(pydb=pydb0, export_options=None)
    db_inc.init_db(params_inc)

    rows_obj = db_inc.rows
    luts_obj = db_inc.f_delay_flat_luts_values
    hcap_obj = db_inc.unit_horizontal_capacities
    hdem_obj = db_inc.initial_horizontal_demand_map

    pydb1 = make_pydb(3)
    summary = db_inc.rebuild_from_pydb(deepcopy(pydb1), rebuild_mode="topo")
    assert summary["incremental_static_reuse"] is True
    assert db_inc.rows is rows_obj
    assert db_inc.f_delay_flat_luts_values is luts_obj
    assert db_inc.unit_horizontal_capacities is hcap_obj
    assert db_inc.initial_horizontal_demand_map is hdem_obj

    params_full = make_params()
    db_full = MacroPlaceDB(SimpleNamespace())
    db_full.pydb = deepcopy(pydb1)
    db_full.rawdb = SimpleNamespace(pydb=db_full.pydb, export_options=None)
    db_full.init_db(params_full)

    for attr in (
        "node_x",
        "node_y",
        "node_size_x",
        "node_size_y",
        "pin_offset_x",
        "pin_offset_y",
        "rows",
        "unit_horizontal_capacities",
        "initial_horizontal_demand_map",
        "f_delay_flat_luts_values",
        "flat_lib_pin_offset_x",
        "inst_cell_id",
        "pin_2_libpin_offset",
    ):
        np.testing.assert_allclose(getattr(db_inc, attr), getattr(db_full, attr))


def test_first_inflation_window_replaced_by_repair_design_leaves_four_s5b1_windows(
    setup, monkeypatch
):
    engine, session, old, model, metric, _ = setup
    engine.params.routability_opt_flag = 1
    engine.params.node_area_adjust_overflow = 0.15
    engine.params.timing_grad_balance_reset_on_restart = 0
    schedule = TimingOptWindowSchedule([], 5)
    old.inflation_s5b1 = SimpleNamespace(
        schedule=schedule,
        round_index=0,
        summaries=[],
        sync_native_sizing=lambda: {"ok": True, "accepted_count": 0, "requested_count": 0},
    )
    old._diff_tdp_step_status = {"timing_objective_active": True, "enabled": True}
    old._timing_grad_balance_state = {"initialized": True, "weight_applied": 0.25}
    model.inflation_state = SimpleNamespace(num_area_adjust=0)
    model.routability_Lsub_iteration = 5
    model._timing_coeff_growth_base = (500.0, 5.0)
    model._timing_coeff_growth_multiplier = 1.5
    model.timing_wns_coeff = 750.0
    model.timing_tns_coeff = 7.5

    def stage(db, **kwargs):
        updated = database(3)
        db.__dict__.update(updated.__dict__)
        db.topology_generation = 1
        return {
            "status": "ok",
            "stages": {
                "repair_design": {
                    "status": "ok",
                    "inserted_count": 1,
                    "resized_count": 1,
                    "plan": {"actions": [{}], "resizes": [{}]},
                }
            },
        }

    run = Mock(side_effect=stage)
    monkeypatch.setattr(repair, "run_repair_buffering_stage", run)

    # Ordinary after_step before first inflation must defer when S5B1 inflation mode is active
    assert session.defers_to_first_inflation_window(old)
    session.after_step(old, model, metric, {}, PlacementCursor(raw_iteration=10), 1.0)
    assert run.call_count == 0
    assert not session.active

    # Triggering at the 1st inflation window consumes Window 1 and Round 0
    metric.overflow = torch.tensor(0.145)
    with pytest.raises(repair.MidplaceRestart):
        session.before_first_inflation(
            old,
            model,
            metric,
            {},
            PlacementCursor(stage=0, gamma=10, density=0, sub=0, raw_iteration=10),
        )
    assert run.call_count == 1
    assert session.resume_sub_iteration == 5
    assert session.schedule.use_restart_density_weight is True

    fresh = session.build_placer(lambda: placer_for(engine.placedb))
    fresh_owner = SimpleNamespace(lane=None)
    session.runtime_ready(fresh, fresh_owner, fresh._routability_controller)
    assert fresh_owner.window_count == 1
    assert fresh_owner.schedule.window_count == 1
    assert fresh_owner.summaries[0]["window"] == 1
    assert fresh_owner.summaries[0]["mode"] == "repair_design"
    assert session.schedule.inflation_rounds == {0: 1}
    assert fresh._routability_controller._last_inflation_step == 3

    # Remaining 4 inflation events correspond to S5B1 windows 2, 3, 4, 5
    remaining_windows = []
    for next_iter in (50, 90, 130, 170, 210):
        evt = fresh_owner.schedule.inflation_event(iteration=next_iter, stage=0)
        if evt is not None:
            remaining_windows.append(fresh_owner.schedule.window_count)
    assert remaining_windows == [2, 3, 4, 5]

    fresh_model = SimpleNamespace(
        apply_timing_grad_balance_state=Mock(),
        enable_l_shape_routability=False,
    )
    session.schedule.restore_model_state(fresh, fresh_model, engine.params)
    assert fresh._diff_tdp_step_status["timing_objective_active"] is True
    assert fresh_model.timing_wns_coeff == pytest.approx(750.0)
    assert fresh_model.timing_tns_coeff == pytest.approx(7.5)
    fresh_model.apply_timing_grad_balance_state.assert_called_once_with(
        {"initialized": True, "weight_applied": 0.25}
    )


def test_s5b1_window_runs_repair_timing_and_preserves_inflated_geometry(
    setup, monkeypatch
):
    engine, session, old, model, metric, _ = setup
    engine.params.routability_opt_flag = 1
    engine.params.timing_opt_buffering_enabled = 1
    engine.params.node_area_adjust_overflow = 0.15
    schedule = TimingOptWindowSchedule([], 5)
    schedule.inflation_event(iteration=10, stage=0)
    schedule.inflation_event(iteration=50, stage=0)
    old.inflation_s5b1 = SimpleNamespace(
        schedule=schedule,
        round_index=5,
        summaries=[
            {
                "window": 2,
                "sizing": {"best_round": 2},
                "buffering": {
                    "status": "deferred_to_midplace_repair_timing",
                    "scheduled": 0,
                    "accepted": 0,
                    "cumulative_count": 0,
                },
            }
        ],
        capacity=1000.0,
        sync_native_sizing=lambda: {"ok": True, "accepted_count": 0, "requested_count": 0},
        _write_summaries=Mock(),
    )
    old.placedb = engine.placedb
    old.data_collections.original_node_size_x = old.data_collections.node_size_x.clone()
    old.data_collections.original_node_size_y = old.data_collections.node_size_y.clone()
    old.data_collections.original_pin_offset_x = torch.zeros(
        len(engine.placedb.pin2node_map), dtype=torch.float32
    )
    old.data_collections.original_pin_offset_y = torch.zeros(
        len(engine.placedb.pin2node_map), dtype=torch.float32
    )
    old.data_collections.pin_offset_x = old.data_collections.original_pin_offset_x.clone()
    old.data_collections.pin_offset_y = old.data_collections.original_pin_offset_y.clone()
    old.data_collections.node_size_x[0] = 6.0
    old.data_collections.node_size_y[0] = 5.0
    old.pos[0].data[0] -= 1.0
    old.pos[0].data[engine.placedb.num_nodes] -= 1.0
    engine.placedb.cooptimization_geometry = old.data_collections
    model.inflation_state = SimpleNamespace(num_area_adjust=2)
    model.routability_Lsub_iteration = 5

    observed_modes = []

    def stage_fn(db, **kwargs):
        observed_modes.append(kwargs["mode"])
        updated = database(3)
        db.__dict__.update(updated.__dict__)
        db.topology_generation = 2
        return {
            "status": "ok",
            "stages": {
                "repair_timing": {
                    "status": "ok",
                    "accepted_count": 1,
                    "verified_kept_count": 1,
                    "timing_before": {"wns_ps": -200.0, "tns_ps": -500.0},
                    "timing_after": {"wns_ps": -100.0, "tns_ps": -250.0},
                    "plan": {"accepted_actions": [{"buffer_master_name": "BUF"}]},
                }
            },
        }

    monkeypatch.setattr(repair, "run_repair_buffering_stage", stage_fn)
    metric.overflow = torch.tensor(0.14)
    with pytest.raises(repair.MidplaceRestart):
        session.after_inflation_window(
            old,
            model,
            metric,
            {},
            PlacementCursor(stage=0, gamma=20, density=0, sub=0, raw_iteration=50),
            window_index=2,
        )

    assert observed_modes == ["repair_timing"]
    assert session.record["window_repairs"][0]["window"] == 2
    assert session.record["window_repairs"][0]["mode"] == "repair_timing"
    assert session.record["window_repairs"][0]["inserted_count"] == 1
    assert old.inflation_s5b1.summaries[-1]["buffering"]["accepted"] == 1
    assert old.inflation_s5b1.summaries[-1]["buffering"]["cumulative_count"] == 1

    def build_fresh():
        fresh_placer = placer_for(engine.placedb)
        fresh_placer.placedb = engine.placedb
        fresh_placer.data_collections.original_node_size_x = (
            fresh_placer.data_collections.node_size_x.clone()
        )
        fresh_placer.data_collections.original_node_size_y = (
            fresh_placer.data_collections.node_size_y.clone()
        )
        fresh_placer.data_collections.original_pin_offset_x = torch.zeros(
            len(engine.placedb.pin2node_map), dtype=torch.float32
        )
        fresh_placer.data_collections.original_pin_offset_y = torch.zeros(
            len(engine.placedb.pin2node_map), dtype=torch.float32
        )
        fresh_placer.data_collections.pin_offset_x = (
            fresh_placer.data_collections.original_pin_offset_x.clone()
        )
        fresh_placer.data_collections.pin_offset_y = (
            fresh_placer.data_collections.original_pin_offset_y.clone()
        )
        fresh_placer.data_collections.target_density = torch.tensor([0.8], dtype=torch.float32)
        fresh_placer.data_collections.sorted_node_map = torch.zeros(
            engine.placedb.num_movable_nodes, dtype=torch.int32
        )
        return fresh_placer

    fresh = session.build_placer(build_fresh)
    fresh_owner = SimpleNamespace(lane=None, capacity=1000.0)
    session.runtime_ready(fresh, fresh_owner, fresh._routability_controller)
    assert session.record["inflated_geometry_restore"] == {"restored_inflated_nodes": 1}
    assert float(fresh.data_collections.node_size_x[0]) == pytest.approx(6.0)
    assert float(fresh.data_collections.node_size_y[0]) == pytest.approx(5.0)
    assert float(fresh.data_collections.original_node_size_x[0]) == pytest.approx(1.0)
    assert float(fresh.data_collections.original_node_size_y[0]) == pytest.approx(1.0)




