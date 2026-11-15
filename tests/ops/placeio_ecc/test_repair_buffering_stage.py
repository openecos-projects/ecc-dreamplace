from types import SimpleNamespace

import pytest
from dreamplace.flows.baselines import repair_buffering
from dreamplace.ops.placeio_ecc.place_io import PlaceIOFunction


class _RawDB:
    timing_enabled = True

    def __init__(self):
        self.pydb = "base"


class _Placedb:
    def __init__(self):
        self.rawdb = _RawDB()
        self.pydb = self.rawdb.pydb
        self.backend_caps = {"backend": "ecc", "supports_buffer_commit": True}
        self.rebuild_calls = []

    def rebuild_from_pydb(self, pydb, rebuild_mode):
        self.rebuild_calls.append((pydb, rebuild_mode))
        self.pydb = pydb
        return {"status": "ok", "post_counts": {"nodes": 2}}


def _install_common(monkeypatch):
    monkeypatch.setattr(
        "dreamplace.ops.rebuffer.timing_model.RebufferTimingModel",
        lambda pydb, **kwargs: SimpleNamespace(baseline=SimpleNamespace(as_dict=lambda: {})),
    )
    monkeypatch.setattr(
        PlaceIOFunction,
        "backend_caps",
        staticmethod(lambda raw_db: {"backend": "ecc", "supports_buffer_commit": True}),
    )


def _install_design_plan(monkeypatch):
    calls = []

    def plan(pydb, timing_model, *, config):
        calls.append((pydb, timing_model, config))
        return {
            "status": "ok",
            "baseline": {"wns_ps": -1.0},
            "inserted_count": 1,
            "resized_count": 1,
            "actions": [{"buffer_master_name": "BUF"}],
            "resizes": [{"node_id": 0, "target_cell_index": 1}],
        }

    def apply(raw_db, plan, *, refresh):
        calls.append(("apply", raw_db, plan, refresh))
        return {"status": "committed", "sizing": {}, "buffer": {}}

    def python_refresh(raw_db):
        raw_db.pydb = "refreshed"
        return raw_db.pydb, {"status": "ok", "ecc_sta_run": False}

    monkeypatch.setattr("dreamplace.ops.rebuffer.repair_design.plan_repair_design_from_pydb", plan)
    monkeypatch.setattr(PlaceIOFunction, "apply_repair_design", staticmethod(apply))
    monkeypatch.setattr(
        "dreamplace.ops.placeio_ecc.timing_carrier.refresh_python_timing_carrier",
        python_refresh,
    )
    return calls


_ACTION = {
    "action_kind": "buffer_insert",
    "net_name": "net0",
    "buffer_master_name": "BUF",
    "candidate_location_x_dbu": 1,
    "candidate_location_y_dbu": 1,
    "downstream_pin_names": ["u/a"],
}


class _Session:
    def __init__(self, raw_db, *, timing_model):
        self.raw_db = raw_db

    def begin(self):
        return {"status": "ok"}

    def rollback(self):
        return {"status": "rolled_back"}


def _install_timing_plan(monkeypatch, actions):
    planner_configs = []
    verified_calls = []

    def plan_timing(session, model, *, max_passes, config):
        planner_configs.append(config)
        return {"status": "ok", "accepted_actions": list(actions), "final": {"tns_ps": -1.0}}

    def verified(raw_db, actions, **kwargs):
        verified_calls.append((list(actions), dict(kwargs)))
        raw_db.pydb = "verified_refreshed"
        return {
            "status": "ok",
            "kept_count": len(actions),
            "reverted_count": 0,
            "timing_after": {"wns_ps": -1.0, "tns_ps": -1.0},
            "kept_actions": list(actions),
            "chunks": [],
        }

    monkeypatch.setattr("dreamplace.ops.placeio_ecc.buffer_session.PlaceIOBufferSession", _Session)
    monkeypatch.setattr("dreamplace.ops.placeio_ecc.repair_timing.plan_timing_repair", plan_timing)
    monkeypatch.setattr(
        "dreamplace.ops.placeio_ecc.commit_verify.commit_buffer_actions_verified", verified
    )
    return planner_configs, verified_calls


def test_repair_design_stage_commits_refreshes_and_rebuilds(tmp_path, monkeypatch):
    placedb = _Placedb()
    _install_common(monkeypatch)
    calls = _install_design_plan(monkeypatch)

    result = repair_buffering.run_repair_buffering_stage(
        placedb, mode="repair_design", artifact_dir=tmp_path
    )

    assert result["status"] == "ok"
    assert result["stages"]["repair_design"]["buffer_histogram"] == {"BUF": 1}
    assert result["stages"]["repair_design"]["timing_before"] == {"wns_ps": -1.0}
    assert placedb.rebuild_calls == [("refreshed", "all")]
    assert calls[0][0] == "base"
    assert calls[1][0] == "apply"
    assert calls[1][3] is False
    assert (tmp_path / "repair_buffering_stage.json").is_file()


def test_repair_design_uses_configured_wire_rc_before_and_after_commit(monkeypatch):
    _install_common(monkeypatch)
    _install_design_plan(monkeypatch)
    captured = []

    def model_factory(pydb, **kwargs):
        captured.append((pydb, kwargs))
        return SimpleNamespace(baseline=SimpleNamespace(as_dict=lambda: {}))

    monkeypatch.setattr(
        "dreamplace.ops.rebuffer.timing_model.RebufferTimingModel", model_factory
    )
    wire_rc = {"r_unit": 0.653, "c_unit": 1.8889e-4}

    result = repair_buffering.run_repair_buffering_stage(
        _Placedb(), mode="repair_design", config=wire_rc
    )

    assert result["status"] == "ok"
    assert captured == [
        ("base", {"use_pin_offset": False, **wire_rc}),
        ("refreshed", {"use_pin_offset": False, **wire_rc}),
    ]


def test_repair_timing_and_fresh_commit_reader_share_configured_wire_rc(monkeypatch):
    _install_common(monkeypatch)
    _, verified_calls = _install_timing_plan(monkeypatch, [_ACTION])
    captured = []

    def model_factory(pydb, **kwargs):
        captured.append((pydb, kwargs))
        return SimpleNamespace(baseline=SimpleNamespace(as_dict=lambda: {}))

    monkeypatch.setattr(
        "dreamplace.ops.rebuffer.timing_model.RebufferTimingModel", model_factory
    )
    monkeypatch.setattr(
        "dreamplace.ops.placeio_ecc.electrical_observation.electrical_observation",
        lambda model, **kwargs: {},
    )
    wire_rc = {"r_unit": 0.653, "c_unit": 1.8889e-4}

    result = repair_buffering.run_repair_buffering_stage(
        _Placedb(), mode="repair_timing", config=wire_rc
    )
    observation = verified_calls[0][1]["reader"]("committed")

    assert result["status"] == "ok"
    assert observation == {"electrical": {}}
    assert captured == [
        (
            "base",
            {
                "use_pin_offset": False,
                "use_persistent_graph": False,
                "use_native_timing_graph": True,
                "use_native_timing_evaluation": True,
                **wire_rc,
            },
        ),
        ("committed", {"use_pin_offset": False, **wire_rc}),
    ]


def test_design_selected_tree_profile_reaches_initial_and_fresh_python_tp(monkeypatch):
    _install_common(monkeypatch)
    calls = _install_design_plan(monkeypatch)
    captured = []

    def model_factory(pydb, **kwargs):
        captured.append((pydb, kwargs))
        return SimpleNamespace(
            baseline=SimpleNamespace(as_dict=lambda: {}),
            steiner_input_identity={"source": pydb, **kwargs},
        )

    monkeypatch.setattr("dreamplace.ops.rebuffer.timing_model.RebufferTimingModel", model_factory)
    config = {"steiner_provider": "openroad_stt", "r_unit": 0.653, "c_unit": 1.8889e-4}
    result = repair_buffering.run_repair_buffering_stage(
        _Placedb(), mode="repair_design", config=config
    )
    expected = {"use_pin_offset": True, **config}
    assert captured == [("base", expected), ("refreshed", expected)]
    assert calls[0][1].steiner_input_identity == {"source": "base", **expected}
    assert result["stages"]["repair_design"]["timing_after_steiner_input_identity"] == {
        "source": "refreshed",
        **expected,
    }


def test_repair_timing_uses_the_verified_native_defaults(tmp_path, monkeypatch):
    placedb = _Placedb()
    _install_common(monkeypatch)
    planner_configs, verified_calls = _install_timing_plan(monkeypatch, [_ACTION])

    result = repair_buffering.run_repair_buffering_stage(
        placedb, mode="repair_timing", max_passes=3, artifact_dir=tmp_path
    )

    stage = result["stages"]["repair_timing"]
    assert result["status"] == "ok"
    assert planner_configs == [repair_buffering.REPAIR_TIMING_DEFAULTS]
    assert stage["accepted_count"] == 1
    assert stage["verified_kept_count"] == 1
    assert stage["buffer_histogram"] == {"BUF": 1}
    assert verified_calls[0][1]["chunk_size"] == 0
    assert placedb.rebuild_calls == [("verified_refreshed", "all")]


def test_repair_timing_defaults_select_the_native_cpp_consumer():
    resolved = repair_buffering.resolve_repair_timing_config({})

    assert {
        name: resolved[name]
        for name in (
            "use_native_timing_graph",
            "use_native_timing_queue",
            "use_native_timing_evaluation",
            "use_native_timing_planning",
        )
    } == {
        "use_native_timing_graph": True,
        "use_native_timing_queue": True,
        "use_native_timing_evaluation": True,
        "use_native_timing_planning": True,
    }


def test_repair_timing_explicit_config_overrides_defaults(monkeypatch):
    _install_common(monkeypatch)
    planner_configs, verified_calls = _install_timing_plan(monkeypatch, [_ACTION])

    repair_buffering.run_repair_buffering_stage(
        _Placedb(),
        mode="repair_timing",
        config={"commit_verify_chunk": 4, "native_tree_round_robin_endpoint_window": 1},
    )

    assert planner_configs == [
        {
            **repair_buffering.REPAIR_TIMING_DEFAULTS,
            "commit_verify_chunk": 4,
            "native_tree_round_robin_endpoint_window": 1,
        }
    ]
    assert verified_calls[0][1]["chunk_size"] == 4


@pytest.mark.parametrize(
    "option", ["use_persistent_graph", "use_native_timing_graph", "use_native_timing_evaluation"]
)
def test_repair_timing_passes_graph_opt_in_to_model(monkeypatch, option):
    captured = []

    def model_factory(pydb, **kwargs):
        captured.append((pydb, kwargs))
        return SimpleNamespace(baseline=SimpleNamespace(as_dict=lambda: {}))

    monkeypatch.setattr(
        "dreamplace.ops.rebuffer.timing_model.RebufferTimingModel", model_factory
    )
    _install_timing_plan(monkeypatch, [_ACTION])

    result = repair_buffering.run_repair_buffering_stage(
        _Placedb(),
        mode="repair_timing",
        config={option: True},
    )

    assert result["status"] == "ok"
    assert captured[0][1]["use_pin_offset"] is False
    assert captured[0][1][option] is True


def test_repair_timing_without_accepted_actions_is_noop(monkeypatch):
    placedb = _Placedb()
    _install_common(monkeypatch)
    _, verified_calls = _install_timing_plan(monkeypatch, [])

    result = repair_buffering.run_repair_buffering_stage(placedb, mode="repair_timing")

    assert result["status"] == "ok"
    assert result["stages"]["repair_timing"]["status"] == "noop"
    assert verified_calls == []
    assert placedb.rebuild_calls == []


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"commit_verify_chunk": -1}, "commit_verify_chunk"),
        ({"python_timing_candidate_top_k": 0}, "python_timing_candidate_top_k"),
    ],
)
def test_repair_timing_rejects_invalid_config_before_planning(tmp_path, override, message):
    result = repair_buffering.run_repair_buffering_stage(
        _Placedb(), mode="repair_timing", config=override, artifact_dir=tmp_path
    )

    assert result["status"] == "invalid_config"
    assert message in result["failure_reason"]
    assert result["stages"] == {}
    assert (tmp_path / "repair_buffering_stage.json").is_file()


def test_combined_mode_runs_timing_on_the_design_committed_database(monkeypatch):
    placedb = _Placedb()
    _install_common(monkeypatch)
    _install_design_plan(monkeypatch)
    _install_timing_plan(monkeypatch, [_ACTION])
    sessions = []
    monkeypatch.setattr(
        "dreamplace.ops.placeio_ecc.buffer_session.PlaceIOBufferSession",
        lambda raw_db, *, timing_model: sessions.append(raw_db.pydb)
        or _Session(raw_db, timing_model=timing_model),
    )

    result = repair_buffering.run_repair_buffering_stage(placedb, mode="repair_design_then_timing")

    assert result["status"] == "ok"
    assert list(result["stages"]) == ["repair_design", "repair_timing"]
    assert sessions == ["refreshed"]


def test_stage_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unsupported repair buffering mode"):
        repair_buffering.run_repair_buffering_stage(_Placedb(), mode="unknown")


def test_reenter_after_repair_buffering_uses_fresh_legalization_placer(tmp_path, monkeypatch):
    import torch

    placedb = SimpleNamespace(topology_generation=7)
    params = SimpleNamespace(
        result_dir=str(tmp_path),
        repair_buffering_enable=1,
        repair_buffering_mode="repair_design",
        repair_buffering_max_passes=4,
        repair_buffering_config={"slew_margin": 0.0},
        global_place_flag=1,
        legalize_flag=0,
        timing_opt_enabled=1,
        routability_opt_flag=1,
        l_shape_routability_flag=1,
        enable_relaxed_buffer_timing=True,
        enable_fillers=1,
    )
    placer = SimpleNamespace(pos=[torch.zeros(2, 2)])
    monkeypatch.setattr(
        "dreamplace.ops.routability.gpugr_context.write_back_movable_lpos",
        lambda pos, params, placedb: None,
    )
    stage_calls = []
    monkeypatch.setattr(
        repair_buffering,
        "run_repair_buffering_stage",
        lambda placedb_arg, **kwargs: stage_calls.append((placedb_arg, kwargs)) or {"status": "ok"},
    )

    class _FreshPlacer:
        def __init__(self, params_arg, placedb_arg, third):
            pass

        def __call__(self, params_arg, placedb_arg):
            return {"status": "ok", "params": params_arg}

    monkeypatch.setattr("dreamplace.NonLinearPlace.NonLinearPlace", _FreshPlacer)

    result = repair_buffering.reenter_after_repair_buffering(placer, params, placedb)

    assert result[2]["repair_buffering_stage"]["status"] == "ok"
    assert stage_calls[0][0] is placedb
    assert stage_calls[0][1]["topology_generation"] == 7
    assert len(stage_calls[0][1]["placement_snapshot_digest"]) == 64
    terminal = result[2]["legalization"]["params"]
    assert terminal.repair_buffering_enable == 0
    assert terminal.global_place_flag == 0
    assert terminal.legalize_flag == 1
    assert terminal.timing_opt_enabled == 0
    assert terminal.result_dir.endswith("_repair_buffering_terminal")


def test_repair_timing_resolves_max_buffer_fraction_from_movable_nodes(monkeypatch):
    _install_common(monkeypatch)
    planner_configs, _ = _install_timing_plan(monkeypatch, [_ACTION])
    placedb = _Placedb()
    placedb.num_movable_nodes = 2500

    result = repair_buffering.run_repair_buffering_stage(
        placedb,
        mode="repair_timing",
        max_passes=20,
        config={"max_buffer_fraction": 0.001},
    )

    assert result["status"] == "ok"
    assert planner_configs[0]["max_buffers"] == 3
