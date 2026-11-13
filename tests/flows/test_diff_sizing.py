import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from dreamplace.Placer import PlacementEngine


def test_sizing_lut_action_dispatches_to_diff_sizing():
    engine = object.__new__(PlacementEngine)
    engine.params = SimpleNamespace(
        diff_opt_enabled=0,
        flow_kind="sizing",
        timing_action_gradient_mode="lut_action",
    )
    sentinel = {"status": "ok", "flow_kind": "sizing"}
    engine.run_diff_sizing = lambda: sentinel

    assert engine.run() is sentinel


def test_diff_sizing_controller_uses_one_action_phase(monkeypatch):
    from dreamplace.flows import diff_sizing

    calls = []

    class Owner:
        capacity = 17.0

        def __init__(self, params, placedb, data, ops, refresh_topology, **kwargs):
            calls.append(("init", kwargs))

        @contextmanager
        def electrical_window(self):
            calls.append(("window", None))
            yield

        def run_sizing(
            self,
            model,
            pos,
            geometry,
            capacity,
            *,
            action_mask=None,
        ):
            calls.append(("sizing", geometry, capacity, action_mask))
            return {
                "best_round": 3,
                "best_loss": -2.0,
                "initial_objective": {"score": -2.0},
                "best_objective": {"score": -1.0},
            }

        def publish_geometry(self, model):
            calls.append(("publish_geometry", model))
            return {"status": "ok"}

        def sync_native_sizing(self):
            calls.append(("commit", None))
            return {"ok": True, "accepted_count": 2}

    monkeypatch.setattr(diff_sizing, "DiscreteTimingOwner", Owner)
    monkeypatch.setattr(
        diff_sizing,
        "SizingWindowGeometry",
        lambda data, placedb, pos: "geometry",
    )
    params = SimpleNamespace(timing_opt_sizing_rounds=1)
    controller = diff_sizing.DiffSizingController(
        params,
        object(),
        object(),
        object(),
        lambda *_: None,
    )
    model = SimpleNamespace()
    pos = object()

    result = controller.run(model, pos, 50)

    assert params.timing_opt_sizing_rounds == 1
    assert calls == [
        (
            "init",
            {"refresh_candidates_after_sizing": False},
        ),
        ("window", None),
        ("sizing", "geometry", 17.0, (True, True, True, True)),
        ("publish_geometry", model),
        ("commit", None),
    ]
    sizing = result["sizing"]
    assert sizing["best_round"] == 3
    assert sizing["best_loss"] == -2.0
    assert sizing["phase_count"] == 1
    assert sizing["phases"][0]["action"] == "quad"
    assert result["sizing_commit"] == {"ok": True, "accepted_count": 2}
    assert not hasattr(params, "discrete_gradient_topk_shared_budget_percent")


def test_diff_sizing_controller_runs_ordered_action_phases(monkeypatch):
    from dreamplace.flows import diff_sizing
    from dreamplace.flows.diff_sizing_spec import parse_diff_sizing_flows

    calls = []

    class Owner:
        capacity = 1.0

        def __init__(self, params, *args, **kwargs):
            self.params = params

        @contextmanager
        def electrical_window(self):
            yield

        def run_sizing(
            self,
            model,
            pos,
            geometry,
            capacity,
            *,
            action_mask=None,
        ):
            calls.append(
                {
                    "mask": action_mask,
                    "rounds": self.params.timing_opt_sizing_rounds,
                    "percent": self.params.discrete_gradient_topk_shared_budget_percent,
                    "coefficients": {
                        name: getattr(model, field)
                        for name, field in diff_sizing._COEFFICIENT_FIELDS.items()
                    },
                }
            )
            return {
                "rounds": [{"round": 1}],
                "best_round": 1,
                "best_loss": -0.5,
                "initial_objective": {"score": -1.0},
                "best_objective": {"score": -0.5},
            }

        def publish_geometry(self, model):
            return {"status": "ok"}

        def sync_native_sizing(self):
            return {"ok": True}

    monkeypatch.setattr(diff_sizing, "DiscreteTimingOwner", Owner)
    monkeypatch.setattr(
        diff_sizing,
        "SizingWindowGeometry",
        lambda data, placedb, pos: "geometry",
    )
    params = SimpleNamespace(
        timing_opt_sizing_rounds=7,
        discrete_gradient_topk_shared_budget_percent=2.0,
    )
    controller = diff_sizing.DiffSizingController(
        params,
        object(),
        object(),
        object(),
        lambda *_: None,
    )
    model = SimpleNamespace(
        timing_wns_coeff=1.0,
        timing_tns_coeff=1.0,
        timing_slew_weight=1.0,
        timing_cap_weight=1.0,
    )
    phases = parse_diff_sizing_flows(
        [
            ["up", 2, [500, 5, 1, 1], 1],
            ["down", 1, [100, 2, 3, 4], 0.1],
            ["quad", 3, [10, 1, 1, 1], 5],
        ]
    )

    result = controller.run(model, object(), phases)

    assert [call["mask"] for call in calls] == [
        (True, False, False, True),
        (False, True, True, False),
        (True, True, True, True),
    ]
    assert [call["rounds"] for call in calls] == [2, 1, 3]
    assert [call["percent"] for call in calls] == [1.0, 0.1, 5.0]
    assert [call["coefficients"] for call in calls] == [
        {"wns": 500.0, "tns": 5.0, "slew": 1.0, "cap": 1.0},
        {"wns": 100.0, "tns": 2.0, "slew": 3.0, "cap": 4.0},
        {"wns": 10.0, "tns": 1.0, "slew": 1.0, "cap": 1.0},
    ]
    assert params.timing_opt_sizing_rounds == 7
    assert params.discrete_gradient_topk_shared_budget_percent == 2.0
    assert model.timing_wns_coeff == 1.0
    assert model.timing_tns_coeff == 1.0
    assert model.timing_slew_weight == 1.0
    assert model.timing_cap_weight == 1.0
    sizing = result["sizing"]
    assert sizing["phase_count"] == 3
    assert sizing["total_rounds"] == 6
    assert [phase["action"] for phase in sizing["phases"]] == [
        "up",
        "down",
        "quad",
    ]
    assert [record["phase"] for record in sizing["rounds"]] == [1, 2, 3]


@pytest.mark.parametrize(
    ("timing_rc_mode", "terminal_gr_sta_enabled", "expected_reason"),
    [
        ("gr", 0, "diff_sizing_terminal_gr_sta_disabled"),
        ("placement", 1, "diff_sizing_timing_rc_mode_placement"),
        ("spef", 0, "diff_sizing_timing_rc_mode_spef"),
    ],
)
def test_run_diff_sizing_commits_refreshes_and_writes_summary(
    monkeypatch, tmp_path, timing_rc_mode, terminal_gr_sta_enabled, expected_reason
):
    from dreamplace.flows import diff_sizing

    calls = []
    gr_session = object()

    class Controller:
        def __init__(self, params, placedb, data, ops, refresh_topology):
            calls.append(("init", params, placedb, data, ops, refresh_topology))

        def run(self, model, pos, phases):
            calls.append(("run", model, pos, phases))
            return {
                "sizing": {"best_round": 2},
                "sizing_commit": {"ok": True, "accepted_count": 1},
            }

    class Engine:
        def __init__(self):
            self.params = SimpleNamespace(
                timing_action_gradient_mode="lut_action",
                flow_kind="sizing",
                timing_rc_mode=timing_rc_mode,
                placement_sizing_mode="size_only",
                diff_sizing_continuous_steps=0,
                global_place_stages=[{"iteration": 2}],
                timing_opt_sizing_rounds=10,
                diff_sizing_terminal_gr_sta_enabled=terminal_gr_sta_enabled,
                scale_factor=2.0,
                result_dir=str(tmp_path),
            )
            self.placedb = SimpleNamespace(
                gr_sizing=(
                    gr_session if timing_rc_mode in {"gr", "spef"} else None
                ),
                pin_offset_x=np.array([0.0, 0.0], dtype=np.float32),
                pin_offset_y=np.array([0.0, 0.0], dtype=np.float32),
                rawdb=SimpleNamespace(
                    pydb=SimpleNamespace(
                        pin_offset_x=[1.5, 2.5],
                        pin_offset_y=[3.5, 4.5],
                        native_state_dirty=True,
                    )
                ),
            )
            self.placer = SimpleNamespace(
                data_collections=SimpleNamespace(pos=[object()]),
                op_collections=object(),
                _refresh_live_timing_topology=object(),
            )
            self.rsmt = 1.0
            self.hpwl = 2.0

        def setup_placedb(self):
            calls.append(("setup_placedb", None))

        def _physical_mutation_backend(self):
            return SimpleNamespace(
                refresh=lambda **kwargs: calls.append(("refresh", kwargs))
                or {"status": "ok"},
            )

    engine = Engine()
    monkeypatch.setattr(diff_sizing, "DiffSizingController", Controller)
    monkeypatch.setattr(
        diff_sizing, "_build_timing_model", lambda current: "model"
    )
    monkeypatch.setattr(
        diff_sizing,
        "_legalize_and_measure",
        lambda current, params: {"wns": -1.0, "tns": -2.0},
    )

    result = diff_sizing.run_diff_sizing(engine)

    assert calls[0] == ("setup_placedb", None)
    assert calls[1][0] == "init"
    assert calls[1][1] is engine.params
    assert calls[1][2] is engine.placedb
    assert calls[1][3] is engine.placer.data_collections
    assert calls[1][4] is engine.placer.op_collections
    assert calls[1][5] is engine.placer._refresh_live_timing_topology
    assert calls[2][0] == "run"
    assert calls[2][1] == "model"
    assert calls[2][2] is engine.placer.data_collections.pos[0]
    phases = calls[2][3]
    assert len(phases) == 1
    assert phases[0].action == "quad"
    assert phases[0].rounds == 2
    assert len(calls) == 3
    assert result["diff_sizing"]["refresh"] == {
        "status": "skipped",
        "reason": "inplace_sizing_sync",
        "refresh_mode": "inplace",
        "rebuild_mode": "none",
        "accepted_count": 1,
        "pin_offsets_synced": True,
    }
    np.testing.assert_allclose(engine.placedb.pin_offset_x, [3.0, 5.0])
    np.testing.assert_allclose(engine.placedb.pin_offset_y, [7.0, 9.0])
    assert engine.placedb.rawdb.pydb.native_state_dirty is False
    assert engine.placedb.gr_sizing is (
        gr_session if timing_rc_mode in {"gr", "spef"} else None
    )
    assert engine.params.timing_action_gradient_mode == "lut_action"
    assert engine.params.timing_opt_sizing_rounds == 10
    assert result["status"] == "ok"
    assert result["diff_sizing"]["terminal_timing"] == {
        "wns": -1.0,
        "tns": -2.0,
    }
    assert result["gr_sizing"]["status"] == "skipped"
    assert result["gr_sizing"]["reason"] == expected_reason
    summary_path = result["diff_sizing_summary_path"]
    summary = json.loads(Path(summary_path).read_text(encoding="utf-8"))
    assert summary["artifact"] == "diff_sizing"
    assert summary["artifact_version"] == 1
    assert summary["mode"] == "lut_action"
    assert summary["status"] == "ok"
    assert summary["final_timing"] == {"wns": -1.0, "tns": -2.0}
    assert engine.last_run_result is result


@pytest.mark.parametrize(
    ("timing_rc_mode", "initially_legal", "expect_legalizer_call"),
    [
        ("spef", True, False),
        ("spef", False, True),
        ("gr", True, True),
    ],
)
def test_legalize_and_measure_preserves_legal_postroute_placement_in_spef_mode(
    monkeypatch, tmp_path, timing_rc_mode, initially_legal, expect_legalizer_call
):
    import dreamplace.NonLinearPlace as nlp_mod
    import torch
    from dreamplace.flows import diff_sizing

    calls = []

    class FakePlacer:
        def __init__(self, terminal_params, placedb, _):
            self.pos = [torch.tensor([1.0, 2.0])]
            self._legal = initially_legal
            self.op_collections = SimpleNamespace(
                legality_check_op=lambda _pos: self._legal,
                hpwl_op=lambda _pos: torch.tensor(123.5),
            )

        def __call__(self, terminal_params, placedb):
            calls.append("legalize")
            self._legal = True
            return 10.0, 20.0, {"timing_stage_summary": {}}

        def run_sta_only(self, terminal_params, placedb):
            calls.append("sta")
            return float("inf"), float("inf"), {
                "timing_stage_summary": {"sta": {"wns": -0.25, "tns": -1.5}}
            }

    monkeypatch.setattr(nlp_mod, "NonLinearPlace", FakePlacer)
    params = SimpleNamespace(timing_rc_mode=timing_rc_mode, result_dir=str(tmp_path))
    engine = SimpleNamespace(placedb=SimpleNamespace(cooptimization_geometry="old"))

    timing = diff_sizing._legalize_and_measure(engine, params)

    assert timing == {"wns": -0.25, "tns": -1.5}
    assert calls == (["legalize", "sta"] if expect_legalizer_call else ["sta"])
    assert engine.hpwl == (20.0 if expect_legalizer_call else 123.5)


def test_build_timing_model_forces_lut_only_and_restores_params():
    from dreamplace.flows import diff_sizing

    seen_modes = []

    class FakeEngine:
        def __init__(self):
            self.params = SimpleNamespace(
                flow_kind="sizing",
                global_place_flag=1,
                legalize_flag=1,
                detailed_place_flag=1,
                global_place_stages=[{"iteration": 50}],
                timing_surrogate_mode="surrogate_only",
                build_model_only=False,
            )
            self.placedb = SimpleNamespace(gr_sizing=object())
            self.placer = None

        def place(self):
            seen_modes.append(self.params.timing_surrogate_mode)
            self.placer = SimpleNamespace(last_global_place_model="fake_model")

    engine = FakeEngine()
    assert diff_sizing._build_timing_model(engine) == "fake_model"
    assert seen_modes == ["lut_only"]
    assert engine.params.timing_surrogate_mode == "surrogate_only"


@pytest.mark.parametrize(
    ("action_mode", "expected_surrogate_mode"),
    [("lut_action", "lut_only"), ("off", "surrogate_only")],
)
def test_configure_gr_sizing_sets_surrogate_mode_by_action_mode(
    action_mode, expected_surrogate_mode
):
    from dreamplace.flows.gr_sizing_config import configure_gr_sizing

    params = SimpleNamespace(
        flow_kind="sizing",
        timing_rc_mode="gr",
        place_io_engine="ecc",
        gpu=0,
        gpugr_backend="cpu_pr_mt",
        routability_opt_flag=0,
        l_shape_routability_flag=0,
        adjust_gpugr_area_flag=0,
        gpugr_final_eval_flag=0,
        enable_net_weighting=0,
        pin2pin_net_weighting=0,
        timing_opt_enabled=0,
        timing_action_gradient_mode=action_mode,
        diff_sizing_continuous_steps=0,
        global_place_stages=[{"iteration": 100}],
    )
    configure_gr_sizing(params)
    assert params.timing_surrogate_mode == expected_surrogate_mode
