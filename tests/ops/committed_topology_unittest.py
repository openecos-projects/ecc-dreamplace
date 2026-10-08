import json
from types import SimpleNamespace

import pytest
from dreamplace.Placer import PlacementEngine


@pytest.mark.parametrize(
    ("backend", "flow_kind"),
    [("openroad", "joint"), ("ecc", "joint"), ("openroad", "buffering"), ("ecc", "buffering")],
)
def test_committed_topology_rebuilds_fresh_runtime(monkeypatch, tmp_path, backend, flow_kind):
    commit_artifact_path = tmp_path / "buffering_coordinate_commit_results.json"
    commit_artifact_path.write_text(
        json.dumps(
            {
                "artifact": "buffering_coordinate_commit_results",
                "results": [
                    {
                        "accepted": True,
                        "inserted_buffer_name": "buffer_coord_buf_0",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    refresh_mode = "full_rebuild" if backend == "ecc" else "topo"
    params = SimpleNamespace(
        result_dir=str(tmp_path),
        base_design_name="unit_design",
        flow_kind=flow_kind,
        place_io_engine=backend,
        joint_post_commit_continue=True,
        joint_post_commit_openroad_eco="none",
        plot_flag=False,
        enable_relaxed_buffer_timing=True,
        buffering_segment_count_tns_gradient=1,
    )
    engine = PlacementEngine.__new__(PlacementEngine)
    engine.params = params
    engine.rsmt = float("nan")
    engine.hpwl = float("nan")
    engine.congestion = float("inf")
    engine.density = float("inf")
    engine.metrics = {
        "post_commit_control_result": {
            "status": "safe_stopped_after_accepted_buffer_commit",
            "topology_mutated": True,
            "refresh_required": True,
            "refresh_mode": refresh_mode,
            "rebuild_mode": refresh_mode,
            "continuation_policy": "rebuild_placer_outer_loop",
            "accepted_action_count": 1,
            "commit": {
                "status": "accepted",
                "accepted_action_count": 1,
                "before_wns": -10.0,
                "before_tns": -20.0,
                "after_wns": -9.0,
                "after_tns": -15.0,
                "delta_wns": 1.0,
                "delta_tns": 5.0,
                "artifact_path": str(commit_artifact_path),
                "committed_def_path": str(tmp_path / "committed.def"),
            },
        }
    }
    joint_flow_artifact_path = tmp_path / "unit_design_joint_flow_summary.json"
    joint_flow_artifact_path.write_text(
        json.dumps(
            {
                "artifact_version": 1,
                "summary": {
                    "buffer_commit_trace": [
                        {
                            "commit": {
                                "status": "accepted",
                                "accepted_action_count": 1,
                            }
                        }
                    ],
                    "post_commit_control_result": {
                        "status": "safe_stopped_after_accepted_buffer_commit"
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    class FakePlaceDB:
        topology_generation = 0

        def __init__(self):
            self.openroad_bridge = SimpleNamespace(eval_tcl_string=lambda cmd: "")

        def refresh_from_openroad_bridge(self, refresh_mode="topo", rebuild_mode="topo"):
            self.topology_generation += 1
            self.refresh_args = (refresh_mode, rebuild_mode)
            return {
                "status": "ok",
                "requested_refresh_mode": refresh_mode,
                "effective_refresh_mode": refresh_mode,
                "rebuild": {
                    "status": "ok",
                    "requested_rebuild_mode": rebuild_mode,
                    "effective_rebuild_mode": rebuild_mode,
                    "topology_generation": self.topology_generation,
                    "pre_counts": {"nodes": 2, "pins": 3, "nets": 1},
                    "post_counts": {"nodes": 3, "pins": 5, "nets": 2},
                },
            }

    constructed = []

    class FakeNonLinearPlace:
        def __init__(self, params_arg, placedb_arg, timer):
            self.params = params_arg
            self.placedb = placedb_arg
            constructed.append(self)

        def run_sta_only(self, params_arg, placedb_arg):
            assert params_arg.enable_relaxed_buffer_timing is False
            assert params_arg.buffering_segment_count_tns_gradient == 0
            return 1.0, 2.0, {"wns": -3.0, "tns": -4.0}

    old_placer = SimpleNamespace(last_joint_flow_artifact_path=str(joint_flow_artifact_path))
    engine.placer = old_placer
    engine.placedb = FakePlaceDB()
    engine._physical_mutation_backend = lambda: SimpleNamespace(
        refresh=engine.placedb.refresh_from_openroad_bridge
    )
    monkeypatch.setattr(
        "dreamplace.Placer.NonLinearPlace.NonLinearPlace",
        FakeNonLinearPlace,
    )

    result = engine._continue_after_joint_buffer_commit(engine._post_commit_control_result())

    assert result["status"] == "ok"
    assert result["fresh_nonlinear_place_constructed"]
    assert result["old_nonlinear_place_id"] == id(old_placer)
    assert result["new_nonlinear_place_id"] == id(constructed[0])
    assert engine.placer is constructed[0]
    assert engine.placedb.refresh_args == (refresh_mode, refresh_mode)
    assert result["post_rebuild_timing_status"] == "ok"
    assert result["post_rebuild_timing_mode"] == "committed_topology_plain_pysta"
    assert result["tns"] == -4.0
    assert params.enable_relaxed_buffer_timing is True
    assert params.buffering_segment_count_tns_gradient == 1
    assert (tmp_path / "unit_design_joint_post_rebuild_summary.json").exists()
    refresh_summary_path = tmp_path / "unit_design_buffer_commit_pydb_refresh_summary.json"
    assert refresh_summary_path.exists()
    refresh_summary = json.loads(refresh_summary_path.read_text(encoding="utf-8"))
    assert refresh_summary["artifact_scope"] == "buffer_commit_pydb_refresh"
    assert refresh_summary["commit_status"] == "accepted"
    assert refresh_summary["accepted_action_count"] == 1
    assert refresh_summary["before_counts"] == {"nodes": 2, "pins": 3, "nets": 1}
    assert refresh_summary["after_counts"] == {"nodes": 3, "pins": 5, "nets": 2}
    assert refresh_summary["committed_buffer_names"] == ["buffer_coord_buf_0"]
    assert refresh_summary["committed_buffer_names_found"] is True
    assert refresh_summary["opensta_before"] == {"wns": -10.0, "tns": -20.0}
    assert refresh_summary["opensta_after"] == {"wns": -9.0, "tns": -15.0}
    assert refresh_summary["delta_tns"] == 5.0
    assert refresh_summary["refresh_status"] == "ok"
    assert refresh_summary["refresh_generation"] == 1
    assert refresh_summary["continuation_policy"] == "rebuild_placer_outer_loop"
    assert result["buffer_commit_pydb_refresh_summary_path"] == str(refresh_summary_path)
    post_rebuild_summary = json.loads(
        (tmp_path / "unit_design_joint_post_rebuild_summary.json").read_text(encoding="utf-8")
    )
    assert post_rebuild_summary["post_rebuild_result"][
        "buffer_commit_pydb_refresh_summary_path"
    ] == str(refresh_summary_path)
    joint_flow_artifact = json.loads(joint_flow_artifact_path.read_text(encoding="utf-8"))
    assert joint_flow_artifact["summary"]["buffer_commit_pydb_refresh_summary_path"] == str(
        refresh_summary_path
    )
    commit_artifact = json.loads(commit_artifact_path.read_text(encoding="utf-8"))
    assert commit_artifact["buffer_commit_pydb_refresh_summary_path"] == str(refresh_summary_path)
