from types import SimpleNamespace

from dreamplace.Placer import PlacementEngine


def test_completed_buffering_lane_is_a_successful_flow():
    engine = PlacementEngine.__new__(PlacementEngine)
    engine.params = SimpleNamespace(flow_kind="buffering")
    engine.setup_placedb = lambda: None
    engine.place = lambda: None
    engine._post_commit_control_result = lambda: None
    engine._continue_after_joint_buffer_commit = lambda _result: None
    engine.placer = SimpleNamespace(
        last_buffering_lane_summary={
            "status": "completed",
            "mode": "candidate",
            "commit": {"status": "accepted"},
        }
    )

    result = engine.run()

    assert result == {
        "status": "ok",
        "flow_kind": "buffering",
        "buffering_mode": "candidate",
        "buffering_commit_status": "accepted",
    }
    assert engine.last_run_result == result
