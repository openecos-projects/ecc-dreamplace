import hashlib
import gc
import json
from types import SimpleNamespace

import torch

from dreamplace.ops.buffer_insertion.discrete_candidate_buffer_runner import (
    run_discrete_candidate_buffer_scheduler,
    update_discrete_candidate_scheduler_trace_commit,
)
from dreamplace.ops.buffer_insertion.optimization_state import (
    build_discrete_candidate_state,
)
from dreamplace.ops.buffer_insertion.projection import project_candidate_buffer_state
from dreamplace.NonLinearPlace import NonLinearPlace


def _candidate(candidate_id, net_id, node_id, x_dbu):
    return {
        "candidate_id": candidate_id,
        "net_id": net_id,
        "candidate_node_id": node_id,
        "buffer_main_type_index": 4,
        "x_dbu": x_dbu,
        "y_dbu": 20,
    }


class _FakeCandidateLane:
    def __init__(self, output_dir, *, improving=True):
        self.config = SimpleNamespace(
            mode="candidate",
            candidate_strategy="discrete_net_gradient",
            active_strategy="discrete_net_gradient",
            fixed_bsu_index=1,
            max_selected_actions=None,
            segment_capacity_enabled=False,
            output_dir=str(output_dir),
            commit_enabled=False,
        )
        self.state = build_discrete_candidate_state(
            [_candidate(11, 5, 2, 10), _candidate(22, 7, 4, 30)],
            buffer_main_type_index=4,
            legal_buffer_count=3,
            fixed_bsu_index=1,
        )
        self.improving = improving
        self.objective_calls = 0
        self.project_calls = 0
        self.refresh_calls = 0

    def _current_state(self):
        return self.state

    def objective(self, _model):
        self.objective_calls += 1
        sign = -1.0 if self.improving else 1.0
        loss = sign * self.state.activation_param.sum()
        return loss, {"loss": loss.detach(), "tns": -loss, "wns": -loss}

    def project(self, **_kwargs):
        self.project_calls += 1
        return project_candidate_buffer_state(self.state)

    def refresh_runtime_state(self, projection):
        self.refresh_calls += 1
        return {
            "status": "ok",
            "projected_buffer_count": projection.projected_buffer_count,
        }

    def summarize(self):
        return {
            "mode": "candidate",
            "state_kind": "candidate",
            "candidate_strategy": "discrete_net_gradient",
            "net_subgraph_forward_backend": "unit",
        }


def test_candidate_runner_applies_binary_rounds_and_terminal_forward(tmp_path):
    lane = _FakeCandidateLane(tmp_path)

    summary = run_discrete_candidate_buffer_scheduler(
        lane=lane,
        model=SimpleNamespace(),
        rounds=2,
    )

    assert summary.strategy == "discrete_net_gradient"
    assert summary.iterations == 2
    assert summary.terminal_reason == "max_rounds"
    assert lane.objective_calls == 3
    assert lane.project_calls == 1
    assert lane.refresh_calls == 1
    assert lane.state.activation_param.tolist() == [1.0, 1.0]
    assert lane.state.activation_param.grad is None
    assert summary.final_projection.projected_buffer_count == 2
    assert summary.terminal_metrics["total_virtual_buffer_count"] == 2
    payload = json.loads((tmp_path / "candidate_discrete_scheduler_trace.json").read_text())
    assert payload["artifact"] == "candidate_discrete_scheduler_trace"
    assert payload["selection_fraction"] == 0.001
    assert payload["rounds"][0]["accepted_action_count"] == 1
    assert payload["rounds"][1]["accepted_action_count"] == 1
    assert payload["rounds"][0]["activation_binary_before"]
    assert payload["rounds"][0]["activation_binary_after"]
    assert payload["rounds"][0]["active_candidate_count_before"] == 0
    assert payload["rounds"][0]["active_candidate_count_after"] == 1
    assert payload["rounds"][0]["raw_grad_nonfinite_count"] == 0
    assert payload["runtime"]["cold_forward_ms"] >= 0.0
    assert payload["projection"]["exact_active_candidate_match"]


def test_candidate_runner_stops_without_mutation_when_no_positive_action(tmp_path):
    lane = _FakeCandidateLane(tmp_path, improving=False)

    summary = run_discrete_candidate_buffer_scheduler(
        lane=lane,
        model=SimpleNamespace(),
        rounds=5,
    )

    assert summary.iterations == 1
    assert summary.terminal_reason == "no_positive_action"
    assert lane.state.activation_param.tolist() == [0.0, 0.0]
    assert summary.final_projection.projected_buffer_count == 0


def test_candidate_runner_restores_gc_after_releasing_projection_records(tmp_path):
    lane = _FakeCandidateLane(tmp_path)
    lane.buffer_relaxed_timing_payload = {
        "metadata": {"python_gc_suspended_until_projection": True}
    }
    lane.data_collections = SimpleNamespace(
        buffering_nets=[{"net_id": 5}],
        buffering_candidate_rows=[{"candidate_id": 11}],
    )
    gc_was_enabled = gc.isenabled()
    gc.disable()
    try:
        summary = run_discrete_candidate_buffer_scheduler(
            lane=lane,
            model=SimpleNamespace(),
            rounds=1,
        )

        assert summary.final_projection.projected_buffer_count == 1
        assert lane.state.candidate_records == ()
        assert lane.data_collections.buffering_nets == ()
        assert lane.data_collections.buffering_candidate_rows == ()
        assert gc.isenabled()
        assert lane.buffer_relaxed_timing_payload["metadata"][
            "candidate_python_state_released"
        ]
    finally:
        if not gc_was_enabled:
            gc.disable()
        elif not gc.isenabled():
            gc.enable()


def test_candidate_trace_commit_update_preserves_artifact_identity(tmp_path):
    lane = _FakeCandidateLane(tmp_path)
    summary = run_discrete_candidate_buffer_scheduler(
        lane=lane,
        model=SimpleNamespace(),
        rounds=1,
    )

    digest = update_discrete_candidate_scheduler_trace_commit(
        summary.trace_path,
        {"status": "accepted", "accepted_action_count": 1},
    )

    payload = json.loads((tmp_path / "candidate_discrete_scheduler_trace.json").read_text())
    assert payload["commit"]["status"] == "accepted"
    assert payload["commit"]["accepted_action_count"] == 1
    assert len(digest) == 64


def test_placement_commit_handoff_updates_candidate_trace_and_summary(tmp_path):
    lane = _FakeCandidateLane(tmp_path)
    summary = run_discrete_candidate_buffer_scheduler(
        lane=lane,
        model=SimpleNamespace(),
        rounds=1,
    )
    engine = object.__new__(NonLinearPlace)
    engine.params = SimpleNamespace(flow_kind="buffering")
    engine.joint_buffer_commit_trace = []
    engine.last_joint_flow_artifact = None
    engine.last_buffering_lane_summary = {
        "mode": "candidate",
        "strategy": "discrete_net_gradient",
        "trace_path": summary.trace_path,
        "trace_sha256": summary.trace_sha256,
        "inner_loop": {"trace_sha256": summary.trace_sha256},
    }
    engine._joint_post_commit_control_result = lambda _result: {
        "status": "safe_stopped_after_accepted_buffer_commit",
        "accepted_action_count": 1,
    }
    engine._update_joint_flow_artifact = lambda _params: None
    engine._buffering_flow_enabled = lambda _params: True
    written = {}
    engine._write_buffering_inner_loop_artifact = (
        lambda _params, payload: written.update(payload=payload)
    )

    NonLinearPlace.record_outer_buffer_commit_result(
        engine,
        {
            "commit_request": {"action_count": 1},
            "commit": {"status": "accepted", "accepted_action_count": 1},
        },
    )

    trace_path = tmp_path / "candidate_discrete_scheduler_trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["commit"] == {"status": "accepted", "accepted_action_count": 1}
    assert written["payload"]["trace_sha256"] == hashlib.sha256(
        trace_path.read_bytes()
    ).hexdigest()
    assert (
        written["payload"]["inner_loop"]["trace_sha256"]
        == written["payload"]["trace_sha256"]
    )
