import json
import hashlib
from pathlib import Path
from types import SimpleNamespace

import torch

from dreamplace.NonLinearPlace import NonLinearPlace
from dreamplace.ops.buffer_insertion.discrete_virtual_buffer_runner import (
    _discrete_count_proximal_action_grad,
    run_discrete_virtual_buffer_scheduler,
    update_discrete_virtual_scheduler_trace_commit,
)


class _FakeDiscreteLane:
    def __init__(self, output_dir):
        self.config = SimpleNamespace(
            fixed_bsu_index=2,
            output_dir=str(output_dir),
            segment_projection_min_z_to_insert=0.5,
            segment_projection_require_setup_criticality=False,
            max_selected_actions=None,
            discrete_count_proximal_lambda=0.0,
        )
        self.state = SimpleNamespace(
            z_param=torch.nn.Parameter(torch.tensor([0.0, 0.0])),
            bsu_index_param=torch.nn.Parameter(torch.tensor([2.0, 2.0])),
            fixed_bsu_index=2,
            segment_net_index=torch.tensor([0, 1], dtype=torch.long),
            net_ids=torch.tensor([10, 20], dtype=torch.long),
            segment_ids=torch.tensor([5, 3], dtype=torch.long),
            max_repeater_count=2,
            segment_rows=(
                {"segment_id": 5, "net_id": 10},
                {"segment_id": 3, "net_id": 20},
            ),
        )
        self.objective_calls = 0
        self.project_calls = 0
        self.refresh_calls = 0

    def _current_state(self):
        return self.state

    def objective(self, _model):
        self.objective_calls += 1
        loss = -self.state.z_param.sum()
        return loss, {
            "loss": loss.detach(),
            "tns": -loss,
            "wns": -loss,
        }

    def project(self, *, iteration, final):
        self.project_calls += 1
        actions = []
        for row_index, segment_id in enumerate(self.state.segment_ids.tolist()):
            count = int(self.state.z_param[row_index].item())
            for _ in range(count):
                actions.append(
                    {
                        "segment_id": segment_id,
                        "projected_repeater_count": count,
                    }
                )
        return SimpleNamespace(
            selected_actions=tuple(actions),
            projected_buffer_count=len(actions),
            metadata={"iteration": iteration, "final": final},
        )

    def refresh_runtime_state(self, projection):
        self.refresh_calls += 1
        return {"projected_buffer_count": projection.projected_buffer_count}


class _NoBenefitDiscreteLane(_FakeDiscreteLane):
    def objective(self, _model):
        self.objective_calls += 1
        loss = self.state.z_param.sum()
        return loss, {
            "loss": loss.detach(),
            "tns": -loss,
            "wns": -loss,
        }


class _BacktrackingDiscreteLane(_FakeDiscreteLane):
    def __init__(self, output_dir):
        super().__init__(output_dir)
        self.state.z_param = torch.nn.Parameter(torch.zeros(2000))
        self.state.bsu_index_param = torch.nn.Parameter(torch.full((2000,), 2.0))
        self.state.segment_net_index = torch.arange(2000, dtype=torch.long)
        self.state.net_ids = torch.arange(2000, dtype=torch.long)
        self.state.segment_ids = torch.arange(2000, dtype=torch.long)
        self.state.segment_rows = tuple(
            {"segment_id": index, "net_id": index} for index in range(2000)
        )

    def objective(self, _model):
        self.objective_calls += 1
        z = self.state.z_param
        loss = -2.0 * z[0] - z[1] + 4.0 * z[0] * z[1] + z[2:].sum()
        return loss, {"loss": loss.detach(), "tns": -loss, "wns": -loss}


def test_discrete_count_proximal_uses_one_step_finite_increment():
    raw_grad = torch.tensor([-2.0, -1.0, 3.0, 4.0])
    counts = torch.tensor([0.0, 1.0, 0.0, 2.0])

    action_grad, metadata = _discrete_count_proximal_action_grad(
        raw_grad,
        counts,
        2.0,
    )

    assert torch.allclose(action_grad, raw_grad + 0.25)
    assert metadata["finite_increment_applied"]
    assert metadata["normalizer"] == 4
    assert metadata["correction"] == 0.25


def test_discrete_runner_uses_z_only_gradient_and_terminal_integer_evaluation(tmp_path):
    lane = _FakeDiscreteLane(tmp_path)

    summary = run_discrete_virtual_buffer_scheduler(
        lane=lane,
        model=object(),
        rounds=2,
    )

    assert summary.strategy == "discrete_net_gradient"
    assert summary.iterations == 2
    assert summary.terminal_reason == "max_rounds"
    assert lane.objective_calls == 5
    assert lane.project_calls == 1
    assert lane.refresh_calls == 1
    assert lane.state.z_param.grad is None
    assert lane.state.bsu_index_param.grad is None
    assert torch.equal(lane.state.z_param.detach(), torch.tensor([0.0, 2.0]))
    assert summary.final_projection.projected_buffer_count == 2
    assert summary.trace_path
    trace = json.loads(Path(summary.trace_path).read_text(encoding="utf-8"))
    assert trace["projection"]["active_segment_count"] == 1
    assert "per_segment_count" not in trace["projection"]
    assert trace["rounds"][0]["selected_rows"]
    assert "selected_rows" not in summary.metrics_trace[0]
    assert trace["commit"] == {
        "status": "disabled_or_no_actions",
        "commit_enabled": False,
        "action_count": 2,
    }

    trace_sha256 = update_discrete_virtual_scheduler_trace_commit(
        summary.trace_path,
        {"status": "accepted", "accepted_action_count": 2},
    )
    updated_trace = json.loads(Path(summary.trace_path).read_text(encoding="utf-8"))
    assert updated_trace["commit"] == {
        "status": "accepted",
        "accepted_action_count": 2,
    }
    assert trace_sha256 == hashlib.sha256(
        Path(summary.trace_path).read_bytes()
    ).hexdigest()


def test_discrete_runner_backtracks_to_largest_improving_prefix(tmp_path):
    lane = _BacktrackingDiscreteLane(tmp_path)

    summary = run_discrete_virtual_buffer_scheduler(
        lane=lane,
        model=object(),
        rounds=1,
    )

    trace = json.loads(Path(summary.trace_path).read_text(encoding="utf-8"))
    round_zero = trace["rounds"][0]
    assert round_zero["gradient_selected_action_count"] == 2
    assert round_zero["accepted_action_count"] == 1
    assert [row["selected_action_count"] for row in round_zero["trial_prefixes"]] == [2, 1]
    assert [row["accepted"] for row in round_zero["trial_prefixes"]] == [False, True]
    assert round_zero["actual_delta_loss"] == -2.0
    assert round_zero["total_virtual_buffer_count"] == 0
    assert round_zero["selected_rows"][0]["n_before"] == 0
    assert round_zero["selected_rows"][0]["n_after"] == 1
    assert lane.state.z_param[:2].tolist() == [1.0, 0.0]


def test_placement_commit_handoff_updates_route_b_trace_and_summary(tmp_path):
    lane = _FakeDiscreteLane(tmp_path)
    summary = run_discrete_virtual_buffer_scheduler(
        lane=lane,
        model=object(),
        rounds=1,
    )
    engine = object.__new__(NonLinearPlace)
    engine.params = SimpleNamespace(flow_kind="buffering")
    engine.joint_buffer_commit_trace = []
    engine.last_joint_flow_artifact = None
    engine.last_buffering_lane_summary = {
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

    trace = json.loads(Path(summary.trace_path).read_text(encoding="utf-8"))
    assert trace["commit"] == {"status": "accepted", "accepted_action_count": 1}
    assert written["payload"]["trace_sha256"] != summary.trace_sha256
    assert (
        written["payload"]["inner_loop"]["trace_sha256"]
        == written["payload"]["trace_sha256"]
    )


def test_discrete_runner_early_stops_without_positive_action_but_evaluates_terminal_state(
    tmp_path,
):
    lane = _NoBenefitDiscreteLane(tmp_path)

    summary = run_discrete_virtual_buffer_scheduler(
        lane=lane,
        model=object(),
        rounds=3,
    )

    assert summary.iterations == 1
    assert summary.terminal_reason == "no_positive_action"
    assert lane.objective_calls == 2
    assert lane.project_calls == 1
    assert lane.refresh_calls == 1
    trace = json.loads(Path(summary.trace_path).read_text(encoding="utf-8"))
    assert trace["rounds"][0]["accepted_action_count"] == 0
    assert trace["terminal"]["total_virtual_buffer_count"] == 0
