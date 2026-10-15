from types import SimpleNamespace
from pathlib import Path

import pytest
import torch


class _FakeOptimizer:
    def __init__(self, trace):
        self.trace = trace

    def zero_grad(self, set_to_none=True):
        self.trace.append(("zero_grad", set_to_none))

    def step(self):
        self.trace.append("optimizer.step")


class _FakeLane:
    def __init__(self):
        self.trace = []
        self.value = torch.nn.Parameter(torch.tensor(2.0))
        self.config = SimpleNamespace(
            segment_integer_projection_interval=0,
            segment_integer_projection_start_step=0,
            segment_integer_projection_project_bsu=False,
            segment_integer_projection_reset_optimizer_state=True,
        )
        self._summary = {}

    def objective(self, model):
        self.trace.append("objective")
        return self.value * self.value, {"source": getattr(model, "name", "unit")}

    def before_step(self, iteration, loss, metrics):
        self.trace.append(("before_step", iteration, metrics["source"]))

    def after_step(self, iteration, loss, metrics):
        self.trace.append(("after_step", iteration, metrics["source"]))

    def should_project(self, iteration):
        return True

    def project(self, **kwargs):
        self.trace.append(("project", kwargs))
        return SimpleNamespace(projected_buffer_count=1, affected_nets=(3,))

    def refresh_runtime_state(self, projection_result):
        self.trace.append(("refresh_runtime_state", projection_result.affected_nets))
        return {"status": "ok"}

    def project_segment_count_optimizer_state_to_integer(
        self,
        *,
        project_bsu,
        reset_optimizer_state,
        optimizer,
        iteration,
    ):
        self.trace.append(
            (
                "integer_project",
                iteration,
                bool(project_bsu),
                bool(reset_optimizer_state),
                optimizer is not None,
            )
        )
        return {
            "status": "ok",
            "z_nonzero_count_after": 1,
            "z_sum_after": 1.0,
            "z_max_after": 1.0,
        }

def test_inner_loop_runs_objective_backward_step_projection_and_refresh_in_order():
    from dreamplace.ops.buffer_insertion.buffering_inner_loop import (
        run_buffering_inner_loop,
    )

    lane = _FakeLane()
    optimizer = _FakeOptimizer(lane.trace)

    summary = run_buffering_inner_loop(
        lane=lane,
        model=SimpleNamespace(name="model"),
        optimizer=optimizer,
        steps=1,
    )

    assert lane.value.grad is not None
    assert lane.value.grad.item() == pytest.approx(4.0)
    assert lane.trace == [
        ("zero_grad", True),
        "objective",
        ("before_step", 0, "model"),
        "optimizer.step",
        ("after_step", 0, "model"),
        ("project", {"iteration": 0, "final": False}),
        ("refresh_runtime_state", (3,)),
        ("project", {"iteration": 1, "final": True}),
        ("refresh_runtime_state", (3,)),
    ]
    assert summary.final_projection.projected_buffer_count == 1
    assert summary.iterations == 1


def test_inner_loop_summary_records_detached_scalar_metrics_trace():
    from dreamplace.ops.buffer_insertion.buffering_inner_loop import (
        run_buffering_inner_loop,
    )

    lane = _FakeLane()
    optimizer = _FakeOptimizer(lane.trace)

    summary = run_buffering_inner_loop(
        lane=lane,
        model=SimpleNamespace(name="model"),
        optimizer=optimizer,
        steps=1,
    )

    assert len(summary.metrics_trace) == 1
    assert summary.metrics_trace[0]["source"] == "model"
    assert summary.metrics_trace[0]["loss"] == pytest.approx(4.0)
    assert summary.metrics_trace[0]["step_wall_ms"] >= 0.0
    assert summary.metrics_trace[0]["objective_step_wall_ms"] >= 0.0
    assert summary.metrics_trace[0]["projection_refresh_wall_ms"] >= 0.0


def test_inner_loop_respects_project_interval_and_still_runs_final_projection():
    from dreamplace.ops.buffer_insertion.buffering_inner_loop import (
        run_buffering_inner_loop,
    )

    lane = _FakeLane()
    optimizer = _FakeOptimizer(lane.trace)

    run_buffering_inner_loop(
        lane=lane,
        model=SimpleNamespace(name="model"),
        optimizer=optimizer,
        steps=3,
        project_interval=2,
    )

    project_calls = [
        item for item in lane.trace if isinstance(item, tuple) and item[0] == "project"
    ]

    assert project_calls == [
        ("project", {"iteration": 1, "final": False}),
        ("project", {"iteration": 3, "final": True}),
    ]


def test_inner_loop_runs_periodic_integer_projection_after_optimizer_step():
    from dreamplace.ops.buffer_insertion.buffering_inner_loop import (
        run_buffering_inner_loop,
    )

    lane = _FakeLane()
    lane.config.segment_integer_projection_interval = 2
    lane.config.segment_integer_projection_start_step = 2
    lane.config.segment_integer_projection_project_bsu = True
    optimizer = _FakeOptimizer(lane.trace)

    summary = run_buffering_inner_loop(
        lane=lane,
        model=SimpleNamespace(name="model"),
        optimizer=optimizer,
        steps=3,
        project_interval=0,
    )

    step_index = lane.trace.index("optimizer.step", lane.trace.index(("zero_grad", True)) + 1)
    after_step_index = lane.trace.index(("after_step", 1, "model"))
    integer_project_index = lane.trace.index(("integer_project", 1, True, True, True))
    assert step_index < after_step_index < integer_project_index
    integer_calls = [
        item
        for item in lane.trace
        if isinstance(item, tuple) and item[0] == "integer_project"
    ]
    assert integer_calls == [("integer_project", 1, True, True, True)]
    assert len(summary.periodic_integer_projection_trace) == 1
    assert summary.metrics_trace[1]["periodic_integer_projection_status"] == "ok"
    assert summary.metrics_trace[1]["periodic_integer_projection_z_max_after"] == 1.0


def test_forced_segment_count_is_clamped_to_state_range(monkeypatch):
    from dreamplace.ops.buffer_insertion.buffering_inner_loop import (
        _forced_z_value,
        _set_forced_z_selection,
    )

    state = SimpleNamespace(
        max_repeater_count=3,
        z_param=torch.nn.Parameter(torch.zeros(3)),
        project_=lambda: None,
    )
    selected = torch.tensor([1], dtype=torch.long)

    monkeypatch.setenv("AIMP_BUFFERING_SEGMENT_FORCE_Z_VALUE", "5")
    value = _forced_z_value(state)
    _set_forced_z_selection(state, selected, value=value)

    assert value == 3.0
    assert state.z_param.detach().tolist() == [0.0, 3.0, 0.0]


def test_forced_z_probe_only_differentiates_segment_state(monkeypatch):
    from dreamplace.ops.buffer_insertion.buffering_inner_loop import (
        _segment_forced_z_probe,
    )

    static_leaf = torch.tensor(2.0, requires_grad=True)
    reused_static_graph = static_leaf.square()
    z_param = torch.nn.Parameter(torch.zeros(1))
    bsu_param = torch.nn.Parameter(torch.zeros(1))
    state = SimpleNamespace(
        max_repeater_count=3,
        z_param=z_param,
        bsu_index_param=bsu_param,
        segment_rows=({"segment_id": 7, "net_id": 3, "parent_node_id": 11},),
        project_=lambda: None,
    )

    class ProbeLane:
        def objective(self, model):
            loss = reused_static_graph - z_param.sum()
            return loss, {"tns": -loss, "wns": -loss}

    lane = ProbeLane()
    initial_loss, _ = lane.objective(SimpleNamespace())
    initial_loss.backward()
    saved_grad = z_param.grad.detach().clone()
    monkeypatch.setenv("AIMP_BUFFERING_SEGMENT_FORCE_Z_TOPKS", "1")

    result = _segment_forced_z_probe(state, lane, SimpleNamespace())

    assert result["status"] == "ok"
    assert result["negative_grad_segment_count"] == 1
    assert result["forced_results"][0]["selected_segments"][0]["segment_id"] == 7
    torch.testing.assert_close(z_param.grad, saved_grad)


def test_inner_loop_module_has_no_physical_commit_or_sta_calls():
    source = Path("dreamplace/ops/buffer_insertion/buffering_inner_loop.py").read_text(
        encoding="utf-8"
    )

    assert "OpenROAD" not in source
    assert "OpenSTA" not in source
    assert "commit_final_projection_if_enabled" not in source
    assert "commit_if_enabled" not in source
