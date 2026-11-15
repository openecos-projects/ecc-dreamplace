import gc
import hashlib
import json
from types import SimpleNamespace

import pytest
import torch
from dreamplace.NonLinearPlace import NonLinearPlace
from dreamplace.ops.buffer_insertion.discrete_candidate_buffer_runner import (
    run_discrete_candidate_buffer_scheduler,
    update_discrete_candidate_scheduler_trace_commit,
)
from dreamplace.ops.buffer_insertion.drive_type_table import (
    EXPECTED_DRIVE_TYPE_MASTERS,
    write_case_drive_type_table,
)
from dreamplace.ops.buffer_insertion.optimization_state import (
    build_discrete_candidate_state,
    build_typed_discrete_candidate_state,
)
from dreamplace.ops.buffer_insertion.projection import project_candidate_buffer_state
from torch.utils import deterministic


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


class _FakeTypedCandidateLane:
    def __init__(self, output_dir, *, candidates=None):
        self.config = SimpleNamespace(
            mode="candidate",
            candidate_strategy="discrete_net_gradient",
            active_strategy="discrete_net_gradient",
            candidate_drive_type_mode="four_strength_table",
            max_selected_actions=None,
            segment_capacity_enabled=False,
            output_dir=str(output_dir),
            commit_enabled=False,
        )
        if candidates is None:
            candidates = [
                _candidate(11, 5, 2, 10),
                _candidate(22, 5, 4, 30),
            ]
        library = {
            index: {
                "cell_id": 100 + index,
                "master_name": name,
                "input_cap": 0.1 + index,
                "delay": 1.0 + index,
                "output_slew": 0.5 + index,
            }
            for index, name in enumerate(EXPECTED_DRIVE_TYPE_MASTERS)
        }
        table_path = output_dir / "drive_strength.json"
        table = write_case_drive_type_table(table_path, buffer_library=library)
        self.state = build_typed_discrete_candidate_state(
            candidates,
            buffer_main_type_index=4,
            legal_buffer_count=4,
            drive_type_table=table,
        )
        self.objective_calls = 0
        self.project_calls = 0
        self.refresh_calls = 0

    def _current_state(self):
        return self.state

    def objective(self, _model):
        self.objective_calls += 1
        # bsu_index is constant with respect to activation during a typed
        # probe, so this gives each type a distinct candidate gradient while
        # exercising the same state/objective boundary as production.
        weights = self.state.bsu_index().to(dtype=self.state.activation_param.dtype) + 1.0
        loss = -(self.state.candidate_bu() * weights).sum()
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
            "candidate_drive_type_mode": "four_strength_table",
            "net_subgraph_forward_backend": "unit",
        }


def test_candidate_runner_restores_the_gp_timing_scope(tmp_path, monkeypatch):
    lane = _FakeCandidateLane(tmp_path)
    original = lane.objective
    observed = []
    timing = SimpleNamespace(use_cell_aat_static_cache=False, _cell_aat_static_cache={})
    model = SimpleNamespace(op_collections=SimpleNamespace(timing_propagation_op=timing))
    monkeypatch.setattr(deterministic, "fill_uninitialized_memory", True)

    def objective(model):
        observed.append((deterministic.fill_uninitialized_memory, timing.use_cell_aat_static_cache))
        return original(model)

    lane.objective = objective
    run_discrete_candidate_buffer_scheduler(lane=lane, model=model, rounds=1)
    assert observed and set(observed) == {(False, True)}
    assert (deterministic.fill_uninitialized_memory, timing.use_cell_aat_static_cache) == (
        True,
        False,
    )


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
    # The terminal state is already best, so its objective is reused.
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


def test_candidate_runner_can_retain_existing_virtual_activation(tmp_path):
    lane = _FakeCandidateLane(tmp_path)
    with torch.no_grad():
        lane.state.activation_param[0] = 1.0

    summary = run_discrete_candidate_buffer_scheduler(
        lane=lane,
        model=SimpleNamespace(),
        rounds=1,
        allow_existing_activation=True,
    )

    assert summary.final_projection.projected_buffer_count == 2
    assert lane.state.activation_param.tolist() == [1.0, 1.0]


def test_candidate_terminal_metrics_follow_restored_state_after_overshoot(tmp_path):
    lane = _FakeCandidateLane(tmp_path)

    def objective(_model):
        count = lane.state.activation_param.sum()
        # The initial gradient selects a buffer, but its discrete cost is worse.
        loss = count.square() - 0.2 * count
        return loss, {"loss": loss.detach(), "tns": -loss, "wns": -loss}

    lane.objective = objective
    summary = run_discrete_candidate_buffer_scheduler(
        lane=lane,
        model=SimpleNamespace(),
        rounds=1,
    )

    assert summary.best_restore["best_round"] == 0
    assert lane.state.activation_param.tolist() == [0.0, 0.0]
    assert summary.final_projection.projected_buffer_count == 0
    assert summary.terminal_metrics["loss"] == summary.best_restore["best_score"] == 0.0
    payload = json.loads((tmp_path / "candidate_discrete_scheduler_trace.json").read_text())
    assert payload["terminal"]["loss"] == 0.0
    assert payload["rounds"][0]["actual_delta_loss"] == pytest.approx(0.8)


def test_candidate_restore_uses_current_loss_instead_of_cached_objective_terms(tmp_path):
    lane = _FakeCandidateLane(tmp_path)
    model = SimpleNamespace(last_timing_objective_terms={"total_loss": -100.0})

    def objective(_model):
        count = lane.state.activation_param.sum()
        loss = count.square() - 0.2 * count
        # TNS alone improves after insertion, while the combined loss worsens.
        return loss, {"loss": loss.detach(), "tns": -10.0 + count, "wns": count}

    lane.objective = objective
    summary = run_discrete_candidate_buffer_scheduler(
        lane=lane,
        model=model,
        rounds=1,
    )

    assert summary.best_restore["best_round"] == 0
    assert summary.best_restore["best_score"] == 0.0
    assert summary.final_projection.projected_buffer_count == 0
    assert summary.terminal_metrics["restore_score"] == 0.0


def test_typed_candidate_runner_probes_four_types_and_projects_master(tmp_path):
    lane = _FakeTypedCandidateLane(tmp_path)

    summary = run_discrete_candidate_buffer_scheduler(
        lane=lane,
        model=SimpleNamespace(),
        rounds=2,
    )

    assert summary.strategy == "discrete_net_gradient"
    assert summary.iterations == 2
    # Four typed passes per round, plus terminal and post-restore forwards.
    assert lane.objective_calls == 9
    assert lane.project_calls == 1
    assert lane.refresh_calls == 1
    assert lane.state.activation_param.sum(dim=1).tolist() == [1.0, 1.0]
    assert lane.state.timing_active_only is True
    selected_types = lane.state.selected_drive_type().tolist()
    assert selected_types == [3, 3]
    assert summary.final_projection.validation["exact_active_action_match"]
    actions = summary.final_projection.selected_actions
    assert [action["drive_type_id"] for action in actions] == [3, 3]
    assert [action["drive_type_master_name"] for action in actions] == [
        "BUFX20H7L",
        "BUFX20H7L",
    ]
    payload = json.loads((tmp_path / "candidate_discrete_scheduler_trace.json").read_text())
    assert payload["candidate_drive_type_mode"] == "four_strength_table"
    assert payload["drive_type_count"] == 4
    assert payload["snapshot_policy"]["mode"] == "frozen_per_round"
    assert payload["snapshot_policy"]["typed_evaluation_order"] == [0, 1, 2, 3]
    assert payload["snapshot_policy"]["refresh_between_types"] is False
    assert payload["rounds"][0]["typed_gradient_metadata"]["gradient_pass_count"] == 4
    assert len(payload["rounds"][0]["snapshot_id"]) == 64
    assert payload["rounds"][0]["drive_type_order"] == [0, 1, 2, 3]
    assert (
        payload["rounds"][0]["typed_gradient_metadata"]["snapshot_id"]
        == (payload["rounds"][0]["snapshot_id"])
    )


def test_typed_candidate_runner_forwards_selection_fraction(tmp_path):
    lane = _FakeTypedCandidateLane(
        tmp_path,
        candidates=[_candidate(i, i, i, i) for i in range(1001)],
    )
    lane.config.route_b_selection_fraction = 0.005

    summary = run_discrete_candidate_buffer_scheduler(
        lane=lane,
        model=SimpleNamespace(),
        rounds=1,
    )

    assert summary.final_projection.projected_buffer_count == 6
    expected = torch.zeros_like(lane.state.activation_param)
    expected[:6, 3] = 1.0
    torch.testing.assert_close(lane.state.activation_param, expected)
    payload = json.loads((tmp_path / "candidate_discrete_scheduler_trace.json").read_text())
    assert payload["selection_fraction"] == 0.005
    assert payload["affected_nets_per_action"] == 200
    assert payload["rounds"][0]["prefix_size"] == 6


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
        assert lane.buffer_relaxed_timing_payload["metadata"]["candidate_python_state_released"]
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
    engine._write_buffering_inner_loop_artifact = lambda _params, payload: written.update(
        payload=payload
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
    assert written["payload"]["trace_sha256"] == hashlib.sha256(trace_path.read_bytes()).hexdigest()
    assert written["payload"]["inner_loop"]["trace_sha256"] == written["payload"]["trace_sha256"]


def test_candidate_runner_tiered_retry_margin_and_round_early_break(tmp_path):
    """In-round tier retry consumes leftover round capacity, then early-breaks on 0-accept."""
    lane = _FakeCandidateLane(tmp_path)
    lane.state = build_discrete_candidate_state(
        [_candidate(11, 5, 2, 10), _candidate(12, 5, 3, 20), _candidate(22, 7, 4, 30)],
        buffer_main_type_index=4,
        legal_buffer_count=3,
        fixed_bsu_index=1,
    )

    class _FakeGate:
        def __init__(self):
            self.blocked_cids = {11}
            self.last_area = 100.0
            self.calls = 0

        def check_available(self, preexisting_virtual_count=0):
            return None

        def current_area(self):
            return self.last_area

        def allowed_untyped_candidates(self, state):
            return torch.tensor(
                [int(cid) not in self.blocked_cids for cid in state.candidate_ids.tolist()],
                dtype=torch.bool,
            )

        def filter_candidate_transition(self, state, transition, *, capacity_override=None):
            self.calls += 1
            rows = transition.selected_candidate_indices.detach().cpu().tolist()
            keep = []
            infos = []
            dirty = []
            for spec_i, row in enumerate(rows):
                cid = int(state.candidate_ids[row].item())
                if cid == 22:
                    self.blocked_cids.add(cid)  # simulate STA rejection on net 7
                elif cid == 12:
                    keep.append(spec_i)
                    self.last_area += 5.0
                    infos.append(
                        {
                            "row_idx": int(row),
                            "candidate_id": cid,
                            "orig_net_id": int(state.candidate_net_id[row].item()),
                            "active_net": 5,
                            "new_net": 99,
                        }
                    )
                    dirty.extend([5, 99])
            keep_t = torch.tensor(keep, dtype=torch.long)
            next_acts = state.activation_param.detach().clone()
            if keep:
                sel_rows = transition.selected_candidate_indices.index_select(0, keep_t)
                next_acts[sel_rows] = 1.0
            from dreamplace.ops.buffer_insertion.discrete_candidate_scheduler import (
                DiscreteCandidateTransition,
            )

            sel_rows = transition.selected_candidate_indices.index_select(0, keep_t)
            filtered = DiscreteCandidateTransition(
                next_activations=next_acts,
                selected_candidate_indices=sel_rows,
                selected_net_indices=transition.selected_net_indices.index_select(0, keep_t),
                selected_candidate_ids=transition.selected_candidate_ids.index_select(0, keep_t),
                selected_raw_grads=transition.selected_raw_grads.index_select(0, keep_t),
                selected_predicted_improvements=transition.selected_predicted_improvements.index_select(
                    0, keep_t
                ),
                affected_net_count=transition.affected_net_count,
                eligible_net_count=transition.eligible_net_count,
                prefix_size=transition.prefix_size,
                positive_prefix_count=transition.positive_prefix_count,
            )
            return filtered, {
                "status": "ok",
                "input_count": len(rows),
                "evaluated_count": len(rows),
                "accepted_count": len(keep),
                "rejected_area_count": 0,
                "rejected_loss_count": len(rows) - len(keep),
                "accepted_action_infos": infos,
                "dirty_nets": sorted(set(dirty)),
            }

    gate = _FakeGate()
    refreshes = []
    # Tier 1 proposes net 5 (cid 12 -> accepted!) and net 7 (cid 22 -> rejected!).
    # After round 0 refreshes dirty=[5, 99], round 1 proposes net 5 again (cid 12 inactive,
    # Wait: suppose no new improving candidates left -> round 1 has accepted=0 -> early break!).
    summary = run_discrete_candidate_buffer_scheduler(
        lane=lane,
        model=SimpleNamespace(),
        rounds=3,
        incremental_gate=gate,
        selection_fraction=0.5,  # ceil(2 * 0.5) = 1 -> triggers tier retry on net 7 rejection!
        round_budget_fn=lambda **kw: {
            "exhausted": False,
            "selection_fraction": 0.5,
            "round_capacity": 120.0,
            "min_buffer_area": 5.0,
        },
        on_round_accepted=lambda infos, dirty: refreshes.append((infos, dirty)) or {"status": "ok"},
    )
    assert len(refreshes) == 1
    assert refreshes[0][1] == [5, 99]
    assert int(lane.state.activation_param.sum().item()) == 1
    # Round 0 accepted 1; Round 1 had accepted=0 -> broke before Round 2.
    assert summary.iterations == 2


def test_candidate_runner_reanchors_best_activation_after_refresh_when_gate_degrades(
    tmp_path,
):
    """Refreshing candidate rows in Round 0 must re-anchor best_activation if Round 1 gate skips."""
    lane = _FakeCandidateLane(tmp_path)
    lane.state = build_discrete_candidate_state(
        [_candidate(11, 5, 2, 30), _candidate(22, 7, 4, 10)],
        buffer_main_type_index=4,
        legal_buffer_count=3,
        fixed_bsu_index=1,
    )

    class _DegradingGate:
        def __init__(self):
            self.round_calls = 0
            self.last_area = 100.0

        def check_available(self, preexisting_virtual_count=0):
            return None

        def current_area(self):
            return self.last_area

        def allowed_untyped_candidates(self, state):
            return None

        def filter_candidate_transition(
            self, state, transition, *, capacity_override=None
        ):
            self.round_calls += 1
            if self.round_calls == 1:
                row = int(transition.selected_candidate_indices[0].item())
                return transition, {
                    "status": "ok",
                    "input_count": 1,
                    "evaluated_count": 1,
                    "accepted_count": int(transition.accepted_count),
                    "rejected_area_count": 0,
                    "rejected_loss_count": 0,
                    "accepted_action_infos": [
                        {
                            "row_idx": row,
                            "candidate_id": int(state.candidate_ids[row].item()),
                            "orig_net_id": int(state.candidate_net_id[row].item()),
                            "active_net": 5,
                            "new_net": 99,
                        }
                    ],
                    "dirty_nets": [5, 99],
                }
            return transition, {"status": "skipped", "reason": "engine_unavailable"}

    def _on_round_accepted(infos, dirty):
        # Expand state from 2 -> 4 rows (locked buffer + refreshed sub-net candidates).
        rebuilt = build_discrete_candidate_state(
            [
                _candidate(11, 5, 2, 0),
                _candidate(31, 5, 6, -5),
                _candidate(32, 5, 8, -5),
                _candidate(22, 7, 4, -5),
            ],
            buffer_main_type_index=4,
            legal_buffer_count=3,
            fixed_bsu_index=1,
        )
        with torch.no_grad():
            rebuilt.activation_param[0] = 1.0
        lane.state = rebuilt
        return {"status": "ok", "candidate_count": 4}

    # Non-convex objective: 1 active buffer improves (-1), but 2 active buffers regress (+5)
    # despite negative local gradient at [1, 0, 0, 0], forcing terminal rollback to best_activation!
    def _nonconvex_objective(_model):
        lane.objective_calls += 1
        act = lane.state.activation_param
        s = act.sum()
        loss = -s + 3.0 * torch.clamp(s - 1.0, min=0.0) ** 2 + (
            4.0 * float((s.detach() > 1.5).item())
        )
        return loss, {"loss": loss.detach(), "tns": -loss.detach(), "wns": -loss.detach()}

    lane.objective = _nonconvex_objective

    summary = run_discrete_candidate_buffer_scheduler(
        lane=lane,
        model=SimpleNamespace(),
        rounds=2,
        incremental_gate=_DegradingGate(),
        selection_fraction=0.5,
        on_round_accepted=_on_round_accepted,
    )
    assert tuple(lane.state.activation_param.shape) == (4,)
    assert lane.state.activation_param.detach().tolist() == [1.0, 0.0, 0.0, 0.0]
    assert summary.final_projection.projected_buffer_count == 1
    assert summary.iterations == 2


def test_candidate_runner_preserves_multi_tier_metrics_and_unlocks_area_capped_in_next_round(
    tmp_path,
):
    """Multi-tier report keeps Tier-1 baseline/timings; area-capped actions unlock next round."""
    lane = _FakeCandidateLane(tmp_path)
    lane.state = build_discrete_candidate_state(
        [
            _candidate(11, 5, 2, 10),
            _candidate(22, 7, 4, 20),
            _candidate(33, 9, 6, 30),
        ],
        buffer_main_type_index=4,
        legal_buffer_count=3,
        fixed_bsu_index=1,
    )

    def _ranked_objective(_model):
        lane.objective_calls += 1
        act = lane.state.activation_param
        # Distinct gradients: cid 11 (-3.0), cid 22 (-2.0), cid 33 (-1.0)
        weights = torch.tensor([3.0, 2.0, 1.0], dtype=act.dtype)
        loss = -(act * weights).sum()
        return loss, {"loss": loss.detach(), "tns": -loss.detach(), "wns": -loss.detach()}

    lane.objective = _ranked_objective

    class _AreaAndTierGate:
        def __init__(self):
            self.sta_blocked = set()
            self.last_area = 100.0
            self.last_loss = 0.0
            self.proposed_cids_by_call = []

        def check_available(self, preexisting_virtual_count=0):
            return None

        def current_area(self):
            return self.last_area

        def allowed_untyped_candidates(self, state):
            return torch.tensor(
                [int(cid) not in self.sta_blocked for cid in state.candidate_ids.tolist()],
                dtype=torch.bool,
            )

        def filter_candidate_transition(self, state, transition, *, capacity_override=None):
            from dreamplace.ops.buffer_insertion.discrete_candidate_scheduler import (
                DiscreteCandidateTransition,
            )

            call_idx = len(self.proposed_cids_by_call)
            rows = transition.selected_candidate_indices.detach().cpu().tolist()
            cids = [int(state.candidate_ids[r].item()) for r in rows]
            self.proposed_cids_by_call.append(cids)

            area_cost = {11: 8.0, 22: 12.0, 33: 5.0}
            init_area = self.last_area
            init_loss = self.last_loss
            keep = []
            rej_area = []
            rej_loss = 0
            infos = []
            actions = []
            dirty = []
            for spec_i, (row, cid) in enumerate(zip(rows, cids, strict=True)):
                proposed = self.last_area + area_cost[cid]
                if capacity_override is not None and proposed > float(capacity_override):
                    rej_area.append(spec_i)
                    continue
                keep.append(spec_i)
                self.last_area = proposed
                net_id = int(state.candidate_net_id[row].item())
                infos.append(
                    {
                        "row_idx": int(row),
                        "candidate_id": cid,
                        "orig_net_id": net_id,
                        "active_net": net_id,
                        "new_net": 100 + net_id,
                    }
                )
                actions.append({"spec_index": spec_i, "active_net": net_id})
                dirty.extend([net_id, 100 + net_id])

            keep_t = torch.tensor(keep, dtype=torch.long)
            next_acts = state.activation_param.detach().clone()
            sel_rows = transition.selected_candidate_indices.index_select(0, keep_t)
            if keep:
                next_acts[sel_rows] = 1.0
            filtered = DiscreteCandidateTransition(
                next_activations=next_acts,
                selected_candidate_indices=sel_rows,
                selected_net_indices=transition.selected_net_indices.index_select(0, keep_t),
                selected_candidate_ids=transition.selected_candidate_ids.index_select(0, keep_t),
                selected_raw_grads=transition.selected_raw_grads.index_select(0, keep_t),
                selected_predicted_improvements=transition.selected_predicted_improvements.index_select(
                    0, keep_t
                ),
                affected_net_count=transition.affected_net_count,
                eligible_net_count=transition.eligible_net_count,
                prefix_size=transition.prefix_size,
                positive_prefix_count=transition.positive_prefix_count,
            )
            self.last_loss = init_loss - 5.0 * len(keep)
            final_loss = self.last_loss
            is_cold_build = call_idx == 0
            return filtered, {
                "status": "ok",
                "cache_hit": not is_cold_build,
                "engine_build_ms": 42.0 if is_cold_build else 0.0,
                "elapsed_ms": 3.0 if is_cold_build else 2.0,
                "total_filter_ms": 45.0 if is_cold_build else 2.5,
                "input_count": len(rows),
                "evaluated_count": len(rows) - len(rej_area),
                "accepted_count": len(keep),
                "rejected_area_count": len(rej_area),
                "rejected_area_indices": rej_area,
                "rejected_loss_count": rej_loss,
                "initial_loss": init_loss,
                "final_loss": final_loss,
                "initial_wns_ps": -200.0 + init_loss,
                "final_wns_ps": -200.0 + final_loss,
                "initial_tns_ps": -1000.0 + init_loss,
                "final_tns_ps": -1000.0 + final_loss,
                "initial_area": init_area,
                "final_area": self.last_area,
                "accepted_indices": keep,
                "accepted_actions": actions,
                "accepted_action_infos": infos,
                "dirty_nets": sorted(set(dirty)),
            }

    gate = _AreaAndTierGate()
    round_caps = {0: 115.0, 1: 130.0}
    summary = run_discrete_candidate_buffer_scheduler(
        lane=lane,
        model=SimpleNamespace(),
        rounds=2,
        incremental_gate=gate,
        selection_fraction=0.6,  # ceil(3 * 0.6) = 2 per tier
        round_budget_fn=lambda round_idx, **_kw: {
            "exhausted": False,
            "selection_fraction": 0.6,
            "round_capacity": round_caps[round_idx],
            "min_buffer_area": 5.0,
        },
    )

    # Round 0 (cap=115):
    # - Tier 1 proposes [11 (+8 -> 108, accepted), 22 (+12 -> 120 > 115, area-capped)].
    # - Remaining cap 7 >= min_buf_area 5 -> Tier 2 skips blocked 22 and accepts [33 (+5 -> 113)].
    # Round 1 (cap=130):
    # - Area-capped 22 was NOT permanently blacklisted -> proposed & accepted (+12 -> 125)!
    assert gate.proposed_cids_by_call == [[11, 22], [33], [22]]
    assert lane.state.activation_param.detach().tolist() == [1.0, 1.0, 1.0]
    assert summary.final_projection.projected_buffer_count == 3

    r0_commit = summary.metrics_trace[0]["incremental_commit"]
    assert r0_commit["tier_attempts"] == 2
    assert r0_commit["cache_hit"] is False
    assert r0_commit["engine_build_ms"] == pytest.approx(42.0)
    assert r0_commit["elapsed_ms"] == pytest.approx(5.0)
    assert r0_commit["total_filter_ms"] == pytest.approx(47.5)
    assert r0_commit["initial_loss"] == pytest.approx(0.0)
    assert r0_commit["final_loss"] == pytest.approx(-10.0)
    assert r0_commit["initial_area"] == pytest.approx(100.0)
    assert r0_commit["final_area"] == pytest.approx(113.0)
    assert r0_commit["input_count"] == 3
    assert r0_commit["evaluated_count"] == 2
    assert r0_commit["accepted_count"] == 2
    assert r0_commit["rejected_area_count"] == 1
    assert r0_commit["accepted_indices"] == [0, 2]
    assert r0_commit["rejected_area_indices"] == [1]
    assert [a["spec_index"] for a in r0_commit["accepted_actions"]] == [0, 2]
    assert [info["candidate_id"] for info in r0_commit["accepted_action_infos"]] == [11, 33]
    assert r0_commit["dirty_nets"] == [5, 9, 105, 109]


