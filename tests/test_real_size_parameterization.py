
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dreamplace import BasicPlace
from dreamplace.NonLinearPlace import NonLinearPlace
from dreamplace import placer_cli
from dreamplace.ops.gate_projection.gate_projection import (
    MainIdCandidateProvider,
    ProjectionFrame,
    VectorizedMainIdCandidateProvider,
)


def _real_size_collection():
    data = BasicPlace.PlaceDataCollection.__new__(BasicPlace.PlaceDataCollection)
    data.sizing_parameterization = "real_size"
    data.size_logits = None
    data.real_size = torch.nn.Parameter(torch.tensor([0.5, 2.0, 3.0]))
    data.inst_size_init = torch.tensor([0.5, 2.0, 3.0])
    data.inst_size_lower = torch.tensor([0.0, 1.0, 3.0])
    data.inst_size_upper = torch.tensor([1.0, 4.0, 3.0])
    data.continuous_size_trainable_mask = torch.tensor([True, False, False])
    return data


def test_effective_initial_size_matches_logit_coordinate():
    initial = np.array([0.0, 2.0, 4.0], dtype=np.float64)
    lower = np.array([0.0, 1.0, 4.0], dtype=np.float64)
    upper = np.array([1.0, 3.0, 4.0], dtype=np.float64)

    effective = BasicPlace.compute_effective_initial_size(
        initial,
        lower,
        upper,
        dtype=np.float64,
    )

    assert effective[0] > lower[0]
    assert effective[0] < upper[0]
    assert effective[1] == initial[1]
    assert effective[2] == initial[2]


def test_real_size_is_canonical_owner_and_has_direct_gradient():
    data = _real_size_collection()
    value = data.get_size_var()
    objective = (value * value).sum()
    objective.backward()

    assert value is data.real_size
    assert data.get_continuous_size_parameter() is data.real_size
    assert torch.equal(data.get_continuous_size_gradient(), 2.0 * data.real_size.detach())
    assert data.size_logits is None


def test_real_size_projection_clamps_and_preserves_fixed_instances():
    data = _real_size_collection()
    with torch.no_grad():
        data.real_size.copy_(torch.tensor([-2.0, 8.0, 9.0]))

    summary = data.project_continuous_size()

    assert torch.equal(data.real_size.detach(), torch.tensor([0.0, 2.0, 3.0]))
    assert summary["lower_clamped_count"] == 1
    assert summary["upper_clamped_count"] == 1
    assert summary["fixed_instance_count"] == 2
    assert summary["max_projection_delta"] == 6.0


def test_real_size_snapshot_restore_keeps_owner_coordinate():
    data = _real_size_collection()
    snapshot = data.capture_continuous_size_state()
    with torch.no_grad():
        data.real_size.add_(1.0)

    assert data.restore_continuous_size_state(snapshot)
    assert torch.equal(data.real_size.detach(), torch.tensor([0.5, 2.0, 3.0]))
    assert torch.equal(snapshot["value"], data.real_size.detach())
    assert torch.equal(snapshot["parameter"], data.real_size.detach())


def test_logit_snapshot_compatibility_restores_parameter_not_size_value():
    data = BasicPlace.PlaceDataCollection.__new__(BasicPlace.PlaceDataCollection)
    data.sizing_parameterization = "logits"
    data.real_size = None
    data.inst_size_lower = torch.tensor([0.0, 0.0])
    data.inst_size_upper = torch.tensor([1.0, 2.0])
    data.size_logits = torch.nn.Parameter(torch.tensor([0.0, 0.0]))

    snapshot = data.capture_continuous_size_state()
    with torch.no_grad():
        data.size_logits.add_(1.0)

    assert data.restore_continuous_size_state(snapshot)
    assert torch.allclose(data.size_logits.detach(), torch.tensor([0.0, 0.0]))


def test_cli_exposes_real_size_configuration(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text('{"design_name": "unit"}', encoding="utf-8")
    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "sizing",
            "--sizing-parameterization",
            "real_size",
            "--continuous-size-dynamics-mode",
            "none",
            "--real-size-learning-rate",
            "0.2",
            "--real-size-execution-mode",
            "warmup_to_discrete",
            "--real-size-warmup-steps",
            "3",
        ]
    )

    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.sizing_parameterization == "real_size"
    assert params.continuous_size_dynamics_mode == "none"
    assert params.real_size_learning_rate == 0.2
    assert params.real_size_execution_mode == "warmup_to_discrete"
    assert params.real_size_warmup_steps == 3
    assert params.real_size_transition_backend == "vectorized"


def test_real_size_transition_backend_switches_exact_provider_type():
    placer = NonLinearPlace.__new__(NonLinearPlace)
    torch.nn.Module.__init__(placer)
    projection_op = SimpleNamespace(candidate_provider=MainIdCandidateProvider())

    assert placer._configure_real_size_projection_backend(
        SimpleNamespace(real_size_transition_backend="vectorized"),
        projection_op,
    ) == "vectorized"
    assert type(projection_op.candidate_provider) is VectorizedMainIdCandidateProvider

    assert placer._configure_real_size_projection_backend(
        SimpleNamespace(real_size_transition_backend="reference"),
        projection_op,
    ) == "reference"
    assert type(projection_op.candidate_provider) is MainIdCandidateProvider

    with pytest.raises(RuntimeError, match="native.*not implemented"):
        placer._configure_real_size_projection_backend(
            SimpleNamespace(real_size_transition_backend="native"),
            projection_op,
        )


def test_real_size_transition_artifact_policy_and_compact_summary(tmp_path):
    placer = NonLinearPlace.__new__(NonLinearPlace)
    torch.nn.Module.__init__(placer)
    params = SimpleNamespace(
        real_size_transition_artifact_policy="summary_only",
        result_dir=str(tmp_path),
        design_name=lambda: "unit",
    )

    assert placer._projection_artifact_policy(
        params,
        stage_timing_summary={"artifact_scope": "real_size_warmup_transition"},
    ) == "summary_only"
    assert placer._projection_artifact_policy(params, stage_timing_summary={}) == "full"
    params.real_size_transition_artifact_policy = "full"
    assert placer._projection_artifact_policy(
        params,
        stage_timing_summary={"artifact_scope": "real_size_warmup_transition"},
    ) == "full"
    params.real_size_transition_artifact_policy = "invalid"
    with pytest.raises(ValueError, match="unsupported.*artifact_policy"):
        placer._projection_artifact_policy(
            params,
            stage_timing_summary={"artifact_scope": "real_size_warmup_transition"},
        )

    projection_result = object()
    placer.last_projection_frame = SimpleNamespace(
        projection_result=projection_result,
        state_digest="sha256:unit",
        changed_inst_ids=torch.tensor([2]),
        changed_pin_ids=torch.tensor([4, 5]),
        topology_generation_before=7,
        timing_model_generation_before=3,
        tensor_contract=(),
    )
    placer.op_collections = SimpleNamespace(
        gate_projection_op=SimpleNamespace(
            build_artifact_summary=lambda *_args, **_kwargs: {
                "validation": {"is_valid": True}
            }
        )
    )
    paths = placer._write_projection_summary_artifact(
        params,
        projection_result,
        metadata={"artifact_scope": "real_size_warmup_transition"},
    )

    assert set(paths) == {"summary"}
    assert paths["summary"].endswith(
        "unit_real_size_transition_projection_summary.json"
    )
    assert not list(tmp_path.glob("*.jsonl"))
    assert not list(tmp_path.glob("*.csv"))
    payload = json.loads(tmp_path.joinpath(
        "unit_real_size_transition_projection_summary.json"
    ).read_text(encoding="utf-8"))
    assert payload["state_digest"] == "sha256:unit"
    assert payload["num_changed_instances"] == 1
    assert payload["num_changed_pins"] == 2
    assert payload["topology_generation"] == 7
    assert payload["timing_model_generation"] == 3


def test_real_size_transition_profile_accounting_uses_exclusive_stages():
    profile = {
        "transition_total_ms": 10.0,
        "projection_pipeline_ms": 3.0,
        "transaction_capture_ms": 1.0,
        "runtime_refresh_total_ms": 2.0,
        "size_to_logits_ms": 1.0,
        "optimizer_state_reset_ms": 1.0,
        "transition_finalize_ms": 1.0,
        "cuda_synchronize_ms": 0.5,
    }

    NonLinearPlace._finalize_real_size_transition_accounting(profile)

    assert profile["accounted_transition_ms"] == pytest.approx(9.5)
    assert profile["unaccounted_transition_ms"] == pytest.approx(0.5)
    assert profile["accounted_transition_ratio"] == pytest.approx(0.95)


def test_real_size_rejects_explicit_discrete_dynamics_before_transition(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text('{"design_name": "unit"}', encoding="utf-8")
    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "sizing",
            "--sizing-parameterization",
            "real_size",
            "--continuous-size-dynamics-mode",
            "discrete_gradient_topk",
        ]
    )

    try:
        placer_cli.build_effective_params_from_args(args)
    except ValueError as error:
        assert "explicit warm-up transition" in str(error)
    else:
        raise AssertionError(
            "real_size must reject discrete dynamics before transition"
        )


def test_real_size_rejects_discrete_dynamics_declared_in_json(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        '{"design_name": "unit", '
        '"continuous_size_dynamics_mode": "discrete_gradient_topk"}',
        encoding="utf-8",
    )
    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "sizing",
            "--sizing-parameterization",
            "real_size",
        ]
    )

    try:
        placer_cli.build_effective_params_from_args(args)
    except ValueError as error:
        assert "explicit warm-up transition" in str(error)
    else:
        raise AssertionError(
            "real_size must reject JSON-declared discrete dynamics before transition"
        )


def test_optimizer_registers_only_the_selected_size_owner():
    data = _real_size_collection()
    data.vt_logits = torch.nn.Parameter(torch.zeros(3, 1))
    placer = NonLinearPlace.__new__(NonLinearPlace)
    torch.nn.Module.__init__(placer)
    placer.data_collections = data
    placer.size_params = torch.nn.ParameterList([data.real_size])
    placer.vt_params = torch.nn.ParameterList([data.vt_logits])
    placer.buffer_params = torch.nn.ParameterList()

    groups = placer._optimizer_parameters(
        SimpleNamespace(placement_sizing_mode="size_only")
    )

    assert len(groups) == 1
    assert groups[0]["group_name"] == "sizing"
    assert any(parameter is data.real_size for parameter in groups[0]["params"])
    assert any(parameter is data.vt_logits for parameter in groups[0]["params"])
    assert all(parameter is not data.size_logits for parameter in groups[0]["params"])


def test_real_size_projection_rejects_invalid_bounds():
    data = _real_size_collection()
    data.inst_size_lower = torch.tensor([2.0, 1.0, 3.0])
    data.inst_size_upper = torch.tensor([1.0, 4.0, 3.0])

    try:
        data.project_continuous_size()
    except ValueError as error:
        assert "greater than or equal" in str(error)
    else:
        raise AssertionError("invalid size bounds must be rejected")


def test_real_size_projection_rejects_nonfinite_owner():
    data = _real_size_collection()
    with torch.no_grad():
        data.real_size[0] = float("nan")

    try:
        data.project_continuous_size()
    except ValueError as error:
        assert "finite" in str(error)
    else:
        raise AssertionError("non-finite real_size must be rejected")


def test_real_size_configuration_requires_positive_warmup_steps(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text('{"design_name": "unit"}', encoding="utf-8")
    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "sizing",
            "--sizing-parameterization",
            "real_size",
            "--real-size-execution-mode",
            "warmup_to_discrete",
            "--real-size-warmup-steps",
            "0",
        ]
    )

    try:
        placer_cli.build_effective_params_from_args(args)
    except ValueError as error:
        assert "warmup_steps" in str(error)
    else:
        raise AssertionError("warmup_to_discrete must require a positive step count")


def test_warmup_transition_replaces_owner_and_resets_sizing_state(tmp_path):
    data = _real_size_collection()
    vt_logits = torch.nn.Parameter(torch.zeros(3, 2))
    data.vt_logits = vt_logits
    placer = NonLinearPlace.__new__(NonLinearPlace)
    torch.nn.Module.__init__(placer)
    placer.data_collections = data
    placer.size_params = torch.nn.ParameterList([data.real_size])
    placer.vt_params = torch.nn.ParameterList([vt_logits])
    placer.discrete_gradient_topk_candidate_cache = {"stale": True}
    placer.discrete_gradient_topk_candidate_cache_key = ("stale",)
    placer._real_size_transition_done = False
    placer.placedb = SimpleNamespace()

    projection_result = SimpleNamespace(
        inst_ids=torch.tensor([0]),
        validation=SimpleNamespace(is_valid=True, issues=[]),
    )
    frame = ProjectionFrame(
        projection_result=projection_result,
        changed_candidate_mask=torch.tensor([False]),
        changed_inst_ids=torch.zeros(0, dtype=torch.long),
        changed_node_mask=torch.zeros(3, dtype=torch.bool),
        changed_pin_ids=torch.zeros(0, dtype=torch.long),
        projected_pin_offset_x=torch.zeros(0),
        projected_pin_offset_y=torch.zeros(0),
        projected_node_size_x=None,
        projected_node_size_y=None,
        projected_node_area=None,
        projected_size_global=torch.tensor([1.0, 2.0, 3.0]),
        projected_vt_global=None,
        projected_cell_id_global=torch.zeros(3, dtype=torch.long),
        projected_libcell_offset_global=torch.zeros(3, dtype=torch.long),
        projected_inst_size_init=None,
        projected_original_pin_offset_x=None,
        projected_original_pin_offset_y=None,
        state_digest="sha256:test",
        topology_generation_before=0,
        topology_generation_after=None,
        timing_model_generation_before=0,
        timing_model_generation_after=None,
    )
    placer._write_projection_artifacts = (
        lambda params, stage_timing_summary: setattr(
            placer, "last_projection_result", projection_result
        )
        or setattr(placer, "last_projection_frame", frame)
        or {"summary": "/tmp/summary.json"}
    )

    def apply_projection(_params, _placedb):
        with torch.no_grad():
            data.real_size.copy_(torch.tensor([1.0, 2.0, 3.0]))
        return {"runtime_refresh_applied": True}

    placer._apply_projected_runtime_refresh = apply_projection
    optimizer = torch.optim.Adam(
        [
            {"params": [data.real_size], "group_name": "sizing"},
            {"params": [vt_logits], "group_name": "sizing"},
        ],
        lr=0.1,
    )
    optimizer.state[data.real_size] = {"step": torch.tensor(4.0)}
    optimizer.state[vt_logits] = {"step": torch.tensor(4.0)}
    params = SimpleNamespace(
        sizing_parameterization="real_size",
        real_size_execution_mode="warmup_to_discrete",
        real_size_warmup_steps=1,
        continuous_size_dynamics_mode="none",
        real_size_transition_profile=True,
        real_size_transition_backend="reference",
        real_size_transition_artifact_policy="summary_only",
        result_dir=str(tmp_path),
        design_name=lambda: "unit",
    )

    summary = placer._maybe_transition_real_size_to_discrete(
        params=params,
        optimizer=optimizer,
        iteration=0,
        model=None,
    )

    assert summary["transitioned"] is True
    assert data.sizing_parameterization == "logits"
    assert data.real_size is None
    assert data.size_logits is placer.size_params[0]
    assert torch.allclose(
        data.get_size_var(), torch.tensor([0.9999, 2.0, 3.0]), atol=1e-6
    )
    assert params.sizing_parameterization == "real_size"
    assert params.continuous_size_dynamics_mode == "discrete_gradient_topk"
    assert placer.discrete_gradient_topk_candidate_cache is None
    assert not optimizer.state
    profile_path = tmp_path / "unit_real_size_transition_profile_latest.json"
    assert profile_path.is_file()
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["status"] == "completed"
    assert profile["transition_total_ms"] >= 0.0


def test_warmup_transition_rolls_back_runtime_owner_and_optimizer_on_failure():
    data = _real_size_collection()
    data.inst_cell_id = torch.tensor([0, 0, 0], dtype=torch.long)
    data.inst_libcell_offset = torch.tensor([0, 0, 0], dtype=torch.long)
    data.runtime_cell_state_generation = 2
    data.timing_model_generation = 3
    vt_logits = torch.nn.Parameter(torch.zeros(3, 2))
    data.vt_logits = vt_logits
    placer = NonLinearPlace.__new__(NonLinearPlace)
    torch.nn.Module.__init__(placer)
    placer.data_collections = data
    placer.size_params = torch.nn.ParameterList([data.real_size])
    placer.vt_params = torch.nn.ParameterList([vt_logits])
    placer.discrete_gradient_topk_candidate_cache = None
    placer.discrete_gradient_topk_candidate_cache_key = None
    placer._real_size_transition_done = False
    placer.placedb = SimpleNamespace()
    old_projection_result = object()
    old_projection_frame = object()
    old_projection_paths = {"summary": "/tmp/old-summary.json"}
    placer.last_projection_result = old_projection_result
    placer.last_projection_frame = old_projection_frame
    placer.last_projection_artifact_paths = old_projection_paths

    projection_result = SimpleNamespace(
        inst_ids=torch.tensor([0]),
        validation=SimpleNamespace(is_valid=True, issues=[]),
    )
    frame = ProjectionFrame(
        projection_result=projection_result,
        changed_candidate_mask=torch.tensor([True]),
        changed_inst_ids=torch.tensor([0]),
        changed_node_mask=torch.tensor([True, False, False]),
        changed_pin_ids=torch.zeros(0, dtype=torch.long),
        projected_pin_offset_x=torch.zeros(0),
        projected_pin_offset_y=torch.zeros(0),
        projected_node_size_x=None,
        projected_node_size_y=None,
        projected_node_area=None,
        projected_size_global=torch.tensor([1.0, 2.0, 3.0]),
        projected_vt_global=None,
        projected_cell_id_global=torch.tensor([1, 0, 0]),
        projected_libcell_offset_global=torch.tensor([1, 0, 0]),
        projected_inst_size_init=None,
        projected_original_pin_offset_x=None,
        projected_original_pin_offset_y=None,
        state_digest="sha256:rollback",
        topology_generation_before=0,
        topology_generation_after=None,
        timing_model_generation_before=3,
        timing_model_generation_after=None,
    )
    placer._write_projection_artifacts = (
        lambda params, stage_timing_summary: setattr(
            placer, "last_projection_result", projection_result
        )
        or setattr(placer, "last_projection_frame", frame)
        or {"summary": "/tmp/summary.json"}
    )

    def fail_after_partial_runtime_apply(_params, _placedb):
        data.inst_cell_id[0] = 1
        data.inst_libcell_offset[0] = 1
        data.runtime_cell_state_generation = 9
        data.timing_model_generation = 10
        raise RuntimeError("injected refresh failure")

    placer._apply_projected_runtime_refresh = fail_after_partial_runtime_apply
    old_real_size = data.real_size
    optimizer = torch.optim.Adam(
        [
            {"params": [old_real_size], "group_name": "sizing"},
            {"params": [vt_logits], "group_name": "sizing"},
        ],
        lr=0.1,
    )
    optimizer.state[old_real_size] = {"step": torch.tensor(4.0)}
    params = SimpleNamespace(
        sizing_parameterization="real_size",
        real_size_execution_mode="warmup_to_discrete",
        real_size_warmup_steps=1,
        continuous_size_dynamics_mode="none",
    )

    with pytest.raises(RuntimeError, match="injected refresh failure"):
        placer._maybe_transition_real_size_to_discrete(
            params=params,
            optimizer=optimizer,
            iteration=0,
            model=None,
        )

    assert torch.equal(data.inst_cell_id, torch.tensor([0, 0, 0]))
    assert torch.equal(data.inst_libcell_offset, torch.tensor([0, 0, 0]))
    assert data.runtime_cell_state_generation == 2
    assert data.timing_model_generation == 3
    assert data.sizing_parameterization == "real_size"
    assert data.real_size is old_real_size
    assert data.size_logits is None
    assert placer.size_params[0] is old_real_size
    assert optimizer.param_groups[0]["params"][0] is old_real_size
    assert old_real_size in optimizer.state
    assert params.continuous_size_dynamics_mode == "none"
    assert placer._real_size_transition_done is False
    assert placer.last_projection_result is old_projection_result
    assert placer.last_projection_frame is old_projection_frame
    assert placer.last_projection_artifact_paths is old_projection_paths


def test_size_only_topology_lifecycle_freezes_existing_generation_without_rebuild():
    placer = NonLinearPlace.__new__(NonLinearPlace)
    torch.nn.Module.__init__(placer)
    calls = []

    class Topology:
        topology_generation = 4
        frozen_topology_generation = None

        @property
        def topology_frozen(self):
            return self.frozen_topology_generation is not None

        def freeze_topology(self):
            calls.append("freeze")
            self.frozen_topology_generation = self.topology_generation
            return self.frozen_topology_generation

        def unfreeze_topology(self):
            calls.append("unfreeze")
            old = self.frozen_topology_generation
            self.frozen_topology_generation = None
            return old

    topology = Topology()
    placer.op_collections = SimpleNamespace(steiner_topo_op=topology)
    placer.data_collections = SimpleNamespace(
        buffering_timing_topology={"topology_generation": 4}
    )
    placer._live_timing_topology_initialized = True
    placer._refresh_live_timing_topology = lambda _pos: (_ for _ in ()).throw(
        AssertionError("initialized topology must not rebuild")
    )
    params = SimpleNamespace(
        placement_sizing_mode="size_only",
        production_fast_loop=False,
        with_sta=True,
    )

    assert placer._activate_size_only_frozen_topology(params, torch.zeros(1)) == 4
    assert topology.topology_frozen
    assert placer.data_collections.buffering_timing_topology[
        "topology_update_kind"
    ] == "size_only_stage_freeze"
    assert placer._release_size_only_frozen_topology(params) == 4
    assert not topology.topology_frozen
    assert calls == ["freeze", "unfreeze"]
