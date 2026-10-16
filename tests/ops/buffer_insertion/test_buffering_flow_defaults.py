from types import SimpleNamespace

import pytest

from dreamplace.flows.flow_config import apply_flow_defaults


def test_buffering_flow_defaults_to_segment_native_projection_recipe():
    params = SimpleNamespace(flow_kind="buffering")

    apply_flow_defaults(params)

    assert params.buffering_mode == "segment"
    assert params.buffering_candidate_strategy == "continuous"
    assert params.buffering_continuous_steps == 120
    assert params.buffering_continuous_lr == 0.01
    assert (
        params.buffering_segment_count_timing_backend
        == "cpp_cuda_segment_transfer_explicit_autograd"
    )
    assert params.buffering_segment_transfer_backend == "segment_transfer_native"
    assert params.buffering_segment_integer_projection_start_step == 0
    assert params.buffering_segment_integer_projection_interval == 0
    assert params.buffering_fixed_buffer_master == "BUFX1H7L"
    assert params.buffering_fixed_bsu_index is None
    assert params.buffering_route_b_selection_fraction == 0.001
    assert params.buffering_commit_enabled == 1
    assert params.buffering_segment_integer_projection_project_bsu == 0
    assert params.buffering_segment_integer_projection_reset_optimizer_state == 1
    assert params.buffering_segment_projection_min_z_to_insert == 0.5
    assert not hasattr(params, "buffering_segment_projection_require_setup_criticality")


def test_buffering_flow_keeps_explicit_backend_override():
    params = SimpleNamespace(
        flow_kind="buffering",
        buffering_segment_count_timing_backend="prepared_python",
        _buffering_segment_count_timing_backend_explicit=True,
    )

    apply_flow_defaults(params)

    assert params.buffering_segment_count_timing_backend == "prepared_python"
    assert params.buffering_segment_transfer_backend == "segment_transfer_native"


def test_discrete_net_gradient_flow_normalizes_inherited_continuous_defaults():
    params = SimpleNamespace(
        flow_kind="buffering",
        buffering_segment_strategy="discrete_net_gradient",
        buffering_fixed_bsu_index=2,
        _buffering_fixed_bsu_index_explicit=True,
    )

    apply_flow_defaults(params)

    assert params.buffering_segment_strategy == "discrete_net_gradient"
    assert params.buffering_segment_count_z_init == 0.0
    assert params.buffering_segment_integer_projection_interval == 0
    assert params.buffering_segment_integer_projection_start_step == 0
    assert params.buffering_segment_integer_projection_project_bsu == 0


def test_candidate_discrete_net_gradient_flow_preserves_exact_candidate_controls():
    params = SimpleNamespace(
        flow_kind="buffering",
        buffering_mode="candidate",
        _buffering_mode_explicit=True,
        buffering_candidate_strategy="discrete_net_gradient",
        _buffering_candidate_strategy_explicit=True,
        buffering_fixed_bsu_index=7,
        _buffering_fixed_bsu_index_explicit=True,
        buffering_continuous_steps=5,
        _buffering_continuous_steps_explicit=True,
    )

    apply_flow_defaults(params)

    assert params.buffering_mode == "candidate"
    assert params.buffering_segment_strategy == "continuous"
    assert params.buffering_candidate_strategy == "discrete_net_gradient"
    assert params.buffering_fixed_buffer_master == "BUFX1H7L"
    assert params.buffering_fixed_bsu_index == 7
    assert params.buffering_continuous_steps == 5


def test_candidate_discrete_net_gradient_flow_rejects_segment_mode():
    params = SimpleNamespace(
        flow_kind="buffering",
        buffering_mode="segment",
        buffering_candidate_strategy="discrete_net_gradient",
        buffering_fixed_bsu_index=7,
    )

    with pytest.raises(ValueError, match="requires buffering_mode=candidate"):
        apply_flow_defaults(params)


def test_candidate_discrete_net_gradient_rejects_explicit_continuous_lr():
    params = SimpleNamespace(
        flow_kind="buffering",
        buffering_mode="candidate",
        buffering_candidate_strategy="discrete_net_gradient",
        buffering_fixed_bsu_index=7,
        buffering_continuous_steps=5,
        buffering_continuous_lr=0.01,
        _buffering_continuous_lr_explicit=True,
    )

    with pytest.raises(ValueError, match="does not use buffering_continuous_lr"):
        apply_flow_defaults(params)


def test_joint_flow_defaults_to_segment_buffering_profile():
    params = SimpleNamespace(flow_kind="joint")

    apply_flow_defaults(params)

    assert params.flow_kind == "joint"
    assert params.placement_sizing_mode == "joint"
    assert params.diff_timing_driven_placement == 1
    assert params.differentiable_timing_obj == 1
    assert params.with_sta == 1
    assert params.buffering_mode == "segment"
    assert params.buffering_segment_count_tns_gradient == 1
    assert params.buffering_segment_count_timing_backend == (
        "cpp_cuda_segment_transfer_explicit_autograd"
    )
    assert params.buffering_segment_transfer_backend == "segment_transfer_native"
    assert params.joint_buffer_commit_period == 100
    assert params.joint_buffer_max_per_segment == 3
    assert params.joint_buffer_overflow_gate == 0.3
    assert params.buffering_segment_integer_projection_interval == 0
    assert params.buffering_segment_projection_min_z_to_insert == 0.5
    assert not hasattr(params, "buffering_segment_projection_require_setup_criticality")


def test_joint_proximal_profile_applies_canonical_defaults():
    params = SimpleNamespace(
        flow_kind="joint",
        joint_quality_profile="proximal_alternating_v1",
    )

    apply_flow_defaults(params)

    assert params.flow_kind == "joint"
    assert params.joint_quality_profile == "proximal_alternating_v1"
    assert params.timing_placement_carrier == "gradient_net_weight"
    assert params.buffering_commit_enabled == 0
    assert params.buffering_segment_projection_min_z_to_insert == 0.15
    assert params.buffering_segment_projection_require_setup_criticality == 1
    assert not hasattr(params, "buffering_max_selected_actions")
    assert params.joint_proximal_enabled == 1
    assert params.joint_proximal_lambda_x > 0.0
    assert hasattr(params, "joint_proximal_lambda_x")
    assert hasattr(params, "joint_proximal_lambda_s")
    assert hasattr(params, "joint_proximal_lambda_z")
    assert hasattr(params, "joint_proximal_lambda_b")
    assert hasattr(params, "joint_proximal_buffer_mu")
    assert not hasattr(params, "joint_proximal_buffer_resource_criticality_weighting")
    assert not hasattr(params, "joint_proximal_buffer_resource_weight_mode")
    assert not hasattr(
        params, "joint_proximal_buffer_resource_criticality_weight_floor"
    )


def test_joint_flow_keeps_explicit_buffering_overrides():
    params = SimpleNamespace(
        flow_kind="joint",
        buffering_segment_count_timing_backend="prepared_python",
        _buffering_segment_count_timing_backend_explicit=True,
    )

    apply_flow_defaults(params)

    assert params.buffering_segment_count_timing_backend == "prepared_python"
    assert params.placement_sizing_mode == "joint"


def test_segment_count_direct_joint_profile_resolves_fixed_pb_recipe():
    params = SimpleNamespace(
        flow_kind="joint",
        joint_quality_profile="segment_count_direct_joint_v1",
    )

    apply_flow_defaults(params)

    assert params.placement_sizing_mode == "joint"
    assert params.timing_placement_carrier == "direct_loss"
    assert params.continuous_size_dynamics_mode == "none"
    assert params.buffering_mode == "segment"
    assert params.buffering_segment_strategy == "discrete_net_gradient"
    assert params.buffering_segment_count_z_init == 0.0
    assert params.buffering_fixed_buffer_master == "BUFX1H7L"
    assert params.buffering_fixed_bsu_index is None
    assert params.buffering_max_repeaters_per_segment == 3
    assert params.buffering_segment_live_geometry == 1
    assert params.buffering_segment_capacity_enabled == 0
    assert params.buffering_segment_integer_projection_interval == 0
    assert params.joint_buffer_commit_period == 0
    assert params.joint_segment_sizing_enabled == 1
    assert params.joint_segment_virtual_density_enabled == 1
    assert params.joint_proximal_enabled == 1
    assert params.joint_proximal_lambda_x == 1.0e-8
    assert params.joint_proximal_lambda_s == 0.0
    assert params.joint_proximal_lambda_z == 1.0e-4
    assert params.joint_proximal_lambda_b == 0.0
    assert params.joint_proximal_buffer_mu == 0.0
    assert params.buffering_commit_enabled == 1
