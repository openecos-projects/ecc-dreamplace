from types import SimpleNamespace

import pytest

from dreamplace.ops.buffer_insertion.buffering_config import (
    build_buffering_config_from_params,
)


def test_config_defaults_to_segment_inner_loop_mode():
    config = build_buffering_config_from_params(
        SimpleNamespace(),
        flow_kind="buffering",
    )

    assert config.mode == "segment"
    assert config.segment_strategy == "continuous"
    assert config.candidate_strategy == "continuous"
    assert config.active_strategy == "continuous"
    assert config.continuous_steps == 120
    assert config.continuous_lr == 0.01
    assert config.max_repeaters_per_segment == 3
    assert config.max_selected_actions is None
    assert config.fixed_bsu_index is None
    assert config.route_b_selection_fraction == 0.001
    assert config.segment_integer_projection_interval == 100
    assert config.segment_integer_projection_start_step == 100
    assert not config.segment_integer_projection_project_bsu
    assert config.segment_integer_projection_reset_optimizer_state
    assert not config.segment_projection_require_setup_criticality
    assert not config.commit_enabled
    assert config.committed_def_path == ""
    assert config.post_commit_pysta_json == ""
    assert config.output_dir == "results/unknown_design/buffering_inner_loop"


def test_config_uses_public_buffering_values():
    config = build_buffering_config_from_params(
        SimpleNamespace(
            buffering_mode="candidate",
            buffering_output_dir="out/buffering",
            buffering_continuous_steps=11,
            buffering_continuous_lr=0.2,
            buffering_max_selected_actions=9,
            buffering_max_repeaters_per_segment=6,
            buffering_fixed_bsu_index=2,
            buffering_route_b_selection_fraction=0.02,
            buffering_segment_integer_projection_interval=5,
            buffering_segment_integer_projection_start_step=10,
            buffering_segment_integer_projection_project_bsu=1,
            buffering_segment_integer_projection_reset_optimizer_state=0,
            buffering_segment_projection_min_z_to_insert=0.2,
            buffering_segment_projection_require_setup_criticality=1,
            buffering_committed_def_path="out/committed.def",
            buffering_post_commit_pysta_json="out/post_commit_pysta.json",
        ),
        flow_kind="buffering",
    )

    assert config.mode == "candidate"
    assert config.output_dir == "out/buffering"
    assert config.continuous_steps == 11
    assert config.continuous_lr == 0.2
    assert config.max_selected_actions == 9
    assert config.max_repeaters_per_segment == 6
    assert config.fixed_bsu_index == 2
    assert config.route_b_selection_fraction == 0.02
    assert config.segment_integer_projection_interval == 5
    assert config.segment_integer_projection_start_step == 10
    assert config.segment_integer_projection_project_bsu
    assert not config.segment_integer_projection_reset_optimizer_state
    assert config.segment_projection_min_z_to_insert == 0.2
    assert config.segment_projection_require_setup_criticality
    assert config.committed_def_path == "out/committed.def"
    assert config.post_commit_pysta_json == "out/post_commit_pysta.json"


def test_config_rejects_nonpositive_max_repeaters_per_segment():
    with pytest.raises(
        ValueError,
        match="buffering_max_repeaters_per_segment must be positive",
    ):
        build_buffering_config_from_params(
            SimpleNamespace(buffering_max_repeaters_per_segment=0),
            flow_kind="buffering",
        )


def test_discrete_net_gradient_config_accepts_resolved_integer_route_b_state():
    config = build_buffering_config_from_params(
        SimpleNamespace(
            buffering_mode="segment",
            buffering_segment_strategy="discrete_net_gradient",
            buffering_fixed_bsu_index=2,
            buffering_segment_count_z_init=0.0,
            buffering_segment_integer_projection_interval=0,
            buffering_segment_integer_projection_start_step=0,
            buffering_segment_integer_projection_project_bsu=0,
            buffering_segment_projection_min_z_to_insert=0.5,
            buffering_discrete_count_proximal_lambda=1.0e-4,
        ),
        flow_kind="buffering",
    )

    assert config.segment_strategy == "discrete_net_gradient"
    assert config.fixed_bsu_index == 2
    assert config.segment_integer_projection_interval == 0
    assert config.discrete_count_proximal_lambda == 1.0e-4
    assert not config.segment_capacity_enabled


def test_candidate_discrete_net_gradient_config_accepts_fixed_bsu_state():
    config = build_buffering_config_from_params(
        SimpleNamespace(
            buffering_mode="candidate",
            buffering_candidate_strategy="discrete_net_gradient",
            buffering_fixed_bsu_index=7,
            buffering_continuous_steps=5,
        ),
        flow_kind="buffering",
    )

    assert config.mode == "candidate"
    assert config.segment_strategy == "continuous"
    assert config.candidate_strategy == "discrete_net_gradient"
    assert config.active_strategy == "discrete_net_gradient"
    assert config.fixed_bsu_index == 7
    assert config.continuous_steps == 5


@pytest.mark.parametrize(
    "overrides, message",
    (
        (
            {"buffering_fixed_bsu_index": None},
            "requires buffering_fixed_buffer_master or buffering_fixed_bsu_index",
        ),
        (
            {"buffering_max_selected_actions": 1},
            "does not allow max selected actions",
        ),
        (
            {"buffering_continuous_steps": 0},
            "buffering_continuous_steps must be positive",
        ),
        (
            {"_buffering_continuous_lr_explicit": True},
            "does not use buffering_continuous_lr",
        ),
    ),
)
def test_candidate_discrete_net_gradient_config_rejects_conflicts(overrides, message):
    values = {
        "buffering_mode": "candidate",
        "buffering_candidate_strategy": "discrete_net_gradient",
        "buffering_fixed_bsu_index": 7,
        "buffering_continuous_steps": 5,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        build_buffering_config_from_params(SimpleNamespace(**values), flow_kind="buffering")


def test_candidate_discrete_net_gradient_config_rejects_non_candidate_mode():
    with pytest.raises(ValueError, match="requires buffering_mode=candidate"):
        build_buffering_config_from_params(
            SimpleNamespace(
                buffering_mode="segment",
                buffering_candidate_strategy="discrete_net_gradient",
                buffering_fixed_bsu_index=7,
            ),
            flow_kind="buffering",
        )


def test_candidate_discrete_net_gradient_config_rejects_joint_flow():
    with pytest.raises(ValueError, match="requires standalone flow_kind=buffering"):
        build_buffering_config_from_params(
            SimpleNamespace(
                buffering_mode="candidate",
                buffering_candidate_strategy="discrete_net_gradient",
                buffering_fixed_bsu_index=7,
            ),
            flow_kind="joint",
        )


@pytest.mark.parametrize(
    "overrides, message",
    (
        ({"buffering_mode": "candidate"}, "requires buffering_mode=segment"),
        (
            {"buffering_fixed_bsu_index": None},
            "requires buffering_fixed_buffer_master or buffering_fixed_bsu_index",
        ),
        (
            {
                "buffering_segment_capacity_enabled": 1,
                "buffering_segment_capacity_grid": "4",
            },
            "does not allow segment capacity control",
        ),
        ({"buffering_max_selected_actions": 1}, "does not allow max selected actions"),
        (
            {"buffering_segment_integer_projection_interval": 1},
            "requires periodic integer projection disabled",
        ),
        ({"buffering_segment_count_z_init": 0.1}, "requires z_init=0"),
    ),
)
def test_discrete_net_gradient_config_rejects_conflicting_controls(overrides, message):
    values = {
        "buffering_mode": "segment",
        "buffering_segment_strategy": "discrete_net_gradient",
        "buffering_fixed_bsu_index": 2,
        "buffering_segment_count_z_init": 0.0,
        "buffering_segment_integer_projection_interval": 0,
        "buffering_segment_integer_projection_start_step": 0,
        "buffering_segment_integer_projection_project_bsu": 0,
        "buffering_segment_projection_min_z_to_insert": 0.5,
    }
    values.update(overrides)

    with pytest.raises(ValueError, match=message):
        build_buffering_config_from_params(SimpleNamespace(**values), flow_kind="buffering")


def test_config_ignores_removed_segment_aliases():
    config = build_buffering_config_from_params(
        SimpleNamespace(
            buffering_segment_count_steps=13,
            buffering_segment_count_lr=0.03,
            buffering_segment_count_max_repeater_count=5,
        ),
        flow_kind="buffering",
    )

    assert config.continuous_steps == 120
    assert config.continuous_lr == 0.01
    assert config.max_repeaters_per_segment == 3


def test_config_can_enable_final_commit_boundary_explicitly_for_buffering_flow():
    config = build_buffering_config_from_params(
        SimpleNamespace(buffering_commit_enabled=1),
        flow_kind="buffering",
    )

    assert config.commit_enabled


def test_config_can_enable_final_commit_boundary_explicitly_for_joint_flow():
    config = build_buffering_config_from_params(
        SimpleNamespace(
            buffering_commit_enabled=1,
            joint_buffer_max_per_segment=5,
        ),
        flow_kind="joint",
    )

    assert config.commit_enabled
    assert config.max_repeaters_per_segment == 5


@pytest.mark.parametrize("grid", ("4", "8"))
def test_segment_capacity_config_accepts_explicit_coarse_grid_and_disables_periodic_projection(
    grid,
):
    config = build_buffering_config_from_params(
        SimpleNamespace(
            buffering_mode="segment",
            buffering_fixed_bsu_index=2,
            buffering_segment_capacity_enabled=1,
            buffering_segment_capacity_grid=grid,
            buffering_segment_integer_projection_interval=100,
        ),
        flow_kind="buffering",
    )

    assert config.segment_capacity_enabled
    assert config.segment_capacity_grid == grid
    assert config.fixed_bsu_index == 2
    assert config.segment_integer_projection_interval == 0


@pytest.mark.parametrize(
    "params, message",
    (
        (
            SimpleNamespace(
                buffering_mode="candidate",
                buffering_fixed_bsu_index=2,
                buffering_segment_capacity_enabled=1,
                buffering_segment_capacity_grid="4",
            ),
            "requires buffering_mode=segment",
        ),
        (
            SimpleNamespace(
                buffering_mode="segment",
                buffering_segment_capacity_enabled=1,
                buffering_segment_capacity_grid="4",
            ),
            "requires buffering_fixed_buffer_master or buffering_fixed_bsu_index",
        ),
        (
            SimpleNamespace(
                buffering_mode="segment",
                buffering_fixed_bsu_index=2,
                buffering_segment_capacity_enabled=1,
                buffering_segment_capacity_grid="4",
                buffering_max_selected_actions=1,
            ),
            "does not allow max selected actions",
        ),
        (
            SimpleNamespace(
                buffering_mode="segment",
                buffering_fixed_bsu_index=2,
                buffering_segment_capacity_enabled=1,
                buffering_segment_capacity_grid="4",
                buffering_segment_projection_require_setup_criticality=1,
            ),
            "does not allow criticality action filtering",
        ),
    ),
)
def test_segment_capacity_config_rejects_incompatible_selection_controls(params, message):
    with pytest.raises(ValueError, match=message):
        build_buffering_config_from_params(params, flow_kind="buffering")


def test_segment_capacity_config_rejects_unknown_coarse_grid():
    with pytest.raises(ValueError, match="must be one of auto, 4, or 8"):
        build_buffering_config_from_params(
            SimpleNamespace(
                buffering_mode="segment",
                buffering_fixed_bsu_index=2,
                buffering_segment_capacity_enabled=1,
                buffering_segment_capacity_grid="16",
            ),
            flow_kind="buffering",
        )


def test_segment_capacity_config_rejects_unselected_auto_grid():
    with pytest.raises(ValueError, match="requires an explicit capacity grid: 4 or 8"):
        build_buffering_config_from_params(
            SimpleNamespace(
                buffering_mode="segment",
                buffering_fixed_bsu_index=2,
                buffering_segment_capacity_enabled=1,
            ),
            flow_kind="buffering",
        )


def test_config_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unsupported buffering mode"):
        build_buffering_config_from_params(
            SimpleNamespace(buffering_mode="unknown_mode"),
            flow_kind="buffering",
        )
