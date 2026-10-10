import json

import pytest

from dreamplace import placer_cli


def test_buffering_segment_count_z_init_cli_overrides_forced_flow_defaults(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "design_name": "unit",
                "buffering_segment_count_z_init": 0.0,
            }
        ),
        encoding="utf-8",
    )

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "buffering",
            "--buffering-mode",
            "segment",
            "--buffering-segment-count-z-init",
            "0.0",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.buffering_segment_count_z_init == 0.0


def test_cli_resolves_discrete_net_gradient_route_b_defaults(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "buffering",
            "--buffering-mode",
            "segment",
            "--buffering-segment-strategy",
            "discrete_net_gradient",
            "--buffering-fixed-bsu-index",
            "2",
        ]
    )
    params, launch = placer_cli.build_effective_params_from_args(args)

    assert params.buffering_segment_strategy == "discrete_net_gradient"
    assert params.buffering_segment_count_z_init == 0.0
    assert params.buffering_segment_integer_projection_interval == 0
    assert (
        launch["effective_params_manifest"]["resolved_params"][
            "buffering_segment_strategy"
        ]
        == "discrete_net_gradient"
    )


def test_cli_preserves_output_verilog_launch_path(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")
    output_verilog = tmp_path / "result" / "final.v"

    args = placer_cli.build_arg_parser().parse_args(
        [str(params_path), "--output-verilog", str(output_verilog)]
    )
    _, launch = placer_cli.build_effective_params_from_args(args)

    assert launch["output_verilog"] == str(output_verilog)


def test_cli_resolves_defaults_once_and_records_deterministic_manifest(
    tmp_path, monkeypatch
):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")
    calls = []
    original = placer_cli.apply_flow_defaults

    def counted(params):
        calls.append(params)
        return original(params)

    monkeypatch.setattr(placer_cli, "apply_flow_defaults", counted)
    args = placer_cli.build_arg_parser().parse_args(
        [str(params_path), "--flow-kind", "sizing"]
    )

    params, launch = placer_cli.build_effective_params_from_args(args)

    assert calls == [params]
    assert params._flow_defaults_resolved is True
    assert (
        launch["effective_params_manifest"]["resolved_params"]["flow_kind"] == "sizing"
    )
    assert len(launch["effective_params_manifest"]["content_sha256"]) == 64


def test_explicit_sizing_flow_replaces_legacy_nesterov_optimizer(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "design_name": "unit",
                "global_place_stages": [{"iteration": 2000, "optimizer": "nesterov"}],
            }
        ),
        encoding="utf-8",
    )

    args = placer_cli.build_arg_parser().parse_args(
        [str(params_path), "--flow-kind", "sizing", "--iterations", "50"]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.global_place_stages[0]["iteration"] == 50
    assert params.global_place_stages[0]["optimizer"] == "adam"


def test_placement_optimizer_override_updates_all_stages(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "design_name": "unit",
                "global_place_stages": [
                    {"iteration": 10, "optimizer": "adam"},
                    {"iteration": 20, "optimizer": "sgd"},
                ],
            }
        ),
        encoding="utf-8",
    )

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "placement",
            "--placement-optimizer",
            "nesterov",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert [stage["optimizer"] for stage in params.global_place_stages] == [
        "nesterov",
        "nesterov",
    ]
    assert params._placement_optimizer_explicit is True


def test_cli_manifest_hash_is_stable_for_equivalent_resolved_params(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")
    args = placer_cli.build_arg_parser().parse_args(
        [str(params_path), "--flow-kind", "buffering", "--buffering-mode", "segment"]
    )

    _, first_launch = placer_cli.build_effective_params_from_args(args)
    _, second_launch = placer_cli.build_effective_params_from_args(args)

    assert (
        first_launch["effective_params_manifest"]["content_sha256"]
        == second_launch["effective_params_manifest"]["content_sha256"]
    )


def test_cli_bin_count_override_updates_placedb_and_all_stages(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "design_name": "unit",
                "num_bins_x": 64,
                "num_bins_y": 64,
                "global_place_stages": [
                    {"num_bins_x": 64, "num_bins_y": 64},
                    {"num_bins_x": 32, "num_bins_y": 32},
                ],
            }
        ),
        encoding="utf-8",
    )
    args = placer_cli.build_arg_parser().parse_args(
        [str(params_path), "--num-bins-x", "128", "--num-bins-y", "256"]
    )

    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.num_bins_x == 128
    assert params.num_bins_y == 256
    assert all(stage["num_bins_x"] == 128 for stage in params.global_place_stages)
    assert all(stage["num_bins_y"] == 256 for stage in params.global_place_stages)
    assert params._placement_bin_count_explicit is True


def test_cli_initial_placement_learning_rate_max(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps({"design_name": "unit", "global_place_stages": [{}]}),
        encoding="utf-8",
    )
    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--placement-initial-learning-rate-max",
            "0.125",
        ]
    )

    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.placement_initial_learning_rate_max == 0.125
    assert params._placement_initial_learning_rate_max_explicit is True


def test_cli_placement_overflow_tolerance(tmp_path):
    params_json = tmp_path / "params.json"
    params_json.write_text(json.dumps({"stop_overflow": 0.1}), encoding="utf-8")
    args = placer_cli.build_arg_parser().parse_args(
        [str(params_json), "--placement-overflow-tolerance", "0.001"]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)
    assert params.placement_overflow_tolerance == 0.001
    assert params._placement_overflow_tolerance_explicit is True


def test_cli_rejects_unsupported_physical_eco_flow(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")
    args = placer_cli.build_arg_parser().parse_args(
        [str(params_path), "--flow-kind", "physical_eco"]
    )

    try:
        placer_cli.build_effective_params_from_args(args)
    except ValueError as error:
        assert "physical_eco" in str(error)
        assert "not supported" in str(error)
    else:
        raise AssertionError("physical_eco must be explicitly rejected")


def test_buffering_flow_defaults_to_lut_timing_like_sta(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "buffering",
            "--buffering-mode",
            "segment",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.timing_surrogate_mode == "lut_only"
    assert params.timing_lut_2d_native_op == "auto"
    assert params.size_interpolated_pin_native_op == "auto"


def test_joint_flow_cli_applies_compact_public_defaults(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "joint",
            "--joint-buffer-commit-period",
            "25",
            "--joint-buffer-outer-iterations",
            "2",
            "--joint-buffer-max-per-segment",
            "2",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.flow_kind == "joint"
    assert params.placement_sizing_mode == "joint"
    assert params.buffering_mode == "segment"
    assert params.diff_timing_driven_placement == 1
    assert params.with_sta == 1
    assert params.joint_buffer_commit_period == 25
    assert params.joint_buffer_outer_iterations == 2
    assert params.buffering_segment_integer_projection_interval == 25
    assert params.joint_buffer_max_per_segment == 2
    assert params.buffering_max_repeaters_per_segment == 2
    assert params.joint_post_commit_openroad_eco == "none"


def test_joint_flow_cli_applies_proximal_profile_defaults(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "joint",
            "--joint-quality-profile",
            "proximal_alternating_v1",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

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


def test_joint_flow_cli_applies_staged_smoke_profile_defaults(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "joint",
            "--joint-quality-profile",
            "staged_smoke_v1",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.flow_kind == "joint"
    assert params.joint_quality_profile == "staged_smoke_v1"
    assert params.joint_staged_smoke_outer_iterations == 1
    assert params.buffering_commit_enabled == 0
    assert params.random_center_init_flag == 0
    assert params.gp_noise_ratio == 0.0


def test_joint_flow_cli_applies_segment_count_direct_profile_defaults(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "joint",
            "--joint-quality-profile",
            "segment_count_direct_joint_v1",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.flow_kind == "joint"
    assert params.joint_quality_profile == "segment_count_direct_joint_v1"
    assert params.timing_placement_carrier == "direct_loss"
    assert params.buffering_segment_strategy == "discrete_net_gradient"
    assert params.buffering_segment_live_geometry == 1
    assert params.joint_segment_sizing_enabled == 1
    assert params.joint_segment_action_schedule_mode == "overflow_milestones"
    assert params.joint_buffer_outer_iterations == 1
    assert params.timing_topology_enable_overflow_threshold == 0.35
    assert params.net_weighting_update_interval == 15
    assert params.joint_segment_milestones == (0.30, 0.25, 0.20, 0.15, 0.10)
    assert params.joint_segment_sizing_rounds == 5
    assert params.joint_segment_sizing_up_percent == 10.0
    assert params.joint_segment_buffering_rounds == 1
    assert params.joint_segment_buffering_selection_fraction == 0.001
    assert params.joint_segment_pin2pin_rebootstrap_enabled == 1
    assert params.buffering_fixed_buffer_master == "BUFX1H7L"
    assert params.buffering_fixed_bsu_index is None


def test_segment_count_direct_profile_allows_explicit_fixed_bsu_override(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "joint",
            "--joint-quality-profile",
            "segment_count_direct_joint_v1",
            "--buffering-fixed-bsu-index",
            "0",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.buffering_fixed_bsu_index == 0


def test_segment_count_direct_profile_allows_explicit_buffer_master(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "joint",
            "--joint-quality-profile",
            "segment_count_direct_joint_v1",
            "--buffering-fixed-buffer-master",
            "BUFX4H7H",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.buffering_fixed_buffer_master == "BUFX4H7H"
    assert params.buffering_fixed_bsu_index is None


def test_segment_direct_joint_profile_inherits_the_placement_gpu_for_timing(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "design_name": "unit",
                # Legacy benchmark configs commonly pin this to CPU. The
                # canonical GPU profile owns the default runtime policy.
                "timing_propagation_device": "cpu",
            }
        ),
        encoding="utf-8",
    )

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "joint",
            "--joint-quality-profile",
            "segment_count_direct_joint_v1",
            "--gpu",
            "1",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.gpu == 1
    assert params.timing_propagation_device == "inherit"


def test_segment_direct_joint_profile_honors_explicit_timing_device_override(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "joint",
            "--joint-quality-profile",
            "segment_count_direct_joint_v1",
            "--gpu",
            "1",
            "--timing-propagation-device",
            "cpu",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.timing_propagation_device == "cpu"
    assert params._timing_propagation_device_explicit is True


@pytest.mark.parametrize(
    "flow_kind, extra_args",
    [
        ("sizing", ["--with-sta"]),
        ("buffering", ["--with-sta"]),
    ],
)
def test_timing_active_flows_inherit_the_placement_gpu_by_default(
    tmp_path,
    flow_kind,
    extra_args,
):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "design_name": "unit",
                "timing_propagation_device": "cpu",
            }
        ),
        encoding="utf-8",
    )

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            flow_kind,
            "--gpu",
            "1",
            *extra_args,
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.with_sta == 1
    assert params.timing_propagation_device == "inherit"


def test_segment_count_direct_profile_accepts_named_milestone_budget_overrides(
    tmp_path,
):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")
    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "joint",
            "--joint-quality-profile",
            "segment_count_direct_joint_v1",
            "--joint-segment-milestone",
            "0.28",
            "--joint-segment-milestone",
            "0.14",
            "--joint-segment-sizing-rounds",
            "3",
            "--joint-segment-sizing-up-percent",
            "8",
            "--joint-segment-buffering-rounds",
            "1",
            "--joint-segment-buffering-selection-fraction",
            "0.002",
            "--joint-segment-pin2pin-rebootstrap-enabled",
            "0",
        ]
    )

    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.joint_segment_milestones == (0.28, 0.14)
    assert params.joint_segment_sizing_rounds == 3
    assert params.joint_segment_sizing_up_percent == 8.0
    assert params.joint_segment_buffering_rounds == 1
    assert params.joint_segment_buffering_selection_fraction == 0.002
    assert params.joint_segment_pin2pin_rebootstrap_enabled == 0


def test_segment_count_direct_profile_allows_timing_only_density_ablation(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "joint",
            "--joint-quality-profile",
            "segment_count_direct_joint_v1",
            "--joint-segment-virtual-density-enabled",
            "0",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.joint_segment_virtual_density_enabled == 0


def test_segment_count_direct_profile_allows_sizing_ablation(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "joint",
            "--joint-quality-profile",
            "segment_count_direct_joint_v1",
            "--joint-segment-sizing-enabled",
            "0",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.joint_segment_sizing_enabled == 0
    assert params.placement_sizing_mode == "place_only"
    assert params.joint_segment_action_schedule_mode == "overflow_milestones"


def test_segment_count_direct_profile_allows_controlled_placement_only_arm(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "joint",
            "--joint-quality-profile",
            "segment_count_direct_joint_v1",
            "--joint-segment-route-b-enabled",
            "0",
            "--joint-segment-virtual-window-steps",
            "20",
            "--joint-segment-virtual-warmup-steps",
            "30",
            "--joint-segment-virtual-density-enabled",
            "0",
            "--buffering-commit-enabled",
            "0",
            "--enable-fillers",
            "0",
            "--legalize",
            "0",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.joint_segment_route_b_enabled == 0
    assert params.joint_segment_virtual_window_steps == 20
    assert params.joint_segment_virtual_warmup_steps == 30
    assert params.joint_segment_virtual_density_enabled == 0
    assert params.buffering_commit_enabled == 0
    assert params.enable_fillers == 0
    assert params.legalize_flag == 0
    assert params.joint_segment_action_schedule_mode == "fixed_window_compat"


def test_segment_count_direct_profile_rejects_multiple_milestone_outer_loops(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "joint",
            "--joint-quality-profile",
            "segment_count_direct_joint_v1",
            "--joint-buffer-outer-iterations",
            "2",
        ]
    )

    with pytest.raises(
        ValueError,
        match="overflow_milestones requires joint_buffer_outer_iterations=1",
    ):
        placer_cli.build_effective_params_from_args(args)


def test_segment_count_direct_profile_allows_explicit_no_commit_diagnostic(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "joint",
            "--joint-quality-profile",
            "segment_count_direct_joint_v1",
            "--buffering-commit-enabled",
            "0",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.buffering_commit_enabled == 0
    assert params.buffering_segment_live_geometry == 1
    assert params.joint_buffer_commit_period == 0


def test_joint_staged_smoke_outer_iterations_cli_override(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "joint",
            "--joint-quality-profile",
            "staged_smoke_v1",
            "--joint-staged-smoke-outer-iterations",
            "2",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.joint_staged_smoke_outer_iterations == 2


def test_joint_proximal_profile_keeps_explicit_carrier_override(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "joint",
            "--joint-quality-profile",
            "proximal_alternating_v1",
            "--timing-placement-carrier",
            "direct_loss",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.joint_quality_profile == "proximal_alternating_v1"
    assert params.timing_placement_carrier == "direct_loss"
    assert params.joint_proximal_enabled == 1


def test_joint_flow_cli_accepts_post_commit_openroad_eco_profile(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "joint",
            "--joint-post-commit-openroad-eco",
            "resynth_once",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.joint_post_commit_openroad_eco == "resynth_once"


def test_diff_timing_driven_placement_cli_applies_profile_defaults(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "placement",
            "--diff-timing-driven-placement",
            "1",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.flow_kind == "placement"
    assert params.diff_timing_driven_placement == 1
    assert params.with_sta == 1
    assert params.differentiable_timing_obj == 1
    assert params.timing_surrogate_mode == "lut_only"
    assert params.timing_objective_lane == "timing_only"
    assert params.timing_topology_refresh_interval == 15
    assert params.timing_topology_enable_overflow_threshold == 0.35
    assert params.timing_wns_coeff == 0.01
    assert params.timing_tns_coeff == 0.0001
    assert params.timing_placement_carrier == "direct_loss"
    assert params.timing_gradient_net_weight_scale == 0.4
    assert params.timing_gradient_net_weight_max == 2.0
    assert params.pin2pin_weight == 0.0005
    assert params.pin2pin_min_weight == 10.0
    assert params.pin2pin_max_weight == 50.0
    assert params.pin2pin_accumulate_weight == 0.2


def test_timing_grad_balance_ratio_schema_default_is_loaded():
    params = placer_cli._load_params_json(None)

    assert params.timing_grad_balance_target_ratio == 0.2


def test_sizing_flow_disables_position_timing_gradient_balance_by_default(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [str(params_path), "--flow-kind", "sizing"]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.placement_sizing_mode == "size_only"
    assert params.diff_timing_driven_placement == 0
    assert params.timing_wns_coeff == 0.01
    assert params.timing_tns_coeff == 0.0001
    assert params.timing_grad_balance_target_ratio == 0.0


def test_diff_timing_grad_balance_ratio_from_params_json_is_preserved(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "design_name": "unit",
                "timing_grad_balance_target_ratio": 0.25,
            }
        ),
        encoding="utf-8",
    )

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "placement",
            "--diff-timing-driven-placement",
            "1",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.timing_grad_balance_target_ratio == 0.25


def test_cli_timing_grad_balance_ratio_overrides_params_json(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "design_name": "unit",
                "timing_grad_balance_target_ratio": 0.25,
            }
        ),
        encoding="utf-8",
    )

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "placement",
            "--diff-timing-driven-placement",
            "1",
            "--timing-grad-balance-target-ratio",
            "0.3",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.timing_grad_balance_target_ratio == 0.3
    assert params._timing_grad_balance_target_ratio_explicit


def test_placement_cli_can_disable_legacy_net_weighting(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps({"design_name": "unit", "enable_net_weighting": 1}),
        encoding="utf-8",
    )

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "placement",
            "--diff-timing-driven-placement",
            "0",
            "--enable-net-weighting",
            "0",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.diff_timing_driven_placement == 0
    assert params.enable_net_weighting == 0


def test_placement_cli_preserves_explicit_sta_for_legacy_net_weighting(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "placement",
            "--with-sta",
            "--diff-timing-driven-placement",
            "0",
            "--enable-net-weighting",
            "1",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.flow_kind == "placement"
    assert params.with_sta == 1
    assert params.diff_timing_driven_placement == 0
    assert params.differentiable_timing_obj == 0
    assert params.enable_net_weighting == 1


def test_placement_cli_configures_pin2pin_net_weighting(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "placement",
            "--with-sta",
            "--diff-timing-driven-placement",
            "0",
            "--enable-net-weighting",
            "1",
            "--net-weighting-scheme",
            "pin2pin",
            "--pin2pin-weight",
            "0.00005",
            "--pin2pin-min-weight",
            "10",
            "--pin2pin-max-weight",
            "50",
            "--pin2pin-accumulate-weight",
            "0.2",
            "--net-weighting-npaths",
            "3.0",
            "--net-weighting-update-interval",
            "15",
            "--timing-topology-enable-overflow-threshold",
            "0.3",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.with_sta == 1
    assert params.diff_timing_driven_placement == 0
    assert params.enable_net_weighting == 1
    assert params.net_weighting_scheme == "pin2pin"
    assert params.pin2pin_weight == 0.00005
    assert params.pin2pin_min_weight == 10.0
    assert params.pin2pin_max_weight == 50.0
    assert params.pin2pin_accumulate_weight == 0.2
    assert params.net_weighting_npaths == 3.0
    assert params.net_weighting_update_interval == 15
    assert params.timing_topology_enable_overflow_threshold == 0.3


def test_placement_cli_configures_endpoint_limited_pin2pin_weighting(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps({"design_name": "unit", "net_weighting_npaths": 3.0}),
        encoding="utf-8",
    )

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "placement",
            "--net-weighting-scheme",
            "pin2pin",
            "--net-weighting-nendpoints",
            "32",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.net_weighting_nendpoints == 32
    assert params.net_weighting_npaths == 3.0
    assert params._net_weighting_nendpoints_explicit is True


def test_placement_cli_rejects_endpoint_percentage_out_of_range(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    with pytest.raises(SystemExit):
        placer_cli.build_arg_parser().parse_args(
            [str(params_path), "--net-weighting-npaths", "100.1"]
        )


def test_placement_cli_rejects_inverted_pin2pin_weight_bounds(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")
    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--pin2pin-min-weight",
            "50",
            "--pin2pin-max-weight",
            "10",
        ]
    )

    with pytest.raises(ValueError, match="pin2pin_min_weight"):
        placer_cli.build_effective_params_from_args(args)


def test_placement_cli_can_force_center_initialization(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "design_name": "unit",
                "random_center_init_flag": 0,
                "init_loc_perc_x": 0.1,
                "init_loc_perc_y": 0.9,
            }
        ),
        encoding="utf-8",
    )

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "placement",
            "--random-center-init",
            "1",
            "--init-loc-perc-x",
            "0.5",
            "--init-loc-perc-y",
            "0.5",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.random_center_init_flag == 1
    assert params.init_loc_perc_x == 0.5
    assert params.init_loc_perc_y == 0.5


def test_explicit_placement_flow_disables_sizing_state_from_legacy_params(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "design_name": "unit",
                "placement_sizing_mode": "size_only",
                "continuous_size_dynamics_mode": "discrete_gradient_topk",
                "discrete_gradient_topk_ranking_mode": "taylor_delta_candidate",
            }
        ),
        encoding="utf-8",
    )

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "placement",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.flow_kind == "placement"
    assert params.placement_sizing_mode == "place_only"
    assert params.continuous_size_dynamics_mode == "none"


def test_explicit_placement_flow_disables_timing_state_from_legacy_params(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "design_name": "unit",
                "with_sta": 1,
                "differentiable_timing_obj": 1,
            }
        ),
        encoding="utf-8",
    )

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "placement",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.flow_kind == "placement"
    assert params.with_sta == 0
    assert params.differentiable_timing_obj == 0
    assert params.diff_timing_driven_placement == 0


def test_joint_cli_explicitly_disables_differentiable_placement_timing(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "joint",
            "--joint-quality-profile",
            "segment_count_direct_joint_v1",
            "--diff-timing-driven-placement",
            "0",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.flow_kind == "joint"
    assert params.diff_timing_driven_placement == 0
    assert params.differentiable_timing_obj == 0
    assert params.with_sta == 1


def test_diff_timing_driven_placement_cli_keeps_gate_overrides(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")

    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "placement",
            "--diff-timing-driven-placement",
            "1",
            "--timing-topology-refresh-interval",
            "5",
            "--timing-topology-enable-overflow-threshold",
            "0.15",
            "--timing-wns-coeff",
            "0.002",
            "--timing-tns-coeff",
            "0.00002",
            "--timing-grad-balance-target-ratio",
            "0.3",
            "--timing-placement-carrier",
            "gradient_net_weight",
            "--timing-gradient-net-weight-scale",
            "2.5",
            "--timing-gradient-net-weight-max",
            "6.0",
        ]
    )
    params, _ = placer_cli.build_effective_params_from_args(args)

    assert params.timing_topology_refresh_interval == 5
    assert params.timing_topology_enable_overflow_threshold == 0.15
    assert params.timing_wns_coeff == 0.002
    assert params.timing_tns_coeff == 0.00002
    assert params.timing_grad_balance_target_ratio == 0.3
    assert params.timing_placement_carrier == "gradient_net_weight"
    assert params.timing_gradient_net_weight_scale == 2.5
    assert params.timing_gradient_net_weight_max == 6.0


def test_segment_driver_cap_mode_reaches_effective_params_manifest(tmp_path):
    params_path = tmp_path / "params.json"
    params_path.write_text(json.dumps({"design_name": "unit"}), encoding="utf-8")
    args = placer_cli.build_arg_parser().parse_args(
        [
            str(params_path),
            "--flow-kind",
            "buffering",
            "--buffering-segment-driver-cap-mode",
            "direct",
        ]
    )
    params, launch = placer_cli.build_effective_params_from_args(args)
    assert params.buffering_segment_driver_cap_mode == "direct"
    assert (
        launch["effective_params_manifest"]["resolved_params"][
            "buffering_segment_driver_cap_mode"
        ]
        == "direct"
    )
