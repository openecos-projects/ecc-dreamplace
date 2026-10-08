import importlib.util
import csv
import json
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parent / "run_random_init_validation.py"
spec = importlib.util.spec_from_file_location("random_init_validation", MODULE_PATH)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def _write_case(root: Path, case: str) -> None:
    design = root / "design" / case
    workspace = design / "workspace"
    params = workspace / "config" / "dreamplace_config" / "param.json"
    params.parent.mkdir(parents=True)
    params.write_text(
        json.dumps(
            {
                "global_place_stages": [
                    {
                        "iteration": 7,
                        "Llambda_density_weight_iteration": 2,
                        "Lsub_iteration": 3,
                        "optimizer": "adam",
                    }
                ],
                "target_density": 0.8,
            }
        ),
        encoding="utf-8",
    )
    for suffix in ("def", "v", "sdc"):
        (design / f"{case}.{suffix}").write_text("fixture\n", encoding="utf-8")
    asap7 = root / "ASAP7"
    for path in runner._asap7_inputs(root)["lef"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    for path in runner._asap7_inputs(root)["lib"]:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    for path in (
        runner._asap7_inputs(root)["tech_lef"],
        runner._asap7_inputs(root)["rc_tcl"],
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")


def test_placement_budget_counts_nested_optimizer_steps(tmp_path):
    case = "NV_NVDLA_partition_m"
    _write_case(tmp_path, case)
    params = runner._case_inputs(tmp_path, case)["params_json"]

    budget = runner._placement_budget(params)

    assert budget["P"] == 42
    assert budget["stages"][0]["optimizer_steps"] == 42


def test_openroad_tcl_uses_requested_thread_count(tmp_path, monkeypatch):
    captured = {}

    def fake_run(command, *, cwd, log_path):
        captured["command"] = command
        return {"status": "ok", "returncode": 0}

    monkeypatch.setattr(runner, "_run_command", fake_run)
    runner._run_openroad_tcl(
        openroad_bin=Path("/openroad"),
        tcl_path=tmp_path / "run.tcl",
        log_path=tmp_path / "run.log",
        openroad_num_threads=16,
    )

    command = captured["command"]
    assert command[command.index("-threads") + 1] == "16"


def test_overflow_milestone_campaign_default_is_500_steps():
    assert runner.DEFAULT_OVERFLOW_MILESTONE_ITERATIONS == 500


def test_placement_arms_enable_fillers_by_default():
    parser = runner.build_arg_parser()
    args = parser.parse_args(["--method", "TDP-pin2pin"])

    assert args.enable_fillers is None
    assert runner._resolve_enable_fillers(args.enable_fillers, ["TDP-pin2pin"])


def test_filler_default_preserves_nonplacement_protocols():
    assert not runner._resolve_enable_fillers(None, ["m1", "m2"])
    assert not runner._resolve_enable_fillers(None, ["S"])
    assert not runner._resolve_enable_fillers(False, ["TDP-pin2pin"])
    with pytest.raises(ValueError, match="qualified only for placement arms"):
        runner._resolve_enable_fillers(True, ["S"])


def test_m2_control_uses_same_joint_profile_without_route_b_or_density(tmp_path):
    case = "NV_NVDLA_partition_m"
    _write_case(tmp_path, case)
    command = runner.build_m2_m4_command(
        python_bin=Path("/python"),
        benchmark_root=tmp_path,
        case=case,
        r0_def=tmp_path / "R0.def",
        method_root=tmp_path / "m2",
        iterations=42,
        warmup_iterations=0,
        outer_iterations=3,
        gpu_id=0,
        placement_optimizer="nesterov",
        route_b_enabled=False,
        virtual_density_enabled=False,
    )
    joined = " ".join(command)

    assert "--joint-quality-profile segment_count_direct_joint_v1" in joined
    assert "--iterations 126" in joined
    assert "--placement-optimizer nesterov" in joined
    assert "--joint-segment-virtual-window-steps 42" in joined
    assert "--joint-segment-route-b-enabled 0" in joined
    assert "--joint-segment-virtual-density-enabled 0" in joined
    assert "--buffering-commit-enabled 0" in joined
    assert "--enable-fillers 0" in joined


def test_m3_uses_three_rounds_and_matching_count_proximal_term(tmp_path):
    case = "NV_NVDLA_partition_m"
    _write_case(tmp_path, case)
    command = runner.build_m3_command(
        python_bin=Path("/python"),
        benchmark_root=tmp_path,
        case=case,
        m2_def=tmp_path / "m2.def",
        method_root=tmp_path / "m3",
        rounds=3,
        gpu_id=0,
    )
    joined = " ".join(command)

    assert "--buffering-continuous-steps 3" in joined
    assert "--buffering-discrete-count-proximal-lambda 0.0001" in joined
    assert "--buffering-segment-strategy discrete_net_gradient" in joined


def test_overflow_milestone_sb_uses_one_outer_loop_without_fixed_window(tmp_path):
    case = "NV_NVDLA_partition_m"
    _write_case(tmp_path, case)

    command = runner.build_overflow_milestone_command(
        python_bin=Path("/python"),
        benchmark_root=tmp_path,
        case=case,
        r0_def=tmp_path / "R0.def",
        method_root=tmp_path / "SB",
        iterations=42,
        gpu_id=0,
        placement_optimizer="nesterov",
        placement_timing_enabled=True,
        sizing_enabled=True,
        route_b_enabled=True,
    )
    joined = " ".join(command)

    assert "--iterations 42" in joined
    assert "--stop-overflow 0.1" in joined
    assert "--joint-buffer-outer-iterations 1" in joined
    assert "--joint-segment-sizing-enabled 1" in joined
    assert "--joint-segment-route-b-enabled 1" in joined
    assert "--joint-segment-virtual-density-enabled 1" in joined
    assert "--buffering-commit-enabled 1" in joined
    assert joined.count("--joint-segment-milestone") == 5
    assert "--joint-segment-sizing-rounds 5" in joined
    assert "--joint-segment-sizing-up-percent 10.0" in joined
    assert "--joint-segment-buffering-rounds 1" in joined
    assert "--joint-segment-buffering-selection-fraction 0.001" in joined
    assert "--joint-segment-pin2pin-rebootstrap-enabled 1" in joined
    assert "--joint-segment-virtual-window-steps" not in joined
    assert "--joint-segment-virtual-warmup-steps" not in joined


def test_overflow_milestone_forwards_complete_v3_schedule(tmp_path):
    case = "NV_NVDLA_partition_m"
    _write_case(tmp_path, case)

    command = runner.build_overflow_milestone_command(
        python_bin=Path("/python"),
        benchmark_root=tmp_path,
        case=case,
        r0_def=tmp_path / "R0.def",
        method_root=tmp_path / "SB",
        iterations=99,
        gpu_id=0,
        placement_optimizer="nesterov",
        placement_timing_enabled=False,
        sizing_enabled=True,
        route_b_enabled=True,
        pin2pin_enabled=True,
        stop_overflow=0.08,
        timing_topology_enable_overflow_threshold=0.36,
        pin2pin_update_interval=17,
        joint_segment_milestones=(0.29, 0.18, 0.09),
        joint_segment_sizing_rounds=3,
        joint_segment_sizing_up_percent=7.5,
        joint_segment_buffering_rounds=1,
        joint_segment_buffering_selection_fraction=0.002,
        joint_segment_pin2pin_rebootstrap_enabled=False,
    )
    joined = " ".join(command)

    assert "--stop-overflow 0.08" in joined
    assert "--timing-topology-enable-overflow-threshold 0.36" in joined
    assert "--net-weighting-update-interval 17" in joined
    assert joined.count("--joint-segment-milestone") == 3
    assert "--joint-segment-sizing-rounds 3" in joined
    assert "--joint-segment-sizing-up-percent 7.5" in joined
    assert "--joint-segment-buffering-selection-fraction 0.002" in joined
    assert "--joint-segment-pin2pin-rebootstrap-enabled 0" in joined


def test_overflow_milestone_stage_ablation_controls_actions_and_commit(tmp_path):
    case = "NV_NVDLA_partition_m"
    _write_case(tmp_path, case)

    commands = {}
    for method in ("P", "TDP", "TDP-pin2pin", "S", "B"):
        arm = runner.OVERFLOW_MILESTONE_ARMS[method]
        commands[method] = " ".join(
            runner.build_overflow_milestone_command(
                python_bin=Path("/python"),
                benchmark_root=tmp_path,
                case=case,
                r0_def=tmp_path / "R0.def",
                method_root=tmp_path / method,
                iterations=42,
                gpu_id=0,
                placement_optimizer="nesterov",
                placement_timing_enabled=arm["placement_timing_enabled"],
                sizing_enabled=arm["sizing_enabled"],
                route_b_enabled=arm["route_b_enabled"],
                pin2pin_enabled=arm["pin2pin_enabled"],
            )
        )

    assert "--joint-segment-sizing-enabled" not in commands["P"]
    assert "--flow-kind placement" in commands["P"]
    assert "--joint-quality-profile" not in commands["P"]
    assert "--diff-timing-driven-placement 0" in commands["P"]
    assert "--enable-net-weighting 0" in commands["P"]
    assert "--buffering-commit-enabled 0" in commands["P"]
    assert "--flow-kind placement" in commands["TDP"]
    assert "--joint-quality-profile" not in commands["TDP"]
    assert "--diff-timing-driven-placement 1" in commands["TDP"]
    assert "--enable-net-weighting 0" in commands["TDP"]
    assert "--buffering-commit-enabled 0" in commands["TDP"]
    assert "--timing-topology-enable-overflow-threshold 0.3" in commands["TDP"]
    assert "--timing-topology-refresh-interval 10" in commands["TDP"]
    assert "--timing-grad-balance-target-ratio 0.1" in commands["TDP"]
    assert "--flow-kind placement" in commands["TDP-pin2pin"]
    assert "--diff-timing-driven-placement 0" in commands["TDP-pin2pin"]
    assert "--enable-net-weighting 1" in commands["TDP-pin2pin"]
    assert "--net-weighting-scheme pin2pin" in commands["TDP-pin2pin"]
    assert "--pin2pin-weight 2.5e-05" in commands["TDP-pin2pin"]
    assert "--pin2pin-min-weight 10.0" in commands["TDP-pin2pin"]
    assert "--pin2pin-max-weight 50.0" in commands["TDP-pin2pin"]
    assert "--pin2pin-accumulate-weight 0.2" in commands["TDP-pin2pin"]
    assert "--net-weighting-npaths 0" in commands["TDP-pin2pin"]
    assert "--net-weighting-update-interval 15" in commands["TDP-pin2pin"]
    assert "--timing-topology-enable-overflow-threshold 0.3" in commands["TDP-pin2pin"]
    assert "--timing-grad-balance-target-ratio" not in commands["TDP-pin2pin"]
    assert "--flow-kind joint" in commands["S"]
    assert "--joint-quality-profile segment_count_direct_joint_v1" in commands["S"]
    assert "--joint-segment-sizing-enabled 1" in commands["S"]
    assert "--joint-segment-route-b-enabled 0" in commands["S"]
    assert "--diff-timing-driven-placement 1" in commands["S"]
    assert "--enable-net-weighting 0" in commands["S"]
    assert "--buffering-commit-enabled 1" in commands["S"]
    assert "--joint-segment-sizing-enabled 0" in commands["B"]
    assert "--joint-segment-route-b-enabled 1" in commands["B"]
    assert "--diff-timing-driven-placement 1" in commands["B"]
    assert "--enable-net-weighting 0" in commands["B"]
    assert "--buffering-commit-enabled 1" in commands["B"]


def test_overflow_milestone_timing_ratio_is_configurable(tmp_path):
    case = "NV_NVDLA_partition_m"
    _write_case(tmp_path, case)

    command = runner.build_overflow_milestone_command(
        python_bin=Path("/python"),
        benchmark_root=tmp_path,
        case=case,
        r0_def=tmp_path / "R0.def",
        method_root=tmp_path / "TDP",
        iterations=42,
        gpu_id=0,
        placement_optimizer="nesterov",
        placement_timing_enabled=True,
        sizing_enabled=False,
        route_b_enabled=False,
        timing_grad_balance_target_ratio=0.3,
    )

    assert "--timing-grad-balance-target-ratio 0.3" in " ".join(command)


def test_overflow_milestone_activation_threshold_is_configurable(tmp_path):
    case = "NV_NVDLA_partition_m"
    _write_case(tmp_path, case)

    command = runner.build_overflow_milestone_command(
        python_bin=Path("/python"),
        benchmark_root=tmp_path,
        case=case,
        r0_def=tmp_path / "R0.def",
        method_root=tmp_path / "TDP-pin2pin",
        iterations=42,
        gpu_id=0,
        placement_optimizer="nesterov",
        placement_timing_enabled=False,
        sizing_enabled=False,
        route_b_enabled=False,
        pin2pin_enabled=True,
        timing_topology_enable_overflow_threshold=0.18,
    )

    assert "--timing-topology-enable-overflow-threshold 0.18" in " ".join(command)


def test_overflow_milestone_fillers_are_explicitly_configurable(tmp_path):
    case = "NV_NVDLA_partition_m"
    _write_case(tmp_path, case)

    command = runner.build_overflow_milestone_command(
        python_bin=Path("/python"),
        benchmark_root=tmp_path,
        case=case,
        r0_def=tmp_path / "R0.def",
        method_root=tmp_path / "TDP-pin2pin",
        iterations=42,
        gpu_id=0,
        placement_optimizer="nesterov",
        placement_timing_enabled=False,
        sizing_enabled=False,
        route_b_enabled=False,
        enable_fillers=True,
        pin2pin_enabled=True,
    )

    assert "--enable-fillers 1" in " ".join(command)


def test_overflow_milestone_legalization_is_explicitly_configurable(tmp_path):
    case = "NV_NVDLA_partition_m"
    _write_case(tmp_path, case)

    command = runner.build_overflow_milestone_command(
        python_bin=Path("/python"),
        benchmark_root=tmp_path,
        case=case,
        r0_def=tmp_path / "R0.def",
        method_root=tmp_path / "TDP-pin2pin",
        iterations=42,
        gpu_id=0,
        placement_optimizer="nesterov",
        enable_fillers=True,
        legalize=True,
        placement_timing_enabled=False,
        sizing_enabled=False,
        route_b_enabled=False,
        pin2pin_enabled=True,
    )

    assert "--legalize 1" in " ".join(command)


def test_pin2pin_sb_arm_composes_existing_three_optimization_lanes(tmp_path):
    case = "NV_NVDLA_partition_m"
    _write_case(tmp_path, case)
    arm = runner.OVERFLOW_MILESTONE_ARMS["TDP-pin2pin-SB"]

    command = runner.build_overflow_milestone_command(
        python_bin=Path("/python"),
        benchmark_root=tmp_path,
        case=case,
        r0_def=tmp_path / "R0.def",
        method_root=tmp_path / "TDP-pin2pin-SB",
        iterations=3000,
        gpu_id=0,
        placement_optimizer="nesterov",
        enable_fillers=True,
        plot_flag=0,
        **arm,
    )
    joined = " ".join(command)

    assert "--flow-kind joint" in joined
    assert "--enable-fillers 1" in joined
    assert "--enable-net-weighting 1" in joined
    assert "--net-weighting-scheme pin2pin" in joined
    assert "--joint-segment-sizing-enabled 1" in joined
    assert "--joint-segment-route-b-enabled 1" in joined
    assert "--joint-segment-virtual-density-enabled 1" in joined
    assert "--buffering-commit-enabled 1" in joined
    assert "--plot 0" in joined
    assert runner._method_raw_def(
        tmp_path,
        case,
        "TDP-pin2pin-SB",
    ) == (
        tmp_path
        / "TDP-pin2pin-SB"
        / "autodmp_result"
        / f"{case}_segment_joint_final.def"
    )


def test_openroad_native_full_flow_owns_repair_before_raw_def(tmp_path):
    case = "NV_NVDLA_partition_m"
    _write_case(tmp_path, case)
    output_def = tmp_path / "native_full.def"
    tcl = runner.generate_openroad_native_full_flow_tcl(
        benchmark_root=tmp_path,
        case=case,
        r0_def=tmp_path / "R0.def",
        output_def=output_def,
        pre_gpl_coordinates=tmp_path / "pre.tsv",
        target_density=0.8,
    )

    assert "global_placement -skip_initial_place -timing_driven" in tcl
    assert tcl.count("repair_design") == 1
    assert tcl.count("repair_timing -setup") == 1
    assert tcl.count("detailed_placement") == 3
    assert tcl.rindex("repair_timing -setup") < tcl.rindex("detailed_placement")
    assert tcl.rindex("check_placement") < tcl.rindex("write_def")
    assert str(output_def) in tcl
    assert "read_verilog" not in tcl


def test_r0_only_is_a_public_experiment_mode():
    parser = runner.build_arg_parser()
    args = parser.parse_args(["--r0-only"])

    assert args.r0_only


def test_openroad_native_full_flow_uses_controlled_k1_protocol():
    assert runner.OPENROAD_NATIVE_FULL_FLOW_METHOD in runner.OPENROAD_PLACEMENT_METHODS
    assert runner.OPENROAD_NATIVE_FULL_FLOW_METHOD in runner.EXPERIMENT_METHODS


def test_placement_budget_records_effective_optimizer_override(tmp_path):
    case = "NV_NVDLA_partition_m"
    _write_case(tmp_path, case)
    params = runner._case_inputs(tmp_path, case)["params_json"]

    budget = runner._placement_budget(params, optimizer_override="nesterov")

    assert budget["stages"][0]["source_optimizer"] == "adam"
    assert budget["stages"][0]["optimizer"] == "nesterov"


def test_cuda_visibility_binding_is_separate_from_logical_gpu():
    parser = runner.build_arg_parser()
    args = parser.parse_args(
        ["--gpu-id", "0", "--cuda-visible-devices", "2"]
    )

    assert args.gpu_id == 0
    assert args.cuda_visible_devices == "2"


def test_source_only_cli_is_opt_in():
    parser = runner.build_arg_parser()

    assert parser.parse_args([]).source_only is False
    assert parser.parse_args(["--source-only"]).source_only is True


def test_prepared_r0_cli_accepts_literal_or_case_template():
    parser = runner.build_arg_parser()

    literal = parser.parse_args(["--input-r0-def", "/tmp/R0.def"])
    templated = parser.parse_args(["--input-r0-def", "/tmp/{case}/R0.def"])

    assert str(literal.input_r0_def) == "/tmp/R0.def"
    assert str(templated.input_r0_def) == "/tmp/{case}/R0.def"


def test_m1_and_track_a_tcl_keep_required_boundaries(tmp_path):
    case = "NV_NVDLA_partition_m"
    _write_case(tmp_path, case)
    m1 = runner.generate_m1_tcl(
        benchmark_root=tmp_path,
        case=case,
        r0_def=tmp_path / "R0.def",
        output_def=tmp_path / "m1.def",
        pre_gpl_coordinates=tmp_path / "coordinates.tsv",
        target_density=0.8,
    )
    reweight_only = runner.generate_m1_tcl(
        benchmark_root=tmp_path,
        case=case,
        r0_def=tmp_path / "R0.def",
        output_def=tmp_path / "or_tdp_reweight_only.def",
        pre_gpl_coordinates=tmp_path / "coordinates.tsv",
        target_density=0.8,
        reweight_only=True,
    )
    plain = runner.generate_m1_tcl(
        benchmark_root=tmp_path,
        case=case,
        r0_def=tmp_path / "R0.def",
        output_def=tmp_path / "or_gpl_plain.def",
        pre_gpl_coordinates=tmp_path / "plain_coordinates.tsv",
        target_density=0.8,
        timing_driven=False,
    )
    track_a = runner.generate_evaluation_tcl(
        benchmark_root=tmp_path,
        case=case,
        input_def=tmp_path / "m1.def",
        output_def=tmp_path / "track_a.def",
        endpoint_csv=tmp_path / "slacks.csv",
        power_report=tmp_path / "power.rpt",
        closure=False,
    )

    assert "global_placement -skip_initial_place -timing_driven" in m1
    assert "-timing_driven_reweight_only" not in m1
    assert (
        "global_placement -skip_initial_place -timing_driven "
        "-timing_driven_reweight_only"
    ) in reweight_only
    assert "global_placement -skip_initial_place -density 0.8" in plain
    assert "-timing_driven" not in plain
    assert "force_center_initial_place" not in m1
    assert "dump_coordinates" in m1
    assert '$status eq "FIRM"' in m1
    assert '$status eq "LOCKED"' in m1
    assert "read_verilog" not in track_a
    assert "detailed_placement" in track_a
    assert "estimate_parasitics -placement" in track_a
    assert "dump_endpoint_slacks" in track_a
    assert "[rsz::design_area] * 1.0e12" in track_a


def test_topology_changing_methods_evaluate_committed_def(tmp_path):
    case = "NV_NVDLA_partition_m"

    assert runner._method_raw_def(tmp_path, case, "m2").name.endswith("_raw.def")
    assert runner._method_raw_def(tmp_path, case, "m3").name.endswith(
        "_committed.def"
    )
    assert runner._method_raw_def(tmp_path, case, "m4").name.endswith(
        "_committed.def"
    )
    assert runner._method_raw_def(tmp_path, case, "m4_no_density").name.endswith(
        "_committed.def"
    )
    assert runner._method_raw_def(tmp_path, case, "P").name.endswith("_raw.def")
    assert runner._method_raw_def(tmp_path, case, "TDP").name.endswith("_raw.def")
    for method in ("S", "B", "SB"):
        assert runner._method_raw_def(tmp_path, case, method) == (
            tmp_path
            / method
            / "autodmp_result"
            / f"{case}_segment_joint_final.def"
        )


def test_milestone_evidence_reads_action_counts_and_terminal_failure(tmp_path):
    case = "NV_NVDLA_partition_m"
    method_root = tmp_path / "SB"
    result_dir = method_root / "autodmp_result"
    result_dir.mkdir(parents=True)
    (result_dir / f"{case}_joint_flow_summary.json").write_text(
        json.dumps(
            {
                "summary": {
                    "segment_action_schedule_mode": "overflow_milestones",
                    "terminal_segment_route_b_reselection": {
                        "status": "reselected",
                        "policy": "rebuild_tree_reset_then_terminal_frame_route_b",
                        "frame": "terminal_final_placement_sizing",
                        "round_budget": 1,
                        "iterations": 1,
                        "terminal_reason": "max_rounds",
                        "historical_buffer_count": 3,
                        "final_buffer_count": 2,
                    },
                    "segment_direct_joint": {
                        "terminal_failure": None,
                        "trace": [
                            {
                                "status": "consumed",
                                "milestone": 0.3,
                                "iteration": 7,
                                "transition": {
                                    "sizing": {
                                        "num_changed_instances": 11,
                                        "area_delta_internal": 4.5,
                                    },
                                    "route_b": {"accepted_action_count": 3},
                                    "runtime_ms": {
                                        "sizing_select": 1.5,
                                        "route_b": 2.5,
                                        "total": 4.0,
                                    },
                                },
                            }
                        ],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    (result_dir / f"{case}_placement_debug_summary.json").write_text(
        json.dumps(
            {
                "final_overflow": 0.098,
                "timing_grad_balance": {
                    "status": "initialized",
                    "initialized": True,
                    "target_ratio": 0.1,
                    "iteration": 7,
                    "weight_applied": 0.25,
                },
            }
        ),
        encoding="utf-8",
    )

    evidence = runner._milestone_evidence(method_root, case)

    assert evidence["schedule_mode"] == "overflow_milestones"
    assert evidence["reached_milestones"] == [0.3]
    assert evidence["sizing_action_count"] == 11
    assert evidence["buffer_action_count"] == 3
    assert evidence["terminal_buffer_reselection"]["final_buffer_count"] == 2
    assert evidence["stage_runtime_ms"] == {
        "sizing_select": 1.5,
        "route_b": 2.5,
        "total": 4.0,
    }
    assert evidence["per_milestone"][0]["sizing_area_delta_internal"] == 4.5
    assert evidence["terminal_failure"] is None
    assert evidence["final_overflow"] == pytest.approx(0.098)
    assert evidence["timing_grad_balance"]["weight_applied"] == 0.25

    runner._add_post_action_step_evidence(evidence, total_optimizer_steps=10)
    assert evidence["post_final_action_placement_steps"] == 2


def test_milestone_evidence_reads_rebootstrap_micro_rounds(tmp_path):
    case = "NV_NVDLA_partition_m"
    result_dir = tmp_path / "SB" / "autodmp_result"
    result_dir.mkdir(parents=True)
    result_dir.joinpath(f"{case}_joint_flow_summary.json").write_text(
        json.dumps(
            {
                "summary": {
                    "segment_action_schedule_mode": "overflow_milestones",
                    "pin2pin_pair_generations": {
                        "generation_history": [
                            {
                                "generation_id": 1,
                                "install_reason": "activation_bootstrap",
                                "status": "timing_violating",
                                "pair_count": 8,
                                "objective_eval_count": 1,
                                "consumed_step_count": 1,
                            },
                            {
                                "generation_id": 2,
                                "install_reason": "milestone_rebootstrap",
                                "status": "timing_clean",
                                "pair_count": 0,
                                "objective_eval_count": 0,
                                "consumed_step_count": 0,
                            },
                        ],
                        "install_count_by_iteration": {"7": 1, "8": 1},
                    },
                    "segment_direct_joint": {
                        "queue": [],
                        "pending_event": None,
                        "terminal_failure": None,
                        "activation_iteration": 7,
                        "trace": [
                            {
                                "status": "consumed",
                                "milestone": 0.3,
                                "iteration": 8,
                                "queued_iteration": 7,
                                "queue_before": [0.3],
                                "queue_after_begin": [],
                                "pair_generation_before_milestone": 1,
                                "stage_trace": [
                                    {"stage": name}
                                    for name in (
                                        "tdp_objective_and_step_consumed",
                                        "sizing_micro_loop_completed",
                                        "buffering_gradient_ready",
                                        "route_b_applied",
                                        "tdp_state_invalidated",
                                        "tdp_rebootstrap_completed",
                                    )
                                ],
                                "transition": {
                                    "sizing_micro_loop": {
                                        "rounds_requested": 2,
                                        "rounds_attempted": 2,
                                        "rounds_completed": 2,
                                        "terminal_reason": "max_rounds",
                                        "changed_instance_count": 9,
                                        "area_delta_internal": 4.0,
                                        "rounds": [
                                            {
                                                "gradient_digest": "g0",
                                                "num_changed_instances": 5,
                                                "runtime_ms": 1.0,
                                            },
                                            {
                                                "gradient_digest": "g1",
                                                "num_changed_instances": 4,
                                                "runtime_ms": 2.0,
                                            },
                                        ],
                                    },
                                    "buffering_gradient": {"runtime_ms": 3.0},
                                    "route_b": {
                                        "accepted_action_count": 1,
                                        "runtime_ms": 4.0,
                                    },
                                    "pair_generation_before": 1,
                                    "pair_generation_after": 2,
                                },
                            }
                        ],
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    result_dir.joinpath(f"{case}_placement_debug_summary.json").write_text(
        json.dumps({"final_overflow": 0.09}),
        encoding="utf-8",
    )

    evidence = runner._milestone_evidence(tmp_path / "SB", case)

    assert evidence["reached_milestones"] == [0.3]
    assert evidence["sizing_action_count"] == 9
    assert evidence["buffer_action_count"] == 1
    assert evidence["pending_queue"] == []
    assert evidence["activation_iteration"] == 7
    assert evidence["per_milestone"][0]["pair_generation_after"] == 2
    assert len(evidence["per_milestone"][0]["sizing_rounds"]) == 2
    assert evidence["stage_runtime_ms"]["sizing_micro_loop"] == 3.0


def test_overflow_milestone_mechanism_audit_checks_physical_ownership():
    result = {
        "milestones": {
            "schedule_mode": "overflow_milestones",
            "reached_milestones": [0.30, 0.25, 0.20, 0.15, 0.10],
            "sizing_action_count": 12,
            "buffer_action_count": 5,
            "terminal_buffer_reselection": {
                "status": "reselected",
                "policy": "rebuild_tree_reset_then_terminal_frame_route_b",
                "frame": "terminal_final_placement_sizing",
                "round_budget": 5,
                "iterations": 5,
                "terminal_reason": "max_rounds",
                "historical_buffer_count": 5,
                "final_buffer_count": 2,
            },
            "terminal_failure": None,
            "final_overflow": 0.098,
            "timing_gate": {
                "threshold": 0.30,
                "timing_active_step_count": 20,
                "topology_refresh_count": 2,
                "first_timing_active_iteration": 20,
                "first_timing_active_overflow": 0.299,
            },
            "timing_grad_balance": {
                "status": "initialized",
                "initialized": True,
                "target_ratio": 0.1,
                "iteration": 20,
                "weight_applied": 0.25,
            },
        },
        "physical_commit": {
            "status": "ok",
            "physical_action_trace": [
                {"stage": "coordinate_sync"},
                {"stage": "apply_sizing"},
                {"stage": "insert_buffers"},
            ],
            "legalization_status": "ok",
            "parasitic_refresh_status": "ok",
            "def_only_opensta_verification_status": "ok",
            "def_only_vs_in_process_gap": {"wns": 0.0, "tns": 0.0},
            "sizing_substitution_count": 10,
            "sizing_master_verification": {"status": "ok"},
            "buffer_count": 2,
            "accepted_buffer_count": 2,
        },
    }

    assert runner._overflow_milestone_method_failures("SB", result) == []

    result["physical_commit"]["buffer_count"] = 1
    assert runner._overflow_milestone_method_failures("SB", result) == [
        "buffer_action_count_parity"
    ]


def test_def_only_gap_audit_allows_small_raw_opensta_replay_error():
    physical = {
        "opensta_after": {
            "wns": -326.1005367668137,
            "tns": -681063.4572431861,
        },
        "def_only_opensta_after": {
            "wns": -326.1005367668137,
            "tns": -681066.1857273023,
        },
        "def_only_vs_in_process_gap": {
            "wns": 0.0,
            "tns": -2.728484116261825,
        },
    }

    assert not runner._def_only_gap_exceeds_tolerance(physical)

    physical["def_only_vs_in_process_gap"]["tns"] = -100.0
    assert runner._def_only_gap_exceeds_tolerance(physical)

    physical["def_only_vs_in_process_gap"]["wns"] = -5.1
    assert runner._def_only_gap_exceeds_tolerance(physical)


def test_segment_physical_evidence_preserves_raw_replay_metrics(tmp_path):
    case = "NV_NVDLA_partition_m"
    result_dir = tmp_path / "SB" / "autodmp_result"
    result_dir.mkdir(parents=True)
    artifact = {
        "status": "ok",
        "def_only_opensta_after": {"wns": -326.0, "tns": -681066.0},
        "def_only_opensta_replay": {"wns": -326.0, "tns": -681066.0},
        "def_only_vs_in_process_gap": {"wns": 0.0, "tns": -2.7},
        "opensta_after": {"wns": -326.0, "tns": -681063.3},
    }
    path = result_dir / f"{case}_segment_joint_committed_verification.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    evidence = runner._segment_joint_physical_evidence(tmp_path / "SB", case)

    assert evidence["def_only_opensta_after"] == artifact["def_only_opensta_after"]
    assert evidence["def_only_opensta_replay"] == artifact["def_only_opensta_replay"]
    assert not runner._def_only_gap_exceeds_tolerance(evidence)


def test_preflight_rejection_with_no_source_mutation_is_a_valid_buffer_noop():
    physical = {
        "qor_status": "buffer_preflight_rejected_preserved_prebuffer_state",
        "buffer_count": 0,
        "accepted_buffer_count": 0,
        "physical_action_trace": [
            {
                "stage": "insert_buffers",
                "status": "preflight_rejected",
                "action_count": 350,
                "accepted_action_count": 0,
                "preflight_status": "rejected",
            }
        ],
    }

    assert not runner._buffer_action_count_mismatch(physical, 350)

    physical["buffer_count"] = 1
    assert runner._buffer_action_count_mismatch(physical, 350)


def test_rebootstrap_milestone_audit_preserves_accumulated_buffers():
    thresholds = [0.30, 0.25, 0.20, 0.15, 0.10]
    stages = [
        "tdp_objective_and_step_consumed",
        "sizing_micro_loop_completed",
        "buffering_gradient_ready",
        "route_b_applied",
        "tdp_state_invalidated",
        "tdp_rebootstrap_completed",
    ]
    result = {
        "expected_milestone_schedule": {
            "sizing_rounds": 1,
            "sizing_up_percent": 10.0,
        },
        "milestones": {
            "schedule_mode": "overflow_milestones",
            "reached_milestones": thresholds,
            "sizing_action_count": 5,
            "buffer_action_count": 5,
            "terminal_failure": None,
            "pending_queue": [],
            "pending_event": None,
            "final_overflow": 0.098,
            "timing_gate": {
                "timing_active_step_count": 0,
                "topology_refresh_count": 0,
            },
            "pair_generations": {
                "generation_history": [
                    {
                        "generation_id": generation,
                        "install_reason": (
                            "activation_bootstrap"
                            if generation == 1
                            else "milestone_rebootstrap"
                        ),
                        "status": "timing_violating",
                        "pair_count": 10,
                        "objective_eval_count": 1,
                        "consumed_step_count": 1,
                    }
                    for generation in range(1, 7)
                ],
                "install_count_by_iteration": {
                    str(iteration): 1 for iteration in range(10, 16)
                },
            },
            "per_milestone": [
                {
                    "milestone": threshold,
                    "transaction_stages": stages,
                    "pair_generation_before": index,
                    "pair_generation_after": index + 1,
                    "sizing_micro_loop": {
                        "rounds_requested": 1,
                        "rounds_attempted": 1,
                        "rounds_completed": 1,
                        "rounds": [
                            {
                                "gradient_digest": f"g{index}",
                                "num_up_improving_instances": 10,
                                "num_up_selected": 1,
                            }
                        ],
                    },
                }
                for index, threshold in enumerate(thresholds, start=1)
            ],
        },
        "physical_commit": {
            "status": "ok",
            "physical_action_trace": [
                {"stage": "coordinate_sync"},
                {"stage": "apply_sizing"},
                {"stage": "insert_buffers"},
            ],
            "legalization_status": "ok",
            "parasitic_refresh_status": "ok",
            "def_only_opensta_verification_status": "ok",
            "def_only_vs_in_process_gap": {"wns": 0.0, "tns": 0.0},
            "sizing_substitution_count": 5,
            "sizing_master_verification": {"status": "ok"},
            "buffer_count": 5,
            "accepted_buffer_count": 5,
        },
    }

    assert runner._overflow_milestone_method_failures(
        "TDP-pin2pin-SB", result
    ) == []

    result["milestones"]["per_milestone"][0]["sizing_micro_loop"]["rounds"][0][
        "num_up_selected"
    ] = 2
    assert "milestone_sizing_selection_denominator" in (
        runner._overflow_milestone_method_failures("TDP-pin2pin-SB", result)
    )


def test_overflow_milestone_audit_accepts_zero_final_buffer_reselection():
    result = {
        "milestones": {
            "schedule_mode": "overflow_milestones",
            "reached_milestones": [0.30, 0.25, 0.20, 0.15, 0.10],
            "sizing_action_count": 12,
            "buffer_action_count": 5,
            "terminal_failure": None,
            "final_overflow": 0.098,
            "timing_gate": {
                "threshold": 0.30,
                "timing_active_step_count": 20,
                "topology_refresh_count": 2,
                "first_timing_active_iteration": 20,
                "first_timing_active_overflow": 0.299,
            },
            "timing_grad_balance": {
                "status": "initialized",
                "initialized": True,
                "target_ratio": 0.1,
                "iteration": 20,
                "weight_applied": 0.25,
            },
            "terminal_buffer_reselection": {
                "status": "reselected",
                "policy": "rebuild_tree_reset_then_terminal_frame_route_b",
                "frame": "terminal_final_placement_sizing",
                "round_budget": 5,
                "iterations": 1,
                "terminal_reason": "no_positive_action",
                "historical_buffer_count": 5,
                "final_buffer_count": 0,
            },
        },
        "physical_commit": {
            "status": "ok",
            "physical_action_trace": [
                {"stage": "coordinate_sync"},
                {"stage": "apply_sizing"},
            ],
            "legalization_status": "ok",
            "parasitic_refresh_status": "ok",
            "def_only_opensta_verification_status": "ok",
            "def_only_vs_in_process_gap": {"wns": 0.0, "tns": 0.0},
            "sizing_substitution_count": 10,
            "sizing_master_verification": {"status": "ok"},
            "buffer_count": 0,
            "accepted_buffer_count": 0,
        },
    }

    assert runner._overflow_milestone_method_failures("SB", result) == []


def test_segment_joint_physical_evidence_preserves_committed_qor(tmp_path):
    case = "hidden5"
    method_root = tmp_path / "method"
    artifact = (
        method_root
        / "autodmp_result"
        / f"{case}_segment_joint_committed_verification.json"
    )
    artifact.parent.mkdir(parents=True)
    artifact.write_text(
        json.dumps(
            {
                "status": "ok",
                "opensta_before": {"wns": -56.3, "tns": -9100.5},
                "opensta_after": {"wns": -265.7, "tns": -153139.5},
                "delta_wns": -209.4,
                "delta_tns": -144039.0,
                "qor_status": "fail_tns_or_wns",
            }
        ),
        encoding="utf-8",
    )

    evidence = runner._segment_joint_physical_evidence(method_root, case)

    assert evidence["qor_status"] == "fail_tns_or_wns"
    assert evidence["qor_pass"] is False
    assert evidence["delta_tns"] == pytest.approx(-144039.0)
    assert evidence["opensta_before"]["tns"] == pytest.approx(-9100.5)
    assert evidence["opensta_after"]["tns"] == pytest.approx(-153139.5)


def test_overflow_milestone_mechanism_audit_separates_p_and_tdp_timing():
    milestones = {
        "schedule_mode": "overflow_milestones",
        "reached_milestones": [0.30, 0.25, 0.20, 0.15, 0.10],
        "sizing_action_count": 0,
        "buffer_action_count": 0,
        "terminal_failure": None,
        "final_overflow": 0.098,
    }
    p_result = {
        "milestones": {
            **milestones,
            "timing_gate": {
                "threshold": 0.30,
                "timing_active_step_count": 0,
                "topology_refresh_count": 0,
            },
        }
    }
    tdp_result = {
        "milestones": {
            **milestones,
            "timing_gate": {
                "threshold": 0.30,
                "timing_active_step_count": 20,
                "topology_refresh_count": 2,
                "first_timing_active_iteration": 20,
                "first_timing_active_overflow": 0.299,
            },
            "timing_grad_balance": {
                "status": "initialized",
                "initialized": True,
                "target_ratio": 0.1,
                "iteration": 20,
                "weight_applied": 0.25,
            },
        }
    }

    assert runner._overflow_milestone_method_failures("P", p_result) == []
    assert runner._overflow_milestone_method_failures("TDP", tdp_result) == []


def test_overflow_milestone_mechanism_audit_qualifies_pin2pin_arm():
    result = {
        "milestones": {
            "final_overflow": 0.098,
            "sizing_action_count": 0,
            "buffer_action_count": 0,
            "timing_gate": {
                "timing_active_step_count": 0,
                "topology_refresh_count": 0,
            },
            "legacy_net_weight_gate": {
                "scheme": "pin2pin",
                "threshold": 0.3,
                "update_count": 3,
                "first_update_overflow": 0.298,
            },
            "legacy_net_weight": {
                "pin2pin_pair_count": 42,
                "pin2pin_objective": 12.5,
                "pin2pin_backend": "cuda",
                "pin2pin_path_pair_backend": "cpp_openmp_transition_aware",
                "pin2pin_native_summary": {
                    "state_domain": "setup_plus_recovery",
                    "recovery_endpoint_count": 2,
                    "pair_normalization_policy": "selected_max_endpoint_gba_wns",
                },
            },
        }
    }

    assert runner._overflow_milestone_method_failures("TDP-pin2pin", result) == []


def test_buffer_budget_helpers_require_commit_to_match_reached_window_actions():
    m4 = {
        "virtual_windows": [
            {"accepted_action_count": 12},
            {"accepted_action_count": 16},
        ],
        "track_a": {"mutation": {"added_count": 28}},
    }

    assert runner._accepted_window_action_count(m4) == 28
    assert runner._committed_buffer_count(m4) == 28


def test_summary_csv_exports_tdp_gate_evidence(tmp_path):
    path = tmp_path / "summary.csv"
    runner._write_summary_csv(
        path,
        [
            {
                "case": "fixture",
                "methods": {
                    "TDP": {
                        "status": "pass",
                        "milestones": {
                            "timing_gate": {
                                "timing_active_step_count": 20,
                                "timing_inactive_step_count": 40,
                                "topology_refresh_count": 3,
                                "first_timing_active_iteration": 40,
                                "first_timing_active_overflow": 0.299,
                            },
                            "timing_grad_balance": {
                                "target_ratio": 0.1,
                                "wirelength_grad_norm": 20.0,
                                "timing_grad_norm": 100.0,
                                "weight_applied": 0.02,
                            },
                        },
                    }
                },
            }
        ],
    )

    row = next(csv.DictReader(path.open(encoding="utf-8")))
    assert row["timing_active_step_count"] == "20"
    assert row["timing_inactive_step_count"] == "40"
    assert row["topology_refresh_count"] == "3"
    assert row["first_timing_active_iteration"] == "40"
    assert row["first_timing_active_overflow"] == "0.299"
    assert row["timing_grad_balance_target_ratio"] == "0.1"
    assert row["timing_grad_balance_wl_grad_l1"] == "20.0"
    assert row["timing_grad_balance_timing_grad_l1"] == "100.0"
    assert row["timing_grad_balance_weight"] == "0.02"


def test_m1_native_action_parser_preserves_resizer_and_rebuffer_semantics(tmp_path):
    log_path = tmp_path / "m1.log"
    log_path.write_text(
        "[INFO RSZ-0039] Resized 8 instances.\n"
        "[INFO RSZ-0038] Inserted 12 buffers in 9 nets.\n"
        "[INFO GPL-0107] Timing-driven: repair_design delta area: 144.750 um^2 (+5.53%)\n"
        "[INFO GPL-0109] Timing-driven: repair_design, gcells created: 3819, deleted: 3246\n"
        "[INFO GPL-0107] Timing-driven: repair_design delta area: 23.415 um^2 (+0.85%)\n"
        "[INFO GPL-0109] Timing-driven: repair_design, gcells created: 4073, deleted: 3807\n",
        encoding="utf-8",
    )

    parsed = runner._parse_m1_native_actions(log_path)

    assert parsed["explicit_resized_instances"] == 8
    assert parsed["explicit_inserted_buffers"] == 12
    assert parsed["repair_design_net_instance_delta"] == 839
    assert parsed["repair_design_area_delta_um2"] == pytest.approx(168.165)


def test_reweight_only_baseline_audit_rejects_physical_actions():
    clean = {
        "r0_coordinate_match": True,
        "native_timing_driven_actions": {
            "explicit_resized_instances": 0,
            "explicit_inserted_buffers": 0,
            "repair_design_net_instance_delta": 0,
            "repair_design_area_delta_um2": 0.0,
        },
        "track_a": {"mutation": {"added_count": 0, "removed_count": 0}},
    }

    assert runner._reweight_only_baseline_failures(clean) == []

    clean["native_timing_driven_actions"]["explicit_inserted_buffers"] = 1
    assert runner._reweight_only_baseline_failures(clean) == [
        "native_timing_driven_action_detected"
    ]


def test_slack_histogram_writes_all_artifacts_with_numpy_2(tmp_path):
    endpoint_csv = tmp_path / "endpoint_slacks.csv"
    with endpoint_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("endpoint", "slack_ns"))
        writer.writeheader()
        writer.writerows(
            (
                {"endpoint": "a", "slack_ns": "-0.20"},
                {"endpoint": "b", "slack_ns": "0.10"},
            )
        )

    result = runner._generate_slack_histogram(
        tmp_path,
        "fixture",
        {"m4": {"track_a": {"endpoint_csv": str(endpoint_csv)}}},
    )

    assert result["status"] == "ok"
    assert result["methods"]["m4"]["violating_endpoint_count"] == 1
    for suffix in ("png", "pdf", "json"):
        assert (tmp_path / f"track_a_slack_histogram.{suffix}").is_file()


def test_m1_coordinate_identity_filters_fixed_components(tmp_path):
    r0_def = tmp_path / "R0.def"
    r0_def.write_text(
        """VERSION 5.8 ;
COMPONENTS 3 ;
- u1 INVx1 + PLACED ( 10 20 ) N ;
- macro0 SRAM + PLACED ( 30 40 ) FS ;
- fixed0 TAP + FIXED ( 50 60 ) N ;
END COMPONENTS
END DESIGN
""",
        encoding="utf-8",
    )
    coordinate_tsv = tmp_path / "coordinates.tsv"
    coordinate_tsv.write_text(
        "u1\t10\t20\nmacro0\t30\t40\nfixed0\t50\t60\n",
        encoding="utf-8",
    )
    records = runner.parse_def_component_placements(r0_def)
    expected = runner.xy_coordinate_sha256(records, {"u1", "macro0"})

    identity = runner._m1_coordinate_identity(
        coordinate_tsv,
        r0_def,
        {"coordinate_identity": {"movable_nodes_xy_sha256": expected}},
    )

    assert identity["pre_gpl_coordinate_count"] == 2
    assert identity["r0_coordinate_match"]
