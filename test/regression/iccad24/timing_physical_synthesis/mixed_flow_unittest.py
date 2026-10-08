#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest


MODULE_PATH = Path(__file__).with_name("mixed_flow.py")
SPEC = importlib.util.spec_from_file_location("mixed_flow", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
mixed = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mixed)


def _args() -> SimpleNamespace:
    return SimpleNamespace(
        python_bin=Path("/opt/python"),
        openroad_bin=Path("/opt/openroad"),
        cpu_threads=16,
        gpu_id="0",
        seed=3000,
        max_steps=3000,
        stop_overflow=0.1,
        sizing_iterations=100,
        sizing_up_percent=30.0,
        sizing_down_percent=0.0,
        placement_optimizer="nesterov",
        timing_activation_overflow=0.3,
        pin2pin_update_interval=15,
        pin2pin_weight=0.0005,
        pin2pin_min_weight=10.0,
        pin2pin_max_weight=50.0,
        pin2pin_accumulate_weight=0.2,
        pin2pin_path_limit=0,
        joint_segment_milestones=[0.30, 0.25, 0.20, 0.15, 0.10],
        joint_segment_sizing_rounds=1,
        joint_segment_sizing_up_percent=10.0,
        joint_segment_buffering_rounds=1,
        joint_segment_buffering_selection_fraction=0.001,
        joint_segment_pin2pin_rebootstrap_enabled=False,
        terminal_buffering_rounds=5,
        terminal_selection_tolerance_ns=1.0e-6,
        buffer_master="BUFx4f_ASAP7_75t_R",
        timeout_sec=21600,
    )


def _unresolved_args(*extra: str):
    return mixed.build_arg_parser().parse_args(
        ["--campaign-root", "/campaign", *extra]
    )


def _source_identity(seed: str = "a") -> dict:
    payload = {
        "artifact": mixed.protocol.SOURCE_IDENTITY_ARTIFACT,
        "artifact_version": mixed.protocol.SOURCE_IDENTITY_VERSION,
        "autodmp_root": "/autodmp",
        "git_head": seed * 64,
        "tracked_dirty": True,
        "tracked_diff_sha256": "b" * 64,
        "source_file_sha256": {"runner.py": "c" * 64},
        "native_library_sha256": {"timing.so": "d" * 64},
    }
    payload["identity_digest"] = mixed.protocol.digest_payload(
        mixed.protocol._source_identity_payload(payload)
    )
    return payload


def test_v3_campaign_source_identity_must_match_current_source():
    recorded = _source_identity("a")
    campaign = {
        "source_identity_digest": recorded["identity_digest"],
        "source_identity": recorded,
    }

    assert (
        mixed._validate_campaign_source_identity(
            campaign=campaign,
            profile_version=3,
            current_source_identity=_source_identity("a"),
        )
        == recorded["identity_digest"]
    )
    try:
        mixed._validate_campaign_source_identity(
            campaign=campaign,
            profile_version=3,
            current_source_identity=_source_identity("e"),
        )
    except ValueError as error:
        assert "differs from the immutable campaign" in str(error)
    else:
        raise AssertionError("source drift must reject a v3 campaign")


def test_legacy_v2_campaign_without_source_identity_remains_loadable():
    assert (
        mixed._validate_campaign_source_identity(
            campaign={},
            profile_version=2,
            current_source_identity=_source_identity(),
        )
        is None
    )


def test_v3_profile_is_default_and_owns_schedule_values():
    args = _unresolved_args()
    profile = mixed.protocol.load_profile(args.profile)

    config, sources = mixed._resolve_effective_config(args, profile)

    assert args.profile == mixed.DEFAULT_PROFILE
    assert config["profile_version"] == 3
    assert config["placement"]["pin2pin_activation_overflow"] == 0.35
    assert config["placement"]["pin2pin_update_interval"] == 15
    assert config["coordinated_milestones"]["thresholds"] == [
        0.30,
        0.25,
        0.20,
        0.15,
        0.10,
    ]
    assert config["coordinated_milestones"]["sizing_rounds"] == 5
    assert config["coordinated_milestones"]["buffering_rounds"] == 1
    assert config["terminal_refinement"]["sizing_iterations"] == 50
    assert config["terminal_refinement"]["buffering_rounds"] == 5
    assert all(source == "profile" for source in sources.values())


def test_explicit_schedule_override_changes_effective_identity():
    base_args = _unresolved_args()
    profile = mixed.protocol.load_profile(base_args.profile)
    base, _ = mixed._resolve_effective_config(base_args, profile)
    override_args = _unresolved_args(
        "--joint-segment-sizing-rounds",
        "3",
        "--timing-activation-overflow",
        "0.4",
    )

    override, sources = mixed._resolve_effective_config(override_args, profile)

    assert override["coordinated_milestones"]["sizing_rounds"] == 3
    assert sources["joint_segment_sizing_rounds"] == "cli_override"
    assert sources["timing_activation_overflow"] == "cli_override"
    assert mixed.protocol.digest_payload(base) != mixed.protocol.digest_payload(override)


def test_v2_profile_remains_resolvable_with_legacy_schedule():
    profile_path = mixed.SCRIPT_DIR / "profiles" / "timing_physical_synthesis_v2.json"
    args = _unresolved_args("--profile", str(profile_path))
    profile = mixed.protocol.load_profile(profile_path)

    config, sources = mixed._resolve_effective_config(args, profile)

    assert config["profile_version"] == 2
    assert config["placement"]["pin2pin_activation_overflow"] == 0.3
    assert config["coordinated_milestones"]["sizing_rounds"] == 1
    assert not config["coordinated_milestones"]["pin2pin_rebootstrap_enabled"]
    assert config["terminal_refinement"]["sizing_iterations"] == 100
    assert sources["joint_segment_sizing_rounds"] == "legacy_v2_contract"


def test_resolved_buffering_profile_records_rounds_and_fraction(tmp_path):
    args = _unresolved_args()
    profile = mixed.protocol.load_profile(args.profile)
    mixed._resolve_effective_config(args, profile)

    path, evidence = mixed._resolved_equal_buffering_profile(
        args=args,
        profile=profile,
        output_root=tmp_path,
    )
    payload = mixed._read_json(path)

    assert payload["segment_mode"]["continuous_steps"] == 5
    assert payload["segment_mode"]["selection_fraction"] == 0.001
    assert evidence["resolved_sha256"] == mixed.protocol.sha256_file(path)


def test_terminal_sizing_rebuilds_from_coordinated_committed_def():
    args = _args()
    benchmark = Path("/bench")
    core_def = Path("/run/coordinated_core.def")
    command, output_def = mixed._sizing_command(
        args=args,
        benchmark_root=benchmark,
        case="ariane136",
        input_def=core_def,
        output_root=Path("/run/terminal_refinement"),
    )
    joined = " ".join(command)
    assert f"--def-input {core_def}" in joined
    assert "--flow-kind sizing" in joined
    assert "--iterations 100" in joined
    assert "--legalize 0" in joined
    assert "--enable-fillers 0" in joined
    assert output_def == Path(
        "/run/terminal_refinement/sizing/ariane136_sized.def"
    )


def test_v3_terminal_sizing_enables_autodmp_legalization():
    command, _ = mixed._sizing_command(
        args=_args(),
        benchmark_root=Path("/bench"),
        case="mempool_tile_wrap",
        input_def=Path("/run/core.def"),
        output_root=Path("/run/refinement"),
        autodmp_legalize=True,
    )
    assert command.count("--legalize") == 1
    assert command[command.index("--legalize") + 1] == "1"


def test_terminal_buffering_consumes_sized_def_with_route_b_profile():
    args = _args()
    sized_def = Path("/run/terminal_refinement/sizing/ariane136_sized.def")
    command, result, committed = mixed._buffer_command(
        args=args,
        benchmark_root=Path("/bench"),
        case="ariane136",
        input_def=sized_def,
        output_root=Path("/run/terminal_refinement"),
    )
    joined = " ".join(command)
    assert f"--def-input {sized_def}" in joined
    assert "--segment-strategy discrete_net_gradient" in joined
    assert str(mixed.EQUAL_BUFFERING_PROFILE) in joined
    assert result.name == "result.json"
    assert committed.name == "ariane136_ours_committed.def"


def test_physical_finalize_is_def_only():
    command, summary = mixed._evaluate_command(
        args=_args(),
        benchmark_root=Path("/bench"),
        case="ariane136",
        input_def=Path("/run/refined.def"),
        output_dir=Path("/run/evaluate"),
    )
    joined = " ".join(command)
    assert "--evaluation-mode physical_finalize" in joined
    assert "--def-input /run/refined.def" in joined
    assert "--verilog" not in joined
    assert summary.name == "physical_finalize_summary.json"


def test_selected_v3_checkpoint_uses_fixed_state_without_dpl():
    args = _args()
    args.profile_version = 3
    command, summary = mixed._evaluate_command(
        args=args,
        benchmark_root=Path("/bench"),
        case="ariane136",
        input_def=Path("/run/selected_legal.def"),
        output_dir=Path("/run/final_verify"),
        evaluation_mode="fixed_state",
    )

    joined = " ".join(command)
    assert "--evaluation-mode fixed_state" in joined
    assert "--def-input /run/selected_legal.def" in joined
    assert "--verilog" not in joined
    assert summary.name == "fixed_state_summary.json"
    assert "--ignore-blocked-layer-violations" in command


def test_terminal_state_selection_keeps_positive_tns_buffering_result():
    selection = mixed._terminal_state_selection(
        {
            "before_wns": -295.72,
            "before_tns": -13964.80,
            "after_wns": -305.20,
            "after_tns": -13092.82,
            "delta_wns": -9.48,
            "delta_tns": 871.98,
            "qor_status": "fail_wns_regression",
            "qor_pass": False,
        }
    )

    assert selection["status"] == "pass"
    assert selection["selected_state"] == "buffered"
    assert selection["inner_qor_pass"] is False


def test_terminal_state_selection_falls_back_on_tns_regression():
    selection = mixed._terminal_state_selection(
        {
            "delta_wns": 38.39,
            "delta_tns": -1160.73,
            "qor_status": "fail_tns_nonpositive",
            "qor_pass": False,
        }
    )

    assert selection["status"] == "pass"
    assert selection["selected_state"] == "sized"
    assert selection["reason"] == "nonpositive_committed_delta_tns"


def test_terminal_state_selection_falls_back_on_neutral_tns_commit():
    selection = mixed._terminal_state_selection(
        {
            "before_tns": -86.834,
            "after_tns": -86.834,
            "delta_tns": 0.0,
            "qor_status": "fail_tns_nonpositive",
            "qor_pass": False,
        }
    )

    assert selection["status"] == "pass"
    assert selection["selected_state"] == "sized"
    assert selection["reason"] == "nonpositive_committed_delta_tns"


def test_terminal_buffering_rounds_accept_stationary_early_stop():
    evidence = {
        "iterations": 2,
        "round_count": 2,
        "terminal_reason": "no_positive_action",
    }

    assert mixed._terminal_buffering_rounds_complete(evidence, requested_rounds=5)
    evidence["round_count"] = 1
    assert not mixed._terminal_buffering_rounds_complete(
        evidence,
        requested_rounds=5,
    )
    evidence.update(round_count=2, terminal_reason="execution_failed")
    assert not mixed._terminal_buffering_rounds_complete(
        evidence,
        requested_rounds=5,
    )


def test_terminal_state_selection_fails_without_committed_tns_evidence():
    assert mixed._terminal_state_selection({}) == {
        "status": "failed",
        "policy": "committed_opensta_tns_improvement",
        "reason": "missing_committed_delta_tns",
    }


def test_terminal_state_selection_reports_nonfinite_evidence_as_json_safe_text():
    assert mixed._terminal_state_selection({"delta_tns": float("nan")}) == {
        "status": "failed",
        "policy": "committed_opensta_tns_improvement",
        "reason": "nonfinite_committed_delta_tns",
        "observed_delta_tns": "nan",
    }


def _checkpoint(stage, tns, wns, area):
    return {
        "stage": stage,
        "status": "pass",
        "legal_def": f"/{stage}.def",
        "legal_def_sha256": f"sha-{stage}",
        "metrics": {
            "tns_ns": tns,
            "wns_ns": wns,
            "total_cell_area_um2": area,
        },
    }


def test_three_checkpoint_selection_uses_tns_then_wns_area_and_stage():
    selection = mixed._select_legal_checkpoints(
        [
            _checkpoint("core", -100.0, -1.0, 10.0),
            _checkpoint("sized", -90.0, -2.0, 12.0),
            _checkpoint("buffered", -95.0, -0.5, 11.0),
        ]
    )
    assert selection["selected_state"] == "sized"

    tied_tns = mixed._select_legal_checkpoints(
        [
            _checkpoint("core", -90.0, -2.0, 10.0),
            _checkpoint("sized", -90.0 + 5.0e-7, -1.0, 12.0),
        ]
    )
    assert tied_tns["selected_state"] == "sized"

    tied_timing = mixed._select_legal_checkpoints(
        [
            _checkpoint("core", -90.0, -1.0, 10.0),
            _checkpoint("sized", -90.0, -1.0, 9.0),
        ]
    )
    assert tied_timing["selected_state"] == "sized"

    exact_tie = mixed._select_legal_checkpoints(
        [
            _checkpoint("core", -90.0, -1.0, 10.0),
            _checkpoint("sized", -90.0, -1.0, 10.0),
        ]
    )
    assert exact_tie["selected_state"] == "core"


def test_three_checkpoint_selection_excludes_failed_state():
    failed = _checkpoint("buffered", 0.0, 0.0, 1.0)
    failed["status"] = "physical_invalid"

    selection = mixed._select_legal_checkpoints(
        [_checkpoint("core", -10.0, -1.0, 10.0), failed]
    )

    assert selection["selected_state"] == "core"
    assert selection["candidates"][1]["exclusion_reason"] == "checkpoint_not_pass"


def _passing_result(tmp_path: Path, arm: str) -> tuple[dict, str]:
    raw_def = tmp_path / f"{arm}.def"
    raw_def.write_text("VERSION 5.8 ;\n", encoding="ascii")
    terminal_summary = tmp_path / f"{arm}_track_a.json"
    terminal_summary.write_text("{}\n", encoding="ascii")
    input_hash = hashlib.sha256(b"R0").hexdigest()
    result = {
        "status": "pass",
        "arm": arm,
        "input_r0_sha256": input_hash,
        "method": {
            "status": "pass",
            "transaction_policy": mixed.TRANSACTION_POLICIES[arm],
            "raw_def": str(raw_def),
            "raw_def_sha256": mixed.protocol.sha256_file(raw_def),
        },
        "terminal": {"status": "pass", "summary": str(terminal_summary)},
    }
    return result, input_hash


def test_staged_and_openroad_results_do_not_require_coordinated_refinement(tmp_path):
    staged, staged_hash = _passing_result(tmp_path, "autodmp_staged_sequential")
    assert not mixed._existing_result_reusable(
        arm="autodmp_staged_sequential",
        result=staged,
        input_r0_sha256=staged_hash,
        sizing_iterations=100,
        buffering_rounds=5,
    )
    staged["method"]["terminal_state_selection"] = {
        "status": "pass",
        "selected_state": "sized",
    }
    assert mixed._existing_result_reusable(
        arm="autodmp_staged_sequential",
        result=staged,
        input_r0_sha256=staged_hash,
        sizing_iterations=100,
        buffering_rounds=5,
    )

    openroad, openroad_hash = _passing_result(tmp_path, "openroad_native_mixed")
    assert mixed._existing_result_reusable(
        arm="openroad_native_mixed",
        result=openroad,
        input_r0_sha256=openroad_hash,
        sizing_iterations=100,
        buffering_rounds=5,
    )


def test_coordinated_result_requires_complete_terminal_refinement(tmp_path):
    result, input_hash = _passing_result(tmp_path, "autodmp_coordinated")
    assert not mixed._existing_result_reusable(
        arm="autodmp_coordinated",
        result=result,
        input_r0_sha256=input_hash,
        sizing_iterations=100,
        buffering_rounds=5,
    )
    result["method"]["terminal_refinement"] = {
        "status": "pass",
        "sizing_iterations": 100,
        "buffering_rounds": 5,
        "topology_rebuild_evidence": {"status": "pass"},
        "buffering_round_evidence": {
            "status": "pass",
            "iterations": 5,
            "round_count": 5,
            "terminal_reason": "max_rounds",
        },
    }
    assert mixed._existing_result_reusable(
        arm="autodmp_coordinated",
        result=result,
        input_r0_sha256=input_hash,
        sizing_iterations=100,
        buffering_rounds=5,
    )


def test_coordinated_result_reuses_valid_terminal_no_action_refinement(tmp_path):
    result, input_hash = _passing_result(tmp_path, "autodmp_coordinated")
    result["method"]["terminal_refinement"] = {
        "status": "pass",
        "sizing_iterations": 100,
        "buffering_rounds": 5,
        "topology_rebuild_evidence": {"status": "pass"},
        "buffering_round_evidence": {
            "status": "skipped",
            "reason": "no_projected_buffer_actions",
            "accepted_action_count": 0,
            "iterations": 1,
        },
    }

    assert mixed._existing_result_reusable(
        arm="autodmp_coordinated",
        result=result,
        input_r0_sha256=input_hash,
        sizing_iterations=100,
        buffering_rounds=5,
    )


def test_coordinated_result_reuses_early_stationary_terminal_refinement(tmp_path):
    result, input_hash = _passing_result(tmp_path, "autodmp_coordinated")
    result["method"]["terminal_refinement"] = {
        "status": "pass",
        "sizing_iterations": 100,
        "buffering_rounds": 5,
        "topology_rebuild_evidence": {"status": "pass"},
        "buffering_round_evidence": {
            "status": "pass",
            "iterations": 2,
            "round_count": 2,
            "terminal_reason": "no_positive_action",
            "accepted_action_count": 2,
        },
    }

    assert mixed._existing_result_reusable(
        arm="autodmp_coordinated",
        result=result,
        input_r0_sha256=input_hash,
        sizing_iterations=100,
        buffering_rounds=5,
    )


def test_buffering_no_action_result_requires_matching_input_def(tmp_path):
    input_def = tmp_path / "sized.def"
    input_def.write_text("VERSION 5.8 ;\n", encoding="ascii")
    digest = mixed.protocol.sha256_file(input_def)
    summary = {
        "status": "completed",
        "inner_loop": {"terminal_reason": "no_positive_action"},
        "commit": {
            "status": "skipped",
            "reason": "no_projected_buffer_actions",
            "action_count": 0,
        },
    }
    result = {
        "status": "skipped",
        "failure": "",
        "skip_reason": "no_projected_buffer_actions",
        "action_audit_status": "skipped",
        "buffer_count": 0,
        "input_def_sha256": digest,
        "final_def": str(input_def),
    }

    assert mixed._is_valid_no_action_buffering_result(
        result,
        input_def=input_def,
        payload={"summary": summary},
    )
    result["input_def_sha256"] = "stale"
    assert not mixed._is_valid_no_action_buffering_result(
        result,
        input_def=input_def,
        payload={"summary": summary},
    )


def test_buffering_no_improving_prefix_is_a_valid_no_action_result(tmp_path):
    input_def = tmp_path / "sized.def"
    input_def.write_text("VERSION 5.8 ;\n", encoding="ascii")
    digest = mixed.protocol.sha256_file(input_def)
    summary = {
        "status": "completed",
        "inner_loop": {"terminal_reason": "no_improving_prefix"},
        "commit": {
            "status": "skipped",
            "reason": "no_projected_buffer_actions",
            "action_count": 0,
        },
    }
    result = {
        "status": "skipped",
        "failure": "",
        "skip_reason": "no_projected_buffer_actions",
        "action_audit_status": "skipped",
        "buffer_count": 0,
        "input_def_sha256": digest,
        "final_def": str(input_def),
    }

    assert mixed._is_valid_no_action_buffering_result(
        result,
        input_def=input_def,
        payload={"summary": summary},
    )


def test_staged_flow_forwards_sizing_def_when_buffering_is_skipped(tmp_path):
    input_r0 = tmp_path / "R0.def"
    place_def = tmp_path / "placed.def"
    sized_def = tmp_path / "sized.def"
    for path in (input_r0, place_def, sized_def):
        path.write_text("VERSION 5.8 ;\n", encoding="ascii")
    buffer_result = tmp_path / "buffer_result.json"
    digest = mixed.protocol.sha256_file(sized_def)
    mixed._write_json(
        buffer_result,
        {
            "result": {
                "status": "skipped",
                "failure": "",
                "skip_reason": "no_projected_buffer_actions",
                "action_audit_status": "skipped",
                "buffer_count": 0,
                "input_def_sha256": digest,
                "final_def": str(sized_def),
            },
            "summary": {
                "status": "completed",
                "inner_loop": {"terminal_reason": "no_positive_action"},
                "commit": {
                    "status": "skipped",
                    "reason": "no_projected_buffer_actions",
                    "action_count": 0,
                },
            },
        },
    )
    args = _args()

    with mock.patch.object(
        mixed,
        "_placement_command",
        return_value=( ["place"], place_def, input_r0),
    ), mock.patch.object(
        mixed,
        "_sizing_command",
        return_value=(["size"], sized_def),
    ), mock.patch.object(
        mixed,
        "_buffer_command",
        return_value=(["buffer"], buffer_result, tmp_path / "missing.def"),
    ), mock.patch.object(
        mixed,
        "_run",
        return_value={"status": "pass", "returncode": 0},
    ):
        result = mixed._run_staged(
            args=args,
            benchmark_root=Path("/bench"),
            case="toy",
            input_r0=input_r0,
            arm_root=tmp_path / "staged",
            environment={},
        )

    assert result["status"] == "pass"
    assert Path(result["raw_def"]) == sized_def
    assert result["buffering_status"] == "skipped"
    assert result["stages"]["buffering"]["result_status"] == "skipped"
    assert result["terminal_state_selection"]["selected_state"] == "sized"


def test_staged_flow_preserves_sizing_def_when_buffering_regresses_tns(tmp_path):
    input_r0 = tmp_path / "R0.def"
    place_def = tmp_path / "placed.def"
    sized_def = tmp_path / "sized.def"
    buffered_def = tmp_path / "buffered.def"
    for path in (input_r0, place_def, sized_def, buffered_def):
        path.write_text(f"VERSION 5.8 ; # {path.name}\n", encoding="ascii")
    inner_summary = tmp_path / "buffering_inner_summary.json"
    mixed._write_json(
        inner_summary,
        {
            "summary": {
                "commit": {
                    "before_tns": -100.0,
                    "after_tns": -101.0,
                    "delta_tns": -1.0,
                    "qor_status": "fail_tns_nonpositive",
                    "qor_pass": False,
                }
            }
        },
    )
    buffer_result = tmp_path / "buffer_result.json"
    mixed._write_json(
        buffer_result,
        {
            "result": {
                "status": "pass",
                "action_audit_status": "pass",
                "autodmp_summary_path": str(inner_summary),
            }
        },
    )

    with mock.patch.object(
        mixed,
        "_placement_command",
        return_value=(["place"], place_def, input_r0),
    ), mock.patch.object(
        mixed,
        "_sizing_command",
        return_value=(["size"], sized_def),
    ), mock.patch.object(
        mixed,
        "_buffer_command",
        return_value=(["buffer"], buffer_result, buffered_def),
    ), mock.patch.object(
        mixed,
        "_run",
        return_value={"status": "pass", "returncode": 0},
    ):
        result = mixed._run_staged(
            args=_args(),
            benchmark_root=Path("/bench"),
            case="toy",
            input_r0=input_r0,
            arm_root=tmp_path / "staged",
            environment={},
        )

    assert result["status"] == "pass"
    assert Path(result["raw_def"]) == sized_def
    assert result["terminal_state_selection"]["selected_state"] == "sized"
    assert Path(result["stage_defs"]["buffering"]) == buffered_def


def test_post_buffer_physical_command_rebuilds_without_optimization(tmp_path):
    config = tmp_path / "bench/design/toy/workspace/config/dreamplace_config/param.json"
    mixed._write_json(config, {"global_place_flag": 1, "detailed_place_flag": 0})
    command, output = mixed._post_buffer_physical_command(
        args=_args(), benchmark_root=tmp_path / "bench", case="toy",
        input_def=tmp_path / "committed.def", output_root=tmp_path / "tail",
    )
    params = mixed._read_json(Path(command[2]))
    assert params["global_place_flag"] == 0
    assert params["detailed_place_flag"] == 1
    assert params["random_center_init_flag"] == 0
    assert params["gp_noise_ratio"] == 0
    assert command[command.index("--legalize") + 1] == "1"
    assert command[command.index("--flow-kind") + 1] == "placement"
    assert command[command.index("--def-input") + 1] == str(tmp_path / "committed.def")
    assert output == tmp_path / "tail/buffered_autodmp_physical/toy.def"


@pytest.mark.parametrize("sizing_valid,physical_valid", [(True, True), (False, True), (True, False)])
def test_v3_tail_checks_autodmp_sized_state_without_dpl(tmp_path, sizing_valid, physical_valid):
    args = _args()
    args.profile_version = 3
    args.sizing_iterations = 50
    args.terminal_buffering_rounds = 5
    core = tmp_path / "core_legal.def"
    sized_output = tmp_path / "sized_output.def"
    sized_legal = sized_output
    buffered_raw = tmp_path / "buffered_raw.def"
    buffered_legal = tmp_path / "buffered_legal.def"
    for path in (core, sized_output, buffered_raw, buffered_legal):
        path.write_text(f"VERSION 5.8 ; # {path.name}\n", encoding="ascii")

    refinement = tmp_path / "arm" / "terminal_refinement_v3"
    topology = (
        refinement
        / "sizing"
        / "autodmp_result"
        / "toy_openroad_aimp_db_summary.json"
    )
    mixed._write_json(
        topology,
        {
            "aimp_db_source": "openroad",
            "counts": {"instances": 1, "nets": 1, "pins": 1, "timing_edges": 1},
        },
    )
    trace = tmp_path / "buffer_trace.json"
    mixed._write_json(
        trace,
        {
            "iterations": 5,
            "rounds": [{"round": index} for index in range(5)],
            "terminal_reason": "max_rounds",
            "commit": {"accepted_action_count": 2},
        },
    )
    inner = tmp_path / "buffer_inner.json"
    mixed._write_json(inner, {"summary": {"trace_path": str(trace)}})
    buffer_result = tmp_path / "buffer_result.json"
    mixed._write_json(
        buffer_result,
        {
            "result": {
                "status": "pass",
                "action_audit_status": "pass",
                "autodmp_summary_path": str(inner),
            }
        },
    )

    calls = []

    def fake_evaluate(**kwargs):
        stage = kwargs["stage"]
        calls.append((stage, kwargs["input_def"], kwargs["evaluation_mode"]))
        if stage == "sized" and not sizing_valid:
            return {
                "stage": stage,
                "status": "failed",
                "failure_status": "evaluation_failed",
                "command": ["check-sized"],
            }
        legal = {
            "core": core,
            "sized": sized_legal,
            "buffered": buffered_legal,
        }[stage]
        metrics = {
            "core": (-100.0, -2.0, 10.0),
            "sized": (-80.0, -1.5, 12.0),
            "buffered": (-70.0, -1.0, 13.0),
        }[stage]
        return {
            "stage": stage,
            "status": "pass",
            "failure_status": None,
            "legal_def": str(legal),
            "legal_def_sha256": mixed.protocol.sha256_file(legal),
            "metrics": {
                "tns_ns": metrics[0],
                "wns_ns": metrics[1],
                "total_cell_area_um2": metrics[2],
            },
            "command": [f"evaluate-{stage}"],
        }

    def fake_sizing(**kwargs):
        assert kwargs["input_def"] == core
        assert kwargs["autodmp_legalize"] is True
        return ["size"], sized_output

    def fake_buffer(**kwargs):
        assert sizing_valid, "Invalid sizing output must not reach buffering"
        assert kwargs["input_def"] == sized_legal
        return ["buffer"], buffer_result, buffered_raw

    def fake_physical(**kwargs):
        assert kwargs["input_def"] == buffered_raw
        return ["physical"], buffered_legal

    def fake_run(command, **kwargs):
        if command == ["physical"] and not physical_valid:
            return {"status": "execution_failed", "returncode": 1}
        return {"status": "pass", "returncode": 0}

    with mock.patch.object(mixed, "_sizing_command", side_effect=fake_sizing), mock.patch.object(
        mixed, "_buffer_command", side_effect=fake_buffer
    ), mock.patch.object(
        mixed, "_evaluate_legal_checkpoint", side_effect=fake_evaluate
    ), mock.patch.object(
        mixed, "_post_buffer_physical_command", side_effect=fake_physical
    ), mock.patch.object(
        mixed, "_run", side_effect=fake_run
    ):
        result = mixed._run_coordinated_v3_tail(
            args=args,
            benchmark_root=Path("/bench"),
            case="toy",
            arm_root=tmp_path / "arm",
            environment={},
            stages={"coordinated": {"status": "pass"}},
            core_legal_def=core,
            coordinated_command=["coordinated"],
            source_diagnostics={},
        )

    if not sizing_valid:
        assert calls == [
            ("core", core, "fixed_state"),
            ("sized", sized_output, "fixed_state"),
        ]
        assert result["status"] == "evaluation_failed"
        assert result["failures"] == ["sized_fixed_state_report"]
        assert result["recoverable_checkpoint"]["stage"] == "core"
        return

    if not physical_valid:
        assert result["status"] == "execution_failed"
        assert result["failures"] == ["buffered_autodmp_physical"]
        assert result["recoverable_checkpoint"]["stage"] == "sized"
        assert calls == [("core", core, "fixed_state"), ("sized", sized_output, "fixed_state")]
        return

    assert result["status"] == "pass"
    assert calls == [
        ("core", core, "fixed_state"),
        ("sized", sized_output, "fixed_state"),
        ("buffered", buffered_legal, "fixed_state"),
    ]
    assert result["terminal_refinement"]["sizing_legalization"] == "autodmp"
    assert result["terminal_refinement"]["buffering_physical_treatment"] == "autodmp_legalization_and_detailed_placement"
    assert "terminal_sizing_raw_def" not in result
    assert Path(result["terminal_sized_def"]) == sized_legal
    assert Path(result["terminal_buffered_def"]) == buffered_legal
    assert Path(result["raw_def"]) == buffered_legal
    assert result["terminal_state_selection"]["selected_state"] == "buffered"
    assert result["terminal_evaluation_mode"] == "fixed_state"


def test_v3_tail_preserves_terminal_failure_with_recoverable_core_checkpoint(
    tmp_path,
):
    args = _args()
    args.profile_version = 3
    args.sizing_iterations = 50
    core = tmp_path / "core_legal.def"
    core.write_text("VERSION 5.8 ;\n", encoding="ascii")
    sized_raw = tmp_path / "sized_raw.def"

    core_checkpoint = {
        "stage": "core",
        "status": "pass",
        "failure_status": None,
        "legal_def": str(core),
        "legal_def_sha256": mixed.protocol.sha256_file(core),
        "metrics": {
            "tns_ns": -100.0,
            "wns_ns": -2.0,
            "total_cell_area_um2": 10.0,
        },
        "command": ["evaluate-core"],
    }

    with mock.patch.object(
        mixed,
        "_evaluate_legal_checkpoint",
        return_value=core_checkpoint,
    ), mock.patch.object(
        mixed,
        "_sizing_command",
        return_value=(["size"], sized_raw),
    ), mock.patch.object(
        mixed,
        "_run",
        return_value={"status": "execution_failed", "returncode": 1},
    ):
        result = mixed._run_coordinated_v3_tail(
            args=args,
            benchmark_root=Path("/bench"),
            case="toy",
            arm_root=tmp_path / "arm",
            environment={},
            stages={"coordinated": {"status": "pass"}},
            core_legal_def=core,
            coordinated_command=["coordinated"],
            source_diagnostics={},
        )

    assert result["status"] == "execution_failed"
    assert result["failures"] == ["terminal_sizing"]
    assert result["terminal_state_selection"]["status"] == "pass"
    assert result["terminal_state_selection"]["selected_state"] == "core"
    assert result["recoverable_checkpoint"] == {
        "stage": "core",
        "def": str(core),
        "def_sha256": mixed.protocol.sha256_file(core),
    }
    assert Path(result["raw_def"]) == core


def test_coordinated_mechanism_failure_is_not_hidden_by_completed_source(tmp_path):
    args = _args()
    input_r0 = tmp_path / "R0.def"
    input_r0.write_text("VERSION 5.8 ;\n", encoding="ascii")
    source_root = tmp_path / "coordinated" / "joint_source"
    result_root = source_root / "run" / "hidden3"
    raw_def = result_root / "TDP-pin2pin-SB" / "autodmp_result" / "hidden3_segment_joint_final.def"
    copied_r0 = result_root / "r0" / "R0.def"
    raw_def.parent.mkdir(parents=True)
    copied_r0.parent.mkdir(parents=True)
    raw_def.write_text("VERSION 5.8 ;\n", encoding="ascii")
    copied_r0.write_text(input_r0.read_text(encoding="ascii"), encoding="ascii")
    mixed._write_json(
        source_root / "run" / "campaign_summary.json",
        {
            "case_results": [
                {
                    "case": "hidden3",
                    "status": "failed",
                    "failures": ["TDP-pin2pin-SB_mechanism_failed"],
                    "methods": {
                        "TDP-pin2pin-SB": {
                            "mechanism_audit": {
                                "status": "failed",
                                "failures": ["pin2pin_empty_pairs"],
                            },
                            "physical_commit": {
                                "status": "ok",
                                "final_def_sha256": mixed.protocol.sha256_file(raw_def),
                            },
                        }
                    },
                }
            ]
        },
    )

    result = mixed._run_coordinated(
        args=args,
        benchmark_root=Path("/bench"),
        case="hidden3",
        input_r0=input_r0,
        arm_root=tmp_path / "coordinated",
        environment={},
    )

    assert result["status"] == "failed"
    assert "coordinated_mechanism_audit" in result["failures"]


def test_reuse_rejects_changed_raw_def(tmp_path):
    result, input_hash = _passing_result(tmp_path, "autodmp_staged_sequential")
    Path(result["method"]["raw_def"]).write_text("changed\n", encoding="ascii")
    assert not mixed._existing_result_reusable(
        arm="autodmp_staged_sequential",
        result=result,
        input_r0_sha256=input_hash,
        sizing_iterations=100,
        buffering_rounds=5,
    )


def test_all_cases_selection_is_explicit():
    assert mixed._selected_cases(None) == mixed.DEFAULT_PILOT_CASES
    assert mixed._selected_cases(None, all_cases=True) == mixed.protocol.CANONICAL_CASES


def test_placement_command_freezes_fillers_and_external_legalization_policy():
    command, _raw_def, _r0_copy = mixed._placement_command(
        args=_args(),
        benchmark_root=Path("/bench"),
        case="NV_NVDLA_partition_m",
        input_r0=Path("/run/R0.def"),
        output_root=Path("/run/staged"),
    )
    assert "--enable-fillers" in command
    assert "--no-legalize" in command
    assert "--stop-overflow" in command
    assert command[command.index("--pin2pin-update-interval") + 1] == "15"


def test_prepared_r0_gate_accepts_complete_evidence(tmp_path):
    case = "NV_NVDLA_partition_m"
    r0_root = tmp_path / "initial_state" / "R0" / case
    r0_root.mkdir(parents=True)
    r0_def = r0_root / "R0.def"
    r0_def.write_text("VERSION 5.8 ;\n", encoding="ascii")
    digest = mixed.protocol.sha256_file(r0_def)
    mixed._write_json(
        r0_root / "R0_manifest.json",
        {"status": "pass", "r0_def_sha256": digest},
    )
    mixed._write_json(
        tmp_path / "initial_state" / "preparation_summary.json",
        {
            "status": "pass",
            "requested_input_domains": ["R0"],
            "cases": {
                case: {
                    "R0": {
                        "status": "pass",
                        "r0_def_sha256": digest,
                        "evaluation": {"status": "pass"},
                    }
                }
            },
        },
    )
    mixed._validate_prepared_r0(tmp_path, (case,))


def test_prepared_r0_gate_rejects_unverified_input(tmp_path):
    mixed._write_json(
        tmp_path / "initial_state" / "preparation_summary.json",
        {
            "status": "pass",
            "requested_input_domains": ["D_post"],
            "cases": {},
        },
    )
    try:
        mixed._validate_prepared_r0(tmp_path, ("NV_NVDLA_partition_m",))
    except ValueError as error:
        assert "did not request R0" in str(error)
    else:
        raise AssertionError("missing R0 preparation must fail")
