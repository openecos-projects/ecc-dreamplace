#!/usr/bin/env python3
"""Run the three primary placement/sizing/buffering mixed-flow arms."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
AUTODMP_ROOT = SCRIPT_DIR.parents[3]
DEFAULT_PROFILE = SCRIPT_DIR / "profiles" / "timing_physical_synthesis_v3.json"
DEFAULT_PILOT_CASES = ("NV_NVDLA_partition_m", "ariane136", "hidden5")
PLACEMENT_RUNNER = (
    AUTODMP_ROOT
    / "test"
    / "regression"
    / "iccad24"
    / "joint_place_sizing_buffering"
    / "run_random_init_validation.py"
)
EQUAL_BUFFERING_RUNNER = (
    AUTODMP_ROOT
    / "test"
    / "regression"
    / "iccad24"
    / "equal_spaced_buffering"
    / "run.py"
)
EQUAL_BUFFERING_PROFILE = (
    AUTODMP_ROOT
    / "test"
    / "regression"
    / "iccad24"
    / "equal_spaced_buffering"
    / "params"
    / "route_b_allcases_0p1pct.json"
)
COMMON_EVALUATOR = AUTODMP_ROOT / "test" / "regression" / "common" / "def_only_track_a.py"
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import campaign_protocol as protocol


ARMS = (
    "autodmp_staged_sequential",
    "autodmp_coordinated",
    "openroad_native_mixed",
)

TRANSACTION_POLICIES = {
    "autodmp_staged_sequential": "staged_def_handoff_no_interstage_dpl",
    "autodmp_coordinated": "coordinated_core_commit_then_terminal_refinement_commit",
    "openroad_native_mixed": "native_openroad_transaction",
}

_TERMINAL_BUFFERING_EARLY_STOP_REASONS = {
    "no_positive_action",
    "no_improving_prefix",
}


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_campaign_source_identity(
    *,
    campaign: dict[str, Any],
    profile_version: int,
    current_source_identity: dict[str, Any],
) -> str | None:
    expected = campaign.get("source_identity_digest")
    recorded = campaign.get("source_identity")
    if expected is None and recorded is None and int(profile_version) < 3:
        return None
    if not expected or not isinstance(recorded, dict):
        raise ValueError("v3 campaign is missing its AutoDMP source identity")
    protocol.validate_source_identity(recorded)
    if recorded.get("identity_digest") != expected:
        raise ValueError("campaign source identity fields are inconsistent")
    protocol.validate_source_identity(current_source_identity)
    if current_source_identity.get("identity_digest") != expected:
        raise ValueError(
            "current AutoDMP source identity differs from the immutable campaign"
        )
    return str(expected)


def _resolve_effective_config(
    args: argparse.Namespace,
    profile: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    """Resolve campaign knobs once, retaining where each value came from."""

    version = int(profile["artifact_version"])
    placement = dict(profile["placement"])
    refinement = dict(profile["mixed_terminal_refinement"])
    buffering = dict(profile["buffering"])
    if version >= 3:
        milestones = dict(profile["coordinated_milestones"])
        milestone_source = "profile"
    else:
        milestones = {
            "thresholds": [0.30, 0.25, 0.20, 0.15, 0.10],
            "sizing_rounds": 1,
            "sizing_up_percent": 10.0,
            "buffering_rounds": 1,
            "buffering_selection_fraction": 0.001,
        }
        milestone_source = "legacy_v2_contract"

    sources: dict[str, str] = {}

    def resolve(name: str, default: Any, *, source: str = "profile") -> Any:
        requested = getattr(args, name, None)
        if requested is None:
            value = copy.deepcopy(default)
            sources[name] = source
        else:
            value = requested
            sources[name] = "cli_override"
        setattr(args, name, value)
        return value

    config = {
        "profile_name": profile["profile_name"],
        "profile_version": version,
        "placement": {
            "optimizer": resolve("placement_optimizer", placement["optimizer"]),
            "max_steps": int(resolve("max_steps", placement["max_steps"])),
            "stop_overflow": float(
                resolve("stop_overflow", placement["stop_overflow"])
            ),
            "pin2pin_activation_overflow": float(
                resolve(
                    "timing_activation_overflow",
                    placement["pin2pin_activation_overflow"],
                )
            ),
            "pin2pin_update_interval": int(
                resolve(
                    "pin2pin_update_interval",
                    placement["pin2pin_update_interval"],
                )
            ),
            "pin2pin_weight": float(
                resolve("pin2pin_weight", placement["pin2pin_weight"])
            ),
            "pin2pin_min_weight": float(
                resolve("pin2pin_min_weight", placement["pin2pin_min_weight"])
            ),
            "pin2pin_max_weight": float(
                resolve("pin2pin_max_weight", placement["pin2pin_max_weight"])
            ),
            "pin2pin_accumulate_weight": float(
                resolve(
                    "pin2pin_accumulate_weight",
                    placement["pin2pin_accumulate_weight"],
                )
            ),
            "pin2pin_path_limit": float(
                resolve("pin2pin_path_limit", placement["pin2pin_path_limit"])
            ),
        },
        "coordinated_milestones": {
            "thresholds": [
                float(value)
                for value in resolve(
                    "joint_segment_milestones",
                    milestones["thresholds"],
                    source=milestone_source,
                )
            ],
            "sizing_rounds": int(
                resolve(
                    "joint_segment_sizing_rounds",
                    milestones["sizing_rounds"],
                    source=milestone_source,
                )
            ),
            "sizing_up_percent": float(
                resolve(
                    "joint_segment_sizing_up_percent",
                    milestones["sizing_up_percent"],
                    source=milestone_source,
                )
            ),
            "buffering_rounds": int(
                resolve(
                    "joint_segment_buffering_rounds",
                    milestones["buffering_rounds"],
                    source=milestone_source,
                )
            ),
            "buffering_selection_fraction": float(
                resolve(
                    "joint_segment_buffering_selection_fraction",
                    milestones["buffering_selection_fraction"],
                    source=milestone_source,
                )
            ),
            "pin2pin_rebootstrap_enabled": version >= 3,
        },
        "terminal_refinement": {
            "sizing_iterations": int(
                resolve("sizing_iterations", refinement["sizing_iterations"])
            ),
            "sizing_up_percent": float(
                resolve("sizing_up_percent", refinement["sizing_up_percent"])
            ),
            "sizing_down_percent": float(
                resolve("sizing_down_percent", refinement["sizing_down_percent"])
            ),
            "buffering_rounds": int(
                resolve(
                    "terminal_buffering_rounds",
                    refinement["buffering_rounds"],
                )
            ),
            "selection_policy": refinement["terminal_selection_policy"],
            "selection_tolerance_ns": float(
                refinement.get("terminal_selection_tolerance_ns", 1.0e-6)
            ),
        },
        "buffer_master": resolve("buffer_master", buffering["buffer_master"]),
    }
    args.profile_version = version
    args.joint_segment_pin2pin_rebootstrap_enabled = version >= 3
    args.terminal_selection_tolerance_ns = config["terminal_refinement"][
        "selection_tolerance_ns"
    ]

    thresholds = config["coordinated_milestones"]["thresholds"]
    if not thresholds or any(
        thresholds[index] <= thresholds[index + 1]
        for index in range(len(thresholds) - 1)
    ):
        raise ValueError("joint segment milestones must be strictly descending")
    if thresholds[0] > config["placement"]["pin2pin_activation_overflow"]:
        raise ValueError("first milestone must not exceed Pin2Pin activation overflow")
    if config["placement"]["stop_overflow"] > thresholds[-1]:
        raise ValueError("stop overflow must not exceed the final milestone")
    if config["coordinated_milestones"]["sizing_rounds"] <= 0:
        raise ValueError("milestone sizing rounds must be positive")
    if config["coordinated_milestones"]["buffering_rounds"] != 1:
        raise ValueError("milestone buffering rounds must equal one")
    if config["placement"]["pin2pin_update_interval"] <= 0:
        raise ValueError("Pin2Pin update interval must be positive")
    if config["terminal_refinement"]["sizing_iterations"] <= 0:
        raise ValueError("terminal sizing iterations must be positive")
    if config["terminal_refinement"]["buffering_rounds"] <= 0:
        raise ValueError("terminal buffering rounds must be positive")
    for name, value in (
        (
            "milestone sizing percentage",
            config["coordinated_milestones"]["sizing_up_percent"] / 100.0,
        ),
        (
            "milestone buffering fraction",
            config["coordinated_milestones"]["buffering_selection_fraction"],
        ),
    ):
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must lie within (0, 1]")
    return config, sources


def _resolved_equal_buffering_profile(
    *,
    args: argparse.Namespace,
    profile: dict[str, Any],
    output_root: Path,
) -> tuple[Path, dict[str, Any]]:
    source = _read_json(EQUAL_BUFFERING_PROFILE)
    source_segment = dict(source.get("segment_mode") or {})
    expected = dict(profile["buffering"])
    contract = {
        "continuous_steps": int(expected["rounds"]),
        "max_repeaters_per_segment": int(expected["max_repeaters_per_segment"]),
        "fixed_bsu_index": int(expected["fixed_bsu_index"]),
        "buffer_master": str(expected["buffer_master"]),
        "timing_backend": str(expected["timing_backend"]),
        "transfer_backend": str(expected["transfer_backend"]),
        "selection_fraction": float(expected["selection_fraction"]),
    }
    mismatches = {
        key: {"expected": value, "observed": source_segment.get(key)}
        for key, value in contract.items()
        if source_segment.get(key) != value
    }
    if mismatches:
        raise ValueError(f"equal-spaced buffering profile contract mismatch: {mismatches}")

    resolved = copy.deepcopy(source)
    segment = resolved["segment_mode"]
    segment["continuous_steps"] = int(args.terminal_buffering_rounds)
    segment["selection_fraction"] = float(
        args.joint_segment_buffering_selection_fraction
    )
    path = output_root / "resolved_equal_buffering_profile.json"
    _write_json(path, resolved)
    evidence = {
        "source_path": str(EQUAL_BUFFERING_PROFILE.resolve()),
        "source_sha256": protocol.sha256_file(EQUAL_BUFFERING_PROFILE.resolve()),
        "resolved_path": str(path),
        "resolved_sha256": protocol.sha256_file(path),
    }
    return path, evidence


def _is_valid_no_action_buffering_result(
    result: dict[str, Any],
    *,
    input_def: Path,
    payload: dict[str, Any] | None = None,
) -> bool:
    """Validate a buffering skip before forwarding the input DEF."""

    if (
        result.get("status") != "skipped"
        or result.get("failure", "")
        or result.get("skip_reason") != "no_projected_buffer_actions"
        or result.get("action_audit_status") != "skipped"
    ):
        return False
    try:
        buffer_count = int(result.get("buffer_count"))
    except (TypeError, ValueError):
        return False
    if buffer_count != 0:
        return False
    input_hash = protocol.sha256_file(input_def)
    if result.get("input_def_sha256") != input_hash:
        return False
    final_def = Path(str(result.get("final_def") or ""))
    if not final_def.is_file() or protocol.sha256_file(final_def) != input_hash:
        return False

    summary = dict((payload or {}).get("summary") or {})
    if not summary:
        summary_path = Path(str(result.get("autodmp_summary_path") or ""))
        if summary_path.is_file():
            summary = dict(_read_json(summary_path).get("summary") or {})
    inner = dict(summary.get("inner_loop") or {})
    commit = dict(summary.get("commit") or {})
    return (
        summary.get("status") == "completed"
        and inner.get("terminal_reason")
        in {"no_positive_action", "no_improving_prefix"}
        and commit.get("status") == "skipped"
        and commit.get("reason") == "no_projected_buffer_actions"
        and int(commit.get("action_count", 0) or 0) == 0
        and int(commit.get("attempted_action_count", 0) or 0) == 0
        and int(commit.get("accepted_action_count", 0) or 0) == 0
        and int(commit.get("failed_action_count", 0) or 0) == 0
        and int(commit.get("rejected_action_count", 0) or 0) == 0
    )


def _existing_result_reusable(
    *,
    arm: str,
    result: dict[str, Any],
    input_r0_sha256: str,
    sizing_iterations: int,
    buffering_rounds: int,
) -> bool:
    if result.get("status") != "pass":
        return False
    if (
        result.get("arm") != arm
        or result.get("input_r0_sha256") != input_r0_sha256
    ):
        return False
    method = dict(result.get("method") or {})
    terminal = dict(result.get("terminal") or {})
    if method.get("status") != "pass" or terminal.get("status") != "pass":
        return False
    if method.get("transaction_policy") != TRANSACTION_POLICIES[arm]:
        return False
    raw_def = Path(str(method.get("raw_def") or ""))
    raw_def_sha256 = method.get("raw_def_sha256")
    if not raw_def.is_file() or not raw_def_sha256:
        return False
    if protocol.sha256_file(raw_def) != raw_def_sha256:
        return False
    terminal_summary = Path(str(terminal.get("summary") or ""))
    if not terminal_summary.is_file():
        return False
    if arm == "autodmp_staged_sequential":
        return dict(method.get("terminal_state_selection") or {}).get("status") == "pass"
    if arm != "autodmp_coordinated":
        return True
    refinement = dict(method.get("terminal_refinement") or {})
    buffering_evidence = dict(refinement.get("buffering_round_evidence") or {})
    buffering_status = buffering_evidence.get("status")
    if buffering_status == "pass":
        buffering_reusable = _terminal_buffering_rounds_complete(
            buffering_evidence,
            requested_rounds=buffering_rounds,
        )
    elif buffering_status == "skipped":
        buffering_reusable = (
            buffering_evidence.get("reason") == "no_projected_buffer_actions"
            and int(buffering_evidence.get("accepted_action_count", 0) or 0) == 0
            and int(buffering_evidence.get("iterations", 0) or 0) > 0
        )
    else:
        buffering_reusable = False
    return (
        refinement.get("status") == "pass"
        and int(refinement.get("sizing_iterations", 0) or 0)
        == int(sizing_iterations)
        and int(refinement.get("buffering_rounds", 0) or 0)
        == int(buffering_rounds)
        and dict(refinement.get("topology_rebuild_evidence") or {}).get("status")
        == "pass"
        and buffering_reusable
    )


def _validate_prepared_r0(campaign_root: Path, cases: tuple[str, ...]) -> None:
    summary_path = campaign_root / "initial_state" / "preparation_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = _read_json(summary_path)
    if summary.get("status") != "pass":
        raise ValueError("R0 preparation summary did not pass")
    if "R0" not in tuple(summary.get("requested_input_domains") or ()):
        raise ValueError("campaign preparation did not request R0")
    prepared_cases = dict(summary.get("cases") or {})
    for case in cases:
        case_summary = dict(prepared_cases.get(case) or {})
        r0_summary = dict(case_summary.get("R0") or {})
        evaluation = dict(r0_summary.get("evaluation") or {})
        r0_def = campaign_root / "initial_state" / "R0" / case / "R0.def"
        producer_manifest_path = (
            campaign_root / "initial_state" / "R0" / case / "R0_manifest.json"
        )
        if (
            r0_summary.get("status") != "pass"
            or evaluation.get("status") != "pass"
            or not r0_def.is_file()
            or not producer_manifest_path.is_file()
        ):
            raise ValueError(f"{case}: prepared R0 evidence is incomplete")
        producer_manifest = _read_json(producer_manifest_path)
        expected_hash = r0_summary.get("r0_def_sha256")
        if (
            producer_manifest.get("status") != "pass"
            or producer_manifest.get("r0_def_sha256") != expected_hash
            or protocol.sha256_file(r0_def) != expected_hash
        ):
            raise ValueError(f"{case}: prepared R0 identity mismatch")


def _terminal_state_selection(commit: dict[str, Any]) -> dict[str, Any]:
    try:
        delta_tns = float(commit["delta_tns"])
    except (KeyError, TypeError, ValueError):
        return {
            "status": "failed",
            "policy": "committed_opensta_tns_improvement",
            "reason": "missing_committed_delta_tns",
        }
    if not math.isfinite(delta_tns):
        return {
            "status": "failed",
            "policy": "committed_opensta_tns_improvement",
            "reason": "nonfinite_committed_delta_tns",
            "observed_delta_tns": str(delta_tns),
        }
    selected_state = "buffered" if delta_tns > 0.0 else "sized"
    return {
        "status": "pass",
        "policy": "committed_opensta_tns_improvement",
        "selected_state": selected_state,
        "reason": (
            "positive_committed_delta_tns"
            if selected_state == "buffered"
            else "nonpositive_committed_delta_tns"
        ),
        "before_wns": commit.get("before_wns"),
        "before_tns": commit.get("before_tns"),
        "after_wns": commit.get("after_wns"),
        "after_tns": commit.get("after_tns"),
        "delta_wns": commit.get("delta_wns"),
        "delta_tns": delta_tns,
        "inner_qor_status": commit.get("qor_status"),
        "inner_qor_pass": commit.get("qor_pass"),
    }


def _select_legal_checkpoints(
    checkpoints: list[dict[str, Any]],
    *,
    tolerance_ns: float = 1.0e-6,
) -> dict[str, Any]:
    """Select a committed legal state using the predeclared v3 ordering."""

    stage_order = {"core": 0, "sized": 1, "buffered": 2}
    candidates = []
    for checkpoint in checkpoints:
        record = copy.deepcopy(checkpoint)
        metrics = dict(record.get("metrics") or {})
        try:
            values = {
                "tns_ns": float(metrics["tns_ns"]),
                "wns_ns": float(metrics["wns_ns"]),
                "total_cell_area_um2": float(metrics["total_cell_area_um2"]),
            }
        except (KeyError, TypeError, ValueError):
            record.update(eligible=False, exclusion_reason="missing_selection_metric")
        else:
            if record.get("status") != "pass" or not all(
                math.isfinite(value) for value in values.values()
            ):
                record.update(
                    eligible=False,
                    exclusion_reason=(
                        "checkpoint_not_pass"
                        if record.get("status") != "pass"
                        else "nonfinite_selection_metric"
                    ),
                )
            elif record.get("stage") not in stage_order:
                record.update(eligible=False, exclusion_reason="unknown_stage")
            else:
                record.update(
                    eligible=True,
                    exclusion_reason=None,
                    selection_values=values,
                    stage_order=stage_order[record["stage"]],
                )
        candidates.append(record)

    eligible = [record for record in candidates if record.get("eligible")]
    if not eligible:
        return {
            "status": "failed",
            "policy": "max_tns_then_wns_then_min_area_then_earliest_stage",
            "reason": "no_eligible_legal_checkpoint",
            "tolerance_ns": float(tolerance_ns),
            "candidates": candidates,
        }

    def better(candidate: dict[str, Any], incumbent: dict[str, Any]) -> bool:
        lhs = candidate["selection_values"]
        rhs = incumbent["selection_values"]
        if lhs["tns_ns"] > rhs["tns_ns"] + tolerance_ns:
            return True
        if abs(lhs["tns_ns"] - rhs["tns_ns"]) > tolerance_ns:
            return False
        if lhs["wns_ns"] > rhs["wns_ns"] + tolerance_ns:
            return True
        if abs(lhs["wns_ns"] - rhs["wns_ns"]) > tolerance_ns:
            return False
        if lhs["total_cell_area_um2"] < rhs["total_cell_area_um2"]:
            return True
        if lhs["total_cell_area_um2"] > rhs["total_cell_area_um2"]:
            return False
        return int(candidate["stage_order"]) < int(incumbent["stage_order"])

    selected = eligible[0]
    for candidate in eligible[1:]:
        if better(candidate, selected):
            selected = candidate
    return {
        "status": "pass",
        "policy": "max_tns_then_wns_then_min_area_then_earliest_stage",
        "tolerance_ns": float(tolerance_ns),
        "selected_state": selected["stage"],
        "selected_def": selected["legal_def"],
        "selected_def_sha256": selected["legal_def_sha256"],
        "selected_metrics": selected["selection_values"],
        "candidates": candidates,
    }


def _terminal_buffering_rounds_complete(
    evidence: dict[str, Any], *, requested_rounds: int
) -> bool:
    try:
        iterations = int(evidence.get("iterations"))
        round_count = int(evidence.get("round_count"))
    except (TypeError, ValueError):
        return False
    if iterations <= 0 or round_count != iterations:
        return False
    if iterations == int(requested_rounds):
        return True
    return (
        iterations < int(requested_rounds)
        and evidence.get("terminal_reason")
        in _TERMINAL_BUFFERING_EARLY_STOP_REASONS
    )


def _run(
    command: list[str],
    *,
    cwd: Path,
    log_path: Path,
    environment: dict[str, str],
    timeout_sec: int,
) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    returncode: int | None = None
    timed_out = False
    with log_path.open("w", encoding="utf-8", errors="ignore") as stream:
        try:
            completed = subprocess.run(
                command,
                cwd=str(cwd),
                env=environment,
                stdout=stream,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
                timeout=timeout_sec,
            )
            returncode = int(completed.returncode)
        except subprocess.TimeoutExpired:
            timed_out = True
    return {
        "status": "pass" if returncode == 0 and not timed_out else "execution_failed",
        "returncode": returncode,
        "timed_out": timed_out,
        "runtime_sec": time.perf_counter() - started,
        "log_path": str(log_path),
        "command": command,
    }


def _environment(args: argparse.Namespace) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": str(args.gpu_id),
            "OMP_NUM_THREADS": str(args.cpu_threads),
            "MKL_NUM_THREADS": str(args.cpu_threads),
            "OPENBLAS_NUM_THREADS": str(args.cpu_threads),
            "NUMEXPR_NUM_THREADS": str(args.cpu_threads),
        }
    )
    return environment


def _tech_args(benchmark_root: Path) -> list[str]:
    tech = protocol.technology_paths(benchmark_root)
    command = ["--tech-lef", str(tech["tech_lef"])]
    for path in tech["lefs"]:
        command.extend(("--lef", str(path)))
    for path in tech["libs"]:
        command.extend(("--lib", str(path)))
    command.extend(("--rc-tcl", str(tech["rc_tcl"])))
    return command


def _case_placer_args(
    benchmark_root: Path,
    case: str,
    input_def: Path,
    result_dir: Path,
    output_def: Path,
) -> list[str]:
    paths = protocol.case_paths(benchmark_root, case)
    params = benchmark_root / "design" / case / "workspace" / "config" / "dreamplace_config" / "param.json"
    workspace = benchmark_root / "design" / case / "workspace"
    return [
        str(params),
        "--place-io-engine",
        "openroad",
        "--workspace",
        str(workspace),
        "--result-dir",
        str(result_dir),
        "--base-design-name",
        case,
        "--def-input",
        str(input_def),
        "--verilog-input",
        str(paths["verilog"]),
        "--sdc",
        str(paths["sdc"]),
        "--output-def",
        str(output_def),
        "--enable-fillers",
        "0",
        "--legalize",
        "0",
        "--plot",
        "0",
    ] + _tech_args(benchmark_root)


def _placement_command(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    case: str,
    input_r0: Path,
    output_root: Path,
) -> tuple[list[str], Path, Path]:
    run_root = output_root / "placement_source"
    case_root = run_root / "run" / case
    command = [
        str(args.python_bin.resolve()),
        str(PLACEMENT_RUNNER),
        "--benchmark-root",
        str(benchmark_root),
        "--output-root",
        str(run_root),
        "--run-id",
        "run",
        "--python-bin",
        str(args.python_bin.resolve()),
        "--openroad-bin",
        str(args.openroad_bin.resolve()),
        "--openroad-num-threads",
        str(args.cpu_threads),
        "--case",
        case,
        "--method",
        "TDP-pin2pin",
        "--outer-iterations",
        "1",
        "--iterations",
        str(args.max_steps),
        "--stop-overflow",
        str(args.stop_overflow),
        "--seed",
        str(args.seed),
        "--gpu-id",
        "0",
        "--cuda-visible-devices",
        str(args.gpu_id),
        "--placement-optimizer",
        str(args.placement_optimizer),
        "--timing-topology-enable-overflow-threshold",
        str(args.timing_activation_overflow),
        "--pin2pin-weight",
        str(args.pin2pin_weight),
        "--pin2pin-min-weight",
        str(args.pin2pin_min_weight),
        "--pin2pin-max-weight",
        str(args.pin2pin_max_weight),
        "--pin2pin-accumulate-weight",
        str(args.pin2pin_accumulate_weight),
        "--net-weighting-npaths",
        str(args.pin2pin_path_limit),
        "--pin2pin-update-interval",
        str(args.pin2pin_update_interval),
        "--input-r0-def",
        str(input_r0),
        "--source-only",
        "--enable-fillers",
        "--no-legalize",
        "--plot",
        "0",
    ]
    return command, case_root / "TDP-pin2pin" / f"{case}_raw.def", case_root / "r0" / "R0.def"


def _sizing_command(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    case: str,
    input_def: Path,
    output_root: Path,
    autodmp_legalize: bool = False,
) -> tuple[list[str], Path]:
    output_def = output_root / "sizing" / f"{case}_sized.def"
    command = [
        str(args.python_bin.resolve()),
        str(AUTODMP_ROOT / "dreamplace" / "Placer.py"),
        *_case_placer_args(
            benchmark_root,
            case,
            input_def,
            output_root / "sizing" / "autodmp_result",
            output_def,
        ),
        "--flow-kind",
        "sizing",
        "--with-sta",
        "--gpu",
        "1",
        "--gpu-id",
        "0",
        "--iterations",
        str(args.sizing_iterations),
        "--discrete-gradient-topk-up-percent",
        str(args.sizing_up_percent),
        "--discrete-gradient-topk-down-percent",
        str(args.sizing_down_percent),
    ]
    command[command.index("--legalize") + 1] = str(int(autodmp_legalize))
    return command, output_def


def _buffer_command(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    case: str,
    input_def: Path,
    output_root: Path,
) -> tuple[list[str], Path, Path]:
    source_root = output_root / "buffer_source"
    command = [
        str(args.python_bin.resolve()),
        str(EQUAL_BUFFERING_RUNNER),
        "--profile",
        str(getattr(args, "equal_buffering_profile", EQUAL_BUFFERING_PROFILE)),
        "--benchmark-root",
        str(benchmark_root),
        "--output-root",
        str(source_root),
        "--run-id",
        "run",
        "--python-bin",
        str(args.python_bin.resolve()),
        "--openroad-bin",
        str(args.openroad_bin.resolve()),
        "--openroad-num-threads",
        str(args.cpu_threads),
        "--case",
        case,
        "--method",
        "ours",
        "--segment-strategy",
        "discrete_net_gradient",
        "--cuda-visible-devices",
        str(args.gpu_id),
        "--logical-gpu-id",
        "0",
        "--def-input",
        str(input_def),
        "--source-only",
    ]
    result = source_root / "run" / case / "ours" / "result.json"
    committed = source_root / "run" / case / "ours" / f"{case}_ours_committed.def"
    return command, result, committed


def _post_buffer_physical_command(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    case: str,
    input_def: Path,
    output_root: Path,
) -> tuple[list[str], Path]:
    physical_root = output_root / "buffered_autodmp_physical"
    output_def = physical_root / f"{case}.def"
    placer_args = _case_placer_args(
        benchmark_root, case, input_def, physical_root / "autodmp_result", output_def
    )
    params = _read_json(Path(placer_args[0]))
    params.update(
        global_place_flag=0, legalize_flag=1, detailed_place_flag=1,
        random_center_init_flag=0, gp_noise_ratio=0, enable_fillers=0,
        macro_place_flag=0, routability_opt_flag=0, num_threads=int(args.cpu_threads),
    )
    params_path = physical_root / "params.json"
    _write_json(params_path, params)
    placer_args[0] = str(params_path)
    placer_args[placer_args.index("--legalize") + 1] = "1"
    command = [
        str(args.python_bin.resolve()), str(AUTODMP_ROOT / "dreamplace" / "Placer.py"),
        *placer_args, "--flow-kind", "placement", "--gpu", "1", "--gpu-id", "0",
    ]
    return command, output_def


def _openroad_tcl(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    case: str,
    input_def: Path,
    output_def: Path,
) -> str:
    paths = protocol.case_paths(benchmark_root, case)
    tech = protocol.technology_paths(benchmark_root)
    lines = []
    for lef in [tech["tech_lef"], *tech["lefs"]]:
        lines.append(f"read_lef {{{lef}}}")
    for lib in tech["libs"]:
        lines.append(f"read_liberty {{{lib}}}")
    lines.extend(
        [
            f"read_def {{{input_def}}}",
            f"read_sdc {{{paths['sdc']}}}",
            "set_ideal_network [all_clocks]",
            f"source {{{tech['rc_tcl']}}}",
            "set_cmd_units -time ns -capacitance pF -current mA -voltage V -resistance kOhm -distance um",
            "set_units -power mW",
            "estimate_parasitics -placement",
            "global_placement -timing_driven -density 0.8 -init_density_penalty 0.01",
            "repair_design",
            "repair_timing -setup",
            f"write_def {{{output_def}}}",
            "exit",
        ]
    )
    return "\n".join(lines) + "\n"


def _evaluate_command(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    case: str,
    input_def: Path,
    output_dir: Path,
    evaluation_mode: str = "physical_finalize",
) -> tuple[list[str], Path]:
    if evaluation_mode not in {"physical_finalize", "fixed_state"}:
        raise ValueError(f"unsupported mixed-flow evaluation mode: {evaluation_mode}")
    summary = output_dir / f"{evaluation_mode}_summary.json"
    command = [
        str(args.python_bin.resolve()),
        str(COMMON_EVALUATOR),
        "--case",
        case,
        "--method-id",
        f"mixed-flow-{evaluation_mode}",
        "--openroad-bin",
        str(args.openroad_bin.resolve()),
        "--openroad-num-threads",
        str(args.cpu_threads),
        "--reference-def",
        str(input_def),
        "--def-input",
        str(input_def),
        *_tech_args(benchmark_root),
        "--sdc",
        str(protocol.case_paths(benchmark_root, case)["sdc"]),
        "--buffer-master",
        str(args.buffer_master),
        "--output-dir",
        str(output_dir),
        "--evaluation-mode",
        evaluation_mode,
    ]
    if evaluation_mode == "fixed_state" and int(getattr(args, "profile_version", 0)) >= 3:
        command.append("--ignore-blocked-layer-violations")
    return command, summary


def _evaluate_legal_checkpoint(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    case: str,
    stage: str,
    input_def: Path,
    output_dir: Path,
    environment: dict[str, str],
    evaluation_mode: str,
) -> dict[str, Any]:
    command, summary_path = _evaluate_command(
        args=args,
        benchmark_root=benchmark_root,
        case=case,
        input_def=input_def,
        output_dir=output_dir,
        evaluation_mode=evaluation_mode,
    )
    execution = _run(
        command,
        cwd=AUTODMP_ROOT,
        log_path=output_dir / f"{evaluation_mode}.log",
        environment=environment,
        timeout_sec=args.timeout_sec,
    )
    record = {
        "stage": stage,
        "status": execution["status"],
        "evaluation_mode": evaluation_mode,
        "input_def": str(input_def),
        "input_def_sha256": (
            protocol.sha256_file(input_def) if input_def.is_file() else None
        ),
        "summary": str(summary_path),
        "execution": execution,
        "command": command,
    }
    if execution["status"] != "pass":
        record["failure_status"] = "execution_failed"
        return record
    if not summary_path.is_file():
        record.update(status="failed", failure_status="evaluation_failed")
        return record
    payload = _read_json(summary_path)
    record.update(
        evaluator_status=payload.get("status"),
        evaluator_failures=list(payload.get("failures") or ()),
        metrics=dict(payload.get("metrics") or {}),
        placement_validation=payload.get("placement_validation"),
        fixed_state_audit=payload.get("fixed_state_audit"),
        artifacts=dict(payload.get("artifacts") or {}),
    )
    if payload.get("status") != "pass":
        failures = " ".join(str(value) for value in payload.get("failures") or ())
        record.update(
            status="failed",
            failure_status=(
                "physical_invalid" if "placement" in failures else "evaluation_failed"
            ),
        )
        return record

    if evaluation_mode == "fixed_state":
        legal_def = input_def
    else:
        legal_def = Path(str(record["artifacts"].get("evaluated_def") or ""))
    expected_hash = (
        record["input_def_sha256"]
        if evaluation_mode == "fixed_state"
        else record["artifacts"].get("evaluated_def_sha256")
    )
    if (
        not legal_def.is_file()
        or not expected_hash
        or protocol.sha256_file(legal_def) != expected_hash
    ):
        record.update(status="failed", failure_status="artifact_contract_failed")
        return record
    record.update(
        status="pass",
        failure_status=None,
        legal_def=str(legal_def),
        legal_def_sha256=expected_hash,
    )
    return record


def _run_staged(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    case: str,
    input_r0: Path,
    arm_root: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    stages: dict[str, Any] = {}
    place_command, place_def, copied_r0 = _placement_command(
        args=args,
        benchmark_root=benchmark_root,
        case=case,
        input_r0=input_r0,
        output_root=arm_root,
    )
    stages["placement"] = _run(
        place_command,
        cwd=AUTODMP_ROOT,
        log_path=arm_root / "placement.log",
        environment=environment,
        timeout_sec=args.timeout_sec,
    )
    if copied_r0.is_file() and protocol.sha256_file(copied_r0) != protocol.sha256_file(input_r0):
        stages["placement"]["status"] = "artifact_contract_failed"
        stages["placement"]["reason"] = "runner R0 differs from prepared R0"
    if stages["placement"]["status"] != "pass" or not place_def.is_file():
        return {"status": "failed", "stages": stages, "failures": ["placement"]}

    size_command, size_def = _sizing_command(
        args=args,
        benchmark_root=benchmark_root,
        case=case,
        input_def=place_def,
        output_root=arm_root,
    )
    stages["sizing"] = _run(
        size_command,
        cwd=AUTODMP_ROOT,
        log_path=arm_root / "sizing.log",
        environment=environment,
        timeout_sec=args.timeout_sec,
    )
    if stages["sizing"]["status"] != "pass" or not size_def.is_file():
        return {"status": "failed", "stages": stages, "failures": ["sizing"]}

    buffer_command, buffer_result, buffer_def = _buffer_command(
        args=args,
        benchmark_root=benchmark_root,
        case=case,
        input_def=size_def,
        output_root=arm_root,
    )
    stages["buffering"] = _run(
        buffer_command,
        cwd=AUTODMP_ROOT,
        log_path=arm_root / "buffering.log",
        environment=environment,
        timeout_sec=args.timeout_sec,
    )
    if stages["buffering"]["status"] != "pass" or not buffer_result.is_file():
        return {"status": "failed", "stages": stages, "failures": ["buffering"]}
    buffering_payload = _read_json(buffer_result)
    buffering_result = dict(buffering_payload.get("result") or {})
    if _is_valid_no_action_buffering_result(
        buffering_result,
        input_def=size_def,
        payload=buffering_payload,
    ):
        stages["buffering"]["result_status"] = "skipped"
        stages["buffering"]["reason"] = "no_projected_buffer_actions"
        terminal_state_selection = {
            "status": "pass",
            "policy": "no_action_preserve_sized_state",
            "selected_state": "sized",
            "reason": "no_projected_buffer_actions",
            "committed_qor_applicable": False,
        }
        return {
            "status": "pass",
            "stages": stages,
            "raw_def": str(size_def),
            "raw_def_sha256": protocol.sha256_file(size_def),
            "stage_defs": {
                "placement": str(place_def),
                "sizing": str(size_def),
                "buffering": str(size_def),
            },
            "buffering_status": "skipped",
            "buffering_result": str(buffer_result),
            "terminal_state_selection": terminal_state_selection,
            "transaction_policy": TRANSACTION_POLICIES["autodmp_staged_sequential"],
            "commands": {
                "placement": place_command,
                "sizing": size_command,
                "buffering": buffer_command,
            },
        }
    if not buffer_def.is_file():
        return {"status": "failed", "stages": stages, "failures": ["buffering"]}
    inner_summary_path = Path(str(buffering_result.get("autodmp_summary_path") or ""))
    inner_summary = _read_json(inner_summary_path) if inner_summary_path.is_file() else {}
    buffering_commit = dict(
        dict(inner_summary.get("summary") or {}).get("commit") or {}
    )
    terminal_state_selection = _terminal_state_selection(buffering_commit)
    if terminal_state_selection["status"] != "pass":
        return {
            "status": "failed",
            "stages": stages,
            "failures": ["buffering_qor_evidence"],
            "stage_defs": {
                "placement": str(place_def),
                "sizing": str(size_def),
                "buffering": str(buffer_def),
            },
            "buffering_result": str(buffer_result),
            "terminal_state_selection": terminal_state_selection,
        }
    selected_def = (
        buffer_def
        if terminal_state_selection["selected_state"] == "buffered"
        else size_def
    )
    return {
        "status": "pass",
        "stages": stages,
        "raw_def": str(selected_def),
        "raw_def_sha256": protocol.sha256_file(selected_def),
        "stage_defs": {
            "placement": str(place_def),
            "sizing": str(size_def),
            "buffering": str(buffer_def),
        },
        "buffering_result": str(buffer_result),
        "buffering_commit": buffering_commit,
        "terminal_state_selection": terminal_state_selection,
        "transaction_policy": TRANSACTION_POLICIES["autodmp_staged_sequential"],
        "commands": {
            "placement": place_command,
            "sizing": size_command,
            "buffering": buffer_command,
        },
    }


def _openroad_native(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    case: str,
    input_r0: Path,
    arm_root: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    raw_def = arm_root / f"{case}_openroad_native_raw.def"
    tcl = arm_root / "openroad_native.tcl"
    tcl.parent.mkdir(parents=True, exist_ok=True)
    tcl.write_text(
        _openroad_tcl(
            args=args,
            benchmark_root=benchmark_root,
            case=case,
            input_def=input_r0,
            output_def=raw_def,
        ),
        encoding="utf-8",
    )
    command = [str(args.openroad_bin.resolve()), "-threads", str(args.cpu_threads), "-exit", str(tcl)]
    execution = _run(
        command,
        cwd=arm_root,
        log_path=arm_root / "openroad_native.log",
        environment=environment,
        timeout_sec=args.timeout_sec,
    )
    if execution["status"] != "pass" or not raw_def.is_file():
        return {"status": "failed", "stages": {"native": execution}, "failures": ["native"]}
    return {
        "status": "pass",
        "stages": {"native": execution},
        "raw_def": str(raw_def),
        "raw_def_sha256": protocol.sha256_file(raw_def),
        "transaction_policy": TRANSACTION_POLICIES["openroad_native_mixed"],
        "commands": {"native": command},
    }


def _run_coordinated(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    case: str,
    input_r0: Path,
    arm_root: Path,
    environment: dict[str, str],
) -> dict[str, Any]:
    source_root = arm_root / "joint_source"
    command = [
        str(args.python_bin.resolve()),
        str(PLACEMENT_RUNNER),
        "--benchmark-root",
        str(benchmark_root),
        "--output-root",
        str(source_root),
        "--run-id",
        "run",
        "--python-bin",
        str(args.python_bin.resolve()),
        "--openroad-bin",
        str(args.openroad_bin.resolve()),
        "--openroad-num-threads",
        str(args.cpu_threads),
        "--case",
        case,
        "--method",
        "TDP-pin2pin-SB",
        "--outer-iterations",
        "1",
        "--iterations",
        str(args.max_steps),
        "--stop-overflow",
        str(args.stop_overflow),
        "--seed",
        str(args.seed),
        "--gpu-id",
        "0",
        "--cuda-visible-devices",
        str(args.gpu_id),
        "--placement-optimizer",
        str(args.placement_optimizer),
        "--timing-topology-enable-overflow-threshold",
        str(args.timing_activation_overflow),
        "--pin2pin-weight",
        str(args.pin2pin_weight),
        "--pin2pin-min-weight",
        str(args.pin2pin_min_weight),
        "--pin2pin-max-weight",
        str(args.pin2pin_max_weight),
        "--pin2pin-accumulate-weight",
        str(args.pin2pin_accumulate_weight),
        "--net-weighting-npaths",
        str(args.pin2pin_path_limit),
        "--pin2pin-update-interval",
        str(args.pin2pin_update_interval),
        "--input-r0-def",
        str(input_r0),
        "--source-only",
        "--enable-fillers",
        "--no-legalize",
        "--plot",
        "0",
    ]
    for milestone in args.joint_segment_milestones:
        command.extend(("--joint-segment-milestone", str(float(milestone))))
    command.extend(
        (
            "--joint-segment-sizing-rounds",
            str(int(args.joint_segment_sizing_rounds)),
            "--joint-segment-sizing-up-percent",
            str(float(args.joint_segment_sizing_up_percent)),
            "--joint-segment-buffering-rounds",
            str(int(args.joint_segment_buffering_rounds)),
            "--joint-segment-buffering-selection-fraction",
            str(float(args.joint_segment_buffering_selection_fraction)),
            "--joint-segment-pin2pin-rebootstrap-enabled",
            "1" if args.joint_segment_pin2pin_rebootstrap_enabled else "0",
        )
    )
    result_root = source_root / "run" / case
    raw_def = result_root / "TDP-pin2pin-SB" / "autodmp_result" / f"{case}_segment_joint_final.def"
    copied_r0 = result_root / "r0" / "R0.def"
    source_summary_path = source_root / "run" / "campaign_summary.json"
    if raw_def.is_file() and copied_r0.is_file() and source_summary_path.is_file():
        execution = {
            "status": "pass",
            "returncode": 0,
            "timed_out": False,
            "runtime_sec": 0.0,
            "log_path": str(arm_root / "coordinated.log"),
            "command": command,
            "reused_source_artifact": True,
        }
    else:
        execution = _run(
            command,
            cwd=AUTODMP_ROOT,
            log_path=arm_root / "coordinated.log",
            environment=environment,
            timeout_sec=args.timeout_sec,
        )
    source_summary = _read_json(source_summary_path) if source_summary_path.is_file() else {}
    case_rows = [
        row
        for row in source_summary.get("case_results", ())
        if row.get("case") == case
    ]
    source_method = (
        dict(case_rows[0].get("methods", {})).get("TDP-pin2pin-SB", {})
        if len(case_rows) == 1
        else {}
    )
    physical_commit = dict(source_method.get("physical_commit") or {})
    source_diagnostics = {
        "case_status": case_rows[0].get("status") if len(case_rows) == 1 else None,
        "case_failures": case_rows[0].get("failures", []) if len(case_rows) == 1 else [],
        "mechanism_audit": source_method.get("mechanism_audit"),
        "physical_commit": physical_commit,
    }
    failures = []
    mechanism_audit = dict(source_diagnostics.get("mechanism_audit") or {})
    if source_diagnostics["case_failures"] or mechanism_audit.get("status") == "failed":
        failures.append("coordinated_mechanism_audit")
    source_completed = (
        raw_def.is_file()
        and physical_commit.get("status") == "ok"
        and physical_commit.get("final_def_sha256") == protocol.sha256_file(raw_def)
    )
    if execution["status"] != "pass" and not source_completed:
        failures.append("coordinated_execution")
    if not raw_def.is_file():
        failures.append("coordinated_raw_def_missing")
    if not copied_r0.is_file() or protocol.sha256_file(copied_r0) != protocol.sha256_file(input_r0):
        failures.append("coordinated_r0_mismatch")
    if failures:
        return {
            "status": "failed",
            "stages": {"coordinated": execution},
            "failures": failures,
            "source_diagnostics": source_diagnostics,
        }
    stages = {"coordinated": execution}
    core_raw_def = raw_def
    if int(getattr(args, "profile_version", 2)) >= 3:
        return _run_coordinated_v3_tail(
            args=args,
            benchmark_root=benchmark_root,
            case=case,
            arm_root=arm_root,
            environment=environment,
            stages=stages,
            core_legal_def=core_raw_def,
            coordinated_command=command,
            source_diagnostics=source_diagnostics,
        )
    refinement_root = arm_root / "terminal_refinement"
    size_command, sized_def = _sizing_command(
        args=args,
        benchmark_root=benchmark_root,
        case=case,
        input_def=core_raw_def,
        output_root=refinement_root,
    )
    stages["terminal_topology_rebuild_and_sizing"] = _run(
        size_command,
        cwd=AUTODMP_ROOT,
        log_path=refinement_root / "sizing.log",
        environment=environment,
        timeout_sec=args.timeout_sec,
    )
    if (
        stages["terminal_topology_rebuild_and_sizing"]["status"] != "pass"
        or not sized_def.is_file()
    ):
        return {
            "status": "failed",
            "stages": stages,
            "failures": ["terminal_topology_rebuild_and_sizing"],
            "source_diagnostics": source_diagnostics,
            "coordinated_core_def": str(core_raw_def),
        }
    topology_summary_path = (
        refinement_root
        / "sizing"
        / "autodmp_result"
        / f"{case}_openroad_aimp_db_summary.json"
    )
    topology_summary = (
        _read_json(topology_summary_path) if topology_summary_path.is_file() else {}
    )
    topology_counts = dict(topology_summary.get("counts") or {})
    topology_rebuild_evidence = {
        "status": "pass",
        "mechanism": "fresh_placer_from_coordinated_committed_def",
        "input_def": str(core_raw_def),
        "input_def_sha256": protocol.sha256_file(core_raw_def),
        "aimp_db_summary": str(topology_summary_path),
        "aimp_db_source": topology_summary.get("aimp_db_source"),
        "counts": topology_counts,
    }
    if (
        topology_summary.get("aimp_db_source") != "openroad"
        or any(int(topology_counts.get(name, 0) or 0) <= 0 for name in ("instances", "nets", "pins", "timing_edges"))
    ):
        topology_rebuild_evidence["status"] = "failed"
        return {
            "status": "failed",
            "stages": stages,
            "failures": ["terminal_topology_rebuild_evidence"],
            "source_diagnostics": source_diagnostics,
            "coordinated_core_def": str(core_raw_def),
            "topology_rebuild_evidence": topology_rebuild_evidence,
        }
    buffer_command, buffer_result, refined_def = _buffer_command(
        args=args,
        benchmark_root=benchmark_root,
        case=case,
        input_def=sized_def,
        output_root=refinement_root,
    )
    stages["terminal_buffering"] = _run(
        buffer_command,
        cwd=AUTODMP_ROOT,
        log_path=refinement_root / "buffering.log",
        environment=environment,
        timeout_sec=args.timeout_sec,
    )
    if (
        stages["terminal_buffering"]["status"] != "pass"
        or not buffer_result.is_file()
    ):
        return {
            "status": "failed",
            "stages": stages,
            "failures": ["terminal_buffering"],
            "source_diagnostics": source_diagnostics,
            "coordinated_core_def": str(core_raw_def),
            "terminal_sized_def": str(sized_def),
        }
    buffering_payload = _read_json(buffer_result)
    buffering_result = dict(buffering_payload.get("result") or {})
    no_action_buffering = _is_valid_no_action_buffering_result(
        buffering_result,
        input_def=sized_def,
        payload=buffering_payload,
    )
    if not no_action_buffering and not refined_def.is_file():
        return {
            "status": "failed",
            "stages": stages,
            "failures": ["terminal_buffering"],
            "source_diagnostics": source_diagnostics,
            "coordinated_core_def": str(core_raw_def),
            "terminal_sized_def": str(sized_def),
        }
    inner_summary_path = Path(str(buffering_result.get("autodmp_summary_path") or ""))
    inner_summary = _read_json(inner_summary_path) if inner_summary_path.is_file() else {}
    trace_path = Path(
        str(dict(inner_summary.get("summary") or {}).get("trace_path") or "")
    )
    trace = _read_json(trace_path) if trace_path.is_file() else {}
    buffering_commit = dict(
        dict(inner_summary.get("summary") or {}).get("commit") or {}
    )
    buffering_round_evidence = {
        "status": "pass",
        "result": str(buffer_result),
        "action_audit_status": buffering_result.get("action_audit_status"),
        "trace": str(trace_path),
        "requested_rounds": int(args.terminal_buffering_rounds),
        "iterations": trace.get("iterations"),
        "round_count": len(trace.get("rounds") or ()),
        "accepted_action_count": int(
            dict(trace.get("commit") or {}).get("accepted_action_count", 0) or 0
        ),
        "terminal_reason": trace.get("terminal_reason"),
        "committed_opensta": buffering_commit,
    }
    if no_action_buffering:
        buffering_round_evidence.update(
            {
                "status": "skipped",
                "reason": "no_projected_buffer_actions",
            }
        )
        terminal_state_selection = {
            "status": "pass",
            "policy": "no_action_preserve_sized_state",
            "selected_state": "sized",
            "reason": "no_projected_buffer_actions",
            "committed_qor_applicable": False,
        }
        return {
            "status": "pass",
            "stages": stages,
            "raw_def": str(sized_def),
            "raw_def_sha256": protocol.sha256_file(sized_def),
            "coordinated_core_def": str(core_raw_def),
            "coordinated_core_def_sha256": protocol.sha256_file(core_raw_def),
            "terminal_sized_def": str(sized_def),
            "terminal_sized_def_sha256": protocol.sha256_file(sized_def),
            "terminal_buffered_def": None,
            "terminal_buffered_def_sha256": "",
            "terminal_buffering_result": str(buffer_result),
            "terminal_refinement": {
                "status": "pass",
                "topology_rebuild": "fresh_placer_from_coordinated_committed_def",
                "sizing_iterations": int(args.sizing_iterations),
                "buffering_rounds": int(args.terminal_buffering_rounds),
                "buffering_status": "skipped",
                "preserved_intermediate_milestone_actions": True,
                "topology_rebuild_evidence": topology_rebuild_evidence,
                "buffering_round_evidence": buffering_round_evidence,
                "terminal_state_selection": terminal_state_selection,
            },
            "transaction_policy": TRANSACTION_POLICIES["autodmp_coordinated"],
            "commands": {
                "coordinated": command,
                "terminal_topology_rebuild_and_sizing": size_command,
                "terminal_buffering": buffer_command,
            },
            "source_diagnostics": source_diagnostics,
        }
    rounds_complete = _terminal_buffering_rounds_complete(
        buffering_round_evidence,
        requested_rounds=int(args.terminal_buffering_rounds),
    )
    if (
        buffering_result.get("status") != "pass"
        or buffering_result.get("action_audit_status") != "pass"
        or not rounds_complete
    ):
        buffering_round_evidence["status"] = "failed"
        return {
            "status": "failed",
            "stages": stages,
            "failures": ["terminal_buffering_round_evidence"],
            "source_diagnostics": source_diagnostics,
            "coordinated_core_def": str(core_raw_def),
            "terminal_sized_def": str(sized_def),
            "topology_rebuild_evidence": topology_rebuild_evidence,
            "buffering_round_evidence": buffering_round_evidence,
        }
    buffering_round_evidence["completion_mode"] = (
        "requested_rounds"
        if int(buffering_round_evidence["iterations"])
        == int(args.terminal_buffering_rounds)
        else "early_stationary"
    )
    terminal_state_selection = _terminal_state_selection(buffering_commit)
    if terminal_state_selection["status"] != "pass":
        return {
            "status": "failed",
            "stages": stages,
            "failures": ["terminal_buffering_qor_evidence"],
            "source_diagnostics": source_diagnostics,
            "coordinated_core_def": str(core_raw_def),
            "terminal_sized_def": str(sized_def),
            "terminal_buffered_def": str(refined_def),
            "topology_rebuild_evidence": topology_rebuild_evidence,
            "buffering_round_evidence": buffering_round_evidence,
            "terminal_state_selection": terminal_state_selection,
        }
    selected_def = (
        refined_def
        if terminal_state_selection["selected_state"] == "buffered"
        else sized_def
    )
    return {
        "status": "pass",
        "stages": stages,
        "raw_def": str(selected_def),
        "raw_def_sha256": protocol.sha256_file(selected_def),
        "coordinated_core_def": str(core_raw_def),
        "coordinated_core_def_sha256": protocol.sha256_file(core_raw_def),
        "terminal_sized_def": str(sized_def),
        "terminal_sized_def_sha256": protocol.sha256_file(sized_def),
        "terminal_buffered_def": str(refined_def),
        "terminal_buffered_def_sha256": protocol.sha256_file(refined_def),
        "terminal_buffering_result": str(buffer_result),
        "terminal_refinement": {
            "status": "pass",
            "topology_rebuild": "fresh_placer_from_coordinated_committed_def",
            "sizing_iterations": int(args.sizing_iterations),
            "buffering_rounds": int(args.terminal_buffering_rounds),
            "preserved_intermediate_milestone_actions": True,
            "topology_rebuild_evidence": topology_rebuild_evidence,
            "buffering_round_evidence": buffering_round_evidence,
            "terminal_state_selection": terminal_state_selection,
        },
        "transaction_policy": TRANSACTION_POLICIES["autodmp_coordinated"],
        "commands": {
            "coordinated": command,
            "terminal_topology_rebuild_and_sizing": size_command,
            "terminal_buffering": buffer_command,
        },
        "source_diagnostics": source_diagnostics,
    }


def _run_coordinated_v3_tail(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    case: str,
    arm_root: Path,
    environment: dict[str, str],
    stages: dict[str, Any],
    core_legal_def: Path,
    coordinated_command: list[str],
    source_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    refinement_root = arm_root / "terminal_refinement_v3"
    checkpoints: list[dict[str, Any]] = []
    commands: dict[str, Any] = {"coordinated": coordinated_command}

    def failed(
        failure: str,
        *,
        failure_status: str,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        selection = _select_legal_checkpoints(
            checkpoints,
            tolerance_ns=float(args.terminal_selection_tolerance_ns),
        )
        result = {
            "status": failure_status,
            "stages": stages,
            "failures": [failure],
            "source_diagnostics": source_diagnostics,
            "legal_checkpoints": copy.deepcopy(checkpoints),
            "terminal_state_selection": selection,
            "terminal_evaluation_mode": "fixed_state",
            "transaction_policy": TRANSACTION_POLICIES["autodmp_coordinated"],
            "commands": commands,
        }
        if selection.get("status") == "pass":
            recoverable = Path(selection["selected_def"])
            result.update(
                raw_def=str(recoverable),
                raw_def_sha256=selection["selected_def_sha256"],
                recoverable_checkpoint={
                    "stage": selection["selected_state"],
                    "def": str(recoverable),
                    "def_sha256": selection["selected_def_sha256"],
                },
            )
        if extra:
            result.update(extra)
        return result

    core_checkpoint = _evaluate_legal_checkpoint(
        args=args,
        benchmark_root=benchmark_root,
        case=case,
        stage="core",
        input_def=core_legal_def,
        output_dir=refinement_root / "core_fixed_state",
        environment=environment,
        evaluation_mode="fixed_state",
    )
    stages["core_fixed_state_report"] = core_checkpoint
    commands["core_fixed_state_report"] = core_checkpoint["command"]
    if core_checkpoint["status"] != "pass":
        return failed(
            "core_fixed_state_report",
            failure_status=core_checkpoint.get("failure_status", "evaluation_failed"),
        )
    checkpoints.append(core_checkpoint)

    size_command, sized_output_def = _sizing_command(
        args=args,
        benchmark_root=benchmark_root,
        case=case,
        input_def=core_legal_def,
        output_root=refinement_root,
        autodmp_legalize=True,
    )
    commands["terminal_sizing"] = size_command
    stages["terminal_sizing"] = _run(
        size_command,
        cwd=AUTODMP_ROOT,
        log_path=refinement_root / "sizing.log",
        environment=environment,
        timeout_sec=args.timeout_sec,
    )
    if stages["terminal_sizing"]["status"] != "pass":
        return failed(
            "terminal_sizing",
            failure_status="execution_failed",
            extra={"terminal_sizing_output_def": str(sized_output_def)},
        )
    if not sized_output_def.is_file():
        return failed(
            "terminal_sizing_output_def_missing",
            failure_status="artifact_contract_failed",
            extra={"terminal_sizing_output_def": str(sized_output_def)},
        )

    topology_summary_path = (
        refinement_root
        / "sizing"
        / "autodmp_result"
        / f"{case}_openroad_aimp_db_summary.json"
    )
    topology_summary = (
        _read_json(topology_summary_path) if topology_summary_path.is_file() else {}
    )
    topology_counts = dict(topology_summary.get("counts") or {})
    topology_rebuild_evidence = {
        "status": "pass",
        "mechanism": "fresh_placer_from_core_legal_def",
        "input_def": str(core_legal_def),
        "input_def_sha256": protocol.sha256_file(core_legal_def),
        "aimp_db_summary": str(topology_summary_path),
        "aimp_db_source": topology_summary.get("aimp_db_source"),
        "counts": topology_counts,
    }
    if (
        topology_summary.get("aimp_db_source") != "openroad"
        or any(
            int(topology_counts.get(name, 0) or 0) <= 0
            for name in ("instances", "nets", "pins", "timing_edges")
        )
    ):
        topology_rebuild_evidence["status"] = "failed"
        return failed(
            "terminal_sizing_topology_rebuild_evidence",
            failure_status="artifact_contract_failed",
            extra={"topology_rebuild_evidence": topology_rebuild_evidence},
        )

    sized_checkpoint = _evaluate_legal_checkpoint(
        args=args,
        benchmark_root=benchmark_root,
        case=case,
        stage="sized",
        input_def=sized_output_def,
        output_dir=refinement_root / "sized_fixed_state_report",
        environment=environment,
        evaluation_mode="fixed_state",
    )
    stages["sized_fixed_state_report"] = sized_checkpoint
    commands["sized_fixed_state_report"] = sized_checkpoint["command"]
    if sized_checkpoint["status"] != "pass":
        return failed(
            "sized_fixed_state_report",
            failure_status=sized_checkpoint.get("failure_status", "evaluation_failed"),
            extra={
                "terminal_sizing_output_def": str(sized_output_def),
                "topology_rebuild_evidence": topology_rebuild_evidence,
            },
        )
    checkpoints.append(sized_checkpoint)
    sized_legal_def = Path(sized_checkpoint["legal_def"])

    buffer_command, buffer_result_path, buffered_raw_def = _buffer_command(
        args=args,
        benchmark_root=benchmark_root,
        case=case,
        input_def=sized_legal_def,
        output_root=refinement_root,
    )
    commands["terminal_buffering"] = buffer_command
    stages["terminal_buffering"] = _run(
        buffer_command,
        cwd=AUTODMP_ROOT,
        log_path=refinement_root / "buffering.log",
        environment=environment,
        timeout_sec=args.timeout_sec,
    )
    if (
        stages["terminal_buffering"]["status"] != "pass"
        or not buffer_result_path.is_file()
    ):
        return failed(
            "terminal_buffering",
            failure_status=(
                "execution_failed"
                if stages["terminal_buffering"]["status"] != "pass"
                else "artifact_contract_failed"
            ),
            extra={
                "terminal_sized_legal_def": str(sized_legal_def),
                "topology_rebuild_evidence": topology_rebuild_evidence,
            },
        )

    buffering_payload = _read_json(buffer_result_path)
    buffering_result = dict(buffering_payload.get("result") or {})
    no_action_buffering = _is_valid_no_action_buffering_result(
        buffering_result,
        input_def=sized_legal_def,
        payload=buffering_payload,
    )
    inner_summary_path = Path(
        str(buffering_result.get("autodmp_summary_path") or "")
    )
    inner_summary = (
        _read_json(inner_summary_path) if inner_summary_path.is_file() else {}
    )
    trace_path = Path(
        str(dict(inner_summary.get("summary") or {}).get("trace_path") or "")
    )
    trace = _read_json(trace_path) if trace_path.is_file() else {}
    buffering_round_evidence = {
        "status": "skipped" if no_action_buffering else "pass",
        "result": str(buffer_result_path),
        "input_def": str(sized_legal_def),
        "input_def_sha256": protocol.sha256_file(sized_legal_def),
        "action_audit_status": buffering_result.get("action_audit_status"),
        "trace": str(trace_path),
        "requested_rounds": int(args.terminal_buffering_rounds),
        "iterations": trace.get("iterations"),
        "round_count": len(trace.get("rounds") or ()),
        "accepted_action_count": int(
            dict(trace.get("commit") or {}).get("accepted_action_count", 0) or 0
        ),
        "terminal_reason": trace.get("terminal_reason"),
    }
    if no_action_buffering:
        buffering_round_evidence["reason"] = "no_projected_buffer_actions"
    else:
        if not buffered_raw_def.is_file():
            return failed(
                "terminal_buffering_raw_def_missing",
                failure_status="artifact_contract_failed",
                extra={"buffering_round_evidence": buffering_round_evidence},
            )
        if (
            buffering_result.get("status") != "pass"
            or buffering_result.get("action_audit_status") != "pass"
            or not _terminal_buffering_rounds_complete(
                buffering_round_evidence,
                requested_rounds=int(args.terminal_buffering_rounds),
            )
        ):
            buffering_round_evidence["status"] = "failed"
            return failed(
                "terminal_buffering_round_evidence",
                failure_status="artifact_contract_failed",
                extra={"buffering_round_evidence": buffering_round_evidence},
            )
        buffering_round_evidence["completion_mode"] = (
            "requested_rounds"
            if int(buffering_round_evidence["iterations"])
            == int(args.terminal_buffering_rounds)
            else "early_stationary"
        )
        physical_command, buffered_physical_def = _post_buffer_physical_command(
            args=args, benchmark_root=benchmark_root, case=case,
            input_def=buffered_raw_def, output_root=refinement_root,
        )
        commands["buffered_autodmp_physical"] = physical_command
        stages["buffered_autodmp_physical"] = _run(
            physical_command, cwd=AUTODMP_ROOT,
            log_path=refinement_root / "buffered_autodmp_physical.log",
            environment=environment, timeout_sec=args.timeout_sec,
        )
        if stages["buffered_autodmp_physical"]["status"] != "pass":
            return failed("buffered_autodmp_physical", failure_status="execution_failed")
        if not buffered_physical_def.is_file():
            return failed("buffered_autodmp_physical_def_missing", failure_status="artifact_contract_failed")
        buffered_checkpoint = _evaluate_legal_checkpoint(
            args=args,
            benchmark_root=benchmark_root,
            case=case,
            stage="buffered",
            input_def=buffered_physical_def,
            output_dir=refinement_root / "buffered_fixed_state_report",
            environment=environment,
            evaluation_mode="fixed_state",
        )
        stages["buffered_fixed_state_report"] = buffered_checkpoint
        commands["buffered_fixed_state_report"] = buffered_checkpoint["command"]
        if buffered_checkpoint["status"] != "pass":
            return failed(
                "buffered_fixed_state_report",
                failure_status=buffered_checkpoint.get(
                    "failure_status", "evaluation_failed"
                ),
                extra={"buffering_round_evidence": buffering_round_evidence},
            )
        checkpoints.append(buffered_checkpoint)

    selection = _select_legal_checkpoints(
        checkpoints,
        tolerance_ns=float(args.terminal_selection_tolerance_ns),
    )
    if selection["status"] != "pass":
        return failed(
            "terminal_checkpoint_selection",
            failure_status="evaluation_failed",
            extra={"buffering_round_evidence": buffering_round_evidence},
        )
    selected_def = Path(selection["selected_def"])
    buffered_checkpoint = next(
        (row for row in checkpoints if row.get("stage") == "buffered"),
        None,
    )
    return {
        "status": "pass",
        "stages": stages,
        "raw_def": str(selected_def),
        "raw_def_sha256": selection["selected_def_sha256"],
        "coordinated_core_def": str(core_legal_def),
        "coordinated_core_def_sha256": protocol.sha256_file(core_legal_def),
        "terminal_sizing_output_def": str(sized_output_def),
        "terminal_sizing_output_def_sha256": protocol.sha256_file(sized_output_def),
        "terminal_sized_def": str(sized_legal_def),
        "terminal_sized_def_sha256": protocol.sha256_file(sized_legal_def),
        "terminal_buffered_raw_def": (
            None if no_action_buffering else str(buffered_raw_def)
        ),
        "terminal_buffered_def": (
            None if buffered_checkpoint is None else buffered_checkpoint["legal_def"]
        ),
        "terminal_buffering_result": str(buffer_result_path),
        "legal_checkpoints": copy.deepcopy(checkpoints),
        "terminal_state_selection": selection,
        "terminal_evaluation_mode": "fixed_state",
        "terminal_refinement": {
            "status": "pass",
            "topology_rebuild": "fresh_placer_from_core_legal_def",
            "sizing_legalization": "autodmp",
            "buffering_physical_treatment": "autodmp_legalization_and_detailed_placement",
            "sizing_evaluation_mode": "fixed_state",
            "fixed_state_placement_policy": "ignore_blocked_layers_only",
            "sizing_iterations": int(args.sizing_iterations),
            "sizing_ranking_mode": "real_size_taylor",
            "sizing_up_percent": float(args.sizing_up_percent),
            "sizing_down_percent": float(args.sizing_down_percent),
            "buffering_rounds": int(args.terminal_buffering_rounds),
            "buffering_status": (
                "skipped_no_action" if no_action_buffering else "completed"
            ),
            "topology_rebuild_evidence": topology_rebuild_evidence,
            "buffering_round_evidence": buffering_round_evidence,
            "terminal_state_selection": selection,
        },
        "transaction_policy": TRANSACTION_POLICIES["autodmp_coordinated"],
        "commands": commands,
        "source_diagnostics": source_diagnostics,
    }


def _terminal_evaluate(
    *,
    args: argparse.Namespace,
    benchmark_root: Path,
    case: str,
    raw_def: Path,
    arm_root: Path,
    environment: dict[str, str],
    evaluation_mode: str = "physical_finalize",
) -> dict[str, Any]:
    command, summary = _evaluate_command(
        args=args,
        benchmark_root=benchmark_root,
        case=case,
        input_def=raw_def,
        output_dir=arm_root / f"terminal_{evaluation_mode}",
        evaluation_mode=evaluation_mode,
    )
    execution = _run(
        command,
        cwd=AUTODMP_ROOT,
        log_path=arm_root / f"terminal_{evaluation_mode}.log",
        environment=environment,
        timeout_sec=args.timeout_sec,
    )
    if execution["status"] != "pass" or not summary.is_file():
        return {
            "status": "failed",
            "execution": execution,
            "summary": str(summary),
        }
    payload = _read_json(summary)
    return {
        "status": payload.get("status"),
        "execution": execution,
        "summary": str(summary),
        "metrics": payload.get("metrics", {}),
        "placement_validation": payload.get("placement_validation"),
        "fixed_state_audit": payload.get("fixed_state_audit"),
        "artifacts": payload.get("artifacts", {}),
    }


def _selected_cases(
    values: list[str] | None, *, all_cases: bool = False
) -> tuple[str, ...]:
    if all_cases and values:
        raise ValueError("--all-cases cannot be combined with --case")
    selected = tuple(
        protocol.CANONICAL_CASES if all_cases else (values or DEFAULT_PILOT_CASES)
    )
    invalid = sorted(set(selected) - set(protocol.CANONICAL_CASES))
    if invalid:
        raise ValueError("unsupported case(s): " + ", ".join(invalid))
    if len(set(selected)) != len(selected):
        raise ValueError("cases must be unique")
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=Path, default=DEFAULT_PROFILE)
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--benchmark-root", type=Path, default=protocol.DEFAULT_BENCHMARK_ROOT)
    parser.add_argument("--python-bin", type=Path, default=Path("/home/zhaoxueyan/anaconda3/envs/PlaceOPT/bin/python"))
    parser.add_argument("--openroad-bin", type=Path, default=Path("/home/zhaoxueyan/code/OpenROAD/build/bin/openroad"))
    parser.add_argument("--case", dest="cases", action="append")
    parser.add_argument(
        "--all-cases",
        action="store_true",
        help="Run the canonical ten-case suite instead of the three-case pilot.",
    )
    parser.add_argument("--arm", dest="arms", action="append", choices=ARMS)
    parser.add_argument("--gpu-id", default="0")
    parser.add_argument("--cpu-threads", type=int, default=16)
    parser.add_argument("--seed", type=int, default=3000)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--stop-overflow", type=float)
    parser.add_argument("--sizing-iterations", type=int)
    parser.add_argument("--sizing-up-percent", type=float)
    parser.add_argument("--sizing-down-percent", type=float)
    parser.add_argument("--terminal-buffering-rounds", type=int)
    parser.add_argument("--placement-optimizer")
    parser.add_argument("--timing-activation-overflow", type=float)
    parser.add_argument("--pin2pin-update-interval", type=int)
    parser.add_argument("--pin2pin-weight", type=float)
    parser.add_argument("--pin2pin-min-weight", type=float)
    parser.add_argument("--pin2pin-max-weight", type=float)
    parser.add_argument("--pin2pin-accumulate-weight", type=float)
    parser.add_argument("--pin2pin-path-limit", type=float)
    parser.add_argument(
        "--joint-segment-milestone",
        dest="joint_segment_milestones",
        action="append",
        type=float,
    )
    parser.add_argument("--joint-segment-sizing-rounds", type=int)
    parser.add_argument("--joint-segment-sizing-up-percent", type=float)
    parser.add_argument("--joint-segment-buffering-rounds", type=int)
    parser.add_argument(
        "--joint-segment-buffering-selection-fraction",
        type=float,
    )
    parser.add_argument("--buffer-master")
    parser.add_argument("--timeout-sec", type=int, default=21600)
    parser.add_argument("--plan-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    profile = protocol.load_profile(args.profile.resolve())
    effective_config, value_sources = _resolve_effective_config(args, profile)
    if args.cpu_threads <= 0 or args.max_steps <= 0 or args.sizing_iterations <= 0:
        raise ValueError("thread and iteration settings must be positive")
    if args.pin2pin_path_limit < 0:
        raise ValueError("pin2pin path limit must be nonnegative")
    benchmark_root = Path(args.benchmark_root)
    cases = _selected_cases(args.cases, all_cases=args.all_cases)
    arms = tuple(args.arms or ARMS)
    benchmark_manifest = protocol.validate_benchmark_inputs(benchmark_root, cases)
    campaign_manifest_path = Path(args.campaign_root).resolve() / "manifest.json"
    if not campaign_manifest_path.is_file():
        raise FileNotFoundError(campaign_manifest_path)
    campaign = _read_json(campaign_manifest_path)
    if (
        campaign.get("profile_name") != profile["profile_name"]
        or campaign.get("profile_sha256") != protocol.sha256_file(args.profile.resolve())
    ):
        raise ValueError("campaign root profile identity differs from selected profile")
    current_source_identity = protocol.build_source_identity(AUTODMP_ROOT)
    source_identity_digest = _validate_campaign_source_identity(
        campaign=campaign,
        profile_version=int(profile["artifact_version"]),
        current_source_identity=current_source_identity,
    )
    selected_manifest = campaign.get("benchmark") or {}
    if selected_manifest.get("benchmark_root_resolved_path") != benchmark_manifest[
        "benchmark_root_resolved_path"
    ]:
        raise ValueError("campaign benchmark root differs from current benchmark root")
    if selected_manifest.get("technology_inputs") != benchmark_manifest[
        "technology_inputs"
    ]:
        raise ValueError("campaign technology inputs differ from current benchmark inputs")
    campaign_cases = set(selected_manifest.get("cases") or ())
    if not set(cases) <= campaign_cases:
        raise ValueError("campaign root does not contain all selected mixed-flow cases")
    for case in cases:
        if selected_manifest.get("case_inputs", {}).get(case) != benchmark_manifest[
            "case_inputs"
        ].get(case):
            raise ValueError(f"campaign benchmark input differs for {case}")
    campaign_root = Path(args.campaign_root).resolve()
    output_root = campaign_root / "mixed_flow_primary"
    output_root.mkdir(parents=True, exist_ok=True)
    args.equal_buffering_profile, buffering_profile_evidence = (
        _resolved_equal_buffering_profile(
            args=args,
            profile=profile,
            output_root=output_root,
        )
    )
    effective_config["equal_buffering_profile"] = buffering_profile_evidence
    identity_config = copy.deepcopy(effective_config)
    identity_config["equal_buffering_profile"] = {
        "source_sha256": buffering_profile_evidence["source_sha256"],
        "resolved_sha256": buffering_profile_evidence["resolved_sha256"],
    }
    effective_config_digest = protocol.digest_payload(identity_config)
    mixed_flow_id = protocol.digest_payload(
        {
            "campaign_id": campaign["campaign_id"],
            "effective_config_digest": effective_config_digest,
            "value_sources": value_sources,
            "source_identity_digest": source_identity_digest,
        }
    )[:16]
    _write_json(
        output_root / "effective_config.json",
        {
            "artifact": "autodmp_timing_physical_synthesis_mixed_effective_config",
            "artifact_version": 2,
            "mixed_flow_id": mixed_flow_id,
            "effective_config_digest": effective_config_digest,
            "values": effective_config,
            "identity_values": identity_config,
            "value_sources": value_sources,
        },
    )
    manifest = {
        "artifact": "autodmp_timing_physical_synthesis_mixed_flow_manifest",
        "artifact_version": 2,
        "campaign_id": campaign["campaign_id"],
        "mixed_flow_id": mixed_flow_id,
        "profile_path": str(args.profile.resolve()),
        "profile_sha256": protocol.sha256_file(args.profile.resolve()),
        "profile_version": int(profile["artifact_version"]),
        "effective_config_digest": effective_config_digest,
        "effective_config": effective_config,
        "value_sources": value_sources,
        "benchmark_manifest_digest": campaign["benchmark_manifest_digest"],
        "source_identity_digest": source_identity_digest,
        "source_identity": campaign.get("source_identity"),
        "cases": list(cases),
        "arms": list(arms),
        "input_state_root": str(campaign_root / "initial_state"),
        "policy": {
            "terminal_evaluator_by_arm": {
                "autodmp_staged_sequential": "physical_finalize",
                "autodmp_coordinated": (
                    "fixed_state"
                    if int(profile["artifact_version"]) >= 3
                    else "physical_finalize"
                ),
                "openroad_native_mixed": "physical_finalize",
            },
            "read_verilog_for_terminal_state": False,
            "placement_only_rerun": False,
            "comparison_scope": "system_qor",
            "strict_coordination_attribution": False,
            "transaction_policy_by_arm": dict(TRANSACTION_POLICIES),
        },
    }
    manifest_path = output_root / "mixed_flow_manifest.json"
    if manifest_path.is_file():
        existing = _read_json(manifest_path)
        immutable_fields = (
            "campaign_id",
            "mixed_flow_id",
            "profile_path",
            "profile_sha256",
            "effective_config_digest",
            "source_identity_digest",
            "input_state_root",
            "policy",
        )
        if any(existing.get(field) != manifest.get(field) for field in immutable_fields):
            raise ValueError("mixed-flow manifest already exists with different identity")
        accepted_benchmark_digests = {
            campaign["benchmark_manifest_digest"],
            protocol.digest_payload(benchmark_manifest),
        }
        existing_cases = tuple(existing.get("cases") or ())
        existing_benchmark = campaign.get("benchmark") or {}
        if existing_cases and set(existing_cases) <= set(existing_benchmark.get("cases") or ()):
            existing_subset = {
                **existing_benchmark,
                "cases": list(existing_cases),
                "case_inputs": {
                    case: existing_benchmark["case_inputs"][case]
                    for case in existing_cases
                },
            }
            accepted_benchmark_digests.add(protocol.digest_payload(existing_subset))
        if existing.get("benchmark_manifest_digest") not in accepted_benchmark_digests:
            raise ValueError("mixed-flow manifest benchmark identity differs from campaign")
        existing_cases = set(existing.get("cases") or ())
        existing_arms = set(existing.get("arms") or ())
        selected_cases = set(manifest["cases"])
        selected_arms = set(manifest["arms"])
        if not selected_cases <= existing_cases or not selected_arms <= existing_arms:
            manifest["cases"] = sorted(existing_cases | selected_cases)
            manifest["arms"] = sorted(existing_arms | selected_arms)
            _write_json(manifest_path, manifest)
        else:
            manifest = existing
    else:
        _write_json(manifest_path, manifest)
    if args.plan_only:
        for case in cases:
            input_r0 = campaign_root / "initial_state" / "R0" / case / "R0.def"
            for arm in arms:
                arm_root = output_root / case / arm
                _write_json(
                    arm_root / "plan.json",
                    {
                        "case": case,
                        "arm": arm,
                        "input_r0": str(input_r0),
                        "input_r0_sha256": protocol.sha256_file(input_r0) if input_r0.is_file() else "plan_only",
                    },
                )
        print(f"mixed_flow_root: {output_root}")
        print(f"status: plan_only")
        return 0

    _validate_prepared_r0(campaign_root, cases)
    environment = _environment(args)
    rows = []
    failures = []
    for case in cases:
        input_r0 = campaign_root / "initial_state" / "R0" / case / "R0.def"
        if not input_r0.is_file():
            raise FileNotFoundError(input_r0)
        for arm in arms:
            arm_root = output_root / case / arm
            result_path = arm_root / "result.json"
            if result_path.is_file():
                existing_result = _read_json(result_path)
                if _existing_result_reusable(
                    arm=arm,
                    result=existing_result,
                    input_r0_sha256=protocol.sha256_file(input_r0),
                    sizing_iterations=int(args.sizing_iterations),
                    buffering_rounds=int(args.terminal_buffering_rounds),
                ):
                    rows.append(existing_result)
                    print(f"{case}/{arm}: pass (reused)")
                    continue
            started = time.perf_counter()
            if arm == "autodmp_staged_sequential":
                method = _run_staged(
                    args=args,
                    benchmark_root=benchmark_root,
                    case=case,
                    input_r0=input_r0,
                    arm_root=arm_root,
                    environment=environment,
                )
            elif arm == "autodmp_coordinated":
                method = _run_coordinated(
                    args=args,
                    benchmark_root=benchmark_root,
                    case=case,
                    input_r0=input_r0,
                    arm_root=arm_root,
                    environment=environment,
                )
            else:
                method = _openroad_native(
                    args=args,
                    benchmark_root=benchmark_root,
                    case=case,
                    input_r0=input_r0,
                    arm_root=arm_root,
                    environment=environment,
                )
            terminal = {"status": "not_run"}
            raw_def = Path(str(method.get("raw_def") or ""))
            if raw_def.is_file():
                terminal = _terminal_evaluate(
                    args=args,
                    benchmark_root=benchmark_root,
                    case=case,
                    raw_def=raw_def,
                    arm_root=arm_root,
                    environment=environment,
                    evaluation_mode=str(
                        method.get("terminal_evaluation_mode")
                        or "physical_finalize"
                    ),
                )
            if method.get("status") == "pass" and terminal.get("status") == "pass":
                status = "pass"
            elif method.get("status") != "pass":
                status = str(method.get("status") or "failed")
            else:
                status = "evaluation_failed"
            if status != "pass":
                failures.append(f"{case}/{arm}")
            row = {
                "case": case,
                "arm": arm,
                "status": status,
                "mixed_flow_id": mixed_flow_id,
                "effective_config_digest": effective_config_digest,
                "method": method,
                "terminal": terminal,
                "input_r0": str(input_r0),
                "input_r0_sha256": protocol.sha256_file(input_r0),
                "runtime_sec": time.perf_counter() - started,
            }
            rows.append(row)
            _write_json(result_path, row)
            print(f"{case}/{arm}: {status}")
    declared_cases = tuple(
        case for case in protocol.CANONICAL_CASES if case in set(manifest["cases"])
    )
    declared_arms = tuple(arm for arm in ARMS if arm in set(manifest["arms"]))
    aggregate_rows = []
    aggregate_failures = []
    for case in declared_cases:
        for arm in declared_arms:
            result_path = output_root / case / arm / "result.json"
            if not result_path.is_file():
                aggregate_failures.append(f"{case}/{arm}:not_run")
                continue
            row = _read_json(result_path)
            aggregate_rows.append(row)
            if row.get("status") != "pass":
                aggregate_failures.append(f"{case}/{arm}:{row.get('status')}")
    _write_json(
        output_root / "mixed_flow_summary.json",
        {
            "artifact": "autodmp_timing_physical_synthesis_mixed_flow_summary",
            "artifact_version": 1,
            "campaign_id": campaign["campaign_id"],
            "mixed_flow_id": mixed_flow_id,
            "effective_config_digest": effective_config_digest,
            "cases": list(declared_cases),
            "arms": list(declared_arms),
            "status": "pass" if not aggregate_failures else "incomplete",
            "failures": aggregate_failures,
            "rows": aggregate_rows,
        },
    )
    print(f"mixed_flow_root: {output_root}")
    print(f"status: {'pass' if not failures else 'failed'}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
