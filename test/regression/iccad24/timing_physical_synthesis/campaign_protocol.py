#!/usr/bin/env python3
"""Protocol helpers for the timing physical synthesis campaign."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


SCRIPT_DIR = Path(__file__).resolve().parent
AUTODMP_ROOT = SCRIPT_DIR.parents[3]
REPO_ROOT = AUTODMP_ROOT.parents[2]
COMMON_DIR = AUTODMP_ROOT / "test" / "regression" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.insert(0, str(COMMON_DIR))

from def_only_track_a import (  # noqa: E402
    EXPECTED_ICCAD24_LIBERTY_BASENAMES,
    validate_complete_iccad24_liberty_manifest,
)

DEFAULT_BENCHMARK_ROOT = (
    REPO_ROOT / "references" / "benchmarks" / "iccad24_benchmark"
)
CANONICAL_CASES = (
    "NV_NVDLA_partition_m",
    "NV_NVDLA_partition_p",
    "aes_256",
    "ariane136",
    "hidden1",
    "hidden2",
    "hidden3",
    "hidden4",
    "hidden5",
    "mempool_tile_wrap",
)
PROFILE_ARTIFACT = "autodmp_timing_physical_synthesis_profile"
SUPPORTED_PROFILE_VERSIONS = (2, 3)
BENCHMARK_MANIFEST_ARTIFACT = "autodmp_timing_physical_synthesis_benchmark_manifest"
BENCHMARK_MANIFEST_VERSION = 1
DPOST_MANIFEST_ARTIFACT = "autodmp_timing_physical_synthesis_dpost_manifest"
DPOST_MANIFEST_VERSION = 1
SOURCE_IDENTITY_ARTIFACT = "autodmp_timing_physical_synthesis_source_identity"
SOURCE_IDENTITY_VERSION = 1
SOURCE_IDENTITY_RELATIVE_FILES = (
    "test/regression/common/def_only_track_a.py",
    "test/regression/iccad24/joint_place_sizing_buffering/random_init_snapshot.py",
    "test/regression/iccad24/joint_place_sizing_buffering/run_random_init_validation.py",
    "test/regression/iccad24/equal_spaced_buffering/run.py",
    "test/regression/iccad24/equal_spaced_buffering/params/route_b_allcases_0p1pct.json",
    "test/regression/iccad24/timing_physical_synthesis/campaign_protocol.py",
    "test/regression/iccad24/timing_physical_synthesis/mixed_flow.py",
    "test/regression/iccad24/timing_physical_synthesis/run.py",
    "test/regression/iccad24/timing_physical_synthesis/profiles/timing_physical_synthesis_v3.json",
)


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")


def digest_payload(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_identity_payload(identity: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact": identity.get("artifact"),
        "artifact_version": identity.get("artifact_version"),
        "git_head": identity.get("git_head"),
        "tracked_dirty": identity.get("tracked_dirty"),
        "tracked_diff_sha256": identity.get("tracked_diff_sha256"),
        "source_file_sha256": identity.get("source_file_sha256"),
        "native_library_sha256": identity.get("native_library_sha256"),
    }


def validate_source_identity(identity: dict[str, Any]) -> None:
    if identity.get("artifact") != SOURCE_IDENTITY_ARTIFACT:
        raise ValueError("invalid timing physical synthesis source identity artifact")
    if int(identity.get("artifact_version", 0) or 0) != SOURCE_IDENTITY_VERSION:
        raise ValueError("unsupported timing physical synthesis source identity version")
    git_head = str(identity.get("git_head") or "")
    if len(git_head) not in (40, 64) or any(
        character not in "0123456789abcdef" for character in git_head.lower()
    ):
        raise ValueError("source identity git_head must be a Git object ID")
    for name in ("tracked_diff_sha256", "identity_digest"):
        value = str(identity.get(name) or "")
        if len(value) != 64:
            raise ValueError(f"source identity {name} must be a SHA-256")
    for name in ("source_file_sha256", "native_library_sha256"):
        records = identity.get(name)
        if not isinstance(records, dict) or not records:
            raise ValueError(f"source identity {name} must be nonempty")
        if any(len(str(value)) != 64 for value in records.values()):
            raise ValueError(f"source identity {name} contains an invalid SHA-256")
    expected = digest_payload(_source_identity_payload(identity))
    if identity.get("identity_digest") != expected:
        raise ValueError("source identity digest mismatch")


def build_source_identity(autodmp_root: Path = AUTODMP_ROOT) -> dict[str, Any]:
    root = _lexical_absolute(autodmp_root)

    def git_output(*arguments: str) -> bytes:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "failed to capture AutoDMP source identity: "
                + completed.stderr.decode("utf-8", errors="ignore")
            )
        return completed.stdout

    git_head = git_output("rev-parse", "HEAD").decode("ascii").strip()
    tracked_diff = git_output("diff", "--binary", "HEAD", "--")
    source_paths = [root / relative for relative in SOURCE_IDENTITY_RELATIVE_FILES]
    missing = [str(path) for path in source_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "missing timing physical synthesis source file(s): " + ", ".join(missing)
        )
    native_paths = sorted((root / "dreamplace").rglob("*.so"))
    if not native_paths:
        raise FileNotFoundError("no AutoDMP native libraries were found")
    identity = {
        "artifact": SOURCE_IDENTITY_ARTIFACT,
        "artifact_version": SOURCE_IDENTITY_VERSION,
        "autodmp_root": str(root),
        "git_head": git_head,
        "tracked_dirty": bool(tracked_diff),
        "tracked_diff_sha256": hashlib.sha256(tracked_diff).hexdigest(),
        "source_file_sha256": {
            str(path.relative_to(root)): sha256_file(path) for path in source_paths
        },
        "native_library_sha256": {
            str(path.relative_to(root)): sha256_file(path) for path in native_paths
        },
    }
    identity["identity_digest"] = digest_payload(_source_identity_payload(identity))
    validate_source_identity(identity)
    return identity


def load_profile(path: Path) -> dict[str, Any]:
    profile = json.loads(path.read_text(encoding="utf-8"))
    validate_profile(profile)
    return profile


def validate_profile(profile: dict[str, Any]) -> None:
    if profile.get("artifact") != PROFILE_ARTIFACT:
        raise ValueError("invalid timing physical synthesis profile artifact")
    version = int(profile.get("artifact_version", -1))
    if version not in SUPPORTED_PROFILE_VERSIONS:
        raise ValueError("unsupported timing physical synthesis profile version")
    expected_name = f"timing_physical_synthesis_v{version}"
    if profile.get("profile_name") != expected_name:
        raise ValueError("timing physical synthesis profile name/version mismatch")
    if tuple(profile.get("suite", ())) != CANONICAL_CASES:
        raise ValueError("profile must contain the canonical ten-case suite")
    source = dict(profile.get("benchmark_source") or {})
    if source.get("path") != "references/benchmarks/iccad24_benchmark":
        raise ValueError("profile benchmark source path mismatch")
    required_design = tuple(source.get("required_design_files", ()))
    expected_design = (
        "design/{case}/{case}.def",
        "design/{case}/{case}.v",
        "design/{case}/{case}.sdc",
    )
    if required_design != expected_design:
        raise ValueError("profile design input layout mismatch")
    required_technology = tuple(source.get("required_technology_files", ()))
    if not required_technology or any(
        not item.startswith("ASAP7/") for item in required_technology
    ):
        raise ValueError("profile technology input layout mismatch")
    r0 = dict(dict(profile.get("initial_state") or {}).get("R0") or {})
    if r0.get("producer") != "joint_place_sizing_buffering/random_init_snapshot.py":
        raise ValueError("profile R0 producer mismatch")
    if int(r0.get("seed", -1)) != 3000:
        raise ValueError("profile R0 seed mismatch")
    if bool(r0.get("enable_fillers")):
        raise ValueError("formal R0 must match the filler-free snapshot producer")
    dpost = dict(dict(profile.get("initial_state") or {}).get("D_post") or {})
    if dpost.get("source_relative_to_benchmark") != "design/{case}/{case}.def":
        raise ValueError("profile D_post source layout mismatch")
    for name, evaluator in dict(profile.get("evaluators") or {}).items():
        runner = str(dict(evaluator).get("runner") or "")
        if runner != "../../common/def_only_track_a.py":
            raise ValueError(f"{name}: evaluator runner must use common DEF-only evaluator")
    if version == 2:
        _validate_v2_profile(profile)
    else:
        _validate_v3_profile(profile)


def _validate_v2_profile(profile: dict[str, Any]) -> None:
    refinement = dict(profile.get("mixed_terminal_refinement") or {})
    expected_refinement = {
        "enabled": True,
        "applies_to": ["autodmp_coordinated"],
        "preserve_intermediate_milestone_actions": True,
        "topology_rebuild": "fresh_placer_from_coordinated_committed_def",
        "sizing_iterations": 100,
        "sizing_up_percent": 30.0,
        "sizing_down_percent": 0.0,
        "buffering_strategy": "discrete_net_gradient",
        "buffering_rounds": 5,
        "terminal_selection_policy": "committed_opensta_tns_improvement",
        "transaction_policy": "coordinated_core_commit_then_terminal_refinement_commit",
    }
    if refinement != expected_refinement:
        raise ValueError("mixed terminal refinement protocol mismatch")


def _validate_v3_profile(profile: dict[str, Any]) -> None:
    placement = dict(profile.get("placement") or {})
    activation = float(placement.get("pin2pin_activation_overflow", float("nan")))
    if not math.isfinite(activation) or not 0.0 < activation <= 1.0:
        raise ValueError("v3 Pin2Pin activation overflow must be in (0, 1]")
    if int(placement.get("pin2pin_update_interval", 0)) <= 0:
        raise ValueError("v3 Pin2Pin update interval must be positive")

    milestones = dict(profile.get("coordinated_milestones") or {})
    thresholds = tuple(float(value) for value in milestones.get("thresholds", ()))
    if not thresholds or any(not math.isfinite(value) for value in thresholds):
        raise ValueError("v3 milestone thresholds must be finite and nonempty")
    if any(
        thresholds[index] <= thresholds[index + 1]
        for index in range(len(thresholds) - 1)
    ):
        raise ValueError("v3 milestone thresholds must be strictly descending")
    if thresholds[0] > activation:
        raise ValueError("v3 first milestone must not exceed activation overflow")
    stop_overflow = float(placement.get("stop_overflow", float("nan")))
    if not math.isfinite(stop_overflow) or stop_overflow > thresholds[-1]:
        raise ValueError("v3 stop overflow must not precede the final milestone")
    if int(milestones.get("sizing_rounds", 0)) <= 0:
        raise ValueError("v3 milestone sizing rounds must be positive")
    sizing_fraction = float(milestones.get("sizing_up_percent", float("nan")))
    if not math.isfinite(sizing_fraction) or not 0.0 < sizing_fraction <= 100.0:
        raise ValueError("v3 milestone sizing percentage must be in (0, 100]")
    if float(milestones.get("sizing_down_percent", float("nan"))) != 0.0:
        raise ValueError("v3 milestone sizing must not downsize")
    if int(milestones.get("buffering_rounds", 0)) != 1:
        raise ValueError("v3 milestone buffering requires one Route-B round")
    buffering_fraction = float(
        milestones.get("buffering_selection_fraction", float("nan"))
    )
    if not math.isfinite(buffering_fraction) or not 0.0 < buffering_fraction <= 1.0:
        raise ValueError("v3 milestone buffering fraction must be in (0, 1]")
    if int(milestones.get("max_pair_generation_installs_per_iteration", 0)) != 1:
        raise ValueError("v3 permits exactly one pair generation install per iteration")
    if milestones.get("timing_clean_policy") != "confirm_current_post_step":
        raise ValueError("v3 timing-clean policy mismatch")

    sizing = dict(profile.get("sizing") or {})
    buffering = dict(profile.get("buffering") or {})
    refinement = dict(profile.get("mixed_terminal_refinement") or {})
    if not refinement.get("enabled"):
        raise ValueError("v3 terminal refinement must be enabled")
    if refinement.get("applies_to") != ["autodmp_coordinated"]:
        raise ValueError("v3 terminal refinement arm mismatch")
    if int(refinement.get("sizing_iterations", 0)) != int(
        sizing.get("iterations", -1)
    ):
        raise ValueError("v3 terminal sizing iteration mismatch")
    if refinement.get("sizing_ranking_mode") != sizing.get("ranking_mode"):
        raise ValueError("v3 terminal sizing ranking mismatch")
    if float(refinement.get("sizing_up_percent", float("nan"))) != float(
        sizing.get("up_percent", float("nan"))
    ) or float(refinement.get("sizing_down_percent", float("nan"))) != float(
        sizing.get("down_percent", float("nan"))
    ):
        raise ValueError("v3 terminal sizing percentage mismatch")
    if not sizing.get("coordinates_fixed") or not sizing.get("topology_frozen"):
        raise ValueError("v3 terminal sizing must keep coordinates and topology fixed")
    if int(refinement.get("buffering_rounds", 0)) != int(buffering.get("rounds", -1)):
        raise ValueError("v3 terminal buffering round mismatch")
    if float(milestones["buffering_selection_fraction"]) != float(
        buffering.get("selection_fraction", float("nan"))
    ):
        raise ValueError("v3 milestone and terminal buffering fractions differ")
    if refinement.get("terminal_selection_policy") != (
        "max_tns_then_wns_then_min_area_then_earliest_stage"
    ):
        raise ValueError("v3 terminal checkpoint selection policy mismatch")


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path.expanduser())))


def _path_record(path: Path, benchmark_root: Path) -> dict[str, Any]:
    lexical = _lexical_absolute(path)
    root = _lexical_absolute(benchmark_root)
    try:
        relative = lexical.relative_to(root)
    except ValueError as error:
        raise ValueError(f"input is outside benchmark root: {lexical}") from error
    if not lexical.is_file():
        raise FileNotFoundError(lexical)
    return {
        "path": str(lexical),
        "resolved_path": str(lexical.resolve()),
        "relative_to_benchmark_root": str(relative),
        "sha256": sha256_file(lexical),
        "size_bytes": lexical.stat().st_size,
    }


def case_paths(benchmark_root: Path, case: str) -> dict[str, Path]:
    root = _lexical_absolute(benchmark_root) / "design" / case
    return {
        "def": root / f"{case}.def",
        "verilog": root / f"{case}.v",
        "sdc": root / f"{case}.sdc",
    }


def technology_paths(benchmark_root: Path) -> dict[str, list[Path] | Path]:
    root = _lexical_absolute(benchmark_root) / "ASAP7"
    libs = [
        root / "lib" / name for name in EXPECTED_ICCAD24_LIBERTY_BASENAMES
    ]
    validate_complete_iccad24_liberty_manifest(libs)
    return {
        "tech_lef": root / "lef" / "asap7_tech_1x_201209.lef",
        "lefs": [
            root / "lef" / "asap7sc7p5t_27_R_1x_201211.lef",
            root / "lef" / "sram_asap7_16x256_1rw.lef",
            root / "lef" / "sram_asap7_32x256_1rw.lef",
            root / "lef" / "sram_asap7_64x256_1rw.lef",
            root / "lef" / "sram_asap7_64x64_1rw.lef",
        ],
        "libs": libs,
        "rc_tcl": root / "setRC.tcl",
    }


def validate_benchmark_inputs(
    benchmark_root: Path,
    cases: Iterable[str] = CANONICAL_CASES,
) -> dict[str, Any]:
    root = _lexical_absolute(benchmark_root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    selected = tuple(cases)
    invalid = sorted(set(selected) - set(CANONICAL_CASES))
    if invalid:
        raise ValueError("unsupported case(s): " + ", ".join(invalid))
    case_records: dict[str, Any] = {}
    for case in selected:
        paths = case_paths(root, case)
        case_records[case] = {
            name: _path_record(path, root) for name, path in paths.items()
        }
    tech = technology_paths(root)
    technology_records: dict[str, Any] = {}
    for name, path_or_paths in tech.items():
        paths = path_or_paths if isinstance(path_or_paths, list) else [path_or_paths]
        technology_records[name] = [_path_record(path, root) for path in paths]
    return {
        "artifact": BENCHMARK_MANIFEST_ARTIFACT,
        "artifact_version": BENCHMARK_MANIFEST_VERSION,
        "benchmark_root_path": str(root),
        "benchmark_root_resolved_path": str(root.resolve()),
        "benchmark_root_is_symlink": root.is_symlink(),
        "cases": list(selected),
        "case_inputs": case_records,
        "technology_inputs": technology_records,
    }


def build_dpost_manifest(
    benchmark_manifest: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    output = _lexical_absolute(output_root)
    records = {}
    for case in benchmark_manifest["cases"]:
        source = dict(benchmark_manifest["case_inputs"][case]["def"])
        normalized = output / case / f"{case}.def"
        records[case] = {
            "producer": "packaged_iccad24_def_passthrough",
            "source": source,
            "source_path": source["path"],
            "source_sha256": source["sha256"],
            "normalization": "none",
            "coordinates_fixed": True,
            "normalized_path": str(normalized),
            "normalized_sha256": source["sha256"],
        }
    return {
        "artifact": DPOST_MANIFEST_ARTIFACT,
        "artifact_version": DPOST_MANIFEST_VERSION,
        "benchmark_root_path": benchmark_manifest["benchmark_root_path"],
        "benchmark_root_resolved_path": benchmark_manifest[
            "benchmark_root_resolved_path"
        ],
        "cases": records,
    }


def build_r0_command(
    *,
    python_bin: Path,
    benchmark_root: Path,
    case: str,
    output_root: Path,
    seed: int,
) -> list[str]:
    paths = case_paths(benchmark_root, case)
    tech = technology_paths(benchmark_root)
    case_root = _lexical_absolute(output_root) / case
    command = [
        str(python_bin),
        str(SCRIPT_DIR.parent / "joint_place_sizing_buffering" / "random_init_snapshot.py"),
        "--case",
        case,
        "--params-json",
        str(_lexical_absolute(benchmark_root) / "design" / case / "workspace" / "config" / "dreamplace_config" / "param.json"),
        "--workspace",
        str(_lexical_absolute(benchmark_root) / "design" / case / "workspace"),
        "--source-def",
        str(paths["def"]),
        "--verilog",
        str(paths["verilog"]),
        "--sdc",
        str(paths["sdc"]),
        "--rc-tcl",
        str(tech["rc_tcl"]),
        "--tech-lef",
        str(tech["tech_lef"]),
        "--seed",
        str(int(seed)),
        "--result-dir",
        str(case_root / "r0" / "result"),
        "--output-def",
        str(case_root / "r0" / "R0.def"),
        "--manifest",
        str(case_root / "r0" / "R0_manifest.json"),
    ]
    for lef in tech["lefs"]:
        command.extend(("--lef", str(lef)))
    for liberty in tech["libs"]:
        command.extend(("--lib", str(liberty)))
    return command


def build_campaign_manifest(
    *,
    profile: dict[str, Any],
    benchmark_manifest: dict[str, Any],
    profile_path: Path,
    source_identity: dict[str, Any],
) -> dict[str, Any]:
    validate_source_identity(source_identity)
    profile_digest = digest_payload(profile)
    benchmark_digest = digest_payload(benchmark_manifest)
    source_identity_digest = str(source_identity["identity_digest"])
    return {
        "artifact": "autodmp_timing_physical_synthesis_campaign_manifest",
        "artifact_version": 2,
        "profile_name": profile["profile_name"],
        "profile_path": str(_lexical_absolute(profile_path)),
        "profile_sha256": sha256_file(_lexical_absolute(profile_path)),
        "profile_digest": profile_digest,
        "benchmark_manifest_digest": benchmark_digest,
        "source_identity_digest": source_identity_digest,
        "campaign_id": digest_payload(
            {
                "profile_digest": profile_digest,
                "benchmark_manifest_digest": benchmark_digest,
                "source_identity_digest": source_identity_digest,
            }
        )[:16],
        "benchmark": benchmark_manifest,
        "source_identity": source_identity,
    }
